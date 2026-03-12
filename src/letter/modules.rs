//! Graph-compatible modules for the attention pipeline.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use flodl::autograd::Variable;
use flodl::nn::{
    Detachable, Linear, Module, NamedInputModule, Parameter, Resettable,
};
use flodl::tensor::{Result, Tensor};

/// Identity passes the input through unchanged.
/// Used as the graph entry point to tag the image before routing.
pub struct Identity;

impl Module for Identity {
    fn name(&self) -> &str { "identity" }
    fn forward(&self, input: &Variable) -> Result<Variable> {
        Ok(input.clone())
    }
}

/// H0Init ignores its input and returns the learned initial hidden state
/// expanded to match the batch dimension.
pub struct H0Init {
    h0: Parameter,
    hidden_dim: i64,
}

impl H0Init {
    pub fn new(hidden_dim: i64) -> Result<Self> {
        let h0_data = Tensor::zeros(&[1, hidden_dim], Default::default())?;
        Ok(H0Init {
            h0: Parameter { variable: Variable::new(h0_data, true), name: "h0".into() },
            hidden_dim,
        })
    }
}

impl Module for H0Init {
    fn name(&self) -> &str { "h0_init" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let b = input.shape()[0];
        self.h0.variable.reshape(&[1, self.hidden_dim])?
            .expand(&[b, self.hidden_dim])
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.h0.clone()]
    }
}

/// AttentionStep is the loop body: receives h as stream, image via ref,
/// manages location as internal recurrent state.
///
/// Implements:
/// - `NamedInputModule` — receives "image" via Using refs
/// - `Resettable` — auto-reset before each forward pass
/// - `Module::trace()` — loop collects fixation trajectory
/// - `Detachable` — breaks gradient chain on carried state
pub struct AttentionStep {
    sensor: Rc<dyn Module>,
    gru: flodl::GRUCell,
    loc_head: Linear,
    location: RefCell<Option<Variable>>,
}

impl AttentionStep {
    pub fn new(sensor: impl Module + 'static, hidden_dim: i64) -> Result<Self> {
        Ok(AttentionStep {
            sensor: Rc::new(sensor),
            gru: flodl::GRUCell::new(hidden_dim, hidden_dim)?,
            loc_head: Linear::new(hidden_dim, 2)?,
            location: RefCell::new(None),
        })
    }

    fn step(&self, h: &Variable, image: &Variable) -> Result<Variable> {
        let new_h = {
            let loc_guard = self.location.borrow();
            let loc = loc_guard.as_ref().expect("AttentionStep: reset not called");

            // Sensor extracts glimpse at current location
            let glimpse = sensor_forward(&*self.sensor, image, loc)?;

            // GRU update
            self.gru.forward_step(&glimpse, Some(h))?
        }; // loc_guard dropped here

        // Update location
        let new_loc = self.loc_head.forward(&new_h)?.tanh_act()?;
        *self.location.borrow_mut() = Some(new_loc);

        Ok(new_h)
    }
}

/// Forward the sensor with (image, location) by concatenating them as a 2-input forward.
/// The sensor's Module::forward takes a single input, but GlimpseSensor expects
/// image and location — we use a special method on GlimpseSensor instead.
fn sensor_forward(sensor: &dyn Module, image: &Variable, location: &Variable) -> Result<Variable> {
    // Downcast not available with trait objects — we pass concatenated input
    // and handle it in GlimpseSensor::forward via shape splitting.
    // Actually, we use NamedInputModule on the sensor.
    if let Some(named) = sensor.as_named_input() {
        let mut refs = HashMap::new();
        refs.insert("location".to_string(), location.clone());
        named.forward_named(image, &refs)
    } else {
        // Fallback: plain forward (shouldn't happen for GlimpseSensor)
        sensor.forward(image)
    }
}

impl Module for AttentionStep {
    fn name(&self) -> &str { "attention_step" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        // Plain forward: no refs, create dummy image
        self.step(input, input)
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> {
        Some(self)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.sensor.parameters();
        params.extend(self.gru.parameters());
        params.extend(self.loc_head.parameters());
        params
    }

    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        vec![self.sensor.clone()]
    }

    fn trace(&self) -> Option<Variable> {
        self.location.borrow().clone()
    }
}

impl NamedInputModule for AttentionStep {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        let image = refs.get("image").expect("AttentionStep requires 'image' ref");
        self.step(input, image)
    }
}

impl Resettable for AttentionStep {
    fn reset(&self) {
        *self.location.borrow_mut() = None;
    }
}

impl Detachable for AttentionStep {
    fn detach_state(&self) {
        let mut loc = self.location.borrow_mut();
        if let Some(v) = loc.take() {
            *loc = Some(v.detach());
        }
    }
}

/// LatentHead projects the final hidden state to the latent space.
pub struct LatentHead {
    fc: Linear,
}

impl LatentHead {
    pub fn new(hidden_dim: i64, latent_dim: i64) -> Result<Self> {
        Ok(LatentHead { fc: Linear::new(hidden_dim, latent_dim)? })
    }
}

impl Module for LatentHead {
    fn name(&self) -> &str { "latent_head" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.fc.forward(input)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.fc.parameters()
    }
}

/// SelectFirst is a merge module that returns the first input.
/// Used after Split+TagGroup when the graph output is irrelevant
/// (we access individual heads via Tagged).
pub struct SelectFirst;

impl Module for SelectFirst {
    fn name(&self) -> &str { "select_first" }
    fn forward(&self, input: &Variable) -> Result<Variable> {
        Ok(input.clone())
    }
}
