//! Graph-compatible modules for the attention pipeline.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use flodl::autograd::Variable;
use flodl::nn::{Identity, Linear, Module, NamedInputModule, Parameter};
use flodl::tensor::{Result, Tensor, TensorOptions};

// Re-export Identity so model.rs etc. can use it from here.
pub use flodl::Identity;

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

/// Shared controller components: GRU and location head.
/// Both ScanStep and AttentionStep hold Rc to the same instance,
/// ensuring shared weights and single parameter collection.
pub struct Controller {
    pub gru: flodl::GRUCell,
    pub loc_head: Linear,
}

impl Controller {
    pub fn new(hidden_dim: i64) -> Result<Self> {
        Ok(Controller {
            gru: flodl::GRUCell::new(hidden_dim, hidden_dim)?,
            loc_head: Linear::new(hidden_dim, 2)?,
        })
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.gru.parameters();
        params.extend(self.loc_head.parameters());
        params
    }
}

/// ScanStep: wide-patch scan with learnable x-position, learned y.
/// Produces coarse overview fixations guided by attention_guide_loss.
///
/// Location update: x = tanh(scan_x[step]), y = tanh(loc_head(h))[y].
/// The x-sweep is learned but constrained, y is free.
pub struct ScanStep {
    sensor: Rc<dyn Module>,
    controller: Rc<Controller>,
    scan_xs: Vec<Parameter>,
    location: RefCell<Option<Variable>>,
    step_idx: RefCell<usize>,
}

impl ScanStep {
    pub fn new(
        sensor: impl Module + 'static,
        controller: Rc<Controller>,
        n_scan: usize,
    ) -> Result<Self> {
        let mut scan_xs = Vec::with_capacity(n_scan);
        for i in 0..n_scan {
            let init_val = if n_scan == 1 {
                0.0
            } else {
                // Spread across [-0.5, 0.5] for multiple scan steps
                -0.5 + (i as f64) / (n_scan as f64 - 1.0)
            };
            let t = Tensor::from_f32(&[init_val as f32], &[1], flodl::tensor::Device::CPU)?;
            scan_xs.push(Parameter {
                variable: Variable::new(t, true),
                name: format!("scan_x_{i}"),
            });
        }
        Ok(ScanStep {
            sensor: Rc::new(sensor),
            controller,
            scan_xs,
            location: RefCell::new(None),
            step_idx: RefCell::new(0),
        })
    }

    fn step(&self, h: &Variable, image: &Variable) -> Result<Variable> {
        let new_h = {
            if self.location.borrow().is_none() {
                let batch = h.shape()[0];
                let device = h.data().device();
                let zeros = Tensor::zeros(&[batch, 2], TensorOptions { device, ..Default::default() })?;
                *self.location.borrow_mut() = Some(Variable::new(zeros, false));
            }

            let loc_guard = self.location.borrow();
            let loc = loc_guard.as_ref().unwrap();

            let mut refs = HashMap::new();
            refs.insert("location".to_string(), loc.clone());
            let glimpse = self.sensor.as_named_input().unwrap()
                .forward_named(image, &refs)?;

            self.controller.gru.forward_step(&glimpse, Some(h))?
        };

        // Location: learnable x, free y from loc_head
        let raw = self.controller.loc_head.forward(&new_h)?.tanh_act()?;
        let y = raw.select(1, 1)?.unsqueeze(1)?; // [B, 1]

        let idx = *self.step_idx.borrow();
        let scan_x = &self.scan_xs[idx.min(self.scan_xs.len() - 1)];
        let x = scan_x.variable.tanh_act()?.expand(&[h.shape()[0], 1])?; // [B, 1]
        *self.step_idx.borrow_mut() = idx + 1;

        let new_loc = x.cat(&y, 1)?; // [B, 2]
        *self.location.borrow_mut() = Some(new_loc);

        Ok(new_h)
    }
}

impl Module for ScanStep {
    fn name(&self) -> &str { "scan_step" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.step(input, input)
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> {
        Some(self)
    }

    fn reset(&self) {
        *self.location.borrow_mut() = None;
        *self.step_idx.borrow_mut() = 0;
    }

    fn detach_state(&self) {
        let mut loc = self.location.borrow_mut();
        if let Some(v) = loc.take() {
            *loc = Some(v.detach());
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        // Only scan-specific params: sensor + scan_xs.
        // GRU and loc_head are shared — collected by AttentionStep.
        let mut params = self.sensor.parameters();
        params.extend(self.scan_xs.iter().cloned());
        params
    }

    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        vec![self.sensor.clone()]
    }

    fn trace(&self) -> Option<Variable> {
        self.location.borrow().clone()
    }
}

impl NamedInputModule for ScanStep {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        let image = refs.get("image").expect("ScanStep requires 'image' ref");
        self.step(input, image)
    }
}

/// AttentionStep is the read loop body: receives h as stream, image via ref,
/// manages location as internal recurrent state. Free (x,y) positioning.
///
/// Implements:
/// - `NamedInputModule` — receives "image" via Using refs
/// - `Module::reset()` — auto-reset before each loop invocation
/// - `Module::trace()` — loop collects fixation trajectory
/// - `Module::detach_state()` — breaks gradient chain on carried state
pub struct AttentionStep {
    sensor: Rc<dyn Module>,
    controller: Rc<Controller>,
    location: RefCell<Option<Variable>>,
}

impl AttentionStep {
    /// Create with a new (unshared) controller. For standalone use or backward compat.
    pub fn new(sensor: impl Module + 'static, hidden_dim: i64) -> Result<Self> {
        Ok(AttentionStep {
            sensor: Rc::new(sensor),
            controller: Rc::new(Controller::new(hidden_dim)?),
            location: RefCell::new(None),
        })
    }

    /// Create with a shared controller (for scan+read architecture).
    pub fn with_shared(sensor: impl Module + 'static, controller: Rc<Controller>) -> Self {
        AttentionStep {
            sensor: Rc::new(sensor),
            controller,
            location: RefCell::new(None),
        }
    }

    fn step(&self, h: &Variable, image: &Variable) -> Result<Variable> {
        let new_h = {
            // Lazy init: derive batch size and device from input on first call.
            if self.location.borrow().is_none() {
                let batch = h.shape()[0];
                let device = h.data().device();
                let zeros = Tensor::zeros(&[batch, 2], TensorOptions { device, ..Default::default() })?;
                *self.location.borrow_mut() = Some(Variable::new(zeros, false));
            }

            let loc_guard = self.location.borrow();
            let loc = loc_guard.as_ref().unwrap();

            // Sensor extracts glimpse at current location
            let mut refs = HashMap::new();
            refs.insert("location".to_string(), loc.clone());
            let glimpse = self.sensor.as_named_input().unwrap()
                .forward_named(image, &refs)?;

            // GRU update
            self.controller.gru.forward_step(&glimpse, Some(h))?
        }; // loc_guard dropped here

        // Update location
        let new_loc = self.controller.loc_head.forward(&new_h)?.tanh_act()?;
        *self.location.borrow_mut() = Some(new_loc);

        Ok(new_h)
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

    fn reset(&self) {
        *self.location.borrow_mut() = None;
    }

    fn detach_state(&self) {
        let mut loc = self.location.borrow_mut();
        if let Some(v) = loc.take() {
            *loc = Some(v.detach());
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.sensor.parameters();
        // AttentionStep owns the controller params (even if shared via Rc).
        // ScanStep deliberately excludes them to avoid double-counting.
        params.extend(self.controller.parameters());
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
