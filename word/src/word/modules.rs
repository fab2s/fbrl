//! Graph-compatible modules for the word attention pipeline.
//!
//! ScanStep: prescribed x-positions with content head (discovers letter locations).
//! ReadStep: free attention that collects hidden states for cross-attention readout.
//! Both own their GRU and loc_head independently.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use flodl::autograd::Variable;
use flodl::nn::{Linear, Module, NamedInputModule, Parameter};
use flodl::tensor::{Result, Tensor, TensorOptions};

pub use flodl::Identity;

/// H0Init: learned initial hidden state expanded to batch dimension.
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
            .repeat(&[b, 1])
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.h0.clone()]
    }
}

/// ScanStep: wide-patch scan with prescribed x-positions.
///
/// 8 scan steps across a 256-wide word canvas. Each step:
/// - x = tanh(scan_xs[step_idx])  (learnable, constrained)
/// - y = tanh(loc_head(h))[y]     (free, from GRU)
/// - content_head(h)              (BCE: does this location have ink?)
pub struct ScanStep {
    sensor: Rc<dyn Module>,
    gru: flodl::GRUCell,
    loc_head: Linear,
    scan_xs: Vec<Parameter>,
    location: RefCell<Option<Variable>>,
    step_idx: RefCell<usize>,
    content_head: Option<Linear>,
    content_logits: Option<Rc<RefCell<Vec<Variable>>>>,
}

impl ScanStep {
    pub fn new(
        sensor: impl Module + 'static,
        hidden_dim: i64,
        n_scan: usize,
        content_head: Option<Linear>,
        content_logits: Option<Rc<RefCell<Vec<Variable>>>>,
    ) -> Result<Self> {
        let gru = flodl::GRUCell::new(hidden_dim, hidden_dim)?;
        let loc_head = Linear::new(hidden_dim, 2)?;

        // Prescribed x-positions spread across [-0.5, 0.5] via atanh.
        let mut scan_xs = Vec::with_capacity(n_scan);
        for i in 0..n_scan {
            let x_target = if n_scan == 1 {
                0.0
            } else {
                -0.5 + (i as f64) / (n_scan as f64 - 1.0)
            };
            // atanh(x) so that tanh(param) recovers x_target.
            let atanh_val = (0.5 * ((1.0 + x_target) / (1.0 - x_target)).ln()) as f32;
            let t = Tensor::from_f32(&[atanh_val], &[1], flodl::tensor::Device::CPU)?;
            scan_xs.push(Parameter {
                variable: Variable::new(t, true),
                name: format!("scan_x_{i}"),
            });
        }

        Ok(ScanStep {
            sensor: Rc::new(sensor),
            gru,
            loc_head,
            scan_xs,
            location: RefCell::new(None),
            step_idx: RefCell::new(0),
            content_head,
            content_logits,
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

            self.gru.forward_step(&glimpse, Some(h))?
        };

        // Content head: predict whether scan location has ink.
        if let (Some(head), Some(buf)) = (&self.content_head, &self.content_logits) {
            buf.borrow_mut().push(head.forward(&new_h)?);
        }

        // Location: prescribed x, free y.
        let raw = self.loc_head.forward(&new_h)?.tanh()?;
        let y = raw.select(1, 1)?.unsqueeze(1)?;

        let idx = *self.step_idx.borrow();
        let scan_x = &self.scan_xs[idx.min(self.scan_xs.len() - 1)];
        let x = scan_x.variable.tanh()?.expand(&[h.shape()[0], 1])?;
        *self.step_idx.borrow_mut() = idx + 1;

        let new_loc = x.cat(&y, 1)?;
        *self.location.borrow_mut() = Some(new_loc);

        Ok(new_h)
    }
}

impl Module for ScanStep {
    fn name(&self) -> &str { "scan_step" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.step(input, input)
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> { Some(self) }

    fn reset(&self) {
        *self.location.borrow_mut() = None;
        *self.step_idx.borrow_mut() = 0;
        if let Some(buf) = &self.content_logits {
            buf.borrow_mut().clear();
        }
    }

    fn detach_state(&self) {
        let mut loc = self.location.borrow_mut();
        if let Some(v) = loc.take() {
            *loc = Some(v.detach());
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.sensor.parameters();
        params.extend(self.gru.parameters());
        params.extend(self.loc_head.parameters());
        params.extend(self.scan_xs.iter().cloned());
        if let Some(head) = &self.content_head {
            params.extend(head.parameters());
        }
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

/// ReadStep: focused read that collects hidden states for cross-attention readout.
///
/// Free (x,y) positioning. Each step's hidden state is pushed to a shared buffer
/// that CrossAttentionReadout consumes after the loop.
pub struct ReadStep {
    sensor: Rc<dyn Module>,
    gru: flodl::GRUCell,
    loc_head: Linear,
    location: RefCell<Option<Variable>>,
    /// Collected hidden states [B, latent_dim] — one per read step.
    read_states: Rc<RefCell<Vec<Variable>>>,
}

impl ReadStep {
    pub fn new(
        sensor: impl Module + 'static,
        hidden_dim: i64,
        read_states: Rc<RefCell<Vec<Variable>>>,
    ) -> Result<Self> {
        Ok(ReadStep {
            sensor: Rc::new(sensor),
            gru: flodl::GRUCell::new(hidden_dim, hidden_dim)?,
            loc_head: Linear::new(hidden_dim, 2)?,
            location: RefCell::new(None),
            read_states,
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

            self.gru.forward_step(&glimpse, Some(h))?
        };

        // Collect hidden state for cross-attention readout.
        self.read_states.borrow_mut().push(new_h.clone());

        // Free (x, y) location update.
        let new_loc = self.loc_head.forward(&new_h)?.tanh()?;
        *self.location.borrow_mut() = Some(new_loc);

        Ok(new_h)
    }
}

impl Module for ReadStep {
    fn name(&self) -> &str { "read_step" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.step(input, input)
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> { Some(self) }

    fn reset(&self) {
        *self.location.borrow_mut() = None;
        self.read_states.borrow_mut().clear();
    }

    fn detach_state(&self) {
        let mut loc = self.location.borrow_mut();
        if let Some(v) = loc.take() {
            *loc = Some(v.detach());
        }
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

impl NamedInputModule for ReadStep {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        let image = refs.get("image").expect("ReadStep requires 'image' ref");
        self.step(input, image)
    }
}
