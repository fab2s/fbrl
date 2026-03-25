//! Graph-compatible modules for the attention pipeline.
//!
//! v2: ScanStep and AttentionStep each own their GRU and loc_head.
//! No shared Controller — clean sub-graph boundaries for word transfer.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use flodl::autograd::Variable;
use flodl::nn::{Linear, Module, NamedInputModule, Parameter};
use flodl::tensor::{Result, Tensor, TensorOptions};

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
            .repeat(&[b, 1])
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.h0.clone()]
    }
}

/// One-way handoff: scan writes its final location here, read picks it up
/// as its initial position (detached, so no gradient flows back to scan).
pub type LocationHandoff = Rc<RefCell<Option<Variable>>>;

/// ScanStep: wide-patch scan with learnable x-position, learned y.
/// Produces coarse overview fixations guided by attention_guide_loss.
///
/// Owns its own GRU and loc_head (independent from AttentionStep).
///
/// **Standalone mode** (scan_start empty): x = tanh(scan_x[step]), y = tanh(loc_head(h))[y].
/// **Composed mode** (scan_start provided): free (x, y) = tanh(loc_head(h)), starting
/// from SubScan's position. The learnable scan_xs are unused in composed mode.
pub struct ScanStep {
    sensor: Rc<dyn Module>,
    gru: flodl::GRUCell,
    loc_head: Linear,
    scan_xs: Vec<Parameter>,
    location: RefCell<Option<Variable>>,
    step_idx: RefCell<usize>,
    content_head: Option<Linear>,
    content_logits: Option<Rc<RefCell<Vec<Variable>>>>,
    handoff: LocationHandoff,
    /// External starting position (for composition with SubScan).
    /// When populated before forward, ScanStep uses it as initial location
    /// and switches to free (x, y) mode.
    scan_start: LocationHandoff,
    /// Whether the current forward pass is using an external start position.
    from_external: Cell<bool>,
}

impl ScanStep {
    pub fn new(
        sensor: impl Module + 'static,
        hidden_dim: i64,
        n_scan: usize,
        content_head: Option<Linear>,
        content_logits: Option<Rc<RefCell<Vec<Variable>>>>,
        handoff: LocationHandoff,
        scan_start: LocationHandoff,
    ) -> Result<Self> {
        let gru = flodl::GRUCell::new(hidden_dim, hidden_dim)?;
        let loc_head = Linear::new(hidden_dim, 2)?;

        let mut scan_xs = Vec::with_capacity(n_scan);
        for i in 0..n_scan {
            let init_val = if n_scan == 1 {
                0.0
            } else {
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
            gru,
            loc_head,
            scan_xs,
            location: RefCell::new(None),
            step_idx: RefCell::new(0),
            content_head,
            content_logits,
            handoff,
            scan_start,
            from_external: Cell::new(false),
        })
    }

    fn step(&self, h: &Variable, image: &Variable) -> Result<Variable> {
        let new_h = {
            if self.location.borrow().is_none() {
                // Check for external start position (composed mode with SubScan).
                let external = self.scan_start.borrow_mut().take();
                self.from_external.set(external.is_some());
                let loc = match external {
                    Some(pos) => pos,
                    None => {
                        let batch = h.shape()[0];
                        let device = h.data().device();
                        let zeros = Tensor::zeros(&[batch, 2], TensorOptions { device, ..Default::default() })?;
                        Variable::new(zeros, false)
                    }
                };
                *self.location.borrow_mut() = Some(loc);
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
            let logit = head.forward(&new_h)?; // [B, 1]
            buf.borrow_mut().push(logit);
        }

        // Location update.
        let raw = self.loc_head.forward(&new_h)?.tanh()?;
        let new_loc = if self.from_external.get() {
            // Composed mode: free (x, y) refinement from SubScan position.
            // Both channels of loc_head are active — scan can follow the letter
            // anywhere on the word image, not just along a learnable x-sweep.
            raw
        } else {
            // Standalone: learnable x, free y from loc_head.
            let y = raw.select(1, 1)?.unsqueeze(1)?; // [B, 1]
            let idx = *self.step_idx.borrow();
            let scan_x = &self.scan_xs[idx.min(self.scan_xs.len() - 1)];
            let x = scan_x.variable.tanh()?.expand(&[h.shape()[0], 1])?; // [B, 1]
            *self.step_idx.borrow_mut() = idx + 1;
            x.cat(&y, 1)?
        };

        *self.location.borrow_mut() = Some(new_loc.clone());

        // Write detached copy to handoff — read phase picks this up as initial position.
        *self.handoff.borrow_mut() = Some(new_loc.detach());

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
        self.from_external.set(false);
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

/// AttentionStep: the read loop body with its own GRU and loc_head.
///
/// Receives h as stream, image via ref, manages location as internal state.
/// Free (x,y) positioning.
///
/// When a `LocationHandoff` is provided, the first read step starts at the
/// scan's final position (detached) instead of (0,0).
pub struct AttentionStep {
    sensor: Rc<dyn Module>,
    gru: flodl::GRUCell,
    loc_head: Linear,
    location: RefCell<Option<Variable>>,
    handoff: Option<LocationHandoff>,
}

impl AttentionStep {
    pub fn new(
        sensor: impl Module + 'static,
        hidden_dim: i64,
        handoff: Option<LocationHandoff>,
    ) -> Result<Self> {
        Ok(AttentionStep {
            sensor: Rc::new(sensor),
            gru: flodl::GRUCell::new(hidden_dim, hidden_dim)?,
            loc_head: Linear::new(hidden_dim, 2)?,
            location: RefCell::new(None),
            handoff,
        })
    }

    fn step(&self, h: &Variable, image: &Variable) -> Result<Variable> {
        let new_h = {
            // Lazy init: use scan handoff if available, otherwise zeros.
            if self.location.borrow().is_none() {
                let init_loc = self.handoff.as_ref()
                    .and_then(|h| h.borrow_mut().take());
                let loc = match init_loc {
                    Some(scan_loc) => scan_loc,
                    None => {
                        let batch = h.shape()[0];
                        let device = h.data().device();
                        Variable::new(
                            Tensor::zeros(&[batch, 2], TensorOptions { device, ..Default::default() })?,
                            false,
                        )
                    }
                };
                *self.location.borrow_mut() = Some(loc);
            }

            let loc_guard = self.location.borrow();
            let loc = loc_guard.as_ref().unwrap();

            // Sensor extracts glimpse at current location
            let mut refs = HashMap::new();
            refs.insert("location".to_string(), loc.clone());
            let glimpse = self.sensor.as_named_input().unwrap()
                .forward_named(image, &refs)?;

            // GRU update
            self.gru.forward_step(&glimpse, Some(h))?
        }; // loc_guard dropped here

        // Update location
        let new_loc = self.loc_head.forward(&new_h)?.tanh()?;
        *self.location.borrow_mut() = Some(new_loc);

        Ok(new_h)
    }
}

impl Module for AttentionStep {
    fn name(&self) -> &str { "attention_step" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.step(input, input)
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> {
        Some(self)
    }

    fn reset(&self) {
        *self.location.borrow_mut() = None;
        // Handoff is NOT cleared here — it persists from scan to read within a forward pass.
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
