//! Graph-compatible modules for the attention pipeline.

use std::cell::RefCell;
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

/// One-way handoff: scan writes its final location here, read picks it up
/// as its initial position (detached, so no gradient flows back to scan).
pub type LocationHandoff = Rc<RefCell<Option<Variable>>>;

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
    content_head: Option<Linear>,
    content_logits: Option<Rc<RefCell<Vec<Variable>>>>,
    handoff: LocationHandoff,
}

impl ScanStep {
    pub fn new(
        sensor: impl Module + 'static,
        controller: Rc<Controller>,
        n_scan: usize,
        content_head: Option<Linear>,
        content_logits: Option<Rc<RefCell<Vec<Variable>>>>,
        handoff: LocationHandoff,
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
            content_head,
            content_logits,
            handoff,
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

        // Content head: predict whether scan location has ink.
        if let (Some(head), Some(buf)) = (&self.content_head, &self.content_logits) {
            let logit = head.forward(&new_h)?; // [B, 1]
            buf.borrow_mut().push(logit);
        }

        // Location: learnable x, free y from loc_head
        let raw = self.controller.loc_head.forward(&new_h)?.tanh()?;
        let y = raw.select(1, 1)?.unsqueeze(1)?; // [B, 1]

        let idx = *self.step_idx.borrow();
        let scan_x = &self.scan_xs[idx.min(self.scan_xs.len() - 1)];
        let x = scan_x.variable.tanh()?.expand(&[h.shape()[0], 1])?; // [B, 1]
        *self.step_idx.borrow_mut() = idx + 1;

        let new_loc = x.cat(&y, 1)?; // [B, 2]
        *self.location.borrow_mut() = Some(new_loc.clone());

        // Write detached copy to handoff — read phase picks this up as initial position.
        // Detached so no gradient flows from read back through the handoff to scan.
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
        // Only scan-specific params: sensor + scan_xs + content_head.
        // GRU and loc_head are shared — collected by AttentionStep.
        let mut params = self.sensor.parameters();
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

/// AttentionStep is the read loop body: receives h as stream, image via ref,
/// manages location as internal recurrent state. Free (x,y) positioning.
///
/// When a `LocationHandoff` is provided, the first read step starts at the
/// scan's final position (detached) instead of (0,0). This matches Python's
/// behavior where a single `location` variable flows from scan to read.
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
    handoff: Option<LocationHandoff>,
}

impl AttentionStep {
    /// Create with a new (unshared) controller. For standalone use or backward compat.
    pub fn new(sensor: impl Module + 'static, hidden_dim: i64) -> Result<Self> {
        Ok(AttentionStep {
            sensor: Rc::new(sensor),
            controller: Rc::new(Controller::new(hidden_dim)?),
            location: RefCell::new(None),
            handoff: None,
        })
    }

    /// Create with a shared controller and location handoff (for scan+read architecture).
    /// The read phase starts at the scan's final position (detached).
    pub fn with_shared(sensor: impl Module + 'static, controller: Rc<Controller>, handoff: LocationHandoff) -> Self {
        AttentionStep {
            sensor: Rc::new(sensor),
            controller,
            location: RefCell::new(None),
            handoff: Some(handoff),
        }
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
            self.controller.gru.forward_step(&glimpse, Some(h))?
        }; // loc_guard dropped here

        // Update location
        let new_loc = self.controller.loc_head.forward(&new_h)?.tanh()?;
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

/// CombinedStep: flat scan+read loop matching Python's `encode_scan_read`.
///
/// A single module runs n_scan + n_read steps with continuous location flow:
/// - Steps 0..n_scan: scan sensor (wide patch), learnable x + free y, content head
/// - Steps n_scan..total: read sensor (square patch), free (x,y)
///
/// All positions are relative to origin. The `loc_head` outputs offsets from
/// origin, `location_fc` in the sensor sees these relative offsets, and
/// `grid_sample` receives `origin + offset` for correct image sampling.
///
/// The graph sees one loop of `n_scan + n_read` iterations. Traces are split
/// into scan and read portions by the model's `forward()`.
pub struct CombinedStep {
    scan_sensor: Rc<dyn Module>,
    read_sensor: Rc<dyn Module>,
    controller: Rc<Controller>,
    n_scan: usize,
    scan_xs: Vec<Parameter>,
    /// Current position as relative offset from origin.
    location: RefCell<Option<Variable>>,
    /// Origin position (set from graph input on first step, constant within a forward pass).
    origin: RefCell<Option<Variable>>,
    step_idx: RefCell<usize>,
    content_head: Option<Linear>,
    content_logits: Option<Rc<RefCell<Vec<Variable>>>>,
}

impl CombinedStep {
    pub fn new(
        scan_sensor: impl Module + 'static,
        read_sensor: impl Module + 'static,
        controller: Rc<Controller>,
        n_scan: usize,
        content_head: Option<Linear>,
        content_logits: Option<Rc<RefCell<Vec<Variable>>>>,
    ) -> Result<Self> {
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
        Ok(CombinedStep {
            scan_sensor: Rc::new(scan_sensor),
            read_sensor: Rc::new(read_sensor),
            controller,
            n_scan,
            scan_xs,
            location: RefCell::new(None),
            origin: RefCell::new(None),
            step_idx: RefCell::new(0),
            content_head,
            content_logits,
        })
    }

    fn step(&self, h: &Variable, image: &Variable, origin: &Variable) -> Result<Variable> {
        let idx = *self.step_idx.borrow();
        let is_scan = idx < self.n_scan;

        // Store origin on first step (constant within a forward pass).
        if self.origin.borrow().is_none() {
            *self.origin.borrow_mut() = Some(origin.clone());
        }

        let new_h = {
            // Lazy init: start at origin position.
            if self.location.borrow().is_none() {
                *self.location.borrow_mut() = Some(origin.clone());
            }

            let loc_guard = self.location.borrow();
            let location = loc_guard.as_ref().unwrap();

            // location_fc sees position relative to origin (always near zero).
            // grid_sample sees absolute position (unchanged from old code).
            let relative = location.sub(origin)?;

            let mut refs = HashMap::new();
            refs.insert("location".to_string(), location.clone());
            refs.insert("relative_location".to_string(), relative);

            // Pick sensor based on phase.
            let sensor = if is_scan { &self.scan_sensor } else { &self.read_sensor };
            let glimpse = sensor.as_named_input().unwrap()
                .forward_named(image, &refs)?;

            self.controller.gru.forward_step(&glimpse, Some(h))?
        };

        // Content head (scan phase only).
        if is_scan
            && let (Some(head), Some(buf)) = (&self.content_head, &self.content_logits) {
                let logit = head.forward(&new_h)?;
                buf.borrow_mut().push(logit);
        }

        // Location update — all positions are absolute (same as old code).
        // Origin translation is isolated: only location_fc sees the difference.
        let new_loc = if is_scan {
            // Scan: learnable x, free y from loc_head.
            let raw = self.controller.loc_head.forward(&new_h)?.tanh()?;
            let y = raw.select(1, 1)?.unsqueeze(1)?;
            let scan_x = &self.scan_xs[idx.min(self.scan_xs.len() - 1)];
            let x = scan_x.variable.tanh()?.expand(&[h.shape()[0], 1])?;
            x.cat(&y, 1)?
        } else {
            // Read: free (x, y).
            self.controller.loc_head.forward(&new_h)?.tanh()?
        };

        *self.location.borrow_mut() = Some(new_loc);
        *self.step_idx.borrow_mut() = idx + 1;

        Ok(new_h)
    }
}

impl Module for CombinedStep {
    fn name(&self) -> &str { "combined_step" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        // Not called directly — NamedInputModule path provides origin via refs.
        self.step(input, input, input)
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> {
        Some(self)
    }

    fn reset(&self) {
        *self.location.borrow_mut() = None;
        *self.origin.borrow_mut() = None;
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
        let mut params = self.scan_sensor.parameters();
        params.extend(self.read_sensor.parameters());
        params.extend(self.controller.parameters());
        params.extend(self.scan_xs.iter().cloned());
        if let Some(head) = &self.content_head {
            params.extend(head.parameters());
        }
        params
    }

    fn sub_modules(&self) -> Vec<Rc<dyn Module>> {
        vec![self.scan_sensor.clone(), self.read_sensor.clone()]
    }

    fn trace(&self) -> Option<Variable> {
        self.location.borrow().clone()
    }
}

impl NamedInputModule for CombinedStep {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        let image = refs.get("image").expect("CombinedStep requires 'image' ref");
        let origin = refs.get("origin").expect("CombinedStep requires 'origin' ref");
        self.step(input, image, origin)
    }
}
