//! SubScan: triangle-glimpse letter localization (Graph-based).
//!
//! Given a region of a word image, SubScan takes 3 blurred glimpses in a
//! triangle pattern to sense ink structure, then outputs the letter center_x.
//!
//! ## Triangle geometry
//!
//! ```text
//!         apex (cx, +h/2)
//!         /\
//!        /  \
//!       /    \
//!      /______\
//!   left       right
//! (cx-w, 0)  (cx+w, 0)
//! ```
//!
//! - **Left base**: senses left ink edge
//! - **Right base**: senses right ink edge
//! - **Apex**: senses vertical structure (distinguishes O from IL)
//!
//! The base width `w` is bounded: min (I-width) to max (<MM-width).
//! The triangle height is fixed (known from line height).
//! SubScan outputs center_x = midpoint of base.
//!
//! ## Architecture (FlowBuilder graph)
//!
//! ```text
//! image → GaussianBlur(σ) → tag("blurred")
//!       → H0Init(hidden_dim)
//!       → Input(["region_center", "region_half_w"])
//!       → Loop(TriangleStep, 3).Using("blurred", "region_center", "region_half_w")
//!           .Tag("scan")
//! ```
//!
//! ## Training
//!
//! Supervised MSE on center_x. No oracle, no REINFORCE.
//! Trained independently — compose with letter model at inference only.

use std::cell::RefCell;
use std::collections::HashMap;

use flodl::autograd::Variable;
use flodl::graph::{FlowBuilder, Graph};
use flodl::nn::{GaussianBlur, Linear, Module, NamedInputModule, Parameter};
use flodl::tensor::{Result, Tensor};

use fbrl::letter::H0Init;
use super::glimpse::GlimpseSensor;

/// SubScan configuration.
pub struct SubScanConfig {
    /// Hidden/latent dimension for GRU and sensor output.
    pub hidden_dim: i64,
    /// Glimpse patch height.
    pub patch_h: i64,
    /// Glimpse patch width.
    pub patch_w: i64,
    /// Number of resolution scales in the glimpse sensor.
    pub n_scales: usize,
    /// Number of glimpse steps (default: 3 for one triangle pass).
    pub n_glimpses: usize,
    /// Gaussian blur sigma applied to the full image.
    pub blur_sigma: f64,

    // Triangle geometry bounds (normalized coords).
    /// Minimum base half-width (about I-width / 2).
    pub min_base_hw: f64,
    /// Maximum base half-width (less than one letter spacing / 2).
    pub max_base_hw: f64,
    /// Triangle height (fixed, about half a letter height).
    pub triangle_height: f64,
}

// ── TriangleStep: loop body ─────────────────────────────────────────

/// One step of the triangle SubScan loop.
///
/// Steps cycle through triangle vertices: left base → right base → apex.
/// After each glimpse, the GRU updates hidden state and the output head
/// refines center_x and base_half_width estimates.
struct TriangleStep {
    sensor: GlimpseSensor,
    gru: flodl::GRUCell,
    output_head: Linear, // hidden_dim → 2 (raw_cx, raw_bw)

    // Triangle bounds
    min_base_hw: f64,
    max_base_hw: f64,
    triangle_h: f64,

    // State
    step_idx: RefCell<usize>,
    center_x: RefCell<Option<Variable>>,
    base_half_w: RefCell<Option<Variable>>,
}

impl TriangleStep {
    fn new(
        sensor: GlimpseSensor,
        gru: flodl::GRUCell,
        output_head: Linear,
        min_base_hw: f64,
        max_base_hw: f64,
        triangle_h: f64,
    ) -> Self {
        TriangleStep {
            sensor, gru, output_head,
            min_base_hw, max_base_hw, triangle_h,
            step_idx: RefCell::new(0),
            center_x: RefCell::new(None),
            base_half_w: RefCell::new(None),
        }
    }

    fn step(
        &self,
        h: &Variable,
        blurred: &Variable,
        region_center: &Variable,
        region_half_w: &Variable,
    ) -> Result<Variable> {
        let idx = *self.step_idx.borrow();
        let dev = blurred.device();
        let b = h.shape()[0];

        // Initialize on first step.
        if idx == 0 {
            let cx = region_center.narrow(1, 0, 1)?; // [B, 1]
            *self.center_x.borrow_mut() = Some(cx);

            let init_hw = (self.min_base_hw + self.max_base_hw) / 2.0;
            let bw_data = vec![init_hw as f32; b as usize];
            let bw = Variable::new(
                Tensor::from_f32(&bw_data, &[b, 1], dev)?, false,
            );
            *self.base_half_w.borrow_mut() = Some(bw);
        }

        let cx = self.center_x.borrow().as_ref().unwrap().clone();
        let bw = self.base_half_w.borrow().as_ref().unwrap().clone();

        // Glimpse position from triangle vertex.
        let glimpse_pos = match idx % 3 {
            0 => {
                // Left base: (center_x - base_half_w, 0)
                let x = cx.sub(&bw)?;
                let y_data = vec![0.0f32; b as usize];
                let y = Variable::new(Tensor::from_f32(&y_data, &[b, 1], dev)?, false);
                x.cat(&y, 1)?
            }
            1 => {
                // Right base: (center_x + base_half_w, 0)
                let x = cx.add(&bw)?;
                let y_data = vec![0.0f32; b as usize];
                let y = Variable::new(Tensor::from_f32(&y_data, &[b, 1], dev)?, false);
                x.cat(&y, 1)?
            }
            _ => {
                // Apex: (center_x, +triangle_height/2)
                let y_val = (self.triangle_h / 2.0) as f32;
                let y_data = vec![y_val; b as usize];
                let y = Variable::new(Tensor::from_f32(&y_data, &[b, 1], dev)?, false);
                cx.cat(&y, 1)?
            }
        };

        // Extract glimpse and update GRU.
        let glimpse = self.sensor.sense(blurred, &glimpse_pos)?;
        let new_h = self.gru.forward_step(&glimpse, Some(h))?;

        // Update center_x and base_half_w from output head.
        let raw = self.output_head.forward(&new_h)?.tanh()?; // [B, 2] in [-1, 1]
        let raw_cx = raw.narrow(1, 0, 1)?;
        let raw_bw = raw.narrow(1, 1, 1)?;

        // center_x: bounded to region via tanh reparameterization.
        let center_ref = region_center.narrow(1, 0, 1)?;
        let new_cx = center_ref.add(&region_half_w.mul(&raw_cx)?)?;

        // base_half_w: bounded to [min, max].
        // tanh in [-1,1] → (tanh+1)/2 in [0,1] → min + range * [0,1]
        let range = self.max_base_hw - self.min_base_hw;
        let new_bw = raw_bw.add_scalar(1.0)?
            .mul_scalar(0.5 * range)?
            .add_scalar(self.min_base_hw)?;

        *self.center_x.borrow_mut() = Some(new_cx);
        *self.base_half_w.borrow_mut() = Some(new_bw);
        *self.step_idx.borrow_mut() = idx + 1;

        Ok(new_h)
    }
}

impl Module for TriangleStep {
    fn name(&self) -> &str { "triangle_step" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.step(input, input, input, input)
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> { Some(self) }

    fn reset(&self) {
        *self.step_idx.borrow_mut() = 0;
        *self.center_x.borrow_mut() = None;
        *self.base_half_w.borrow_mut() = None;
    }

    fn detach_state(&self) {
        let mut cx = self.center_x.borrow_mut();
        if let Some(v) = cx.take() { *cx = Some(v.detach()); }
        let mut bw = self.base_half_w.borrow_mut();
        if let Some(v) = bw.take() { *bw = Some(v.detach()); }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.sensor.parameters();
        params.extend(self.gru.parameters());
        params.extend(self.output_head.parameters());
        params
    }

    fn trace(&self) -> Option<Variable> {
        // Return center_x as [B, 1] trace for the graph.
        self.center_x.borrow().clone()
    }
}

impl NamedInputModule for TriangleStep {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        let blurred = refs.get("blurred")
            .expect("TriangleStep requires 'blurred' ref");
        let region_center = refs.get("region_center")
            .expect("TriangleStep requires 'region_center' ref");
        let region_half_w = refs.get("region_half_w")
            .expect("TriangleStep requires 'region_half_w' ref");
        self.step(input, blurred, region_center, region_half_w)
    }
}

// ── SubScanModel: Graph wrapper ──────────────────────────────────────

/// SubScan model with triangle-constrained glimpses.
///
/// Three blurred glimpses (left base, right base, apex) sense ink structure.
/// Outputs center_x for handoff to the letter model.
pub struct SubScanModel {
    pub graph: Graph,
}

impl SubScanModel {
    /// Create the triangle SubScan model.
    pub fn new(cfg: &SubScanConfig) -> Result<Self> {
        let sensor = GlimpseSensor::new(
            cfg.patch_h, cfg.patch_w, cfg.n_scales, cfg.hidden_dim,
        )?;
        let gru = flodl::GRUCell::new(cfg.hidden_dim, cfg.hidden_dim)?;
        let output_head = Linear::new(cfg.hidden_dim, 2)?;

        let step = TriangleStep::new(
            sensor, gru, output_head,
            cfg.min_base_hw, cfg.max_base_hw, cfg.triangle_height,
        );

        let graph = FlowBuilder::from(GaussianBlur::new(cfg.blur_sigma))
            .label("SubScan")
            .tag("blurred")
            .input(&["region_center", "region_half_w"])
            .through(H0Init::new(cfg.hidden_dim)?)
            .loop_body(step).for_n(cfg.n_glimpses)
                .using(&["blurred", "region_center", "region_half_w"])
                .tag("scan")
            .build()?;

        Ok(SubScanModel { graph })
    }

    /// Localize a letter within a bounded region.
    ///
    /// - `image`: `[B, 1, H, W]` — full word image.
    /// - `region_center`: `[B, 2]` — noisy starting position (x, y).
    /// - `region_half_w`: `[B, 1]` — region half-width bounding center_x.
    ///
    /// Returns `[B, 1]` — predicted center_x.
    pub fn forward(
        &self,
        image: &Variable,
        region_center: &Variable,
        region_half_w: &Variable,
    ) -> Result<Variable> {
        self.graph.forward_multi(&[
            image.clone(),
            region_center.clone(),
            region_half_w.clone(),
        ])?;

        // The last trace is center_x [B, 1] after all 3 glimpses.
        let traces = self.graph.traces("scan").unwrap_or_default();
        assert!(!traces.is_empty(), "no scan traces");
        Ok(traces.last().unwrap().clone())
    }

    /// Center_x traces from each step (one per glimpse).
    pub fn center_traces(&self) -> Vec<Variable> {
        self.graph.traces("scan").unwrap_or_default()
    }

    /// All learnable parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        self.graph.parameters()
    }
}
