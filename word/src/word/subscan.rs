//! SubScan: bounded-region letter localization (Graph-based).
//!
//! Given a region of a word image (defined by center + half-width), SubScan
//! takes N short, wide, blurred glimpses within the bounded region and
//! infers a letter center position for handoff to the letter model.
//!
//! The output position is free — it does not need to coincide with any
//! glimpse location. SubScan observes partial ink structure and infers
//! where the letter center must be.
//!
//! ## Architecture (FlowBuilder graph)
//!
//! ```text
//! image → GaussianBlur(σ) → tag("blurred")
//!       → H0Init(hidden_dim)
//!       → Loop(SubScanStep, n_glimpses).Using("blurred", "region_center", "region_half_w")
//!           .Tag("scan")
//! ```
//!
//! ## Region bounding
//!
//! SubScan's location head is reparameterized to stay within its region:
//! ```text
//! x = region_center + region_half_width * tanh(raw_x)
//! y = tanh(raw_y)   // full vertical range
//! ```
//!
//! ## Glimpse design
//!
//! - **Short and wide** (~8x28 pixels): horizontal localization emphasis
//! - **2/3 letter width**: one glimpse cannot see the whole letter
//! - **Blurred**: gaussian blur on the full image before glimpse extraction
//! - **Two free glimpses**: minimum for triangulating the letter center

use std::cell::RefCell;
use std::collections::HashMap;

use flodl::autograd::Variable;
use flodl::graph::{FlowBuilder, Graph};
use flodl::nn::{GaussianBlur, Linear, Module, NamedInputModule, Parameter};
use flodl::tensor::Result;

use fbrl::letter::H0Init;
use super::glimpse::GlimpseSensor;

/// SubScan configuration.
pub struct SubScanConfig {
    /// Hidden/latent dimension for GRU and sensor output.
    pub hidden_dim: i64,
    /// Glimpse patch height (short: ~8 pixels).
    pub patch_h: i64,
    /// Glimpse patch width (wide: ~28 pixels, ~2/3 letter width).
    pub patch_w: i64,
    /// Number of resolution scales in the glimpse sensor.
    pub n_scales: usize,
    /// Number of glimpse steps (default: 2).
    pub n_glimpses: usize,
    /// Gaussian blur sigma applied to the full image before glimpse extraction.
    /// Ensures SubScan sees ink density, not letterform detail.
    pub blur_sigma: f64,
}

// ── SubScanStep: loop body NamedInputModule ──────────────────────────

/// One step of the SubScan loop.
///
/// Receives hidden state `h` as stream and accesses blurred image,
/// region center, and region half-width from graph refs.
///
/// Internal state: current location (lazy-initialized to region center).
struct SubScanStep {
    sensor: GlimpseSensor,
    gru: flodl::GRUCell,
    loc_head: Linear,
    location: RefCell<Option<Variable>>,
}

impl SubScanStep {
    fn new(sensor: GlimpseSensor, gru: flodl::GRUCell, loc_head: Linear) -> Self {
        SubScanStep {
            sensor,
            gru,
            loc_head,
            location: RefCell::new(None),
        }
    }

    fn step(
        &self,
        h: &Variable,
        blurred: &Variable,
        region_center: &Variable,
        region_half_w: &Variable,
    ) -> Result<Variable> {
        let new_h = {
            // Lazy init: start at region_center [B, 2] (noisy x and y).
            if self.location.borrow().is_none() {
                *self.location.borrow_mut() = Some(region_center.clone());
            }

            let loc_guard = self.location.borrow();
            let loc = loc_guard.as_ref().unwrap();

            let glimpse = self.sensor.sense(blurred, loc)?;
            self.gru.forward_step(&glimpse, Some(h))?
        }; // loc_guard dropped

        // Location update: x bounded to region, y free.
        let raw = self.loc_head.forward(&new_h)?.tanh()?; // [B, 2] in [-1, 1]
        let raw_x = raw.narrow(1, 0, 1)?;                  // [B, 1]
        let raw_y = raw.narrow(1, 1, 1)?;                  // [B, 1]
        let center_x = region_center.narrow(1, 0, 1)?;     // [B, 1]
        let x = center_x.add(&region_half_w.mul(&raw_x)?)?;
        let new_loc = x.cat(&raw_y, 1)?;                   // [B, 2]

        *self.location.borrow_mut() = Some(new_loc);
        Ok(new_h)
    }
}

impl Module for SubScanStep {
    fn name(&self) -> &str { "subscan_step" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        // Not called directly — NamedInputModule path is used.
        self.step(input, input, input, input)
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> { Some(self) }

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
        params.extend(self.gru.parameters());
        params.extend(self.loc_head.parameters());
        params
    }

    fn trace(&self) -> Option<Variable> {
        self.location.borrow().clone()
    }
}

impl NamedInputModule for SubScanStep {
    fn forward_named(
        &self,
        input: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        let blurred = refs.get("blurred").expect("SubScanStep requires 'blurred' ref");
        let region_center = refs.get("region_center")
            .expect("SubScanStep requires 'region_center' ref");
        let region_half_w = refs.get("region_half_w")
            .expect("SubScanStep requires 'region_half_w' ref");
        self.step(input, blurred, region_center, region_half_w)
    }
}

// ── SubScanModel: Graph wrapper ──────────────────────────────────────

/// SubScan model built as a computation graph.
///
/// Two short, wide, blurred glimpses within a bounded region, then infer
/// the letter center position. The output position does not need to
/// coincide with either glimpse location — it is an inference from
/// accumulated evidence.
pub struct SubScanModel {
    pub graph: Graph,
}

impl SubScanModel {
    /// Create the SubScan model as a Graph.
    pub fn new(cfg: &SubScanConfig) -> Result<Self> {
        let sensor = GlimpseSensor::new(
            cfg.patch_h, cfg.patch_w, cfg.n_scales, cfg.hidden_dim,
        )?;
        let gru = flodl::GRUCell::new(cfg.hidden_dim, cfg.hidden_dim)?;
        let loc_head = Linear::new(cfg.hidden_dim, 2)?;

        let step = SubScanStep::new(sensor, gru, loc_head);

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
    /// - `region_center`: `[B, 2]` — noisy starting position (x, y) in normalized coords.
    /// - `region_half_w`: `[B, 1]` — normalized half-width bounding the x-axis.
    ///
    /// Returns `[B, 2]` — midpoint of all glimpse positions (triangulated center).
    ///
    /// Using the midpoint instead of the last position creates an inductive bias:
    /// the glimpses must bracket the letter to produce a good center estimate.
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
        let traces = self.graph.traces("scan").unwrap_or_default();
        assert!(!traces.is_empty(), "no scan traces");
        // Midpoint of all glimpse positions: triangulated center.
        // Each trace is [B, 2]. Stack → [B, N, 2], mean over dim 1 → [B, 2].
        let stacked = Variable::stack(&traces, 1)?;
        stacked.mean_dim(1, false)
    }

    /// Location traces from the most recent forward pass (one per glimpse step).
    pub fn locations(&self) -> Vec<Variable> {
        self.graph.traces("scan").unwrap_or_default()
    }

    /// All learnable parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        self.graph.parameters()
    }
}
