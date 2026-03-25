//! LetterModel — single-letter recognition via foveal attention.
//!
//! Built as a graph with two inputs (image, case label):
//!
//! ```text
//! image → Tag("image") → H0Init
//!   → [Loop(ScanStep).Using("image").Tag("scan")]     // optional scan phase
//!   → Loop(AttentionStep).Using("image").Tag("read")   // read phase
//!   → Linear → Tag("latent") → Fork(letterHead).Tag("heads_0")
//!   → Fork(caseHead).Tag("heads_1") → Decoder.Using("latent", "case") → Tag("recon")
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use flodl::autograd::Variable;
use flodl::graph::{FlowBuilder, Graph};
use flodl::nn::{Buffer, Linear, Module, NamedInputModule, Parameter};
use flodl::tensor::{Device, Result};

use super::decoder::VisualDecoder;
use super::glimpse::GlimpseSensor;
use super::modules::{AttentionStep, H0Init, Identity, LocationHandoff, ScanStep};

/// Wraps an Rc<VisualDecoder> so it can live in the graph while
/// LetterModel retains a reference for recode.
struct SharedDecoder(Rc<VisualDecoder>);

impl Module for SharedDecoder {
    fn name(&self) -> &str { self.0.name() }
    fn forward(&self, input: &Variable) -> Result<Variable> { self.0.forward(input) }
    fn as_named_input(&self) -> Option<&dyn NamedInputModule> { Some(self) }
    fn parameters(&self) -> Vec<Parameter> { self.0.parameters() }
    fn buffers(&self) -> Vec<Buffer> { self.0.buffers() }
    fn set_training(&self, training: bool) { self.0.set_training(training) }
    fn move_to_device(&self, device: Device) { self.0.move_to_device(device) }
}

impl NamedInputModule for SharedDecoder {
    fn forward_named(
        &self,
        _stream: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        let latent = refs.get("latent").expect("VisualDecoder requires 'latent' ref");
        let case_label = refs.get("case").expect("VisualDecoder requires 'case' ref");
        self.0.forward(&latent.cat(case_label, 1)?)
    }
}

/// Holds everything from a forward pass.
pub struct LetterResult {
    pub recon: Variable,             // [B, 1, 128, 128]
    pub letter_logits: Variable,     // [B, n_classes]
    pub case_logits: Variable,       // [B, 2]
    pub scan_locations: Vec<Variable>,  // scan fixation trajectory
    pub read_locations: Vec<Variable>,  // read fixation trajectory
    pub latent: Variable,            // [B, latent_dim]
}

/// Single-letter model built as a computation graph.
///
/// v2: separate ScanStep and AttentionStep loops, each with own GRU.
/// Traces are collected per-loop via separate tags ("scan" / "read").
///
/// **Composition support:** when used inside a SubScan model, call
/// [`set_scan_start`] before each forward pass to inject SubScan's
/// output position. ScanStep will start from that position and use
/// free (x, y) mode instead of learnable x-sweep.
pub struct LetterModel {
    pub graph: Graph,
    /// Decoder reference for recode (shared with graph node via Rc).
    pub decoder: Rc<VisualDecoder>,
    /// Content logits buffer shared with ScanStep (populated during forward).
    content_logits_buf: Rc<RefCell<Vec<Variable>>>,
    /// Number of scan steps — used to know whether to look for scan traces.
    n_scan: usize,
    /// Shared with ScanStep — write here before forward to inject an
    /// external starting position (from SubScan).
    scan_start: Option<LocationHandoff>,
    /// Scan-phase parameters (h0 + ScanStep), collected before graph build.
    /// In composed mode these stay trainable while read params are frozen.
    scan_phase_params: Vec<Parameter>,
}

impl LetterModel {
    /// Create the single-letter model.
    ///
    /// n_scan=0: read-only (v1 baseline).
    /// n_scan=1: 1 wide scan + n_read reads (v2, matches Python v7).
    pub fn new(
        n_classes: usize,
        n_scan: usize,
        n_read: usize,
        patch_size: i64,
        scan_patch_w: i64,
        n_scales: usize,
        latent_dim: i64,
    ) -> Result<Self> {
        let letter_head = Linear::new(latent_dim, n_classes as i64)?;
        let case_head = Linear::new(latent_dim, 2)?;
        let decoder = Rc::new(VisualDecoder::new(latent_dim + 1, 128, 128)?);

        let content_logits_buf = Rc::new(RefCell::new(Vec::new()));
        let handoff: LocationHandoff = Rc::new(RefCell::new(None));

        // Build graph: H0Init → [ScanLoop] → ReadLoop → heads → decoder
        // Collect scan-phase params before modules are consumed by the builder.
        let h0_init = H0Init::new(latent_dim)?;
        let mut scan_phase_params = h0_init.parameters();

        let mut builder = FlowBuilder::from(Identity)
            .label("LetterModel")
            .tag("image")
            .input(&["case"])
            .through(h0_init);

        // Optional scan phase (own GRU + loc_head).
        let scan_start: Option<LocationHandoff> = if n_scan > 0 {
            let scan_start: LocationHandoff = Rc::new(RefCell::new(None));
            let scan_sensor = GlimpseSensor::new(patch_size, scan_patch_w, n_scales, latent_dim)?;
            let content_head = Linear::new(latent_dim, 1)?;
            let scan_step = ScanStep::new(
                scan_sensor, latent_dim, n_scan,
                Some(content_head), Some(content_logits_buf.clone()),
                handoff.clone(),
                scan_start.clone(),
            )?;
            scan_phase_params.extend(scan_step.parameters());
            builder = builder
                .loop_body(scan_step).for_n(n_scan)
                .using(&["image"]).tag("scan");
            Some(scan_start)
        } else {
            None
        };

        // Read phase (own GRU + loc_head, picks up scan handoff if present).
        let read_sensor = GlimpseSensor::new(patch_size, patch_size, n_scales, latent_dim)?;
        let read_step = AttentionStep::new(
            read_sensor, latent_dim,
            if n_scan > 0 { Some(handoff) } else { None },
        )?;

        let graph = builder
            .loop_body(read_step).for_n(n_read)
            .using(&["image"]).tag("read")
            .through(Linear::new(latent_dim, latent_dim)?).tag("latent")
            .fork(letter_head).tag("heads_0")
            .fork(case_head).tag("heads_1")
            .through(SharedDecoder(decoder.clone())).using(&["latent", "case"]).tag("recon")
            .build()?;

        Ok(LetterModel { graph, decoder, content_logits_buf, n_scan, scan_start, scan_phase_params })
    }

    /// Run the full pipeline: encode → classify → decode.
    ///
    /// img: [B, 1, 128, 128] input image.
    /// case_label: [B, 1] float — 0.0=upper, 1.0=lower.
    pub fn forward(&self, img: &Variable, case_label: &Variable) -> Result<LetterResult> {
        self.graph.forward_multi(&[img.clone(), case_label.clone()])?;

        let scan_locs = if self.n_scan > 0 {
            self.graph.traces("scan").unwrap_or_default()
        } else {
            Vec::new()
        };
        let read_locs = self.graph.traces("read").unwrap_or_default();

        Ok(LetterResult {
            letter_logits: self.graph.tagged("heads_0").expect("heads_0"),
            case_logits: self.graph.tagged("heads_1").expect("heads_1"),
            latent: self.graph.tagged("latent").expect("latent"),
            scan_locations: scan_locs,
            read_locations: read_locs,
            recon: self.graph.tagged("recon").expect("recon"),
        })
    }

    /// Inject an external starting position for the scan phase.
    ///
    /// Call before `forward()` when composing with SubScan. ScanStep will
    /// start from this position and use free (x, y) refinement instead of
    /// the learnable x-sweep used in standalone mode.
    ///
    /// The position is consumed on the next forward pass — call again for
    /// each subsequent forward.
    pub fn set_scan_start(&self, pos: Variable) {
        if let Some(ref start) = self.scan_start {
            *start.borrow_mut() = Some(pos);
        }
    }

    /// Scan-phase parameters (h0 + ScanStep).
    ///
    /// In composed mode, these stay trainable while read-phase params
    /// (AttentionStep, latent projection, heads, decoder) are frozen.
    pub fn scan_phase_params(&self) -> &[Parameter] {
        &self.scan_phase_params
    }

    /// Content logits from the most recent forward pass (one per scan step).
    pub fn content_logits(&self) -> Vec<Variable> {
        self.content_logits_buf.borrow().clone()
    }

    /// Parameters returns all learnable parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        self.graph.parameters()
    }

    /// Set training mode.
    pub fn train(&self) { self.graph.train(); }

    /// Set eval mode.
    pub fn eval(&self) { self.graph.eval(); }
}
