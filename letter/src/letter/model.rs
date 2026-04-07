//! LetterModel — single-letter recognition via foveal attention.
//!
//! Built as a graph with three inputs (image, case label, origin):
//!
//! ```text
//! image → Tag("image") → H0Init
//!       → Input(["case", "origin"])
//!       → Loop(CombinedStep).Using("image", "origin").Tag("attn")
//!       → Linear → Tag("latent") → Fork(letterHead).Tag("heads_0")
//!       → Fork(caseHead).Tag("heads_1") → Decoder.Using("latent", "case") → Tag("recon")
//! ```
//!
//! All attention positions are relative to `origin`. The loc_head outputs
//! offsets, `location_fc` sees these offsets (always near zero), and
//! `grid_sample` receives `origin + offset` for correct image sampling.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use flodl::autograd::Variable;
use flodl::graph::{FlowBuilder, Graph};
use flodl::nn::{Buffer, Linear, Module, NamedInputModule, Parameter};
use flodl::tensor::{Device, Result};

use super::decoder::VisualDecoder;
use super::glimpse::GlimpseSensor;
use super::modules::{CombinedStep, Controller, H0Init, Identity};

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
pub struct LetterModel {
    pub graph: Graph,
    /// Decoder reference for recode (shared with graph node via Rc).
    pub decoder: Rc<VisualDecoder>,
    /// Content logits buffer shared with CombinedStep (populated during forward).
    content_logits_buf: Rc<RefCell<Vec<Variable>>>,
    /// Number of scan steps — used to split traces into scan/read.
    n_scan: usize,
}

impl LetterModel {
    /// Create the single-letter model.
    ///
    /// n_scan=0: read-only (v1 baseline).
    /// n_scan=1: 1 wide scan + n_read reads (v2, matches Python v7).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_classes: usize,
        n_scan: usize,
        n_read: usize,
        patch_size: i64,
        scan_patch_w: i64,
        n_scales: usize,
        latent_dim: i64,
        img_h: i64,
        img_w: i64,
    ) -> Result<Self> {
        let controller = Rc::new(Controller::new(latent_dim)?);

        let letter_head = Linear::new(latent_dim, n_classes as i64)?;
        let case_head = Linear::new(latent_dim, 2)?;
        let decoder = Rc::new(VisualDecoder::new(latent_dim + 1, img_h, img_w)?);

        // Content head + shared buffer for scan ink detection (only when scanning).
        let content_logits_buf = Rc::new(RefCell::new(Vec::new()));
        let content_head = if n_scan > 0 {
            Some(Linear::new(latent_dim, 1)?)
        } else {
            None
        };

        let total_steps = n_scan + n_read;
        let scan_sensor = GlimpseSensor::new(patch_size, scan_patch_w, n_scales, latent_dim)?;
        let read_sensor = GlimpseSensor::new(patch_size, patch_size, n_scales, latent_dim)?;

        let combined = CombinedStep::new(
            scan_sensor, read_sensor, controller,
            n_scan, content_head, Some(content_logits_buf.clone()),
        )?;

        let graph = FlowBuilder::from(Identity)
            .label("LetterModel")
            .tag("image")
            .input(&["case", "origin"])
            .through(H0Init::new(latent_dim)?)
            .loop_body(combined).for_n(total_steps)
            .using(&["image", "origin"]).tag("attn")
            .through(Linear::new(latent_dim, latent_dim)?).tag("latent")
            .fork(letter_head).tag("heads_0")
            .fork(case_head).tag("heads_1")
            .through(SharedDecoder(decoder.clone())).using(&["latent", "case"]).tag("recon")
            .build()?;

        Ok(LetterModel { graph, decoder, content_logits_buf, n_scan })
    }

    /// Run the full pipeline: encode → classify → decode.
    ///
    /// - `img`: `[B, 1, H, W]` input image.
    /// - `case_label`: `[B, 1]` float — 0.0=upper, 1.0=lower.
    /// - `origin`: `[B, 2]` coordinate origin for the attention mechanism.
    ///   Standalone training: `(0,0) + noise`. Composition: SubScan output.
    pub fn forward(&self, img: &Variable, case_label: &Variable, origin: &Variable) -> Result<LetterResult> {
        self.graph.forward_multi(&[img.clone(), case_label.clone(), origin.clone()])?;

        // Split flat traces into scan and read portions.
        let all_traces = self.graph.traces("attn").unwrap_or_default();
        let (scan_locs, read_locs) = if self.n_scan > 0 && all_traces.len() > self.n_scan {
            let (s, r) = all_traces.split_at(self.n_scan);
            (s.to_vec(), r.to_vec())
        } else {
            (Vec::new(), all_traces)
        };

        Ok(LetterResult {
            letter_logits: self.graph.tagged("heads_0").expect("heads_0"),
            case_logits: self.graph.tagged("heads_1").expect("heads_1"),
            latent: self.graph.tagged("latent").expect("latent"),
            scan_locations: scan_locs,
            read_locations: read_locs,
            recon: self.graph.tagged("recon").expect("recon"),
        })
    }

    /// Content logits from the most recent forward pass (one per scan step).
    pub fn content_logits(&self) -> Vec<Variable> {
        self.content_logits_buf.borrow().clone()
    }

    /// Extract the graph for use as a DDP replica.
    /// Consumes the LetterModel; the graph's internal nodes retain all state via Rc.
    pub fn into_graph(self) -> Graph {
        self.graph
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

impl Module for LetterModel {
    fn name(&self) -> &str { "LetterModel" }
    fn forward(&self, input: &Variable) -> Result<Variable> { self.graph.forward(input) }
    fn parameters(&self) -> Vec<Parameter> { self.graph.parameters() }
    fn buffers(&self) -> Vec<Buffer> { self.graph.buffers() }
    fn set_training(&self, training: bool) { self.graph.set_training(training) }
    fn move_to_device(&self, device: Device) { self.graph.move_to_device(device) }
    fn as_graph(&self) -> Option<&flodl::Graph> { Some(&self.graph) }
}
