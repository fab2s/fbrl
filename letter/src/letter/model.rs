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
use flodl::nn::{Linear, Module, NamedInputModule, Parameter};
use flodl::tensor::{Device, Result};

use super::decoder::VisualDecoder;
use super::glimpse::GlimpseSensor;
use super::modules::{AttentionStep, Controller, H0Init, Identity, ScanStep};

/// Wraps an Rc<VisualDecoder> so it can live in the graph while
/// LetterModel retains a reference for recode.
struct SharedDecoder(Rc<VisualDecoder>);

impl Module for SharedDecoder {
    fn name(&self) -> &str { self.0.name() }
    fn forward(&self, input: &Variable) -> Result<Variable> { self.0.forward(input) }
    fn as_named_input(&self) -> Option<&dyn NamedInputModule> { Some(self) }
    fn parameters(&self) -> Vec<Parameter> { self.0.parameters() }
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
    /// Content logits buffer shared with ScanStep (populated during forward).
    content_logits_buf: Rc<RefCell<Vec<Variable>>>,
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
        let controller = Rc::new(Controller::new(latent_dim)?);

        let read_sensor = GlimpseSensor::new(patch_size, patch_size, n_scales, latent_dim)?;
        let read_step = AttentionStep::with_shared(read_sensor, controller.clone());

        let letter_head = Linear::new(latent_dim, n_classes as i64)?;
        let case_head = Linear::new(latent_dim, 2)?;
        let decoder = Rc::new(VisualDecoder::new(latent_dim + 1, 128, 128)?);

        // Content head + shared buffer for scan ink detection (only when scanning).
        let content_logits_buf = Rc::new(RefCell::new(Vec::new()));
        let content_head = if n_scan > 0 {
            Some(Linear::new(latent_dim, 1)?)
        } else {
            None
        };

        let mut builder = FlowBuilder::from(Identity)
            .label("LetterModel")
            .tag("image")
            .input(&["case"])
            .through(H0Init::new(latent_dim)?);

        if n_scan > 0 {
            let scan_sensor = GlimpseSensor::new(patch_size, scan_patch_w, n_scales, latent_dim)?;
            let scan_step = ScanStep::new(
                scan_sensor, controller.clone(), n_scan,
                content_head, Some(content_logits_buf.clone()),
            )?;
            builder = builder
                .loop_body(scan_step).for_n(n_scan)
                .using(&["image"]).tag("scan");
        }

        let graph = builder
            .loop_body(read_step).for_n(n_read)
            .using(&["image"]).tag("read")
            .through(Linear::new(latent_dim, latent_dim)?).tag("latent")
            .fork(letter_head).tag("heads_0")
            .fork(case_head).tag("heads_1")
            .through(SharedDecoder(decoder.clone())).using(&["latent", "case"]).tag("recon")
            .build()?;

        Ok(LetterModel { graph, decoder, content_logits_buf })
    }

    /// Run the full pipeline: encode → classify → decode.
    ///
    /// img: [B, 1, 128, 128] input image.
    /// case_label: [B, 1] float — 0.0=upper, 1.0=lower.
    pub fn forward(&self, img: &Variable, case_label: &Variable) -> Result<LetterResult> {
        self.graph.forward_multi(&[img.clone(), case_label.clone()])?;

        Ok(LetterResult {
            letter_logits: self.graph.tagged("heads_0").expect("heads_0"),
            case_logits: self.graph.tagged("heads_1").expect("heads_1"),
            latent: self.graph.tagged("latent").expect("latent"),
            scan_locations: self.graph.traces("scan").unwrap_or_default(),
            read_locations: self.graph.traces("read").unwrap_or_default(),
            recon: self.graph.tagged("recon").expect("recon"),
        })
    }

    /// Content logits from the most recent forward pass (one per scan step).
    pub fn content_logits(&self) -> Vec<Variable> {
        self.content_logits_buf.borrow().clone()
    }

    /// Parameters returns all learnable parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        self.graph.parameters()
    }

    /// Propagate training mode.
    pub fn set_training(&self, training: bool) {
        self.graph.set_training(training);
    }
}
