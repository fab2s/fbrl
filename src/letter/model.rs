//! LetterModel — single-letter recognition via foveal attention.
//!
//! Built as a graph with two inputs (image, case label):
//!
//! ```text
//! image → Tag("image") → H0Init → Loop(AttentionStep).Using("image").Tag("attention")
//!   → LatentHead → Tag("latent") → Split(letterHead, caseHead)
//!   → TagGroup("heads") → Merge → Decoder.Using("latent", "case") → Tag("recon")
//! ```

use flodl::autograd::Variable;
use flodl::graph::{FlowBuilder, Graph, MergeOp};
use flodl::nn::{Linear, Module, Parameter};
use flodl::tensor::Result;
use flodl::modules;

use super::decoder::VisualDecoder;
use super::glimpse::GlimpseSensor;
use super::modules::{AttentionStep, H0Init, Identity, LatentHead};

/// Holds everything from a forward pass.
pub struct LetterResult {
    pub recon: Variable,           // [B, 1, 128, 128]
    pub letter_logits: Variable,   // [B, n_classes]
    pub case_logits: Variable,     // [B, 2]
    pub locations: Vec<Variable>,  // fixation trajectory
    pub latent: Variable,          // [B, latent_dim]
}

/// Single-letter model built as a computation graph.
pub struct LetterModel {
    pub graph: Graph,
}

impl LetterModel {
    /// Create the single-letter model.
    pub fn new(
        n_classes: usize,
        n_glimpses: usize,
        patch_size: i64,
        n_scales: usize,
        latent_dim: i64,
    ) -> Result<Self> {
        let sensor = GlimpseSensor::new(patch_size, patch_size, n_scales, latent_dim)?;
        let step = AttentionStep::new(sensor, latent_dim)?;
        let letter_head = Linear::new(latent_dim, n_classes as i64)?;
        let case_head = Linear::new(latent_dim, 2)?;
        let decoder = VisualDecoder::new(latent_dim + 1, 128, 128)?;

        let graph = FlowBuilder::from(Identity)
            .tag("image")
            .input(&["case"])
            .through(H0Init::new(latent_dim)?)
            .loop_body(step).for_n(n_glimpses)
            .using(&["image"]).tag("attention")
            .through(LatentHead::new(latent_dim, latent_dim)?).tag("latent")
            .split(modules![letter_head, case_head])
            .tag_group("heads")
            .merge(MergeOp::Add) // SelectFirst via merge; heads accessed via tagged
            .through(decoder).using(&["latent", "case"]).tag("recon")
            .build()?;

        Ok(LetterModel { graph })
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
            locations: self.graph.traces("attention").unwrap_or_default(),
            recon: self.graph.tagged("recon").expect("recon"),
        })
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
