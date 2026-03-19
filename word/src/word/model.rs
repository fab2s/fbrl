//! WordModel — 4-letter word recognition via foveal attention.
//!
//! Two-phase attention graph:
//! ```text
//! image → H0Init → Loop(ScanStep, 8).Using("image").Tag("scan")
//!       → Loop(ReadStep, 12).Using("image").Tag("read")
//! ```
//!
//! Post-graph:
//! ```text
//! read_states [B, 12, D] → CrossAttentionReadout → [B, 4, D]
//!   → 4 × classifier → per-position letter logits
//!   → flatten → WordDecoder → [B, 1, 128, 256] reconstruction
//! ```

use std::cell::RefCell;
use std::rc::Rc;

use flodl::autograd::Variable;
use flodl::graph::{FlowBuilder, Graph};
use flodl::nn::{Linear, Module, Parameter, Buffer};
use flodl::tensor::{Device, Result};

use super::decoder::WordDecoder;
use super::glimpse::GlimpseSensor;
use super::modules::{H0Init, Identity, ReadStep, ScanStep};
use super::readout::CrossAttentionReadout;

/// Holds everything from a word model forward pass.
pub struct WordResult {
    pub recon: Variable,                      // [B, 1, 128, 256]
    pub position_logits: Vec<Variable>,       // 4 × [B, 26]
    pub readout_states: Variable,             // [B, 4, latent_dim]
    pub scan_locations: Vec<Variable>,        // 8 scan fixation locations
    pub read_locations: Vec<Variable>,        // 12 read fixation locations
}

/// 4-letter word model.
pub struct WordModel {
    /// Attention graph (scan + read loops).
    pub graph: Graph,
    /// Cross-attention readout (outside graph — consumes collected read states).
    pub readout: CrossAttentionReadout,
    /// Per-position classifiers: 4 × Linear(latent_dim, 26).
    pub classifiers: Vec<Linear>,
    /// Decoder for reconstruction.
    pub decoder: WordDecoder,
    /// Read states buffer shared with ReadStep.
    read_states_buf: Rc<RefCell<Vec<Variable>>>,
    /// Content logits buffer shared with ScanStep.
    content_logits_buf: Rc<RefCell<Vec<Variable>>>,
    /// Group boundaries for cross-attention (e.g. [0, 3, 6, 9]).
    group_boundaries: Vec<usize>,
    n_positions: usize,
}

impl WordModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_classes: usize,
        n_positions: usize,
        n_scan: usize,
        n_read: usize,
        patch_size: i64,
        scan_patch_w: i64,
        n_scales: usize,
        latent_dim: i64,
    ) -> Result<Self> {
        let content_logits_buf = Rc::new(RefCell::new(Vec::new()));
        let read_states_buf = Rc::new(RefCell::new(Vec::new()));

        // Build attention graph: H0Init → ScanLoop → ReadLoop.
        let scan_sensor = GlimpseSensor::new(patch_size, scan_patch_w, n_scales, latent_dim)?;
        let content_head = Linear::new(latent_dim, 1)?;
        let scan_step = ScanStep::new(
            scan_sensor, latent_dim, n_scan,
            Some(content_head), Some(content_logits_buf.clone()),
        )?;

        let read_sensor = GlimpseSensor::new(patch_size, patch_size, n_scales, latent_dim)?;
        let read_step = ReadStep::new(
            read_sensor, latent_dim, read_states_buf.clone(),
        )?;

        let graph = FlowBuilder::from(Identity)
            .label("WordModel")
            .tag("image")
            .through(H0Init::new(latent_dim)?)
            .loop_body(scan_step).for_n(n_scan)
            .using(&["image"]).tag("scan")
            .loop_body(read_step).for_n(n_read)
            .using(&["image"]).tag("read")
            .build()?;

        // Post-graph modules.
        let readout = CrossAttentionReadout::new(latent_dim, n_positions)?;

        let mut classifiers = Vec::with_capacity(n_positions);
        for _ in 0..n_positions {
            classifiers.push(Linear::new(latent_dim, n_classes as i64)?);
        }

        let decoder = WordDecoder::new(
            latent_dim * n_positions as i64,
            128, 256,
        )?;

        // Group boundaries: evenly divide reads across positions.
        let reads_per_pos = n_read / n_positions;
        let group_boundaries: Vec<usize> = (0..n_positions)
            .map(|p| p * reads_per_pos)
            .collect();

        Ok(WordModel {
            graph,
            readout,
            classifiers,
            decoder,
            read_states_buf,
            content_logits_buf,
            group_boundaries,
            n_positions,
        })
    }

    /// Run the full pipeline.
    ///
    /// img: [B, 1, 128, 256] word image.
    pub fn forward(&self, img: &Variable) -> Result<WordResult> {
        // Run attention graph (populates read_states_buf via ReadStep).
        self.graph.forward(img)?;

        // Collect scan/read traces.
        let scan_locs = self.graph.traces("scan").unwrap_or_default();
        let read_locs = self.graph.traces("read").unwrap_or_default();

        // Stack read hidden states: Vec<[B, D]> → [B, T, D].
        let states = self.read_states_buf.borrow();
        let read_states = Variable::stack(&states, 1)?; // [B, T, D]

        // Cross-attention readout: [B, n_positions, D].
        let readout_states = self.readout.forward(&read_states, &self.group_boundaries)?;

        // Per-position classification.
        let mut position_logits = Vec::with_capacity(self.n_positions);
        for (p, clf) in self.classifiers.iter().enumerate() {
            let pos_latent = readout_states.select(1, p as i64)?; // [B, D]
            position_logits.push(clf.forward(&pos_latent)?);
        }

        // Reconstruction from flattened readout.
        let recon_input = readout_states.flatten(1, -1)?; // [B, n_pos * D]
        let recon = self.decoder.decode(&recon_input)?;

        Ok(WordResult {
            recon,
            position_logits,
            readout_states,
            scan_locations: scan_locs,
            read_locations: read_locs,
        })
    }

    /// Content logits from the most recent forward pass (one per scan step).
    pub fn content_logits(&self) -> Vec<Variable> {
        self.content_logits_buf.borrow().clone()
    }

    /// All learnable parameters (graph + readout + classifiers + decoder).
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.graph.parameters();
        params.extend(self.readout.parameters());
        for clf in &self.classifiers {
            params.extend(clf.parameters());
        }
        params.extend(self.decoder.parameters());
        params
    }

    /// All buffers (decoder BatchNorm).
    pub fn buffers(&self) -> Vec<Buffer> {
        self.decoder.buffers()
    }

    pub fn train(&self) {
        self.graph.train();
        self.decoder.set_training(true);
    }

    pub fn eval(&self) {
        self.graph.eval();
        self.decoder.set_training(false);
    }

    pub fn set_device(&self, device: Device) {
        self.graph.set_device(device);
        // Move post-graph modules (not part of graph) to device.
        let extra_params = self.readout.parameters().into_iter()
            .chain(self.classifiers.iter().flat_map(|c| c.parameters()))
            .chain(self.decoder.parameters());
        for p in extra_params {
            if p.variable.data().device() != device {
                if let Ok(t) = p.variable.data().detach()
                    .and_then(|d| d.to_device(device))
                {
                    p.variable.set_data(t);
                }
            }
        }
        self.decoder.move_to_device(device);
    }
}
