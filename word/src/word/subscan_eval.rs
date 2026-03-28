//! Composition eval: SubScan + LetterModel on word images.
//!
//! Two independently trained models, composed for the first time:
//! - SubScan: triangle glimpses → center_x (trained on word images alone)
//! - LetterModel: center → letter classification (trained on single letters)
//!
//! For each letter position in each word:
//!   noisy_start → SubScan → center_x → LetterModel → predicted letter
//!
//! Reports per-position accuracy, overall accuracy, and position error.

use std::fs;

use flodl::autograd::{no_grad, Variable};
use flodl::nn::Module;
use flodl::tensor::{cuda_available, Device, DType, Result, Tensor, TensorError};
use serde::{Serialize, Deserialize};

use super::data::{WordDataset, WordLoader};
use super::subscan::{SubScanModel, SubScanConfig};

use fbrl::letter::LetterModel;

/// Number of letter positions in a word image.
const N_POSITIONS: usize = 4;

/// Normalized x-centers for the 4 letter positions in a 128x256 word image.
const LETTER_CENTERS: [f64; N_POSITIONS] = [-0.75, -0.25, 0.25, 0.75];

/// Configuration for composition eval.
#[derive(Serialize, Deserialize)]
pub struct SubScanEvalConfig {
    // --- SubScan architecture ---
    pub hidden_dim: i64,
    pub subscan_patch_h: i64,
    pub subscan_patch_w: i64,
    pub subscan_n_scales: usize,
    pub subscan_n_glimpses: usize,
    pub subscan_blur_sigma: f64,
    pub min_base_hw: f64,
    pub max_base_hw: f64,
    pub triangle_height: f64,

    // --- Letter model architecture ---
    pub letter_n_classes: usize,
    pub letter_n_scan: usize,
    pub letter_n_read: usize,
    pub letter_patch_size: i64,
    pub letter_scan_patch_w: i64,
    pub letter_n_scales: usize,
    pub letter_latent_dim: i64,

    // --- Checkpoints ---
    pub subscan_checkpoint: String,
    pub letter_checkpoint: String,

    // --- Region bounding ---
    pub region_half_w: f64,

    // --- Input noise (simulates word model imprecision) ---
    pub noise_x: f64,
    pub noise_y: f64,

    // --- Data ---
    pub word_data: String,
    pub batch_size: usize,

    // --- Output ---
    #[serde(default)]
    pub save_dir: String,
}

impl Default for SubScanEvalConfig {
    fn default() -> Self {
        SubScanEvalConfig {
            hidden_dim: 256,
            subscan_patch_h: 8,
            subscan_patch_w: 28,
            subscan_n_scales: 1,
            subscan_n_glimpses: 3,
            subscan_blur_sigma: 4.0,
            min_base_hw: 0.03,
            max_base_hw: 0.20,
            triangle_height: 0.3,

            letter_n_classes: 26,
            letter_n_scan: 1,
            letter_n_read: 6,
            letter_patch_size: 12,
            letter_scan_patch_w: 18,
            letter_n_scales: 1,
            letter_latent_dim: 256,

            subscan_checkpoint: String::new(),
            letter_checkpoint: String::new(),

            region_half_w: 0.5,

            noise_x: 0.10,
            noise_y: 0.05,

            word_data: String::new(),
            batch_size: 32,

            save_dir: String::new(),
        }
    }
}

/// Per-position results.
#[derive(Debug)]
pub struct PositionResult {
    pub position: usize,
    pub gt_center: f64,
    pub total: usize,
    pub correct: usize,
    pub mean_err_x: f64,
}

/// Overall eval results.
pub struct EvalResults {
    pub positions: [PositionResult; N_POSITIONS],
    pub total: usize,
    pub correct: usize,
    pub accuracy: f64,
    pub mean_err_x: f64,
}

/// Run composition eval: SubScan → center_x → LetterModel.
pub fn eval_subscan_composition(
    cfg: &SubScanEvalConfig,
    word_ds: &WordDataset,
) -> Result<EvalResults> {
    // --- Build SubScan ---
    let subscan = SubScanModel::new(&SubScanConfig {
        hidden_dim: cfg.hidden_dim,
        patch_h: cfg.subscan_patch_h,
        patch_w: cfg.subscan_patch_w,
        n_scales: cfg.subscan_n_scales,
        n_glimpses: cfg.subscan_n_glimpses,
        blur_sigma: cfg.subscan_blur_sigma,
        min_base_hw: cfg.min_base_hw,
        max_base_hw: cfg.max_base_hw,
        triangle_height: cfg.triangle_height,
    })?;

    if !cfg.subscan_checkpoint.is_empty() {
        let report = subscan.graph.load_checkpoint(&cfg.subscan_checkpoint)?;
        eprintln!("SubScan checkpoint: {} params, {} skipped, {} missing",
            report.loaded.len(), report.skipped.len(), report.missing.len());
    }

    // --- Build LetterModel ---
    let letter = LetterModel::new(
        cfg.letter_n_classes, cfg.letter_n_scan, cfg.letter_n_read,
        cfg.letter_patch_size, cfg.letter_scan_patch_w,
        cfg.letter_n_scales, cfg.letter_latent_dim,
    )?;

    if !cfg.letter_checkpoint.is_empty() {
        let report = letter.graph.load_checkpoint(&cfg.letter_checkpoint)?;
        eprintln!("Letter checkpoint: {} params, {} skipped, {} missing",
            report.loaded.len(), report.skipped.len(), report.missing.len());
    }

    // --- Move to device ---
    let device = if cuda_available() {
        eprintln!("Using CUDA");
        flodl::tensor::set_cudnn_benchmark(true);
        subscan.graph.set_device(Device::CUDA(0));
        letter.graph.set_device(Device::CUDA(0));
        Device::CUDA(0)
    } else {
        eprintln!("Using CPU");
        Device::CPU
    };

    // Both models in eval mode, all params frozen.
    subscan.graph.eval();
    letter.eval();
    for p in &subscan.parameters() { p.freeze()?; }
    for p in &letter.parameters() { p.freeze()?; }

    // --- Data loader ---
    let mut loader = WordLoader::new(word_ds, cfg.batch_size, false);
    loader.set_device(device);

    // --- RNG for noise ---
    let mut rng_state: u64 = 0xE0A1_CAFE;
    #[inline]
    fn rng_next(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *state
    }
    #[inline]
    fn rng_uniform(state: &mut u64) -> f64 {
        let v = rng_next(state);
        ((v >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    // --- Per-position accumulators ---
    let mut pos_correct = [0usize; N_POSITIONS];
    let mut pos_total = [0usize; N_POSITIONS];
    let mut pos_err_sum = [0.0f64; N_POSITIONS];

    // --- Eval loop ---
    no_grad(|| {
        while let Some(batch) = loader.next_batch()? {
            let img_var = Variable::new(batch.image, false);
            let b = img_var.shape()[0] as usize;

            // Region half-width.
            let half_w_data = vec![cfg.region_half_w as f32; b];
            let region_half_w = Variable::new(
                Tensor::from_f32(&half_w_data, &[b as i64, 1], device)?, false,
            );

            // Case label (all lowercase).
            let case_data = vec![1.0f32; b];
            let case_var = Variable::new(
                Tensor::from_f32(&case_data, &[b as i64, 1], device)?, false,
            );

            for pos in 0..N_POSITIONS {
                let gt_x = LETTER_CENTERS[pos];

                // Noisy starting position (simulates word model).
                let start_data: Vec<f32> = (0..b).flat_map(|_| {
                    let x = (gt_x + rng_uniform(&mut rng_state) * cfg.noise_x) as f32;
                    let y = (rng_uniform(&mut rng_state) * cfg.noise_y) as f32;
                    [x, y]
                }).collect();
                let start_pos = Variable::new(
                    Tensor::from_f32(&start_data, &[b as i64, 2], device)?, false,
                );

                // ── SubScan → center_x ──
                let center_x = subscan.forward(&img_var, &start_pos, &region_half_w)?;

                // Build origin [B, 2] for letter model: (center_x, 0).
                let y_data = vec![0.0f32; b];
                let y_var = Variable::new(
                    Tensor::from_f32(&y_data, &[b as i64, 1], device)?, false,
                );
                let origin = center_x.cat(&y_var, 1)?; // [B, 2]

                // ── LetterModel → classification ──
                let result = letter.forward(&img_var, &case_var, &origin)?;

                // Accuracy.
                let preds = result.letter_logits.data().argmax(1, false)?;
                let correct_mask = preds.eq_tensor(&batch.letter_idx[pos])?
                    .to_dtype(DType::Float32)?;
                let n_correct: f64 = correct_mask.sum_dim(0, false)?.item()?;

                // Position error.
                let gt_data = vec![gt_x as f32; b];
                let gt_tensor = Tensor::from_f32(&gt_data, &[b as i64, 1], device)?;
                let err: f64 = center_x.data().sub(&gt_tensor)?
                    .abs()?.mean()?.item()?;

                pos_correct[pos] += n_correct as usize;
                pos_total[pos] += b;
                pos_err_sum[pos] += err * b as f64;

                subscan.graph.detach_state();
                letter.graph.detach_state();
            }
        }
        Ok(())
    })?;

    // --- Compile results ---
    let mut total = 0usize;
    let mut correct = 0usize;
    let mut err_sum = 0.0f64;
    let mut positions = Vec::with_capacity(N_POSITIONS);

    for pos in 0..N_POSITIONS {
        let n = pos_total[pos];
        let c = pos_correct[pos];
        let err = if n > 0 { pos_err_sum[pos] / n as f64 } else { 0.0 };
        positions.push(PositionResult {
            position: pos,
            gt_center: LETTER_CENTERS[pos],
            total: n,
            correct: c,
            mean_err_x: err,
        });
        total += n;
        correct += c;
        err_sum += pos_err_sum[pos];
    }

    let accuracy = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
    let mean_err_x = if total > 0 { err_sum / total as f64 } else { 0.0 };

    let results = EvalResults {
        positions: positions.try_into().unwrap(),
        total,
        correct,
        accuracy,
        mean_err_x,
    };

    // --- Save results ---
    if !cfg.save_dir.is_empty() {
        fs::create_dir_all(&cfg.save_dir)
            .map_err(|e| TensorError::new(&format!("create save dir: {e}")))?;

        let report = serde_json::json!({
            "model": "subscan + letter composition",
            "noise_x": cfg.noise_x,
            "noise_y": cfg.noise_y,
            "total": total,
            "correct": correct,
            "accuracy": format!("{:.1}%", accuracy * 100.0),
            "mean_err_x": format!("{:.4}", mean_err_x),
            "per_position": results.positions.iter().map(|p| {
                serde_json::json!({
                    "pos": p.position,
                    "gt_center": p.gt_center,
                    "total": p.total,
                    "correct": p.correct,
                    "accuracy": format!("{:.1}%", if p.total > 0 { p.correct as f64 / p.total as f64 * 100.0 } else { 0.0 }),
                    "mean_err_x": format!("{:.4}", p.mean_err_x),
                })
            }).collect::<Vec<_>>(),
            "subscan_checkpoint": cfg.subscan_checkpoint,
            "letter_checkpoint": cfg.letter_checkpoint,
        });

        let path = format!("{}/eval.json", cfg.save_dir);
        if let Err(e) = fs::write(&path, serde_json::to_string_pretty(&report).unwrap_or_default()) {
            eprintln!("warning: write eval: {e}");
        }
    }

    Ok(results)
}
