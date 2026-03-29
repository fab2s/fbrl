//! Direct letter eval: LetterModel on word images with GT origins.
//!
//! Bypasses SubScan entirely — sets origin to the known letter center
//! for each position. This gives a clean upper-bound baseline:
//! "how well does the letter model perform given perfect positioning?"
//!
//! Supports variable-length words with per-sample GT centers.

use std::fs;

use flodl::autograd::{no_grad, Variable};
use flodl::tensor::{cuda_available, Device, DType, Result, Tensor, TensorError};
use serde::{Serialize, Deserialize};

use super::data::{WordDataset, WordLoader};
use fbrl::letter::LetterModel;

/// Configuration for direct letter eval.
#[derive(Serialize, Deserialize)]
pub struct DirectEvalConfig {
    // --- Letter model architecture (auto-loaded from manifest) ---
    pub letter_n_classes: usize,
    pub letter_n_scan: usize,
    pub letter_n_read: usize,
    pub letter_patch_size: i64,
    pub letter_scan_patch_w: i64,
    pub letter_n_scales: usize,
    pub letter_latent_dim: i64,
    pub letter_img_h: i64,
    pub letter_img_w: i64,

    // --- Checkpoint ---
    pub letter_checkpoint: String,

    // --- Data ---
    pub word_data: String,
    pub batch_size: usize,

    // --- Output ---
    #[serde(default)]
    pub save_dir: String,
}

impl Default for DirectEvalConfig {
    fn default() -> Self {
        DirectEvalConfig {
            letter_n_classes: 26,
            letter_n_scan: 1,
            letter_n_read: 6,
            letter_patch_size: 12,
            letter_scan_patch_w: 18,
            letter_n_scales: 1,
            letter_latent_dim: 256,
            letter_img_h: 128,
            letter_img_w: 256,

            letter_checkpoint: String::new(),

            word_data: String::new(),
            batch_size: 32,

            save_dir: String::new(),
        }
    }
}

/// Per-position results.
#[derive(Debug)]
pub struct DirectPositionResult {
    pub position: usize,
    pub total: usize,
    pub correct: usize,
    pub matched_other_pos: Vec<usize>,
}

/// Overall eval results.
pub struct DirectEvalResults {
    pub positions: Vec<DirectPositionResult>,
    pub total: usize,
    pub correct: usize,
    pub accuracy: f64,
    /// Prediction frequency per letter index (0-25).
    pub pred_histogram: [usize; 26],
}

/// Load letter model config from manifest.json next to the checkpoint.
pub fn load_letter_config_for_direct(cfg: &mut DirectEvalConfig) {
    if cfg.letter_checkpoint.is_empty() { return; }
    let ckpt_path = std::path::Path::new(&cfg.letter_checkpoint);
    let manifest_path = ckpt_path.parent()
        .map(|p| p.join("manifest.json"))
        .unwrap_or_default();
    if !manifest_path.exists() { return; }

    let Ok(text) = fs::read_to_string(&manifest_path) else { return };
    let Ok(manifest): std::result::Result<serde_json::Value, _> = serde_json::from_str(&text) else { return };
    let Some(config) = manifest.get("config") else { return };

    if let Some(v) = config.get("n_classes").and_then(|v| v.as_u64()) { cfg.letter_n_classes = v as usize; }
    if let Some(v) = config.get("n_scan").and_then(|v| v.as_u64()) { cfg.letter_n_scan = v as usize; }
    if let Some(v) = config.get("n_read").and_then(|v| v.as_u64()) { cfg.letter_n_read = v as usize; }
    if let Some(v) = config.get("patch_size").and_then(|v| v.as_i64()) { cfg.letter_patch_size = v; }
    if let Some(v) = config.get("scan_patch_w").and_then(|v| v.as_i64()) { cfg.letter_scan_patch_w = v; }
    if let Some(v) = config.get("n_scales").and_then(|v| v.as_u64()) { cfg.letter_n_scales = v as usize; }
    if let Some(v) = config.get("latent_dim").and_then(|v| v.as_i64()) { cfg.letter_latent_dim = v; }
    if let Some(v) = config.get("img_h").and_then(|v| v.as_i64()) { cfg.letter_img_h = v; }
    if let Some(v) = config.get("img_w").and_then(|v| v.as_i64()) { cfg.letter_img_w = v; }

    eprintln!("Letter config from {}: {}x{}, latent={}, scan={}, read={}",
        manifest_path.display(), cfg.letter_img_h, cfg.letter_img_w,
        cfg.letter_latent_dim, cfg.letter_n_scan, cfg.letter_n_read);
}

/// Run direct eval: GT origin → LetterModel → classification.
pub fn eval_letter_direct(
    cfg: &DirectEvalConfig,
    word_ds: &WordDataset,
) -> Result<DirectEvalResults> {
    // --- Build LetterModel ---
    let letter = LetterModel::new(
        cfg.letter_n_classes, cfg.letter_n_scan, cfg.letter_n_read,
        cfg.letter_patch_size, cfg.letter_scan_patch_w,
        cfg.letter_n_scales, cfg.letter_latent_dim,
        cfg.letter_img_h, cfg.letter_img_w,
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
        letter.graph.set_device(Device::CUDA(0));
        Device::CUDA(0)
    } else {
        eprintln!("Using CPU");
        Device::CPU
    };

    letter.eval();
    for p in &letter.parameters() { p.freeze()?; }

    // --- Data loader ---
    let mut loader = WordLoader::new(word_ds, cfg.batch_size, false);
    loader.set_device(device);

    // --- Per-position accumulators ---
    let max_pos = 4;
    let mut pos_correct = vec![0usize; max_pos];
    let mut pos_total = vec![0usize; max_pos];
    let mut pos_matched_other: Vec<Vec<usize>> = (0..max_pos).map(|_| vec![0usize; max_pos]).collect();
    let mut pred_histogram = [0usize; 26];

    // --- Eval loop ---
    no_grad(|| {
        while let Some(batch) = loader.next_batch()? {
            let img_var = Variable::new(batch.image, false);
            let b = img_var.shape()[0] as usize;

            // Fetch all GT letter indices for cross-position matching.
            let gt_all: Vec<Vec<i64>> = (0..batch.word_len)
                .map(|pos| batch.letter_idx[pos].to_i64_vec())
                .collect::<Result<_>>()?;

            let case_var = Variable::new(
                Tensor::from_f32(&vec![1.0f32; b], &[b as i64, 1], device)?, false,
            );

            for pos in 0..batch.word_len {
                let gt_vec = batch.centers[pos].to_f32_vec()?;

                // GT origin: (center_x, 0.0) per sample.
                let origin_data: Vec<f32> = gt_vec.iter()
                    .flat_map(|&gt_x| [gt_x, 0.0f32])
                    .collect();
                let origin = Variable::new(
                    Tensor::from_f32(&origin_data, &[b as i64, 2], device)?, false,
                );

                // ── LetterModel → classification ──
                let result = letter.forward(&img_var, &case_var, &origin)?;

                let preds = result.letter_logits.data().argmax(1, false)?;
                let pred_vec = preds.to_i64_vec()?;
                let correct_mask = preds.eq_tensor(&batch.letter_idx[pos])?
                    .to_dtype(DType::Float32)?;
                let n_correct: f64 = correct_mask.sum_dim(0, false)?.item()?;

                pos_correct[pos] += n_correct as usize;
                pos_total[pos] += b;

                // Cross-position analysis.
                for (sample_i, &pred) in pred_vec.iter().enumerate() {
                    if pred >= 0 && pred < 26 {
                        pred_histogram[pred as usize] += 1;
                    }
                    for other_pos in 0..batch.word_len {
                        if pred == gt_all[other_pos][sample_i] {
                            pos_matched_other[pos][other_pos] += 1;
                        }
                    }
                }

                letter.graph.detach_state();
            }
        }
        Ok(())
    })?;

    // --- Compile results ---
    let mut total = 0usize;
    let mut correct = 0usize;
    let mut positions = Vec::new();

    for pos in 0..max_pos {
        if pos_total[pos] == 0 { continue; }
        let n = pos_total[pos];
        let c = pos_correct[pos];
        positions.push(DirectPositionResult {
            position: pos,
            total: n,
            correct: c,
            matched_other_pos: pos_matched_other[pos].clone(),
        });
        total += n;
        correct += c;
    }

    let accuracy = if total > 0 { correct as f64 / total as f64 } else { 0.0 };

    let results = DirectEvalResults {
        positions,
        total,
        correct,
        accuracy,
        pred_histogram,
    };

    // --- Save results ---
    if !cfg.save_dir.is_empty() {
        fs::create_dir_all(&cfg.save_dir)
            .map_err(|e| TensorError::new(&format!("create save dir: {e}")))?;

        let mut pred_sorted: Vec<(usize, usize)> = pred_histogram.iter()
            .enumerate().map(|(i, &c)| (i, c)).collect();
        pred_sorted.sort_by(|a, b| b.1.cmp(&a.1));
        let top_preds: Vec<String> = pred_sorted.iter().take(5).map(|(i, c)| {
            format!("{}={}", (b'A' + *i as u8) as char, c)
        }).collect();

        let report = serde_json::json!({
            "model": "letter direct (GT origin)",
            "total": total,
            "correct": correct,
            "accuracy": format!("{:.1}%", accuracy * 100.0),
            "top_predictions": top_preds,
            "per_position": results.positions.iter().map(|p| {
                let n = p.total.max(1) as f64;
                serde_json::json!({
                    "pos": p.position,
                    "total": p.total,
                    "correct": p.correct,
                    "accuracy": format!("{:.1}%", p.correct as f64 / n * 100.0),
                    "pred_matches_pos": p.matched_other_pos.iter().enumerate()
                        .filter(|&(_, c)| *c > 0)
                        .map(|(op, c)| format!("pos{}={:.1}%", op, *c as f64 / n * 100.0))
                        .collect::<Vec<_>>(),
                })
            }).collect::<Vec<_>>(),
            "letter_checkpoint": cfg.letter_checkpoint,
        });

        let path = format!("{}/eval.json", cfg.save_dir);
        if let Err(e) = fs::write(&path, serde_json::to_string_pretty(&report).unwrap_or_default()) {
            eprintln!("warning: write eval: {e}");
        }
    }

    Ok(results)
}
