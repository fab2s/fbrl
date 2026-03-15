//! Training loop and configuration for the letter model.

use std::fs;
use std::io::Write;
use std::time::Instant;

use flodl::autograd::Variable;
use flodl::nn::{
    cross_entropy_loss, mse_loss, clip_grad_norm,
    Adam, CosineScheduler, Optimizer,
};
use flodl::monitor::Monitor;
use flodl::tensor::{cuda_available, Device, DType, Result, Tensor, TensorError};
use serde::{Serialize, Deserialize};

use super::data::{LetterDataset, LetterLoader};
use super::loss::{attention_guide_loss, fixation_diversity_loss, fixation_hit_rate};
use super::model::LetterModel;

/// Hyperparameters for letter model training.
#[derive(Serialize, Deserialize)]
pub struct LetterConfig {
    // Model architecture.
    pub n_classes: usize,
    pub n_scan: usize,
    pub n_read: usize,
    pub patch_size: i64,
    pub scan_patch_w: i64,
    pub n_scales: usize,
    pub latent_dim: i64,

    // Training.
    pub batch_size: usize,
    pub epochs: usize,
    pub lr: f64,
    pub min_lr: f64,
    pub max_grad_norm: f64,

    // Loss weights.
    pub scan_guide_weight: f64,
    pub read_guide_weight: f64,
    pub diversity_weight: f64,
    pub recon_weight: f64,
    pub recode_weight: f64,
    pub blur_sigma_ratio: f64,
    pub diversity_sigma: f64,
    pub diversity_vy: f64,

    // Checkpointing.
    #[serde(default)]
    pub save_dir: String,
    #[serde(default = "default_checkpoint_interval")]
    pub checkpoint_interval: usize,

    // Live monitoring.
    #[serde(skip)]
    pub monitor_port: Option<u16>,
}

impl Default for LetterConfig {
    fn default() -> Self {
        LetterConfig {
            n_classes: 26,
            n_scan: 1,
            n_read: 6,
            patch_size: 12,
            scan_patch_w: 18,
            n_scales: 1,
            latent_dim: 256,

            batch_size: 32,
            epochs: 100,
            lr: 0.001,
            min_lr: 0.0,
            max_grad_norm: 5.0,

            scan_guide_weight: 8.0,
            read_guide_weight: 0.0,
            diversity_weight: 1.0,
            recon_weight: 1.0,
            recode_weight: 0.0,
            blur_sigma_ratio: 0.16,
            diversity_sigma: 0.1,
            diversity_vy: 1.0,

            save_dir: "training".into(),
            checkpoint_interval: 50,

            monitor_port: None,
        }
    }
}

fn default_checkpoint_interval() -> usize { 50 }

/// Per-epoch averaged metrics.
pub struct EpochStats {
    pub epoch: usize,
    pub letter_loss: f64,
    pub case_loss: f64,
    pub letter_acc: f64,
    pub case_acc: f64,
    pub recon_loss: f64,
    pub guide_loss: f64,
    pub div_loss: f64,
    pub total_loss: f64,
    pub hit_rate: f64,
    pub lr: f64,
    pub duration: std::time::Duration,
    pub eta: std::time::Duration,
}

/// Run the full training loop for the letter model.
pub fn train_letter(
    cfg: &LetterConfig,
    ds: &LetterDataset,
    on_epoch: Option<&dyn Fn(&EpochStats)>,
) -> Result<()> {
    let model = LetterModel::new(
        cfg.n_classes, cfg.n_scan, cfg.n_read, cfg.patch_size, cfg.scan_patch_w,
        cfg.n_scales, cfg.latent_dim,
    )?;

    // Move model to CUDA if available.
    let device = if cuda_available() {
        eprintln!("Using CUDA");
        model.graph.set_device(Device::CUDA);
        Device::CUDA
    } else {
        eprintln!("Using CPU");
        Device::CPU
    };

    let params = model.parameters();
    let mut optimizer = Adam::new(&params, cfg.lr);
    let scheduler = CosineScheduler::new(cfg.lr, cfg.min_lr, cfg.epochs);
    model.set_training(true);

    let mut loader = LetterLoader::new(ds, cfg.batch_size, true);
    loader.set_device(device);

    // Ensure save directory exists.
    if !cfg.save_dir.is_empty() {
        fs::create_dir_all(&cfg.save_dir)
            .map_err(|e| TensorError::new(&format!("create save dir: {e}")))?;
    }

    // Open streaming log.
    let mut log_file = if !cfg.save_dir.is_empty() {
        let path = format!("{}/training.log", cfg.save_dir);
        let mut f = fs::File::create(&path)
            .map_err(|e| TensorError::new(&format!("create log: {e}")))?;
        writeln!(f, "# fbrl letter training").ok();
        writeln!(f, "# epochs={}  batch={}  lr={:.4}→{:.4}  scan={}  read={}",
            cfg.epochs, cfg.batch_size, cfg.lr, cfg.min_lr, cfg.n_scan, cfg.n_read).ok();
        Some(f)
    } else {
        None
    };

    // Live monitor (optional).
    let mut monitor = if let Some(port) = cfg.monitor_port {
        let mut m = Monitor::new(cfg.epochs);
        if let Err(e) = m.serve(port) {
            eprintln!("warning: monitor server: {e}");
        }
        m.watch(&model.graph);
        m.set_metadata(serde_json::to_value(cfg).unwrap_or_default());
        if !cfg.save_dir.is_empty() {
            m.save_html(&format!("{}/dashboard.html", cfg.save_dir));
        }
        Some(m)
    } else {
        None
    };

    let metric_tags: &[&str] = &[
        "letter_ce", "case_ce", "letter_acc", "case_acc",
        "recon_mse", "guide", "diversity", "total", "hit_rate", "lr",
    ];

    for epoch in 0..cfg.epochs {
        // Enable profiling on the last epoch so finish_with() captures node timings.
        if epoch + 1 == cfg.epochs {
            model.graph.enable_profiling();
        }

        loader.reset();
        let epoch_start = Instant::now();
        let mut n_batches = 0usize;

        // Update LR at start of epoch.
        let current_lr = scheduler.lr(epoch);
        optimizer.set_lr(current_lr);

        while let Some(batch) = loader.next_batch()? {
            let img_var = Variable::new(batch.image, false);
            let case_var = Variable::new(batch.case_label.clone(), false);
            let clean_var = Variable::new(batch.clean, false);

            let result = model.forward(&img_var, &case_var)?;

            // Classification losses (cross_entropy accepts [B] int64 indices).
            let letter_idx = batch.letter_idx;
            let case_idx = case_idx_from_float(&batch.case_label)?;
            let letter_target = Variable::new(letter_idx.clone(), false);
            let case_target = Variable::new(case_idx.clone(), false);
            let letter_loss = cross_entropy_loss(&result.letter_logits, &letter_target)?;
            let case_loss = cross_entropy_loss(&result.case_logits, &case_target)?;

            // Reconstruction loss.
            let recon_loss = mse_loss(&result.recon, &img_var)?;

            // Attention losses — separate guide weights for scan vs read.
            let scan_guide = attention_guide_loss(
                &clean_var, &result.scan_locations, cfg.blur_sigma_ratio,
            )?;
            let read_guide = attention_guide_loss(
                &clean_var, &result.read_locations, cfg.blur_sigma_ratio,
            )?;

            // Diversity across all locations combined.
            let all_locs: Vec<Variable> = result.scan_locations.iter()
                .chain(result.read_locations.iter()).cloned().collect();
            let div_loss = fixation_diversity_loss(
                &all_locs, cfg.diversity_sigma, cfg.diversity_vy,
            )?;

            // Total loss.
            let total = letter_loss.add(&case_loss)?;
            let total = total.add(&recon_loss.mul_scalar(cfg.recon_weight)?)?;
            let total = total.add(&scan_guide.mul_scalar(cfg.scan_guide_weight)?)?;
            let total = total.add(&read_guide.mul_scalar(cfg.read_guide_weight)?)?;
            let total = total.add(&div_loss.mul_scalar(cfg.diversity_weight)?)?;

            optimizer.zero_grad();
            total.backward()?;
            clip_grad_norm(&params, cfg.max_grad_norm)?;
            optimizer.step()?;

            // Break gradient chain.
            model.graph.detach_state();

            // Record metrics.
            let (hr, _) = fixation_hit_rate(&clean_var, &all_locs, 0.3)?;
            let letter_acc = accuracy(&result.letter_logits.data(), &letter_idx)?;
            let case_acc_val = accuracy(&result.case_logits.data(), &case_idx)?;

            model.graph.record_scalar("letter_ce", letter_loss.item()?);
            model.graph.record_scalar("case_ce", case_loss.item()?);
            model.graph.record_scalar("letter_acc", letter_acc);
            model.graph.record_scalar("case_acc", case_acc_val);
            model.graph.record_scalar("recon_mse", recon_loss.item()?);
            model.graph.record_scalar("guide",
                scan_guide.item()? * cfg.scan_guide_weight
                + read_guide.item()? * cfg.read_guide_weight);
            model.graph.record_scalar("diversity", div_loss.item()?);
            model.graph.record_scalar("total", total.item()?);
            model.graph.record_scalar("hit_rate", hr);
            model.graph.record_scalar("lr", current_lr);

            n_batches += 1;
        }

        if n_batches == 0 {
            return Err(TensorError::new(
                &format!("epoch {epoch}: no batches produced"),
            ));
        }

        // Flush batch means → epoch history.
        model.graph.flush(&[]);

        let epoch_dur = epoch_start.elapsed();
        let eta_secs = model.graph.eta(cfg.epochs);
        let eta = std::time::Duration::from_secs_f64(eta_secs);

        let stats = EpochStats {
            epoch,
            letter_loss: model.graph.trend("letter_ce").latest(),
            case_loss: model.graph.trend("case_ce").latest(),
            letter_acc: model.graph.trend("letter_acc").latest(),
            case_acc: model.graph.trend("case_acc").latest(),
            recon_loss: model.graph.trend("recon_mse").latest(),
            guide_loss: model.graph.trend("guide").latest(),
            div_loss: model.graph.trend("diversity").latest(),
            total_loss: model.graph.trend("total").latest(),
            hit_rate: model.graph.trend("hit_rate").latest(),
            lr: current_lr,
            duration: epoch_dur,
            eta,
        };

        if let Some(cb) = on_epoch {
            cb(&stats);
        }

        // Push to live monitor (graph already has all metrics from record_scalar + flush).
        if let Some(ref mut m) = monitor {
            m.log(epoch, epoch_dur, &model.graph);
        }

        // Append to streaming log.
        if let Some(ref mut f) = log_file {
            writeln!(f,
                "epoch {:3}  ltr={:.4}({:.0}%)  case={:.4}({:.0}%)  recon={:.4}  guide={:.4}  div={:.4}  hit={:.0}%  lr={:.6}  [{:?}]  ETA {:?}",
                epoch + 1,
                stats.letter_loss, stats.letter_acc * 100.0,
                stats.case_loss, stats.case_acc * 100.0,
                stats.recon_loss, stats.guide_loss, stats.div_loss,
                stats.hit_rate * 100.0, stats.lr,
                stats.duration, stats.eta,
            ).ok();
            f.flush().ok();
        }

        // Checkpoint.
        if !cfg.save_dir.is_empty() && cfg.checkpoint_interval > 0
            && (epoch + 1) % cfg.checkpoint_interval == 0
        {
            let path = format!("{}/checkpoint_epoch_{}.fdl.gz", cfg.save_dir, epoch + 1);
            model.graph.save_checkpoint(&path)?;
        }
    }

    // Finalize live monitor.
    if let Some(ref mut m) = monitor {
        m.finish_with(&model.graph);
    }

    // Save final outputs.
    if !cfg.save_dir.is_empty() {
        model.graph.save_checkpoint(
            &format!("{}/model_final.fdl.gz", cfg.save_dir),
        )?;
        if let Err(e) = model.graph.plot_html(
            &format!("{}/training.html", cfg.save_dir), metric_tags,
        ) {
            eprintln!("warning: plot HTML: {e}");
        }
        if let Err(e) = model.graph.export_trends(
            &format!("{}/training.csv", cfg.save_dir), metric_tags,
        ) {
            eprintln!("warning: export CSV: {e}");
        }

        // Save manifest with config + final results.
        let manifest = serde_json::json!({
            "framework": "flodl",
            "model": "letter",
            "config": cfg,
            "results": {
                "letter_acc": model.graph.trend("letter_acc").latest(),
                "case_acc": model.graph.trend("case_acc").latest(),
                "letter_ce": model.graph.trend("letter_ce").latest(),
                "recon_mse": model.graph.trend("recon_mse").latest(),
            },
            "files": {
                "model": "model_final.fdl.gz",
                "dashboard": if cfg.monitor_port.is_some() { "dashboard.html" } else { "" },
            },
            "parent": null,
        });
        if let Err(e) = fs::write(
            format!("{}/manifest.json", cfg.save_dir),
            serde_json::to_string_pretty(&manifest).unwrap_or_default(),
        ) {
            eprintln!("warning: write manifest: {e}");
        }
    }

    Ok(())
}

/// Convert [B, 1] float case labels to [B] int64 indices.
fn case_idx_from_float(case_label: &Tensor) -> Result<Tensor> {
    case_label.squeeze(1)?.gt_scalar(0.5)?.to_dtype(DType::Int64)
}

/// Compute classification accuracy from logits and target indices (on-device).
fn accuracy(logits: &Tensor, targets: &Tensor) -> Result<f64> {
    let preds = logits.argmax(1, false)?;
    preds.eq_tensor(targets)?.mean()?.item()
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use super::*;
    use crate::letter::new_synthetic_dataset;

    #[test]
    fn train_letter_smoke() {
        let ds = new_synthetic_dataset(64).expect("synthetic dataset");

        let cfg = LetterConfig {
            n_scan: 1,
            n_read: 2,
            patch_size: 8,
            scan_patch_w: 12,
            latent_dim: 32,
            batch_size: 16,
            epochs: 2,
            lr: 0.001,
            save_dir: String::new(), // no checkpoints
            ..Default::default()
        };

        let stats = RefCell::new(Vec::new());
        train_letter(&cfg, &ds, Some(&|s: &EpochStats| {
            eprintln!(
                "epoch {:2}  ltr={:.4}({:.0}%)  case={:.4}({:.0}%)  recon={:.4}  \
                 guide={:.4}  div={:.4}  total={:.4}  hit={:.0}%  lr={:.6}  [{:?}]  ETA {:?}",
                s.epoch + 1,
                s.letter_loss, s.letter_acc * 100.0,
                s.case_loss, s.case_acc * 100.0,
                s.recon_loss, s.guide_loss, s.div_loss,
                s.total_loss, s.hit_rate * 100.0, s.lr,
                s.duration, s.eta,
            );
            stats.borrow_mut().push((s.letter_loss, s.recon_loss, s.lr));
        }))
        .expect("train_letter should succeed");

        let stats = stats.into_inner();
        assert_eq!(stats.len(), 2, "expected 2 epoch stats");

        // Losses should be finite positive numbers.
        for (i, &(letter, recon, _)) in stats.iter().enumerate() {
            assert!(letter > 0.0, "epoch {i}: letter loss should be positive, got {letter}");
            assert!(recon > 0.0, "epoch {i}: recon loss should be positive, got {recon}");
        }

        // LR should decrease with cosine schedule.
        assert!(
            stats[1].2 < stats[0].2,
            "LR should decrease: epoch 0={}, epoch 1={}",
            stats[0].2, stats[1].2,
        );
    }
}
