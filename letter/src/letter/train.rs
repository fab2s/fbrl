//! Training loop and configuration for the letter model.

use std::fs;
use std::io::Write;
use std::time::Instant;

use flodl::autograd::{Variable, grid_sample};
use flodl::nn::{
    cross_entropy_loss, mse_loss, bce_with_logits_loss, clip_grad_norm,
    Adam, CosineScheduler, Module, Optimizer,
};
use flodl::monitor::Monitor;
use flodl::tensor::{cuda_available, Device, DType, Result, Tensor, TensorError, TensorOptions};
use flodl::CpuWorker;
use serde::{Serialize, Deserialize};

use super::data::{LetterDataset, LetterLoader};
use super::loss::{attention_guide_loss, build_void_grid, fixation_diversity_loss, fixation_hit_rate, void_repulsion_with_grid};
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
    pub scan_void_weight: f64,
    pub void_weight: f64,
    pub diversity_weight: f64,
    pub recon_weight: f64,
    pub recode_weight: f64,
    pub content_weight: f64,
    pub blur_sigma_ratio: f64,
    pub diversity_sigma: f64,
    pub scan_vy: f64,
    pub read_vy: f64,

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

            batch_size: 52,
            epochs: 100,
            lr: 0.001,
            min_lr: 0.0,
            max_grad_norm: 5.0,

            scan_guide_weight: 8.0,
            read_guide_weight: 0.0,
            scan_void_weight: 1.5,
            void_weight: 0.5,
            diversity_weight: 1.0,
            recon_weight: 1.0,
            recode_weight: 1.0,
            content_weight: 0.5,
            blur_sigma_ratio: 0.16,
            diversity_sigma: 0.1,
            scan_vy: 0.3,
            read_vy: 1.0,

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
    pub recode_loss: f64,
    pub content_loss: f64,
    pub guide_loss: f64,
    pub void_loss: f64,
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
        flodl::tensor::set_cudnn_benchmark(true);
        model.graph.set_device(Device::CUDA(0));
        Device::CUDA(0)
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

    // Ensure save directory exists and rotate prior run artifacts.
    if !cfg.save_dir.is_empty() {
        fs::create_dir_all(&cfg.save_dir)
            .map_err(|e| TensorError::new(&format!("create save dir: {e}")))?;
        rotate_prior_run(&cfg.save_dir);
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

    // Background worker for async checkpointing.
    let mut worker = CpuWorker::new();

    let metric_tags: &[&str] = &[
        "letter_ce", "case_ce", "letter_acc", "case_acc",
        "recon_mse", "recode", "content", "guide", "void", "diversity",
        "total", "hit_rate", "lr",
    ];

    // Pre-build void repulsion grids (image dims are constant across training).
    let img_shape = ds.samples[0].image.shape();
    let (img_h, img_w) = (img_shape[1], img_shape[2]);
    let scan_void_grid = if cfg.scan_void_weight > 0.0 {
        Some(build_void_grid(cfg.patch_size, cfg.scan_patch_w, img_h, img_w, device)?)
    } else { None };
    let read_void_grid = if cfg.void_weight > 0.0 {
        Some(build_void_grid(cfg.patch_size, cfg.patch_size, img_h, img_w, device)?)
    } else { None };

    let train_start = Instant::now();
    let mut epoch_times: Vec<f64> = Vec::with_capacity(cfg.epochs);

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
            let partner_clean_var = Variable::new(batch.partner_clean, false);

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

            // Recode loss: decode latent with flipped case, compare to partner clean.
            let recode_loss = if cfg.recode_weight > 0.0 && ds.has_partners {
                let b = batch.case_label.shape()[0];
                let ones_t = Tensor::from_f32(
                    &vec![1.0f32; b as usize], &[b, 1], device,
                )?;
                let ones_var = Variable::new(ones_t, false);
                let flipped_case = ones_var.sub(&case_var)?;
                let z_recode = result.latent.cat(&flipped_case, 1)?;
                let recode = model.decoder.forward(&z_recode)?;
                mse_loss(&recode, &partner_clean_var)?
            } else {
                Variable::new(Tensor::zeros(&[1], TensorOptions { device, ..Default::default() })?, false)
            };

            // Content loss: BCE on whether scan fixation has ink.
            let content_loss = if cfg.content_weight > 0.0 && cfg.n_scan > 0 {
                let scan_logits = model.content_logits();
                let scan_locs = &result.scan_locations;
                if !scan_logits.is_empty() {
                    let zero = Variable::new(
                        Tensor::zeros(&[1], TensorOptions { device, ..Default::default() })?, false,
                    );
                    let mut loss_sum = zero;
                    for (loc, logit) in scan_locs.iter().zip(scan_logits.iter()) {
                        let grid = loc.unsqueeze(1)?.unsqueeze(2)?; // [B, 1, 1, 2]
                        let sampled = grid_sample(&clean_var, &grid, 0, 0, true)?;
                        let label_t = sampled.data().reshape(&[-1, 1])?
                            .gt_scalar(0.1)?.to_dtype(DType::Float32)?;
                        let label = Variable::new(label_t, false);
                        let step_loss = bce_with_logits_loss(logit, &label)?;
                        loss_sum = loss_sum.add(&step_loss)?;
                    }
                    loss_sum.mul_scalar(1.0 / scan_logits.len() as f64)?
                } else {
                    Variable::new(Tensor::zeros(&[1], TensorOptions { device, ..Default::default() })?, false)
                }
            } else {
                Variable::new(Tensor::zeros(&[1], TensorOptions { device, ..Default::default() })?, false)
            };

            // Attention losses — separate guide weights for scan vs read.
            let scan_guide = if cfg.scan_guide_weight > 0.0 {
                attention_guide_loss(
                    &clean_var, &result.scan_locations, cfg.blur_sigma_ratio,
                )?
            } else {
                Variable::new(Tensor::zeros(&[1], TensorOptions { device, ..Default::default() })?, false)
            };
            let read_guide = if cfg.read_guide_weight > 0.0 {
                attention_guide_loss(
                    &clean_var, &result.read_locations, cfg.blur_sigma_ratio,
                )?
            } else {
                Variable::new(Tensor::zeros(&[1], TensorOptions { device, ..Default::default() })?, false)
            };

            // Void repulsion — penalize fixations landing in empty space.
            let scan_void = if cfg.scan_void_weight > 0.0 {
                void_repulsion_with_grid(
                    &clean_var, &result.scan_locations,
                    cfg.patch_size, cfg.scan_patch_w, 0.1,
                    scan_void_grid.as_ref(),
                )?
            } else {
                Variable::new(Tensor::zeros(&[1], TensorOptions { device, ..Default::default() })?, false)
            };
            let read_void = if cfg.void_weight > 0.0 {
                void_repulsion_with_grid(
                    &clean_var, &result.read_locations,
                    cfg.patch_size, cfg.patch_size, 0.1,
                    read_void_grid.as_ref(),
                )?
            } else {
                Variable::new(Tensor::zeros(&[1], TensorOptions { device, ..Default::default() })?, false)
            };

            // Split diversity: separate scan/read with different VY scaling.
            let scan_div = fixation_diversity_loss(
                &result.scan_locations, cfg.diversity_sigma, cfg.scan_vy,
            )?;
            let read_div = fixation_diversity_loss(
                &result.read_locations, cfg.diversity_sigma, cfg.read_vy,
            )?;
            let div_loss = scan_div.add(&read_div)?;

            // Total loss.
            let total = letter_loss.add(&case_loss)?;
            let total = total.add(&recon_loss.mul_scalar(cfg.recon_weight)?)?;
            let total = total.add(&recode_loss.mul_scalar(cfg.recode_weight)?)?;
            let total = total.add(&content_loss.mul_scalar(cfg.content_weight)?)?;
            let total = total.add(&scan_guide.mul_scalar(cfg.scan_guide_weight)?)?;
            let total = total.add(&read_guide.mul_scalar(cfg.read_guide_weight)?)?;
            let total = total.add(&scan_void.mul_scalar(cfg.scan_void_weight)?)?;
            let total = total.add(&read_void.mul_scalar(cfg.void_weight)?)?;
            let total = total.add(&div_loss.mul_scalar(cfg.diversity_weight)?)?;

            optimizer.zero_grad();
            total.backward()?;
            clip_grad_norm(&params, cfg.max_grad_norm)?;
            optimizer.step()?;

            // Break gradient chain.
            model.graph.detach_state();

            // Record metrics.
            let all_locs: Vec<Variable> = result.scan_locations.iter()
                .chain(result.read_locations.iter()).cloned().collect();
            let (hr, _) = fixation_hit_rate(&clean_var, &all_locs, 0.3)?;
            let letter_acc = accuracy(&result.letter_logits.data(), &letter_idx)?;
            let case_acc_val = accuracy(&result.case_logits.data(), &case_idx)?;

            model.graph.record_scalar("letter_ce", letter_loss.item()?);
            model.graph.record_scalar("case_ce", case_loss.item()?);
            model.graph.record_scalar("letter_acc", letter_acc);
            model.graph.record_scalar("case_acc", case_acc_val);
            model.graph.record_scalar("recon_mse", recon_loss.item()?);
            model.graph.record_scalar("recode", recode_loss.item()?);
            model.graph.record_scalar("content", content_loss.item()?);
            model.graph.record_scalar("guide",
                scan_guide.item()? * cfg.scan_guide_weight
                + read_guide.item()? * cfg.read_guide_weight);
            model.graph.record_scalar("void",
                scan_void.item()? * cfg.scan_void_weight
                + read_void.item()? * cfg.void_weight);
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
        epoch_times.push((epoch_dur.as_secs_f64() * 100.0).round() / 100.0);
        let eta_secs = model.graph.eta(cfg.epochs);
        let eta = std::time::Duration::from_secs_f64(eta_secs);

        let stats = EpochStats {
            epoch,
            letter_loss: model.graph.trend("letter_ce").latest(),
            case_loss: model.graph.trend("case_ce").latest(),
            letter_acc: model.graph.trend("letter_acc").latest(),
            case_acc: model.graph.trend("case_acc").latest(),
            recon_loss: model.graph.trend("recon_mse").latest(),
            recode_loss: model.graph.trend("recode").latest(),
            content_loss: model.graph.trend("content").latest(),
            guide_loss: model.graph.trend("guide").latest(),
            void_loss: model.graph.trend("void").latest(),
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
                "epoch {:3}  ltr={:.4}({:.0}%)  case={:.4}({:.0}%)  recon={:.4}  recode={:.4}  content={:.4}  guide={:.4}  void={:.4}  div={:.4}  hit={:.0}%  lr={:.6}  [{:?}]  ETA {:?}",
                epoch + 1,
                stats.letter_loss, stats.letter_acc * 100.0,
                stats.case_loss, stats.case_acc * 100.0,
                stats.recon_loss, stats.recode_loss, stats.content_loss,
                stats.guide_loss, stats.void_loss, stats.div_loss,
                stats.hit_rate * 100.0, stats.lr,
                stats.duration, stats.eta,
            ).ok();
            f.flush().ok();
        }

        // Async checkpoint — skip if worker still saving the previous one.
        if !cfg.save_dir.is_empty() && cfg.checkpoint_interval > 0
            && (epoch + 1) % cfg.checkpoint_interval == 0
        {
            if worker.is_idle() {
                let path = format!("{}/checkpoint_epoch_{}.fdl.gz", cfg.save_dir, epoch + 1);
                let snap = model.graph.snapshot_cpu()?;
                worker.submit(move || {
                    if let Err(e) = snap.save_file(&path) {
                        eprintln!("warning: async checkpoint: {e}");
                    }
                });
            } else {
                eprintln!("epoch {}: skipping checkpoint (worker busy)", epoch + 1);
            }
        }
    }

    // Finalize live monitor.
    if let Some(ref mut m) = monitor {
        m.finish_with(&model.graph);
    }

    // Save final outputs.
    if !cfg.save_dir.is_empty() {
        // Submit final checkpoint to worker, then generate artifacts while it saves.
        let final_path = format!("{}/model_final.fdl.gz", cfg.save_dir);
        let snap = model.graph.snapshot_cpu()?;
        worker.submit(move || {
            if let Err(e) = snap.save_file(&final_path) {
                eprintln!("warning: final checkpoint: {e}");
            }
        });

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

        // Benchmark report.
        let total_time = train_start.elapsed().as_secs_f64();
        let rss_mb = (flodl::tensor::rss_kb() as f64 / 1024.0).round() as i64;
        let mut bench = serde_json::json!({
            "framework": "flodl",
            "model": "letter",
            "config": {
                "n_scan": cfg.n_scan,
                "n_read": cfg.n_read,
                "latent_dim": cfg.latent_dim,
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
            },
            "ram_peak_rss_mb": rss_mb,
            "total_time_s": (total_time * 10.0).round() / 10.0,
            "avg_epoch_s": ((total_time / cfg.epochs as f64) * 10.0).round() / 10.0,
            "epoch_times_s": epoch_times,
        });
        if device != Device::CPU {
            if let Ok((used, total)) = flodl::tensor::cuda_memory_info() {
                let gpu = flodl::tensor::cuda_device_name().unwrap_or_default();
                bench["gpu"] = serde_json::json!(gpu);
                bench["vram"] = serde_json::json!({
                    "device_used_mb": (used as f64 / 1024.0 / 1024.0).round() as i64,
                    "device_total_mb": (total as f64 / 1024.0 / 1024.0).round() as i64,
                });
            }
        }
        let bench_path = format!("{}/benchmark.json", cfg.save_dir);
        if let Err(e) = fs::write(
            &bench_path,
            serde_json::to_string_pretty(&bench).unwrap_or_default(),
        ) {
            eprintln!("warning: write benchmark: {e}");
        }

        // Print summary.
        if let Some(gpu) = bench.get("gpu") {
            eprintln!("\n--- Benchmark ---");
            eprintln!("GPU:         {}", gpu.as_str().unwrap_or("?"));
            if let Some(vram) = bench.get("vram") {
                eprintln!("VRAM:        {} MB device",
                    vram["device_used_mb"]);
            }
        } else {
            eprintln!("\n--- Benchmark ---");
        }
        eprintln!("RAM:         {} MB peak RSS", bench["ram_peak_rss_mb"]);
        eprintln!("Avg epoch:   {}s  ({} epochs, {}s total)",
            bench["avg_epoch_s"], cfg.epochs, bench["total_time_s"]);
        eprintln!("Saved:       {bench_path}");

        // Wait for all background saves to complete.
        worker.finish();
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

/// Rotate prior run artifacts in place by appending a timestamp suffix.
///
/// Renames `training.log` → `training_YYYYMMDD_HHMMSS.log`, etc.
/// Also rotates `manifest.json`, `dashboard.html`, and `*.fdl.gz` checkpoints.
/// No-op if no log file exists.
fn rotate_prior_run(save_dir: &str) {
    let log_path = format!("{save_dir}/training.log");
    if !std::path::Path::new(&log_path).exists() {
        return;
    }

    // Timestamp from the log file's modification time, fallback to now.
    let ts = fs::metadata(&log_path)
        .and_then(|m| m.modified())
        .unwrap_or_else(|_| std::time::SystemTime::now());
    let secs = ts.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
    let (s, mi, h) = (secs % 60, (secs / 60) % 60, (secs / 3600) % 24);
    let days = secs / 86400;
    let (y, mo, d) = civil_from_days(days as i64);
    let stamp = format!("{y:04}{mo:02}{d:02}_{h:02}{mi:02}{s:02}");

    // Files with extensions: rename base_STAMP.ext
    let ext_files = [
        "training.log", "training.csv", "training.html",
        "manifest.json", "dashboard.html",
    ];
    for name in &ext_files {
        let src = format!("{save_dir}/{name}");
        if std::path::Path::new(&src).exists() {
            let rotated = if let Some(dot) = name.rfind('.') {
                format!("{save_dir}/{}_{stamp}.{}", &name[..dot], &name[dot + 1..])
            } else {
                format!("{save_dir}/{name}_{stamp}")
            };
            let _ = fs::rename(&src, &rotated);
        }
    }

    eprintln!("Rotated prior run artifacts with suffix _{stamp}");
}

/// Convert days since Unix epoch to (year, month, day). Civil calendar.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    // Algorithm from Howard Hinnant (public domain).
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
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
                 guide={:.4}  void={:.4}  div={:.4}  total={:.4}  hit={:.0}%  lr={:.6}  [{:?}]  ETA {:?}",
                s.epoch + 1,
                s.letter_loss, s.letter_acc * 100.0,
                s.case_loss, s.case_acc * 100.0,
                s.recon_loss, s.guide_loss, s.void_loss, s.div_loss,
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
