//! Training loop for the word model.

use std::fs;
use std::io::Write;
use std::time::Instant;

use flodl::autograd::Variable;
use flodl::nn::{
    cross_entropy_loss, mse_loss, clip_grad_norm,
    Adam, CosineScheduler, Module, Optimizer,
};
use flodl::monitor::Monitor;
use flodl::tensor::{cuda_available, Device, DType, Result, Tensor, TensorError, TensorOptions};
use flodl::CpuWorker;
use serde::{Serialize, Deserialize};

use super::data::{WordDataset, WordLoader};
use super::loss;
use super::model::WordModel;

/// Hyperparameters for word model training.
#[derive(Serialize, Deserialize)]
pub struct WordConfig {
    // Model architecture.
    pub n_classes: usize,
    pub n_positions: usize,
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
    pub content_weight: f64,
    pub isolation_weight: f64,
    pub blur_sigma_ratio: f64,
    pub diversity_sigma: f64,
    pub scan_vy: f64,
    pub read_vy: f64,

    // Scaffold.
    pub scaffold_ratio: f64,
    pub scaffold_floor: f64,

    // Data.
    #[serde(default)]
    pub isolation_data_dir: String,

    // Checkpointing.
    #[serde(default)]
    pub save_dir: String,
    #[serde(default = "default_checkpoint_interval")]
    pub checkpoint_interval: usize,

    // Transfer.
    #[serde(default)]
    pub transfer_from: String,

    // Live monitoring.
    #[serde(skip)]
    pub monitor_port: Option<u16>,
}

impl Default for WordConfig {
    fn default() -> Self {
        WordConfig {
            n_classes: 26,
            n_positions: 4,
            n_scan: 8,
            n_read: 12,
            patch_size: 12,
            scan_patch_w: 18,
            n_scales: 1,
            latent_dim: 256,

            batch_size: 32,
            epochs: 200,
            lr: 0.001,
            min_lr: 0.0,
            max_grad_norm: 5.0,

            scan_guide_weight: 8.0,
            read_guide_weight: 8.0,
            diversity_weight: 1.0,
            recon_weight: 1.0,
            content_weight: 0.5,
            isolation_weight: 0.5,
            blur_sigma_ratio: 0.16,
            diversity_sigma: 0.1,
            scan_vy: 0.3,
            read_vy: 1.5,

            scaffold_ratio: 0.67,
            scaffold_floor: 0.0,

            isolation_data_dir: String::new(),
            save_dir: "training".into(),
            checkpoint_interval: 50,
            transfer_from: String::new(),

            monitor_port: None,
        }
    }
}

fn default_checkpoint_interval() -> usize { 50 }

/// Per-epoch averaged metrics.
pub struct WordEpochStats {
    pub epoch: usize,
    pub position_losses: [f64; 4],
    pub position_accs: [f64; 4],
    pub recon_loss: f64,
    pub guide_loss: f64,
    pub div_loss: f64,
    pub content_loss: f64,
    pub isolation_loss: f64,
    pub hit_rate: f64,
    pub scaffold_weight: f64,
    pub total_loss: f64,
    pub lr: f64,
    pub duration: std::time::Duration,
    pub eta: std::time::Duration,
}

/// Run the full training loop for the word model.
pub fn train_word(
    cfg: &WordConfig,
    ds: &WordDataset,
    isolation_ds: Option<&super::data::IsolationDataset>,
    on_epoch: Option<&dyn Fn(&WordEpochStats)>,
) -> Result<()> {
    let model = WordModel::new(
        cfg.n_classes, cfg.n_positions, cfg.n_scan, cfg.n_read,
        cfg.patch_size, cfg.scan_patch_w, cfg.n_scales, cfg.latent_dim,
    )?;

    // Transfer learning: load letter model weights into word model.
    if !cfg.transfer_from.is_empty() {
        let report = model.graph.load_checkpoint(&cfg.transfer_from)?;
        eprintln!("Transfer: {} loaded, {} skipped, {} missing from {}",
            report.loaded.len(), report.skipped.len(), report.missing.len(),
            cfg.transfer_from);
    }

    let device = if cuda_available() {
        eprintln!("Using CUDA");
        flodl::tensor::set_cudnn_benchmark(true);
        model.set_device(Device::CUDA(0));
        Device::CUDA(0)
    } else {
        eprintln!("Using CPU");
        Device::CPU
    };

    let params = model.parameters();
    let mut optimizer = Adam::new(&params, cfg.lr);
    let scheduler = CosineScheduler::new(cfg.lr, cfg.min_lr, cfg.epochs);
    model.train();

    let mut loader = WordLoader::new(ds, cfg.batch_size, true);
    loader.set_device(device);

    let scaffold_epochs = (cfg.scaffold_ratio * cfg.epochs as f64).round() as usize;

    // Ensure save directory exists.
    if !cfg.save_dir.is_empty() {
        fs::create_dir_all(&cfg.save_dir)
            .map_err(|e| TensorError::new(&format!("create save dir: {e}")))?;
    }

    // Streaming log.
    let mut log_file = if !cfg.save_dir.is_empty() {
        let path = format!("{}/training.log", cfg.save_dir);
        let mut f = fs::File::create(&path)
            .map_err(|e| TensorError::new(&format!("create log: {e}")))?;
        writeln!(f, "# fbrl word training").ok();
        writeln!(f, "# epochs={}  batch={}  lr={:.4}->{:.4}  scan={}  read={}  positions={}",
            cfg.epochs, cfg.batch_size, cfg.lr, cfg.min_lr,
            cfg.n_scan, cfg.n_read, cfg.n_positions).ok();
        Some(f)
    } else {
        None
    };

    // Live monitor.
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

    let mut worker = CpuWorker::new();

    let metric_tags: &[&str] = &[
        "pos1_ce", "pos2_ce", "pos3_ce", "pos4_ce",
        "pos1_acc", "pos2_acc", "pos3_acc", "pos4_acc",
        "recon_mse", "guide", "diversity", "content", "isolation",
        "total", "hit_rate", "scaffold", "lr",
    ];

    // Group boundaries for cross-attention readout.
    let reads_per_pos = cfg.n_read / cfg.n_positions;
    let group_boundaries: Vec<usize> = (0..cfg.n_positions)
        .map(|p| p * reads_per_pos)
        .collect();

    let train_start = Instant::now();
    let mut epoch_times: Vec<f64> = Vec::with_capacity(cfg.epochs);

    for epoch in 0..cfg.epochs {
        if epoch + 1 == cfg.epochs {
            model.graph.enable_profiling();
        }

        loader.reset();
        let epoch_start = Instant::now();
        let mut n_batches = 0usize;

        let current_lr = scheduler.lr(epoch);
        optimizer.set_lr(current_lr);

        // Scaffold weight: 1.0 → floor over scaffold_epochs.
        let scaffold_weight = if scaffold_epochs > 0 {
            (cfg.scaffold_floor).max(1.0 - epoch as f64 / scaffold_epochs as f64)
        } else {
            cfg.scaffold_floor
        };

        while let Some(batch) = loader.next_batch()? {
            let img_var = Variable::new(batch.image, false);
            let clean_var = Variable::new(batch.clean, false);

            let result = model.forward(&img_var)?;

            // Per-position classification losses.
            let mut cls_total = zero_var(device)?;
            for p in 0..cfg.n_positions {
                let target = Variable::new(batch.letter_idx[p].clone(), false);
                let pos_loss = cross_entropy_loss(&result.position_logits[p], &target)?;
                cls_total = cls_total.add(&pos_loss)?;
                model.graph.record_scalar(
                    &format!("pos{}_ce", p + 1), pos_loss.item()?);

                let acc = accuracy(&result.position_logits[p].data(), &batch.letter_idx[p])?;
                model.graph.record_scalar(&format!("pos{}_acc", p + 1), acc);
            }

            // Reconstruction loss.
            let recon_loss = mse_loss(&result.recon, &img_var)?;

            // Scan attention guide.
            let scan_guide = loss::scan_guide_loss(
                &clean_var, &result.scan_locations, cfg.blur_sigma_ratio,
            )?;

            // Read scaffold guide.
            let read_guide = loss::read_scaffold_loss(
                &clean_var, &result.read_locations, cfg.blur_sigma_ratio,
                &group_boundaries, cfg.n_positions, scaffold_weight,
            )?;

            // Diversity losses (split scan/read).
            let scan_div = loss::fixation_diversity_loss(
                &result.scan_locations, cfg.diversity_sigma, cfg.scan_vy,
            )?;
            let read_div = loss::fixation_diversity_loss(
                &result.read_locations, cfg.diversity_sigma, cfg.read_vy,
            )?;
            let div_loss = scan_div.add(&read_div)?;

            // Content detection (scan only).
            let scan_logits = model.content_logits();
            let content = loss::content_loss(
                &clean_var, &result.scan_locations, &scan_logits, device,
            )?;

            // Isolation loss: forward 128×128 single-letter images through word model.
            let isolation_loss = if cfg.isolation_weight > 0.0 {
                if let Some(iso) = isolation_ds {
                    let iso_k = (n_batches % cfg.n_positions) as usize; // rotate position
                    let b = batch.letter_idx[0].shape()[0] as usize;

                    // Build isolation batch: for each sample, pick the letter at position iso_k.
                    let mut iso_images = Vec::with_capacity(b);
                    let mut iso_targets = Vec::with_capacity(b);
                    let targets_data = batch.letter_idx[iso_k].to_i64_vec()?;
                    for bi in 0..b {
                        let letter_idx = targets_data[bi];
                        // Find matching isolation image.
                        if let Some(sample) = iso.samples.iter()
                            .find(|s| s.letter_idx == letter_idx)
                        {
                            iso_images.push(&sample.image);
                            iso_targets.push(letter_idx);
                        }
                    }

                    if !iso_images.is_empty() {
                        let iso_img = Tensor::stack(&iso_images, 0)?.to_device(device)?;
                        let iso_var = Variable::new(iso_img, false);
                        let iso_result = model.forward(&iso_var)?;
                        let iso_target = Tensor::from_i64(
                            &iso_targets, &[iso_targets.len() as i64], device,
                        )?;
                        cross_entropy_loss(
                            &iso_result.position_logits[iso_k],
                            &Variable::new(iso_target, false),
                        )?
                    } else {
                        zero_var(device)?
                    }
                } else {
                    zero_var(device)?
                }
            } else {
                zero_var(device)?
            };

            // Total loss.
            let total = cls_total
                .add(&recon_loss.mul_scalar(cfg.recon_weight)?)?
                .add(&scan_guide.mul_scalar(cfg.scan_guide_weight)?)?
                .add(&read_guide.mul_scalar(cfg.read_guide_weight)?)?
                .add(&div_loss.mul_scalar(cfg.diversity_weight)?)?
                .add(&content.mul_scalar(cfg.content_weight)?)?
                .add(&isolation_loss.mul_scalar(cfg.isolation_weight)?)?;

            optimizer.zero_grad();
            total.backward()?;
            clip_grad_norm(&params, cfg.max_grad_norm)?;
            optimizer.step()?;

            model.graph.detach_state();

            // Record metrics.
            model.graph.record_scalar("recon_mse", recon_loss.item()?);
            model.graph.record_scalar("guide",
                scan_guide.item()? * cfg.scan_guide_weight
                + read_guide.item()? * cfg.read_guide_weight);
            model.graph.record_scalar("diversity", div_loss.item()?);
            model.graph.record_scalar("content", content.item()?);
            model.graph.record_scalar("isolation", isolation_loss.item()?);
            model.graph.record_scalar("total", total.item()?);
            model.graph.record_scalar("scaffold", scaffold_weight);
            model.graph.record_scalar("lr", current_lr);

            // Hit rate.
            let all_locs: Vec<Variable> = result.scan_locations.iter()
                .chain(result.read_locations.iter()).cloned().collect();
            let hr = hit_rate(&clean_var, &all_locs)?;
            model.graph.record_scalar("hit_rate", hr);

            n_batches += 1;
        }

        if n_batches == 0 {
            return Err(TensorError::new(&format!("epoch {epoch}: no batches")));
        }

        model.graph.flush(&[]);

        let epoch_dur = epoch_start.elapsed();
        epoch_times.push((epoch_dur.as_secs_f64() * 100.0).round() / 100.0);
        let eta_secs = model.graph.eta(cfg.epochs);
        let eta = std::time::Duration::from_secs_f64(eta_secs);

        let stats = WordEpochStats {
            epoch,
            position_losses: [
                model.graph.trend("pos1_ce").latest(),
                model.graph.trend("pos2_ce").latest(),
                model.graph.trend("pos3_ce").latest(),
                model.graph.trend("pos4_ce").latest(),
            ],
            position_accs: [
                model.graph.trend("pos1_acc").latest(),
                model.graph.trend("pos2_acc").latest(),
                model.graph.trend("pos3_acc").latest(),
                model.graph.trend("pos4_acc").latest(),
            ],
            recon_loss: model.graph.trend("recon_mse").latest(),
            guide_loss: model.graph.trend("guide").latest(),
            div_loss: model.graph.trend("diversity").latest(),
            content_loss: model.graph.trend("content").latest(),
            isolation_loss: model.graph.trend("isolation").latest(),
            hit_rate: model.graph.trend("hit_rate").latest(),
            scaffold_weight,
            total_loss: model.graph.trend("total").latest(),
            lr: current_lr,
            duration: epoch_dur,
            eta,
        };

        if let Some(cb) = on_epoch { cb(&stats); }

        if let Some(ref mut m) = monitor {
            m.log(epoch, epoch_dur, &model.graph);
        }

        // Streaming log line.
        if let Some(ref mut f) = log_file {
            let pos_str = (0..4).map(|p| format!(
                "{:.4}({:.0}%)", stats.position_losses[p], stats.position_accs[p] * 100.0
            )).collect::<Vec<_>>().join("  ");
            writeln!(f,
                "epoch {:3}  {}  recon={:.4}  guide={:.4}  div={:.4}  cont={:.4}  iso={:.4}  \
                 hit={:.0}%  scaff={:.2}  lr={:.6}  [{:?}]  ETA {:?}",
                epoch + 1, pos_str,
                stats.recon_loss, stats.guide_loss, stats.div_loss,
                stats.content_loss, stats.isolation_loss,
                stats.hit_rate * 100.0, scaffold_weight, stats.lr,
                stats.duration, stats.eta,
            ).ok();
            f.flush().ok();
        }

        // Async checkpoint.
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
            }
        }
    }

    // Finalize.
    if let Some(ref mut m) = monitor {
        m.finish_with(&model.graph);
    }

    if !cfg.save_dir.is_empty() {
        let final_path = format!("{}/model_final.fdl.gz", cfg.save_dir);
        let snap = model.graph.snapshot_cpu()?;
        worker.submit(move || {
            if let Err(e) = snap.save_file(&final_path) {
                eprintln!("warning: final checkpoint: {e}");
            }
        });

        if let Err(e) = model.graph.plot_html(
            &format!("{}/training.html", cfg.save_dir), metric_tags,
        ) { eprintln!("warning: plot: {e}"); }

        if let Err(e) = model.graph.export_trends(
            &format!("{}/training.csv", cfg.save_dir), metric_tags,
        ) { eprintln!("warning: csv: {e}"); }

        // Manifest.
        let manifest = serde_json::json!({
            "framework": "flodl",
            "model": "word",
            "config": cfg,
            "results": {
                "pos1_acc": model.graph.trend("pos1_acc").latest(),
                "pos2_acc": model.graph.trend("pos2_acc").latest(),
                "pos3_acc": model.graph.trend("pos3_acc").latest(),
                "pos4_acc": model.graph.trend("pos4_acc").latest(),
                "recon_mse": model.graph.trend("recon_mse").latest(),
            },
        });
        if let Err(e) = fs::write(
            format!("{}/manifest.json", cfg.save_dir),
            serde_json::to_string_pretty(&manifest).unwrap_or_default(),
        ) { eprintln!("warning: manifest: {e}"); }

        // Benchmark.
        let total_time = train_start.elapsed().as_secs_f64();
        let rss_mb = (flodl::tensor::rss_kb() as f64 / 1024.0).round() as i64;
        let mut bench = serde_json::json!({
            "framework": "flodl",
            "model": "word",
            "config": {
                "n_positions": cfg.n_positions,
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
        if let Err(e) = fs::write(
            format!("{}/benchmark.json", cfg.save_dir),
            serde_json::to_string_pretty(&bench).unwrap_or_default(),
        ) { eprintln!("warning: benchmark: {e}"); }

        if let Some(gpu) = bench.get("gpu") {
            eprintln!("\n--- Benchmark ---");
            eprintln!("GPU:         {}", gpu.as_str().unwrap_or("?"));
            if let Some(vram) = bench.get("vram") {
                eprintln!("VRAM:        {} MB device", vram["device_used_mb"]);
            }
        } else {
            eprintln!("\n--- Benchmark ---");
        }
        eprintln!("RAM:         {} MB peak RSS", bench["ram_peak_rss_mb"]);
        eprintln!("Avg epoch:   {}s  ({} epochs, {}s total)",
            bench["avg_epoch_s"], cfg.epochs, bench["total_time_s"]);

        worker.finish();
    }

    Ok(())
}

fn zero_var(device: Device) -> Result<Variable> {
    let z = Tensor::zeros(&[1], TensorOptions { device, ..Default::default() })?;
    Ok(Variable::new(z, false))
}

fn accuracy(logits: &Tensor, targets: &Tensor) -> Result<f64> {
    let preds = logits.argmax(1, false)?;
    preds.eq_tensor(targets)?.mean()?.item()
}

fn hit_rate(image: &Variable, locations: &[Variable], ) -> Result<f64> {
    if locations.is_empty() { return Ok(0.0); }
    let img_data = image.data();
    let loc_tensors: Vec<Tensor> = locations.iter().map(|l| l.data()).collect();
    let loc_refs: Vec<&Tensor> = loc_tensors.iter().collect();
    let stacked = Tensor::stack(&loc_refs, 1)?;
    let grid = stacked.unsqueeze(2)?;
    let sampled = img_data.grid_sample(&grid, 0, 0, true)?;
    let vals = sampled.to_f32_vec()?;
    let hits = vals.iter().filter(|&&v| v as f64 > 0.3).count();
    Ok(hits as f64 / vals.len().max(1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::word::new_synthetic_dataset;

    #[test]
    fn train_word_smoke() {
        let samples = new_synthetic_dataset(64).expect("synthetic data");
        let ds = WordDataset { samples };

        let cfg = WordConfig {
            n_scan: 2,
            n_read: 4,
            patch_size: 8,
            scan_patch_w: 12,
            latent_dim: 32,
            batch_size: 16,
            epochs: 2,
            lr: 0.001,
            save_dir: String::new(),
            ..Default::default()
        };

        train_word(&cfg, &ds, None, Some(&|s: &WordEpochStats| {
            let pos_str = (0..4).map(|p| format!(
                "P{}={:.4}({:.0}%)", p + 1,
                s.position_losses[p], s.position_accs[p] * 100.0
            )).collect::<Vec<_>>().join("  ");
            eprintln!(
                "epoch {:2}  {}  recon={:.4}  guide={:.4}  total={:.4}  [{:?}]",
                s.epoch + 1, pos_str, s.recon_loss, s.guide_loss,
                s.total_loss, s.duration,
            );
        }))
        .expect("train_word should succeed");
    }
}
