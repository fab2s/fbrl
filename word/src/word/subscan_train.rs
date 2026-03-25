//! Training loop for SubScan + Letter composition (Step 2).
//!
//! Trains SubScan to localize letters within bounded regions of word images,
//! using a frozen letter read module for classification.
//!
//! Per batch, for each of 4 letter positions:
//! 1. Compute a noisy region around the ground-truth letter center
//! 2. SubScan localizes within the region → position
//! 3. LetterModel.scan refines from that position → LetterModel.read classifies
//! 4. CE + reconstruction loss per position
//! Then: SubScan diversity loss across the 4 positions, backward, step.

use std::fs;
use std::io::Write;
use std::time::Instant;

use flodl::autograd::Variable;
use flodl::nn::{
    cross_entropy_loss, mse_loss, clip_grad_norm,
    Adam, CosineScheduler, Optimizer, Parameter,
};
use flodl::monitor::Monitor;
use flodl::tensor::{cuda_available, Device, Result, Tensor, TensorError, TensorOptions};
use flodl::CpuWorker;
use serde::{Serialize, Deserialize};

use super::data::{WordDataset, WordLoader, IsolationDataset};
use super::loss::fixation_diversity_loss;
use super::subscan::{SubScan, SubScanConfig};

use fbrl::letter::LetterModel;

/// Number of letter positions in a word image.
const N_POSITIONS: usize = 4;

/// Normalized x-centers for the 4 letter positions in a 128×256 word image.
/// Letters are evenly spaced: centers at pixels 32, 96, 160, 224.
const LETTER_CENTERS: [f64; N_POSITIONS] = [-0.75, -0.25, 0.25, 0.75];

/// SubScan training configuration.
#[derive(Serialize, Deserialize)]
pub struct SubScanTrainConfig {
    // --- SubScan architecture ---
    pub hidden_dim: i64,
    pub subscan_patch_h: i64,
    pub subscan_patch_w: i64,
    pub subscan_n_scales: usize,
    pub subscan_n_glimpses: usize,
    pub subscan_blur_sigma: f64,

    // --- Letter model (loaded from checkpoint) ---
    pub letter_checkpoint: String,
    pub letter_n_classes: usize,
    pub letter_n_scan: usize,
    pub letter_n_read: usize,
    pub letter_patch_size: i64,
    pub letter_scan_patch_w: i64,
    pub letter_n_scales: usize,
    pub letter_latent_dim: i64,

    // --- Region noise curriculum ---
    /// Starting noise magnitude (fraction of letter width in normalized coords).
    pub noise_start: f64,
    /// Final noise magnitude.
    pub noise_end: f64,
    /// Epochs over which noise ramps from start to end.
    pub noise_ramp_epochs: usize,
    /// Region half-width in normalized coords (~1 letter width = 0.5).
    pub region_half_w: f64,

    // --- Training ---
    pub batch_size: usize,
    pub epochs: usize,
    pub subscan_lr: f64,
    pub scan_lr: f64,
    pub min_lr_ratio: f64,
    pub max_grad_norm: f64,

    // --- Loss weights ---
    pub recon_weight: f64,
    pub diversity_weight: f64,
    pub diversity_sigma: f64,

    // --- Data paths ---
    pub word_data: String,
    pub isolation_data: String,

    // --- Checkpointing ---
    #[serde(default)]
    pub save_dir: String,
    #[serde(default = "default_checkpoint_interval")]
    pub checkpoint_interval: usize,

    // --- Live monitoring ---
    #[serde(skip)]
    pub monitor_port: Option<u16>,
}

fn default_checkpoint_interval() -> usize { 25 }

impl Default for SubScanTrainConfig {
    fn default() -> Self {
        SubScanTrainConfig {
            hidden_dim: 256,
            subscan_patch_h: 8,
            subscan_patch_w: 28,
            subscan_n_scales: 1,
            subscan_n_glimpses: 2,
            subscan_blur_sigma: 4.0,

            letter_checkpoint: String::new(),
            letter_n_classes: 26,
            letter_n_scan: 1,
            letter_n_read: 6,
            letter_patch_size: 12,
            letter_scan_patch_w: 18,
            letter_n_scales: 1,
            letter_latent_dim: 256,

            noise_start: 0.08,
            noise_end: 0.20,
            noise_ramp_epochs: 50,
            region_half_w: 0.5,

            batch_size: 32,
            epochs: 100,
            subscan_lr: 0.001,
            scan_lr: 0.0001,
            min_lr_ratio: 0.01,
            max_grad_norm: 5.0,

            recon_weight: 1.0,
            diversity_weight: 1.0,
            diversity_sigma: 0.1,

            word_data: String::new(),
            isolation_data: String::new(),

            save_dir: "training".into(),
            checkpoint_interval: 25,

            monitor_port: None,
        }
    }
}

/// Per-epoch averaged metrics.
pub struct SubScanEpochStats {
    pub epoch: usize,
    pub ce_loss: f64,
    pub recon_loss: f64,
    pub div_loss: f64,
    pub total_loss: f64,
    pub accuracy: f64,
    pub lr_subscan: f64,
    pub lr_scan: f64,
    pub noise_range: f64,
    pub duration: std::time::Duration,
    pub eta: std::time::Duration,
}

/// Run SubScan + Letter composition training (Step 2).
pub fn train_subscan(
    cfg: &SubScanTrainConfig,
    word_ds: &WordDataset,
    iso_ds: &IsolationDataset,
    on_epoch: Option<&dyn Fn(&SubScanEpochStats)>,
) -> Result<()> {
    // --- Build SubScan ---
    let subscan = SubScan::new(&SubScanConfig {
        hidden_dim: cfg.hidden_dim,
        patch_h: cfg.subscan_patch_h,
        patch_w: cfg.subscan_patch_w,
        n_scales: cfg.subscan_n_scales,
        n_glimpses: cfg.subscan_n_glimpses,
        blur_sigma: cfg.subscan_blur_sigma,
    })?;

    // --- Build and load LetterModel ---
    let letter = LetterModel::new(
        cfg.letter_n_classes, cfg.letter_n_scan, cfg.letter_n_read,
        cfg.letter_patch_size, cfg.letter_scan_patch_w,
        cfg.letter_n_scales, cfg.letter_latent_dim,
    )?;
    if !cfg.letter_checkpoint.is_empty() {
        let report = letter.graph.load_checkpoint(&cfg.letter_checkpoint)?;
        eprintln!("Loaded letter checkpoint: {} params, {} skipped, {} missing",
            report.loaded.len(), report.skipped.len(), report.missing.len());
        if !report.missing.is_empty() {
            eprintln!("  missing: {:?}", &report.missing[..report.missing.len().min(5)]);
        }
    }

    // --- Move to device ---
    let device = if cuda_available() {
        eprintln!("Using CUDA");
        flodl::tensor::set_cudnn_benchmark(true);
        letter.graph.set_device(Device::CUDA(0));
        subscan.set_device(Device::CUDA(0));
        Device::CUDA(0)
    } else {
        eprintln!("Using CPU");
        Device::CPU
    };

    // --- Freeze read-phase params, keep scan-phase trainable ---
    let scan_params: Vec<Parameter> = letter.scan_phase_params().to_vec();
    let scan_param_count = scan_params.len();
    let all_letter_params = letter.parameters();

    // Freeze everything in the letter model.
    for p in &all_letter_params {
        p.freeze()?;
    }
    // Unfreeze scan-phase params.
    for p in &scan_params {
        p.unfreeze()?;
    }
    let frozen_count = all_letter_params.len() - scan_param_count;
    eprintln!("Letter model: {} scan params (trainable), {} read params (frozen)",
        scan_param_count, frozen_count);

    // --- Optimizer: two groups ---
    let subscan_params = subscan.parameters();
    eprintln!("SubScan: {} params", subscan_params.len());

    let mut optimizer = Adam::with_groups()
        .group(&subscan_params, cfg.subscan_lr)
        .group(&scan_params, cfg.scan_lr)
        .build();
    let subscan_scheduler = CosineScheduler::new(
        cfg.subscan_lr, cfg.subscan_lr * cfg.min_lr_ratio, cfg.epochs,
    );
    let scan_scheduler = CosineScheduler::new(
        cfg.scan_lr, cfg.scan_lr * cfg.min_lr_ratio, cfg.epochs,
    );

    // All trainable params for grad clipping.
    let mut all_trainable: Vec<Parameter> = subscan_params.clone();
    all_trainable.extend(scan_params.iter().cloned());

    letter.train();
    subscan.set_training(true);

    // --- Data loader ---
    let mut loader = WordLoader::new(word_ds, cfg.batch_size, true);
    loader.set_device(device);

    // --- Save directory ---
    if !cfg.save_dir.is_empty() {
        fs::create_dir_all(&cfg.save_dir)
            .map_err(|e| TensorError::new(&format!("create save dir: {e}")))?;
    }

    // --- Streaming log ---
    let mut log_file = if !cfg.save_dir.is_empty() {
        let path = format!("{}/training.log", cfg.save_dir);
        let mut f = fs::File::create(&path)
            .map_err(|e| TensorError::new(&format!("create log: {e}")))?;
        writeln!(f, "# fbrl-word subscan training (step 2)").ok();
        writeln!(f, "# epochs={}  batch={}  subscan_lr={:.4}  scan_lr={:.4}",
            cfg.epochs, cfg.batch_size, cfg.subscan_lr, cfg.scan_lr).ok();
        Some(f)
    } else {
        None
    };

    // --- Live monitor (optional) ---
    let mut monitor = if let Some(port) = cfg.monitor_port {
        let mut m = Monitor::new(cfg.epochs);
        if let Err(e) = m.serve(port) {
            eprintln!("warning: monitor server: {e}");
        }
        m.watch(&letter.graph);
        m.set_metadata(serde_json::to_value(cfg).unwrap_or_default());
        if !cfg.save_dir.is_empty() {
            m.save_html(&format!("{}/dashboard.html", cfg.save_dir));
        }
        Some(m)
    } else {
        None
    };

    // --- Background worker for async checkpointing ---
    let mut worker = CpuWorker::new();

    let metric_tags: &[&str] = &[
        "ce", "recon", "diversity", "total", "accuracy", "lr_subscan", "lr_scan", "noise",
    ];

    // --- RNG for isolation image selection ---
    let mut rng_state: u64 = 0xCAFE_BABE;
    let mut rng = |bound: usize| -> usize {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng_state >> 33) as usize % bound
    };

    let opts = TensorOptions { device, ..Default::default() };
    let train_start = Instant::now();
    let mut epoch_times: Vec<f64> = Vec::with_capacity(cfg.epochs);

    for epoch in 0..cfg.epochs {
        loader.reset();
        let epoch_start = Instant::now();
        let mut n_batches = 0usize;

        // Noise curriculum.
        let noise_progress = if cfg.noise_ramp_epochs > 0 {
            (epoch as f64 / cfg.noise_ramp_epochs as f64).min(1.0)
        } else {
            1.0
        };
        let noise_range = cfg.noise_start + (cfg.noise_end - cfg.noise_start) * noise_progress;

        // Update LR.
        let current_subscan_lr = subscan_scheduler.lr(epoch);
        let current_scan_lr = scan_scheduler.lr(epoch);
        optimizer.set_group_lr(0, current_subscan_lr);
        optimizer.set_group_lr(1, current_scan_lr);

        while let Some(batch) = loader.next_batch()? {
            let img_var = Variable::new(batch.image, false);
            let b = img_var.shape()[0];

            // Pre-extract letter indices for isolation lookup.
            let letter_indices: Vec<Vec<i64>> = (0..N_POSITIONS).map(|pos| {
                batch.letter_idx[pos].to_i64_vec().unwrap_or_default()
            }).collect();

            // Accumulate losses across the 4 letter positions.
            let zero = Variable::new(Tensor::zeros(&[1], opts)?, false);
            let mut total_ce = zero.clone();
            let mut total_recon = zero.clone();
            let mut subscan_positions: Vec<Variable> = Vec::with_capacity(N_POSITIONS);
            let mut total_acc = 0.0f64;

            for pos in 0..N_POSITIONS {
                // Region bounds: ground-truth center + uniform noise.
                let gt_center = LETTER_CENTERS[pos];
                let noise_data: Vec<f32> = (0..b as usize).map(|_| {
                    let u = rng(10000) as f64 / 10000.0; // [0, 1)
                    let offset = (u * 2.0 - 1.0) * noise_range;
                    (gt_center + offset) as f32
                }).collect();
                let region_center = Variable::new(
                    Tensor::from_f32(&noise_data, &[b, 1], device)?, false,
                );
                let half_w_data = vec![cfg.region_half_w as f32; b as usize];
                let region_half_w = Variable::new(
                    Tensor::from_f32(&half_w_data, &[b, 1], device)?, false,
                );

                // SubScan: localize within region.
                let subscan_pos = subscan.forward(&img_var, &region_center, &region_half_w)?;
                subscan_positions.push(subscan_pos.clone());

                // LetterModel: scan from SubScan's position, read classifies.
                // Case label: assume lowercase (1.0) for word letters.
                let case_data = vec![1.0f32; b as usize];
                let case_var = Variable::new(
                    Tensor::from_f32(&case_data, &[b, 1], device)?, false,
                );
                letter.set_scan_start(subscan_pos);
                let result = letter.forward(&img_var, &case_var)?;

                // Per-position CE loss.
                let target = Variable::new(batch.letter_idx[pos].clone(), false);
                let ce = cross_entropy_loss(&result.letter_logits, &target)?;
                total_ce = total_ce.add(&ce)?;

                // Per-position reconstruction loss.
                // Target: matching letter from isolation dataset (128×128).
                if cfg.recon_weight > 0.0 {
                    let recon_targets: Vec<Tensor> = (0..b as usize).map(|bi| {
                        let lidx = letter_indices[pos][bi];
                        iso_ds.get_random_image(lidx, &mut rng).clone()
                    }).collect();
                    let recon_target_refs: Vec<&Tensor> = recon_targets.iter().collect();
                    let recon_target_batch = Tensor::stack(&recon_target_refs, 0)?;
                    let recon_target = Variable::new(
                        recon_target_batch.to_device(device)?, false,
                    );
                    let recon = mse_loss(&result.recon, &recon_target)?;
                    total_recon = total_recon.add(&recon)?;
                }

                // Accuracy tracking (per-position mean, then averaged).
                let preds = result.letter_logits.data().argmax(1, false)?;
                let acc = preds.eq_tensor(&batch.letter_idx[pos])?.mean()?.item()?;
                total_acc += acc;

                // Detach state between positions (no gradient across positions).
                letter.graph.detach_state();
            }

            // SubScan diversity: repel the 4 output positions from each other.
            let div_loss = if cfg.diversity_weight > 0.0 {
                fixation_diversity_loss(&subscan_positions, cfg.diversity_sigma, 0.3)?
            } else {
                zero.clone()
            };

            // Total loss.
            let avg_ce = total_ce.mul_scalar(1.0 / N_POSITIONS as f64)?;
            let avg_recon = total_recon.mul_scalar(1.0 / N_POSITIONS as f64)?;
            let total = avg_ce.add(&avg_recon.mul_scalar(cfg.recon_weight)?)?
                .add(&div_loss.mul_scalar(cfg.diversity_weight)?)?;

            optimizer.zero_grad();
            total.backward()?;
            clip_grad_norm(&all_trainable, cfg.max_grad_norm)?;
            optimizer.step()?;

            // Break gradient chain for next batch.
            letter.graph.detach_state();
            subscan.reset();

            // Record metrics.
            let acc = total_acc / N_POSITIONS as f64;
            letter.graph.record_scalar("ce", avg_ce.item()?);
            letter.graph.record_scalar("recon", avg_recon.item()?);
            letter.graph.record_scalar("diversity", div_loss.item()?);
            letter.graph.record_scalar("total", total.item()?);
            letter.graph.record_scalar("accuracy", acc);
            letter.graph.record_scalar("lr_subscan", current_subscan_lr);
            letter.graph.record_scalar("lr_scan", current_scan_lr);
            letter.graph.record_scalar("noise", noise_range);

            n_batches += 1;
        }

        if n_batches == 0 {
            return Err(TensorError::new(&format!("epoch {epoch}: no batches")));
        }

        // Flush batch means → epoch history.
        letter.graph.flush(&[]);

        let epoch_dur = epoch_start.elapsed();
        epoch_times.push((epoch_dur.as_secs_f64() * 100.0).round() / 100.0);
        let eta_secs = letter.graph.eta(cfg.epochs);
        let eta = std::time::Duration::from_secs_f64(eta_secs);

        let stats = SubScanEpochStats {
            epoch,
            ce_loss: letter.graph.trend("ce").latest(),
            recon_loss: letter.graph.trend("recon").latest(),
            div_loss: letter.graph.trend("diversity").latest(),
            total_loss: letter.graph.trend("total").latest(),
            accuracy: letter.graph.trend("accuracy").latest(),
            lr_subscan: current_subscan_lr,
            lr_scan: current_scan_lr,
            noise_range,
            duration: epoch_dur,
            eta,
        };

        if let Some(cb) = on_epoch {
            cb(&stats);
        }

        if let Some(ref mut m) = monitor {
            m.log(epoch, epoch_dur, &letter.graph);
        }

        if let Some(ref mut f) = log_file {
            writeln!(f,
                "epoch {:3}  ce={:.4}  recon={:.4}  div={:.4}  total={:.4}  acc={:.1}%  \
                 noise={:.3}  lr_sub={:.6}  lr_scan={:.6}  [{:?}]  ETA {:?}",
                epoch + 1,
                stats.ce_loss, stats.recon_loss, stats.div_loss, stats.total_loss,
                stats.accuracy * 100.0, stats.noise_range,
                stats.lr_subscan, stats.lr_scan,
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
                let snap = letter.graph.snapshot_cpu()?;
                worker.submit(move || {
                    if let Err(e) = snap.save_file(&path) {
                        eprintln!("warning: async checkpoint: {e}");
                    }
                });
                // TODO: also save SubScan params separately (not in letter graph)
            }
        }
    }

    // --- Finalize ---
    if let Some(ref mut m) = monitor {
        m.finish_with(&letter.graph);
    }

    if !cfg.save_dir.is_empty() {
        // Save final letter graph checkpoint.
        let letter_path = format!("{}/letter_composed.fdl.gz", cfg.save_dir);
        let snap = letter.graph.snapshot_cpu()?;
        worker.submit(move || {
            if let Err(e) = snap.save_file(&letter_path) {
                eprintln!("warning: final letter checkpoint: {e}");
            }
        });

        // TODO: save SubScan params as a separate checkpoint
        // (SubScan is not a Graph, so no snapshot_cpu — need manual tensor saving)

        if let Err(e) = letter.graph.plot_html(
            &format!("{}/training.html", cfg.save_dir), metric_tags,
        ) {
            eprintln!("warning: plot HTML: {e}");
        }
        if let Err(e) = letter.graph.export_trends(
            &format!("{}/training.csv", cfg.save_dir), metric_tags,
        ) {
            eprintln!("warning: export CSV: {e}");
        }

        // Manifest.
        let manifest = serde_json::json!({
            "framework": "flodl",
            "model": "subscan+letter",
            "step": 2,
            "config": cfg,
            "results": {
                "accuracy": letter.graph.trend("accuracy").latest(),
                "ce": letter.graph.trend("ce").latest(),
                "recon": letter.graph.trend("recon").latest(),
            },
            "files": {
                "letter_model": "letter_composed.fdl.gz",
            },
            "parent": cfg.letter_checkpoint,
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
            "model": "subscan+letter",
            "config": {
                "hidden_dim": cfg.hidden_dim,
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
        ) {
            eprintln!("warning: write benchmark: {e}");
        }

        worker.finish();
    }

    Ok(())
}
