//! Training loop for SubScan (Step 2) — REINFORCE with frozen letter oracle.
//!
//! SubScan's job: given a noisy starting position near a letter center,
//! use two blurred glimpses to triangulate the actual letter center.
//! The midpoint of the glimpse positions is the center estimate.
//!
//! ## Training mechanism
//!
//! A retry loop per letter IS the training:
//! - SubScan forward → sample position → oracle reads → reward/penalty
//! - Negative reinforcement on failed reads: backward + step changes weights,
//!   so the next forward pass produces a different output
//! - Positive reinforcement on success: speed bonus + correct letter nudge
//! - Each failed attempt physically shifts the model via gradient step
//!
//! ## REINFORCE gradient
//!
//! SubScan outputs a deterministic mean position μ. A small Gaussian action
//! noise ε is added for the REINFORCE gradient computation:
//!   action = μ + ε,  ε ~ N(0, σ²)
//!   loss = reward × |action - μ|² / (2σ²)
//!
//! This gives gradient: -reward × ε/σ² × ∂μ/∂θ
//! - Negative reward → push μ away from the failed action
//! - Positive reward → pull μ toward the successful action
//!
//! The action noise is NOT exploration noise on the input — it is the
//! mathematical mechanism that gives REINFORCE a gradient direction.
//! Between retries, weight updates shift μ, which is the real exploration.

use std::fs;
use std::io::Write;
use std::time::Instant;

use flodl::autograd::{no_grad, Variable};
use flodl::nn::{clip_grad_norm, Adam, Module, Optimizer, Parameter};
use flodl::monitor::Monitor;
use flodl::tensor::{cuda_available, Device, DType, Result, Tensor, TensorError};
use flodl::CpuWorker;
use serde::{Serialize, Deserialize};

use super::data::{WordDataset, WordLoader};
use super::subscan::{SubScanModel, SubScanConfig};

use fbrl::letter::LetterModel;

/// Number of letter positions in a word image.
const N_POSITIONS: usize = 4;

/// Normalized x-centers for the 4 letter positions in a 128x256 word image.
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

    // --- Letter model (oracle, fully frozen) ---
    pub letter_checkpoint: String,
    pub letter_n_classes: usize,
    pub letter_n_scan: usize,
    pub letter_n_read: usize,
    pub letter_patch_size: i64,
    pub letter_scan_patch_w: i64,
    pub letter_n_scales: usize,
    pub letter_latent_dim: i64,

    // --- Region bounding ---
    pub region_half_w: f64,

    // --- Noise curriculum on starting position (simulates word model imprecision) ---
    pub noise_x_start: f64,
    pub noise_x_end: f64,
    pub noise_y_start: f64,
    pub noise_y_end: f64,
    pub noise_ramp_pct: f64,

    // --- REINFORCE ---
    /// Maximum retry attempts per letter before giving up.
    pub max_attempts: usize,
    /// Negative reward on each failed read attempt.
    pub fail_penalty: f64,
    /// Extra positive reward when the oracle reads the correct target letter.
    pub target_bonus: f64,
    /// Std dev of Gaussian action noise for REINFORCE gradient computation.
    pub action_sigma: f64,
    /// Minimum softmax confidence for the oracle read to count as a success.
    pub confidence_threshold: f64,

    // --- Training ---
    pub batch_size: usize,
    pub epochs: usize,
    pub subscan_lr: f64,
    pub max_grad_norm: f64,

    // --- Data ---
    pub word_data: String,

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

            region_half_w: 0.5,

            noise_x_start: 0.02,
            noise_x_end: 0.18,
            noise_y_start: 0.01,
            noise_y_end: 0.08,
            noise_ramp_pct: 0.6,

            max_attempts: 30,
            fail_penalty: -0.1,
            target_bonus: 0.2,
            action_sigma: 0.03,
            confidence_threshold: 0.5,

            batch_size: 32,
            epochs: 100,
            subscan_lr: 0.001,
            max_grad_norm: 5.0,

            word_data: String::new(),

            save_dir: "training".into(),
            checkpoint_interval: 25,

            monitor_port: None,
        }
    }
}

/// Per-epoch metrics.
pub struct SubScanEpochStats {
    pub epoch: usize,
    pub mean_attempts: f64,
    pub max_attempt: usize,
    pub success_rate: f64,
    pub target_acc: f64,
    pub mean_reward: f64,
    pub noise_x: f64,
    pub noise_y: f64,
    pub lr: f64,
    pub duration: std::time::Duration,
    pub eta: std::time::Duration,
}

/// Train SubScan via REINFORCE with frozen letter oracle.
///
/// The retry loop IS the training mechanism. Each failed attempt does
/// backward + step, changing weights so the next forward produces a
/// different output. Success gives positive reinforcement scaled by
/// speed and correct-letter nudge.
pub fn train_subscan(
    cfg: &SubScanTrainConfig,
    word_ds: &WordDataset,
    on_epoch: Option<&dyn Fn(&SubScanEpochStats)>,
) -> Result<()> {
    // --- Build SubScan ---
    let subscan = SubScanModel::new(&SubScanConfig {
        hidden_dim: cfg.hidden_dim,
        patch_h: cfg.subscan_patch_h,
        patch_w: cfg.subscan_patch_w,
        n_scales: cfg.subscan_n_scales,
        n_glimpses: cfg.subscan_n_glimpses,
        blur_sigma: cfg.subscan_blur_sigma,
    })?;

    // --- Build letter model (oracle — frozen, eval mode) ---
    let letter = LetterModel::new(
        cfg.letter_n_classes, cfg.letter_n_scan, cfg.letter_n_read,
        cfg.letter_patch_size, cfg.letter_scan_patch_w,
        cfg.letter_n_scales, cfg.letter_latent_dim,
    )?;
    if !cfg.letter_checkpoint.is_empty() {
        let report = letter.graph.load_checkpoint(&cfg.letter_checkpoint)?;
        eprintln!("Loaded letter checkpoint: {} params, {} skipped, {} missing",
            report.loaded.len(), report.skipped.len(), report.missing.len());
    }

    // --- Move to device ---
    let device = if cuda_available() {
        eprintln!("Using CUDA");
        flodl::tensor::set_cudnn_benchmark(true);
        letter.graph.set_device(Device::CUDA(0));
        subscan.graph.set_device(Device::CUDA(0));
        Device::CUDA(0)
    } else {
        eprintln!("Using CPU");
        Device::CPU
    };

    // Freeze letter model.
    for p in &letter.parameters() {
        p.freeze()?;
    }
    letter.eval();
    eprintln!("Letter model: {} params (frozen)", letter.parameters().len());

    // --- SubScan optimizer (fixed LR) ---
    let subscan_params = subscan.graph.parameters();
    eprintln!("SubScan: {} params (trainable)", subscan_params.len());

    let mut optimizer = Adam::new(&subscan_params, cfg.subscan_lr);
    let all_trainable: Vec<Parameter> = subscan_params.clone();

    subscan.graph.train();

    // --- Data loader ---
    let mut loader = WordLoader::new(word_ds, cfg.batch_size, true);
    loader.set_device(device);

    // --- Save directory ---
    if !cfg.save_dir.is_empty() {
        fs::create_dir_all(&cfg.save_dir)
            .map_err(|e| TensorError::new(&format!("create save dir: {e}")))?;
    }

    // --- Log file ---
    let mut log_file = if !cfg.save_dir.is_empty() {
        let path = format!("{}/training.log", cfg.save_dir);
        let mut f = fs::File::create(&path)
            .map_err(|e| TensorError::new(&format!("create log: {e}")))?;
        writeln!(f, "# fbrl-word subscan training (step 2) — REINFORCE").ok();
        writeln!(f, "# epochs={}  batch={}  lr={:.4}  max_attempts={}",
            cfg.epochs, cfg.batch_size, cfg.subscan_lr, cfg.max_attempts).ok();
        writeln!(f, "# noise_x={:.3}→{:.3}  noise_y={:.3}→{:.3}  ramp={:.0}%",
            cfg.noise_x_start, cfg.noise_x_end, cfg.noise_y_start, cfg.noise_y_end,
            cfg.noise_ramp_pct * 100.0).ok();
        writeln!(f, "# fail_penalty={:.2}  target_bonus={:.2}  sigma={:.3}  threshold={:.2}",
            cfg.fail_penalty, cfg.target_bonus, cfg.action_sigma, cfg.confidence_threshold).ok();
        Some(f)
    } else {
        None
    };

    // --- Monitor ---
    let mut monitor = if let Some(port) = cfg.monitor_port {
        let mut m = Monitor::new(cfg.epochs);
        if let Err(e) = m.serve(port) {
            eprintln!("warning: monitor server: {e}");
        }
        m.watch(&subscan.graph);
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
        "mean_attempts", "success_rate", "target_acc", "mean_reward",
        "noise_x", "noise_y", "lr",
    ];

    // --- RNG ---
    let mut rng_state: u64 = 0xCAFE_BABE;
    #[inline]
    fn rng_next(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *state
    }
    /// Uniform in [-1, 1].
    #[inline]
    fn rng_uniform(state: &mut u64) -> f64 {
        let v = rng_next(state);
        ((v >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    /// Uniform in (0, 1] — needed for Box-Muller (must avoid 0 for log).
    #[inline]
    fn rng_open01(state: &mut u64) -> f64 {
        let v = rng_next(state);
        ((v >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
    }
    /// Standard normal via Box-Muller.
    #[inline]
    fn rng_normal(state: &mut u64) -> f64 {
        let u1 = rng_open01(state);
        let u2 = rng_open01(state);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn shuffle(slice: &mut [usize], state: &mut u64) {
        for i in (1..slice.len()).rev() {
            let j = (rng_next(state) >> 33) as usize % (i + 1);
            slice.swap(i, j);
        }
    }

    let inv_two_sigma_sq = 1.0 / (2.0 * cfg.action_sigma * cfg.action_sigma);
    let train_start = Instant::now();
    let mut epoch_times: Vec<f64> = Vec::with_capacity(cfg.epochs);

    for epoch in 0..cfg.epochs {
        loader.reset();
        let epoch_start = Instant::now();

        // Noise curriculum.
        let noise_ramp_epochs = (cfg.noise_ramp_pct * cfg.epochs as f64).max(1.0);
        let noise_progress = (epoch as f64 / noise_ramp_epochs).min(1.0);
        let noise_x = cfg.noise_x_start + (cfg.noise_x_end - cfg.noise_x_start) * noise_progress;
        let noise_y = cfg.noise_y_start + (cfg.noise_y_end - cfg.noise_y_start) * noise_progress;

        // Epoch accumulators.
        let mut total_attempts = 0usize;
        let mut total_letters = 0usize;
        let mut total_successes = 0usize;
        let mut total_target_correct = 0usize;
        let mut max_attempt_epoch = 0usize;
        let mut reward_sum = 0.0f64;
        let mut reward_count = 0usize;

        while let Some(batch) = loader.next_batch()? {
            let img_var = Variable::new(batch.image, false);
            let b = img_var.shape()[0] as usize;

            // Case label for letter model (all lowercase).
            let case_data = vec![1.0f32; b];
            let case_var = Variable::new(
                Tensor::from_f32(&case_data, &[b as i64, 1], device)?, false,
            );

            // Region half-width.
            let half_w_data = vec![cfg.region_half_w as f32; b];
            let region_half_w = Variable::new(
                Tensor::from_f32(&half_w_data, &[b as i64, 1], device)?, false,
            );

            // Shuffle letter order within word.
            let mut letter_order = [0usize, 1, 2, 3];
            shuffle(&mut letter_order, &mut rng_state);

            for &pos in &letter_order {
                let gt_x = LETTER_CENTERS[pos];

                // Curriculum-noised starting position (fixed for this letter's retry loop).
                let start_data: Vec<f32> = (0..b).flat_map(|_| {
                    let x = (gt_x + rng_uniform(&mut rng_state) * noise_x) as f32;
                    let y = (rng_uniform(&mut rng_state) * noise_y) as f32;
                    [x, y]
                }).collect();
                let start_pos = Variable::new(
                    Tensor::from_f32(&start_data, &[b as i64, 2], device)?, false,
                );

                // Per-sample tracking: which samples have succeeded.
                let mut succeeded = vec![false; b];
                let mut attempt_counts = vec![0usize; b];

                // ── Retry loop ──────────────────────────────────────
                for attempt in 0..cfg.max_attempts {
                    if succeeded.iter().all(|&s| s) { break; }

                    // SubScan forward → mean position μ [B, 2].
                    let mu = subscan.forward(&img_var, &start_pos, &region_half_w)?;

                    // Action noise for REINFORCE gradient: ε ~ N(0, σ²).
                    let noise_data: Vec<f32> = (0..b * 2).map(|_| {
                        (rng_normal(&mut rng_state) * cfg.action_sigma) as f32
                    }).collect();
                    let noise_tensor = Tensor::from_f32(&noise_data, &[b as i64, 2], device)?;

                    // Sampled action: a = μ_detached + ε (fixed sample, no grad).
                    let action_tensor = mu.data().add(&noise_tensor)?;
                    let action = Variable::new(action_tensor, false);

                    // Oracle reads from sampled action position.
                    let (conf_vals, target_matches) = no_grad(|| {
                        let result = letter.forward(&img_var, &case_var, &action)?;
                        let logits = result.letter_logits.data();

                        // Softmax confidence: max(softmax(logits)).
                        let probs = logits.softmax(1)?;
                        let max_probs = probs.max_dim(1, false)?; // [B] float

                        // Prediction matches target letter? (as float to avoid int64 item())
                        let preds = logits.argmax(1, false)?; // [B] int64
                        let is_target = preds.eq_tensor(&batch.letter_idx[pos])?
                            .to_dtype(DType::Float32)?; // [B] float 0/1

                        // Pull to CPU for per-sample processing.
                        let max_probs_cpu = max_probs.to_device(Device::CPU)?;
                        let is_target_cpu = is_target.to_device(Device::CPU)?;

                        let mut confs = Vec::with_capacity(b);
                        let mut matches = Vec::with_capacity(b);
                        for i in 0..b {
                            confs.push(max_probs_cpu.select(0, i as i64)?.item()?);
                            matches.push(is_target_cpu.select(0, i as i64)?.item()? > 0.5);
                        }
                        Ok((confs, matches))
                    })?;

                    letter.graph.detach_state();

                    // Per-sample reward.
                    let mut rewards = vec![0.0f32; b];
                    let mut any_active = false;

                    for i in 0..b {
                        if succeeded[i] { continue; }
                        any_active = true;
                        attempt_counts[i] = attempt + 1;

                        if conf_vals[i] >= cfg.confidence_threshold {
                            // ── Success: oracle read something ──
                            succeeded[i] = true;
                            let speed = (cfg.max_attempts - attempt) as f64
                                / cfg.max_attempts as f64;
                            let mut r = speed;
                            if target_matches[i] {
                                r += cfg.target_bonus;
                                total_target_correct += 1;
                            }
                            rewards[i] = r as f32;
                            total_successes += 1;
                            reward_sum += r;
                        } else {
                            // ── Failure: negative reinforcement ──
                            rewards[i] = cfg.fail_penalty as f32;
                            reward_sum += cfg.fail_penalty;
                        }
                        reward_count += 1;
                    }

                    if !any_active { break; }

                    // REINFORCE loss: reward × |action - μ|² / (2σ²).
                    // action is fixed (no grad), μ is in the graph.
                    // Gradient: -reward × (action - μ) / σ² × ∂μ/∂θ.
                    //   reward > 0 → pull μ toward action (reinforce)
                    //   reward < 0 → push μ away from action (penalize)
                    let diff = action.sub(&mu)?;
                    let sq_dist = diff.pow_scalar(2.0)?.sum_dim(1, false)?; // [B]

                    let reward_tensor = Variable::new(
                        Tensor::from_f32(&rewards, &[b as i64], device)?, false,
                    );

                    // loss = mean_over_batch(reward_i × sq_dist_i / (2σ²))
                    let loss = sq_dist.mul(&reward_tensor)?
                        .mul_scalar(inv_two_sigma_sq)?
                        .mean()?;

                    optimizer.zero_grad();
                    loss.backward()?;
                    clip_grad_norm(&all_trainable, cfg.max_grad_norm)?;
                    optimizer.step()?;

                    subscan.graph.detach_state();
                }

                // Max attempts exhausted: count remaining failures.
                for i in 0..b {
                    if !succeeded[i] {
                        attempt_counts[i] = cfg.max_attempts;
                    }
                }

                total_letters += b;
                for &ac in &attempt_counts {
                    total_attempts += ac;
                    if ac > max_attempt_epoch {
                        max_attempt_epoch = ac;
                    }
                }
            }
        }

        if total_letters == 0 {
            return Err(TensorError::new(&format!("epoch {epoch}: no data")));
        }

        // Epoch metrics.
        let mean_attempts = total_attempts as f64 / total_letters as f64;
        let success_rate = total_successes as f64 / total_letters as f64;
        let target_acc = if total_successes > 0 {
            total_target_correct as f64 / total_successes as f64
        } else {
            0.0
        };
        let mean_reward = if reward_count > 0 {
            reward_sum / reward_count as f64
        } else {
            0.0
        };

        subscan.graph.record_scalar("mean_attempts", mean_attempts);
        subscan.graph.record_scalar("success_rate", success_rate);
        subscan.graph.record_scalar("target_acc", target_acc);
        subscan.graph.record_scalar("mean_reward", mean_reward);
        subscan.graph.record_scalar("noise_x", noise_x);
        subscan.graph.record_scalar("noise_y", noise_y);
        subscan.graph.record_scalar("lr", cfg.subscan_lr);

        subscan.graph.flush(&[]);

        let epoch_dur = epoch_start.elapsed();
        epoch_times.push((epoch_dur.as_secs_f64() * 100.0).round() / 100.0);
        let eta_secs = subscan.graph.eta(cfg.epochs);
        let eta = std::time::Duration::from_secs_f64(eta_secs);

        let stats = SubScanEpochStats {
            epoch,
            mean_attempts,
            max_attempt: max_attempt_epoch,
            success_rate,
            target_acc,
            mean_reward,
            noise_x,
            noise_y,
            lr: cfg.subscan_lr,
            duration: epoch_dur,
            eta,
        };

        if let Some(cb) = on_epoch {
            cb(&stats);
        }

        if let Some(ref mut m) = monitor {
            m.log(epoch, epoch_dur, &subscan.graph);
        }

        if let Some(ref mut f) = log_file {
            writeln!(f,
                "epoch {:3}  attempts={:.1}(max {})  success={:.1}%  target={:.1}%  \
                 reward={:.3}  noise=({:.3},{:.3})  lr={:.6}  [{:?}]  ETA {:?}",
                epoch + 1,
                stats.mean_attempts, stats.max_attempt,
                stats.success_rate * 100.0,
                stats.target_acc * 100.0,
                stats.mean_reward,
                stats.noise_x, stats.noise_y,
                stats.lr,
                stats.duration, stats.eta,
            ).ok();
            f.flush().ok();
        }

        // Async checkpoint.
        if !cfg.save_dir.is_empty() && cfg.checkpoint_interval > 0
            && (epoch + 1) % cfg.checkpoint_interval == 0
            && worker.is_idle()
        {
            let path = format!("{}/checkpoint_epoch_{}.fdl.gz", cfg.save_dir, epoch + 1);
            let snap = subscan.graph.snapshot_cpu()?;
            worker.submit(move || {
                if let Err(e) = snap.save_file(&path) {
                    eprintln!("warning: async checkpoint: {e}");
                }
            });
        }
    }

    // --- Finalize ---
    if let Some(ref mut m) = monitor {
        m.finish_with(&subscan.graph);
    }

    if !cfg.save_dir.is_empty() {
        let subscan_path = format!("{}/subscan_final.fdl.gz", cfg.save_dir);
        let snap = subscan.graph.snapshot_cpu()?;
        worker.submit(move || {
            if let Err(e) = snap.save_file(&subscan_path) {
                eprintln!("warning: final subscan checkpoint: {e}");
            }
        });

        if let Err(e) = subscan.graph.plot_html(
            &format!("{}/training.html", cfg.save_dir), metric_tags,
        ) {
            eprintln!("warning: plot HTML: {e}");
        }
        if let Err(e) = subscan.graph.export_trends(
            &format!("{}/training.csv", cfg.save_dir), metric_tags,
        ) {
            eprintln!("warning: export CSV: {e}");
        }

        let manifest = serde_json::json!({
            "framework": "flodl",
            "model": "subscan",
            "step": 2,
            "training": "REINFORCE",
            "config": cfg,
            "results": {
                "mean_attempts": subscan.graph.trend("mean_attempts").latest(),
                "success_rate": subscan.graph.trend("success_rate").latest(),
                "target_acc": subscan.graph.trend("target_acc").latest(),
                "mean_reward": subscan.graph.trend("mean_reward").latest(),
            },
            "files": { "subscan_model": "subscan_final.fdl.gz" },
            "parent": cfg.letter_checkpoint,
        });
        if let Err(e) = fs::write(
            format!("{}/manifest.json", cfg.save_dir),
            serde_json::to_string_pretty(&manifest).unwrap_or_default(),
        ) {
            eprintln!("warning: write manifest: {e}");
        }

        let total_time = train_start.elapsed().as_secs_f64();
        let rss_mb = (flodl::tensor::rss_kb() as f64 / 1024.0).round() as i64;
        let mut bench = serde_json::json!({
            "framework": "flodl",
            "model": "subscan",
            "config": {
                "hidden_dim": cfg.hidden_dim,
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
                "max_attempts": cfg.max_attempts,
            },
            "ram_peak_rss_mb": rss_mb,
            "total_time_s": (total_time * 10.0).round() / 10.0,
            "avg_epoch_s": ((total_time / cfg.epochs as f64) * 10.0).round() / 10.0,
            "epoch_times_s": epoch_times,
        });
        if device != Device::CPU
            && let Ok((used, total)) = flodl::tensor::cuda_memory_info() {
                let gpu = flodl::tensor::cuda_device_name().unwrap_or_default();
                bench["gpu"] = serde_json::json!(gpu);
                bench["vram"] = serde_json::json!({
                    "device_used_mb": (used as f64 / 1024.0 / 1024.0).round() as i64,
                    "device_total_mb": (total as f64 / 1024.0 / 1024.0).round() as i64,
                });
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
