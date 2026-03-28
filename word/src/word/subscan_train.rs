//! Training loop for SubScan (Step 2) — supervised MSE on center_x.
//!
//! SubScan learns to find letter centers in word images using triangle
//! glimpses. Trained independently — no letter model in the loop.
//!
//! Loss: MSE(predicted_center_x, gt_center_x).
//! The ink density is the input (through blurred glimpse features),
//! not the loss. SubScan learns spatial patterns → center position.

use std::fs;
use std::io::Write;
use std::time::Instant;

use flodl::autograd::Variable;
use flodl::nn::{mse_loss, clip_grad_norm, Adam, CosineScheduler, Module, Optimizer, Parameter};
use flodl::monitor::Monitor;
use flodl::tensor::{cuda_available, Device, Result, Tensor, TensorError};
use flodl::CpuWorker;
use serde::{Serialize, Deserialize};

use super::data::{WordDataset, WordLoader};
use super::subscan::{SubScanModel, SubScanConfig};

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

    // --- Triangle geometry (normalized coords) ---
    pub min_base_hw: f64,
    pub max_base_hw: f64,
    pub triangle_height: f64,

    // --- Region bounding ---
    pub region_half_w: f64,

    // --- Noise curriculum on starting position ---
    pub noise_x_start: f64,
    pub noise_x_end: f64,
    pub noise_y_start: f64,
    pub noise_y_end: f64,
    pub noise_ramp_pct: f64,

    // --- Training ---
    pub batch_size: usize,
    pub epochs: usize,
    pub subscan_lr: f64,
    pub min_lr_ratio: f64,
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
            subscan_n_glimpses: 3,
            subscan_blur_sigma: 4.0,

            min_base_hw: 0.03,
            max_base_hw: 0.20,
            triangle_height: 0.3,

            region_half_w: 0.5,

            noise_x_start: 0.02,
            noise_x_end: 0.18,
            noise_y_start: 0.01,
            noise_y_end: 0.08,
            noise_ramp_pct: 0.6,

            batch_size: 32,
            epochs: 100,
            subscan_lr: 0.001,
            min_lr_ratio: 0.01,
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
    pub mse: f64,
    pub mean_err_x: f64,
    pub lr: f64,
    pub noise_x: f64,
    pub noise_y: f64,
    pub duration: std::time::Duration,
    pub eta: std::time::Duration,
}

/// Train SubScan with supervised MSE on center_x.
///
/// No letter model in the loop. SubScan learns from ink structure
/// (through triangle glimpse features) to predict letter centers.
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
        min_base_hw: cfg.min_base_hw,
        max_base_hw: cfg.max_base_hw,
        triangle_height: cfg.triangle_height,
    })?;

    // --- Move to device ---
    let device = if cuda_available() {
        eprintln!("Using CUDA");
        flodl::tensor::set_cudnn_benchmark(true);
        subscan.graph.set_device(Device::CUDA(0));
        Device::CUDA(0)
    } else {
        eprintln!("Using CPU");
        Device::CPU
    };

    // --- Optimizer ---
    let subscan_params = subscan.graph.parameters();
    eprintln!("SubScan: {} params (trainable)", subscan_params.len());

    let mut optimizer = Adam::new(&subscan_params, cfg.subscan_lr);
    let scheduler = CosineScheduler::new(
        cfg.subscan_lr, cfg.subscan_lr * cfg.min_lr_ratio, cfg.epochs,
    );
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
        writeln!(f, "# fbrl-word subscan training (step 2) — triangle MSE").ok();
        writeln!(f, "# epochs={}  batch={}  lr={:.4}  glimpses={}",
            cfg.epochs, cfg.batch_size, cfg.subscan_lr, cfg.subscan_n_glimpses).ok();
        writeln!(f, "# triangle: min_base_hw={:.3}  max_base_hw={:.3}  height={:.3}",
            cfg.min_base_hw, cfg.max_base_hw, cfg.triangle_height).ok();
        writeln!(f, "# noise_x={:.3}→{:.3}  noise_y={:.3}→{:.3}  ramp={:.0}%",
            cfg.noise_x_start, cfg.noise_x_end, cfg.noise_y_start, cfg.noise_y_end,
            cfg.noise_ramp_pct * 100.0).ok();
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
    let metric_tags: &[&str] = &["mse", "err_x", "lr", "noise_x", "noise_y"];

    // --- RNG ---
    let mut rng_state: u64 = 0xCAFE_BABE;
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
    fn shuffle(slice: &mut [usize], state: &mut u64) {
        for i in (1..slice.len()).rev() {
            let j = (rng_next(state) >> 33) as usize % (i + 1);
            slice.swap(i, j);
        }
    }

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

        let current_lr = scheduler.lr(epoch);
        optimizer.set_lr(current_lr);

        while let Some(batch) = loader.next_batch()? {
            let img_var = Variable::new(batch.image, false);
            let b = img_var.shape()[0] as usize;

            // Region half-width.
            let half_w_data = vec![cfg.region_half_w as f32; b];
            let region_half_w = Variable::new(
                Tensor::from_f32(&half_w_data, &[b as i64, 1], device)?, false,
            );

            // Shuffle letter order.
            let mut letter_order = [0usize, 1, 2, 3];
            shuffle(&mut letter_order, &mut rng_state);

            for &pos in &letter_order {
                let gt_x = LETTER_CENTERS[pos];

                // Noisy starting position.
                let start_data: Vec<f32> = (0..b).flat_map(|_| {
                    let x = (gt_x + rng_uniform(&mut rng_state) * noise_x) as f32;
                    let y = (rng_uniform(&mut rng_state) * noise_y) as f32;
                    [x, y]
                }).collect();
                let start_pos = Variable::new(
                    Tensor::from_f32(&start_data, &[b as i64, 2], device)?, false,
                );

                // GT target: center_x [B, 1].
                let gt_data: Vec<f32> = vec![gt_x as f32; b];
                let gt_target = Variable::new(
                    Tensor::from_f32(&gt_data, &[b as i64, 1], device)?, false,
                );

                // SubScan forward → center_x [B, 1].
                let pred_cx = subscan.forward(&img_var, &start_pos, &region_half_w)?;

                // MSE loss on center_x.
                let loss = mse_loss(&pred_cx, &gt_target)?;

                optimizer.zero_grad();
                loss.backward()?;
                clip_grad_norm(&all_trainable, cfg.max_grad_norm)?;
                optimizer.step()?;

                // Position error for metrics.
                let err_x: f64 = pred_cx.detach().sub(&gt_target)?.abs()?.mean()?.item()?;

                let mse_val: f64 = loss.item()?;
                subscan.graph.record_scalar("mse", mse_val);
                subscan.graph.record_scalar("err_x", err_x);
                subscan.graph.record_scalar("lr", current_lr);
                subscan.graph.record_scalar("noise_x", noise_x);
                subscan.graph.record_scalar("noise_y", noise_y);

                subscan.graph.detach_state();
            }
        }

        subscan.graph.flush(&[]);

        let epoch_dur = epoch_start.elapsed();
        epoch_times.push((epoch_dur.as_secs_f64() * 100.0).round() / 100.0);
        let eta_secs = subscan.graph.eta(cfg.epochs);
        let eta = std::time::Duration::from_secs_f64(eta_secs);

        let stats = SubScanEpochStats {
            epoch,
            mse: subscan.graph.trend("mse").latest(),
            mean_err_x: subscan.graph.trend("err_x").latest(),
            lr: current_lr,
            noise_x,
            noise_y,
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
                "epoch {:3}  mse={:.6}  err_x={:.4}  noise=({:.3},{:.3})  lr={:.6}  [{:?}]  ETA {:?}",
                epoch + 1,
                stats.mse, stats.mean_err_x,
                stats.noise_x, stats.noise_y,
                stats.lr, stats.duration, stats.eta,
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
            "model": "subscan-triangle",
            "step": 2,
            "training": "supervised MSE",
            "config": cfg,
            "results": {
                "mse": subscan.graph.trend("mse").latest(),
                "err_x": subscan.graph.trend("err_x").latest(),
            },
            "files": { "subscan_model": "subscan_final.fdl.gz" },
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
            "model": "subscan-triangle",
            "config": {
                "hidden_dim": cfg.hidden_dim,
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
                "n_glimpses": cfg.subscan_n_glimpses,
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
