//! Letter model adaptation to word images (Step 1b).
//!
//! Fine-tunes the letter model on word images with origin injection.
//! The model learns: given a word image and a starting position,
//! read the letter at that position, ignore everything else.
//!
//! This produces the frozen oracle for SubScan composition (Step 2).

use std::fs;
use std::io::Write;
use std::time::Instant;

use flodl::autograd::Variable;
use flodl::nn::{
    cross_entropy_loss, clip_grad_norm,
    Adam, CosineScheduler, Optimizer,
};
use flodl::monitor::Monitor;
use flodl::tensor::{cuda_available, Device, Result, Tensor, TensorError};
use flodl::CpuWorker;
use serde::{Serialize, Deserialize};

use super::data::{WordDataset, WordLoader};

use fbrl::letter::LetterModel;

/// Number of letter positions in a word image.
const N_POSITIONS: usize = 4;

/// Normalized x-centers for the 4 letter positions in a 128x256 word image.
const LETTER_CENTERS: [f64; N_POSITIONS] = [-0.75, -0.25, 0.25, 0.75];

/// Configuration for letter model word-image training.
#[derive(Serialize, Deserialize)]
pub struct LetterWordConfig {
    // --- Letter model architecture ---
    pub n_classes: usize,
    pub n_scan: usize,
    pub n_read: usize,
    pub patch_size: i64,
    pub scan_patch_w: i64,
    pub n_scales: usize,
    pub latent_dim: i64,

    // --- Pre-trained checkpoint (starting point) ---
    pub checkpoint: String,

    // --- Position noise (same range as original noise training) ---
    pub noise_x: f64,
    pub noise_y: f64,

    // --- Training ---
    pub batch_size: usize,
    pub epochs: usize,
    pub lr: f64,
    pub min_lr: f64,
    pub max_grad_norm: f64,

    // --- Data ---
    pub word_data: String,

    // --- Checkpointing ---
    #[serde(default)]
    pub save_dir: String,
    #[serde(default = "default_checkpoint_interval")]
    pub checkpoint_interval: usize,

    // --- Monitoring ---
    #[serde(skip)]
    pub monitor_port: Option<u16>,
}

fn default_checkpoint_interval() -> usize { 25 }

impl Default for LetterWordConfig {
    fn default() -> Self {
        LetterWordConfig {
            n_classes: 26,
            n_scan: 1,
            n_read: 6,
            patch_size: 12,
            scan_patch_w: 18,
            n_scales: 1,
            latent_dim: 256,

            checkpoint: String::new(),

            noise_x: 0.3,
            noise_y: 0.15,

            batch_size: 32,
            epochs: 100,
            lr: 0.0003,
            min_lr: 0.0,
            max_grad_norm: 5.0,

            word_data: String::new(),

            save_dir: "training".into(),
            checkpoint_interval: 25,

            monitor_port: None,
        }
    }
}

/// Per-epoch stats for letter word training.
pub struct LetterWordEpochStats {
    pub epoch: usize,
    pub ce_loss: f64,
    pub accuracy: f64,
    pub lr: f64,
    pub duration: std::time::Duration,
    pub eta: std::time::Duration,
}

/// Train the letter model on word images with position injection.
///
/// Each batch: pick a random letter position, inject noisy starting position
/// via origin injection, forward on the word image, CE loss on the target letter.
/// The model learns to focus on one letter in a multi-letter image.
pub fn train_letter_on_words(
    cfg: &LetterWordConfig,
    word_ds: &WordDataset,
    on_epoch: Option<&dyn Fn(&LetterWordEpochStats)>,
) -> Result<()> {
    // --- Build LetterModel ---
    let model = LetterModel::new(
        cfg.n_classes, cfg.n_scan, cfg.n_read,
        cfg.patch_size, cfg.scan_patch_w,
        cfg.n_scales, cfg.latent_dim, 128, 128,
    )?;

    // --- Load pre-trained checkpoint ---
    if !cfg.checkpoint.is_empty() {
        let report = model.graph.load_checkpoint(&cfg.checkpoint)?;
        eprintln!("Loaded checkpoint: {} params, {} skipped, {} missing",
            report.loaded.len(), report.skipped.len(), report.missing.len());
        if !report.missing.is_empty() {
            eprintln!("  missing: {:?}", &report.missing[..report.missing.len().min(5)]);
        }
    }

    // --- Move to device ---
    let device = if cuda_available() {
        eprintln!("Using CUDA");
        flodl::tensor::set_cudnn_benchmark(true);
        model.graph.set_device(Device::CUDA(0));
        Device::CUDA(0)
    } else {
        eprintln!("Using CPU");
        Device::CPU
    };

    // --- ALL params trainable ---
    model.train();
    let params = model.parameters();
    eprintln!("Letter model: {} params (all trainable)", params.len());

    let mut optimizer = Adam::new(&params, cfg.lr);
    let scheduler = CosineScheduler::new(
        cfg.lr, cfg.min_lr, cfg.epochs,
    );

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
        writeln!(f, "# letter model word-image training (step 1b)").ok();
        writeln!(f, "# epochs={}  batch={}  lr={:.4}  noise_x={:.2}  noise_y={:.2}",
            cfg.epochs, cfg.batch_size, cfg.lr, cfg.noise_x, cfg.noise_y).ok();
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

    let metric_tags: &[&str] = &["ce", "accuracy", "lr"];

    // --- RNG ---
    let mut rng_state: u64 = 0xDEAD_BEEF;
    #[inline]
    fn rng_next(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *state
    }
    /// Uniform in [-1, 1].
    #[inline]
    fn rng_f(state: &mut u64) -> f64 {
        let v = rng_next(state);
        ((v >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    let train_start = Instant::now();
    let mut epoch_times: Vec<f64> = Vec::with_capacity(cfg.epochs);

    for epoch in 0..cfg.epochs {
        loader.reset();
        let epoch_start = Instant::now();
        let mut n_batches = 0usize;

        let current_lr = scheduler.lr(epoch);
        optimizer.set_lr(current_lr);

        while let Some(batch) = loader.next_batch()? {
            let img_var = Variable::new(batch.image, false);
            let b = img_var.shape()[0] as usize;

            // Pick a random letter position.
            let pos = (rng_next(&mut rng_state) >> 33) as usize % N_POSITIONS;
            let gt_center = LETTER_CENTERS[pos];

            // Noisy starting position: ground truth + noise.
            let mut offsets = Vec::with_capacity(b * 2);
            for _ in 0..b {
                offsets.push((gt_center + rng_f(&mut rng_state) * cfg.noise_x) as f32);
                offsets.push((rng_f(&mut rng_state) * cfg.noise_y) as f32);
            }
            let scan_start = Variable::new(
                Tensor::from_f32(&offsets, &[b as i64, 2], device)?, false,
            );
            // Case: all lowercase (1.0) for word images.
            let case_data = vec![1.0f32; b];
            let case_var = Variable::new(
                Tensor::from_f32(&case_data, &[b as i64, 1], device)?, false,
            );

            // Forward on word image — origin = noisy starting position.
            let result = model.forward(&img_var, &case_var, &scan_start)?;

            // CE loss on target letter at this position.
            let target = Variable::new(batch.letter_idx[pos].clone(), false);
            let ce = cross_entropy_loss(&result.letter_logits, &target)?;

            // Case loss (all lowercase = class 1).
            let case_target = Variable::new(
                Tensor::from_i64(&vec![1i64; b], &[b as i64], device)?, false,
            );
            let case_loss = cross_entropy_loss(&result.case_logits, &case_target)?;

            let total = ce.add(&case_loss)?;

            // Accuracy.
            let preds = result.letter_logits.data().argmax(1, false)?;
            let acc: f64 = preds.eq_tensor(&batch.letter_idx[pos])?.mean()?.item()?;

            optimizer.zero_grad();
            total.backward()?;
            clip_grad_norm(&params, cfg.max_grad_norm)?;
            optimizer.step()?;

            model.graph.detach_state();

            let ce_val: f64 = ce.item()?;
            model.graph.record_scalar("ce", ce_val);
            model.graph.record_scalar("accuracy", acc);
            model.graph.record_scalar("lr", current_lr);

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

        let stats = LetterWordEpochStats {
            epoch,
            ce_loss: model.graph.trend("ce").latest(),
            accuracy: model.graph.trend("accuracy").latest(),
            lr: current_lr,
            duration: epoch_dur,
            eta,
        };

        if let Some(cb) = on_epoch {
            cb(&stats);
        }

        if let Some(ref mut m) = monitor {
            m.log(epoch, epoch_dur, &model.graph);
        }

        if let Some(ref mut f) = log_file {
            writeln!(f,
                "epoch {:3}  ce={:.4}  acc={:.1}%  lr={:.6}  [{:?}]  ETA {:?}",
                epoch + 1,
                stats.ce_loss, stats.accuracy * 100.0,
                stats.lr, stats.duration, stats.eta,
            ).ok();
            f.flush().ok();
        }

        // Checkpoint.
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

    // --- Finalize ---
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
        ) {
            eprintln!("warning: plot HTML: {e}");
        }
        if let Err(e) = model.graph.export_trends(
            &format!("{}/training.csv", cfg.save_dir), metric_tags,
        ) {
            eprintln!("warning: export CSV: {e}");
        }

        let manifest = serde_json::json!({
            "framework": "flodl",
            "model": "letter-word",
            "step": "1b",
            "config": cfg,
            "results": {
                "accuracy": model.graph.trend("accuracy").latest(),
                "ce": model.graph.trend("ce").latest(),
            },
            "parent": cfg.checkpoint,
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
            "model": "letter-word",
            "config": {
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
        ) {
            eprintln!("warning: write benchmark: {e}");
        }

        worker.finish();
    }

    Ok(())
}
