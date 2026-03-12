//! Training loop and configuration for the letter model.

use std::fs;
use std::io::Write;
use std::time::Instant;

use flodl::autograd::Variable;
use flodl::nn::{
    cross_entropy_loss, mse_loss, clip_grad_norm,
    Adam, CosineScheduler, Optimizer,
};
use flodl::tensor::{cuda_available, Device, DType, Result, Tensor, TensorError};

use super::data::{LetterDataset, LetterLoader};
use super::loss::{attention_guide_loss, fixation_diversity_loss, fixation_hit_rate};
use super::model::LetterModel;

/// Hyperparameters for letter model training.
pub struct LetterConfig {
    // Model architecture.
    pub n_classes: usize,
    pub n_glimpses: usize,
    pub patch_size: i64,
    pub n_scales: usize,
    pub latent_dim: i64,

    // Training.
    pub batch_size: usize,
    pub epochs: usize,
    pub lr: f64,
    pub min_lr: f64,
    pub max_grad_norm: f64,

    // Loss weights.
    pub guide_weight: f64,
    pub diversity_weight: f64,
    pub recon_weight: f64,
    pub recode_weight: f64,
    pub blur_sigma_ratio: f64,
    pub diversity_sigma: f64,
    pub diversity_vy: f64,

    // Checkpointing.
    pub save_dir: String,
    pub checkpoint_interval: usize,

    // Profiling.
    pub profile: bool,
}

impl Default for LetterConfig {
    fn default() -> Self {
        LetterConfig {
            n_classes: 26,
            n_glimpses: 8,
            patch_size: 12,
            n_scales: 1,
            latent_dim: 128,

            batch_size: 32,
            epochs: 100,
            lr: 0.001,
            min_lr: 0.0,
            max_grad_norm: 5.0,

            guide_weight: 8.0,
            diversity_weight: 1.0,
            recon_weight: 1.0,
            recode_weight: 0.0,
            blur_sigma_ratio: 0.16,
            diversity_sigma: 0.1,
            diversity_vy: 1.0,

            save_dir: "letter/training".into(),
            checkpoint_interval: 50,

            profile: false,
        }
    }
}

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
}

/// Run the full training loop for the letter model.
pub fn train_letter(
    cfg: &LetterConfig,
    ds: &LetterDataset,
    on_epoch: Option<&dyn Fn(&EpochStats)>,
) -> Result<()> {
    let model = LetterModel::new(
        cfg.n_classes, cfg.n_glimpses, cfg.patch_size, cfg.n_scales, cfg.latent_dim,
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

    if cfg.profile {
        model.graph.enable_profiling();
    }

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
        writeln!(f, "# epochs={}  batch={}  lr={:.4}→{:.4}  glimpses={}",
            cfg.epochs, cfg.batch_size, cfg.lr, cfg.min_lr, cfg.n_glimpses).ok();
        Some(f)
    } else {
        None
    };

    let metric_tags: &[&str] = &[
        "letter_ce", "case_ce", "letter_acc", "case_acc",
        "recon_mse", "guide", "diversity", "total", "hit_rate", "lr",
    ];

    for epoch in 0..cfg.epochs {
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

            // Classification losses.
            let letter_target = Variable::new(batch.letter_idx, false);
            let case_target = Variable::new(
                case_idx_from_float(&batch.case_label)?, false,
            );
            let letter_loss = cross_entropy_loss(&result.letter_logits, &letter_target)?;
            let case_loss = cross_entropy_loss(&result.case_logits, &case_target)?;

            // Reconstruction loss.
            let recon_loss = mse_loss(&result.recon, &img_var)?;

            // Attention losses.
            let guide_loss = attention_guide_loss(
                &clean_var, &result.locations, cfg.blur_sigma_ratio,
            )?;
            let div_loss = fixation_diversity_loss(
                &result.locations, cfg.diversity_sigma, cfg.diversity_vy,
            )?;

            // Total loss.
            let total = letter_loss.add(&case_loss)?;
            let total = total.add(&recon_loss.mul_scalar(cfg.recon_weight)?)?;
            let total = total.add(&guide_loss.mul_scalar(cfg.guide_weight)?)?;
            let total = total.add(&div_loss.mul_scalar(cfg.diversity_weight)?)?;

            optimizer.zero_grad();
            total.backward()?;
            clip_grad_norm(&params, cfg.max_grad_norm)?;
            optimizer.step()?;

            // Break gradient chain.
            model.graph.detach_state();

            // Record metrics.
            let (hr, _) = fixation_hit_rate(&clean_var, &result.locations, 0.3)?;
            let letter_acc = accuracy(&result.letter_logits.data(), &letter_target.data())?;
            let case_acc_val = accuracy(&result.case_logits.data(), &case_target.data())?;

            model.graph.record("letter_ce", &[letter_loss.item()?]);
            model.graph.record("case_ce", &[case_loss.item()?]);
            model.graph.record("letter_acc", &[letter_acc]);
            model.graph.record("case_acc", &[case_acc_val]);
            model.graph.record("recon_mse", &[recon_loss.item()?]);
            model.graph.record("guide", &[guide_loss.item()? * cfg.guide_weight]);
            model.graph.record("diversity", &[div_loss.item()?]);
            model.graph.record("total", &[total.item()?]);
            model.graph.record("hit_rate", &[hr]);
            model.graph.record("lr", &[current_lr]);

            if cfg.profile {
                model.graph.collect_timings(metric_tags);
            }

            n_batches += 1;
        }

        if n_batches == 0 {
            return Err(TensorError::new(
                &format!("epoch {epoch}: no batches produced"),
            ));
        }

        // Flush batch means → epoch history.
        model.graph.flush(&[]);
        if cfg.profile {
            model.graph.flush_timings(&[]);
        }

        let epoch_dur = epoch_start.elapsed();

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
        };

        if let Some(cb) = on_epoch {
            cb(&stats);
        }

        // Append to streaming log.
        if let Some(ref mut f) = log_file {
            writeln!(f,
                "epoch {:3}  ltr={:.4}({:.0}%)  case={:.4}({:.0}%)  recon={:.4}  guide={:.4}  div={:.4}  hit={:.0}%  lr={:.6}  [{:?}]",
                epoch + 1,
                stats.letter_loss, stats.letter_acc * 100.0,
                stats.case_loss, stats.case_acc * 100.0,
                stats.recon_loss, stats.guide_loss, stats.div_loss,
                stats.hit_rate * 100.0, stats.lr,
                stats.duration,
            ).ok();
            f.flush().ok();
        }

        // Checkpoint.
        if !cfg.save_dir.is_empty() && cfg.checkpoint_interval > 0
            && (epoch + 1) % cfg.checkpoint_interval == 0
        {
            let path = format!("{}/checkpoint_epoch_{}.bin", cfg.save_dir, epoch + 1);
            flodl::save_parameters_file(&path, &params)?;
        }
    }

    // Save final outputs.
    if !cfg.save_dir.is_empty() {
        let params = model.parameters();
        flodl::save_parameters_file(&format!("{}/model_final.bin", cfg.save_dir), &params)?;
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
        if cfg.profile
            && let Err(e) = model.graph.plot_timings_html(
                &format!("{}/timings.html", cfg.save_dir), metric_tags,
            )
        {
            eprintln!("warning: plot timings: {e}");
        }
    }

    Ok(())
}

/// Convert [B, 1] float case labels to [B] int64 indices.
fn case_idx_from_float(case_label: &Tensor) -> Result<Tensor> {
    case_label.squeeze(1)?.gt_scalar(0.5)?.to_dtype(DType::Int64)
}

/// Compute classification accuracy from logits and target indices.
fn accuracy(logits: &Tensor, targets: &Tensor) -> Result<f64> {
    let preds = logits.to_device(Device::CPU)?.argmax(1, false)?.to_i64_vec()?;
    let truth = targets.to_device(Device::CPU)?.to_i64_vec()?;
    if preds.is_empty() {
        return Ok(0.0);
    }
    let correct = preds.iter().zip(truth.iter()).filter(|(p, t)| p == t).count();
    Ok(correct as f64 / preds.len() as f64)
}
