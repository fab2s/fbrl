//! Training loop and configuration for the letter model.

use std::cell::Cell;
use std::fs;
use std::io::Write;
use std::rc::Rc;
use std::time::Instant;

use std::sync::atomic::{AtomicU64, Ordering};

use flodl::autograd::{Variable, grid_sample};
use flodl::nn::{
    cross_entropy_loss, mse_loss, bce_with_logits_loss, clip_grad_norm,
    Adam, CosineScheduler, Module,
};
use flodl::{Ddp, DdpConfig, ApplyPolicy, AverageBackend};
use flodl::monitor::Monitor;
use flodl::tensor::{
    cuda_available, cuda_device_count, Device, DType, Result, Tensor, TensorError, TensorOptions,
};
use flodl::{CpuWorker, EpochMetrics, LossContext};
use serde::{Serialize, Deserialize};

use super::data::{LetterBatchAdapter, LetterDataset, BATCH_NAMES};
use super::loss::{attention_guide_loss, build_void_grid, fixation_diversity_loss, fixation_hit_rate, leash_loss, void_repulsion_with_grid};
use super::model::LetterModel;

/// Arc wrapper so LetterBatchAdapter can be shared between training loop and DataLoader.
struct ArcAdapter(std::sync::Arc<LetterBatchAdapter>);
unsafe impl Send for ArcAdapter {}
unsafe impl Sync for ArcAdapter {}
impl flodl::BatchDataSet for ArcAdapter {
    fn len(&self) -> usize { self.0.dataset.samples.len() }
    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> { self.0.get_batch(indices) }
}

/// DDP mode selection.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub enum DdpMode {
    /// Synchronous DDP with El Che per-batch backward (default).
    #[default]
    SyncElChe,
    /// Async DDP: thread-per-GPU with configurable policy and backend.
    Async {
        policy: AsyncPolicy,
        backend: AsyncBackend,
    },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum AsyncPolicy { Sync, Cadence, Async }

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum AsyncBackend { Nccl, Cpu }

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

    // Image dimensions (set from dataset during training, read back during eval).
    #[serde(default = "default_128")]
    pub img_h: i64,
    #[serde(default = "default_128")]
    pub img_w: i64,

    // Training.
    pub batch_size: usize,
    pub epochs: usize,
    pub lr: f64,
    pub min_lr: f64,
    pub max_grad_norm: f64,

    // Origin noise curriculum — defines the tolerance band for composition.
    pub origin_noise_x: f64,
    pub origin_noise_y: f64,
    pub noise_start: f64,
    /// Fraction of total epochs over which noise ramps from noise_start to 1.0.
    pub noise_ramp_ratio: f64,

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

    // Recon annealing: cosine decay from recon_weight to recon_end_weight.
    // Set recon_end_weight < recon_weight to anneal reconstruction pressure.
    #[serde(default = "default_none_f64")]
    pub recon_end_weight: Option<f64>,

    // Leash: keeps read fixations near origin.
    #[serde(default)]
    pub leash_weight: f64,
    #[serde(default = "default_leash_radius")]
    pub leash_radius: f64,
    /// Fraction of leash_weight applied at epoch 0 (ramps to 1.0 over leash_ramp_ratio).
    #[serde(default = "default_leash_start")]
    pub leash_start: f64,
    /// Fraction of total epochs over which leash ramps from leash_start to 1.0.
    #[serde(default = "default_leash_ramp")]
    pub leash_ramp_ratio: f64,

    // DDP mode.
    #[serde(default)]
    pub ddp_mode: DdpMode,

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

            img_h: 128,
            img_w: 128,

            batch_size: 52,
            epochs: 100,
            lr: 0.001,
            min_lr: 0.0,
            max_grad_norm: 5.0,

            origin_noise_x: 0.3,
            origin_noise_y: 0.15,
            noise_start: 0.0,
            noise_ramp_ratio: 0.5,

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

            recon_end_weight: None,

            leash_weight: 0.0,
            leash_radius: 0.5,
            leash_start: 0.3,
            leash_ramp_ratio: 0.3,

            ddp_mode: DdpMode::default(),

            save_dir: "training".into(),
            checkpoint_interval: 50,

            monitor_port: None,
        }
    }
}

fn default_none_f64() -> Option<f64> { None }
fn default_128() -> i64 { 128 }
fn default_leash_radius() -> f64 { 0.5 }
fn default_leash_start() -> f64 { 0.3 }
fn default_leash_ramp() -> f64 { 0.3 }
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

/// Lightweight metrics for El Che: only classification + recon from gathered tags.
/// Avoids image-scale ops (recode, void, guide, blur) that OOM on gathered data
/// (anchor×devices batches concatenated).
fn record_el_che_metrics(
    model: &LetterModel,
    batch: &flodl::Batch,
    device: Device,
) -> Result<()> {
    let to_dev = |t: &Tensor| -> Result<Tensor> {
        if t.device() == device { Ok(t.clone()) } else { t.to_device(device) }
    };

    let letter_logits = model.graph.tagged("heads_0").expect("heads_0");
    let case_logits = model.graph.tagged("heads_1").expect("heads_1");
    let recon = model.graph.tagged("recon").expect("recon");

    let letter_idx = to_dev(&batch["letter_idx"])?;
    let case_idx = case_idx_from_float(&to_dev(&batch["case"])?)?;
    let img = to_dev(&batch["image"])?;

    let letter_ce = cross_entropy_loss(&letter_logits, &Variable::new(letter_idx.clone(), false))?.item()?;
    let case_ce = cross_entropy_loss(&case_logits, &Variable::new(case_idx.clone(), false))?.item()?;
    let recon_mse = mse_loss(&recon, &Variable::new(img, false))?.item()?;

    model.graph.record_scalar("letter_ce", letter_ce);
    model.graph.record_scalar("case_ce", case_ce);
    model.graph.record_scalar("letter_acc", accuracy(&letter_logits.data(), &letter_idx)?);
    model.graph.record_scalar("case_acc", accuracy(&case_logits.data(), &case_idx)?);
    model.graph.record_scalar("recon_mse", recon_mse);
    model.graph.record_scalar("recode", 0.0);
    model.graph.record_scalar("content", 0.0);
    model.graph.record_scalar("guide", 0.0);
    model.graph.record_scalar("void", 0.0);
    model.graph.record_scalar("diversity", 0.0);
    model.graph.record_scalar("leash", 0.0);
    model.graph.record_scalar("total", letter_ce + case_ce + recon_mse);
    model.graph.record_scalar("hit_rate", 0.0);

    Ok(())
}

/// Compute full loss stack from gathered tags, traces, and batch targets.
/// Works identically for 1 or N GPUs — all data is on the gather device.
#[allow(clippy::too_many_arguments)]
fn compute_loss(
    cfg: &LetterConfig,
    model: &LetterModel,
    batch: &flodl::Batch,
    device: Device,
    cur_recon_weight: f64,
    cur_leash_weight: f64,
    has_partners: bool,
    scan_void_grid: Option<&Tensor>,
    read_void_grid: Option<&Tensor>,
) -> Result<Variable> {
    // Move batch tensors to model device (gather device may be CPU when streaming).
    let to_dev = |t: &Tensor| -> Result<Tensor> {
        if t.device() == device { Ok(t.clone()) } else { t.to_device(device) }
    };
    let zero = || Variable::new(Tensor::zeros(&[1], TensorOptions { device, ..Default::default() }).unwrap(), false);

    // Read gathered outputs from graph tags + traces.
    let letter_logits = model.graph.tagged("heads_0").expect("heads_0");
    let case_logits = model.graph.tagged("heads_1").expect("heads_1");
    let recon = model.graph.tagged("recon").expect("recon");
    let latent = model.graph.tagged("latent").expect("latent");

    let all_traces = model.graph.traces("attn").unwrap_or_default();
    let (scan_locs, read_locs) = if cfg.n_scan > 0 && all_traces.len() > cfg.n_scan {
        let (s, r) = all_traces.split_at(cfg.n_scan);
        (s.to_vec(), r.to_vec())
    } else {
        (Vec::new(), all_traces)
    };

    let img_var = Variable::new(to_dev(&batch["image"])?, false);
    let case_var = Variable::new(to_dev(&batch["case"])?, false);
    let clean_var = Variable::new(to_dev(&batch["clean"])?, false);
    let origin_var = Variable::new(to_dev(&batch["origin"])?, false);

    // Classification.
    let letter_target = Variable::new(to_dev(&batch["letter_idx"])?, false);
    let case_idx = case_idx_from_float(&to_dev(&batch["case"])?)?;
    let case_target = Variable::new(case_idx.clone(), false);
    let letter_loss = cross_entropy_loss(&letter_logits, &letter_target)?;
    let case_loss = cross_entropy_loss(&case_logits, &case_target)?;

    // Reconstruction.
    let recon_loss = mse_loss(&recon, &img_var)?;

    // Recode.
    let recode_loss = if cfg.recode_weight > 0.0 && has_partners {
        let b = batch["case"].shape()[0];
        let ones_t = Tensor::from_f32(&vec![1.0f32; b as usize], &[b, 1], device)?;
        let ones_var = Variable::new(ones_t, false);
        let flipped_case = ones_var.sub(&case_var)?;
        let z_recode = latent.cat(&flipped_case, 1)?;
        let recode = model.decoder.forward(&z_recode)?;
        mse_loss(&recode, &Variable::new(to_dev(&batch["partner_clean"])?, false))?
    } else { zero() };

    // Content (from main graph's RefCell — not gathered across replicas).
    // Skip when batch sizes don't match (El Che gathers traces but not RefCell state).
    let content_loss = if cfg.content_weight > 0.0 && cfg.n_scan > 0 {
        let scan_logits = model.content_logits();
        let logit_b = scan_logits.first().map(|l| l.shape()[0]).unwrap_or(0);
        let trace_b = scan_locs.first().map(|l| l.shape()[0]).unwrap_or(0);
        if !scan_logits.is_empty() && !scan_locs.is_empty() && logit_b == trace_b {
            let mut loss_sum = zero();
            for (loc, logit) in scan_locs.iter().zip(scan_logits.iter()) {
                let grid = loc.unsqueeze(1)?.unsqueeze(2)?;
                let sampled = grid_sample(&clean_var, &grid, 0, 0, true)?;
                let label_t = sampled.data().reshape(&[-1, 1])?.gt_scalar(0.1)?.to_dtype(DType::Float32)?;
                let label = Variable::new(label_t, false);
                loss_sum = loss_sum.add(&bce_with_logits_loss(logit, &label)?)?;
            }
            loss_sum.mul_scalar(1.0 / scan_logits.len() as f64)?
        } else { zero() }
    } else { zero() };

    // Attention guide (gathered traces).
    let scan_guide = if cfg.scan_guide_weight > 0.0 && !scan_locs.is_empty() {
        attention_guide_loss(&clean_var, &scan_locs, cfg.blur_sigma_ratio)?
    } else { zero() };
    let read_guide = if cfg.read_guide_weight > 0.0 && !read_locs.is_empty() {
        attention_guide_loss(&clean_var, &read_locs, cfg.blur_sigma_ratio)?
    } else { zero() };

    // Void repulsion (gathered traces).
    let scan_void = if cfg.scan_void_weight > 0.0 && !scan_locs.is_empty() {
        void_repulsion_with_grid(&clean_var, &scan_locs, cfg.patch_size, cfg.scan_patch_w, 0.1, scan_void_grid)?
    } else { zero() };
    let read_void = if cfg.void_weight > 0.0 && !read_locs.is_empty() {
        void_repulsion_with_grid(&clean_var, &read_locs, cfg.patch_size, cfg.patch_size, 0.1, read_void_grid)?
    } else { zero() };

    // Diversity (gathered traces).
    let div_loss = if !scan_locs.is_empty() || !read_locs.is_empty() {
        let sd = fixation_diversity_loss(&scan_locs, cfg.diversity_sigma, cfg.scan_vy)?;
        let rd = fixation_diversity_loss(&read_locs, cfg.diversity_sigma, cfg.read_vy)?;
        sd.add(&rd)?
    } else { zero() };

    // Leash (gathered traces + gathered origin).
    let leash = if cur_leash_weight > 0.0 && !read_locs.is_empty() {
        leash_loss(&read_locs, &origin_var, cfg.leash_radius)?
    } else { zero() };

    // Total.
    let total = letter_loss.add(&case_loss)?;
    let total = total.add(&recon_loss.mul_scalar(cur_recon_weight)?)?;
    let total = total.add(&recode_loss.mul_scalar(cfg.recode_weight)?)?;
    let total = total.add(&content_loss.mul_scalar(cfg.content_weight)?)?;
    let total = total.add(&scan_guide.mul_scalar(cfg.scan_guide_weight)?)?;
    let total = total.add(&read_guide.mul_scalar(cfg.read_guide_weight)?)?;
    let total = total.add(&scan_void.mul_scalar(cfg.scan_void_weight)?)?;
    let total = total.add(&read_void.mul_scalar(cfg.void_weight)?)?;
    let total = total.add(&div_loss.mul_scalar(cfg.diversity_weight)?)?;
    let total = total.add(&leash.mul_scalar(cur_leash_weight)?)?;

    // Record scalar metrics on graph (for monitor/plotting).
    let all_locs: Vec<Variable> = scan_locs.iter().chain(read_locs.iter()).cloned().collect();
    let (hr, _) = fixation_hit_rate(&clean_var, &all_locs, 0.3)?;
    model.graph.record_scalar("letter_ce", letter_loss.item()?);
    model.graph.record_scalar("case_ce", case_loss.item()?);
    model.graph.record_scalar("letter_acc", accuracy(&letter_logits.data(), &to_dev(&batch["letter_idx"])?)?);
    model.graph.record_scalar("case_acc", accuracy(&case_logits.data(), &case_idx)?);
    model.graph.record_scalar("recon_mse", recon_loss.item()?);
    model.graph.record_scalar("recode", recode_loss.item()?);
    model.graph.record_scalar("content", content_loss.item()?);
    model.graph.record_scalar("guide",
        scan_guide.item()? * cfg.scan_guide_weight + read_guide.item()? * cfg.read_guide_weight);
    model.graph.record_scalar("void",
        scan_void.item()? * cfg.scan_void_weight + read_void.item()? * cfg.void_weight);
    model.graph.record_scalar("diversity", div_loss.item()?);
    model.graph.record_scalar("leash", leash.item()?);
    model.graph.record_scalar("total", total.item()?);
    model.graph.record_scalar("hit_rate", hr);

    Ok(total)
}

/// Run the full training loop for the letter model.
pub fn train_letter(
    cfg: &LetterConfig,
    ds: &LetterDataset,
    on_epoch: Option<&dyn Fn(&EpochStats)>,
) -> Result<()> {
    // Dispatch based on DDP mode.
    if let DdpMode::Async { ref policy, ref backend } = cfg.ddp_mode {
        return train_letter_async(cfg, ds, policy, backend);
    }

    // --- SyncElChe path (existing) ---

    let img_shape = ds.samples[0].image.shape();
    let (img_h, img_w) = (img_shape[1], img_shape[2]);

    let model = LetterModel::new(
        cfg.n_classes, cfg.n_scan, cfg.n_read, cfg.patch_size, cfg.scan_patch_w,
        cfg.n_scales, cfg.latent_dim, img_h, img_w,
    )?;

    // Move model to CUDA if available.
    let device = if cuda_available() {
        flodl::tensor::set_cudnn_benchmark(true);
        model.graph.set_device(Device::CUDA(0));
        Device::CUDA(0)
    } else {
        eprintln!("Using CPU");
        Device::CPU
    };

    let scheduler = CosineScheduler::new(cfg.lr, cfg.min_lr, cfg.epochs);

    // Transparent DDP: Ddp::setup_with detects GPUs, creates replicas,
    // sets per-replica optimizers, enables El Che for heterogeneous hardware.
    // Training loop is identical for 1 or N GPUs.
    let n_classes = cfg.n_classes;
    let n_scan = cfg.n_scan;
    let n_read = cfg.n_read;
    let patch_size = cfg.patch_size;
    let scan_patch_w = cfg.scan_patch_w;
    let n_scales = cfg.n_scales;
    let latent_dim = cfg.latent_dim;
    let lr = cfg.lr;

    Ddp::setup_with(
        &model.graph,
        |dev| {
            let replica = LetterModel::new(
                n_classes, n_scan, n_read, patch_size, scan_patch_w,
                n_scales, latent_dim, img_h, img_w,
            )?;
            replica.graph.set_device(dev);
            Ok(replica.into_graph())
        },
        |p| Adam::new(p, lr),
        DdpConfig::new()
            .speed_hint(1, 2.3)
            .overhead_target(0.10),
    )?;

    // El Che per-batch backward: loss closure runs inside forward_batch,
    // one activation graph in VRAM at a time. Anchor scales freely.
    let is_el_che = cuda_device_count() > 1;

    // Per-epoch dynamic values shared with loss closure via interior mutability.
    let recon_w_cell = Rc::new(Cell::new(cfg.recon_weight));
    let leash_w_cell = Rc::new(Cell::new(cfg.leash_weight));

    if is_el_che {
        let n_scan = cfg.n_scan;
        let patch_size = cfg.patch_size;
        let scan_patch_w = cfg.scan_patch_w;
        let scan_gw = cfg.scan_guide_weight;
        let read_gw = cfg.read_guide_weight;
        let scan_vw = cfg.scan_void_weight;
        let void_w = cfg.void_weight;
        let div_w = cfg.diversity_weight;
        let blur_sr = cfg.blur_sigma_ratio;
        let div_sigma = cfg.diversity_sigma;
        let s_vy = cfg.scan_vy;
        let r_vy = cfg.read_vy;
        let l_radius = cfg.leash_radius;
        let rw = recon_w_cell.clone();
        let lw = leash_w_cell.clone();

        // Pure loss computation — no .item() calls, no GPU→CPU syncs.
        // Metrics are computed AFTER forward_batch from gathered detached data.
        model.graph.set_loss_fn(move |ctx: &LossContext| {
            let dev = ctx.output.device();
            let zero = || Variable::new(
                Tensor::zeros(&[1], TensorOptions { device: dev, ..Default::default() }).unwrap(), false
            );

            let letter_logits = ctx.tags.get("heads_0").expect("heads_0");
            let case_logits = ctx.tags.get("heads_1").expect("heads_1");
            let recon = ctx.tags.get("recon").expect("recon");

            let all_traces = ctx.traces.get("attn").map(|v| v.as_slice()).unwrap_or(&[]);
            let (scan_locs, read_locs) = if n_scan > 0 && all_traces.len() > n_scan {
                all_traces.split_at(n_scan)
            } else {
                (&[][..], all_traces)
            };

            let img = Variable::new(ctx.batch["image"].clone(), false);
            let clean = Variable::new(ctx.batch["clean"].clone(), false);
            let origin = Variable::new(ctx.batch["origin"].clone(), false);
            let letter_target = Variable::new(ctx.batch["letter_idx"].clone(), false);
            let case_idx = case_idx_from_float(&ctx.batch["case"])?;
            let case_target = Variable::new(case_idx, false);

            let letter_loss = cross_entropy_loss(letter_logits, &letter_target)?;
            let case_loss = cross_entropy_loss(case_logits, &case_target)?;
            let recon_loss = mse_loss(recon, &img)?;

            let scan_guide = if scan_gw > 0.0 && !scan_locs.is_empty() {
                attention_guide_loss(&clean, scan_locs, blur_sr)?
            } else { zero() };
            let read_guide = if read_gw > 0.0 && !read_locs.is_empty() {
                attention_guide_loss(&clean, read_locs, blur_sr)?
            } else { zero() };

            let scan_void = if scan_vw > 0.0 && !scan_locs.is_empty() {
                void_repulsion_with_grid(&clean, scan_locs, patch_size, scan_patch_w, 0.1, None)?
            } else { zero() };
            let read_void = if void_w > 0.0 && !read_locs.is_empty() {
                void_repulsion_with_grid(&clean, read_locs, patch_size, patch_size, 0.1, None)?
            } else { zero() };

            let div_loss = if !scan_locs.is_empty() || !read_locs.is_empty() {
                fixation_diversity_loss(scan_locs, div_sigma, s_vy)?
                    .add(&fixation_diversity_loss(read_locs, div_sigma, r_vy)?)?
            } else { zero() };

            let cur_leash = lw.get();
            let leash = if cur_leash > 0.0 && !read_locs.is_empty() {
                leash_loss(read_locs, &origin, l_radius)?
            } else { zero() };

            let cur_recon = rw.get();
            letter_loss.add(&case_loss)?
                .add(&recon_loss.mul_scalar(cur_recon)?)?
                .add(&scan_guide.mul_scalar(scan_gw)?)?
                .add(&read_guide.mul_scalar(read_gw)?)?
                .add(&scan_void.mul_scalar(scan_vw)?)?
                .add(&read_void.mul_scalar(void_w)?)?
                .add(&div_loss.mul_scalar(div_w)?)?
                .add(&leash.mul_scalar(cur_leash)?)
        });

        eprintln!("  El Che: per-batch backward enabled, anchor auto-tuned");
    }

    // flodl DataLoader: VRAM-aware auto-depth, origin noise injected per batch.
    let adapter = std::sync::Arc::new(LetterBatchAdapter::new(ds.clone()));
    let loader = {
        let a = std::sync::Arc::clone(&adapter);
        flodl::DataLoader::from_batch_dataset(ArcAdapter(a))
            .batch_size(cfg.batch_size)
            .device(device)
            .vram_max_usage(0.90)
            .names(BATCH_NAMES)
            .build()?
    };
    eprintln!("  loader: {} (depth {})", if loader.is_resident() { "resident" } else { "streaming" }, loader.prefetch_depth());
    model.graph.set_data_loader(loader, "image")?;

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

        // Noise curriculum: ramp from noise_start to 1.0 over noise_ramp_ratio of total epochs.
        let noise_progress = if cfg.noise_ramp_ratio > 0.0 {
            let ramp_epochs = (cfg.noise_ramp_ratio * cfg.epochs as f64).max(1.0);
            let t = (epoch as f64 / ramp_epochs).min(1.0);
            cfg.noise_start + (1.0 - cfg.noise_start) * t
        } else {
            1.0
        };
        adapter.set_noise(cfg.origin_noise_x * noise_progress, cfg.origin_noise_y * noise_progress);

        // Per-epoch dynamic weights.
        let cur_leash_weight = cfg.leash_weight;
        let cur_recon_weight = match cfg.recon_end_weight {
            Some(end_w) if end_w < cfg.recon_weight => {
                let t = epoch as f64 / (cfg.epochs.max(1) - 1) as f64;
                let cos_decay = 0.5 * (1.0 + (std::f64::consts::PI * t).cos());
                end_w + (cfg.recon_weight - end_w) * cos_decay
            }
            _ => cfg.recon_weight,
        };

        // Push per-epoch values into shared cells (read by El Che loss closure).
        recon_w_cell.set(cur_recon_weight);
        leash_w_cell.set(cur_leash_weight);

        let current_lr = scheduler.lr(epoch);
        model.graph.set_lr(current_lr);

        let epoch_start = Instant::now();
        let mut n_batches = 0usize;

        for batch in model.graph.epoch(epoch).activate() {
            let batch = batch?;
            let _out = model.graph.forward_batch(&batch)?;

            if is_el_che {
                // Per-batch backward already happened inside forward_batch.
                // Lightweight metrics from gathered detached tags (small tensors only).
                // Avoid image-scale ops (recode, void, guide) — they OOM on gathered data.
                record_el_che_metrics(&model, &batch, device)?;
            } else {
                // Single-GPU: external loss + backward.
                let total = compute_loss(
                    cfg, &model, &batch, device,
                    cur_recon_weight, cur_leash_weight, ds.has_partners,
                    scan_void_grid.as_ref(), read_void_grid.as_ref(),
                )?;
                total.backward()?;
            }

            clip_grad_norm(&model.parameters(), cfg.max_grad_norm)?;
            model.graph.step()?;

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
        // Inject actual image dimensions (cfg may have defaults).
        let mut config_json = serde_json::to_value(cfg).unwrap_or_default();
        config_json["img_h"] = serde_json::json!(img_h);
        config_json["img_w"] = serde_json::json!(img_w);
        let manifest = serde_json::json!({
            "framework": "flodl",
            "model": "letter",
            "config": config_json,
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
        if device != Device::CPU
            && let Ok((used, total)) = flodl::tensor::cuda_memory_info() {
                let gpu = flodl::tensor::cuda_device_name().unwrap_or_default();
                bench["gpu"] = serde_json::json!(gpu);
                bench["vram"] = serde_json::json!({
                    "device_used_mb": (used as f64 / 1024.0 / 1024.0).round() as i64,
                    "device_total_mb": (total as f64 / 1024.0 / 1024.0).round() as i64,
                });
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

// ---------------------------------------------------------------------------
// Async DDP path: framework-managed training via Ddp::builder
// ---------------------------------------------------------------------------

/// Train with async DDP (thread-per-GPU, framework-managed epoch loop).
fn train_letter_async(
    cfg: &LetterConfig,
    ds: &LetterDataset,
    policy: &AsyncPolicy,
    backend: &AsyncBackend,
) -> Result<()> {
    let img_shape = ds.samples[0].image.shape();
    let (img_h, img_w) = (img_shape[1], img_shape[2]);

    // Config values for closures (must be Copy/Clone, not references).
    let n_classes = cfg.n_classes;
    let n_scan = cfg.n_scan;
    let n_read = cfg.n_read;
    let patch_size = cfg.patch_size;
    let scan_patch_w = cfg.scan_patch_w;
    let n_scales = cfg.n_scales;
    let latent_dim = cfg.latent_dim;
    let lr = cfg.lr;
    let min_lr = cfg.min_lr;
    let epochs = cfg.epochs;
    let has_partners = ds.has_partners;

    // Static loss weights.
    let scan_gw = cfg.scan_guide_weight;
    let read_gw = cfg.read_guide_weight;
    let scan_vw = cfg.scan_void_weight;
    let void_w = cfg.void_weight;
    let div_w = cfg.diversity_weight;
    let recode_w = cfg.recode_weight;
    let content_w = cfg.content_weight;
    let blur_sr = cfg.blur_sigma_ratio;
    let div_sigma = cfg.diversity_sigma;
    let s_vy = cfg.scan_vy;
    let r_vy = cfg.read_vy;
    let l_radius = cfg.leash_radius;

    // Dynamic weights via atomics (updated in epoch_fn, read in train_fn).
    let recon_w = std::sync::Arc::new(AtomicU64::new(cfg.recon_weight.to_bits()));
    let leash_w = std::sync::Arc::new(AtomicU64::new(cfg.leash_weight.to_bits()));
    let rw_epoch = recon_w.clone();
    let lw_epoch = leash_w.clone();
    let rw_train = recon_w.clone();
    let lw_train = leash_w.clone();

    // Noise/annealing config for epoch_fn.
    let noise_x = cfg.origin_noise_x;
    let noise_y = cfg.origin_noise_y;
    let noise_start = cfg.noise_start;
    let noise_ramp = cfg.noise_ramp_ratio;
    let recon_weight_base = cfg.recon_weight;
    let recon_end_weight = cfg.recon_end_weight;
    let leash_weight_base = cfg.leash_weight;

    // Dataset adapter (thread-safe via AtomicU64 noise params).
    let adapter = std::sync::Arc::new(LetterBatchAdapter::new(ds.clone()));
    let adapter_epoch = adapter.clone();

    let policy = match policy {
        AsyncPolicy::Sync => ApplyPolicy::Sync,
        AsyncPolicy::Cadence => ApplyPolicy::Cadence,
        AsyncPolicy::Async => ApplyPolicy::Async,
    };
    let backend = match backend {
        AsyncBackend::Nccl => AverageBackend::Nccl,
        AsyncBackend::Cpu => AverageBackend::Cpu,
    };

    // Ensure save directory.
    if !cfg.save_dir.is_empty() {
        fs::create_dir_all(&cfg.save_dir)
            .map_err(|e| TensorError::new(&format!("create save dir: {e}")))?;
        rotate_prior_run(&cfg.save_dir);
    }

    let save_dir = cfg.save_dir.clone();
    let save_dir_main = cfg.save_dir.clone();
    let ckpt_interval = cfg.checkpoint_interval;

    eprintln!("  mode: async DDP (policy={policy:?}, backend={backend:?})");

    // Live monitor (optional).
    let mut monitor = if let Some(port) = cfg.monitor_port {
        let mut m = Monitor::new(cfg.epochs);
        if let Err(e) = m.serve(port) {
            eprintln!("warning: monitor server: {e}");
        }
        m.set_metadata(serde_json::to_value(cfg).unwrap_or_default());
        if !save_dir_main.is_empty() {
            m.save_html(&format!("{}/dashboard.html", save_dir_main));
        }
        Some(m)
    } else {
        None
    };

    // Open streaming log.
    let mut log_file = if !save_dir_main.is_empty() {
        let path = format!("{}/training.log", save_dir_main);
        let mut f = fs::File::create(&path)
            .map_err(|e| TensorError::new(&format!("create log: {e}")))?;
        writeln!(f, "# fbrl letter training (async DDP)").ok();
        writeln!(f, "# epochs={}  batch={}  lr={:.4}\u{2192}{:.4}  scan={}  read={}  policy={:?}  backend={:?}",
            cfg.epochs, cfg.batch_size, cfg.lr, cfg.min_lr, cfg.n_scan, cfg.n_read, policy, backend).ok();
        Some(f)
    } else {
        None
    };

    let handle = Ddp::builder(
        // model_factory: one LetterModel per GPU.
        move |dev| {
            let m = LetterModel::new(
                n_classes, n_scan, n_read, patch_size, scan_patch_w,
                n_scales, latent_dim, img_h, img_w,
            )?;
            m.graph.set_device(dev);
            Ok(m)
        },
        // optim_factory
        move |params| Adam::new(params, lr),
        // train_fn: full loss stack, called per batch per worker.
        move |model: &LetterModel, batch: &[Tensor]| {
            let dev = batch[0].device();
            let zero = || Variable::new(
                Tensor::zeros(&[1], TensorOptions { device: dev, ..Default::default() }).unwrap(), false
            );

            // Batch fields in BATCH_NAMES order:
            // [0]=image, [1]=clean, [2]=partner_clean, [3]=letter_idx, [4]=case, [5]=origin
            let img = Variable::new(batch[0].clone(), false);
            let clean = Variable::new(batch[1].clone(), false);
            let case = Variable::new(batch[4].clone(), false);
            let origin = Variable::new(batch[5].clone(), false);

            let result = model.forward(&img, &case, &origin)?;

            // Classification.
            let letter_target = Variable::new(batch[3].clone(), false);
            let case_idx = case_idx_from_float(&batch[4])?;
            let case_target = Variable::new(case_idx, false);
            let letter_loss = cross_entropy_loss(&result.letter_logits, &letter_target)?;
            let case_loss = cross_entropy_loss(&result.case_logits, &case_target)?;

            // Reconstruction.
            let recon_loss = mse_loss(&result.recon, &img)?;

            // Recode.
            let recode_loss = if recode_w > 0.0 && has_partners {
                let b = batch[4].shape()[0];
                let ones = Variable::new(
                    Tensor::from_f32(&vec![1.0f32; b as usize], &[b, 1], dev)?, false,
                );
                let flipped = ones.sub(&case)?;
                let z_recode = result.latent.cat(&flipped, 1)?;
                let recode = model.decoder.forward(&z_recode)?;
                mse_loss(&recode, &Variable::new(batch[2].clone(), false))?
            } else { zero() };

            // Content.
            let content_loss = if content_w > 0.0 && n_scan > 0 {
                let scan_logits = model.content_logits();
                if !scan_logits.is_empty() && !result.scan_locations.is_empty() {
                    let mut loss_sum = zero();
                    for (loc, logit) in result.scan_locations.iter().zip(scan_logits.iter()) {
                        let grid = loc.unsqueeze(1)?.unsqueeze(2)?;
                        let sampled = grid_sample(&clean, &grid, 0, 0, true)?;
                        let label_t = sampled.data().reshape(&[-1, 1])?
                            .gt_scalar(0.1)?.to_dtype(DType::Float32)?;
                        let label = Variable::new(label_t, false);
                        loss_sum = loss_sum.add(&bce_with_logits_loss(logit, &label)?)?;
                    }
                    loss_sum.mul_scalar(1.0 / scan_logits.len() as f64)?
                } else { zero() }
            } else { zero() };

            // Attention guide.
            let scan_guide = if scan_gw > 0.0 && !result.scan_locations.is_empty() {
                attention_guide_loss(&clean, &result.scan_locations, blur_sr)?
            } else { zero() };
            let read_guide = if read_gw > 0.0 && !result.read_locations.is_empty() {
                attention_guide_loss(&clean, &result.read_locations, blur_sr)?
            } else { zero() };

            // Void repulsion (grid built per call — cheap meshgrid).
            let scan_void = if scan_vw > 0.0 && !result.scan_locations.is_empty() {
                void_repulsion_with_grid(&clean, &result.scan_locations, patch_size, scan_patch_w, 0.1, None)?
            } else { zero() };
            let read_void = if void_w > 0.0 && !result.read_locations.is_empty() {
                void_repulsion_with_grid(&clean, &result.read_locations, patch_size, patch_size, 0.1, None)?
            } else { zero() };

            // Diversity.
            let div_loss = if !result.scan_locations.is_empty() || !result.read_locations.is_empty() {
                fixation_diversity_loss(&result.scan_locations, div_sigma, s_vy)?
                    .add(&fixation_diversity_loss(&result.read_locations, div_sigma, r_vy)?)?
            } else { zero() };

            // Leash.
            let cur_leash = f64::from_bits(lw_train.load(Ordering::Relaxed));
            let leash = if cur_leash > 0.0 && !result.read_locations.is_empty() {
                leash_loss(&result.read_locations, &origin, l_radius)?
            } else { zero() };

            // Total.
            let cur_recon = f64::from_bits(rw_train.load(Ordering::Relaxed));
            let total = letter_loss.add(&case_loss)?
                .add(&recon_loss.mul_scalar(cur_recon)?)?
                .add(&recode_loss.mul_scalar(recode_w)?)?
                .add(&content_loss.mul_scalar(content_w)?)?
                .add(&scan_guide.mul_scalar(scan_gw)?)?
                .add(&read_guide.mul_scalar(read_gw)?)?
                .add(&scan_void.mul_scalar(scan_vw)?)?
                .add(&read_void.mul_scalar(void_w)?)?
                .add(&div_loss.mul_scalar(div_w)?)?
                .add(&leash.mul_scalar(cur_leash)?)?;

            // Record per-component metrics (thread-local, drained at epoch boundary).
            flodl::record_scalar("letter_ce", letter_loss.item()?);
            flodl::record_scalar("case_ce", case_loss.item()?);
            flodl::record_scalar("letter_acc", accuracy(&result.letter_logits.data(), &batch[3])?);
            flodl::record_scalar("case_acc", accuracy(&result.case_logits.data(), &case_target.data())?);
            flodl::record_scalar("recon_mse", recon_loss.item()?);
            flodl::record_scalar("recode", recode_loss.item()?);
            flodl::record_scalar("content", content_loss.item()?);
            flodl::record_scalar("guide",
                scan_guide.item()? * scan_gw + read_guide.item()? * read_gw);
            flodl::record_scalar("void",
                scan_void.item()? * scan_vw + read_void.item()? * void_w);
            flodl::record_scalar("diversity", div_loss.item()?);
            flodl::record_scalar("leash", leash.item()?);
            flodl::record_scalar("total", total.item()?);
            let all_locs: Vec<Variable> = result.scan_locations.iter()
                .chain(result.read_locations.iter()).cloned().collect();
            let (hr, _) = fixation_hit_rate(&clean, &all_locs, 0.3)?;
            flodl::record_scalar("hit_rate", hr);

            Ok(total)
        },
    )
    .dataset(std::sync::Arc::new(ArcAdapter(adapter.clone())))
    .batch_size(cfg.batch_size)
    .num_epochs(cfg.epochs)
    .policy(policy)
    .backend(backend)
    .overhead_target(0.10)
    .epoch_fn(move |epoch: usize, worker: &mut flodl::GpuWorker<LetterModel>| {
        // LR schedule.
        let sched = CosineScheduler::new(lr, min_lr, epochs);
        let current_lr = sched.lr(epoch);
        worker.set_lr(current_lr);
        flodl::record_scalar("lr", current_lr);

        // Noise curriculum.
        let noise_progress = if noise_ramp > 0.0 {
            let ramp_epochs = (noise_ramp * epochs as f64).max(1.0);
            let t = (epoch as f64 / ramp_epochs).min(1.0);
            noise_start + (1.0 - noise_start) * t
        } else { 1.0 };
        adapter_epoch.set_noise(noise_x * noise_progress, noise_y * noise_progress);

        // Recon weight annealing.
        let cur_recon = match recon_end_weight {
            Some(end_w) if end_w < recon_weight_base => {
                let t = epoch as f64 / (epochs.max(1) - 1) as f64;
                let cos_decay = 0.5 * (1.0 + (std::f64::consts::PI * t).cos());
                end_w + (recon_weight_base - end_w) * cos_decay
            }
            _ => recon_weight_base,
        };
        rw_epoch.store(cur_recon.to_bits(), Ordering::Relaxed);
        lw_epoch.store(leash_weight_base.to_bits(), Ordering::Relaxed);
    })
    .checkpoint_every(ckpt_interval)
    .checkpoint_fn(move |ver, model: &LetterModel| {
        if !save_dir.is_empty() {
            let path = format!("{}/checkpoint_epoch_{}.fdl.gz", save_dir, ver);
            model.graph.snapshot_cpu()?.save_file(&path)?;
        }
        Ok(())
    })
    .run()?;

    // Wire architecture SVG + metadata to dashboard.
    if let Some(ref mut mon) = monitor {
        handle.setup_monitor(mon);
    }

    eprintln!("  async DDP training launched, waiting for completion...");

    // Consume epoch metrics: feed monitor, write log, collect for artifacts.
    let train_start = Instant::now();
    let mut last_metrics: Option<EpochMetrics> = None;
    let mut epoch_times: Vec<f64> = Vec::with_capacity(cfg.epochs);

    while let Some(m) = handle.next_metrics() {
        let dur = std::time::Duration::from_millis(m.epoch_ms as u64);
        epoch_times.push(m.epoch_ms / 1000.0);

        // Feed live monitor.
        if let Some(ref mut mon) = monitor {
            mon.log(m.epoch, dur, &m);
        }

        // Streaming log (matches sync format).
        let s = |k: &str| *m.scalars.get(k).unwrap_or(&0.0);
        eprintln!(
            "epoch {:3}  ltr={:.4}({:.0}%)  case={:.4}({:.0}%)  recon={:.4}  recode={:.4}  content={:.4}  \
             guide={:.4}  void={:.4}  div={:.4}  hit={:.0}%  lr={:.6}  [{:?}]",
            m.epoch + 1,
            s("letter_ce"), s("letter_acc") * 100.0,
            s("case_ce"), s("case_acc") * 100.0,
            s("recon_mse"), s("recode"), s("content"),
            s("guide"), s("void"), s("diversity"),
            s("hit_rate") * 100.0, s("lr"),
            dur,
        );
        if let Some(ref mut f) = log_file {
            writeln!(f,
                "epoch {:3}  ltr={:.4}({:.0}%)  case={:.4}({:.0}%)  recon={:.4}  recode={:.4}  content={:.4}  \
                 guide={:.4}  void={:.4}  div={:.4}  hit={:.0}%  lr={:.6}  [{:?}]",
                m.epoch + 1,
                s("letter_ce"), s("letter_acc") * 100.0,
                s("case_ce"), s("case_acc") * 100.0,
                s("recon_mse"), s("recode"), s("content"),
                s("guide"), s("void"), s("diversity"),
                s("hit_rate") * 100.0, s("lr"),
                dur,
            ).ok();
            f.flush().ok();
        }

        last_metrics = Some(m);
    }

    let n_gpus = handle.devices().len();
    let _state = handle.join()?;

    // Finalize monitor.
    if let Some(ref mut m) = monitor {
        m.finish();
    }

    // Save post-training artifacts.
    if !save_dir_main.is_empty() {
        // CSV export from monitor (has all epoch data from log() calls).
        if let Some(ref m) = monitor
            && let Err(e) = m.export_csv(&format!("{}/training.csv", save_dir_main))
        {
            eprintln!("warning: export CSV: {e}");
        }

        // Manifest with config + final results.
        let mut config_json = serde_json::to_value(cfg).unwrap_or_default();
        config_json["img_h"] = serde_json::json!(img_h);
        config_json["img_w"] = serde_json::json!(img_w);
        let final_scalars = last_metrics.as_ref().map(|m| &m.scalars);
        let sf = |k: &str| final_scalars.and_then(|s| s.get(k).copied()).unwrap_or(0.0);
        let manifest = serde_json::json!({
            "framework": "flodl",
            "model": "letter",
            "mode": "async_ddp",
            "config": config_json,
            "results": {
                "letter_acc": sf("letter_acc"),
                "case_acc": sf("case_acc"),
                "letter_ce": sf("letter_ce"),
                "recon_mse": sf("recon_mse"),
            },
            "files": {
                "model": "model_final.fdl.gz",
                "dashboard": if cfg.monitor_port.is_some() { "dashboard.html" } else { "" },
            },
            "parent": null,
        });
        if let Err(e) = fs::write(
            format!("{}/manifest.json", save_dir_main),
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
            "mode": "async_ddp",
            "gpus": n_gpus,
            "config": {
                "n_scan": cfg.n_scan,
                "n_read": cfg.n_read,
                "latent_dim": cfg.latent_dim,
                "batch_size": cfg.batch_size,
                "epochs": cfg.epochs,
            },
            "ram_peak_rss_mb": rss_mb,
            "total_time_s": (total_time * 10.0).round() / 10.0,
            "avg_epoch_s": if !epoch_times.is_empty() {
                (epoch_times.iter().sum::<f64>() / epoch_times.len() as f64 * 10.0).round() / 10.0
            } else { 0.0 },
            "epoch_times_s": epoch_times,
        });
        // Per-GPU info from last metrics.
        if let Some(ref m) = last_metrics {
            let gpu_info: Vec<serde_json::Value> = m.device_indices.iter().enumerate().map(|(i, &dev)| {
                serde_json::json!({
                    "device": dev,
                    "throughput_samples_per_ms": m.per_rank_throughput.get(i).unwrap_or(&0.0),
                    "batch_share": m.per_rank_batch_share.get(i).unwrap_or(&0.0),
                })
            }).collect();
            bench["per_gpu"] = serde_json::json!(gpu_info);
        }
        let bench_path = format!("{}/benchmark.json", save_dir_main);
        if let Err(e) = fs::write(
            &bench_path,
            serde_json::to_string_pretty(&bench).unwrap_or_default(),
        ) {
            eprintln!("warning: write benchmark: {e}");
        }

        // Print summary.
        eprintln!("\n--- Benchmark ({n_gpus} GPU async DDP) ---");
        eprintln!("RAM:         {} MB peak RSS", bench["ram_peak_rss_mb"]);
        eprintln!("Avg epoch:   {}s  ({} epochs, {}s total)",
            bench["avg_epoch_s"], cfg.epochs, bench["total_time_s"]);
        if let Some(ref m) = last_metrics {
            for (i, &dev) in m.device_indices.iter().enumerate() {
                let tput = m.per_rank_throughput.get(i).unwrap_or(&0.0);
                let share = m.per_rank_batch_share.get(i).unwrap_or(&0.0);
                eprintln!("  GPU {dev}:     {:.1} samples/ms, {:.0}% share", tput, share * 100.0);
            }
        }
        eprintln!("Saved:       {bench_path}");
    }

    eprintln!("Training complete.");
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
