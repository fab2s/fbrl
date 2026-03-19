//! Loss functions for word model training.
//!
//! Reuses letter-level losses (attention guide, diversity, content BCE)
//! plus word-specific scaffold and isolation losses.

use flodl::autograd::{Variable, grid_sample};
use flodl::nn::{gaussian_blur_2d, mse_loss, bce_with_logits_loss};
use flodl::tensor::{Result, Tensor, TensorOptions};

/// Scan attention guide: pull scan fixations toward image content (full image).
///
/// Same as letter's attention_guide_loss.
pub fn scan_guide_loss(
    image: &Variable,
    locations: &[Variable],
    blur_sigma_ratio: f64,
) -> Result<Variable> {
    attention_guide_loss(image, locations, blur_sigma_ratio)
}

/// Read attention guide with temporal scaffold.
///
/// During scaffold phase: each read group is guided toward its position's
/// vertical stripe. As scaffold_weight → 0, guide covers the full image.
///
/// image: [B, 1, H, W] clean image.
/// locations: read fixation positions.
/// group_boundaries: [0, 3, 6, 9] — start of each group.
/// n_positions: 4.
/// scaffold_weight: 1.0 → 0.0 over training.
pub fn read_scaffold_loss(
    image: &Variable,
    locations: &[Variable],
    blur_sigma_ratio: f64,
    group_boundaries: &[usize],
    n_positions: usize,
    scaffold_weight: f64,
) -> Result<Variable> {
    if locations.is_empty() {
        return zero_var(image.device());
    }

    let shape = image.shape();
    let h = shape[2];
    let w = shape[3];
    let blur_sigma = blur_sigma_ratio * (h.min(w) as f64);

    // Full-image guide (no scaffold).
    let full_guide = gaussian_blur_2d(&Variable::new(image.data(), false), blur_sigma)?;

    if scaffold_weight < 1e-6 {
        // No scaffold — just use full guide for all reads.
        return attention_guide_loss_with_guide(&full_guide, locations);
    }

    let stripe_w = w / n_positions as i64;
    let device = image.device();

    let mut total = zero_var(device)?;
    let n_locs = locations.len() as f64;

    for (t, loc) in locations.iter().enumerate() {
        // Assign read to position group.
        let pos = group_boundaries.iter()
            .rposition(|&start| t >= start)
            .unwrap_or(0);

        // Build stripe-masked guide for this position.
        let stripe_start = (pos as i64) * stripe_w;
        let stripe_end = stripe_start + stripe_w;

        // Stripe guide: zero outside the stripe, blurred inside.
        let stripe_guide = mask_stripe(&full_guide, stripe_start, stripe_end, w)?;

        // Blend: scaffold_weight * stripe + (1 - scaffold_weight) * full.
        let blended = stripe_guide.mul_scalar(scaffold_weight)?
            .add(&full_guide.mul_scalar(1.0 - scaffold_weight)?)?;

        // Sample guide at fixation location.
        let grid = loc.unsqueeze(1)?.unsqueeze(2)?; // [B, 1, 1, 2]
        let sampled = grid_sample(&blended, &grid, 0, 0, true)?;
        total = total.sub(&sampled.mean()?)?; // Negate: maximize guide value.
    }

    total.mul_scalar(1.0 / n_locs)
}

/// Mask a guide image to a vertical stripe [x_start, x_end).
fn mask_stripe(
    guide: &Variable,
    x_start: i64,
    x_end: i64,
    w: i64,
) -> Result<Variable> {
    let shape = guide.shape();
    let b = shape[0];
    let c = shape[1];
    let h = shape[2];

    // Build a [1, 1, 1, W] mask: 1.0 inside stripe, 0.0 outside.
    let mut mask_data = vec![0.0f32; w as usize];
    for x in x_start..x_end {
        mask_data[x as usize] = 1.0;
    }
    let mask = Tensor::from_f32(&mask_data, &[1, 1, 1, w], guide.device())?;
    let mask_var = Variable::new(mask, false);

    guide.mul(&mask_var)
}

/// Attention guide loss (full image, no stripe masking).
pub fn attention_guide_loss(
    image: &Variable,
    locations: &[Variable],
    blur_sigma_ratio: f64,
) -> Result<Variable> {
    if locations.is_empty() {
        return zero_var(image.device());
    }

    let shape = image.shape();
    let h = shape[2];
    let w = shape[3];
    let blur_sigma = blur_sigma_ratio * (h.min(w) as f64);

    let guide = gaussian_blur_2d(&Variable::new(image.data(), false), blur_sigma)?;
    attention_guide_loss_with_guide(&guide, locations)
}

fn attention_guide_loss_with_guide(
    guide: &Variable,
    locations: &[Variable],
) -> Result<Variable> {
    let stacked = Variable::stack(locations, 1)?;     // [B, T, 2]
    let grid = stacked.unsqueeze(2)?;                 // [B, T, 1, 2]
    let sampled = grid_sample(guide, &grid, 0, 0, true)?;
    sampled.mean()?.mul_scalar(-1.0)
}

/// Fixation diversity loss (pairwise RBF repulsion).
pub fn fixation_diversity_loss(
    locations: &[Variable],
    sigma: f64,
    vy: f64,
) -> Result<Variable> {
    let t = locations.len();
    if t < 2 {
        return if !locations.is_empty() {
            zero_var(locations[0].device())
        } else {
            Ok(Variable::new(Tensor::zeros(&[1], Default::default())?, false))
        };
    }

    let inv_two_sigma_sq = -1.0 / (2.0 * sigma * sigma);
    let vy_squared = vy * vy;

    let stacked = Variable::stack(locations, 1)?;
    let diff = stacked.unsqueeze(2)?.sub(&stacked.unsqueeze(1)?)?;
    let dx = diff.select(3, 0)?;
    let dy = diff.select(3, 1)?;

    let dist_sq = if (vy_squared - 1.0).abs() > 1e-8 {
        dx.pow_scalar(2.0)?.add(&dy.pow_scalar(2.0)?.mul_scalar(vy_squared)?)?
    } else {
        dx.pow_scalar(2.0)?.add(&dy.pow_scalar(2.0)?)?
    };

    let rbf = dist_sq.mul_scalar(inv_two_sigma_sq)?.exp()?;
    let masked = rbf.triu(1)?;
    masked.mean()?.mul_scalar(2.0)
}

/// Content detection loss: BCE on whether scan location has ink.
pub fn content_loss(
    clean: &Variable,
    scan_locations: &[Variable],
    scan_logits: &[Variable],
    device: flodl::tensor::Device,
) -> Result<Variable> {
    if scan_logits.is_empty() {
        return zero_var(device);
    }

    let mut total = zero_var(device)?;
    for (loc, logit) in scan_locations.iter().zip(scan_logits.iter()) {
        let grid = loc.unsqueeze(1)?.unsqueeze(2)?;
        let sampled = grid_sample(clean, &grid, 0, 0, true)?;
        let label_t = sampled.data().reshape(&[-1, 1])?
            .gt_scalar(0.1)?.to_dtype(flodl::tensor::DType::Float32)?;
        let label = Variable::new(label_t, false);
        total = total.add(&bce_with_logits_loss(logit, &label)?)?;
    }
    total.mul_scalar(1.0 / scan_logits.len() as f64)
}

/// Reconstruction loss.
pub fn recon_loss(recon: &Variable, target: &Variable) -> Result<Variable> {
    mse_loss(recon, target)
}

fn zero_var(device: flodl::tensor::Device) -> Result<Variable> {
    let z = Tensor::zeros(&[1], TensorOptions { device, ..Default::default() })?;
    Ok(Variable::new(z, false))
}
