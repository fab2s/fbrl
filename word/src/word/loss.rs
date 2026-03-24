//! Shared loss functions for word model training phases.
//!
//! Reuses letter-level patterns: attention guide, diversity, content BCE.
//! Phase-specific losses live in their respective training modules.

use flodl::autograd::{Variable, grid_sample};
use flodl::nn::{gaussian_blur_2d, mse_loss, bce_with_logits_loss};
use flodl::tensor::{Result, Tensor, TensorOptions};

/// Attention guide loss: pull fixations toward image content.
///
/// Blurs the clean image to create a "scent trail", samples at fixation
/// locations, and returns a loss that attracts fixations toward ink.
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
    let stacked = Variable::stack(locations, 1)?;     // [B, T, 2]
    let grid = stacked.unsqueeze(2)?;                 // [B, T, 1, 2]
    let sampled = grid_sample(&guide, &grid, 0, 0, true)?;
    sampled.mean()?.mul_scalar(-1.0)
}

/// Fixation diversity loss (pairwise RBF repulsion).
///
/// Prevents fixation collapse by penalizing nearby fixation pairs.
/// `vy` scales the vertical distance component (anisotropic repulsion).
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

/// Content detection loss: BCE on whether a fixation location has ink.
pub fn content_loss(
    clean: &Variable,
    locations: &[Variable],
    logits: &[Variable],
    device: flodl::tensor::Device,
) -> Result<Variable> {
    if logits.is_empty() {
        return zero_var(device);
    }

    let mut total = zero_var(device)?;
    for (loc, logit) in locations.iter().zip(logits.iter()) {
        let grid = loc.unsqueeze(1)?.unsqueeze(2)?;
        let sampled = grid_sample(clean, &grid, 0, 0, true)?;
        let label_t = sampled.data().reshape(&[-1, 1])?
            .gt_scalar(0.1)?.to_dtype(flodl::tensor::DType::Float32)?;
        let label = Variable::new(label_t, false);
        total = total.add(&bce_with_logits_loss(logit, &label)?)?;
    }
    total.mul_scalar(1.0 / logits.len() as f64)
}

/// Reconstruction loss (MSE).
pub fn recon_loss(recon: &Variable, target: &Variable) -> Result<Variable> {
    mse_loss(recon, target)
}

/// Zero scalar on a device (utility for empty-input loss returns).
pub fn zero_var(device: flodl::tensor::Device) -> Result<Variable> {
    let z = Tensor::zeros(&[1], TensorOptions { device, ..Default::default() })?;
    Ok(Variable::new(z, false))
}
