//! Loss functions for the letter model — attention guidance, fixation
//! diversity, reconstruction, and non-differentiable diagnostics.

use flodl::autograd::{Variable, grid_sample};
use flodl::nn::mse_loss;
use flodl::tensor::{Result, Tensor, TensorOptions};

/// Guides fixation locations toward image content.
///
/// Blurs the image to create a soft "scent field" around strokes, then samples
/// this field at each fixation point. Minimizing the loss pushes fixations
/// onto high-content regions.
///
/// image: [B, 1, H, W] clean image (unnoised).
/// locations: learned fixation positions (all are sampled — caller skips any init).
/// blur_sigma_ratio: Gaussian sigma as fraction of min(H,W). 0.16 is the default.
pub fn attention_guide_loss(
    image: &Variable,
    locations: &[Variable],
    blur_sigma_ratio: f64,
) -> Result<Variable> {
    if locations.is_empty() {
        let device = image.data().device();
        let z = Tensor::zeros(&[1], TensorOptions { device, ..Default::default() })?;
        return Ok(Variable::new(z, false));
    }

    let img_data = image.data();
    let shape = img_data.shape(); // [B, C, H, W]
    let h = shape[2];
    let w = shape[3];
    let device = img_data.device();

    let blur_sigma = blur_sigma_ratio * (h.min(w) as f64);

    // Build 1D Gaussian kernel.
    let k = (4.0 * blur_sigma) as usize | 1;
    let half_k = k / 2;
    let mut kernel_data = vec![0.0f32; k];
    let mut k_sum = 0.0f64;
    for (i, val) in kernel_data.iter_mut().enumerate() {
        let x = (i as f64) - (half_k as f64);
        let v = (-x * x / (2.0 * blur_sigma * blur_sigma)).exp();
        *val = v as f32;
        k_sum += v;
    }
    for val in &mut kernel_data {
        *val /= k_sum as f32;
    }

    // Separable blur: convolve rows then columns.
    let gauss = Tensor::from_f32(&kernel_data, &[k as i64], device)?;
    let kernel_h = gauss.reshape(&[1, 1, k as i64, 1])?;
    let kernel_w = gauss.reshape(&[1, 1, 1, k as i64])?;

    let guide = img_data.conv2d(
        &kernel_h, None,
        [1, 1], [half_k as i64, 0], [1, 1], 1,
    )?;
    let guide = guide.conv2d(
        &kernel_w, None,
        [1, 1], [0, half_k as i64], [1, 1], 1,
    )?;
    let guide_var = Variable::new(guide, false);

    // Stack all fixations and sample in one grid_sample call.
    let stacked = Variable::stack(locations, 1)?;     // [B, T, 2]
    let grid = stacked.unsqueeze(2)?;                 // [B, T, 1, 2]
    let sampled = grid_sample(&guide_var, &grid, 0, 0, true)?; // [B, 1, T, 1]

    // Negate: maximize guide values → minimize negative.
    sampled.mean()?.mul_scalar(-1.0)
}

/// Pairwise Gaussian RBF repulsion between fixation points.
/// Fixations closer than ~sigma repel each other, encouraging spatial spread.
///
/// locations: learned fixation positions (caller skips any init).
/// sigma: repulsion radius in [-1,1] coords. 0.1 means ~10% of image width.
/// vy: vertical scale factor. vy > 1.0 penalizes vertical clustering more.
pub fn fixation_diversity_loss(
    locations: &[Variable],
    sigma: f64,
    vy: f64,
) -> Result<Variable> {
    let locs = locations;
    let t = locs.len();
    if t < 2 {
        if !locs.is_empty() {
            let device = locs[0].data().device();
            let z = Tensor::zeros(&[1], TensorOptions { device, ..Default::default() })?;
            return Ok(Variable::new(z, false));
        }
        let z = Tensor::zeros(&[1], Default::default())?;
        return Ok(Variable::new(z, false));
    }

    let inv_two_sigma_sq = -1.0 / (2.0 * sigma * sigma);
    let vy_squared = vy * vy;

    // Stack locations: [B, 2] × T → [B, T, 2]
    let stacked = Variable::stack(locs, 1)?;
    // Broadcast pairwise differences: [B, T, 1, 2] - [B, 1, T, 2] → [B, T, T, 2]
    let diff = stacked.unsqueeze(2)?.sub(&stacked.unsqueeze(1)?)?;
    let dx = diff.select(3, 0)?;  // [B, T, T]
    let dy = diff.select(3, 1)?;  // [B, T, T]

    let dist_sq = if (vy_squared - 1.0).abs() > 1e-8 {
        dx.pow_scalar(2.0)?.add(&dy.pow_scalar(2.0)?.mul_scalar(vy_squared)?)?
    } else {
        dx.pow_scalar(2.0)?.add(&dy.pow_scalar(2.0)?)?
    };

    // RBF repulsion with upper-triangle mask (excludes diagonal and duplicates).
    let rbf = dist_sq.mul_scalar(inv_two_sigma_sq)?.exp()?; // [B, T, T]
    let masked = rbf.triu(1)?;                               // [B, T, T]

    // mean over [B,T,T] counts all T*T elements; ×2 recovers full pairwise sum.
    masked.mean()?.mul_scalar(2.0)
}

/// Measures what fraction of fixations land on actual letter pixels.
/// Returns (hit_rate, mean_intensity). Both in [0, 1].
///
/// image: [B, 1, H, W] clean image (sharp, unblurred).
/// locations: learned fixation positions (caller skips any init).
/// threshold: intensity above which a sample counts as a hit. 0.3 is typical.
pub fn fixation_hit_rate(
    image: &Variable,
    locations: &[Variable],
    threshold: f64,
) -> Result<(f64, f64)> {
    if locations.is_empty() {
        return Ok((0.0, 0.0));
    }

    let img_data = image.data();
    let mut hits = 0usize;
    let mut total_intensity = 0.0f64;
    let mut n = 0usize;

    for loc in locations {
        let grid = loc.data().unsqueeze(1)?.unsqueeze(2)?;
        let sampled = img_data.grid_sample(&grid, 0, 0, true)?;
        let vals = sampled.to_f32_vec()?;
        for v in &vals {
            total_intensity += *v as f64;
            if *v as f64 > threshold {
                hits += 1;
            }
            n += 1;
        }
    }

    if n == 0 {
        return Ok((0.0, 0.0));
    }
    Ok((hits as f64 / n as f64, total_intensity / n as f64))
}

/// Reconstruction loss: MSE between reconstructed and target.
pub fn recode_loss(recon: &Variable, target: &Variable) -> Result<Variable> {
    mse_loss(recon, target)
}
