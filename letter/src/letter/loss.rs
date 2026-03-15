//! Loss functions for the letter model — attention guidance, fixation
//! diversity, reconstruction, and non-differentiable diagnostics.

use flodl::autograd::{Variable, grid_sample};
use flodl::nn::{mse_loss, gaussian_blur_2d};
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

    let shape = image.shape(); // [B, C, H, W]
    let h = shape[2];
    let w = shape[3];
    let blur_sigma = blur_sigma_ratio * (h.min(w) as f64);

    // Blur image to create a soft "scent field" (no gradients through the guide).
    let guide_var = gaussian_blur_2d(&Variable::new(image.data(), false), blur_sigma)?;

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

    // Stack all locations and sample in one grid_sample call.
    let loc_tensors: Vec<Tensor> = locations.iter().map(|l| l.data()).collect();
    let loc_refs: Vec<&Tensor> = loc_tensors.iter().collect();
    let stacked = Tensor::stack(&loc_refs, 1)?;           // [B, T, 2]
    let grid = stacked.unsqueeze(2)?;                     // [B, T, 1, 2]
    let sampled = img_data.grid_sample(&grid, 0, 0, true)?; // [B, 1, T, 1]

    // Single GPU→CPU transfer.
    let vals = sampled.to_f32_vec()?;
    let n = vals.len();
    if n == 0 {
        return Ok((0.0, 0.0));
    }
    let hits = vals.iter().filter(|&&v| v as f64 > threshold).count();
    let total_intensity: f64 = vals.iter().map(|&v| v as f64).sum();
    Ok((hits as f64 / n as f64, total_intensity / n as f64))
}

/// Penalizes fixations that land in empty space (void repulsion).
///
/// For each location, extracts the full patch (matching GlimpseSensor resolution)
/// and checks whether any ink is present. Penalty is active only in void — once
/// ink is found, gradient saturates to zero. This avoids the center-pull problem
/// of attention guides while still preventing fixations from sitting in empty space.
///
/// image: [B, 1, H, W] clean image.
/// locations: fixation positions.
/// patch_h, patch_w: glimpse patch size (must match GlimpseSensor).
/// threshold: ink detection threshold (0.1 typical).
/// base_grid: optional cached [1, ph, pw, 2] grid — pass None to build fresh.
pub fn void_repulsion_loss(
    image: &Variable,
    locations: &[Variable],
    patch_h: i64,
    patch_w: i64,
    threshold: f64,
) -> Result<Variable> {
    void_repulsion_with_grid(image, locations, patch_h, patch_w, threshold, None)
}

/// Void repulsion with optional cached base grid (avoids rebuilding per call).
pub fn void_repulsion_with_grid(
    image: &Variable,
    locations: &[Variable],
    patch_h: i64,
    patch_w: i64,
    threshold: f64,
    cached_grid: Option<&Tensor>,
) -> Result<Variable> {
    if locations.is_empty() {
        let device = image.data().device();
        let z = Tensor::zeros(&[1], TensorOptions { device, ..Default::default() })?;
        return Ok(Variable::new(z, false));
    }

    let base_grid = match cached_grid {
        Some(g) => g.clone(),
        None => {
            let shape = image.shape();
            build_void_grid(patch_h, patch_w, shape[2], shape[3], image.device())?
        }
    };

    let inv_threshold = 1.0 / threshold;
    let t = locations.len();
    let b = locations[0].shape()[0];

    // Batch all locations: tile image T times, stack all grids, one grid_sample.
    // image: [B, 1, H, W] → [B*T, 1, H, W]
    let img_tiled = image.repeat(&[t as i64, 1, 1, 1])?;

    // Build offset grids for all locations and concatenate in one kernel.
    let expanded = base_grid.expand(&[b, patch_h, patch_w, 2])?;
    let grid_var = Variable::new(expanded, false);
    let grids: Vec<Variable> = locations.iter()
        .map(|loc| grid_var.add(&loc.unsqueeze_many(&[1, 2]).unwrap()).unwrap())
        .collect();
    let grid_refs: Vec<&Variable> = grids.iter().collect();
    let all_grids = Variable::cat_many(&grid_refs, 0)?; // [B*T, ph, pw, 2]

    // Single grid_sample for all locations: [B*T, 1, ph, pw]
    let sampled = grid_sample(&img_tiled, &all_grids, 0, 0, true)?;

    // Max pixel per patch: [B*T, ph*pw] → max_dim(1) → [B*T]
    let flat = sampled.flatten(1, -1)?;
    let patch_max = flat.max_dim(1, false)?;

    // Saturating penalty: 0 in void, 1 when ink found.
    let saturation = patch_max.mul_scalar(inv_threshold)?.clamp(0.0, 1.0)?;

    // Negate mean: -1.0 when all patches have ink, 0.0 when all void.
    saturation.mean()?.mul_scalar(-1.0)
}

/// Build a base sampling grid for void repulsion.
/// Returns [1, patch_h, patch_w, 2].
pub fn build_void_grid(
    patch_h: i64, patch_w: i64,
    img_h: i64, img_w: i64,
    device: flodl::tensor::Device,
) -> Result<Tensor> {
    let opts = TensorOptions { device, ..Default::default() };
    let delta_h = patch_h as f64 / img_h as f64;
    let delta_w = patch_w as f64 / img_w as f64;
    let grid_y = Tensor::linspace(-delta_h, delta_h, patch_h, opts)?;
    let grid_x = Tensor::linspace(-delta_w, delta_w, patch_w, opts)?;
    let grids = Tensor::meshgrid(&[&grid_y, &grid_x])?;
    let base_grid = Tensor::stack(&[&grids[1], &grids[0]], 2)?;
    base_grid.unsqueeze(0)
}

/// Reconstruction loss: MSE between reconstructed and target.
pub fn recode_loss(recon: &Variable, target: &Variable) -> Result<Variable> {
    mse_loss(recon, target)
}
