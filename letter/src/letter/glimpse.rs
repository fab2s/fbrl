//! Foveal attention sensor — extracts multi-resolution patches via grid_sample.

use std::collections::HashMap;

use flodl::autograd::{Variable, grid_sample};
use flodl::nn::{Conv2d, Linear, Module, NamedInputModule, Parameter};
use flodl::tensor::{Device, Result, Tensor, TensorOptions};

/// Extracts a small patch from the image at a given (x,y) location,
/// encodes it through a CNN, fuses with the location embedding, and returns
/// a single latent vector representing "what + where".
///
/// This is the foveal attention core: the model only sees what it looks at.
pub struct GlimpseSensor {
    patch_h: i64,
    patch_w: i64,
    scales: Vec<usize>,

    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    glimpse_fc: Linear,
    location_fc: Linear,
    combine_fc: Linear,
}

impl GlimpseSensor {
    /// Create a sensor with the given patch size and scale count.
    pub fn new(
        patch_h: i64, patch_w: i64, n_scales: usize, latent_dim: i64,
    ) -> Result<Self> {
        let scales: Vec<usize> = (0..n_scales).map(|i| 1 << i).collect();

        let conv1 = Conv2d::build(
            n_scales as i64, 32, 3, true,
            [1, 1], [1, 1], [1, 1], 1, Device::CPU,
        )?;
        let conv2 = Conv2d::build(
            32, 64, 3, true,
            [2, 2], [1, 1], [1, 1], 1, Device::CPU,
        )?;
        let conv3 = Conv2d::build(
            64, 128, 3, true,
            [2, 2], [1, 1], [1, 1], 1, Device::CPU,
        )?;

        Ok(GlimpseSensor {
            patch_h,
            patch_w,
            scales,
            conv1,
            conv2,
            conv3,
            glimpse_fc: Linear::new(128, latent_dim)?,
            location_fc: Linear::new(2, 128)?,
            combine_fc: Linear::new(latent_dim + 128, latent_dim)?,
        })
    }

    /// Extract, encode, and fuse.
    /// image: [B, C, H, W], location: [B, 2] in [-1, 1].
    /// Returns: [B, latent_dim].
    fn sense(&self, image: &Variable, location: &Variable) -> Result<Variable> {
        let img_shape = image.shape();
        let h = img_shape[2];
        let w = img_shape[3];
        let dev = image.device();

        // Extract one patch per scale via differentiable grid_sample.
        let mut combined: Option<Variable> = None;
        for &scale in &self.scales {
            let grid = make_grid(location, scale, self.patch_h, self.patch_w, h, w, dev)?;
            let patch = grid_sample(image, &grid, 0, 0, true)?;
            combined = Some(match combined {
                None => patch,
                Some(prev) => prev.cat(&patch, 1)?,
            });
        }
        let combined = combined.unwrap();

        // CNN -> pool -> flatten
        let feat = self.conv1.forward(&combined)?.relu()?;
        let feat = self.conv2.forward(&feat)?.relu()?;
        let feat = self.conv3.forward(&feat)?.relu()?;
        let feat = flodl::adaptive_avg_pool2d(&feat, [1, 1])?;
        let feat = feat.flatten(1, -1)?;

        // Fuse "what I see" + "where I am"
        let glimpse_feat = self.glimpse_fc.forward(&feat)?.relu()?;
        let loc_feat = self.location_fc.forward(location)?.relu()?;
        let fused = glimpse_feat.cat(&loc_feat, 1)?;
        self.combine_fc.forward(&fused)?.relu()
    }
}

impl Module for GlimpseSensor {
    fn name(&self) -> &str { "glimpse_sensor" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        // Shouldn't be called directly — use as_named_input
        self.sense(input, input)
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> {
        Some(self)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.conv1.parameters();
        params.extend(self.conv2.parameters());
        params.extend(self.conv3.parameters());
        params.extend(self.glimpse_fc.parameters());
        params.extend(self.location_fc.parameters());
        params.extend(self.combine_fc.parameters());
        params
    }
}

impl NamedInputModule for GlimpseSensor {
    fn forward_named(
        &self,
        image: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        let location = refs.get("location").expect("GlimpseSensor requires 'location' ref");
        self.sense(image, location)
    }
}

/// Build a sampling grid for grid_sample, centered on location.
/// Returns a Variable of shape [B, patch_h, patch_w, 2].
fn make_grid(
    location: &Variable,
    scale: usize,
    patch_h: i64,
    patch_w: i64,
    img_h: i64,
    img_w: i64,
    device: Device,
) -> Result<Variable> {
    let b = location.shape()[0];
    let opts = TensorOptions { device, ..Default::default() };

    // Normalized extent in [-1, 1] coords.
    let delta_h = (scale as f64) * (patch_h as f64) / (img_h as f64);
    let delta_w = (scale as f64) * (patch_w as f64) / (img_w as f64);

    // Build centered grid on target device (no gradient needed for the base grid).
    let grid_y = Tensor::linspace(-delta_h, delta_h, patch_h, opts)?;
    let grid_x = Tensor::linspace(-delta_w, delta_w, patch_w, opts)?;

    // Meshgrid + stack → [patch_h, patch_w, 2] as (x, y).
    let grids = Tensor::meshgrid(&[&grid_y, &grid_x])?;
    let base_grid = Tensor::stack(&[&grids[1], &grids[0]], 2)?;
    let base_grid = base_grid.unsqueeze(0)?;          // [1, patch_h, patch_w, 2]
    let base_grid = base_grid.expand(&[b, patch_h, patch_w, 2])?;
    let grid_var = Variable::new(base_grid, false);

    // Shift by location: [B, 2] -> [B, 1, 1, 2] for broadcast.
    let loc_reshaped = location.unsqueeze_many(&[1, 2])?;
    grid_var.add(&loc_reshaped)
}
