//! VisualDecoder — reconstructs images from latent vectors.

use std::collections::HashMap;

use flodl::autograd::Variable;
use flodl::nn::{
    BatchNorm, Conv2d, ConvTranspose2d, Linear, Module, NamedInputModule, Parameter,
};
use flodl::tensor::{Device, Result};

/// Generates images from latent vectors using transposed convolutions.
///
/// FC projects to spatial feature map, then two stride-2 deconvs to output shape.
/// For the letter model: input_dim = latent_dim + 1 (latent + case label), output 128x128.
///
/// Implements `NamedInputModule`: receives "latent" and "case" via Using refs.
pub struct VisualDecoder {
    spatial_h: i64,
    spatial_w: i64,

    fc: Linear,
    deconv1: ConvTranspose2d,
    bn1: BatchNorm,
    deconv2: ConvTranspose2d,
    bn2: BatchNorm,
    conv: Conv2d,
}

impl VisualDecoder {
    /// Create a decoder for the given input dimension and output shape.
    pub fn new(input_dim: i64, output_h: i64, output_w: i64) -> Result<Self> {
        let spatial_h = output_h / 4;
        let spatial_w = output_w / 4;

        let deconv1 = ConvTranspose2d::build(
            128, 64, 3, true,
            [2, 2], [1, 1], [1, 1], [1, 1], 1, Device::CPU,
        )?;
        let deconv2 = ConvTranspose2d::build(
            64, 32, 5, true,
            [2, 2], [2, 2], [1, 1], [1, 1], 1, Device::CPU,
        )?;
        let conv = Conv2d::build(
            32, 1, 3, true,
            [1, 1], [1, 1], [1, 1], 1, Device::CPU,
        )?;

        Ok(VisualDecoder {
            spatial_h,
            spatial_w,
            fc: Linear::new(input_dim, 128 * spatial_h * spatial_w)?,
            deconv1,
            bn1: BatchNorm::new(64)?,
            deconv2,
            bn2: BatchNorm::new(32)?,
            conv,
        })
    }

    fn decode(&self, z: &Variable) -> Result<Variable> {
        let b = z.shape()[0];
        let x = self.fc.forward(z)?.relu()?;
        let x = x.reshape(&[b, 128, self.spatial_h, self.spatial_w])?;

        let x = self.deconv1.forward(&x)?;
        let x = batch_norm_2d(&x, &self.bn1)?;
        let x = x.relu()?;

        let x = self.deconv2.forward(&x)?;
        let x = batch_norm_2d(&x, &self.bn2)?;
        let x = x.relu()?;

        let x = self.conv.forward(&x)?;
        x.sigmoid()
    }
}

/// Apply 1D batch norm to a 4D tensor by reshaping.
/// [B, C, H, W] -> [B*H*W, C] -> BN -> [B, C, H, W]
fn batch_norm_2d(x: &Variable, bn: &BatchNorm) -> Result<Variable> {
    let shape = x.shape(); // [B, C, H, W]
    let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let xt = x.permute(&[0, 2, 3, 1])?; // [B, H, W, C]
    let flat = xt.reshape(&[b * h * w, c])?;
    let normed = bn.forward(&flat)?;
    let back = normed.reshape(&[b, h, w, c])?;
    back.permute(&[0, 3, 1, 2]) // [B, C, H, W]
}

impl Module for VisualDecoder {
    fn name(&self) -> &str { "visual_decoder" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.decode(input)
    }

    fn as_named_input(&self) -> Option<&dyn NamedInputModule> {
        Some(self)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.fc.parameters();
        params.extend(self.deconv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.deconv2.parameters());
        params.extend(self.bn2.parameters());
        params.extend(self.conv.parameters());
        params
    }

    fn set_training(&self, training: bool) {
        self.bn1.set_training(training);
        self.bn2.set_training(training);
    }

    fn move_to_device(&self, device: Device) {
        self.bn1.move_to_device(device);
        self.bn2.move_to_device(device);
    }
}

impl NamedInputModule for VisualDecoder {
    fn forward_named(
        &self,
        _stream: &Variable,
        refs: &HashMap<String, Variable>,
    ) -> Result<Variable> {
        let latent = refs.get("latent").expect("VisualDecoder requires 'latent' ref");
        let case_label = refs.get("case").expect("VisualDecoder requires 'case' ref");
        self.decode(&latent.cat(case_label, 1)?)
    }
}
