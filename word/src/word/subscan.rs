//! SubScan: bounded-region letter localization.
//!
//! Given a region of a word image (defined by center + half-width), SubScan
//! takes two short, wide, blurred glimpses within the bounded region and
//! infers a letter center position for handoff to the letter model.
//!
//! The output position is free — it does not need to coincide with either
//! glimpse location. SubScan observes partial ink structure and infers
//! where the letter center must be.
//!
//! ## Region bounding
//!
//! SubScan's location head is reparameterized to stay within its region:
//! ```text
//! x = region_center + region_half_width * tanh(raw_x)
//! y = tanh(raw_y)   // full vertical range
//! ```
//!
//! ## Glimpse design
//!
//! - **Short and wide** (~8x28 pixels): horizontal localization emphasis
//! - **2/3 letter width**: one glimpse cannot see the whole letter
//! - **Blurred**: gaussian blur on the full image before glimpse extraction
//! - **Two free glimpses**: minimum for triangulating the letter center

use std::cell::RefCell;

use flodl::autograd::Variable;
use flodl::nn::{gaussian_blur_2d, Linear, Module, Parameter};
use flodl::tensor::{Device, Result, Tensor, TensorOptions};

use super::glimpse::GlimpseSensor;

/// SubScan configuration.
pub struct SubScanConfig {
    /// Hidden/latent dimension for GRU and sensor output.
    pub hidden_dim: i64,
    /// Glimpse patch height (short: ~8 pixels).
    pub patch_h: i64,
    /// Glimpse patch width (wide: ~28 pixels, ~2/3 letter width).
    pub patch_w: i64,
    /// Number of resolution scales in the glimpse sensor.
    pub n_scales: usize,
    /// Number of glimpse steps (default: 2).
    pub n_glimpses: usize,
    /// Gaussian blur sigma applied to the full image before glimpse extraction.
    /// Ensures SubScan sees ink density, not letterform detail.
    pub blur_sigma: f64,
}

/// SubScan: bounded-region letter localization module.
///
/// Two short, wide, blurred glimpses within a bounded region, then infer
/// the letter center position. The output position does not need to
/// coincide with either glimpse location — it is an inference from
/// accumulated evidence.
///
/// # Forward flow
///
/// ```text
/// h = h0, loc = (region_center, 0)
/// for each glimpse:
///     glimpse = sensor(blur(image), loc)
///     h = GRU(glimpse, h)
///     loc = (region_center + region_half_w * tanh(raw_x), tanh(raw_y))
/// return loc
/// ```
pub struct SubScan {
    sensor: GlimpseSensor,
    gru: flodl::GRUCell,
    loc_head: Linear,
    h0: Parameter,
    hidden_dim: i64,
    n_glimpses: usize,
    blur_sigma: f64,
    /// Location traces from the most recent forward pass.
    locations: RefCell<Vec<Variable>>,
}

impl SubScan {
    pub fn new(cfg: &SubScanConfig) -> Result<Self> {
        let sensor = GlimpseSensor::new(
            cfg.patch_h, cfg.patch_w, cfg.n_scales, cfg.hidden_dim,
        )?;
        let gru = flodl::GRUCell::new(cfg.hidden_dim, cfg.hidden_dim)?;
        let loc_head = Linear::new(cfg.hidden_dim, 2)?;
        let h0_data = Tensor::zeros(&[1, cfg.hidden_dim], Default::default())?;
        let h0 = Parameter {
            variable: Variable::new(h0_data, true),
            name: "h0".into(),
        };

        Ok(SubScan {
            sensor,
            gru,
            loc_head,
            h0,
            hidden_dim: cfg.hidden_dim,
            n_glimpses: cfg.n_glimpses,
            blur_sigma: cfg.blur_sigma,
            locations: RefCell::new(Vec::new()),
        })
    }

    /// Localize a letter within a bounded region.
    ///
    /// - `image`: `[B, 1, H, W]` — full word image.
    /// - `region_center`: `[B, 1]` — normalized x-center of the region in `[-1, 1]`.
    /// - `region_half_w`: `[B, 1]` — normalized half-width of the region.
    ///
    /// Returns `[B, 2]` — inferred letter center position, x bounded to region, y free.
    pub fn forward(
        &self,
        image: &Variable,
        region_center: &Variable,
        region_half_w: &Variable,
    ) -> Result<Variable> {
        let batch = image.shape()[0];
        let device = image.device();
        let opts = TensorOptions { device, ..Default::default() };

        // Blur the image — SubScan sees density, not letterforms.
        let blurred = gaussian_blur_2d(image, self.blur_sigma)?;

        // Initialize hidden state from learned h0.
        let mut h = self.h0.variable.reshape(&[1, self.hidden_dim])?
            .repeat(&[batch, 1])?;

        // Start at region center.
        let y_zero = Variable::new(Tensor::zeros(&[batch, 1], opts)?, false);
        let mut loc = region_center.cat(&y_zero, 1)?; // [B, 2]

        let mut locs = Vec::with_capacity(self.n_glimpses);

        for _step in 0..self.n_glimpses {
            // Extract blurred glimpse at current location.
            let glimpse = self.sensor.sense(&blurred, &loc)?;

            // GRU integrates the partial view.
            h = self.gru.forward_step(&glimpse, Some(&h))?;

            // Update location (bounded to region).
            let raw = self.loc_head.forward(&h)?.tanh()?; // [B, 2] in [-1, 1]
            let raw_x = raw.narrow(1, 0, 1)?;             // [B, 1]
            let raw_y = raw.narrow(1, 1, 1)?;             // [B, 1]
            let x = region_center.add(&region_half_w.mul(&raw_x)?)?;
            loc = x.cat(&raw_y, 1)?;                      // [B, 2]

            locs.push(loc.clone());
        }

        *self.locations.borrow_mut() = locs;
        Ok(loc)
    }

    /// Location traces from the most recent forward pass (one per glimpse step).
    pub fn locations(&self) -> Vec<Variable> {
        self.locations.borrow().clone()
    }

    /// All learnable parameters.
    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.h0.clone()];
        params.extend(self.sensor.parameters());
        params.extend(self.gru.parameters());
        params.extend(self.loc_head.parameters());
        params
    }

    /// Clear per-forward state.
    pub fn reset(&self) {
        self.locations.borrow_mut().clear();
    }

    /// Move all parameters to a device.
    pub fn set_device(&self, device: Device) {
        // Move parameters using the same pattern as Graph::set_device:
        // detach, move to device, set_data.
        for p in self.parameters() {
            if p.variable.data().device() != device {
                if let Ok(t) = p.variable.data().detach()
                    .and_then(|d| d.to_device(device))
                {
                    p.variable.set_data(t);
                }
            }
        }
        // Move non-parameter state (BatchNorm running stats, if any).
        self.sensor.move_to_device(device);
        self.gru.move_to_device(device);
        self.loc_head.move_to_device(device);
    }

    /// Set training/eval mode (affects sensor internals).
    pub fn set_training(&self, training: bool) {
        self.sensor.set_training(training);
    }
}
