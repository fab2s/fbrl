//! CrossAttentionReadout — position tokens attend over grouped read states.
//!
//! 4 learned position query tokens, each attending to its group of read states
//! (grouped by read_group_boundaries). Produces [B, n_positions, latent_dim]
//! for per-position classification.

use flodl::autograd::Variable;
use flodl::nn::{Linear, Module, Parameter};
use flodl::tensor::{Device, Result, Tensor};

/// Cross-attention readout with grouped masking.
///
/// Each position token attends only to its group of read states.
/// With n_positions=4 and n_read=12, each group has 3 reads.
pub struct CrossAttentionReadout {
    query_tokens: Parameter,  // [n_positions, latent_dim]
    query_proj: Linear,
    key_proj: Linear,
    value_proj: Linear,
    out_proj: Linear,
    n_positions: usize,
    scale: f64,
}

impl CrossAttentionReadout {
    pub fn new(latent_dim: i64, n_positions: usize) -> Result<Self> {
        // Initialize query tokens with spatial x-bias.
        let mut init_data = vec![0.0f32; n_positions * latent_dim as usize];
        for p in 0..n_positions {
            // First dim gets a position-dependent bias.
            let x_bias = -0.75 + 1.5 * (p as f32) / (n_positions as f32 - 1.0);
            init_data[p * latent_dim as usize] = x_bias;
        }
        let qt = Tensor::from_f32(
            &init_data,
            &[n_positions as i64, latent_dim],
            Device::CPU,
        )?;

        let scale = 1.0 / (latent_dim as f64).sqrt();

        Ok(CrossAttentionReadout {
            query_tokens: Parameter {
                variable: Variable::new(qt, true),
                name: "query_tokens".into(),
            },
            query_proj: Linear::new(latent_dim, latent_dim)?,
            key_proj: Linear::new(latent_dim, latent_dim)?,
            value_proj: Linear::new(latent_dim, latent_dim)?,
            out_proj: Linear::new(latent_dim, latent_dim)?,
            n_positions,
            scale,
        })
    }

    /// Forward pass.
    ///
    /// read_states: [B, T, latent_dim] — stacked read hidden states.
    /// group_boundaries: [0, 3, 6, 9] — start index of each group in T dimension.
    ///
    /// Returns [B, n_positions, latent_dim].
    pub fn forward(
        &self,
        read_states: &Variable,
        group_boundaries: &[usize],
    ) -> Result<Variable> {
        let shape = read_states.shape();
        let b = shape[0];
        let t = shape[1];
        let np = self.n_positions as i64;

        // Expand query tokens to batch: [n_pos, D] → [B, n_pos, D]
        let q = self.query_tokens.variable
            .unsqueeze(0)?.expand(&[b, np, shape[2]])?;
        let q = self.query_proj.forward(&q)?;        // [B, n_pos, D]
        let k = self.key_proj.forward(read_states)?;  // [B, T, D]
        let v = self.value_proj.forward(read_states)?; // [B, T, D]

        // Attention scores: [B, n_pos, T]
        let k_t = k.transpose(1, 2)?;                 // [B, D, T]
        let attn = q.matmul(&k_t)?.mul_scalar(self.scale)?;

        // Group mask: position p attends only to reads in its group.
        let mask = self.build_group_mask(t, group_boundaries, read_states.device())?;
        let attn = attn.add(&Variable::new(mask, false))?;

        // Softmax over T dimension.
        let attn = attn.softmax(2)?;

        // Weighted sum: [B, n_pos, D]
        let out = attn.matmul(&v)?;
        self.out_proj.forward(&out)
    }

    /// Build a [n_pos, T] mask: 0.0 for allowed, -1e9 for blocked.
    fn build_group_mask(
        &self,
        t: i64,
        boundaries: &[usize],
        device: Device,
    ) -> Result<Tensor> {
        let np = self.n_positions;
        // Start with all blocked.
        let mut mask_data = vec![-1e9f32; np * t as usize];

        // Unblock each group's segment.
        for p in 0..np {
            let start = boundaries[p];
            let end = if p + 1 < boundaries.len() {
                boundaries[p + 1]
            } else {
                t as usize
            };
            for j in start..end {
                mask_data[p * t as usize + j] = 0.0;
            }
        }

        Tensor::from_f32(&mask_data, &[np as i64, t], device)
    }

    pub fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.query_tokens.clone()];
        params.extend(self.query_proj.parameters());
        params.extend(self.key_proj.parameters());
        params.extend(self.value_proj.parameters());
        params.extend(self.out_proj.parameters());
        params
    }

    pub fn move_to_device(&self, _device: Device) {
        // Parameters are moved by the optimizer/graph — nothing extra here.
    }
}
