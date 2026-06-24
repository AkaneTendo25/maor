use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::activation::GELUApprox;
use crate::lora::{self, LoraConfig};

/// FeedForward network: GELUApprox(dim→inner_dim) → Linear(inner_dim→dim_out).
///
/// inner_dim = dim * mult (default mult=4).
#[derive(Debug)]
pub struct FeedForward {
    gelu_proj: GELUApprox,
    out_proj: candle_nn::Linear,
}

impl FeedForward {
    pub fn new(dim: usize, dim_out: usize, mult: usize, vb: VarBuilder) -> Result<Self> {
        Self::new_with_lora(dim, dim_out, mult, vb, None)
    }

    pub fn new_with_lora(
        dim: usize,
        dim_out: usize,
        mult: usize,
        vb: VarBuilder,
        lora: Option<&LoraConfig>,
    ) -> Result<Self> {
        let inner_dim = dim * mult;
        let gelu_lora = lora.map(|l| l.pp("net.0"));
        let out_lora = lora.map(|l| l.pp("net.2"));
        let gelu_proj =
            GELUApprox::new_with_lora(dim, inner_dim, vb.pp("net.0"), gelu_lora.as_ref())?;
        let out_proj = lora::linear(inner_dim, dim_out, vb.pp("net.2"), out_lora.as_ref())?;
        Ok(Self {
            gelu_proj,
            out_proj,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.gelu_proj.forward(x)?;
        self.out_proj.forward(&x)
    }
}
