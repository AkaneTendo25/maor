use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::lora::{self, LoraConfig};

/// GELU with tanh approximation followed by linear projection.
///
/// Linear(dim_in, dim_out) followed by GELU with tanh approximation.
#[derive(Debug)]
pub struct GELUApprox {
    proj: Linear,
}

impl GELUApprox {
    pub fn new(dim_in: usize, dim_out: usize, vb: VarBuilder) -> Result<Self> {
        Self::new_with_lora(dim_in, dim_out, vb, None)
    }

    pub fn new_with_lora(
        dim_in: usize,
        dim_out: usize,
        vb: VarBuilder,
        lora: Option<&LoraConfig>,
    ) -> Result<Self> {
        let proj_lora = lora.map(|l| l.pp("proj"));
        let proj = lora::linear(dim_in, dim_out, vb.pp("proj"), proj_lora.as_ref())?;
        Ok(Self { proj })
    }
}

impl Module for GELUApprox {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.proj.forward(x)?.gelu()
    }
}

/// SiLU activation (Sigmoid Linear Unit): `x * sigmoid(x)`.
///
/// A free function (not a struct) because SiLU has no learnable parameters.
/// Compare with [`GELUApprox`] which wraps a Linear projection layer.
pub fn silu(x: &Tensor) -> Result<Tensor> {
    let sigmoid = candle_nn::ops::sigmoid(x)?;
    x * sigmoid
}
