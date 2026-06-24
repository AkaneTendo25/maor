use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::lora::{self, LoraConfig};

/// PixArt-Alpha text projection: Linear → GELU(tanh) → Linear.
///
/// Projects caption embeddings from `in_features` to `out_features`.
#[derive(Debug)]
pub struct PixArtAlphaTextProjection {
    linear_1: Linear,
    linear_2: Linear,
    use_silu: bool,
}

impl PixArtAlphaTextProjection {
    pub fn new(
        in_features: usize,
        hidden_size: usize,
        out_features: Option<usize>,
        act_fn: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_with_lora(in_features, hidden_size, out_features, act_fn, vb, None)
    }

    pub fn new_with_lora(
        in_features: usize,
        hidden_size: usize,
        out_features: Option<usize>,
        act_fn: &str,
        vb: VarBuilder,
        lora: Option<&LoraConfig>,
    ) -> Result<Self> {
        let out_features = out_features.unwrap_or(hidden_size);
        let linear_1_lora = lora.map(|l| l.pp("linear_1"));
        let linear_2_lora = lora.map(|l| l.pp("linear_2"));
        let linear_1 = lora::linear(
            in_features,
            hidden_size,
            vb.pp("linear_1"),
            linear_1_lora.as_ref(),
        )?;
        let linear_2 = lora::linear(
            hidden_size,
            out_features,
            vb.pp("linear_2"),
            linear_2_lora.as_ref(),
        )?;
        let use_silu = act_fn == "silu";
        Ok(Self {
            linear_1,
            linear_2,
            use_silu,
        })
    }
}

impl Module for PixArtAlphaTextProjection {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear_1.forward(x)?;
        let x = if self.use_silu {
            crate::activation::silu(&x)?
        } else {
            x.gelu()?
        };
        self.linear_2.forward(&x)
    }
}
