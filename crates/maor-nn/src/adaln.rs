use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::activation::silu;
use crate::lora::{self, LoraConfig};
use crate::timestep_embedding::PixArtAlphaCombinedTimestepSizeEmbeddings;

/// Adaptive Layer Norm Single (adaLN-single).
///
/// From PixArt-Alpha (Section 2.3). Maps timestep embeddings to
/// `embedding_coefficient * embedding_dim` scale/shift/gate parameters.
#[derive(Debug)]
pub struct AdaLayerNormSingle {
    emb: PixArtAlphaCombinedTimestepSizeEmbeddings,
    linear: Linear,
}

impl AdaLayerNormSingle {
    pub fn new(embedding_dim: usize, embedding_coefficient: usize, vb: VarBuilder) -> Result<Self> {
        Self::new_with_lora(embedding_dim, embedding_coefficient, vb, None)
    }

    pub fn new_with_lora(
        embedding_dim: usize,
        embedding_coefficient: usize,
        vb: VarBuilder,
        lora: Option<&LoraConfig>,
    ) -> Result<Self> {
        let emb_lora = lora.map(|l| l.pp("emb"));
        let linear_lora = lora.map(|l| l.pp("linear"));
        let emb = PixArtAlphaCombinedTimestepSizeEmbeddings::new_with_lora(
            embedding_dim,
            vb.pp("emb"),
            emb_lora.as_ref(),
        )?;
        let linear = lora::linear(
            embedding_dim,
            embedding_coefficient * embedding_dim,
            vb.pp("linear"),
            linear_lora.as_ref(),
        )?;
        Ok(Self { emb, linear })
    }
}

impl AdaLayerNormSingle {
    /// Forward pass: returns (scale_shift_gate, embedded_timestep).
    ///
    /// `scale_shift_gate` has shape (B, coeff * dim), typically coeff=6 for
    /// (shift1, scale1, gate1, shift2, scale2, gate2).
    pub fn forward_with_embedded(
        &self,
        timestep: &Tensor,
        hidden_dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let embedded = self.emb.forward(timestep)?;
        let activated = silu(&embedded)?;
        let out = self.linear.forward(&activated.to_dtype(hidden_dtype)?)?;
        Ok((out, embedded))
    }
}
