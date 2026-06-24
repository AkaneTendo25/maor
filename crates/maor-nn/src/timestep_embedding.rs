use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::activation::silu;
use crate::lora::{self, LoraConfig};

/// Sinusoidal timestep embedding.
///
/// Creates [N, embedding_dim] embeddings from [N] timesteps using sin/cos frequencies.
pub fn get_timestep_embedding(
    timesteps: &Tensor,
    embedding_dim: usize,
    flip_sin_to_cos: bool,
    downscale_freq_shift: f64,
    scale: f64,
    max_period: usize,
) -> Result<Tensor> {
    let half_dim = embedding_dim / 2;
    let device = timesteps.device();

    // exponent = -log(max_period) * arange(0, half_dim) / (half_dim - downscale_freq_shift)
    let log_max = -(max_period as f64).ln();
    let indices: Vec<f32> = (0..half_dim).map(|i| i as f32).collect();
    let exponent = Tensor::from_vec(indices, half_dim, device)?;
    let exponent = (exponent * (log_max / (half_dim as f64 - downscale_freq_shift)))?;
    let emb = exponent.exp()?;

    // timesteps[:, None] * emb[None, :]
    let ts = timesteps.to_dtype(DType::F32)?.unsqueeze(1)?;
    let emb = emb.unsqueeze(0)?;
    let emb = ts.broadcast_mul(&emb)?;

    // Scale
    let emb = if scale != 1.0 { (emb * scale)? } else { emb };

    // Concat sin and cos
    let sin_emb = emb.sin()?;
    let cos_emb = emb.cos()?;
    let emb = if flip_sin_to_cos {
        Tensor::cat(&[&cos_emb, &sin_emb], candle_core::D::Minus1)?
    } else {
        Tensor::cat(&[&sin_emb, &cos_emb], candle_core::D::Minus1)?
    };

    // Zero pad if odd dimension
    if embedding_dim % 2 == 1 {
        let (n, _d) = emb.dims2()?;
        let pad = Tensor::zeros((n, 1), DType::F32, device)?;
        Tensor::cat(&[&emb, &pad], candle_core::D::Minus1)
    } else {
        Ok(emb)
    }
}

/// Timesteps module: wraps get_timestep_embedding with fixed parameters.
#[derive(Debug, Clone)]
pub struct Timesteps {
    pub num_channels: usize,
    pub flip_sin_to_cos: bool,
    pub downscale_freq_shift: f64,
    pub scale: f64,
}

impl Timesteps {
    pub fn new(
        num_channels: usize,
        flip_sin_to_cos: bool,
        downscale_freq_shift: f64,
        scale: f64,
    ) -> Self {
        Self {
            num_channels,
            flip_sin_to_cos,
            downscale_freq_shift,
            scale,
        }
    }
}

impl Module for Timesteps {
    fn forward(&self, timesteps: &Tensor) -> Result<Tensor> {
        get_timestep_embedding(
            timesteps,
            self.num_channels,
            self.flip_sin_to_cos,
            self.downscale_freq_shift,
            self.scale,
            10000,
        )
    }
}

/// TimestepEmbedding: Linear → SiLU → Linear.
///
/// Projects sinusoidal timestep embeddings to model dimension.
#[derive(Debug)]
pub struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    pub fn new(
        in_channels: usize,
        time_embed_dim: usize,
        out_dim: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_with_lora(in_channels, time_embed_dim, out_dim, vb, None)
    }

    pub fn new_with_lora(
        in_channels: usize,
        time_embed_dim: usize,
        out_dim: Option<usize>,
        vb: VarBuilder,
        lora: Option<&LoraConfig>,
    ) -> Result<Self> {
        let out_dim = out_dim.unwrap_or(time_embed_dim);
        let linear_1_lora = lora.map(|l| l.pp("linear_1"));
        let linear_2_lora = lora.map(|l| l.pp("linear_2"));
        let linear_1 = lora::linear(
            in_channels,
            time_embed_dim,
            vb.pp("linear_1"),
            linear_1_lora.as_ref(),
        )?;
        let linear_2 = lora::linear(
            time_embed_dim,
            out_dim,
            vb.pp("linear_2"),
            linear_2_lora.as_ref(),
        )?;
        Ok(Self { linear_1, linear_2 })
    }
}

impl Module for TimestepEmbedding {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Sinusoidal embeddings are built in F32; cast to the weight dtype
        // (e.g. bf16) before the matmul.
        let x = x.to_dtype(self.linear_1.weight().dtype())?;
        let x = self.linear_1.forward(&x)?;
        let x = silu(&x)?;
        self.linear_2.forward(&x)
    }
}

/// PixArtAlpha combined timestep + size embeddings.
///
/// Timesteps(256, flip=true, shift=0) → TimestepEmbedding(256, embedding_dim)
#[derive(Debug)]
pub struct PixArtAlphaCombinedTimestepSizeEmbeddings {
    time_proj: Timesteps,
    timestep_embedder: TimestepEmbedding,
}

impl PixArtAlphaCombinedTimestepSizeEmbeddings {
    pub fn new(embedding_dim: usize, vb: VarBuilder) -> Result<Self> {
        Self::new_with_lora(embedding_dim, vb, None)
    }

    pub fn new_with_lora(
        embedding_dim: usize,
        vb: VarBuilder,
        lora: Option<&LoraConfig>,
    ) -> Result<Self> {
        let time_proj = Timesteps::new(256, true, 0.0, 1.0);
        let embedder_lora = lora.map(|l| l.pp("timestep_embedder"));
        let timestep_embedder = TimestepEmbedding::new_with_lora(
            256,
            embedding_dim,
            None,
            vb.pp("timestep_embedder"),
            embedder_lora.as_ref(),
        )?;
        Ok(Self {
            time_proj,
            timestep_embedder,
        })
    }
}

impl Module for PixArtAlphaCombinedTimestepSizeEmbeddings {
    fn forward(&self, timestep: &Tensor) -> Result<Tensor> {
        let proj = self.time_proj.forward(timestep)?;
        self.timestep_embedder.forward(&proj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_sinusoidal_embedding_shape() -> Result<()> {
        let dev = &Device::Cpu;
        let ts = Tensor::new(&[0.5f32, 1.0], dev)?;
        let emb = get_timestep_embedding(&ts, 256, true, 0.0, 1.0, 10000)?;
        assert_eq!(emb.dims(), &[2, 256]);
        Ok(())
    }

    #[test]
    fn test_sinusoidal_embedding_odd_dim() -> Result<()> {
        let dev = &Device::Cpu;
        let ts = Tensor::new(&[0.5f32], dev)?;
        let emb = get_timestep_embedding(&ts, 257, false, 1.0, 1.0, 10000)?;
        assert_eq!(emb.dims(), &[1, 257]);
        Ok(())
    }
}
