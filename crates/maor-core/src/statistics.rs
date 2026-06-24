use candle_core::{bail, Result, Tensor};
use candle_nn::VarBuilder;

/// Per-channel statistics for normalizing/denormalizing latents.
///
/// Used by the Video VAE to map latent values to a standard distribution.
/// Checkpoints store these buffers as "std-of-means" and "mean-of-means".
#[derive(Debug, Clone)]
pub struct PerChannelStatistics {
    /// Per-channel mean, shape (1, C, 1, 1, 1) for video or (1, C, 1, 1) for audio.
    pub mean: Tensor,
    /// Per-channel std, same shape as mean.
    pub std: Tensor,
}

impl PerChannelStatistics {
    /// Load from VarBuilder (safetensors weights).
    ///
    /// Expects keys "mean-of-means" and "std-of-means" as 1D tensors of length C.
    /// Reshapes to 5D (1, C, 1, 1, 1) for video latent broadcasting.
    pub fn from_vb(latent_channels: usize, vb: VarBuilder) -> Result<Self> {
        let mean = vb.get(latent_channels, "mean-of-means")?;
        let std = vb.get(latent_channels, "std-of-means")?;
        let mean = mean.reshape((1, latent_channels, 1, 1, 1))?;
        let std = std.reshape((1, latent_channels, 1, 1, 1))?;
        Ok(Self { mean, std })
    }

    /// Create from raw mean/std vectors, reshaping to 5D for video latents.
    pub fn new_video(mean: &[f32], std: &[f32], device: &candle_core::Device) -> Result<Self> {
        let c = mean.len();
        if c != std.len() {
            bail!(
                "PerChannelStatistics: mean len ({c}) != std len ({})",
                std.len()
            );
        }
        let mean = Tensor::from_vec(mean.to_vec(), (1, c, 1, 1, 1), device)?;
        let std = Tensor::from_vec(std.to_vec(), (1, c, 1, 1, 1), device)?;
        Ok(Self { mean, std })
    }

    /// Create from raw mean/std vectors, reshaping to 4D for audio latents.
    pub fn new_audio(mean: &[f32], std: &[f32], device: &candle_core::Device) -> Result<Self> {
        let c = mean.len();
        if c != std.len() {
            bail!(
                "PerChannelStatistics: mean len ({c}) != std len ({})",
                std.len()
            );
        }
        let mean = Tensor::from_vec(mean.to_vec(), (1, c, 1, 1), device)?;
        let std = Tensor::from_vec(std.to_vec(), (1, c, 1, 1), device)?;
        Ok(Self { mean, std })
    }

    /// Normalize latents: (x - mean) / std
    pub fn normalize(&self, x: &Tensor) -> Result<Tensor> {
        let mean = self.mean.to_dtype(x.dtype())?.to_device(x.device())?;
        let std = self.std.to_dtype(x.dtype())?.to_device(x.device())?;
        x.broadcast_sub(&mean)?.broadcast_div(&std)
    }

    /// Denormalize latents: x * std + mean
    pub fn denormalize(&self, x: &Tensor) -> Result<Tensor> {
        let mean = self.mean.to_dtype(x.dtype())?.to_device(x.device())?;
        let std = self.std.to_dtype(x.dtype())?.to_device(x.device())?;
        x.broadcast_mul(&std)?.broadcast_add(&mean)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_normalize_denormalize_roundtrip() -> Result<()> {
        let dev = &Device::Cpu;
        let stats = PerChannelStatistics::new_video(&[1.0, 2.0], &[0.5, 0.25], dev)?;

        // Create a (1, 2, 1, 1, 1) tensor
        let x = Tensor::from_vec(vec![3.0f32, 4.0], &[1, 2, 1, 1, 1], dev)?;

        let normed = stats.normalize(&x)?;
        let recovered = stats.denormalize(&normed)?;

        let orig: Vec<f32> = x.flatten_all()?.to_vec1()?;
        let rec: Vec<f32> = recovered.flatten_all()?.to_vec1()?;
        for (a, b) in orig.iter().zip(rec.iter()) {
            assert!((a - b).abs() < 1e-5, "roundtrip failed: {a} vs {b}");
        }

        // Check normalized values: (3-1)/0.5=4, (4-2)/0.25=8
        let normed_vals: Vec<f32> = normed.flatten_all()?.to_vec1()?;
        assert!((normed_vals[0] - 4.0).abs() < 1e-5);
        assert!((normed_vals[1] - 8.0).abs() < 1e-5);

        Ok(())
    }
}
