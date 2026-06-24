use candle_core::{DType, Result, Tensor};
use candle_nn::Module;

/// Per-pixel (per-location) RMS normalization.
///
/// y = x / sqrt(mean(x^2, dim=dim, keepdim=True) + eps)
///
/// Typically applied along the channel dimension (dim=1).
#[derive(Debug, Clone)]
pub struct PixelNorm {
    dim: usize,
    eps: f64,
}

impl PixelNorm {
    pub fn new(dim: usize, eps: f64) -> Self {
        Self { dim, eps }
    }
}

impl Default for PixelNorm {
    fn default() -> Self {
        Self { dim: 1, eps: 1e-8 }
    }
}

impl Module for PixelNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let mean_sq = x_f32.sqr()?.mean_keepdim(self.dim)?;
        let rms = (mean_sq + self.eps)?.sqrt()?;
        let normed = x_f32.broadcast_div(&rms)?;
        normed.to_dtype(x.dtype())
    }
}
