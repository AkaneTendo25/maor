use candle_core::Tensor;

/// Bundle of latent tensors being denoised in the diffusion process.
#[derive(Debug, Clone)]
pub struct LatentState {
    /// Current noisy latent being denoised. Shape: (B, T, D) after patchification.
    pub latent: Tensor,
    /// Denoising strength per token. 1.0 = full denoise, 0.0 = no denoise.
    pub denoise_mask: Tensor,
    /// Positional indices for positional embeddings.
    pub positions: Tensor,
    /// Initial latent before denoising (may include conditioning frames).
    pub clean_latent: Tensor,
}

impl LatentState {
    pub fn new(
        latent: Tensor,
        denoise_mask: Tensor,
        positions: Tensor,
        clean_latent: Tensor,
    ) -> Self {
        Self {
            latent,
            denoise_mask,
            positions,
            clean_latent,
        }
    }
}
