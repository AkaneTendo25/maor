use candle_core::{Result, Tensor};
use candle_nn::{Conv2dConfig, Module, VarBuilder};

use maor_nn::pixel_norm::PixelNorm;

/// Vanilla self-attention block for the audio VAE.
///
/// Q, K, V are 1x1 Conv2d projections. Attention: softmax(Q^T K / sqrt(C)) * V + skip.
#[derive(Debug)]
pub struct AttnBlock {
    norm: PixelNorm,
    q: candle_nn::Conv2d,
    k: candle_nn::Conv2d,
    v: candle_nn::Conv2d,
    proj_out: candle_nn::Conv2d,
}

impl AttnBlock {
    pub fn new(in_channels: usize, vb: VarBuilder) -> Result<Self> {
        let norm = PixelNorm::new(1, 1e-6);
        let cfg = Conv2dConfig {
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let q = candle_nn::conv2d(in_channels, in_channels, 1, cfg, vb.pp("q"))?;
        let k = candle_nn::conv2d(in_channels, in_channels, 1, cfg, vb.pp("k"))?;
        let v = candle_nn::conv2d(in_channels, in_channels, 1, cfg, vb.pp("v"))?;
        let proj_out = candle_nn::conv2d(in_channels, in_channels, 1, cfg, vb.pp("proj_out"))?;
        Ok(Self {
            norm,
            q,
            k,
            v,
            proj_out,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.norm.forward(x)?;
        let q = self.q.forward(&h)?;
        let k = self.k.forward(&h)?;
        let v = self.v.forward(&h)?;

        let (b, c, height, width) = q.dims4()?;
        let hw = height * width;
        let scale = (c as f64).powf(-0.5);

        // q: (B, C, H, W) → (B, C, H*W) → (B, H*W, C)
        let q = q.reshape((b, c, hw))?.permute((0, 2, 1))?.contiguous()?;
        // k: (B, C, H, W) → (B, C, H*W)
        let k = k.reshape((b, c, hw))?.contiguous()?;
        // w: (B, H*W, H*W) = Q K^T / sqrt(C)
        let w = q.matmul(&k)?.affine(scale, 0.0)?;
        let w = candle_nn::ops::softmax(&w, 2)?;

        // v: (B, C, H, W) → (B, C, H*W)
        let v = v.reshape((b, c, hw))?.contiguous()?;
        // Transpose attention weights for value multiplication
        let w = w.permute((0, 2, 1))?.contiguous()?;
        // h = V @ W: (B, C, H*W) → (B, C, H, W)
        let h = v.matmul(&w)?.reshape((b, c, height, width))?.contiguous()?;

        let h = self.proj_out.forward(&h)?;
        x + h
    }
}
