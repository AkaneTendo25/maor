use candle_core::{bail, DType, Device, Result, Tensor};

use crate::types::{AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape};

/// Video latent patchifier.
///
/// Converts between spatial (B, C, F, H, W) and token (B, T, D) representations.
/// Patch size is (1, patch_size, patch_size) — temporal patch is always 1.
#[derive(Debug, Clone, Copy)]
pub struct VideoLatentPatchifier {
    pub patch_size: usize,
}

impl VideoLatentPatchifier {
    pub fn new(patch_size: usize) -> Self {
        Self { patch_size }
    }

    /// Number of tokens after patchification.
    pub fn get_token_count(&self, shape: &VideoLatentShape) -> usize {
        let p = self.patch_size;
        // temporal patch = 1, so tokens = frames * (height/p) * (width/p)
        shape.frames * (shape.height / p) * (shape.width / p)
    }

    /// Patchify: (B, C, F, H, W) -> (B, F*Hp*Wp, C*1*p*p)
    ///
    /// einops: "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)" with p1=1
    pub fn patchify(&self, latents: &Tensor) -> Result<Tensor> {
        let (b, c, f, h, w) = latents.dims5()?;
        let p = self.patch_size;
        if h % p != 0 {
            bail!("height {h} not divisible by patch_size {p}");
        }
        if w % p != 0 {
            bail!("width {w} not divisible by patch_size {p}");
        }

        let hp = h / p;
        let wp = w / p;

        // Reshape: (B, C, F, Hp, p, Wp, p) — split spatial dims into patch grid + patch interior
        let x = latents.reshape(&[b, c, f, hp, p, wp, p])?;
        // Permute to: (B, F, Hp, Wp, C, p, p) — group spatial grid dims together, then channel+patch
        //   indices: [B=0, F=2, Hp=3, Wp=5, C=1, p=4, p=6]
        let x = x.permute([0usize, 2, 3, 5, 1, 4, 6].as_slice())?;
        // Flatten tokens and patch dims: (B, F*Hp*Wp, C*p*p)
        let num_tokens = f * hp * wp;
        let patch_dim = c * p * p;
        x.reshape((b, num_tokens, patch_dim))
    }

    /// Unpatchify: (B, F*Hp*Wp, C*p*p) -> (B, C, F, H, W)
    ///
    /// einops: "b (f h w) (c p q) -> b c f (h p) (w q)"
    pub fn unpatchify(&self, latents: &Tensor, output_shape: &VideoLatentShape) -> Result<Tensor> {
        let (b, _num_tokens, _patch_dim) = latents.dims3()?;
        let p = self.patch_size;
        let c = output_shape.channels;
        let f = output_shape.frames;
        let hp = output_shape.height / p;
        let wp = output_shape.width / p;

        // Reshape tokens back to grid: (B, F, Hp, Wp, C, p, p)
        let x = latents.reshape(&[b, f, hp, wp, c, p, p])?;
        // Permute to: (B, C, F, Hp, p, Wp, p) — inverse of patchify permutation
        //   indices: [B=0, C=4, F=1, Hp=2, p=5, Wp=3, p=6]
        let x = x.permute([0usize, 4, 1, 2, 5, 3, 6].as_slice())?;
        // Merge spatial: (B, C, F, H, W)
        x.reshape((b, c, f, output_shape.height, output_shape.width))
    }

    /// Compute patch grid bounds: (B, 3, num_patches, 2) with [start, end) per dimension.
    ///
    /// 3 dimensions = (frame, height, width).
    pub fn get_patch_grid_bounds(
        &self,
        output_shape: &VideoLatentShape,
        device: &Device,
    ) -> Result<Tensor> {
        let p = self.patch_size;
        let f = output_shape.frames;
        let hp = output_shape.height / p;
        let wp = output_shape.width / p;
        let num_patches = f * hp * wp;

        // Generate grid indices
        let mut bounds = vec![0f32; output_shape.batch * 3 * num_patches * 2];

        for b in 0..output_shape.batch {
            let mut idx = 0;
            for fi in 0..f {
                for hi in 0..hp {
                    for wi in 0..wp {
                        let base = b * 3 * num_patches * 2;
                        let dim_stride = num_patches * 2;
                        // frame dimension (step=1, temporal patch=1)
                        bounds[base + idx * 2] = fi as f32;
                        bounds[base + idx * 2 + 1] = (fi + 1) as f32;
                        // height dimension
                        bounds[base + dim_stride + idx * 2] = (hi * p) as f32;
                        bounds[base + dim_stride + idx * 2 + 1] = ((hi + 1) * p) as f32;
                        // width dimension
                        bounds[base + 2 * dim_stride + idx * 2] = (wi * p) as f32;
                        bounds[base + 2 * dim_stride + idx * 2 + 1] = ((wi + 1) * p) as f32;
                        idx += 1;
                    }
                }
            }
        }

        Tensor::from_vec(bounds, (output_shape.batch, 3, num_patches, 2), device)
    }
}

/// Convert latent-space patch coordinates to pixel-space coordinates.
///
/// Multiplies by VAE scale factors. Optional causal fix adjusts temporal axis
/// for causal VAE (first frame stride=1).
pub fn get_pixel_coords(
    latent_coords: &Tensor,
    scale_factors: &SpatioTemporalScaleFactors,
    causal_fix: bool,
) -> Result<Tensor> {
    // latent_coords: (B, 3, T, 2) — 3 dims = (frame, height, width)
    let scale = Tensor::new(
        &[
            scale_factors.time as f32,
            scale_factors.height as f32,
            scale_factors.width as f32,
        ],
        latent_coords.device(),
    )?
    .reshape((1, 3, 1, 1))?;

    let mut pixel_coords = latent_coords.to_dtype(DType::F32)?.broadcast_mul(&scale)?;

    if causal_fix {
        // Adjust temporal axis: pixel_coords[:, 0, ...] = (val + 1 - time_scale).clamp(min=0)
        let time_scale = scale_factors.time as f64;
        let temporal = pixel_coords.narrow(1, 0, 1)?;
        let adjusted = (temporal + (1.0 - time_scale))?.clamp(0.0, f64::INFINITY)?;
        // Replace the temporal slice
        let height_width = pixel_coords.narrow(1, 1, 2)?;
        pixel_coords = Tensor::cat(&[&adjusted, &height_width], 1)?;
    }

    Ok(pixel_coords)
}

/// Audio patchifier.
///
/// Converts between (B, C, T, mel_bins) and token (B, T, C*mel_bins) representations.
#[derive(Debug, Clone, Copy)]
pub struct AudioPatchifier {
    pub patch_size: usize,
    pub sample_rate: usize,
    pub hop_length: usize,
    pub audio_latent_downsample_factor: usize,
    pub is_causal: bool,
}

impl AudioPatchifier {
    pub fn new(patch_size: usize) -> Self {
        Self {
            patch_size,
            sample_rate: 16000,
            hop_length: 160,
            audio_latent_downsample_factor: 4,
            is_causal: true,
        }
    }

    /// Number of tokens = number of time frames.
    pub fn get_token_count(&self, shape: &AudioLatentShape) -> usize {
        shape.frames
    }

    /// Patchify: (B, C, T, mel_bins) -> (B, T, C*mel_bins)
    pub fn patchify(&self, latents: &Tensor) -> Result<Tensor> {
        let (b, c, t, mel) = latents.dims4()?;
        // Permute to (B, T, C, mel) then flatten last two dims
        let x = latents.permute((0, 2, 1, 3))?;
        x.reshape((b, t, c * mel))
    }

    /// Unpatchify: (B, T, C*mel_bins) -> (B, C, T, mel_bins)
    pub fn unpatchify(&self, latents: &Tensor, output_shape: &AudioLatentShape) -> Result<Tensor> {
        let (b, t, _) = latents.dims3()?;
        let c = output_shape.channels;
        let mel = output_shape.mel_bins;
        // Reshape to (B, T, C, mel) then permute to (B, C, T, mel)
        let x = latents.reshape((b, t, c, mel))?;
        x.permute((0, 2, 1, 3))
    }

    /// Convert latent frame index to time in seconds.
    pub fn latent_time_in_secs(&self, latent_frame: usize) -> f64 {
        let mut mel_frame = latent_frame * self.audio_latent_downsample_factor;
        if self.is_causal {
            mel_frame = mel_frame
                .saturating_add(1)
                .saturating_sub(self.audio_latent_downsample_factor);
        }
        (mel_frame * self.hop_length) as f64 / self.sample_rate as f64
    }

    /// Compute audio temporal bounds: (B, 1, T, 2) with [start_time, end_time] in seconds.
    pub fn get_patch_grid_bounds(
        &self,
        output_shape: &AudioLatentShape,
        device: &Device,
    ) -> Result<Tensor> {
        let t = output_shape.frames;
        let mut bounds = vec![0f32; output_shape.batch * t * 2];

        for b in 0..output_shape.batch {
            for i in 0..t {
                let start = self.latent_time_in_secs(i) as f32;
                let end = self.latent_time_in_secs(i + 1) as f32;
                let base = b * t * 2;
                bounds[base + i * 2] = start;
                bounds[base + i * 2 + 1] = end;
            }
        }

        Tensor::from_vec(bounds, (output_shape.batch, 1, t, 2), device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patchify_unpatchify_roundtrip() -> Result<()> {
        let dev = &Device::Cpu;
        let shape = VideoLatentShape {
            batch: 1,
            channels: 4,
            frames: 2,
            height: 4,
            width: 4,
        };
        let patchifier = VideoLatentPatchifier::new(2);

        // Create test tensor
        let numel = shape.batch * shape.channels * shape.frames * shape.height * shape.width;
        let data: Vec<f32> = (0..numel).map(|i| i as f32).collect();
        let input = Tensor::from_vec(data, shape.to_vec(), dev)?;

        let tokens = patchifier.patchify(&input)?;
        // Expected: (1, 2*2*2, 4*2*2) = (1, 8, 16)
        assert_eq!(tokens.dims(), &[1, 8, 16]);

        let recovered = patchifier.unpatchify(&tokens, &shape)?;
        assert_eq!(recovered.dims(), shape.to_vec().as_slice());

        // Check values match
        let orig: Vec<f32> = input.flatten_all()?.to_vec1()?;
        let rec: Vec<f32> = recovered.flatten_all()?.to_vec1()?;
        for (a, b) in orig.iter().zip(rec.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
        Ok(())
    }

    #[test]
    fn test_token_count() {
        let patchifier = VideoLatentPatchifier::new(2);
        let shape = VideoLatentShape {
            batch: 1,
            channels: 128,
            frames: 5,
            height: 16,
            width: 16,
        };
        // 5 * (16/2) * (16/2) = 5 * 8 * 8 = 320
        assert_eq!(patchifier.get_token_count(&shape), 320);
    }

    #[test]
    fn test_audio_patchify_roundtrip() -> Result<()> {
        let dev = &Device::Cpu;
        let shape = AudioLatentShape {
            batch: 1,
            channels: 8,
            frames: 10,
            mel_bins: 16,
        };
        let patchifier = AudioPatchifier::new(1);

        let numel = shape.batch * shape.channels * shape.frames * shape.mel_bins;
        let data: Vec<f32> = (0..numel).map(|i| i as f32).collect();
        let input = Tensor::from_vec(data, shape.to_vec(), dev)?;

        let tokens = patchifier.patchify(&input)?;
        assert_eq!(tokens.dims(), &[1, 10, 128]); // 8*16 = 128

        let recovered = patchifier.unpatchify(&tokens, &shape)?;
        assert_eq!(recovered.dims(), shape.to_vec().as_slice());

        let orig: Vec<f32> = input.flatten_all()?.to_vec1()?;
        let rec: Vec<f32> = recovered.flatten_all()?.to_vec1()?;
        for (a, b) in orig.iter().zip(rec.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_audio_latent_time() {
        let p = AudioPatchifier::new(1);
        // frame 0 with causal: mel_frame = 0*4 + 1 - 4 = clamp(0) = 0
        assert!((p.latent_time_in_secs(0) - 0.0).abs() < 1e-6);
        // frame 1: mel_frame = 1*4 + 1 - 4 = 1 -> 1*160/16000 = 0.01
        assert!((p.latent_time_in_secs(1) - 0.01).abs() < 1e-6);
    }
}
