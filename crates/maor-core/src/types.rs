use candle_core::{bail, Result};

/// Spatio-temporal downscaling factors for the Video VAE.
///
/// Default: time=8, width=32, height=32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpatioTemporalScaleFactors {
    pub time: usize,
    pub width: usize,
    pub height: usize,
}

impl Default for SpatioTemporalScaleFactors {
    fn default() -> Self {
        Self {
            time: 8,
            width: 32,
            height: 32,
        }
    }
}

/// Video latent shape: (batch, channels, frames, height, width).
///
/// Represents a 5D tensor in the VAE latent space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoLatentShape {
    pub batch: usize,
    pub channels: usize,
    pub frames: usize,
    pub height: usize,
    pub width: usize,
}

impl VideoLatentShape {
    /// Create from raw 5D tensor dimensions.
    pub fn from_dims(dims: &[usize]) -> Result<Self> {
        if dims.len() != 5 {
            bail!("VideoLatentShape: expected 5D shape, got {}D", dims.len());
        }
        Ok(Self {
            batch: dims[0],
            channels: dims[1],
            frames: dims[2],
            height: dims[3],
            width: dims[4],
        })
    }

    /// Convert to a Vec usable with candle tensor operations.
    pub fn to_vec(&self) -> Vec<usize> {
        vec![
            self.batch,
            self.channels,
            self.frames,
            self.height,
            self.width,
        ]
    }

    /// Shape with channels=1 (for masks).
    pub fn mask_shape(&self) -> Self {
        Self {
            channels: 1,
            ..*self
        }
    }

    /// Compute latent shape from pixel-space video dimensions.
    pub fn from_pixel_shape(
        batch: usize,
        frames: usize,
        height: usize,
        width: usize,
        latent_channels: usize,
        scale_factors: &SpatioTemporalScaleFactors,
    ) -> Self {
        Self {
            batch,
            channels: latent_channels,
            frames: (frames - 1) / scale_factors.time + 1,
            height: height / scale_factors.height,
            width: width / scale_factors.width,
        }
    }

    /// Upscale back to pixel-space dimensions (approximate inverse of from_pixel_shape).
    pub fn upscale(&self, scale_factors: &SpatioTemporalScaleFactors) -> (usize, usize, usize) {
        let frames = (self.frames - 1) * scale_factors.time + 1;
        let height = self.height * scale_factors.height;
        let width = self.width * scale_factors.width;
        (frames, height, width)
    }

    /// Total number of spatial-temporal elements (frames * height * width).
    pub fn spatial_temporal_numel(&self) -> usize {
        self.frames * self.height * self.width
    }
}

/// Audio latent shape: (batch, channels, frames, mel_bins).
///
/// Mel_bins = frequency bins from mel-spectrogram.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioLatentShape {
    pub batch: usize,
    pub channels: usize,
    pub frames: usize,
    pub mel_bins: usize,
}

impl AudioLatentShape {
    /// Create from raw 4D tensor dimensions.
    pub fn from_dims(dims: &[usize]) -> Result<Self> {
        if dims.len() != 4 {
            bail!("AudioLatentShape: expected 4D shape, got {}D", dims.len());
        }
        Ok(Self {
            batch: dims[0],
            channels: dims[1],
            frames: dims[2],
            mel_bins: dims[3],
        })
    }

    /// Convert to a Vec usable with candle tensor operations.
    pub fn to_vec(&self) -> Vec<usize> {
        vec![self.batch, self.channels, self.frames, self.mel_bins]
    }

    /// Shape with channels=1, mel_bins=1 (for masks).
    pub fn mask_shape(&self) -> Self {
        Self {
            channels: 1,
            mel_bins: 1,
            ..*self
        }
    }

    /// Compute audio latent shape from duration in seconds.
    pub fn from_duration(
        batch: usize,
        duration_secs: f64,
        channels: usize,
        mel_bins: usize,
        sample_rate: usize,
        hop_length: usize,
        audio_latent_downsample_factor: usize,
    ) -> Self {
        let total_samples = (duration_secs * sample_rate as f64) as usize;
        let mel_frames = total_samples / hop_length;
        let frames = mel_frames / audio_latent_downsample_factor;
        Self {
            batch,
            channels,
            frames,
            mel_bins,
        }
    }

    /// Compute audio latent shape from a video pixel shape.
    pub fn from_video_frames(
        batch: usize,
        video_frames: usize,
        fps: f64,
        channels: usize,
        mel_bins: usize,
        sample_rate: usize,
        hop_length: usize,
        audio_latent_downsample_factor: usize,
    ) -> Self {
        let duration_secs = video_frames as f64 / fps;
        Self::from_duration(
            batch,
            duration_secs,
            channels,
            mel_bins,
            sample_rate,
            hop_length,
            audio_latent_downsample_factor,
        )
    }
}

/// Video pixel-space shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoPixelShape {
    pub batch: usize,
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub fps: f64,
}

/// Default video VAE scale factors (time=8, width=32, height=32).
pub const VIDEO_SCALE_FACTORS: SpatioTemporalScaleFactors = SpatioTemporalScaleFactors {
    time: 8,
    width: 32,
    height: 32,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_latent_from_pixel_shape() {
        // 33 frames at 8x temporal downscale -> (33-1)/8 + 1 = 5
        // 512 / 32 = 16, 768 / 32 = 24
        let shape = VideoLatentShape::from_pixel_shape(1, 33, 512, 768, 128, &VIDEO_SCALE_FACTORS);
        assert_eq!(shape.batch, 1);
        assert_eq!(shape.channels, 128);
        assert_eq!(shape.frames, 5);
        assert_eq!(shape.height, 16);
        assert_eq!(shape.width, 24);
    }

    #[test]
    fn test_video_latent_upscale_roundtrip() {
        let shape = VideoLatentShape {
            batch: 1,
            channels: 128,
            frames: 5,
            height: 16,
            width: 16,
        };
        let (f, h, w) = shape.upscale(&VIDEO_SCALE_FACTORS);
        assert_eq!(f, 33);
        assert_eq!(h, 512);
        assert_eq!(w, 512);
    }

    #[test]
    fn test_audio_latent_mask_shape() {
        let shape = AudioLatentShape {
            batch: 2,
            channels: 8,
            frames: 10,
            mel_bins: 16,
        };
        let mask = shape.mask_shape();
        assert_eq!(mask.channels, 1);
        assert_eq!(mask.mel_bins, 1);
        assert_eq!(mask.batch, 2);
        assert_eq!(mask.frames, 10);
    }
}
