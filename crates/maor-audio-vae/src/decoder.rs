use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

use maor_nn::activation::silu;
use maor_nn::pixel_norm::PixelNorm;

use crate::attention::AttnBlock;
use crate::causal_conv2d::{CausalConv2d, CausalityAxis};
use crate::resnet::ResnetBlock2d;
use crate::upsample::Upsample2d;

const LATENT_DOWNSAMPLE_FACTOR: usize = 4;

/// Per-channel statistics for audio latent denormalization.
///
/// Operates on (B, C, T, F) tensors by patchifying to (B, T, C*F),
/// applying per-element stats, then unpatchifying back.
#[derive(Debug)]
struct AudioPerChannelStatistics {
    mean: Tensor, // shape: (latent_channels,) i.e. (C*F_latent,)
    std: Tensor,
}

impl AudioPerChannelStatistics {
    fn from_vb(latent_channels: usize, vb: VarBuilder) -> Result<Self> {
        let mean = vb.get(latent_channels, "mean-of-means")?;
        let std = vb.get(latent_channels, "std-of-means")?;
        Ok(Self { mean, std })
    }

    /// Denormalize: patchify → x * std + mean → unpatchify.
    fn denormalize(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, f) = x.dims4()?;
        // Patchify: (B, C, T, F) → (B, T, C*F)
        let patched = x
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((b, t, c * f))?;
        // un_normalize: x * std + mean
        let std = self.std.to_dtype(patched.dtype())?;
        let mean = self.mean.to_dtype(patched.dtype())?;
        let denormed = patched.broadcast_mul(&std)?.broadcast_add(&mean)?;
        // Unpatchify: (B, T, C*F) → (B, C, T, F)
        denormed
            .reshape((b, t, c, f))?
            .permute((0, 2, 1, 3))?
            .contiguous()
    }
}

/// A single decoder upsample level: ResNet blocks + optional attention + optional upsample.
#[derive(Debug)]
struct UpLevel {
    blocks: Vec<ResnetBlock2d>,
    attns: Vec<AttnBlock>,
    upsample: Option<Upsample2d>,
}

/// Mid-block: ResnetBlock → AttnBlock → ResnetBlock.
#[derive(Debug)]
struct MidBlock {
    block_1: ResnetBlock2d,
    attn_1: Option<AttnBlock>,
    block_2: ResnetBlock2d,
}

impl MidBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.block_1.forward(x)?;
        if let Some(attn) = &self.attn_1 {
            h = attn.forward(&h)?;
        }
        self.block_2.forward(&h)
    }
}

/// Audio VAE Decoder.
///
/// Reconstructs audio spectrograms from latent features.
/// Architecture: denormalize → conv_in → mid → up_levels → norm → SiLU → conv_out.
///
/// Default LTX-2.3 config: ch=128, out_ch=2, ch_mult=(1,2,4), num_res_blocks=2,
/// z_channels=8, causality_axis=Height.
#[derive(Debug)]
pub struct AudioDecoder {
    per_channel_statistics: AudioPerChannelStatistics,
    conv_in: CausalConv2d,
    mid: MidBlock,
    up_levels: Vec<UpLevel>,
    norm_out: PixelNorm,
    conv_out: CausalConv2d,
    out_ch: usize,
    causality_axis: CausalityAxis,
    mel_bins: Option<usize>,
}

impl AudioDecoder {
    pub fn new(
        ch: usize,
        out_ch: usize,
        ch_mult: &[usize],
        num_res_blocks: usize,
        z_channels: usize,
        causality_axis: CausalityAxis,
        mel_bins: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_with_roots(
            ch,
            out_ch,
            ch_mult,
            num_res_blocks,
            z_channels,
            causality_axis,
            mel_bins,
            vb.pp("per_channel_statistics"),
            vb,
        )
    }

    pub fn new_with_roots(
        ch: usize,
        out_ch: usize,
        ch_mult: &[usize],
        num_res_blocks: usize,
        z_channels: usize,
        causality_axis: CausalityAxis,
        mel_bins: Option<usize>,
        stats_vb: VarBuilder,
        decoder_vb: VarBuilder,
    ) -> Result<Self> {
        let num_resolutions = ch_mult.len();
        let base_block_channels = ch * ch_mult[ch_mult.len() - 1];

        let per_channel_statistics = AudioPerChannelStatistics::from_vb(ch, stats_vb)?;

        let conv_in = CausalConv2d::new(
            z_channels,
            base_block_channels,
            3,
            1,
            1,
            1,
            causality_axis,
            decoder_vb.pp("conv_in"),
        )?;

        // Mid block
        let mid_vb = decoder_vb.pp("mid");
        let mid = MidBlock {
            block_1: ResnetBlock2d::new(
                base_block_channels,
                base_block_channels,
                causality_axis,
                mid_vb.pp("block_1"),
            )?,
            attn_1: if mid_vb.pp("attn_1").pp("q").contains_tensor("weight") {
                Some(AttnBlock::new(base_block_channels, mid_vb.pp("attn_1"))?)
            } else {
                None
            },
            block_2: ResnetBlock2d::new(
                base_block_channels,
                base_block_channels,
                causality_axis,
                mid_vb.pp("block_2"),
            )?,
        };

        // Up levels are stored finest-to-coarsest and executed coarsest-to-finest.
        let mut up_levels = Vec::new();
        let up_vb = decoder_vb.pp("up");
        let mut block_in = base_block_channels;

        for level in (0..num_resolutions).rev() {
            let block_out = ch * ch_mult[level];
            let level_vb = up_vb.pp(level);

            let mut blocks = Vec::new();
            let blocks_vb = level_vb.pp("block");
            for i in 0..(num_res_blocks + 1) {
                blocks.push(ResnetBlock2d::new(
                    block_in,
                    block_out,
                    causality_axis,
                    blocks_vb.pp(i),
                )?);
                block_in = block_out;
            }

            // Attention path is skipped when the checkpoint has empty attn_resolutions.
            let attns = Vec::new();

            let upsample = if level != 0 {
                Some(Upsample2d::new(
                    block_in,
                    causality_axis,
                    level_vb.pp("upsample"),
                )?)
            } else {
                None
            };

            up_levels.push(UpLevel {
                blocks,
                attns,
                upsample,
            });
        }

        let norm_out = PixelNorm::new(1, 1e-6);

        let conv_out = CausalConv2d::new(
            block_in,
            out_ch,
            3,
            1,
            1,
            1,
            causality_axis,
            decoder_vb.pp("conv_out"),
        )?;

        Ok(Self {
            per_channel_statistics,
            conv_in,
            mid,
            up_levels,
            norm_out,
            conv_out,
            out_ch,
            causality_axis,
            mel_bins,
        })
    }

    /// Decode latent features to audio spectrogram.
    ///
    /// Input: (B, z_channels, frames, mel_bins) latent tensor.
    /// Output: (B, out_ch, time, frequency) decoded spectrogram.
    pub fn forward(&self, sample: &Tensor) -> Result<Tensor> {
        // Denormalize latents (patchify → per-channel stats → unpatchify)
        let sample = self.per_channel_statistics.denormalize(sample)?;

        // Compute target output shape
        let (_, _, in_frames, in_mel) = sample.dims4()?;
        let target_frames = if self.causality_axis != CausalityAxis::None {
            (in_frames * LATENT_DOWNSAMPLE_FACTOR)
                .saturating_sub(LATENT_DOWNSAMPLE_FACTOR - 1)
                .max(1)
        } else {
            in_frames * LATENT_DOWNSAMPLE_FACTOR
        };
        let target_mel = self.mel_bins.unwrap_or(in_mel);

        let mut h = self.conv_in.forward(&sample)?;
        h = self.mid.forward(&h)?;

        // Upsampling path
        for level in &self.up_levels {
            for (i, block) in level.blocks.iter().enumerate() {
                h = block.forward(&h)?;
                if i < level.attns.len() {
                    h = level.attns[i].forward(&h)?;
                }
            }
            if let Some(upsample) = &level.upsample {
                h = upsample.forward(&h)?;
            }
        }

        h = self.norm_out.forward(&h)?;
        h = silu(&h)?;
        h = self.conv_out.forward(&h)?;

        // Adjust output shape to match target dimensions
        adjust_output_shape(&h, self.out_ch, target_frames, target_mel)
    }
}

/// Crop/pad output to exact target shape.
fn adjust_output_shape(
    x: &Tensor,
    target_ch: usize,
    target_time: usize,
    target_freq: usize,
) -> Result<Tensor> {
    let (_, c, t, f) = x.dims4()?;

    // Narrow to min of current and target
    let c_out = c.min(target_ch);
    let t_out = t.min(target_time);
    let f_out = f.min(target_freq);

    let mut out = x.narrow(1, 0, c_out)?;
    out = out.narrow(2, 0, t_out)?;
    out = out.narrow(3, 0, f_out)?;

    // Pad if needed
    let time_pad = target_time.saturating_sub(t_out);
    let freq_pad = target_freq.saturating_sub(f_out);
    if time_pad > 0 || freq_pad > 0 {
        out = out.pad_with_zeros(3, 0, freq_pad)?;
        out = out.pad_with_zeros(2, 0, time_pad)?;
    }

    Ok(out)
}
