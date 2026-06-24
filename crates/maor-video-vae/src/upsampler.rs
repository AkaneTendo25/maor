use candle_core::{Module, Result, Tensor};
use candle_nn::{group_norm, Conv2d, Conv2dConfig, GroupNorm, VarBuilder};

use maor_nn::activation::silu;
use maor_nn::conv3d::Conv3d;

#[derive(Debug)]
struct UpsamplerResBlock {
    conv1: Conv3d,
    norm1: GroupNorm,
    conv2: Conv3d,
    norm2: GroupNorm,
}

impl UpsamplerResBlock {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let conv1 = Conv3d::new(
            channels,
            channels,
            3,
            (1, 1, 1),
            (1, 1, 1),
            1,
            true,
            vb.pp("conv1"),
        )?;
        let norm1 = group_norm(32, channels, 1e-5, vb.pp("norm1"))?;
        let conv2 = Conv3d::new(
            channels,
            channels,
            3,
            (1, 1, 1),
            (1, 1, 1),
            1,
            true,
            vb.pp("conv2"),
        )?;
        let norm2 = group_norm(32, channels, 1e-5, vb.pp("norm2"))?;
        Ok(Self {
            conv1,
            norm1,
            conv2,
            norm2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.conv1.forward(x)?;
        let x = self.norm1.forward(&x)?;
        let x = silu(&x)?;
        let x = self.conv2.forward(&x)?;
        let x = self.norm2.forward(&x)?;
        silu(&(x + residual)?)
    }
}

#[derive(Debug)]
struct SpatialX2 {
    conv: Conv2d,
    channels: usize,
}

impl SpatialX2 {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let weight = vb.get((4 * channels, channels, 3, 3), "weight")?;
        let bias = Some(vb.get(4 * channels, "bias")?);
        Ok(Self {
            conv: Conv2d::new(weight, bias, cfg),
            channels,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, f, h, w) = x.dims5()?;
        let x_2d = x.transpose(1, 2)?.contiguous()?.reshape((b * f, c, h, w))?;
        let x_2d = self.conv.forward(&x_2d)?;
        let x_2d = pixel_shuffle_2d(&x_2d, self.channels, 2)?;
        let (_, _, h2, w2) = x_2d.dims4()?;
        x_2d.reshape((b, f, self.channels, h2, w2))?
            .transpose(1, 2)?
            .contiguous()
    }
}

/// LTX-2.3 latent spatial upsampler.
#[derive(Debug)]
pub struct LatentUpsampler {
    initial_conv: Conv3d,
    initial_norm: GroupNorm,
    res_blocks: Vec<UpsamplerResBlock>,
    upsampler: SpatialX2,
    post_upsample_res_blocks: Vec<UpsamplerResBlock>,
    final_conv: Conv3d,
}

impl LatentUpsampler {
    pub fn new_x2(
        in_channels: usize,
        mid_channels: usize,
        blocks: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let initial_conv = Conv3d::new(
            in_channels,
            mid_channels,
            3,
            (1, 1, 1),
            (1, 1, 1),
            1,
            true,
            vb.pp("initial_conv"),
        )?;
        let initial_norm = group_norm(32, mid_channels, 1e-5, vb.pp("initial_norm"))?;

        let mut res_blocks = Vec::with_capacity(blocks);
        for idx in 0..blocks {
            res_blocks.push(UpsamplerResBlock::new(
                mid_channels,
                vb.pp("res_blocks").pp(idx),
            )?);
        }

        let upsampler = SpatialX2::new(mid_channels, vb.pp("upsampler").pp(0))?;

        let mut post_upsample_res_blocks = Vec::with_capacity(blocks);
        for idx in 0..blocks {
            post_upsample_res_blocks.push(UpsamplerResBlock::new(
                mid_channels,
                vb.pp("post_upsample_res_blocks").pp(idx),
            )?);
        }

        let final_conv = Conv3d::new(
            mid_channels,
            in_channels,
            3,
            (1, 1, 1),
            (1, 1, 1),
            1,
            true,
            vb.pp("final_conv"),
        )?;

        Ok(Self {
            initial_conv,
            initial_norm,
            res_blocks,
            upsampler,
            post_upsample_res_blocks,
            final_conv,
        })
    }

    pub fn forward(&self, latent: &Tensor) -> Result<Tensor> {
        let mut x = self.initial_conv.forward(latent)?;
        x = self.initial_norm.forward(&x)?;
        x = silu(&x)?;

        for block in &self.res_blocks {
            x = block.forward(&x)?;
        }

        x = self.upsampler.forward(&x)?;

        for block in &self.post_upsample_res_blocks {
            x = block.forward(&x)?;
        }

        self.final_conv.forward(&x)
    }
}

fn pixel_shuffle_2d(x: &Tensor, out_channels: usize, scale: usize) -> Result<Tensor> {
    let (b, c_total, h, w) = x.dims4()?;
    let expected = out_channels * scale * scale;
    if c_total != expected {
        candle_core::bail!("pixel shuffle expected {expected} channels, got {c_total}");
    }
    x.reshape((b, out_channels, scale, scale, h, w))?
        .permute((0, 1, 4, 2, 5, 3))?
        .contiguous()?
        .reshape((b, out_channels, h * scale, w * scale))
}
