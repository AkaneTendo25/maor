use candle_core::{Module, Result, Tensor};
use candle_nn::{group_norm, GroupNorm, VarBuilder};

use maor_nn::activation::silu;
use maor_nn::conv3d::{CausalConv3d, Conv3d, SpatialPaddingMode};
use maor_nn::pixel_norm::PixelNorm;
use maor_nn::timestep_embedding::PixArtAlphaCombinedTimestepSizeEmbeddings;

/// Normalization layer enum for the VAE.
#[derive(Debug)]
pub enum NormLayer {
    PixelNorm(PixelNorm),
    GroupNorm(GroupNorm),
    Identity,
}

impl NormLayer {
    pub fn new_pixel_norm() -> Self {
        Self::PixelNorm(PixelNorm::default())
    }

    pub fn new_group_norm(
        num_groups: usize,
        num_channels: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self::GroupNorm(group_norm(
            num_groups,
            num_channels,
            eps,
            vb,
        )?))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::PixelNorm(pn) => pn.forward(x),
            Self::GroupNorm(gn) => gn.forward(x),
            Self::Identity => Ok(x.clone()),
        }
    }
}

/// 3D ResNet block with optional timestep conditioning.
///
/// Architecture: norm1 → [scale/shift] → SiLU → conv1 → norm2 → [scale/shift] → SiLU → conv2 + skip
/// Skip connection: norm3 → conv_shortcut (1x1x1) when in_channels != out_channels.
#[derive(Debug)]
pub struct ResnetBlock3D {
    norm1: NormLayer,
    conv1: CausalConv3d,
    norm2: NormLayer,
    conv2: CausalConv3d,
    /// 1x1x1 convolution for channel projection in skip connection (None if in==out).
    conv_shortcut: Option<Conv3d>,
    /// GroupNorm(1, in_ch) for skip when in_ch != out_ch (acts as LayerNorm).
    norm3: NormLayer,
    /// Timestep conditioning table: (4, in_channels) → shift1, scale1, shift2, scale2.
    scale_shift_table: Option<Tensor>,
}

impl ResnetBlock3D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        use_pixel_norm: bool,
        timestep_conditioning: bool,
        groups: usize,
        eps: f64,
        spatial_padding_mode: SpatialPaddingMode,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = if use_pixel_norm {
            NormLayer::new_pixel_norm()
        } else {
            NormLayer::new_group_norm(groups, in_channels, eps, vb.pp("norm1"))?
        };

        let conv1 = CausalConv3d::new_with_padding_mode(
            in_channels,
            out_channels,
            3,
            (1, 1, 1),
            1,
            true,
            spatial_padding_mode,
            vb.pp("conv1"),
        )?;

        let norm2 = if use_pixel_norm {
            NormLayer::new_pixel_norm()
        } else {
            NormLayer::new_group_norm(groups, out_channels, eps, vb.pp("norm2"))?
        };

        let conv2 = CausalConv3d::new_with_padding_mode(
            out_channels,
            out_channels,
            3,
            (1, 1, 1),
            1,
            true,
            spatial_padding_mode,
            vb.pp("conv2"),
        )?;

        let (conv_shortcut, norm3) = if in_channels != out_channels {
            let cs = Conv3d::new(
                in_channels,
                out_channels,
                1,
                (1, 1, 1),
                (0, 0, 0),
                1,
                true,
                vb.pp("conv_shortcut"),
            )?;
            let n3 = NormLayer::new_group_norm(1, in_channels, eps, vb.pp("norm3"))?;
            (Some(cs), n3)
        } else {
            (None, NormLayer::Identity)
        };

        let scale_shift_table = if timestep_conditioning {
            Some(vb.get(&[4, in_channels], "scale_shift_table")?)
        } else {
            None
        };

        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            conv_shortcut,
            norm3,
            scale_shift_table,
        })
    }

    /// Forward pass.
    ///
    /// `timestep`: optional timestep embedding of shape (B, 4*in_ch, 1, 1, 1)
    /// when timestep_conditioning is enabled.
    pub fn forward(&self, x: &Tensor, causal: bool, timestep: Option<&Tensor>) -> Result<Tensor> {
        let batch_size = x.dims()[0];
        let mut h = self.norm1.forward(x)?;

        // Apply timestep scale/shift if available
        let (shift2, scale2) = if let (Some(table), Some(ts)) = (&self.scale_shift_table, timestep)
        {
            // table: (4, in_ch) → (1, 4, in_ch, 1, 1, 1)
            // ts: (B, 4*in_ch, 1, 1, 1) → (B, 4, in_ch, 1, 1, 1)
            let in_ch = table.dim(1)?;
            let table = table
                .unsqueeze(0)?
                .reshape((1, 4, in_ch, 1, 1, 1))?
                .to_dtype(h.dtype())?;
            let ts_reshaped =
                ts.reshape(&[batch_size, 4, in_ch, ts.dim(2)?, ts.dim(3)?, ts.dim(4)?])?;
            let ada_values = (table + ts_reshaped)?;

            let shift1 = ada_values.narrow(1, 0, 1)?.squeeze(1)?;
            let scale1 = ada_values.narrow(1, 1, 1)?.squeeze(1)?;
            let shift2 = ada_values.narrow(1, 2, 1)?.squeeze(1)?;
            let scale2 = ada_values.narrow(1, 3, 1)?.squeeze(1)?;

            h = (h.broadcast_mul(&(scale1 + 1.0)?)? + shift1)?;
            (Some(shift2), Some(scale2))
        } else {
            (None, None)
        };

        h = silu(&h)?;
        h = self.conv1.forward(&h, causal)?;

        h = self.norm2.forward(&h)?;

        if let (Some(shift2), Some(scale2)) = (shift2, scale2) {
            h = (h.broadcast_mul(&(scale2 + 1.0)?)? + shift2)?;
        }

        h = silu(&h)?;
        // Dropout skipped for inference
        h = self.conv2.forward(&h, causal)?;

        // Skip connection
        let skip = self.norm3.forward(x)?;
        let skip = match &self.conv_shortcut {
            Some(cs) => cs.forward(&skip)?,
            None => skip,
        };

        skip + h
    }
}

/// UNet mid-block with multiple ResnetBlock3D layers and optional timestep conditioning.
#[derive(Debug)]
pub struct UNetMidBlock3D {
    res_blocks: Vec<ResnetBlock3D>,
    time_embedder: Option<PixArtAlphaCombinedTimestepSizeEmbeddings>,
}

impl UNetMidBlock3D {
    pub fn new(
        in_channels: usize,
        num_layers: usize,
        use_pixel_norm: bool,
        timestep_conditioning: bool,
        groups: usize,
        eps: f64,
        spatial_padding_mode: SpatialPaddingMode,
        vb: VarBuilder,
    ) -> Result<Self> {
        let time_embedder = if timestep_conditioning {
            Some(PixArtAlphaCombinedTimestepSizeEmbeddings::new(
                in_channels * 4,
                vb.pp("time_embedder"),
            )?)
        } else {
            None
        };

        let mut res_blocks = Vec::with_capacity(num_layers);
        let blocks_vb = vb.pp("res_blocks");
        for i in 0..num_layers {
            res_blocks.push(ResnetBlock3D::new(
                in_channels,
                in_channels,
                use_pixel_norm,
                timestep_conditioning,
                groups,
                eps,
                spatial_padding_mode,
                blocks_vb.pp(i),
            )?);
        }

        Ok(Self {
            res_blocks,
            time_embedder,
        })
    }

    /// Forward pass.
    ///
    /// `timestep`: raw timestep tensor (B,) when timestep_conditioning is enabled.
    pub fn forward(&self, x: &Tensor, causal: bool, timestep: Option<&Tensor>) -> Result<Tensor> {
        let timestep_embed = if let (Some(embedder), Some(ts)) = (&self.time_embedder, timestep) {
            let batch_size = x.dims()[0];
            let emb = embedder.forward(&ts.flatten_all()?)?;
            let emb_dim = emb.dim(emb.rank() - 1)?;
            Some(emb.reshape(&[batch_size, emb_dim, 1, 1, 1])?)
        } else {
            None
        };

        let mut h = x.clone();
        for block in &self.res_blocks {
            h = block.forward(&h, causal, timestep_embed.as_ref())?;
        }
        Ok(h)
    }
}
