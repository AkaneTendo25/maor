use candle_core::{Result, Tensor};
use candle_nn::{Conv1dConfig, Module, VarBuilder};

use maor_nn::activation::silu;
use maor_nn::pixel_norm::PixelNorm;

use crate::causal_conv2d::{leaky_relu, CausalConv2d, CausalityAxis};

const LRELU_SLOPE: f64 = 0.1;

/// ResNet block for the audio VAE decoder (2D convolutions).
///
/// Architecture: PixelNorm → SiLU → CausalConv2d → PixelNorm → SiLU → CausalConv2d + skip.
/// When in_channels != out_channels, skip uses a 1x1 CausalConv2d.
#[derive(Debug)]
pub struct ResnetBlock2d {
    norm1: PixelNorm,
    conv1: CausalConv2d,
    norm2: PixelNorm,
    conv2: CausalConv2d,
    nin_shortcut: Option<CausalConv2d>,
}

impl ResnetBlock2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        causality_axis: CausalityAxis,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = PixelNorm::new(1, 1e-6);
        let conv1 = CausalConv2d::new(
            in_channels,
            out_channels,
            3,
            1,
            1,
            1,
            causality_axis,
            vb.pp("conv1"),
        )?;
        let norm2 = PixelNorm::new(1, 1e-6);
        let conv2 = CausalConv2d::new(
            out_channels,
            out_channels,
            3,
            1,
            1,
            1,
            causality_axis,
            vb.pp("conv2"),
        )?;

        let nin_shortcut = if in_channels != out_channels {
            Some(CausalConv2d::new(
                in_channels,
                out_channels,
                1,
                1,
                1,
                1,
                causality_axis,
                vb.pp("nin_shortcut"),
            )?)
        } else {
            None
        };

        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            nin_shortcut,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.norm1.forward(x)?;
        let h = silu(&h)?;
        let h = self.conv1.forward(&h)?;

        let h = self.norm2.forward(&h)?;
        let h = silu(&h)?;
        let h = self.conv2.forward(&h)?;

        let skip = match &self.nin_shortcut {
            Some(conv) => conv.forward(x)?,
            None => x.clone(),
        };

        skip + h
    }
}

/// HiFi-GAN residual block (1D convolutions) for the vocoder.
///
/// 3 iterations of: LeakyReLU → dilated_conv → LeakyReLU → conv1d + residual.
#[derive(Debug)]
pub struct ResBlock1 {
    convs1: Vec<candle_nn::Conv1d>,
    convs2: Vec<candle_nn::Conv1d>,
}

impl ResBlock1 {
    pub fn new(
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        let vb1 = vb.pp("convs1");
        let vb2 = vb.pp("convs2");

        for (i, &dilation) in dilations.iter().enumerate() {
            // "same" padding: dilation * (kernel_size - 1) / 2
            let padding1 = dilation * (kernel_size - 1) / 2;
            let cfg1 = Conv1dConfig {
                padding: padding1,
                stride: 1,
                dilation,
                groups: 1,
                ..Default::default()
            };
            convs1.push(candle_nn::conv1d(
                channels,
                channels,
                kernel_size,
                cfg1,
                vb1.pp(i),
            )?);

            let padding2 = (kernel_size - 1) / 2;
            let cfg2 = Conv1dConfig {
                padding: padding2,
                stride: 1,
                dilation: 1,
                groups: 1,
                ..Default::default()
            };
            convs2.push(candle_nn::conv1d(
                channels,
                channels,
                kernel_size,
                cfg2,
                vb2.pp(i),
            )?);
        }

        Ok(Self { convs1, convs2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (conv1, conv2) in self.convs1.iter().zip(self.convs2.iter()) {
            let xt = leaky_relu(&x, LRELU_SLOPE)?;
            let xt = conv1.forward(&xt)?;
            let xt = leaky_relu(&xt, LRELU_SLOPE)?;
            let xt = conv2.forward(&xt)?;
            x = (xt + x)?;
        }
        Ok(x)
    }
}
