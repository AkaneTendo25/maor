use candle_core::{Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder};

/// Causality axis for audio VAE convolutions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalityAxis {
    None,
    Width,
    Height,
}

/// Causal 2D convolution with asymmetric padding.
///
/// Ensures output at time `t` only depends on inputs at `t` and earlier.
/// Applies asymmetric padding before a zero-padding Conv2d.
#[derive(Debug)]
pub struct CausalConv2d {
    conv: Conv2d,
    /// Padding: (pad_left, pad_right, pad_top, pad_bottom)
    padding: (usize, usize, usize, usize),
}

impl CausalConv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        causality_axis: CausalityAxis,
        vb: VarBuilder,
    ) -> Result<Self> {
        let pad_h = (kernel_size - 1) * dilation;
        let pad_w = (kernel_size - 1) * dilation;

        let padding = match causality_axis {
            CausalityAxis::None => (pad_w / 2, pad_w - pad_w / 2, pad_h / 2, pad_h - pad_h / 2),
            CausalityAxis::Width => (pad_w, 0, pad_h / 2, pad_h - pad_h / 2),
            CausalityAxis::Height => (pad_w / 2, pad_w - pad_w / 2, pad_h, 0),
        };

        let cfg = Conv2dConfig {
            stride,
            padding: 0,
            dilation,
            groups,
            ..Default::default()
        };
        let weight = vb.pp("conv").get(
            &[out_channels, in_channels / groups, kernel_size, kernel_size],
            "weight",
        )?;
        let bias = vb.pp("conv").get(out_channels, "bias").ok();
        let conv = Conv2d::new(weight, bias, cfg);

        Ok(Self { conv, padding })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (pl, pr, pt, pb) = self.padding;
        let x = if pl > 0 || pr > 0 {
            x.pad_with_zeros(3, pl, pr)?
        } else {
            x.clone()
        };
        let x = if pt > 0 || pb > 0 {
            x.pad_with_zeros(2, pt, pb)?
        } else {
            x
        };
        self.conv.forward(&x)
    }
}

/// Leaky ReLU activation: max(0, x) + slope * min(0, x)
pub fn leaky_relu(x: &Tensor, negative_slope: f64) -> Result<Tensor> {
    let pos = x.relu()?;
    let neg = (x - &pos)?.affine(negative_slope, 0.0)?;
    pos + neg
}
