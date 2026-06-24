use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::causal_conv2d::{CausalConv2d, CausalityAxis};

/// 2x nearest-neighbor upsample → CausalConv2d → trim first element on causal axis.
///
/// Undoes the causal padding added during encoding by trimming the first
/// row (Height) or column (Width) after convolution.
#[derive(Debug)]
pub struct Upsample2d {
    conv: CausalConv2d,
    causality_axis: CausalityAxis,
}

impl Upsample2d {
    pub fn new(in_channels: usize, causality_axis: CausalityAxis, vb: VarBuilder) -> Result<Self> {
        let conv = CausalConv2d::new(
            in_channels,
            in_channels,
            3,
            1,
            1,
            1,
            causality_axis,
            vb.pp("conv"),
        )?;
        Ok(Self {
            conv,
            causality_axis,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        let x = x.upsample_nearest2d(h * 2, w * 2)?;
        let x = self.conv.forward(&x)?;

        // Trim first element on the causal axis to undo encoder's padding
        match self.causality_axis {
            CausalityAxis::Height => {
                let h_out = x.dim(2)?;
                x.narrow(2, 1, h_out - 1)
            }
            CausalityAxis::Width => {
                let w_out = x.dim(3)?;
                x.narrow(3, 1, w_out - 1)
            }
            CausalityAxis::None => Ok(x),
        }
    }
}
