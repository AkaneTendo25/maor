use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use maor_nn::conv3d::{CausalConv3d, SpatialPaddingMode};

/// Depth-to-space upsampling: conv → rearrange channels to spatial dims.
///
/// Expands spatial/temporal dimensions by moving channels into space.
/// If stride=(2,2,2), each voxel becomes a 2x2x2 block of voxels.
/// When stride[0]==2, the first temporal frame is removed after rearrangement
/// (inverse of the temporal padding added during encoding).
#[derive(Debug)]
pub struct DepthToSpaceUpsample {
    conv: CausalConv3d,
    stride: (usize, usize, usize),
}

impl DepthToSpaceUpsample {
    pub fn new(
        in_channels: usize,
        stride: (usize, usize, usize),
        out_channels_reduction_factor: usize,
        spatial_padding_mode: SpatialPaddingMode,
        vb: VarBuilder,
    ) -> Result<Self> {
        let out_channels =
            stride.0 * stride.1 * stride.2 * in_channels / out_channels_reduction_factor;

        let conv = CausalConv3d::new_with_padding_mode(
            in_channels,
            out_channels,
            3,
            (1, 1, 1), // stride=1 for the conv; upsampling is done by rearrange
            1,
            true,
            spatial_padding_mode,
            vb.pp("conv"),
        )?;

        Ok(Self { conv, stride })
    }

    /// Forward pass: conv → depth-to-space rearrange → optional trim first frame.
    pub fn forward(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        let x = self.conv.forward(x, causal)?;

        // Depth-to-space: (B, C*p1*p2*p3, D, H, W) → (B, C, D*p1, H*p2, W*p3)
        let x = depth_to_space_3d(&x, self.stride)?;

        // If temporal stride is 2, remove the first frame
        // (reverses the temporal padding added during encoding)
        if self.stride.0 == 2 {
            let d = x.dim(2)?;
            x.narrow(2, 1, d - 1)
        } else {
            Ok(x)
        }
    }
}

/// 3D depth-to-space rearrange.
///
/// (B, C*p1*p2*p3, D, H, W) → (B, C, D*p1, H*p2, W*p3)
///
/// einops equivalent: "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)"
fn depth_to_space_3d(x: &Tensor, stride: (usize, usize, usize)) -> Result<Tensor> {
    let (b, c_total, d, h, w) = x.dims5()?;
    let (p1, p2, p3) = stride;
    let c = c_total / (p1 * p2 * p3);

    // Reshape: (B, C*p1*p2*p3, D, H, W) → (B, C, p1, p2, p3, D, H, W)
    let x = x.reshape(&[b, c, p1, p2, p3, d, h, w])?;

    // Permute to: (B, C, D, p1, H, p2, W, p3)
    let x = x.permute([0usize, 1, 5, 2, 6, 3, 7, 4].as_slice())?;

    // Reshape to: (B, C, D*p1, H*p2, W*p3)
    x.contiguous()?.reshape((b, c, d * p1, h * p2, w * p3))
}
