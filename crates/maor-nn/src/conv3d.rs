use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, VarBuilder};

/// Spatial padding mode for VAE-style 3D convolutions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialPaddingMode {
    Zeros,
    Reflect,
}

impl SpatialPaddingMode {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "zeros" => Ok(Self::Zeros),
            "reflect" => Ok(Self::Reflect),
            other => candle_core::bail!("unsupported spatial padding mode: {other}"),
        }
    }
}

/// Conv3d implemented as a sum of Conv2d operations across the temporal dimension.
///
/// For each temporal kernel position, extracts a temporal slice of the input,
/// applies Conv2d with that position's 2D kernel, and accumulates results.
/// This avoids needing native 3D convolution support in candle.
///
/// Weight format: standard 5D tensor (out_ch, in_ch/groups, kT, kH, kW).
/// Used for checkpoints stored with standard 3D convolution weights.
#[derive(Debug)]
pub struct Conv3d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    spatial_padding_mode: SpatialPaddingMode,
    groups: usize,
}

impl Conv3d {
    /// Load from VarBuilder with standard "weight" and "bias" keys.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        groups: usize,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_with_padding_mode(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            bias,
            SpatialPaddingMode::Zeros,
            vb,
        )
    }

    /// Load from VarBuilder and configure spatial padding mode.
    pub fn new_with_padding_mode(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        groups: usize,
        bias: bool,
        spatial_padding_mode: SpatialPaddingMode,
        vb: VarBuilder,
    ) -> Result<Self> {
        let k = kernel_size;
        let weight = vb.get(&[out_channels, in_channels / groups, k, k, k], "weight")?;
        let bias = if bias {
            Some(vb.get(out_channels, "bias")?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            spatial_padding_mode,
            groups,
        })
    }

    /// Forward pass: (B, C_in, D, H, W) → (B, C_out, D', H', W')
    ///
    /// Decomposes the 3D convolution into kT separate Conv2d operations,
    /// one per temporal kernel position, and sums the results.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c_in, d, h, w) = x.dims5()?;
        let (s_t, s_h, s_w) = self.stride;
        let (p_t, p_h, p_w) = self.padding;
        let k_t = self.weight.dim(2)?;

        // Temporal padding (zero-pad if needed)
        let x = if p_t > 0 {
            let pad = Tensor::zeros(&[b, c_in, p_t, h, w], x.dtype(), x.device())?;
            Tensor::cat(&[&pad, x, &pad], 2)?
        } else {
            x.clone()
        };
        let d_padded = d + 2 * p_t;
        let d_out = (d_padded - k_t) / s_t + 1;

        let mut output: Option<Tensor> = None;

        for kt in 0..k_t {
            // Extract temporal slice for this kernel position
            let slice = if s_t == 1 {
                // Efficient: narrow is a view (zero-copy)
                x.narrow(2, kt, d_out)?
            } else {
                // Strided: select frames at kt, kt+s_t, kt+2*s_t, ...
                let indices: Vec<u32> = (0..d_out).map(|d| (d * s_t + kt) as u32).collect();
                let idx = Tensor::from_vec(indices, d_out, x.device())?;
                x.index_select(&idx, 2)?
            };

            // (B, C_in, D_out, H, W) → (B, D_out, C_in, H, W) → (B*D_out, C_in, H, W)
            let x_2d = slice
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b * d_out, c_in, h, w))?;

            // 2D weight for this temporal position: (C_out, C_in/groups, kH, kW)
            let w_2d = self.weight.narrow(2, kt, 1)?.squeeze(2)?;

            // Conv2d (spatial only, no bias — add bias once at end).
            // For reflect padding, materialize the spatial pad first and run
            // Candle's Conv2d with zero internal padding, matching PyTorch's
            // nn.Conv3d padding_mode="reflect" with temporal padding disabled.
            let (x_2d, conv_padding) = match self.spatial_padding_mode {
                SpatialPaddingMode::Zeros => (x_2d, (p_h, p_w)),
                SpatialPaddingMode::Reflect => {
                    let x_2d = reflect_pad_2d(&x_2d, p_h, p_w)?;
                    (x_2d, (0, 0))
                }
            };
            let y_2d = conv2d_manual(&x_2d, &w_2d, None, (s_h, s_w), conv_padding, self.groups)?;
            let (_, c_out, h_out, w_out) = y_2d.dims4()?;

            // (B*D_out, C_out, H_out, W_out) → (B, D_out, C_out, H_out, W_out) → (B, C_out, D_out, H_out, W_out)
            let y = y_2d
                .reshape((b, d_out, c_out, h_out, w_out))?
                .transpose(1, 2)?
                .contiguous()?;

            output = Some(match output {
                Some(acc) => (acc + y)?,
                None => y,
            });
        }

        let output = output.ok_or(candle_core::Error::Msg(
            "conv3d: zero temporal kernel size".into(),
        ))?;

        // Add bias: (C_out,) broadcast to (1, C_out, 1, 1, 1)
        match &self.bias {
            Some(bias) => {
                let bias = bias.reshape((1, bias.elem_count(), 1, 1, 1))?;
                output.broadcast_add(&bias)
            }
            None => Ok(output),
        }
    }
}

/// CausalConv3d: temporal causal padding (replicate first/last frame) + Conv3d.
///
/// Causal wrapper around standard 3D convolution weights.
#[derive(Debug)]
pub struct CausalConv3d {
    conv: Conv3d,
    time_kernel_size: usize,
}

impl CausalConv3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: (usize, usize, usize),
        groups: usize,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_with_padding_mode(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups,
            bias,
            SpatialPaddingMode::Zeros,
            vb,
        )
    }

    pub fn new_with_padding_mode(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: (usize, usize, usize),
        groups: usize,
        bias: bool,
        spatial_padding_mode: SpatialPaddingMode,
        vb: VarBuilder,
    ) -> Result<Self> {
        let height_pad = kernel_size / 2;
        let width_pad = kernel_size / 2;

        // Inner conv has spatial padding but no temporal padding
        // (temporal padding is handled externally via frame replication)
        let conv = Conv3d::new_with_padding_mode(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            (0, height_pad, width_pad),
            groups,
            bias,
            spatial_padding_mode,
            vb.pp("conv"),
        )?;

        Ok(Self {
            conv,
            time_kernel_size: kernel_size,
        })
    }

    /// Forward with temporal frame-replication padding.
    ///
    /// Causal: replicates first frame `kernel_size - 1` times before input.
    /// Non-causal: replicates first/last frames `(kernel_size - 1) / 2` times on each side.
    pub fn forward(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        let padded = if causal {
            let first_frame = x.narrow(2, 0, 1)?;
            let pad_count = self.time_kernel_size - 1;
            let padding: Vec<&Tensor> = (0..pad_count).map(|_| &first_frame).collect();
            let mut parts = padding;
            parts.push(x);
            Tensor::cat(&parts, 2)?
        } else {
            let first_frame = x.narrow(2, 0, 1)?;
            let (_, _, d, _, _) = x.dims5()?;
            let last_frame = x.narrow(2, d - 1, 1)?;
            let half_pad = (self.time_kernel_size - 1) / 2;
            let pre: Vec<&Tensor> = (0..half_pad).map(|_| &first_frame).collect();
            let post: Vec<&Tensor> = (0..half_pad).map(|_| &last_frame).collect();
            let mut parts = pre;
            parts.push(x);
            parts.extend(post);
            Tensor::cat(&parts, 2)?
        };

        self.conv.forward(&padded)
    }
}

/// DualConv3d: Decomposed 3D convolution as spatial Conv2d + temporal Conv1d.
///
/// For checkpoints stored with separate spatial and temporal weights.
/// NOT used for standard nn.Conv3d weights — use Conv3d for that.
#[derive(Debug)]
pub struct DualConv3d {
    weight1: Tensor,
    bias1: Option<Tensor>,
    stride1: (usize, usize),
    padding1: (usize, usize),
    weight2: Tensor,
    bias2: Option<Tensor>,
    stride2: usize,
    padding2: usize,
    groups: usize,
}

impl DualConv3d {
    /// Load from VarBuilder with weight1/bias1/weight2/bias2 keys.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        groups: usize,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let intermediate_channels = if in_channels < out_channels {
            out_channels
        } else {
            in_channels
        };

        let weight1 = vb.get(
            (
                intermediate_channels,
                in_channels / groups,
                1,
                kernel_size,
                kernel_size,
            ),
            "weight1",
        )?;
        let weight1 = weight1.squeeze(2)?;

        let bias1 = if bias {
            Some(vb.get(intermediate_channels, "bias1")?)
        } else {
            None
        };

        let weight2 = vb.get(
            (
                out_channels,
                intermediate_channels / groups,
                kernel_size,
                1,
                1,
            ),
            "weight2",
        )?;
        let weight2 = weight2.squeeze(4)?.squeeze(3)?;

        let bias2 = if bias {
            Some(vb.get(out_channels, "bias2")?)
        } else {
            None
        };

        Ok(Self {
            weight1,
            bias1,
            stride1: (stride.1, stride.2),
            padding1: (padding.1, padding.2),
            weight2,
            bias2,
            stride2: stride.0,
            padding2: padding.0,
            groups,
        })
    }

    /// Forward pass using 2D + 1D decomposition.
    ///
    /// Input: (B, C, D, H, W) → Output: (B, C', D', H', W')
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, d, h, w) = x.dims5()?;

        // Spatial conv: (B,C,D,H,W) → (B,D,C,H,W) → (B*D,C,H,W) → Conv2d
        let x_2d = x.transpose(1, 2)?.contiguous()?.reshape((b * d, c, h, w))?;
        let x_2d = conv2d_manual(
            &x_2d,
            &self.weight1,
            self.bias1.as_ref(),
            self.stride1,
            self.padding1,
            self.groups,
        )?;

        let (_, mid_c, h_out, w_out) = x_2d.dims4()?;

        // Temporal conv: (B*D,mid,H',W') → (B,D,mid,H',W') → (B,H',W',mid,D) → (B*H'*W',mid,D) → Conv1d
        let x_3d = x_2d.reshape((b, d, mid_c, h_out, w_out))?;
        let x_1d = x_3d
            .permute((0, 3, 4, 2, 1))? // (B, H', W', mid, D)
            .contiguous()?
            .reshape((b * h_out * w_out, mid_c, d))?;

        let x_1d = conv1d_manual(
            &x_1d,
            &self.weight2,
            self.bias2.as_ref(),
            self.stride2,
            self.padding2,
            self.groups,
        )?;

        let (_, out_c, d_out) = x_1d.dims3()?;

        // (B*H'*W', out, D') → (B, H', W', out, D') → (B, out, D', H', W')
        let out = x_1d.reshape((b, h_out, w_out, out_c, d_out))?;
        out.permute((0, 3, 4, 1, 2))
    }
}

/// Manual Conv2d using weight tensor.
///
/// Requires square stride and padding (sH=sW, pH=pW) since candle's Conv2dConfig
/// only supports a single value. This is always satisfied for LTX-2.3 models.
fn conv2d_manual(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    groups: usize,
) -> Result<Tensor> {
    if stride.0 != stride.1 {
        return Err(candle_core::Error::Msg(
            "non-square stride not yet supported".into(),
        ));
    }
    if padding.0 != padding.1 {
        return Err(candle_core::Error::Msg(
            "non-square padding not yet supported".into(),
        ));
    }
    let cfg = Conv2dConfig {
        stride: stride.0,
        padding: padding.0,
        groups,
        ..Default::default()
    };
    let input = input.contiguous()?;
    let weight = weight.contiguous()?;
    let conv = Conv2d::new(weight, bias.cloned(), cfg);
    conv.forward(&input)
}

fn reflect_pad_2d(input: &Tensor, pad_h: usize, pad_w: usize) -> Result<Tensor> {
    if pad_h == 0 && pad_w == 0 {
        return Ok(input.clone());
    }

    let (_, _, h, w) = input.dims4()?;
    let mut x = input.clone();

    if pad_h > 0 {
        if pad_h >= h {
            candle_core::bail!(
                "reflect height padding {pad_h} must be smaller than input height {h}"
            );
        }
        let indices = reflect_indices(h, pad_h);
        let idx = Tensor::from_vec(indices, h + 2 * pad_h, input.device())?;
        x = x.index_select(&idx, 2)?;
    }

    if pad_w > 0 {
        if pad_w >= w {
            candle_core::bail!(
                "reflect width padding {pad_w} must be smaller than input width {w}"
            );
        }
        let indices = reflect_indices(w, pad_w);
        let idx = Tensor::from_vec(indices, w + 2 * pad_w, input.device())?;
        x = x.index_select(&idx, 3)?;
    }

    Ok(x)
}

fn reflect_indices(size: usize, pad: usize) -> Vec<u32> {
    let mut indices = Vec::with_capacity(size + 2 * pad);
    indices.extend((1..=pad).rev().map(|idx| idx as u32));
    indices.extend((0..size).map(|idx| idx as u32));
    indices.extend(((size - pad - 1)..=(size - 2)).rev().map(|idx| idx as u32));
    indices
}

/// Manual Conv1d using weight tensor.
fn conv1d_manual(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    groups: usize,
) -> Result<Tensor> {
    let cfg = Conv1dConfig {
        stride,
        padding,
        groups,
        ..Default::default()
    };
    let input = input.contiguous()?;
    let weight = weight.contiguous()?;
    let conv = Conv1d::new(weight, bias.cloned(), cfg);
    conv.forward(&input)
}
