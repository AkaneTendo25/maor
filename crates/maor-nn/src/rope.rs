use candle_core::{DType, Device, Result, Tensor};

/// RoPE type: interleaved or split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LTXRopeType {
    Interleaved,
    Split,
}

/// Apply rotary positional embeddings to input tensor.
///
/// `cos_freqs` and `sin_freqs` are precomputed frequency tensors.
pub fn apply_rotary_emb(
    input: &Tensor,
    cos_freqs: &Tensor,
    sin_freqs: &Tensor,
    rope_type: LTXRopeType,
) -> Result<Tensor> {
    match rope_type {
        LTXRopeType::Interleaved => apply_interleaved_rotary_emb(input, cos_freqs, sin_freqs),
        LTXRopeType::Split => apply_split_rotary_emb(input, cos_freqs, sin_freqs),
    }
}

/// Interleaved RoPE: pairs adjacent elements and rotates.
///
/// For each pair (x1, x2): output = (x1*cos - x2*sin, x1*sin + x2*cos)
fn apply_interleaved_rotary_emb(
    input: &Tensor,
    cos_freqs: &Tensor,
    sin_freqs: &Tensor,
) -> Result<Tensor> {
    let dims = input.dims();
    let last_dim = dims[dims.len() - 1];
    if !last_dim.is_multiple_of(2) {
        return Err(candle_core::Error::Msg(
            "last dim must be even for interleaved RoPE".into(),
        ));
    }

    // Reshape to (..., d, 2) pairs
    let mut new_shape = dims.to_vec();
    new_shape.pop();
    new_shape.push(last_dim / 2);
    new_shape.push(2);
    let paired = input.reshape(new_shape.as_slice())?;

    // Extract t1, t2 from last dimension
    let t1 = paired.narrow(dims.len(), 0, 1)?.squeeze(dims.len())?;
    let t2 = paired.narrow(dims.len(), 1, 1)?.squeeze(dims.len())?;

    // Rotated: (-t2, t1)
    let neg_t2 = t2.neg()?;

    // Stack back: [(-t2, t1)] interleaved
    let rot_t1 = neg_t2.unsqueeze(dims.len())?;
    let rot_t2 = t1.unsqueeze(dims.len())?;
    let rotated = Tensor::cat(&[&rot_t1, &rot_t2], dims.len())?;
    let rotated = rotated.reshape(dims)?;

    // input * cos + rotated * sin
    let out = input
        .broadcast_mul(cos_freqs)?
        .add(&rotated.broadcast_mul(sin_freqs)?)?;
    Ok(out)
}

/// Split RoPE: splits tensor in half and rotates each half.
///
/// First half rotated by cos/sin, second half rotated by -sin/cos.
/// Uses complex multiplication: (a + bi)(cos + i*sin) = (a*cos - b*sin) + i(a*sin + b*cos)
fn apply_split_rotary_emb(
    input: &Tensor,
    cos_freqs: &Tensor,
    sin_freqs: &Tensor,
) -> Result<Tensor> {
    // Reference split RoPE stores freqs as (B, H, T, D/2). Attention passes
    // Q/K as (B, T, H*D), so reshape to heads before applying the rotation.
    if input.dims().len() != 4 && cos_freqs.dims().len() == 4 {
        let (b, h, t, half_per_head) = cos_freqs.dims4()?;
        let dims = input.dims().to_vec();
        let per_head = half_per_head * 2;
        let x = input.reshape((b, t, h, per_head))?.transpose(1, 2)?;
        let x = apply_split_rotary_emb(&x, cos_freqs, sin_freqs)?;
        return x.transpose(1, 2)?.contiguous()?.reshape(dims.as_slice());
    }

    let dims = input.dims();
    let last_dim = dims[dims.len() - 1];
    let half = last_dim / 2;

    let x1 = input.narrow(dims.len() - 1, 0, half)?;
    let x2 = input.narrow(dims.len() - 1, half, half)?;

    // (x1 * cos - x2 * sin, x1 * sin + x2 * cos)
    let out1 = x1
        .broadcast_mul(cos_freqs)?
        .sub(&x2.broadcast_mul(sin_freqs)?)?;
    let out2 = x1
        .broadcast_mul(sin_freqs)?
        .add(&x2.broadcast_mul(cos_freqs)?)?;

    Tensor::cat(&[&out1, &out2], dims.len() - 1)
}

/// Precompute RoPE frequency tensors for multi-dimensional positions.
///
/// Returns `(cos_freqs, sin_freqs)` ready for [`apply_rotary_emb`].
///
/// # How it works
/// For each spatial dimension d (frame, height, width):
/// 1. Extract positions from `indices_grid[:, d, :, :]`
/// 2. If bounds `[start, end)` are provided and `use_middle_indices_grid`, use midpoint
/// 3. Normalize to `[0, 1]` by dividing by `max_pos[d]`
/// 4. Multiply by base frequencies `theta^(-2k/freq_dim)` for `k = 0..freq_dim/2`
/// 5. Scale by π to get angular frequencies
/// 6. Compute cos/sin and concatenate across all spatial dimensions
///
/// The result has `ndim * freq_dim/2` frequency components total, split evenly
/// across spatial dimensions. For video with ndim=3 and dim=128:
/// `freq_dim = 128/3 ≈ 42`, so ~21 frequency pairs per spatial dimension.
///
/// - `indices_grid`: `(B, ndim, T, 2)` with `[start, end)` bounds, or `(B, ndim, T)`
/// - `dim`: head dimension (total frequency components = dim/2)
/// - `theta`: base frequency (default 10000)
/// - `max_pos`: per-dimension max position for normalization
pub fn precompute_freqs_cis(
    indices_grid: &Tensor,
    dim: usize,
    out_dtype: DType,
    theta: f64,
    max_pos: &[usize],
    use_middle_indices_grid: bool,
    _num_attention_heads: usize,
    rope_type: LTXRopeType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let ndim = max_pos.len(); // typically 3 for video (frame, height, width)
    let n_elem = 2 * ndim;
    let freq_count = dim / n_elem;
    if freq_count == 0 {
        return Err(candle_core::Error::Msg(format!(
            "RoPE dim {dim} is too small for {ndim} position dimensions"
        )));
    }

    // Frequency progression:
    // theta ** linspace(log_theta(1), log_theta(theta), dim / (2 * ndim)),
    // then multiplied by pi/2.
    let denom = if freq_count > 1 { freq_count - 1 } else { 1 };
    let freq_indices: Vec<f32> = (0..freq_count)
        .map(|i| {
            let exp = i as f64 / denom as f64;
            (theta.powf(exp) * std::f64::consts::PI / 2.0) as f32
        })
        .collect();
    let freq_base = Tensor::from_vec(freq_indices, freq_count, device)?;

    // For each spatial dimension, compute (position / max_pos * 2 - 1) * freqs.
    // Stack as (B, T, freq, axis), then flatten to match reference ordering.
    let mut all_freqs = Vec::new();

    for (d, &max_p_val) in max_pos.iter().enumerate() {
        // Extract positions for this dimension: indices_grid[:, d, :, :]
        let positions = indices_grid.narrow(1, d, 1)?.squeeze(1)?;

        // If using middle indices grid and positions have start/end bounds
        let frac_positions =
            if positions.dims().len() >= 3 && positions.dims()[positions.dims().len() - 1] == 2 {
                // positions: (B, T, 2) with [start, end)
                let start = positions.narrow(positions.dims().len() - 1, 0, 1)?;
                let end = positions.narrow(positions.dims().len() - 1, 1, 1)?;
                if use_middle_indices_grid {
                    // Use midpoint
                    let mid = ((&start + &end)? * 0.5)?;
                    mid.squeeze(positions.dims().len() - 1)?
                } else {
                    start.squeeze(positions.dims().len() - 1)?
                }
            } else {
                positions
            };

        // Reference centers fractional coordinates into [-1, 1].
        let fractional = (frac_positions.to_dtype(DType::F32)? * (1.0 / max_p_val as f64))?;
        let centered = ((fractional * 2.0)? - 1.0)?;

        // Compute freqs: outer product of normalized positions and freq_base
        // positions: (B, T), freq_base: (freq_count,)
        // Result: (B, T, freq_count)
        let pos_expanded = centered.unsqueeze(centered.dims().len())?;
        let freq_expanded = freq_base.unsqueeze(0)?;
        // Handle batched positions
        let freqs = if pos_expanded.dims().len() == 3 {
            let freq_expanded = freq_expanded.unsqueeze(0)?; // (1, 1, freq_count)
            pos_expanded.broadcast_mul(&freq_expanded)?
        } else {
            pos_expanded.broadcast_mul(&freq_expanded)?
        };

        all_freqs.push(freqs);
    }

    // Reference ordering: stack axes in the last dim, then flatten
    // (B, T, freq_count, ndim) -> (B, T, freq_count * ndim).
    let stack_dim = all_freqs[0].dims().len();
    let freqs = Tensor::stack(&all_freqs, stack_dim)?;
    let b = freqs.dims()[0];
    let t = freqs.dims()[1];
    let freqs = freqs.reshape((b, t, freq_count * ndim))?;

    // Format for the specific RoPE type
    match rope_type {
        LTXRopeType::Interleaved => {
            // Repeat each element twice: [c0, c0, c1, c1, ...]
            let mut cos_out = repeat_interleave_last(&freqs.cos()?, 2)?;
            let mut sin_out = repeat_interleave_last(&freqs.sin()?, 2)?;
            let pad_size = dim % n_elem;
            if pad_size != 0 {
                let pad_shape = (b, t, pad_size);
                let cos_pad = Tensor::ones(pad_shape, DType::F32, device)?;
                let sin_pad = Tensor::zeros(pad_shape, DType::F32, device)?;
                cos_out = Tensor::cat(&[&cos_pad, &cos_out], 2)?;
                sin_out = Tensor::cat(&[&sin_pad, &sin_out], 2)?;
            }
            Ok((cos_out.to_dtype(out_dtype)?, sin_out.to_dtype(out_dtype)?))
        }
        LTXRopeType::Split => {
            let expected_freqs = dim / 2;
            let current_freqs = freq_count * ndim;
            let pad_size = expected_freqs
                .checked_sub(current_freqs)
                .ok_or_else(|| candle_core::Error::Msg("RoPE split freqs exceed dim/2".into()))?;
            if !expected_freqs.is_multiple_of(_num_attention_heads) {
                return Err(candle_core::Error::Msg(format!(
                    "RoPE split dim/2 {expected_freqs} not divisible by heads {_num_attention_heads}"
                )));
            }

            let mut cos_freq = freqs.cos()?;
            let mut sin_freq = freqs.sin()?;
            if pad_size != 0 {
                let pad_shape = (b, t, pad_size);
                let cos_pad = Tensor::ones(pad_shape, DType::F32, device)?;
                let sin_pad = Tensor::zeros(pad_shape, DType::F32, device)?;
                cos_freq = Tensor::cat(&[&cos_pad, &cos_freq], 2)?;
                sin_freq = Tensor::cat(&[&sin_pad, &sin_freq], 2)?;
            }

            let half_per_head = expected_freqs / _num_attention_heads;
            let cos_freq = cos_freq
                .reshape((b, t, _num_attention_heads, half_per_head))?
                .transpose(1, 2)?;
            let sin_freq = sin_freq
                .reshape((b, t, _num_attention_heads, half_per_head))?
                .transpose(1, 2)?;
            Ok((cos_freq.to_dtype(out_dtype)?, sin_freq.to_dtype(out_dtype)?))
        }
    }
}

/// Repeat each element along the last dimension `n` times.
/// [a, b, c] with n=2 → [a, a, b, b, c, c]
fn repeat_interleave_last(x: &Tensor, n: usize) -> Result<Tensor> {
    let dims = x.dims();
    let last = dims[dims.len() - 1];

    // Unsqueeze, expand, reshape
    let expanded = x.unsqueeze(x.dims().len())?; // (..., D, 1)
    let repeat_shape = {
        let mut v = vec![1usize; dims.len() + 1];
        v[dims.len()] = n;
        v
    };
    let expanded = expanded.repeat(repeat_shape.as_slice())?;

    // Flatten last two dims
    let mut new_shape = dims.to_vec();
    new_shape[dims.len() - 1] = last * n;
    expanded.reshape(new_shape.as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interleaved_rotary_identity_at_zero() -> Result<()> {
        let dev = &Device::Cpu;
        let input = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], dev)?;
        let cos = Tensor::ones((1, 4), DType::F32, dev)?;
        let sin = Tensor::zeros((1, 4), DType::F32, dev)?;

        let out = apply_interleaved_rotary_emb(&input, &cos, &sin)?;
        let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
        assert!((vals[0] - 1.0).abs() < 1e-5);
        assert!((vals[1] - 2.0).abs() < 1e-5);
        assert!((vals[2] - 3.0).abs() < 1e-5);
        assert!((vals[3] - 4.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_split_rotary_identity_at_zero() -> Result<()> {
        let dev = &Device::Cpu;
        let input = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], dev)?;
        let cos = Tensor::ones((1, 2), DType::F32, dev)?;
        let sin = Tensor::zeros((1, 2), DType::F32, dev)?;

        let out = apply_split_rotary_emb(&input, &cos, &sin)?;
        let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
        assert!((vals[0] - 1.0).abs() < 1e-5);
        assert!((vals[1] - 2.0).abs() < 1e-5);
        assert!((vals[2] - 3.0).abs() < 1e-5);
        assert!((vals[3] - 4.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_repeat_interleave() -> Result<()> {
        let dev = &Device::Cpu;
        let x = Tensor::new(&[1.0f32, 2.0, 3.0], dev)?;
        let out = repeat_interleave_last(&x, 2)?;
        let vals: Vec<f32> = out.to_vec1()?;
        assert_eq!(vals, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        Ok(())
    }
}
