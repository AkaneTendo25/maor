use candle_core::{DType, Result, Tensor};

/// Create a candle error for shape mismatches.
pub fn shape_err(expected: &str, got: &str) -> candle_core::Error {
    candle_core::Error::Msg(format!("shape mismatch: expected {expected}, got {got}"))
}

/// Create a candle error for missing components.
pub fn missing_err(what: &str) -> candle_core::Error {
    candle_core::Error::Msg(format!("missing: {what}"))
}

/// RMS normalization over the last dimension.
///
/// `rms_norm(x, weight, eps)`:
///   norm = x * rsqrt(mean(x^2, dim=-1, keepdim=True) + eps)
///   if weight: norm * weight
pub fn rms_norm(x: &Tensor, weight: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let rsqrt = (variance + eps)?.sqrt()?.recip()?;
    let normed = x_f32.broadcast_mul(&rsqrt)?;
    let normed = normed.to_dtype(x.dtype())?;
    match weight {
        Some(w) => normed.broadcast_mul(w),
        None => Ok(normed),
    }
}

/// Convert denoised sample to velocity prediction.
///
/// velocity = (sample - denoised) / sigma
pub fn to_velocity(sample: &Tensor, sigma: &Tensor, denoised: &Tensor) -> Result<Tensor> {
    let sample_f32 = sample.to_dtype(DType::F32)?;
    let denoised_f32 = denoised.to_dtype(DType::F32)?;
    let sigma_f32 = sigma.to_dtype(DType::F32)?;
    let velocity = sample_f32.sub(&denoised_f32)?.broadcast_div(&sigma_f32)?;
    velocity.to_dtype(sample.dtype())
}

/// Convert velocity prediction to denoised sample.
///
/// denoised = sample - velocity * sigma
pub fn to_denoised(sample: &Tensor, velocity: &Tensor, sigma: &Tensor) -> Result<Tensor> {
    let sample_f32 = sample.to_dtype(DType::F32)?;
    let velocity_f32 = velocity.to_dtype(DType::F32)?;
    let sigma_f32 = sigma.to_dtype(DType::F32)?;
    let denoised = sample_f32.sub(&velocity_f32.broadcast_mul(&sigma_f32)?)?;
    denoised.to_dtype(sample.dtype())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_rms_norm_no_weight() -> Result<()> {
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &Device::Cpu)?;
        let out = rms_norm(&x, None, 1e-6)?;
        // rms of [1,2,3] = sqrt((1+4+9)/3) = sqrt(14/3) ≈ 2.1602
        // normalized: [1/2.1602, 2/2.1602, 3/2.1602] ≈ [0.4629, 0.9258, 1.3887]
        let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
        assert!((vals[0] - 0.4629).abs() < 0.001);
        assert!((vals[1] - 0.9258).abs() < 0.001);
        assert!((vals[2] - 1.3887).abs() < 0.001);
        Ok(())
    }

    #[test]
    fn test_velocity_denoised_roundtrip() -> Result<()> {
        let dev = &Device::Cpu;
        let sample = Tensor::new(&[1.0f32, 2.0, 3.0], dev)?;
        let denoised = Tensor::new(&[0.5f32, 1.5, 2.5], dev)?;
        let sigma = Tensor::new(&[0.8f32], dev)?;

        let vel = to_velocity(&sample, &sigma, &denoised)?;
        let recovered = to_denoised(&sample, &vel, &sigma)?;

        let recovered_vals: Vec<f32> = recovered.to_vec1()?;
        let denoised_vals: Vec<f32> = denoised.to_vec1()?;
        for (a, b) in recovered_vals.iter().zip(denoised_vals.iter()) {
            assert!((a - b).abs() < 1e-5, "roundtrip failed: {a} vs {b}");
        }
        Ok(())
    }
}
