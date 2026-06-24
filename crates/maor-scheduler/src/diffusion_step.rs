use candle_core::{DType, Result, Tensor};

/// Euler diffusion step (first-order ODE solver).
///
/// Given a sample at noise level `sigmas[step_index]`, and the model's denoised prediction,
/// computes the next sample at noise level `sigmas[step_index + 1]`.
///
/// `x_{t+1} = x_t + velocity * (sigma_{t+1} - sigma_t)`
pub fn euler_step(
    sample: &Tensor,
    denoised_sample: &Tensor,
    sigmas: &[f32],
    step_index: usize,
) -> Result<Tensor> {
    let sigma = sigmas[step_index] as f64;
    let sigma_next = sigmas[step_index + 1] as f64;
    let dt = sigma_next - sigma;

    // velocity = (sample - denoised) / sigma
    let sample_f32 = sample.to_dtype(DType::F32)?;
    let denoised_f32 = denoised_sample.to_dtype(DType::F32)?;
    let velocity = ((&sample_f32 - &denoised_f32)? * (1.0 / sigma))?;

    // next = sample + velocity * dt
    let next = (sample_f32 + (velocity * dt)?)?;
    next.to_dtype(sample.dtype())
}

/// RES_2S midpoint predictor input.
///
/// Computes the midpoint sample for the two-stage RES_2S update. The returned
/// tensor is the midpoint sample; the scalar is the midpoint sigma to use for
/// the second model evaluation. Returns `None` for the final zero-sigma step.
pub fn res2s_midpoint(
    sample: &Tensor,
    denoised_sample: &Tensor,
    sigma_current: f32,
    sigma_next: f32,
) -> Result<Option<(Tensor, f32)>> {
    if sigma_next <= 0.0 {
        return Ok(None);
    }

    let sigma_current = sigma_current as f64;
    let sigma_next = sigma_next as f64;
    let h = -(sigma_next / sigma_current).ln();
    let midpoint_fraction = 0.5f64;
    let phi1_mid = phi(1, -h * midpoint_fraction);
    let advance_weight = midpoint_fraction * phi1_mid;

    let sample_f32 = sample.to_dtype(DType::F32)?;
    let denoised_f32 = denoised_sample.to_dtype(DType::F32)?;
    let delta = (&denoised_f32 - &sample_f32)?;
    let midpoint = (sample_f32 + (delta * (h * advance_weight))?)?;
    let sigma_midpoint = (sigma_current.ln() - h * midpoint_fraction).exp() as f32;

    Ok(Some((midpoint.to_dtype(sample.dtype())?, sigma_midpoint)))
}

/// RES_2S ODE step using the stage-1 and midpoint denoised predictions.
pub fn res2s_step(
    sample: &Tensor,
    denoised_stage1: &Tensor,
    denoised_stage2: &Tensor,
    sigmas: &[f32],
    step_index: usize,
) -> Result<Tensor> {
    let sigma_current = sigmas[step_index] as f64;
    let sigma_next = sigmas[step_index + 1] as f64;
    if sigma_next <= 0.0 {
        return denoised_stage1.to_dtype(sample.dtype());
    }

    let h = -(sigma_next / sigma_current).ln();
    let phi1 = phi(1, -h);
    let phi2 = phi(2, -h);
    let midpoint_fraction = 0.5f64;
    let weight_stage2 = phi2 / midpoint_fraction;
    let weight_stage1 = phi1 - weight_stage2;

    let sample_f32 = sample.to_dtype(DType::F32)?;
    let stage1_f32 = denoised_stage1.to_dtype(DType::F32)?;
    let stage2_f32 = denoised_stage2.to_dtype(DType::F32)?;
    let stage1_delta = ((stage1_f32 - &sample_f32)? * weight_stage1)?;
    let stage2_delta = ((stage2_f32 - &sample_f32)? * weight_stage2)?;
    let update = ((stage1_delta + stage2_delta)? * h)?;
    let next = (sample_f32 + update)?;
    next.to_dtype(sample.dtype())
}

fn phi(order: usize, z: f64) -> f64 {
    let eps = f64::EPSILON * 16.0;
    if order == 1 {
        if z.abs() < eps {
            1.0 + z / 2.0 + z.powi(2) / 6.0 + z.powi(3) / 24.0
        } else {
            z.exp_m1() / z
        }
    } else if order == 2 {
        if z.abs() < eps {
            0.5 + z / 6.0 + z.powi(2) / 24.0 + z.powi(3) / 120.0
        } else {
            (z.exp_m1() - z) / z.powi(2)
        }
    } else {
        unreachable!("only phi_1 and phi_2 are supported")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_euler_step_reduces_noise() -> Result<()> {
        let dev = &Device::Cpu;
        let sample = Tensor::new(&[1.0f32, 0.5, -0.3], dev)?;
        let denoised = Tensor::new(&[0.8f32, 0.4, -0.2], dev)?;
        let sigmas = vec![0.5f32, 0.3, 0.1, 0.0];

        let next = euler_step(&sample, &denoised, &sigmas, 0)?;
        let next_vals: Vec<f32> = next.to_vec1()?;
        // Velocity = (sample - denoised) / sigma = [0.4, 0.2, -0.2]
        // dt = 0.3 - 0.5 = -0.2
        // next = sample + velocity * dt = [1.0 + 0.4*(-0.2), ...] = [0.92, 0.46, -0.26]
        assert!((next_vals[0] - 0.92).abs() < 1e-5);
        assert!((next_vals[1] - 0.46).abs() < 1e-5);
        assert!((next_vals[2] - (-0.26)).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_euler_step_to_zero_sigma() -> Result<()> {
        let dev = &Device::Cpu;
        let sample = Tensor::new(&[1.0f32], dev)?;
        let denoised = Tensor::new(&[0.5f32], dev)?;
        // sigma=1.0 → sigma_next=0.0: dt=-1, vel=(1-0.5)/1=0.5, next=1+0.5*(-1)=0.5
        let sigmas = vec![1.0f32, 0.0];

        let next = euler_step(&sample, &denoised, &sigmas, 0)?;
        let val: Vec<f32> = next.to_vec1()?;
        assert!((val[0] - 0.5).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_res2s_final_zero_returns_stage1() -> Result<()> {
        let dev = &Device::Cpu;
        let sample = Tensor::new(&[1.0f32], dev)?;
        let denoised_stage1 = Tensor::new(&[0.25f32], dev)?;
        let denoised_stage2 = Tensor::new(&[0.75f32], dev)?;
        let sigmas = vec![0.1f32, 0.0];

        let next = res2s_step(&sample, &denoised_stage1, &denoised_stage2, &sigmas, 0)?;
        let val: Vec<f32> = next.to_vec1()?;
        assert!((val[0] - 0.25).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_res2s_midpoint_between_sigmas() -> Result<()> {
        let dev = &Device::Cpu;
        let sample = Tensor::new(&[1.0f32], dev)?;
        let denoised = Tensor::new(&[0.0f32], dev)?;

        let (midpoint, midpoint_sigma) = res2s_midpoint(&sample, &denoised, 1.0, 0.25)?.unwrap();
        let val: Vec<f32> = midpoint.to_vec1()?;
        assert!(midpoint_sigma < 1.0 && midpoint_sigma > 0.25);
        assert!(val[0] < 1.0 && val[0] > 0.0);
        Ok(())
    }
}
