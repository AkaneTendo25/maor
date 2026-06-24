use candle_core::{DType, Result, Tensor};

/// Classifier-Free Guidance.
///
/// `denoised = cond + (scale - 1) * (cond - uncond)`
/// Equivalent to: `denoised = uncond + scale * (cond - uncond)`
pub fn cfg_delta(cond: &Tensor, uncond: &Tensor, scale: f64) -> Result<Tensor> {
    let diff = (cond - uncond)?;
    diff * (scale - 1.0)
}

/// Compute guided denoised: `cond + delta`.
pub fn apply_guidance(cond: &Tensor, delta: &Tensor) -> Result<Tensor> {
    cond + delta
}

/// Multi-modal guider: combines CFG, STG, and modality guidance.
///
/// `pred = cond + (cfg-1)*(cond-uncond_text) + stg*(cond-uncond_perturbed) + (mod-1)*(cond-uncond_modality)`
/// Optional rescaling: `pred *= lerp(1, cond.std/pred.std, rescale_scale)`
pub fn multi_modal_guide(
    cond: &Tensor,
    uncond_text: Option<&Tensor>,
    uncond_perturbed: Option<&Tensor>,
    uncond_modality: Option<&Tensor>,
    cfg_scale: f64,
    stg_scale: f64,
    modality_scale: f64,
    rescale_scale: f64,
) -> Result<Tensor> {
    let cond_f32 = cond.to_dtype(DType::F32)?;
    let mut pred = cond_f32.clone();

    if cfg_scale != 1.0 {
        if let Some(uncond) = uncond_text {
            let uncond_f32 = uncond.to_dtype(DType::F32)?;
            let delta = ((&cond_f32 - &uncond_f32)? * (cfg_scale - 1.0))?;
            pred = (&pred + &delta)?;
        }
    }

    if stg_scale != 0.0 {
        if let Some(perturbed) = uncond_perturbed {
            let perturbed_f32 = perturbed.to_dtype(DType::F32)?;
            let delta = ((&cond_f32 - &perturbed_f32)? * stg_scale)?;
            pred = (&pred + &delta)?;
        }
    }

    if modality_scale != 1.0 {
        if let Some(uncond_mod) = uncond_modality {
            let uncond_mod_f32 = uncond_mod.to_dtype(DType::F32)?;
            let delta = ((&cond_f32 - &uncond_mod_f32)? * (modality_scale - 1.0))?;
            pred = (&pred + &delta)?;
        }
    }

    // Rescaling
    if rescale_scale != 0.0 {
        let cond_std = cond_f32.flatten_all()?.to_dtype(DType::F32)?;
        let pred_flat = pred.flatten_all()?;

        let cond_std_val = std_scalar(&cond_std)?;
        let pred_std_val = std_scalar(&pred_flat)?;

        if pred_std_val > 1e-8 {
            let factor = rescale_scale * (cond_std_val / pred_std_val) + (1.0 - rescale_scale);
            pred = (pred * factor)?;
        }
    }

    pred.to_dtype(cond.dtype())
}

/// Projection coefficient: dot(a, b) / dot(b, b).
pub fn projection_coef(to_project: &Tensor, project_onto: &Tensor) -> Result<f64> {
    let a = to_project.flatten_all()?.to_dtype(DType::F64)?;
    let b = project_onto.flatten_all()?.to_dtype(DType::F64)?;
    let dot: f64 = (&a * &b)?.sum_all()?.to_scalar()?;
    let norm_sq: f64 = (&b * &b)?.sum_all()?.to_scalar()?;
    Ok(dot / (norm_sq + 1e-8))
}

/// Standard deviation of a 1D tensor.
fn std_scalar(x: &Tensor) -> Result<f64> {
    let x_f64 = x.to_dtype(DType::F64)?;
    let mean: f64 = x_f64.mean_all()?.to_scalar()?;
    let diff = (x_f64 - mean)?;
    let var: f64 = diff.sqr()?.mean_all()?.to_scalar()?;
    Ok(var.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_cfg_no_guidance() -> Result<()> {
        let dev = &Device::Cpu;
        let cond = Tensor::new(&[1.0f32, 2.0], dev)?;
        let uncond = Tensor::new(&[0.5f32, 1.0], dev)?;
        // scale=1.0 means no guidance
        let delta = cfg_delta(&cond, &uncond, 1.0)?;
        let vals: Vec<f32> = delta.to_vec1()?;
        assert!(vals[0].abs() < 1e-6);
        assert!(vals[1].abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_cfg_scale_7_5() -> Result<()> {
        let dev = &Device::Cpu;
        let cond = Tensor::new(&[1.0f32], dev)?;
        let uncond = Tensor::new(&[0.0f32], dev)?;
        // delta = (7.5-1) * (1-0) = 6.5
        let delta = cfg_delta(&cond, &uncond, 7.5)?;
        let val: Vec<f32> = delta.to_vec1()?;
        assert!((val[0] - 6.5).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_projection_coef_parallel() -> Result<()> {
        let dev = &Device::Cpu;
        let a = Tensor::new(&[2.0f32, 4.0], dev)?;
        let b = Tensor::new(&[1.0f32, 2.0], dev)?;
        // dot(a,b) = 2+8 = 10, dot(b,b) = 1+4 = 5, coeff = 2.0
        let coef = projection_coef(&a, &b)?;
        assert!((coef - 2.0).abs() < 1e-5);
        Ok(())
    }
}
