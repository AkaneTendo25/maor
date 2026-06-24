//! Sigma schedule generators for the LTX-2.3 diffusion process.
//!
//! All schedulers return a `Vec<f32>` of sigma values from high noise to low noise,
//! with length = steps + 1 (including terminal sigma = 0).

/// Anchor points for token-dependent shift interpolation.
/// The shift amount is linearly interpolated between `base_shift` at 1024 tokens
/// and `max_shift` at 4096 tokens.
const BASE_SHIFT_ANCHOR: f64 = 1024.0;
const MAX_SHIFT_ANCHOR: f64 = 4096.0;

/// LTX-2.3 scheduler: linear sigmas with token-dependent shift and optional stretch.
///
/// This is the primary scheduler used by LTX-2.3 inference.
pub fn maor_schedule(
    steps: usize,
    num_tokens: Option<usize>,
    max_shift: f64,
    base_shift: f64,
    stretch: bool,
    terminal: f64,
) -> Vec<f32> {
    let tokens = num_tokens.unwrap_or(MAX_SHIFT_ANCHOR as usize) as f64;

    // Linear sigmas from 1.0 to 0.0
    let mut sigmas: Vec<f64> = (0..=steps).map(|i| 1.0 - i as f64 / steps as f64).collect();

    // Token-dependent shift via linear interpolation
    let mm = (max_shift - base_shift) / (MAX_SHIFT_ANCHOR - BASE_SHIFT_ANCHOR);
    let b = base_shift - mm * BASE_SHIFT_ANCHOR;
    let sigma_shift = tokens * mm + b;

    // Apply logit-space shift: sigma' = sigmoid(logit(sigma) + shift)
    // Equivalent to: sigma' = exp(shift) / (exp(shift) + (1/sigma - 1))
    // This shifts the noise schedule in logit space, pushing more steps toward higher noise.
    let exp_shift = sigma_shift.exp();
    for s in sigmas.iter_mut() {
        if *s > 0.0 && *s < 1.0 {
            let inv_minus_one = 1.0 / *s - 1.0;
            *s = exp_shift / (exp_shift + inv_minus_one);
        }
        // s == 0.0 stays 0.0, s == 1.0 stays 1.0
    }

    // Optional stretching to terminal value
    if stretch {
        // Find non-zero sigmas
        let non_zero: Vec<usize> = sigmas
            .iter()
            .enumerate()
            .filter(|(_, s)| **s != 0.0)
            .map(|(i, _)| i)
            .collect();

        if let Some(&last_nz) = non_zero.last() {
            let one_minus_last = 1.0 - sigmas[last_nz];
            if one_minus_last > 0.0 {
                let scale_factor = one_minus_last / (1.0 - terminal);
                for &i in &non_zero {
                    let one_minus = 1.0 - sigmas[i];
                    sigmas[i] = 1.0 - one_minus / scale_factor;
                }
            }
        }
    }

    sigmas.iter().map(|&s| s as f32).collect()
}

/// Linear-quadratic two-phase scheduler.
///
/// First phase is linear, second phase is quadratic. Used as an alternative scheduler.
pub fn linear_quadratic_schedule(
    steps: usize,
    threshold_noise: f64,
    linear_steps: Option<usize>,
) -> Vec<f32> {
    if steps == 1 {
        return vec![1.0, 0.0];
    }

    let linear_steps = linear_steps.unwrap_or(steps / 2);
    let quadratic_steps = steps - linear_steps;

    // Linear phase
    let mut sigma_schedule: Vec<f64> = (0..linear_steps)
        .map(|i| i as f64 * threshold_noise / linear_steps as f64)
        .collect();

    // Quadratic phase
    if quadratic_steps > 0 {
        let qs2 = (quadratic_steps * quadratic_steps) as f64;
        let ls = linear_steps as f64;
        let s = steps as f64;

        let tns_diff = ls - threshold_noise * s; // threshold_noise_step_diff
        let quadratic_coef = tns_diff / (ls * qs2);
        let linear_coef = threshold_noise / ls - 2.0 * tns_diff / qs2;
        let constant = quadratic_coef * ls * ls;

        for i in linear_steps..steps {
            let fi = i as f64;
            let val = quadratic_coef * fi * fi + linear_coef * fi + constant;
            sigma_schedule.push(val);
        }
    }

    sigma_schedule.push(1.0);

    // Flip: sigmas go from 1.0 down to 0.0
    sigma_schedule.iter().map(|&x| (1.0 - x) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maor_schedule_length() {
        let sigmas = maor_schedule(40, None, 2.05, 0.95, true, 0.1);
        assert_eq!(sigmas.len(), 41); // steps + 1
    }

    #[test]
    fn test_maor_schedule_endpoints() {
        let sigmas = maor_schedule(20, None, 2.05, 0.95, true, 0.1);
        // First sigma should be close to 1.0, last should be 0.0
        assert!((sigmas[0] - 1.0).abs() < 0.01, "first sigma: {}", sigmas[0]);
        assert!(
            sigmas.last().unwrap().abs() < 1e-6,
            "last sigma: {}",
            sigmas.last().unwrap()
        );
    }

    #[test]
    fn test_maor_schedule_monotonic_decreasing() {
        let sigmas = maor_schedule(40, Some(4096), 2.05, 0.95, true, 0.1);
        for i in 0..sigmas.len() - 1 {
            assert!(
                sigmas[i] >= sigmas[i + 1],
                "not monotonic at {}: {} < {}",
                i,
                sigmas[i],
                sigmas[i + 1]
            );
        }
    }

    #[test]
    fn test_maor_schedule_stretch_terminal() {
        let sigmas = maor_schedule(20, Some(4096), 2.05, 0.95, true, 0.1);
        // With stretch, the last non-zero sigma should be near terminal
        let last_nonzero = sigmas.iter().rposition(|s| *s > 0.0).unwrap();
        assert!(
            (sigmas[last_nonzero] - 0.1).abs() < 0.01,
            "last non-zero sigma: {}",
            sigmas[last_nonzero]
        );
    }

    #[test]
    fn test_linear_quadratic_schedule_length() {
        let sigmas = linear_quadratic_schedule(20, 0.025, None);
        assert_eq!(sigmas.len(), 21);
    }

    #[test]
    fn test_linear_quadratic_schedule_single_step() {
        let sigmas = linear_quadratic_schedule(1, 0.025, None);
        assert_eq!(sigmas, vec![1.0, 0.0]);
    }

    #[test]
    fn test_linear_quadratic_monotonic() {
        let sigmas = linear_quadratic_schedule(40, 0.025, None);
        for i in 0..sigmas.len() - 1 {
            assert!(
                sigmas[i] >= sigmas[i + 1],
                "not monotonic at {}: {} < {}",
                i,
                sigmas[i],
                sigmas[i + 1]
            );
        }
    }
}
