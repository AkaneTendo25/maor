use candle_core::{DType, Result, Tensor};
use candle_nn::{linear, linear_no_bias, Linear, Module, VarBuilder};

/// Gemma feature extractor: normalizes and projects multi-layer hidden states.
///
/// Takes all hidden states from Gemma (49 layers), normalizes per-layer,
/// flattens to [B, T, D*L], then projects to [B, T, D] via a single Linear.
///
/// Weights: `aggregate_embed.weight` of shape (D, D*L) = (3840, 188160).
#[derive(Debug)]
pub struct GemmaFeatureExtractor {
    projection: GemmaFeatureProjection,
    hidden_dim: usize,
    num_layers: usize,
}

#[derive(Debug)]
enum GemmaFeatureProjection {
    Single {
        aggregate_embed: Linear,
    },
    Dual {
        video_aggregate_embed: Linear,
        audio_aggregate_embed: Linear,
        video_out_dim: usize,
        audio_out_dim: usize,
    },
}

impl GemmaFeatureExtractor {
    /// `hidden_dim` is Gemma's hidden_size (3840), `num_layers` = 49 (including embedding).
    pub fn new(hidden_dim: usize, num_layers: usize, vb: VarBuilder) -> Result<Self> {
        Self::new_single(hidden_dim, num_layers, vb)
    }

    /// Single-projection feature extractor.
    pub fn new_single(hidden_dim: usize, num_layers: usize, vb: VarBuilder) -> Result<Self> {
        let input_dim = hidden_dim * num_layers;
        let aggregate_embed = linear_no_bias(input_dim, hidden_dim, vb.pp("aggregate_embed"))?;
        Ok(Self {
            projection: GemmaFeatureProjection::Single { aggregate_embed },
            hidden_dim,
            num_layers,
        })
    }

    /// LTX-2.3 feature extractor: per-token RMS normalization + separate biased
    /// video/audio projections.
    pub fn new_dual(
        hidden_dim: usize,
        num_layers: usize,
        video_out_dim: usize,
        audio_out_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input_dim = hidden_dim * num_layers;
        let video_aggregate_embed =
            linear(input_dim, video_out_dim, vb.pp("video_aggregate_embed"))?;
        let audio_aggregate_embed =
            linear(input_dim, audio_out_dim, vb.pp("audio_aggregate_embed"))?;
        Ok(Self {
            projection: GemmaFeatureProjection::Dual {
                video_aggregate_embed,
                audio_aggregate_embed,
                video_out_dim,
                audio_out_dim,
            },
            hidden_dim,
            num_layers,
        })
    }

    /// Extract features from all hidden states.
    ///
    /// - `hidden_states`: Vec of (B, T, D) tensors, one per layer
    /// - `attention_mask`: (B, T) with 1 = valid, 0 = padded
    /// - `padding_side`: "left" or "right"
    pub fn forward(
        &self,
        hidden_states: &[Tensor],
        attention_mask: &Tensor,
        padding_side: &str,
    ) -> Result<Tensor> {
        Ok(self
            .forward_av(hidden_states, attention_mask, padding_side)?
            .0)
    }

    /// Extract video/audio features from separate modality projections when
    /// available.
    pub fn forward_av(
        &self,
        hidden_states: &[Tensor],
        attention_mask: &Tensor,
        padding_side: &str,
    ) -> Result<(Tensor, Tensor)> {
        if hidden_states.len() != self.num_layers {
            return Err(candle_core::Error::Msg(format!(
                "expected {} hidden states, got {}",
                self.num_layers,
                hidden_states.len()
            )));
        }
        let orig_dtype = hidden_states[0].dtype();

        // Stack: (B, T, D, L)
        let stacked = Tensor::stack(hidden_states, 3)?;

        match &self.projection {
            GemmaFeatureProjection::Single { aggregate_embed } => {
                // Normalize and concatenate
                let normed = norm_and_concat_padded_batch(&stacked, attention_mask, padding_side)?;

                // Project: (B, T, D*L) -> (B, T, D)
                let features = aggregate_embed.forward(&normed.to_dtype(orig_dtype)?)?;
                Ok((features.clone(), features))
            }
            GemmaFeatureProjection::Dual {
                video_aggregate_embed,
                audio_aggregate_embed,
                video_out_dim,
                audio_out_dim,
            } => {
                let normed = norm_and_concat_per_token_rms(&stacked, attention_mask)?;
                let normed = normed.to_dtype(orig_dtype)?;
                let video_scale = (*video_out_dim as f64 / self.hidden_dim as f64).sqrt();
                let audio_scale = (*audio_out_dim as f64 / self.hidden_dim as f64).sqrt();
                let video = video_aggregate_embed.forward(&(&normed * video_scale)?)?;
                let audio = audio_aggregate_embed.forward(&(&normed * audio_scale)?)?;
                Ok((video, audio))
            }
        }
    }
}

/// Normalize each token/layer by RMS over the hidden dimension and flatten layers.
///
/// This is the 2.3 `GemmaFeaturesExtractorProjDualLinear` normalization path.
fn norm_and_concat_per_token_rms(encoded_text: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let (b, t, d, l) = encoded_text.dims4()?;
    let encoded_f32 = encoded_text.to_dtype(DType::F32)?;
    let variance = encoded_f32.sqr()?.mean_keepdim(2)?; // (B, T, 1, L)
    let denom = (variance + 1e-6)?.sqrt()?;
    let normed = encoded_f32.broadcast_div(&denom)?;
    let normed = normed.reshape((b, t, d * l))?;

    let mask = attention_mask
        .to_dtype(DType::F32)?
        .unsqueeze(2)?
        .expand((b, t, d * l))?;
    normed.broadcast_mul(&mask)
}

/// Normalize and flatten multi-layer hidden states, respecting padding.
///
/// Input: `(B, T, D, L)` stacked hidden states from all Gemma layers.
/// Output: `(B, T, D*L)` normalized and flattened.
///
/// For each batch item and layer:
/// 1. Compute mean over valid (non-padded) tokens and hidden dims
/// 2. Compute min/max over valid tokens (padded positions filled with ±inf to exclude them)
/// 3. Normalize: `8 * (x - mean) / (max - min + eps)`
/// 4. Flatten layers into the feature dimension and zero out padded positions
///
/// The scale factor of 8 matches the LTX-2.3 feature normalization.
fn norm_and_concat_padded_batch(
    encoded_text: &Tensor,
    attention_mask: &Tensor,
    padding_side: &str,
) -> Result<Tensor> {
    let (b, t, d, l) = encoded_text.dims4()?;
    let device = encoded_text.device();

    // Compute sequence lengths from attention_mask: (B,)
    let seq_lengths = attention_mask.to_dtype(DType::F32)?.sum(1)?;

    // Build mask: (B, T) → (B, T, 1, 1)
    let token_indices = Tensor::arange(0u32, t as u32, device)?
        .to_dtype(DType::F32)?
        .unsqueeze(0)?; // (1, T)

    let mask = match padding_side {
        "left" => {
            // Valid tokens are at the end: index >= (T - seq_len)
            let start = seq_lengths.affine(-1.0, t as f64)?; // (B,) = t - seq_len
            let start = start.unsqueeze(1)?; // (B, 1)
            token_indices.broadcast_ge(&start)? // (B, T)
        }
        _ => {
            // Right padding: valid tokens at the start: index < seq_len
            let lengths = seq_lengths.unsqueeze(1)?; // (B, 1)
            token_indices.broadcast_lt(&lengths)? // (B, T)
        }
    };
    let mask_4d = mask.to_dtype(DType::F32)?.unsqueeze(2)?.unsqueeze(3)?; // (B, T, 1, 1)

    let encoded_f32 = encoded_text.to_dtype(DType::F32)?;
    let eps = 1e-6;

    // Masked mean: sum over (T, D), divide by (seq_len * D) → (B, 1, 1, L)
    let masked = encoded_f32.broadcast_mul(&mask_4d)?;
    let denom = (&seq_lengths * d as f64)?.reshape((b, 1, 1, 1))?;
    let mean = masked.sum((1, 2))?.unsqueeze(1)?.unsqueeze(1)?;
    let mean = mean.broadcast_div(&(denom + eps)?)?;

    // Masked min/max → (B, 1, 1, L)
    let big_pos = Tensor::new(f32::MAX, device)?;
    let big_neg = Tensor::new(f32::MIN, device)?;
    let inv_mask = (mask_4d.neg()? + 1.0)?; // 1 where padded, 0 where valid
    let fill_inf = encoded_f32.broadcast_add(&inv_mask.broadcast_mul(&big_pos)?)?;
    let fill_ninf = encoded_f32.broadcast_add(&inv_mask.broadcast_mul(&big_neg)?)?;

    // amin/amax over (T, D) → (B, L) → (B, 1, 1, L)
    let x_min = fill_inf
        .flatten(1, 2)? // (B, T*D, L)
        .min(1)? // (B, L)
        .unsqueeze(1)?
        .unsqueeze(1)?;
    let x_max = fill_ninf
        .flatten(1, 2)?
        .max(1)?
        .unsqueeze(1)?
        .unsqueeze(1)?;
    let range = (x_max - x_min)?;

    // Normalize: 8 * (x - mean) / (range + eps)
    let normed = (encoded_f32.broadcast_sub(&mean)? * 8.0)?;
    let normed = normed.broadcast_div(&(range + eps)?)?;

    // Flatten layers: (B, T, D, L) → (B, T, D*L)
    let normed = normed.reshape((b, t, d * l))?;

    // Zero out padded positions
    let mask_flat = mask
        .to_dtype(DType::F32)?
        .unsqueeze(2)? // (B, T, 1)
        .expand((b, t, d * l))?;
    normed.broadcast_mul(&mask_flat)
}
