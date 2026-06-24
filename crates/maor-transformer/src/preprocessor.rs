use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module};

use maor_nn::adaln::AdaLayerNormSingle;
use maor_nn::rope::{precompute_freqs_cis, LTXRopeType};
use maor_nn::text_projection::PixArtAlphaTextProjection;

use crate::modality::{Modality, TransformerArgs};

/// Preprocesses a single modality's inputs into TransformerArgs.
///
/// Owns patchify projection, AdaLN, and caption projection layers.
#[derive(Debug)]
pub struct TransformerArgsPreprocessor {
    patchify_proj: Linear,
    adaln: AdaLayerNormSingle,
    caption_projection: Option<PixArtAlphaTextProjection>,
    prompt_adaln: Option<AdaLayerNormSingle>,
    pub inner_dim: usize,
    pub max_pos: Vec<usize>,
    pub num_attention_heads: usize,
    pub use_middle_indices_grid: bool,
    pub timestep_scale_multiplier: usize,
    pub positional_embedding_theta: f64,
    pub rope_type: LTXRopeType,
}

impl TransformerArgsPreprocessor {
    pub fn new(
        patchify_proj: Linear,
        adaln: AdaLayerNormSingle,
        caption_projection: Option<PixArtAlphaTextProjection>,
        prompt_adaln: Option<AdaLayerNormSingle>,
        inner_dim: usize,
        max_pos: Vec<usize>,
        num_attention_heads: usize,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: usize,
        positional_embedding_theta: f64,
        rope_type: LTXRopeType,
    ) -> Self {
        Self {
            patchify_proj,
            adaln,
            caption_projection,
            prompt_adaln,
            inner_dim,
            max_pos,
            num_attention_heads,
            use_middle_indices_grid,
            timestep_scale_multiplier,
            positional_embedding_theta,
            rope_type,
        }
    }

    /// Prepare timestep embeddings: scale -> flatten -> adaln -> reshape.
    ///
    /// Returns (timestep_embedding, embedded_timestep).
    /// timestep_embedding: (B, T, coeff*inner_dim)
    /// embedded_timestep: (B, T, inner_dim)
    fn prepare_timestep(
        &self,
        timestep: &Tensor,
        batch_size: usize,
        hidden_dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        self.prepare_timestep_with_adaln(
            timestep,
            batch_size,
            hidden_dtype,
            &self.adaln,
            self.timestep_scale_multiplier as f64,
        )
    }

    fn prepare_timestep_with_adaln(
        &self,
        timestep: &Tensor,
        batch_size: usize,
        hidden_dtype: DType,
        adaln: &AdaLayerNormSingle,
        scale: f64,
    ) -> Result<(Tensor, Tensor)> {
        let scaled = (timestep * scale)?;
        let flat = scaled.flatten_all()?;
        let (ts_out, embedded) = adaln.forward_with_embedded(&flat, hidden_dtype)?;
        let ts_dim = *ts_out
            .dims()
            .last()
            .ok_or(candle_core::Error::Msg("empty timestep dims".into()))?;
        let emb_dim = *embedded
            .dims()
            .last()
            .ok_or(candle_core::Error::Msg("empty embedded dims".into()))?;
        let n_tokens = ts_out.elem_count() / (batch_size * ts_dim);
        let ts_out = ts_out.reshape(&[batch_size, n_tokens, ts_dim])?;
        let embedded = embedded.reshape(&[batch_size, n_tokens, emb_dim])?;
        Ok((ts_out, embedded))
    }

    /// Project text context to inner_dim.
    fn prepare_context(&self, context: &Tensor, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.dims()[0];
        let x_dim = *x
            .dims()
            .last()
            .ok_or(candle_core::Error::Msg("empty x dims".into()))?;
        let projected = match &self.caption_projection {
            Some(proj) => proj.forward(context)?,
            None => context.clone(),
        };
        let seq_len = projected.elem_count() / (batch_size * x_dim);
        projected.reshape(&[batch_size, seq_len, x_dim])
    }

    /// Compute RoPE positional embeddings for given positions.
    pub fn prepare_positional_embeddings(
        &self,
        positions: &Tensor,
        inner_dim: usize,
        max_pos: &[usize],
        use_middle_indices_grid: bool,
        num_attention_heads: usize,
        x_dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        precompute_freqs_cis(
            positions,
            inner_dim,
            x_dtype,
            self.positional_embedding_theta,
            max_pos,
            use_middle_indices_grid,
            num_attention_heads,
            self.rope_type,
            positions.device(),
        )
    }

    /// Full preprocessing: patchify -> timestep -> context -> mask -> RoPE.
    pub fn prepare(&self, modality: &Modality) -> Result<TransformerArgs> {
        let x = self.patchify_proj.forward(&modality.latent)?;
        let batch_size = x.dims()[0];
        let hidden_dtype = modality.latent.dtype();

        let (timesteps, embedded_timestep) =
            self.prepare_timestep(&modality.timesteps, batch_size, hidden_dtype)?;

        let prompt_timestep = match (&self.prompt_adaln, modality.sigma.as_ref()) {
            (Some(prompt_adaln), Some(sigma)) => Some(
                self.prepare_timestep_with_adaln(
                    sigma,
                    batch_size,
                    hidden_dtype,
                    prompt_adaln,
                    self.timestep_scale_multiplier as f64,
                )?
                .0,
            ),
            _ => None,
        };

        let context = self.prepare_context(&modality.context, &x)?;

        let context_mask = prepare_attention_mask(modality.context_mask.as_ref(), hidden_dtype)?;

        let pe = self.prepare_positional_embeddings(
            &modality.positions,
            self.inner_dim,
            &self.max_pos,
            self.use_middle_indices_grid,
            self.num_attention_heads,
            hidden_dtype,
        )?;

        Ok(TransformerArgs {
            x,
            context,
            context_mask,
            timesteps,
            embedded_timestep,
            positional_embeddings: pe,
            cross_positional_embeddings: None,
            cross_scale_shift_timestep: None,
            cross_gate_timestep: None,
            prompt_timestep,
            enabled: modality.enabled,
        })
    }
}

/// Extends preprocessing for multi-modal (audio-video) scenarios.
///
/// Adds cross-attention positional embeddings and cross-attention timestep embeddings.
#[derive(Debug)]
pub struct MultiModalTransformerArgsPreprocessor {
    pub simple: TransformerArgsPreprocessor,
    cross_scale_shift_adaln: AdaLayerNormSingle,
    cross_gate_adaln: AdaLayerNormSingle,
    cross_pe_max_pos: usize,
    audio_cross_attention_dim: usize,
    av_ca_timestep_scale_multiplier: f64,
}

impl MultiModalTransformerArgsPreprocessor {
    pub fn new(
        simple: TransformerArgsPreprocessor,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        cross_pe_max_pos: usize,
        audio_cross_attention_dim: usize,
        av_ca_timestep_scale_multiplier: f64,
    ) -> Self {
        Self {
            simple,
            cross_scale_shift_adaln,
            cross_gate_adaln,
            cross_pe_max_pos,
            audio_cross_attention_dim,
            av_ca_timestep_scale_multiplier,
        }
    }

    fn prepare_cross_attention_timestep(
        &self,
        timestep: &Tensor,
        batch_size: usize,
        hidden_dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let ts_scale = self.simple.timestep_scale_multiplier;
        let scaled = (timestep * ts_scale as f64)?;
        let av_ca_factor = self.av_ca_timestep_scale_multiplier / ts_scale as f64;

        // Scale-shift timestep
        let (ss_ts, _) = self
            .cross_scale_shift_adaln
            .forward_with_embedded(&scaled.flatten_all()?, hidden_dtype)?;
        let ss_dim = *ss_ts
            .dims()
            .last()
            .ok_or(candle_core::Error::Msg("empty scale-shift dims".into()))?;
        let n_tok = ss_ts.elem_count() / (batch_size * ss_dim);
        let ss_ts = ss_ts.reshape(&[batch_size, n_tok, ss_dim])?;

        // Gate timestep (with av_ca_factor)
        let gate_input = (scaled.flatten_all()? * av_ca_factor)?;
        let (gate_ts, _) = self
            .cross_gate_adaln
            .forward_with_embedded(&gate_input, hidden_dtype)?;
        let gate_dim = *gate_ts
            .dims()
            .last()
            .ok_or(candle_core::Error::Msg("empty gate dims".into()))?;
        let n_tok2 = gate_ts.elem_count() / (batch_size * gate_dim);
        let gate_ts = gate_ts.reshape(&[batch_size, n_tok2, gate_dim])?;

        Ok((ss_ts, gate_ts))
    }

    pub fn prepare(&self, modality: &Modality) -> Result<TransformerArgs> {
        self.prepare_with_cross_modality(modality, None)
    }

    pub fn prepare_with_cross_modality(
        &self,
        modality: &Modality,
        cross_modality: Option<&Modality>,
    ) -> Result<TransformerArgs> {
        let mut args = self.simple.prepare(modality)?;
        let batch_size = args.x.dims()[0];
        let hidden_dtype = modality.latent.dtype();

        // Cross-positional embeddings: use only temporal dimension
        let temporal_positions = modality.positions.narrow(1, 0, 1)?;
        let cross_pe = self.simple.prepare_positional_embeddings(
            &temporal_positions,
            self.audio_cross_attention_dim,
            &[self.cross_pe_max_pos],
            true,
            self.simple.num_attention_heads,
            hidden_dtype,
        )?;
        args.cross_positional_embeddings = Some(cross_pe);

        // Cross-attention timestep embeddings. LTX-2.3 uses the cross
        // modality's scalar sigma here when available, not the local per-token
        // denoise mask.
        let cross_timestep = if let Some(sigma) = cross_modality.and_then(|m| m.sigma.as_ref()) {
            let sigma = if sigma.dims().len() == 1 {
                sigma.reshape((batch_size, 1))?
            } else {
                sigma.clone()
            };
            sigma.expand(modality.timesteps.dims())?
        } else {
            modality.timesteps.clone()
        };
        let (cross_ss, cross_gate) =
            self.prepare_cross_attention_timestep(&cross_timestep, batch_size, hidden_dtype)?;
        args.cross_scale_shift_timestep = Some(cross_ss);
        args.cross_gate_timestep = Some(cross_gate);

        Ok(args)
    }
}

/// Enum wrapping both preprocessor variants.
#[derive(Debug)]
pub enum ArgsPreprocessor {
    Simple(TransformerArgsPreprocessor),
    MultiModal(MultiModalTransformerArgsPreprocessor),
}

impl ArgsPreprocessor {
    pub fn prepare(&self, modality: &Modality) -> Result<TransformerArgs> {
        self.prepare_with_cross_modality(modality, None)
    }

    pub fn prepare_with_cross_modality(
        &self,
        modality: &Modality,
        cross_modality: Option<&Modality>,
    ) -> Result<TransformerArgs> {
        match self {
            Self::Simple(p) => p.prepare(modality),
            Self::MultiModal(p) => p.prepare_with_cross_modality(modality, cross_modality),
        }
    }
}

/// Convert attention mask from bool/int to float format.
///
/// Input: (B, seq_len) with 1=attend, 0=mask
/// Output: (B, 1, 1, seq_len) with 0.0=attend, -large=mask
pub fn prepare_attention_mask(mask: Option<&Tensor>, dtype: DType) -> Result<Option<Tensor>> {
    let mask = match mask {
        None => return Ok(None),
        Some(m) => m,
    };
    // If already additive float mask, pass through. A 2D float mask is still a
    // binary mask from the text encoder and must be converted.
    if matches!(
        mask.dtype(),
        DType::F32 | DType::F16 | DType::BF16 | DType::F64
    ) && mask.dims().len() != 2
    {
        return Ok(Some(mask.clone()));
    }

    // Convert: (mask - 1) * big_value => 0 for attend, -big for mask
    let b = mask.dims()[0];
    let seq_len = *mask
        .dims()
        .last()
        .ok_or(candle_core::Error::Msg("empty mask dims".into()))?;
    let float_mask = (mask.to_dtype(DType::F32)? - 1.0)?;
    let result = (float_mask * 10000.0)?.to_dtype(dtype)?;
    let result = result.reshape(&[b, 1, 1, seq_len])?;
    Ok(Some(result))
}
