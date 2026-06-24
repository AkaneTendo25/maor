use candle_core::{DType, Module, Result, Tensor};
use candle_nn::VarBuilder;

use maor_core::config::TransformerConfig;
use maor_core::ops::rms_norm;
use maor_nn::attention::Attention;
use maor_nn::feed_forward::FeedForward;
use maor_nn::rope::{precompute_freqs_cis, LTXRopeType};

/// A single transformer block for the 1D connector.
///
/// RMS norm → self-attention → skip → RMS norm → FFN → skip.
#[derive(Debug)]
struct BasicTransformerBlock1D {
    attn: Attention,
    ff: FeedForward,
}

impl BasicTransformerBlock1D {
    fn new(
        dim: usize,
        heads: usize,
        dim_head: usize,
        rope_type: LTXRopeType,
        apply_gated_attention: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let attn = Attention::new(
            dim,
            None, // self-attention
            heads,
            dim_head,
            1e-8,
            rope_type,
            apply_gated_attention,
            vb.pp("attn1"),
        )?;
        let ff = FeedForward::new(dim, dim, 4, vb.pp("ff"))?;
        Ok(Self { attn, ff })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        pe: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        // Self-attention with pre-norm
        let normed = rms_norm(x, None, 1e-8)?;
        let attn_out = self.attn.forward(&normed, None, mask, pe, None)?;
        let x = (attn_out + x)?;

        // FFN with pre-norm
        let normed = rms_norm(&x, None, 1e-8)?;
        let ff_out = self.ff.forward(&normed)?;
        ff_out + x
    }
}

/// 1D embeddings connector: replaces padding with learnable registers,
/// runs 2-layer transformer, outputs context embeddings.
///
/// Used separately for video and audio context extraction from the same
/// Gemma feature-extracted embeddings.
#[derive(Debug)]
pub struct Embeddings1DConnector {
    blocks: Vec<BasicTransformerBlock1D>,
    learnable_registers: Option<Tensor>,
    num_learnable_registers: usize,
    inner_dim: usize,
    num_heads: usize,
    theta: f64,
    max_pos: Vec<usize>,
    rope_type: LTXRopeType,
    double_precision_rope: bool,
}

/// Debug trace from an embeddings connector forward pass.
#[derive(Debug)]
pub struct Embeddings1DConnectorTrace {
    pub after_registers: Tensor,
    pub block_outputs: Vec<Tensor>,
    pub output: Tensor,
    pub mask: Tensor,
}

impl Embeddings1DConnector {
    pub fn new(
        attention_head_dim: usize,
        num_attention_heads: usize,
        num_layers: usize,
        theta: f64,
        max_pos: Vec<usize>,
        num_learnable_registers: usize,
        rope_type: LTXRopeType,
        double_precision_rope: bool,
        apply_gated_attention: bool,
        blocks_key: &str,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner_dim = num_attention_heads * attention_head_dim;

        let mut blocks = Vec::with_capacity(num_layers);
        let blocks_vb = vb.pp(blocks_key);
        for i in 0..num_layers {
            blocks.push(BasicTransformerBlock1D::new(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                rope_type,
                apply_gated_attention,
                blocks_vb.pp(i),
            )?);
        }

        let learnable_registers = if num_learnable_registers > 0 {
            Some(vb.get(&[num_learnable_registers, inner_dim], "learnable_registers")?)
        } else {
            None
        };

        Ok(Self {
            blocks,
            learnable_registers,
            num_learnable_registers,
            inner_dim,
            num_heads: num_attention_heads,
            theta,
            max_pos,
            rope_type,
            double_precision_rope,
        })
    }

    /// Default LTX-2.3 connector: 30 heads x 128 dim = 3840, 2 layers, 128 registers.
    pub fn new_default(vb: VarBuilder) -> Result<Self> {
        Self::new(
            128,
            30,
            2,
            10000.0,
            vec![1],
            128,
            LTXRopeType::Split,
            false,
            false,
            "transformer_1d_blocks",
            vb,
        )
    }

    pub fn new_video_from_config(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        Self::new(
            cfg.connector_attention_head_dim,
            cfg.connector_num_attention_heads,
            cfg.connector_num_layers,
            cfg.positional_embedding_theta,
            cfg.connector_positional_embedding_max_pos.clone(),
            cfg.connector_num_learnable_registers,
            parse_rope_type(&cfg.rope_type),
            cfg.double_precision_rope(),
            cfg.connector_apply_gated_attention,
            "transformer_1d_blocks",
            vb,
        )
    }

    pub fn new_audio_from_config(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let attention_head_dim = if cfg.audio_connector_attention_head_dim == 0 {
            cfg.connector_attention_head_dim
        } else {
            cfg.audio_connector_attention_head_dim
        };
        let num_attention_heads = if cfg.audio_connector_num_attention_heads == 0 {
            cfg.connector_num_attention_heads
        } else {
            cfg.audio_connector_num_attention_heads
        };
        let num_layers = if cfg.audio_connector_num_layers == 0 {
            cfg.connector_num_layers
        } else {
            cfg.audio_connector_num_layers
        };
        Self::new(
            attention_head_dim,
            num_attention_heads,
            num_layers,
            cfg.positional_embedding_theta,
            cfg.connector_positional_embedding_max_pos.clone(),
            cfg.connector_num_learnable_registers,
            parse_rope_type(&cfg.rope_type),
            cfg.double_precision_rope(),
            cfg.connector_apply_gated_attention,
            "transformer_1d_blocks",
            vb,
        )
    }

    /// Forward pass.
    ///
    /// - `hidden_states`: (B, T, D) projected features from Gemma
    /// - `attention_mask`: (B, T) with 1 = valid, 0 = padded
    ///
    /// Returns (processed_features, updated_attention_mask).
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let trace = self.forward_trace(hidden_states, attention_mask)?;
        Ok((trace.output, trace.mask))
    }

    /// Forward pass with intermediate tensors for diagnostics.
    pub fn forward_trace(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Embeddings1DConnectorTrace> {
        let dtype = hidden_states.dtype();

        // Convert to additive mask: (B, T) → (B, 1, T, T)
        let additive_mask = to_additive_mask(attention_mask, dtype)?;

        // Replace padded positions with learnable registers
        let (mut hs, mask) = if let Some(ref registers) = self.learnable_registers {
            replace_padded_with_registers(
                hidden_states,
                &additive_mask,
                registers,
                self.num_learnable_registers,
            )?
        } else {
            (hidden_states.clone(), additive_mask)
        };
        let after_registers = hs.clone();

        // Compute 1D RoPE frequencies
        let seq_len = hs.dim(1)?;
        let device = hs.device();
        let indices = Tensor::arange(0u32, seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .unsqueeze(0)?
            .unsqueeze(0)?; // (1, 1, T)
        let pe = precompute_freqs_cis(
            &indices,
            self.inner_dim,
            dtype,
            self.theta,
            &self.max_pos,
            self.double_precision_rope,
            self.num_heads,
            self.rope_type,
            device,
        )?;

        // Run transformer blocks
        let mut block_outputs = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            hs = block.forward(&hs, Some(&mask), Some((&pe.0, &pe.1)))?;
            block_outputs.push(hs.clone());
        }

        // Final RMS norm
        hs = rms_norm(&hs, None, 1e-8)?;

        Ok(Embeddings1DConnectorTrace {
            after_registers,
            block_outputs,
            output: hs,
            mask,
        })
    }
}

fn parse_rope_type(value: &str) -> LTXRopeType {
    match value {
        "interleaved" => LTXRopeType::Interleaved,
        _ => LTXRopeType::Split,
    }
}

/// Convert (B, T) binary mask to additive attention mask (B, 1, 1, T).
///
/// Valid=0, padded = -large_value.
fn to_additive_mask(attention_mask: &Tensor, dtype: DType) -> Result<Tensor> {
    // Build in f32 with a finite bias. Multiplying BF16 zero by a value that
    // rounds to infinity can produce NaNs, which makes every token look padded
    // during register replacement.
    let mask = attention_mask.to_dtype(DType::F32)?;
    let mask = ((mask - 1.0)? * 10000.0)?;
    mask.to_dtype(dtype)?.unsqueeze(1)?.unsqueeze(1) // (B, 1, 1, T)
}

/// Replace padded positions with tiled learnable registers.
///
/// Moves valid tokens to the front, fills remaining positions with registers,
/// and sets the attention mask to all-valid.
fn replace_padded_with_registers(
    hidden_states: &Tensor,
    additive_mask: &Tensor,
    registers: &Tensor,
    num_registers: usize,
) -> Result<(Tensor, Tensor)> {
    let (b, t, _d) = hidden_states.dims3()?;
    let dtype = hidden_states.dtype();
    let device = hidden_states.device();

    if t % num_registers != 0 {
        return Err(candle_core::Error::Msg(format!(
            "seq_len {} must be divisible by num_registers {}",
            t, num_registers
        )));
    }

    let num_dup = t / num_registers;
    // Tile registers: (num_registers, D) → (T, D)
    let tiled_regs = registers.to_dtype(dtype)?.repeat((num_dup, 1))?;

    // Binary mask from additive mask: (B, 1, 1, T) → (B, T)
    let binary = additive_mask
        .squeeze(1)?
        .squeeze(1)?
        .ge(-9000.0_f64)?
        .to_dtype(DType::F32)?;

    // For batch_size=1 (common case): extract non-padded tokens and rearrange
    // For simplicity, process each batch element independently
    let mut result_slices = Vec::with_capacity(b);
    for bi in 0..b {
        let hs_i = hidden_states.narrow(0, bi, 1)?.squeeze(0)?; // (T, D)
        let mask_i = binary.narrow(0, bi, 1)?.squeeze(0)?; // (T,)

        // Count valid tokens
        let valid_count = mask_i.sum_all()?.to_scalar::<f32>()? as usize;
        // Gather valid tokens and reorder: valid first, padded after
        // (valid=1 first, padded=0 after)
        // Instead, use index-based approach:
        let mut indices = Vec::with_capacity(t);
        let mask_vec: Vec<f32> = mask_i.to_vec1()?;
        // Valid token indices first
        for (j, &m) in mask_vec.iter().enumerate() {
            if m > 0.5 {
                indices.push(j as u32);
            }
        }
        // Padded indices after
        for (j, &m) in mask_vec.iter().enumerate() {
            if m <= 0.5 {
                indices.push(j as u32);
            }
        }
        let idx_tensor = Tensor::from_vec(indices, t, device)?;
        let reordered = hs_i.index_select(&idx_tensor, 0)?; // (T, D)

        // Blend: valid positions get reordered tokens, rest get registers
        // Create blend mask: 1 for first valid_count positions, 0 for rest
        let mut blend = vec![0.0f32; t];
        for v in blend.iter_mut().take(valid_count) {
            *v = 1.0;
        }
        let blend_mask = Tensor::from_vec(blend, (t, 1), device)?.to_dtype(dtype)?;
        let inv_blend = (blend_mask.neg()? + 1.0)?;

        let blended = (reordered.to_dtype(dtype)?.broadcast_mul(&blend_mask)?
            + tiled_regs.broadcast_mul(&inv_blend)?)?;

        result_slices.push(blended.unsqueeze(0)?);
    }

    let result = Tensor::cat(&result_slices, 0)?;

    // All-valid attention mask
    let new_mask = Tensor::zeros((b, 1, 1, t), dtype, device)?;

    Ok((result, new_mask))
}
