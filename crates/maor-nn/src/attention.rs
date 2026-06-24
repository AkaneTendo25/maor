use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::rope::{apply_rotary_emb, LTXRopeType};
use maor_core::ops::rms_norm;

use crate::lora::{self, LoraConfig};

/// Multi-head attention with RMSNorm on Q/K and optional RoPE.
///
/// Supports self-attention (context=None) and cross-attention.
#[derive(Debug)]
pub struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    to_gate_logits: Option<Linear>,
    heads: usize,
    dim_head: usize,
    norm_eps: f64,
    rope_type: LTXRopeType,
}

impl Attention {
    pub fn new(
        query_dim: usize,
        context_dim: Option<usize>,
        heads: usize,
        dim_head: usize,
        norm_eps: f64,
        rope_type: LTXRopeType,
        apply_gated_attention: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_with_lora(
            query_dim,
            context_dim,
            heads,
            dim_head,
            norm_eps,
            rope_type,
            apply_gated_attention,
            vb,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_lora(
        query_dim: usize,
        context_dim: Option<usize>,
        heads: usize,
        dim_head: usize,
        norm_eps: f64,
        rope_type: LTXRopeType,
        apply_gated_attention: bool,
        vb: VarBuilder,
        lora: Option<&LoraConfig>,
    ) -> Result<Self> {
        let inner_dim = dim_head * heads;
        let context_dim = context_dim.unwrap_or(query_dim);

        let to_q_lora = lora.map(|l| l.pp("to_q"));
        let to_k_lora = lora.map(|l| l.pp("to_k"));
        let to_v_lora = lora.map(|l| l.pp("to_v"));
        let to_out_lora = lora.map(|l| l.pp("to_out").pp(0));
        let gate_lora = lora.map(|l| l.pp("to_gate_logits"));

        let to_q = lora::linear(query_dim, inner_dim, vb.pp("to_q"), to_q_lora.as_ref())?;
        let to_k = lora::linear(context_dim, inner_dim, vb.pp("to_k"), to_k_lora.as_ref())?;
        let to_v = lora::linear(context_dim, inner_dim, vb.pp("to_v"), to_v_lora.as_ref())?;
        let to_out = lora::linear(
            inner_dim,
            query_dim,
            vb.pp("to_out.0"),
            to_out_lora.as_ref(),
        )?;

        // RMSNorm weights for Q and K. Checkpoints may use either naming pair.
        let q_norm_weight = if vb.pp("norm_q").contains_tensor("weight") {
            vb.pp("norm_q").get(inner_dim, "weight")?
        } else {
            vb.pp("q_norm").get(inner_dim, "weight")?
        };
        let k_norm_weight = if vb.pp("norm_k").contains_tensor("weight") {
            vb.pp("norm_k").get(inner_dim, "weight")?
        } else {
            vb.pp("k_norm").get(inner_dim, "weight")?
        };

        let to_gate_logits = if apply_gated_attention {
            Some(lora::linear(
                query_dim,
                heads,
                vb.pp("to_gate_logits"),
                gate_lora.as_ref(),
            )?)
        } else {
            None
        };

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            q_norm_weight,
            k_norm_weight,
            to_gate_logits,
            heads,
            dim_head,
            norm_eps,
            rope_type,
        })
    }

    /// Forward pass.
    ///
    /// - `x`: query input (B, T, query_dim)
    /// - `context`: key/value input for cross-attention, or None for self-attention
    /// - `mask`: optional attention mask
    /// - `pe`: (cos, sin) RoPE frequencies for Q (and K if k_pe is None)
    /// - `k_pe`: separate RoPE for K (for cross-attention with different positions)
    pub fn forward(
        &self,
        x: &Tensor,
        context: Option<&Tensor>,
        mask: Option<&Tensor>,
        pe: Option<(&Tensor, &Tensor)>,
        k_pe: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let ctx = context.unwrap_or(x);

        let mut q = self.to_q.forward(x)?;
        let mut k = self.to_k.forward(ctx)?;
        let v = self.to_v.forward(ctx)?;

        // RMSNorm on Q and K
        q = rms_norm(&q, Some(&self.q_norm_weight), self.norm_eps)?;
        k = rms_norm(&k, Some(&self.k_norm_weight), self.norm_eps)?;

        // Apply RoPE
        if let Some((cos, sin)) = pe {
            q = apply_rotary_emb(&q, cos, sin, self.rope_type)?;
            let (k_cos, k_sin) = k_pe.unwrap_or((cos, sin));
            k = apply_rotary_emb(&k, k_cos, k_sin, self.rope_type)?;
        }

        // Scaled dot-product attention
        let out = sdpa(&q, &k, &v, self.heads, self.dim_head, mask)?;

        // Optional per-head gating
        let out = if let Some(ref gate_linear) = self.to_gate_logits {
            let gate_logits = gate_linear.forward(x)?; // (B, T, H)
            let (b, t, _) = out.dims3()?;
            let out_4d = out.reshape((b, t, self.heads, self.dim_head))?;
            // 2 * sigmoid(gate) so zero-init → identity (2 * 0.5 = 1.0)
            let gates = (candle_nn::ops::sigmoid(&gate_logits)? * 2.0)?;
            let gates = gates.unsqueeze(3)?; // (B, T, H, 1)
            let gated = out_4d.broadcast_mul(&gates)?;
            gated.reshape((b, t, self.heads * self.dim_head))?
        } else {
            out
        };

        self.to_out.forward(&out)
    }
}

/// Scaled dot-product attention.
///
/// Input shapes: (B, T, H*D) → reshape to (B, H, T, D) → SDPA → (B, T, H*D)
pub fn sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    heads: usize,
    dim_head: usize,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (b, t_q, _) = q.dims3()?;
    let (_, t_k, _) = k.dims3()?;
    let out_dtype = q.dtype();

    // Reshape to (B, H, T, D) — contiguous required for CUDA matmul
    let q = q
        .reshape((b, t_q, heads, dim_head))?
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(DType::F32)?;
    let k = k
        .reshape((b, t_k, heads, dim_head))?
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(DType::F32)?;
    let v = v
        .reshape((b, t_k, heads, dim_head))?
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(DType::F32)?;

    let scale = 1.0 / (dim_head as f64).sqrt();
    let k_t = k.t()?.contiguous()?;
    let mask = prepare_sdpa_mask(mask, DType::F32)?;
    let score_elems = b
        .saturating_mul(heads)
        .saturating_mul(t_q)
        .saturating_mul(t_k);

    let out = if score_elems <= 64 * 1024 * 1024 {
        sdpa_chunk(&q, &k_t, &v, mask.as_ref(), 0, t_q, scale)?
    } else {
        let target_score_elems = 128 * 1024 * 1024usize;
        let denom = b.saturating_mul(heads).saturating_mul(t_k).max(1);
        let chunk = (target_score_elems / denom).clamp(1, t_q);
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < t_q {
            let len = (t_q - start).min(chunk);
            chunks.push(sdpa_chunk(&q, &k_t, &v, mask.as_ref(), start, len, scale)?);
            start += len;
        }
        let refs: Vec<&Tensor> = chunks.iter().collect();
        Tensor::cat(&refs, 2)?
    };

    // Reshape back: (B, H, T, D) → (B, T, H*D)
    let out = out.transpose(1, 2)?.contiguous()?;
    out.reshape((b, t_q, heads * dim_head))?.to_dtype(out_dtype)
}

fn prepare_sdpa_mask(mask: Option<&Tensor>, dtype: DType) -> Result<Option<Tensor>> {
    let Some(mask) = mask else {
        return Ok(None);
    };
    let mask = if mask.dims().len() == 2 {
        mask.unsqueeze(0)?.unsqueeze(0)?
    } else if mask.dims().len() == 3 {
        mask.unsqueeze(1)?
    } else {
        mask.clone()
    };
    mask.to_dtype(dtype).map(Some)
}

fn sdpa_chunk(
    q: &Tensor,
    k_t: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    q_start: usize,
    q_len: usize,
    scale: f64,
) -> Result<Tensor> {
    let q = q.narrow(2, q_start, q_len)?;
    let scores = (q.matmul(k_t)? * scale)?;
    let scores = if let Some(mask) = mask {
        let mask = if mask.dims().len() == 4 && mask.dim(2)? > 1 {
            mask.narrow(2, q_start, q_len)?
        } else {
            mask.clone()
        };
        scores.broadcast_add(&mask)?
    } else {
        scores
    };
    let weights = candle_nn::ops::softmax_last_dim(&scores)?;
    weights.contiguous()?.matmul(v)
}
