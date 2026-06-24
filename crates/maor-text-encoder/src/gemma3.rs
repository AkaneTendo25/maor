use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, Module, VarBuilder};

/// Gemma 3 text model configuration.
#[derive(Debug, Clone)]
pub struct Gemma3Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_local_base_freq: f64,
    pub rope_scaling_factor: f64,
    pub sliding_window: usize,
    pub sliding_window_pattern: usize,
    pub vocab_size: usize,
}

impl Default for Gemma3Config {
    fn default() -> Self {
        Self {
            hidden_size: 3840,
            num_hidden_layers: 48,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 256,
            intermediate_size: 15360,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_local_base_freq: 10_000.0,
            rope_scaling_factor: 8.0,
            sliding_window: 1024,
            sliding_window_pattern: 6,
            vocab_size: 262208,
        }
    }
}

// ── RoPE helpers ──────────────────────────────────────────────────────
// Note: Gemma3 uses standard sequential RoPE (positions 0..T with a single theta),
// while the DiT transformer (maor-nn::rope) uses multi-dimensional fractional RoPE
// (frame/height/width positions normalized by max_pos). The rotation math is the
// same split-half formula, but the frequency computation differs enough that they
// remain separate implementations.

/// Compute inverse frequency for RoPE.
fn compute_inv_freq(
    head_dim: usize,
    theta: f64,
    scaling_factor: f64,
    device: &Device,
) -> Result<Tensor> {
    let half = head_dim / 2;
    let vals: Vec<f32> = (0..half)
        .map(|i| {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            (freq / scaling_factor) as f32
        })
        .collect();
    Tensor::from_vec(vals, half, device)
}

/// Precompute cos/sin tables for positions 0..max_len.
fn precompute_rope_tables(
    max_len: usize,
    inv_freq: &Tensor,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let pos = Tensor::arange(0u32, max_len as u32, device)?.to_dtype(DType::F32)?;
    // (T, 1) × (1, D/2) → (T, D/2)
    let pos = pos.unsqueeze(1)?;
    let freq = inv_freq.to_dtype(DType::F32)?.unsqueeze(0)?;
    let freqs = pos.broadcast_mul(&freq)?;
    Ok((freqs.cos()?, freqs.sin()?))
}

/// Apply split-half RoPE rotation to a 4D tensor.
///
/// x: (B, H, T, D), cos/sin: (1, 1, T, D/2)
fn apply_split_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let d = x.dim(3)?;
    let half = d / 2;
    let cos = cos.to_dtype(x.dtype())?;
    let sin = sin.to_dtype(x.dtype())?;
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    let out1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let out2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
    Tensor::cat(&[&out1, &out2], 3)
}

/// Repeat KV heads for GQA: (B, KVH, T, D) → (B, H, T, D).
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, kv_h, t, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, kv_h, n_rep, t, d))?
        .reshape((b, kv_h * n_rep, t, d))
}

// ── Attention mask ────────────────────────────────────────────────────

/// Build a causal mask (upper-triangle = -inf) with optional sliding window.
///
/// Returns `(1, 1, T, T)` suitable for broadcasting over `(B, H, T, T)`.
/// Uses tensor operations instead of element-wise loops for efficiency at large T.
fn build_causal_mask(
    seq_len: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let rows = Tensor::arange(0u32, seq_len as u32, device)?
        .to_dtype(DType::F32)?
        .unsqueeze(1)?; // (T, 1)
    let cols = Tensor::arange(0u32, seq_len as u32, device)?
        .to_dtype(DType::F32)?
        .unsqueeze(0)?; // (1, T)

    // Upper triangle: mask where col > row (future tokens)
    let mask_value = -10000.0;
    let causal = cols.broadcast_gt(&rows)?;
    let mut mask = causal.to_dtype(DType::F32)?.affine(mask_value, 0.0)?;

    // Sliding window: also mask where row - col > window_size (too-distant past)
    if let Some(w) = sliding_window {
        let distance = rows.broadcast_sub(&cols)?;
        let too_far = distance.broadcast_gt(&Tensor::new(w as f32, device)?)?;
        let sliding = too_far.to_dtype(DType::F32)?.affine(mask_value, 0.0)?;
        mask = (mask + sliding)?;
    }

    mask.unsqueeze(0)?.unsqueeze(0)?.to_dtype(dtype)
}

/// Combine causal mask with padding mask from attention_mask (B, T).
fn combine_masks(causal_mask: &Tensor, attention_mask: &Tensor, dtype: DType) -> Result<Tensor> {
    // attention_mask: (B, T) with 1=valid, 0=padded → convert to additive mask
    let pad_mask = attention_mask.to_dtype(DType::F32)?;
    let pad_mask = ((pad_mask - 1.0)? * 10000.0)?;
    let pad_mask = pad_mask.unsqueeze(1)?.unsqueeze(1)?; // (B, 1, 1, T)
    (causal_mask.to_dtype(DType::F32)?.broadcast_add(&pad_mask))?.to_dtype(dtype)
}

// ── Model layers ──────────────────────────────────────────────────────

/// Per-head RMS normalization: (B, H, T, D) with weight (D,).
fn per_head_rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let mean_sq = x_f32.sqr()?.mean_keepdim(3)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    let normed = x_f32.broadcast_div(&rms)?;
    let w = (weight.to_dtype(DType::F32)? + 1.0)?;
    (normed.broadcast_mul(&w))?.to_dtype(x.dtype())
}

/// Gemma3 RMSNorm uses a zero-initialized learned delta: output *= 1 + weight.
fn gemma_rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let rsqrt = (variance + eps)?.sqrt()?.recip()?;
    let normed = x_f32.broadcast_mul(&rsqrt)?;
    let w = (weight.to_dtype(DType::F32)? + 1.0)?;
    (normed.broadcast_mul(&w))?.to_dtype(x.dtype())
}

/// Gemma 3 self-attention with GQA and per-head norms.
#[derive(Debug)]
struct Gemma3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    rms_norm_eps: f64,
}

impl Gemma3Attention {
    fn new(cfg: &Gemma3Config, vb: VarBuilder) -> Result<Self> {
        let h_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        Ok(Self {
            q_proj: linear_no_bias(cfg.hidden_size, h_dim, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(cfg.hidden_size, kv_dim, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(cfg.hidden_size, kv_dim, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(h_dim, cfg.hidden_size, vb.pp("o_proj"))?,
            q_norm_weight: vb.pp("q_norm").get(cfg.head_dim, "weight")?,
            k_norm_weight: vb.pp("k_norm").get(cfg.head_dim, "weight")?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            scaling: (cfg.head_dim as f64).powf(-0.5),
            rms_norm_eps: cfg.rms_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // (B, T, H*D) → (B, H, T, D)
        let q = q
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head RMS norm on Q and K
        let q = per_head_rms_norm(&q, &self.q_norm_weight, self.rms_norm_eps)?;
        let k = per_head_rms_norm(&k, &self.k_norm_weight, self.rms_norm_eps)?;

        // RoPE — cos/sin are (T, D/2), reshape to (1, 1, T, D/2) for broadcast
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
        let q = apply_split_rope(&q, &cos, &sin)?;
        let k = apply_split_rope(&k, &cos, &sin)?;

        // GQA repeat
        let k = repeat_kv(&k, self.num_kv_groups)?;
        let v = repeat_kv(&v, self.num_kv_groups)?;

        // Scaled dot-product attention (contiguous required for CUDA matmul)
        let attn = (q.contiguous()?.matmul(&k.t()?.contiguous()?)? * self.scaling)?;
        let attn = attn
            .to_dtype(DType::F32)?
            .broadcast_add(&mask.to_dtype(DType::F32)?)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?.to_dtype(q.dtype())?;
        let out = attn.contiguous()?.matmul(&v.contiguous()?)?;

        // (B, H, T, D) → (B, T, H*D)
        let out =
            out.transpose(1, 2)?
                .contiguous()?
                .reshape((b, t, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }
}

/// Gemma 3 gated MLP: GELU(gate) * up → down.
#[derive(Debug)]
struct Gemma3MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Gemma3MLP {
    fn new(cfg: &Gemma3Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = gelu_pytorch_tanh(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Hugging Face Gemma3 uses `hidden_activation="gelu_pytorch_tanh"`.
fn gelu_pytorch_tanh(x: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let x_cube = x_f32.sqr()?.broadcast_mul(&x_f32)?;
    let inner = (&x_f32 + (x_cube * 0.044_715_f64)?)?;
    let inner = (inner * 0.797_884_560_802_865_4_f64)?.tanh()?;
    let y = (x_f32 * 0.5)?.broadcast_mul(&(inner + 1.0)?)?;
    y.to_dtype(dtype)
}

/// Gemma 3 decoder layer.
#[derive(Debug)]
struct Gemma3DecoderLayer {
    input_layernorm: Tensor,
    self_attn: Gemma3Attention,
    post_attention_layernorm: Tensor,
    pre_feedforward_layernorm: Tensor,
    mlp: Gemma3MLP,
    post_feedforward_layernorm: Tensor,
    eps: f64,
}

impl Gemma3DecoderLayer {
    fn new(cfg: &Gemma3Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_layernorm: vb.pp("input_layernorm").get(cfg.hidden_size, "weight")?,
            self_attn: Gemma3Attention::new(cfg, vb.pp("self_attn"))?,
            post_attention_layernorm: vb
                .pp("post_attention_layernorm")
                .get(cfg.hidden_size, "weight")?,
            pre_feedforward_layernorm: vb
                .pp("pre_feedforward_layernorm")
                .get(cfg.hidden_size, "weight")?,
            mlp: Gemma3MLP::new(cfg, vb.pp("mlp"))?,
            post_feedforward_layernorm: vb
                .pp("post_feedforward_layernorm")
                .get(cfg.hidden_size, "weight")?,
            eps: cfg.rms_norm_eps,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // Self-attention with pre/post norms
        let residual = x;
        let h = gemma_rms_norm(x, &self.input_layernorm, self.eps)?;
        let h = self.self_attn.forward(&h, cos, sin, mask)?;
        let h = gemma_rms_norm(&h, &self.post_attention_layernorm, self.eps)?;
        let x = (residual + h)?;

        // MLP with pre/post norms
        let residual = &x;
        let h = gemma_rms_norm(&x, &self.pre_feedforward_layernorm, self.eps)?;
        let h = self.mlp.forward(&h)?;
        let h = gemma_rms_norm(&h, &self.post_feedforward_layernorm, self.eps)?;
        residual + h
    }
}

// ── Full model ────────────────────────────────────────────────────────

/// Gemma 3 text-only model that returns all hidden states for feature extraction.
///
/// Returns 49 tensors: embedding + 47 layer outputs + final normed output.
#[derive(Debug)]
pub struct Gemma3TextModel {
    embed_tokens: Embedding,
    layers: Vec<Gemma3DecoderLayer>,
    norm: Tensor,
    embed_scale: f64,
    global_cos: Tensor,
    global_sin: Tensor,
    local_cos: Tensor,
    local_sin: Tensor,
    sliding_window: usize,
    sliding_window_pattern: usize,
    rms_norm_eps: f64,
}

impl Gemma3TextModel {
    /// Load the model.
    ///
    /// `vb` should be rooted at the language model prefix
    /// (e.g., `language_model.model` in the checkpoint).
    pub fn new(cfg: &Gemma3Config, max_seq_len: usize, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let layers_vb = vb.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Gemma3DecoderLayer::new(cfg, layers_vb.pp(i))?);
        }

        let norm = vb.pp("norm").get(cfg.hidden_size, "weight")?;
        let device = norm.device();

        // Precompute RoPE tables
        let global_inv = compute_inv_freq(
            cfg.head_dim,
            cfg.rope_theta,
            cfg.rope_scaling_factor,
            device,
        )?;
        let local_inv = compute_inv_freq(cfg.head_dim, cfg.rope_local_base_freq, 1.0, device)?;

        let (global_cos, global_sin) = precompute_rope_tables(max_seq_len, &global_inv, device)?;
        let (local_cos, local_sin) = precompute_rope_tables(max_seq_len, &local_inv, device)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            embed_scale: (cfg.hidden_size as f64).sqrt(),
            global_cos,
            global_sin,
            local_cos,
            local_sin,
            sliding_window: cfg.sliding_window,
            sliding_window_pattern: cfg.sliding_window_pattern,
            rms_norm_eps: cfg.rms_norm_eps,
        })
    }

    /// Device the model resides on.
    pub fn device(&self) -> &Device {
        self.norm.device()
    }

    /// Forward pass returning all hidden states (49 for the default 48-layer model).
    ///
    /// - `input_ids`: (B, T) token IDs
    /// - `attention_mask`: (B, T) with 1 = valid, 0 = padding
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Vec<Tensor>> {
        let (_b, t) = input_ids.dims2()?;
        let dtype = self.norm.dtype();
        let device = self.norm.device();

        // Embed + scale
        let mut xs = (self.embed_tokens.forward(input_ids)? * self.embed_scale)?;

        // Build masks (once)
        let global_causal = build_causal_mask(t, None, dtype, device)?;
        let sliding_causal = build_causal_mask(t, Some(self.sliding_window), dtype, device)?;
        let global_mask = combine_masks(&global_causal, attention_mask, dtype)?;
        let sliding_mask = combine_masks(&sliding_causal, attention_mask, dtype)?;

        // Slice RoPE tables to seq_len
        let g_cos = self.global_cos.narrow(0, 0, t)?;
        let g_sin = self.global_sin.narrow(0, 0, t)?;
        let l_cos = self.local_cos.narrow(0, 0, t)?;
        let l_sin = self.local_sin.narrow(0, 0, t)?;

        // Collect hidden states (before each layer, then final normed output)
        let mut hidden_states = Vec::with_capacity(self.layers.len() + 1);

        for (idx, layer) in self.layers.iter().enumerate() {
            hidden_states.push(xs.clone());

            let is_sliding = (idx + 1) % self.sliding_window_pattern > 0;
            let (cos, sin, mask) = if is_sliding {
                (&l_cos, &l_sin, &sliding_mask)
            } else {
                (&g_cos, &g_sin, &global_mask)
            };

            xs = layer.forward(&xs, cos, sin, mask)?;
        }

        // Final RMS norm
        xs = gemma_rms_norm(&xs, &self.norm, self.rms_norm_eps)?;
        hidden_states.push(xs);

        Ok(hidden_states)
    }
}
