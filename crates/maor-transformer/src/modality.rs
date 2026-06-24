use candle_core::Tensor;

/// Input data for a single modality (video or audio) in the transformer.
///
/// Bundles latent tokens, timestep embeddings, positional info, and text context.
#[derive(Debug, Clone)]
pub struct Modality {
    /// Patchified latent tokens. Shape: (B, T, D)
    pub latent: Tensor,
    /// Per-token timesteps (sigma * denoise_mask). Shape: (B, T)
    pub timesteps: Tensor,
    /// Positional indices. Shape: (B, ndim, T, 2) for video (ndim=3)
    pub positions: Tensor,
    /// Text conditioning context. Shape: (B, seq_len, context_dim)
    pub context: Tensor,
    /// Whether this modality is active in this forward pass.
    pub enabled: bool,
    /// Optional attention mask for context. Shape: (B, 1, 1, seq_len) or None.
    pub context_mask: Option<Tensor>,
    /// Optional sample-level sigma for 2.3 prompt AdaLN and AV cross-attention conditioning.
    /// Shape: (B,) or (B, 1).
    pub sigma: Option<Tensor>,
}

/// Preprocessed args for a single modality inside the transformer blocks.
#[derive(Debug, Clone)]
pub struct TransformerArgs {
    /// Hidden state being processed. Shape: (B, T, inner_dim)
    pub x: Tensor,
    /// Projected text context. Shape: (B, seq_len, inner_dim)
    pub context: Tensor,
    /// Prepared attention mask (float, with -inf for masked positions).
    pub context_mask: Option<Tensor>,
    /// AdaLN timestep embedding (scale/shift/gate). Shape: (B, n_tokens, 6*dim)
    pub timesteps: Tensor,
    /// Raw embedded timestep for output modulation. Shape: (B, n_tokens, dim)
    pub embedded_timestep: Tensor,
    /// RoPE positional embeddings (cos, sin) for self-attention.
    pub positional_embeddings: (Tensor, Tensor),
    /// RoPE for cross-modal attention (None if single-modality).
    pub cross_positional_embeddings: Option<(Tensor, Tensor)>,
    /// Cross-attention scale/shift timestep embedding.
    pub cross_scale_shift_timestep: Option<Tensor>,
    /// Cross-attention gate timestep embedding.
    pub cross_gate_timestep: Option<Tensor>,
    /// Prompt AdaLN timestep embedding for 2.3 text cross-attention.
    pub prompt_timestep: Option<Tensor>,
    /// Whether this modality is active.
    pub enabled: bool,
}
