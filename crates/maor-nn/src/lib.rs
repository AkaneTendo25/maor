//! Reusable neural network primitives for LTX-2.3.
//!
//! Building blocks shared across the transformer, VAE, and text encoder:
//! - [`attention`]: Scaled dot-product attention with RoPE and optional cross-attention
//! - [`rope`]: Rotary positional embeddings (split and interleaved variants)
//! - [`conv3d`]: 3D convolution decomposed into 2D ops (candle only has native 2D)
//! - [`adaln`]: Adaptive Layer Norm (adaLN-single from PixArt-Alpha)
//! - [`timestep_embedding`]: Sinusoidal timestep embeddings and MLP projections
//! - [`activation`]: SiLU and GELU activation functions
//! - [`feed_forward`]: Gated feed-forward network (SiLU gate)
//! - [`pixel_norm`]: Per-pixel L2 normalization
//! - [`text_projection`]: PixArt-Alpha text projection module
//! - [`lora`]: Runtime LoRA weight merging

pub mod activation;
pub mod adaln;
pub mod attention;
pub mod conv3d;
pub mod feed_forward;
pub mod lora;
pub mod pixel_norm;
pub mod rope;
pub mod text_projection;
pub mod timestep_embedding;
