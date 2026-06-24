//! LTX-2.3 diffusion transformer (DiT).
//!
//! 48-layer audio-video transformer with cross-modal attention.
//! Data flow: `Modality` → preprocessor → 48× `BasicAVTransformerBlock` → scale-shift → proj_out
//!
//! - [`model::LTXModel`]: Top-level model (load weights, run forward pass)
//! - [`block::BasicAVTransformerBlock`]: Single transformer block with self-attn, cross-attn, AV cross-attn, FFN
//! - [`modality::Modality`]: Input bundle (latent, timesteps, positions, text context)
//! - [`preprocessor`]: Converts `Modality` into internal `TransformerArgs`

pub mod block;
pub mod modality;
pub mod model;
pub mod preprocessor;
