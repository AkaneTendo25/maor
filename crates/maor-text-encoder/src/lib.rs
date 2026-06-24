//! Text encoding pipeline for LTX-2.
//!
//! Converts text prompts into conditioning tensors for the diffusion transformer.
//! Pipeline: tokenize → Gemma3 → feature extract → connector → video/audio embeddings.
//!
//! - [`encoder::AVGemmaTextEncoder`]: Top-level API (tokenize + encode + project)
//! - [`gemma3::Gemma3TextModel`]: Gemma 3 language model returning all hidden states
//! - [`feature_extractor::GemmaFeatureExtractor`]: Normalizes and projects multi-layer hidden states
//! - [`connector`]: Perceiver-style cross-attention with learnable registers
//! - [`tokenizer`]: Gemma tokenizer wrapper with left-padding

pub mod connector;
pub mod encoder;
pub mod feature_extractor;
pub mod gemma3;
pub mod tokenizer;
