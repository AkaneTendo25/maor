//! Core types, configuration, and math operations for LTX-2.3 inference.
//!
//! This crate is the foundation layer shared by all other Ner crates.
//! It provides:
//! - [`config`]: Model configuration structs (transformer, VAE, scheduler, guider)
//! - [`types`]: Tensor shape descriptors (`VideoLatentShape`, `AudioLatentShape`)
//! - [`patchify`]: Spatial ↔ token conversion for video and audio latents
//! - [`ops`]: RMS normalization, velocity/denoised conversion, error helpers
//! - [`statistics`]: Per-channel latent normalization/denormalization
//! - [`latent_state`]: Bundle of latent tensors during denoising

pub mod config;
pub mod latent_state;
pub mod ops;
pub mod patchify;
pub mod statistics;
pub mod types;
