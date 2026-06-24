//! Video VAE decoder for LTX-2.3.
//!
//! Decodes video latents `(B, 128, F, H, W)` to RGB frames `(B, 3, F', H', W')`.
//! Scale factors: time×8, height×32, width×32.
//!
//! - [`decoder::VideoDecoder`]: Full decoder with unpatchify, resnet blocks, and upsampling
//! - [`resnet`]: ResNet blocks with optional timestep conditioning (AdaGN)
//! - [`upsample`]: Depth-to-space and interpolation upsampling

pub mod decoder;
pub mod resnet;
pub mod upsample;
pub mod upsampler;
