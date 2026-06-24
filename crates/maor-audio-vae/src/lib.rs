//! Audio VAE decoder and vocoder for LTX-2.
//!
//! Two-stage audio decoding:
//! 1. [`decoder::AudioDecoder`]: Latents `(B, 8, T, mel)` → mel spectrogram
//! 2. [`vocoder::Vocoder`]: Mel spectrogram → PCM waveform
//!
//! - [`causal_conv2d`]: Causal 2D convolution with configurable causality axis
//! - [`resnet`]: Audio ResNet blocks
//! - [`attention`]: Self-attention for audio decoder
//! - [`upsample`]: Audio upsampling layers

pub mod attention;
pub mod causal_conv2d;
pub mod decoder;
pub mod resnet;
pub mod upsample;
pub mod vocoder;
