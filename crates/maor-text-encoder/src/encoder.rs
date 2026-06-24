use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;

use maor_core::config::TransformerConfig;

use crate::connector::Embeddings1DConnector;
use crate::feature_extractor::GemmaFeatureExtractor;
use crate::gemma3::{Gemma3Config, Gemma3TextModel};
use crate::tokenizer::GemmaTokenizer;

/// Output from the AV Gemma text encoder.
pub struct AVGemmaEncoderOutput {
    /// Video context embeddings: (B, T, 4096).
    pub video_encoding: Tensor,
    /// Audio context embeddings: (B, T, 2048).
    pub audio_encoding: Tensor,
    /// Attention mask after connector processing: (B, T).
    pub attention_mask: Tensor,
}

/// Debug trace from the text encoder.
pub struct AVGemmaEncoderTrace {
    pub input_ids: Tensor,
    pub tokenizer_attention_mask: Tensor,
    pub embedding_hidden_state: Tensor,
    pub final_hidden_state: Tensor,
    pub video_projected: Tensor,
    pub audio_projected: Tensor,
    pub video_after_registers: Tensor,
    pub audio_after_registers: Tensor,
    pub video_block_outputs: Vec<Tensor>,
    pub audio_block_outputs: Vec<Tensor>,
    pub video_connector_output: Tensor,
    pub audio_connector_output: Tensor,
    pub video_connector_mask: Tensor,
    pub output: AVGemmaEncoderOutput,
}

/// Complete text encoding pipeline for LTX-2.3 audio-visual generation.
///
/// Orchestrates: tokenize → Gemma3 → feature extract → video/audio connectors.
#[derive(Debug)]
pub struct AVGemmaTextEncoder {
    gemma: Gemma3TextModel,
    feature_extractor: GemmaFeatureExtractor,
    video_connector: Embeddings1DConnector,
    audio_connector: Embeddings1DConnector,
    tokenizer: GemmaTokenizer,
}

impl AVGemmaTextEncoder {
    /// Load the full text encoding pipeline.
    ///
    /// Weight key mapping from safetensors:
    /// - `gemma_vb`: rooted at `language_model.model.` prefix
    /// - `feature_vb`: rooted at `text_embedding_projection.` prefix
    /// - `video_conn_vb`: rooted at `model.diffusion_model.video_embeddings_connector.` prefix
    /// - `audio_conn_vb`: rooted at `model.diffusion_model.audio_embeddings_connector.` prefix
    pub fn new(
        tokenizer_path: &str,
        gemma_vb: VarBuilder,
        feature_vb: VarBuilder,
        video_conn_vb: VarBuilder,
        audio_conn_vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_with_transformer_config(
            tokenizer_path,
            &TransformerConfig::default(),
            gemma_vb,
            feature_vb,
            video_conn_vb,
            audio_conn_vb,
        )
    }

    pub fn new_with_transformer_config(
        tokenizer_path: &str,
        transformer_cfg: &TransformerConfig,
        gemma_vb: VarBuilder,
        feature_vb: VarBuilder,
        video_conn_vb: VarBuilder,
        audio_conn_vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Gemma3Config::default();
        let max_seq_len = 1024;

        let tokenizer = GemmaTokenizer::from_file(tokenizer_path, max_seq_len)?;
        let gemma = Gemma3TextModel::new(&cfg, max_seq_len, gemma_vb)?;
        let feature_extractor = if transformer_cfg.caption_proj_before_connector {
            GemmaFeatureExtractor::new_dual(
                cfg.hidden_size,
                cfg.num_hidden_layers + 1, // 49 = embedding + 48 layers
                transformer_cfg.video_inner_dim(),
                transformer_cfg.audio_inner_dim(),
                feature_vb,
            )?
        } else {
            GemmaFeatureExtractor::new_single(
                cfg.hidden_size,
                cfg.num_hidden_layers + 1, // 49 = embedding + 48 layers
                feature_vb,
            )?
        };
        let video_connector = if transformer_cfg.caption_proj_before_connector {
            Embeddings1DConnector::new_video_from_config(transformer_cfg, video_conn_vb)?
        } else {
            Embeddings1DConnector::new_default(video_conn_vb)?
        };
        let audio_connector = if transformer_cfg.caption_proj_before_connector {
            Embeddings1DConnector::new_audio_from_config(transformer_cfg, audio_conn_vb)?
        } else {
            Embeddings1DConnector::new_default(audio_conn_vb)?
        };

        Ok(Self {
            gemma,
            feature_extractor,
            video_connector,
            audio_connector,
            tokenizer,
        })
    }

    /// Encode text into video and audio context embeddings.
    pub fn forward(&self, text: &str) -> Result<AVGemmaEncoderOutput> {
        Ok(self.forward_trace(text)?.output)
    }

    /// Encode text and return intermediate tensors for diagnostics.
    pub fn forward_trace(&self, text: &str) -> Result<AVGemmaEncoderTrace> {
        let device = self.gemma.device();

        // Tokenize
        let (input_ids, attention_mask) = self.tokenizer.encode(text, device)?;
        let tokenizer_attention_mask = attention_mask.clone();

        // Run Gemma to get all hidden states
        let hidden_states = self.gemma.forward(&input_ids, &attention_mask)?;
        let embedding_hidden_state = hidden_states[0].clone();
        let final_hidden_state = hidden_states
            .last()
            .ok_or_else(|| candle_core::Error::Msg("Gemma returned no hidden states".to_string()))?
            .clone();

        // Feature extraction: hidden states → projected features
        let (video_projected, audio_projected) =
            self.feature_extractor
                .forward_av(&hidden_states, &attention_mask, "left")?;

        // Run video connector
        let video_trace = self
            .video_connector
            .forward_trace(&video_projected, &attention_mask)?;
        let video_encoded = video_trace.output.clone();
        let video_mask = video_trace.mask.clone();

        // Restore mask to binary: mask values < 0.000001 → 1 (valid), else → 0
        let binary_mask = video_mask
            .squeeze(1)?
            .squeeze(1)?
            .lt(0.000001_f64)?
            .to_dtype(DType::F32)?;

        // Mask the video output: (B, T, D)
        let mask_3d = binary_mask.unsqueeze(2)?; // (B, T, 1)
        let video_encoding =
            video_encoded.broadcast_mul(&mask_3d.to_dtype(video_projected.dtype())?)?;

        // Run audio connector (independently from projected features)
        let audio_trace = self
            .audio_connector
            .forward_trace(&audio_projected, &attention_mask)?;
        let audio_encoding = audio_trace.output.clone();
        let audio_connector_output = audio_encoding.clone();

        // Final attention mask: (B, T)
        let attention_mask = binary_mask;

        let output = AVGemmaEncoderOutput {
            video_encoding,
            audio_encoding,
            attention_mask,
        };

        Ok(AVGemmaEncoderTrace {
            input_ids,
            tokenizer_attention_mask,
            embedding_hidden_state,
            final_hidden_state,
            video_projected,
            audio_projected,
            video_after_registers: video_trace.after_registers,
            audio_after_registers: audio_trace.after_registers,
            video_block_outputs: video_trace.block_outputs,
            audio_block_outputs: audio_trace.block_outputs,
            video_connector_output: video_encoded,
            audio_connector_output,
            video_connector_mask: video_mask,
            output,
        })
    }
}
