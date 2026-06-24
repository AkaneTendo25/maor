use serde::{Deserialize, Deserializer, Serialize};

/// Top-level model config (wraps sub-configs keyed by component name).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub transformer: TransformerConfig,
    #[serde(default)]
    pub vae: VaeConfig,
}

/// Video VAE configuration.
///
/// Matches the "vae" key in safetensors metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaeConfig {
    #[serde(default = "default_3")]
    pub dims: usize,
    #[serde(default = "default_128")]
    pub latent_channels: usize,
    #[serde(default = "default_3")]
    pub out_channels: usize,
    #[serde(default)]
    pub decoder_blocks: Vec<(String, serde_json::Value)>,
    #[serde(
        default = "default_reflect",
        deserialize_with = "deserialize_nullable_string_reflect"
    )]
    pub decoder_spatial_padding_mode: String,
    #[serde(default = "default_4")]
    pub patch_size: usize,
    #[serde(default = "default_pixel_norm")]
    pub norm_layer: String,
    #[serde(default)]
    pub causal_decoder: bool,
    #[serde(default = "default_true")]
    pub timestep_conditioning: bool,
}

impl Default for VaeConfig {
    fn default() -> Self {
        Self {
            dims: 3,
            latent_channels: 128,
            out_channels: 3,
            decoder_blocks: Vec::new(),
            decoder_spatial_padding_mode: "reflect".to_string(),
            patch_size: 4,
            norm_layer: "pixel_norm".to_string(),
            causal_decoder: false,
            timestep_conditioning: true,
        }
    }
}

/// Transformer (DiT) configuration.
///
/// Matches the JSON config loaded from safetensors metadata / config files.
/// Defaults are set for the supported LTX-2.3 inference layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    // --- Video stream ---
    #[serde(default = "default_32")]
    pub num_attention_heads: usize,
    #[serde(default = "default_128")]
    pub attention_head_dim: usize,
    #[serde(default = "default_128")]
    pub in_channels: usize,
    #[serde(default = "default_128")]
    pub out_channels: usize,
    #[serde(default = "default_48")]
    pub num_layers: usize,
    #[serde(default = "default_4096")]
    pub cross_attention_dim: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default = "default_attention_type")]
    pub attention_type: String,
    #[serde(default = "default_3840")]
    pub caption_channels: usize,
    #[serde(default = "default_10000f")]
    pub positional_embedding_theta: f64,
    #[serde(default = "default_pos_max")]
    pub positional_embedding_max_pos: Vec<usize>,
    #[serde(default = "default_1000")]
    pub timestep_scale_multiplier: usize,
    #[serde(default = "default_true")]
    pub use_middle_indices_grid: bool,
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
    #[serde(default = "default_float64")]
    pub frequencies_precision: Option<String>,
    #[serde(default = "default_false")]
    pub apply_gated_attention: bool,

    // --- Text embeddings connector ---
    #[serde(default = "default_32")]
    pub connector_num_attention_heads: usize,
    #[serde(default = "default_128")]
    pub connector_attention_head_dim: usize,
    #[serde(default = "default_8")]
    pub connector_num_layers: usize,
    #[serde(default = "default_connector_pos_max")]
    pub connector_positional_embedding_max_pos: Vec<usize>,
    #[serde(default = "default_128")]
    pub connector_num_learnable_registers: usize,
    #[serde(default = "default_true")]
    pub connector_apply_gated_attention: bool,

    // --- Audio stream ---
    #[serde(default = "default_32")]
    pub audio_num_attention_heads: usize,
    #[serde(default = "default_64")]
    pub audio_attention_head_dim: usize,
    #[serde(default = "default_128")]
    pub audio_in_channels: usize,
    #[serde(default = "default_128")]
    pub audio_out_channels: usize,
    #[serde(default = "default_2048")]
    pub audio_cross_attention_dim: usize,
    #[serde(default = "default_audio_pos_max")]
    pub audio_positional_embedding_max_pos: Vec<usize>,
    #[serde(default = "default_1f")]
    pub av_ca_timestep_scale_multiplier: f64,
    #[serde(default = "default_32")]
    pub audio_connector_num_attention_heads: usize,
    #[serde(default = "default_64")]
    pub audio_connector_attention_head_dim: usize,
    #[serde(default = "default_8")]
    pub audio_connector_num_layers: usize,

    // --- LTX-2.3 transformer switches ---
    #[serde(default = "default_true")]
    pub cross_attention_adaln: bool,
    #[serde(default = "default_true")]
    pub caption_proj_before_connector: bool,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            num_attention_heads: 32,
            attention_head_dim: 128,
            in_channels: 128,
            out_channels: 128,
            num_layers: 48,
            cross_attention_dim: 4096,
            norm_eps: 1e-6,
            attention_type: "default".to_string(),
            caption_channels: 3840,
            positional_embedding_theta: 10000.0,
            positional_embedding_max_pos: vec![20, 2048, 2048],
            timestep_scale_multiplier: 1000,
            use_middle_indices_grid: true,
            rope_type: "split".to_string(),
            frequencies_precision: Some("float64".to_string()),
            apply_gated_attention: false,
            connector_num_attention_heads: 32,
            connector_attention_head_dim: 128,
            connector_num_layers: 8,
            connector_positional_embedding_max_pos: vec![4096],
            connector_num_learnable_registers: 128,
            connector_apply_gated_attention: true,
            audio_num_attention_heads: 32,
            audio_attention_head_dim: 64,
            audio_in_channels: 128,
            audio_out_channels: 128,
            audio_cross_attention_dim: 2048,
            audio_positional_embedding_max_pos: vec![20],
            av_ca_timestep_scale_multiplier: 1.0,
            audio_connector_num_attention_heads: 32,
            audio_connector_attention_head_dim: 64,
            audio_connector_num_layers: 8,
            cross_attention_adaln: true,
            caption_proj_before_connector: true,
        }
    }
}

impl TransformerConfig {
    /// Video inner dimension = num_attention_heads * attention_head_dim.
    pub fn video_inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Audio inner dimension = audio_num_attention_heads * audio_attention_head_dim.
    pub fn audio_inner_dim(&self) -> usize {
        self.audio_num_attention_heads * self.audio_attention_head_dim
    }

    /// Whether to use float64 precision for RoPE frequency computation.
    pub fn double_precision_rope(&self) -> bool {
        self.frequencies_precision.as_deref() == Some("float64")
    }
}

/// Model type for the transformer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LtxModelType {
    AudioVideo,
    VideoOnly,
    AudioOnly,
}

impl LtxModelType {
    pub fn is_video_enabled(&self) -> bool {
        matches!(self, Self::AudioVideo | Self::VideoOnly)
    }

    pub fn is_audio_enabled(&self) -> bool {
        matches!(self, Self::AudioVideo | Self::AudioOnly)
    }
}

/// RoPE type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RopeType {
    Interleaved,
    Split,
}

impl RopeType {
    pub fn parse(s: &str) -> Self {
        match s {
            "split" => Self::Split,
            _ => Self::Interleaved,
        }
    }
}

/// Scheduler configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    #[serde(default = "default_2_05")]
    pub max_shift: f64,
    #[serde(default = "default_0_95")]
    pub base_shift: f64,
    #[serde(default = "default_true")]
    pub stretch: bool,
    #[serde(default = "default_0_1")]
    pub terminal: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_shift: 2.05,
            base_shift: 0.95,
            stretch: true,
            terminal: 0.1,
        }
    }
}

/// CFG guider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuiderConfig {
    #[serde(default = "default_1f")]
    pub cfg_scale: f64,
    #[serde(default)]
    pub stg_scale: f64,
    #[serde(default)]
    pub stg_blocks: Option<Vec<usize>>,
    #[serde(default)]
    pub rescale_scale: f64,
    #[serde(default = "default_1f")]
    pub modality_scale: f64,
    #[serde(default)]
    pub skip_step: usize,
}

impl Default for GuiderConfig {
    fn default() -> Self {
        Self {
            cfg_scale: 1.0,
            stg_scale: 0.0,
            stg_blocks: None,
            rescale_scale: 0.0,
            modality_scale: 1.0,
            skip_step: 0,
        }
    }
}

// ── Serde default helpers ────────────────────────────────────────────
// Required by serde(default = "...") for deserialization from partial JSON.

macro_rules! serde_default {
    ($name:ident, $t:ty, $val:expr) => {
        fn $name() -> $t {
            $val
        }
    };
}

// Integer defaults
serde_default!(default_3, usize, 3); // VAE dims / out_channels
serde_default!(default_4, usize, 4); // VAE patch_size
serde_default!(default_8, usize, 8); // connector depth
serde_default!(default_32, usize, 32); // num_attention_heads (video & audio)
serde_default!(default_64, usize, 64); // audio_attention_head_dim
serde_default!(default_128, usize, 128); // latent_channels / attention_head_dim
serde_default!(default_48, usize, 48); // num_layers (DiT depth)
serde_default!(default_1000, usize, 1000); // timestep_scale_multiplier
serde_default!(default_3840, usize, 3840); // caption_channels (= Gemma3 hidden_size)
serde_default!(default_4096, usize, 4096); // cross_attention_dim (= video inner dim)
serde_default!(default_2048, usize, 2048); // audio_cross_attention_dim

// Float defaults
serde_default!(default_norm_eps, f64, 1e-6);
serde_default!(default_10000f, f64, 10000.0); // RoPE base frequency (theta)
serde_default!(default_2_05, f64, 2.05); // scheduler max_shift
serde_default!(default_0_95, f64, 0.95); // scheduler base_shift
serde_default!(default_0_1, f64, 0.1); // scheduler terminal sigma
serde_default!(default_1f, f64, 1.0); // CFG/modality scale (1.0 = off)

// Bool defaults
serde_default!(default_false, bool, false);
serde_default!(default_true, bool, true);

// String/vec defaults
fn default_attention_type() -> String {
    "default".to_string()
}
fn default_rope_type() -> String {
    "split".to_string()
}
fn default_float64() -> Option<String> {
    Some("float64".to_string())
}
fn default_pos_max() -> Vec<usize> {
    vec![20, 2048, 2048]
} // [frames, height, width] max positions
fn default_audio_pos_max() -> Vec<usize> {
    vec![20]
} // [frames] max positions
fn default_connector_pos_max() -> Vec<usize> {
    vec![4096]
}
fn default_reflect() -> String {
    "reflect".to_string()
}
fn default_pixel_norm() -> String {
    "pixel_norm".to_string()
}

fn deserialize_nullable_string_reflect<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<String>::deserialize(deserializer)?.unwrap_or_else(default_reflect))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_transformer_config() {
        let cfg = TransformerConfig::default();
        assert_eq!(cfg.video_inner_dim(), 4096); // 32 * 128
        assert_eq!(cfg.audio_inner_dim(), 2048); // 32 * 64
        assert_eq!(cfg.num_layers, 48);
        assert!(cfg.double_precision_rope());
    }

    #[test]
    fn test_deserialize_partial_config() {
        let json = r#"{"num_layers": 24, "attention_head_dim": 64}"#;
        let cfg: TransformerConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.attention_head_dim, 64);
        // defaults filled in
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.in_channels, 128);
    }

    #[test]
    fn test_model_type() {
        assert!(LtxModelType::AudioVideo.is_video_enabled());
        assert!(LtxModelType::AudioVideo.is_audio_enabled());
        assert!(LtxModelType::VideoOnly.is_video_enabled());
        assert!(!LtxModelType::VideoOnly.is_audio_enabled());
    }
}
