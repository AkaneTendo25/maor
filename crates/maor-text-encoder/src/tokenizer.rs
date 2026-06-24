use candle_core::{Device, Result, Tensor};

/// Gemma tokenizer wrapper with left-padding.
pub struct GemmaTokenizer {
    tokenizer: tokenizers::Tokenizer,
    max_length: usize,
}

impl GemmaTokenizer {
    /// Load from a tokenizer.json file.
    pub fn from_file(path: &str, max_length: usize) -> Result<Self> {
        let mut tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| candle_core::Error::Msg(format!("tokenizer load: {e}")))?;

        // Determine pad token and ID
        let pad_id = tokenizer
            .token_to_id("<pad>")
            .or_else(|| tokenizer.token_to_id("</s>"))
            .unwrap_or(0);
        let pad_token = tokenizer
            .id_to_token(pad_id)
            .unwrap_or_else(|| "<pad>".to_string());

        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(max_length),
            direction: tokenizers::PaddingDirection::Left,
            pad_id,
            pad_token,
            ..Default::default()
        }));

        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length,
                ..Default::default()
            }))
            .map_err(|e| candle_core::Error::Msg(format!("truncation config: {e}")))?;

        Ok(Self {
            tokenizer,
            max_length,
        })
    }

    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// Tokenize text, returning (input_ids, attention_mask) as tensors.
    pub fn encode(&self, text: &str, device: &Device) -> Result<(Tensor, Tensor)> {
        let text = text.trim();
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| candle_core::Error::Msg(format!("encode: {e}")))?;

        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let mask: Vec<u32> = encoding.get_attention_mask().to_vec();

        let input_ids = Tensor::from_vec(ids, (1, self.max_length), device)?;
        let attention_mask = Tensor::from_vec(mask, (1, self.max_length), device)?;
        Ok((input_ids, attention_mask))
    }
}

impl std::fmt::Debug for GemmaTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GemmaTokenizer")
            .field("max_length", &self.max_length)
            .finish()
    }
}
