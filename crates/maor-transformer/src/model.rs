use candle_core::{bail, DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use maor_core::config::{LtxModelType, TransformerConfig};
use maor_nn::adaln::AdaLayerNormSingle;
use maor_nn::lora::{self as lora_ops, LoraConfig};
use maor_nn::rope::LTXRopeType;
use maor_nn::text_projection::PixArtAlphaTextProjection;

use crate::block::{BasicAVTransformerBlock, BlockConfig};
use crate::modality::Modality;
use crate::preprocessor::{
    ArgsPreprocessor, MultiModalTransformerArgsPreprocessor, TransformerArgsPreprocessor,
};

/// The LTX-2.3 transformer model.
///
/// Supports video-only, audio-only, or joint audio-video generation.
/// Architecture: patchify -> 48x BasicAVTransformerBlock -> scale-shift -> proj_out
#[derive(Debug)]
pub struct LTXModel {
    video_preprocessor: Option<ArgsPreprocessor>,
    audio_preprocessor: Option<ArgsPreprocessor>,
    blocks: Vec<BasicAVTransformerBlock>,
    // Video output
    scale_shift_table: Option<Tensor>,
    proj_out: Option<Linear>,
    norm_eps: f64,
    // Audio output
    audio_scale_shift_table: Option<Tensor>,
    audio_proj_out: Option<Linear>,
}

impl LTXModel {
    pub fn new(
        config: &TransformerConfig,
        model_type: LtxModelType,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_with_lora(config, model_type, vb, None)
    }

    pub fn new_with_lora(
        config: &TransformerConfig,
        model_type: LtxModelType,
        vb: VarBuilder,
        lora: Option<&LoraConfig>,
    ) -> Result<Self> {
        let rope_type = match config.rope_type.as_str() {
            "split" => LTXRopeType::Split,
            _ => LTXRopeType::Interleaved,
        };

        let video_inner_dim = config.video_inner_dim();
        let audio_inner_dim = config.audio_inner_dim();
        let is_multimodal = model_type.is_video_enabled() && model_type.is_audio_enabled();
        let adaln_coeff = if config.cross_attention_adaln { 9 } else { 6 };

        // === Build video simple preprocessor ===
        let video_simple = if model_type.is_video_enabled() {
            let patchify_name = if vb.pp("proj_in").contains_tensor("weight") {
                "proj_in"
            } else {
                "patchify_proj"
            };
            let adaln_name = if vb.pp("time_embed").pp("linear").contains_tensor("weight") {
                "time_embed"
            } else {
                "adaln_single"
            };
            let caption_projection = if config.caption_proj_before_connector {
                None
            } else {
                let caption_lora = lora.map(|l| l.pp("caption_projection"));
                Some(PixArtAlphaTextProjection::new_with_lora(
                    config.caption_channels,
                    video_inner_dim,
                    None,
                    "gelu",
                    vb.pp("caption_projection"),
                    caption_lora.as_ref(),
                )?)
            };
            let prompt_adaln = if config.cross_attention_adaln {
                let prompt_lora = lora.map(|l| l.pp("prompt_adaln_single"));
                Some(AdaLayerNormSingle::new_with_lora(
                    video_inner_dim,
                    2,
                    vb.pp("prompt_adaln_single"),
                    prompt_lora.as_ref(),
                )?)
            } else {
                None
            };
            let patchify_lora = lora.map(|l| l.pp(patchify_name));
            let adaln_lora = lora.map(|l| l.pp(adaln_name));
            Some(TransformerArgsPreprocessor::new(
                lora_ops::linear(
                    config.in_channels,
                    video_inner_dim,
                    vb.pp(patchify_name),
                    patchify_lora.as_ref(),
                )?,
                AdaLayerNormSingle::new_with_lora(
                    video_inner_dim,
                    adaln_coeff,
                    vb.pp(adaln_name),
                    adaln_lora.as_ref(),
                )?,
                caption_projection,
                prompt_adaln,
                video_inner_dim,
                config.positional_embedding_max_pos.clone(),
                config.num_attention_heads,
                config.use_middle_indices_grid,
                config.timestep_scale_multiplier,
                config.positional_embedding_theta,
                rope_type,
            ))
        } else {
            None
        };

        let scale_shift_table = if model_type.is_video_enabled() {
            Some(vb.get(&[2, video_inner_dim], "scale_shift_table")?)
        } else {
            None
        };
        let proj_out = if model_type.is_video_enabled() {
            let proj_lora = lora.map(|l| l.pp("proj_out"));
            Some(lora_ops::linear(
                video_inner_dim,
                config.out_channels,
                vb.pp("proj_out"),
                proj_lora.as_ref(),
            )?)
        } else {
            None
        };

        // === Build audio simple preprocessor ===
        let audio_simple = if model_type.is_audio_enabled() {
            let patchify_name = if vb.pp("audio_proj_in").contains_tensor("weight") {
                "audio_proj_in"
            } else {
                "audio_patchify_proj"
            };
            let adaln_name = if vb
                .pp("audio_time_embed")
                .pp("linear")
                .contains_tensor("weight")
            {
                "audio_time_embed"
            } else {
                "audio_adaln_single"
            };
            let caption_projection = if config.caption_proj_before_connector {
                None
            } else {
                let caption_lora = lora.map(|l| l.pp("audio_caption_projection"));
                Some(PixArtAlphaTextProjection::new_with_lora(
                    config.caption_channels,
                    audio_inner_dim,
                    None,
                    "gelu",
                    vb.pp("audio_caption_projection"),
                    caption_lora.as_ref(),
                )?)
            };
            let prompt_adaln = if config.cross_attention_adaln {
                let prompt_lora = lora.map(|l| l.pp("audio_prompt_adaln_single"));
                Some(AdaLayerNormSingle::new_with_lora(
                    audio_inner_dim,
                    2,
                    vb.pp("audio_prompt_adaln_single"),
                    prompt_lora.as_ref(),
                )?)
            } else {
                None
            };
            let patchify_lora = lora.map(|l| l.pp(patchify_name));
            let adaln_lora = lora.map(|l| l.pp(adaln_name));
            Some(TransformerArgsPreprocessor::new(
                lora_ops::linear(
                    config.audio_in_channels,
                    audio_inner_dim,
                    vb.pp(patchify_name),
                    patchify_lora.as_ref(),
                )?,
                AdaLayerNormSingle::new_with_lora(
                    audio_inner_dim,
                    adaln_coeff,
                    vb.pp(adaln_name),
                    adaln_lora.as_ref(),
                )?,
                caption_projection,
                prompt_adaln,
                audio_inner_dim,
                config.audio_positional_embedding_max_pos.clone(),
                config.audio_num_attention_heads,
                config.use_middle_indices_grid,
                config.timestep_scale_multiplier,
                config.positional_embedding_theta,
                rope_type,
            ))
        } else {
            None
        };

        let audio_scale_shift_table = if model_type.is_audio_enabled() {
            Some(vb.get(&[2, audio_inner_dim], "audio_scale_shift_table")?)
        } else {
            None
        };
        let audio_proj_out = if model_type.is_audio_enabled() {
            let proj_lora = lora.map(|l| l.pp("audio_proj_out"));
            Some(lora_ops::linear(
                audio_inner_dim,
                config.audio_out_channels,
                vb.pp("audio_proj_out"),
                proj_lora.as_ref(),
            )?)
        } else {
            None
        };

        // === Wrap preprocessors (Simple or MultiModal) ===
        let (video_preprocessor, audio_preprocessor) = if is_multimodal {
            let cross_pe_max_pos = *config
                .positional_embedding_max_pos
                .first()
                .unwrap_or(&20)
                .max(
                    config
                        .audio_positional_embedding_max_pos
                        .first()
                        .unwrap_or(&20),
                );

            let vs = match video_simple {
                Some(v) => v,
                None => bail!("video preprocessor required for multimodal mode"),
            };
            let video_cross_ss_name = if vb
                .pp("av_cross_attn_video_scale_shift")
                .pp("linear")
                .contains_tensor("weight")
            {
                "av_cross_attn_video_scale_shift"
            } else {
                "av_ca_video_scale_shift_adaln_single"
            };
            let video_cross_gate_name = if vb
                .pp("av_cross_attn_video_a2v_gate")
                .pp("linear")
                .contains_tensor("weight")
            {
                "av_cross_attn_video_a2v_gate"
            } else {
                "av_ca_a2v_gate_adaln_single"
            };
            let vp = ArgsPreprocessor::MultiModal(MultiModalTransformerArgsPreprocessor::new(
                vs,
                AdaLayerNormSingle::new_with_lora(
                    video_inner_dim,
                    4,
                    vb.pp(video_cross_ss_name),
                    lora.map(|l| l.pp(video_cross_ss_name)).as_ref(),
                )?,
                AdaLayerNormSingle::new_with_lora(
                    video_inner_dim,
                    1,
                    vb.pp(video_cross_gate_name),
                    lora.map(|l| l.pp(video_cross_gate_name)).as_ref(),
                )?,
                cross_pe_max_pos,
                config.audio_cross_attention_dim,
                config.av_ca_timestep_scale_multiplier,
            ));

            let as_ = match audio_simple {
                Some(a) => a,
                None => bail!("audio preprocessor required for multimodal mode"),
            };
            let audio_cross_ss_name = if vb
                .pp("av_cross_attn_audio_scale_shift")
                .pp("linear")
                .contains_tensor("weight")
            {
                "av_cross_attn_audio_scale_shift"
            } else {
                "av_ca_audio_scale_shift_adaln_single"
            };
            let audio_cross_gate_name = if vb
                .pp("av_cross_attn_audio_v2a_gate")
                .pp("linear")
                .contains_tensor("weight")
            {
                "av_cross_attn_audio_v2a_gate"
            } else {
                "av_ca_v2a_gate_adaln_single"
            };
            let ap = ArgsPreprocessor::MultiModal(MultiModalTransformerArgsPreprocessor::new(
                as_,
                AdaLayerNormSingle::new_with_lora(
                    audio_inner_dim,
                    4,
                    vb.pp(audio_cross_ss_name),
                    lora.map(|l| l.pp(audio_cross_ss_name)).as_ref(),
                )?,
                AdaLayerNormSingle::new_with_lora(
                    audio_inner_dim,
                    1,
                    vb.pp(audio_cross_gate_name),
                    lora.map(|l| l.pp(audio_cross_gate_name)).as_ref(),
                )?,
                cross_pe_max_pos,
                config.audio_cross_attention_dim,
                config.av_ca_timestep_scale_multiplier,
            ));

            (Some(vp), Some(ap))
        } else {
            (
                video_simple.map(ArgsPreprocessor::Simple),
                audio_simple.map(ArgsPreprocessor::Simple),
            )
        };

        // === Transformer blocks ===
        let video_block_config = if model_type.is_video_enabled() {
            Some(BlockConfig {
                dim: video_inner_dim,
                heads: config.num_attention_heads,
                d_head: config.attention_head_dim,
                context_dim: config.cross_attention_dim,
                apply_gated_attention: config.apply_gated_attention,
                cross_attention_adaln: config.cross_attention_adaln,
            })
        } else {
            None
        };

        let audio_block_config = if model_type.is_audio_enabled() {
            Some(BlockConfig {
                dim: audio_inner_dim,
                heads: config.audio_num_attention_heads,
                d_head: config.audio_attention_head_dim,
                context_dim: config.audio_cross_attention_dim,
                apply_gated_attention: config.apply_gated_attention,
                cross_attention_adaln: config.cross_attention_adaln,
            })
        } else {
            None
        };

        let mut blocks = Vec::with_capacity(config.num_layers);
        let blocks_vb = vb.pp("transformer_blocks");
        for idx in 0..config.num_layers {
            let block_lora = lora.map(|l| l.pp("transformer_blocks").pp(idx));
            blocks.push(BasicAVTransformerBlock::new_with_lora(
                idx,
                video_block_config.as_ref(),
                audio_block_config.as_ref(),
                rope_type,
                config.norm_eps,
                blocks_vb.pp(idx),
                block_lora.as_ref(),
            )?);
        }

        Ok(Self {
            video_preprocessor,
            audio_preprocessor,
            blocks,
            scale_shift_table,
            proj_out,
            norm_eps: config.norm_eps,
            audio_scale_shift_table,
            audio_proj_out,
        })
    }

    /// Process output: scale-shift modulation -> LayerNorm -> proj_out.
    fn process_output(
        scale_shift_table: &Tensor,
        proj_out: &Linear,
        x: &Tensor,
        embedded_timestep: &Tensor,
        norm_eps: f64,
    ) -> Result<Tensor> {
        // AdaLN structural fix: force the entire output modulation in fp32
        // (matches reference _process_output), cast back only before proj_out.
        let out_dtype = x.dtype();
        // scale_shift_table: (2, dim) -> (1, 1, 2, dim)
        // embedded_timestep: (B, T, dim) -> (B, T, 1, dim)
        // combined: (B, T, 2, dim)
        let table = scale_shift_table
            .unsqueeze(0)?
            .unsqueeze(0)?
            .to_dtype(DType::F32)?;
        let emb = embedded_timestep.to_dtype(DType::F32)?.unsqueeze(2)?;
        let combined = table.broadcast_add(&emb)?;

        let shift = combined.narrow(2, 0, 1)?.squeeze(2)?;
        let scale = combined.narrow(2, 1, 1)?.squeeze(2)?;

        // LayerNorm (elementwise_affine=False) in fp32, kept in fp32.
        let xf = x.to_dtype(DType::F32)?;
        let mean = xf.mean_keepdim(candle_core::D::Minus1)?;
        let diff = xf.broadcast_sub(&mean)?;
        let var = diff.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let std = (var + norm_eps)?.sqrt()?;
        let normed = diff.broadcast_div(&std)?;
        let x = (normed.broadcast_mul(&(scale + 1.0)?)? + shift)?;
        proj_out.forward(&x.to_dtype(out_dtype)?)
    }

    /// Forward pass.
    ///
    /// Returns (video_velocity, audio_velocity) where each is None if that
    /// modality is not enabled.
    pub fn forward(
        &self,
        video: Option<&Modality>,
        audio: Option<&Modality>,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward_perturbed(video, audio, &[], &[])
    }

    /// Forward pass with STG (Spatio-Temporal Guidance) perturbations: at the
    /// given block indices, that stream's self-attention sublayer is skipped
    /// (becomes identity), matching the reference SKIP_*_SELF_ATTN.
    pub fn forward_perturbed(
        &self,
        video: Option<&Modality>,
        audio: Option<&Modality>,
        skip_video_self_attn_blocks: &[usize],
        skip_audio_self_attn_blocks: &[usize],
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        // Preprocess
        let video_args = match (video, &self.video_preprocessor) {
            (Some(v), Some(p)) => Some(p.prepare_with_cross_modality(v, audio)?),
            _ => None,
        };
        let audio_args = match (audio, &self.audio_preprocessor) {
            (Some(a), Some(p)) => Some(p.prepare_with_cross_modality(a, video)?),
            _ => None,
        };

        // Run through transformer blocks
        let mut va = video_args;
        let mut aa = audio_args;
        for block in &self.blocks {
            let sv = skip_video_self_attn_blocks.contains(&block.idx);
            let sa = skip_audio_self_attn_blocks.contains(&block.idx);
            let (v_out, a_out) = block.forward(va.as_ref(), aa.as_ref(), sv, sa)?;
            va = v_out;
            aa = a_out;
        }

        // Process outputs
        let vx = match (
            va.as_ref(),
            self.scale_shift_table.as_ref(),
            self.proj_out.as_ref(),
        ) {
            (Some(v), Some(table), Some(proj)) => Some(Self::process_output(
                table,
                proj,
                &v.x,
                &v.embedded_timestep,
                self.norm_eps,
            )?),
            _ => None,
        };

        let ax = match (
            aa.as_ref(),
            self.audio_scale_shift_table.as_ref(),
            self.audio_proj_out.as_ref(),
        ) {
            (Some(a), Some(table), Some(proj)) => Some(Self::process_output(
                table,
                proj,
                &a.x,
                &a.embedded_timestep,
                self.norm_eps,
            )?),
            _ => None,
        };

        Ok((vx, ax))
    }
}
