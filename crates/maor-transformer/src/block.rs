use candle_core::{bail, DType, Module, Result, Tensor};
use candle_nn::VarBuilder;

use maor_core::ops::{missing_err, rms_norm};
use maor_nn::attention::Attention;
use maor_nn::feed_forward::FeedForward;
use maor_nn::lora::LoraConfig;
use maor_nn::rope::LTXRopeType;

use crate::modality::TransformerArgs;

/// Per-modality config for a transformer block.
pub struct BlockConfig {
    pub dim: usize,
    pub heads: usize,
    pub d_head: usize,
    pub context_dim: usize,
    pub apply_gated_attention: bool,
    pub cross_attention_adaln: bool,
}

/// Audio-video transformer block.
///
/// Contains self-attention, cross-attention to text, bidirectional AV cross-attention,
/// and feed-forward networks for both video and audio modalities.
#[derive(Debug)]
pub struct BasicAVTransformerBlock {
    pub idx: usize,
    // Video layers
    attn1: Option<Attention>,
    attn2: Option<Attention>,
    ff: Option<FeedForward>,
    scale_shift_table: Option<Tensor>,
    prompt_scale_shift_table: Option<Tensor>,
    // Audio layers
    audio_attn1: Option<Attention>,
    audio_attn2: Option<Attention>,
    audio_ff: Option<FeedForward>,
    audio_scale_shift_table: Option<Tensor>,
    audio_prompt_scale_shift_table: Option<Tensor>,
    // AV cross-modal layers
    audio_to_video_attn: Option<Attention>,
    video_to_audio_attn: Option<Attention>,
    scale_shift_table_a2v_ca_audio: Option<Tensor>,
    scale_shift_table_a2v_ca_video: Option<Tensor>,
    // Config
    norm_eps: f64,
}

impl BasicAVTransformerBlock {
    pub fn new(
        idx: usize,
        video: Option<&BlockConfig>,
        audio: Option<&BlockConfig>,
        rope_type: LTXRopeType,
        norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_with_lora(idx, video, audio, rope_type, norm_eps, vb, None)
    }

    pub fn new_with_lora(
        idx: usize,
        video: Option<&BlockConfig>,
        audio: Option<&BlockConfig>,
        rope_type: LTXRopeType,
        norm_eps: f64,
        vb: VarBuilder,
        lora: Option<&LoraConfig>,
    ) -> Result<Self> {
        let mut block = Self {
            idx,
            attn1: None,
            attn2: None,
            ff: None,
            scale_shift_table: None,
            prompt_scale_shift_table: None,
            audio_attn1: None,
            audio_attn2: None,
            audio_ff: None,
            audio_scale_shift_table: None,
            audio_prompt_scale_shift_table: None,
            audio_to_video_attn: None,
            video_to_audio_attn: None,
            scale_shift_table_a2v_ca_audio: None,
            scale_shift_table_a2v_ca_video: None,
            norm_eps,
        };

        if let Some(v) = video {
            let attn1_lora = lora.map(|l| l.pp("attn1"));
            let attn2_lora = lora.map(|l| l.pp("attn2"));
            let ff_lora = lora.map(|l| l.pp("ff"));
            block.attn1 = Some(Attention::new_with_lora(
                v.dim,
                None,
                v.heads,
                v.d_head,
                norm_eps,
                rope_type,
                v.apply_gated_attention,
                vb.pp("attn1"),
                attn1_lora.as_ref(),
            )?);
            block.attn2 = Some(Attention::new_with_lora(
                v.dim,
                Some(v.context_dim),
                v.heads,
                v.d_head,
                norm_eps,
                rope_type,
                v.apply_gated_attention,
                vb.pp("attn2"),
                attn2_lora.as_ref(),
            )?);
            block.ff = Some(FeedForward::new_with_lora(
                v.dim,
                v.dim,
                4,
                vb.pp("ff"),
                ff_lora.as_ref(),
            )?);
            block.scale_shift_table = Some(vb.get(
                &[if v.cross_attention_adaln { 9 } else { 6 }, v.dim],
                "scale_shift_table",
            )?);
            if v.cross_attention_adaln {
                block.prompt_scale_shift_table =
                    Some(vb.get(&[2, v.dim], "prompt_scale_shift_table")?);
            }
        }

        if let Some(a) = audio {
            let attn1_lora = lora.map(|l| l.pp("audio_attn1"));
            let attn2_lora = lora.map(|l| l.pp("audio_attn2"));
            let ff_lora = lora.map(|l| l.pp("audio_ff"));
            block.audio_attn1 = Some(Attention::new_with_lora(
                a.dim,
                None,
                a.heads,
                a.d_head,
                norm_eps,
                rope_type,
                a.apply_gated_attention,
                vb.pp("audio_attn1"),
                attn1_lora.as_ref(),
            )?);
            block.audio_attn2 = Some(Attention::new_with_lora(
                a.dim,
                Some(a.context_dim),
                a.heads,
                a.d_head,
                norm_eps,
                rope_type,
                a.apply_gated_attention,
                vb.pp("audio_attn2"),
                attn2_lora.as_ref(),
            )?);
            block.audio_ff = Some(FeedForward::new_with_lora(
                a.dim,
                a.dim,
                4,
                vb.pp("audio_ff"),
                ff_lora.as_ref(),
            )?);
            block.audio_scale_shift_table = Some(vb.get(
                &[if a.cross_attention_adaln { 9 } else { 6 }, a.dim],
                "audio_scale_shift_table",
            )?);
            if a.cross_attention_adaln {
                block.audio_prompt_scale_shift_table =
                    Some(vb.get(&[2, a.dim], "audio_prompt_scale_shift_table")?);
            }
        }

        if let (Some(v), Some(a)) = (video, audio) {
            // Q: Video, K/V: Audio
            let a2v_lora = lora.map(|l| l.pp("audio_to_video_attn"));
            let v2a_lora = lora.map(|l| l.pp("video_to_audio_attn"));
            block.audio_to_video_attn = Some(Attention::new_with_lora(
                v.dim,
                Some(a.dim),
                a.heads,
                a.d_head,
                norm_eps,
                rope_type,
                v.apply_gated_attention,
                vb.pp("audio_to_video_attn"),
                a2v_lora.as_ref(),
            )?);
            // Q: Audio, K/V: Video
            block.video_to_audio_attn = Some(Attention::new_with_lora(
                a.dim,
                Some(v.dim),
                a.heads,
                a.d_head,
                norm_eps,
                rope_type,
                a.apply_gated_attention,
                vb.pp("video_to_audio_attn"),
                v2a_lora.as_ref(),
            )?);
            block.scale_shift_table_a2v_ca_audio = Some(
                if vb.contains_tensor("audio_a2v_cross_attn_scale_shift_table") {
                    vb.get(&[5, a.dim], "audio_a2v_cross_attn_scale_shift_table")?
                } else {
                    vb.get(&[5, a.dim], "scale_shift_table_a2v_ca_audio")?
                },
            );
            block.scale_shift_table_a2v_ca_video = Some(
                if vb.contains_tensor("video_a2v_cross_attn_scale_shift_table") {
                    vb.get(&[5, v.dim], "video_a2v_cross_attn_scale_shift_table")?
                } else {
                    vb.get(&[5, v.dim], "scale_shift_table_a2v_ca_video")?
                },
            );
        }

        Ok(block)
    }

    pub fn forward(
        &self,
        video: Option<&TransformerArgs>,
        audio: Option<&TransformerArgs>,
        skip_video_self_attn: bool,
        skip_audio_self_attn: bool,
    ) -> Result<(Option<TransformerArgs>, Option<TransformerArgs>)> {
        let batch_size = match video.or(audio) {
            Some(a) => a.x.dims()[0],
            None => bail!("at least one modality required"),
        };

        let mut vx = video.map(|v| v.x.clone());
        let mut ax = audio.map(|a| a.x.clone());

        let run_vx = video.is_some_and(|v| v.enabled && v.x.elem_count() > 0);
        let run_ax = audio.is_some_and(|a| a.enabled && a.x.elem_count() > 0);
        let run_a2v = run_vx && audio.is_some_and(|a| a.x.elem_count() > 0);
        let run_v2a = run_ax && video.is_some_and(|v| v.x.elem_count() > 0);

        // === Video self-attention + text cross-attention ===
        if run_vx {
            let v = video.ok_or(missing_err("video modality"))?;
            let vx_ref = vx.as_ref().ok_or(missing_err("video hidden state"))?;
            vx = Some(apply_self_attn_and_cross_attn(
                vx_ref,
                self.attn1.as_ref().ok_or(missing_err("video attn1"))?,
                self.attn2.as_ref().ok_or(missing_err("video attn2"))?,
                self.scale_shift_table
                    .as_ref()
                    .ok_or(missing_err("video scale_shift_table"))?,
                self.prompt_scale_shift_table.as_ref(),
                v,
                batch_size,
                self.norm_eps,
                skip_video_self_attn,
            )?);
        }

        // === Audio self-attention + text cross-attention ===
        if run_ax {
            let a = audio.ok_or(missing_err("audio modality"))?;
            let ax_ref = ax.as_ref().ok_or(missing_err("audio hidden state"))?;
            ax = Some(apply_self_attn_and_cross_attn(
                ax_ref,
                self.audio_attn1
                    .as_ref()
                    .ok_or(missing_err("audio attn1"))?,
                self.audio_attn2
                    .as_ref()
                    .ok_or(missing_err("audio attn2"))?,
                self.audio_scale_shift_table
                    .as_ref()
                    .ok_or(missing_err("audio scale_shift_table"))?,
                self.audio_prompt_scale_shift_table.as_ref(),
                a,
                batch_size,
                self.norm_eps,
                skip_audio_self_attn,
            )?);
        }

        // === Audio-Video bidirectional cross-attention ===
        // Audio-to-Video: Q=video tokens, K/V=audio tokens (audio informs video)
        // Video-to-Audio: Q=audio tokens, K/V=video tokens (video informs audio)
        if run_a2v || run_v2a {
            let video = video.ok_or(missing_err("video modality for AV cross-attn"))?;
            let audio = audio.ok_or(missing_err("audio modality for AV cross-attn"))?;
            let vx_ref = vx.as_ref().ok_or(missing_err("video hidden state"))?;
            let ax_ref = ax.as_ref().ok_or(missing_err("audio hidden state"))?;

            let vx_norm3 = rms_norm(vx_ref, None, self.norm_eps)?;
            let ax_norm3 = rms_norm(ax_ref, None, self.norm_eps)?;

            // Ada values from 5-row tables: rows 0-3 = scale/shift, row 4 = gate
            let audio_table = self
                .scale_shift_table_a2v_ca_audio
                .as_ref()
                .ok_or(missing_err("audio AV cross-attn table"))?;
            let audio_cross_ss = audio
                .cross_scale_shift_timestep
                .as_ref()
                .ok_or(missing_err("audio cross_scale_shift_timestep"))?;
            let audio_cross_gate = audio
                .cross_gate_timestep
                .as_ref()
                .ok_or(missing_err("audio cross_gate_timestep"))?;
            let audio_av_ada =
                get_av_ca_ada_values(audio_table, batch_size, audio_cross_ss, audio_cross_gate)?;

            let video_table = self
                .scale_shift_table_a2v_ca_video
                .as_ref()
                .ok_or(missing_err("video AV cross-attn table"))?;
            let video_cross_ss = video
                .cross_scale_shift_timestep
                .as_ref()
                .ok_or(missing_err("video cross_scale_shift_timestep"))?;
            let video_cross_gate = video
                .cross_gate_timestep
                .as_ref()
                .ok_or(missing_err("video cross_gate_timestep"))?;
            let video_av_ada =
                get_av_ca_ada_values(video_table, batch_size, video_cross_ss, video_cross_gate)?;

            // Audio-to-Video: Q=video, K/V=audio
            if run_a2v {
                let vx_scaled =
                    modulate_normed(&vx_norm3, &video_av_ada.shift_a2v, &video_av_ada.scale_a2v)?;
                let ax_scaled =
                    modulate_normed(&ax_norm3, &audio_av_ada.shift_a2v, &audio_av_ada.scale_a2v)?;

                let v_cross_pe = video
                    .cross_positional_embeddings
                    .as_ref()
                    .map(|(c, s)| (c, s));
                let a_cross_pe = audio
                    .cross_positional_embeddings
                    .as_ref()
                    .map(|(c, s)| (c, s));

                let a2v_attn = self
                    .audio_to_video_attn
                    .as_ref()
                    .ok_or(missing_err("audio_to_video_attn"))?;
                let a2v_out =
                    a2v_attn.forward(&vx_scaled, Some(&ax_scaled), None, v_cross_pe, a_cross_pe)?;
                vx = Some((vx_ref + a2v_out.broadcast_mul(&video_av_ada.gate)?)?);
            }

            // Video-to-Audio: Q=audio, K/V=video.
            // Reuse the PRE-A2V vx_norm3 computed above (matching the reference,
            // which normalizes the video stream once before both AV branches).
            if run_v2a {
                let ax_scaled =
                    modulate_normed(&ax_norm3, &audio_av_ada.shift_v2a, &audio_av_ada.scale_v2a)?;
                let vx_scaled =
                    modulate_normed(&vx_norm3, &video_av_ada.shift_v2a, &video_av_ada.scale_v2a)?;

                let a_cross_pe = audio
                    .cross_positional_embeddings
                    .as_ref()
                    .map(|(c, s)| (c, s));
                let v_cross_pe = video
                    .cross_positional_embeddings
                    .as_ref()
                    .map(|(c, s)| (c, s));

                let v2a_attn = self
                    .video_to_audio_attn
                    .as_ref()
                    .ok_or(missing_err("video_to_audio_attn"))?;
                let v2a_out =
                    v2a_attn.forward(&ax_scaled, Some(&vx_scaled), None, a_cross_pe, v_cross_pe)?;
                ax = Some((ax_ref + v2a_out.broadcast_mul(&audio_av_ada.gate)?)?);
            }
        }

        // === Video FFN ===
        if run_vx {
            let v = video.ok_or(missing_err("video modality for FFN"))?;
            let vx_ref = vx.as_ref().ok_or(missing_err("video hidden state"))?;
            vx = Some(apply_ffn(
                vx_ref,
                self.ff.as_ref().ok_or(missing_err("video ff"))?,
                self.scale_shift_table
                    .as_ref()
                    .ok_or(missing_err("video scale_shift_table"))?,
                v,
                batch_size,
                self.norm_eps,
            )?);
        }

        // === Audio FFN ===
        if run_ax {
            let a = audio.ok_or(missing_err("audio modality for FFN"))?;
            let ax_ref = ax.as_ref().ok_or(missing_err("audio hidden state"))?;
            ax = Some(apply_ffn(
                ax_ref,
                self.audio_ff.as_ref().ok_or(missing_err("audio ff"))?,
                self.audio_scale_shift_table
                    .as_ref()
                    .ok_or(missing_err("audio scale_shift_table"))?,
                a,
                batch_size,
                self.norm_eps,
            )?);
        }

        // Return updated TransformerArgs with new x values
        let video_out = video.map(|v| TransformerArgs {
            x: vx.unwrap_or_else(|| v.x.clone()),
            ..v.clone()
        });
        let audio_out = audio.map(|a| TransformerArgs {
            x: ax.unwrap_or_else(|| a.x.clone()),
            ..a.clone()
        });

        Ok((video_out, audio_out))
    }
}

// ── Extracted forward helpers ──────────────────────────────────────────

/// Apply self-attention with AdaLN modulation, followed by text cross-attention.
/// Used identically for both video and audio streams.
#[allow(clippy::too_many_arguments)]
fn apply_self_attn_and_cross_attn(
    x: &Tensor,
    attn1: &Attention,
    attn2: &Attention,
    table: &Tensor,
    prompt_table: Option<&Tensor>,
    args: &TransformerArgs,
    batch_size: usize,
    norm_eps: f64,
    skip_self_attn: bool,
) -> Result<Tensor> {
    // Self-attention with AdaLN (offset=0: shift, scale, gate). When the STG
    // perturbation skips this block's self-attention, the sublayer becomes
    // identity (x unchanged), matching reference SKIP_*_SELF_ATTN (mask=0).
    let x = if skip_self_attn {
        x.clone()
    } else {
        let (shift_msa, scale_msa, gate_msa) =
            get_ada_values_3(table, batch_size, &args.timesteps, 0)?;
        let norm_x = rms_norm(x, None, norm_eps)?;
        let norm_x = modulate_normed(&norm_x, &shift_msa, &scale_msa)?;
        let sa_out = attn1.forward(
            &norm_x,
            None,
            None,
            Some((&args.positional_embeddings.0, &args.positional_embeddings.1)),
            None,
        )?;
        (x + sa_out.broadcast_mul(&gate_msa)?)?
    };

    // Text cross-attention with optional query and prompt modulation.
    let ca_out = if table.dim(0)? >= 9 {
        let (shift_q, scale_q, gate_q) = get_ada_values_3(table, batch_size, &args.timesteps, 6)?;
        let norm_x = rms_norm(&x, None, norm_eps)?;
        let attn_input = modulate_normed(&norm_x, &shift_q, &scale_q)?;
        let context = if let (Some(prompt_table), Some(prompt_timestep)) =
            (prompt_table, args.prompt_timestep.as_ref())
        {
            modulate_prompt_context(&args.context, prompt_table, prompt_timestep, batch_size)?
        } else {
            args.context.clone()
        };
        let ca_out = attn2.forward(
            &attn_input,
            Some(&context),
            args.context_mask.as_ref(),
            None,
            None,
        )?;
        ca_out.broadcast_mul(&gate_q)?
    } else {
        let norm_x = rms_norm(&x, None, norm_eps)?;
        attn2.forward(
            &norm_x,
            Some(&args.context),
            args.context_mask.as_ref(),
            None,
            None,
        )?
    };
    &x + ca_out
}

/// Apply feed-forward network with AdaLN modulation.
/// Used identically for both video and audio streams.
fn apply_ffn(
    x: &Tensor,
    ff: &FeedForward,
    table: &Tensor,
    args: &TransformerArgs,
    batch_size: usize,
    norm_eps: f64,
) -> Result<Tensor> {
    // FFN with AdaLN (offset=3: shift, scale, gate)
    let (shift_mlp, scale_mlp, gate_mlp) = get_ada_values_3(table, batch_size, &args.timesteps, 3)?;

    let norm_x = rms_norm(x, None, norm_eps)?;
    let x_scaled = modulate_normed(&norm_x, &shift_mlp, &scale_mlp)?;
    let ff_out = ff.forward(&x_scaled)?;
    x.add(&ff_out.broadcast_mul(&gate_mlp)?)
}

/// Apply AdaLN scale/shift in fp32, matching the reference stability fix.
fn modulate_normed(normed: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let dtype = normed.dtype();
    let normed = normed.to_dtype(DType::F32)?;
    let shift = shift.to_dtype(DType::F32)?;
    let scale = scale.to_dtype(DType::F32)?;
    let out = (normed.broadcast_mul(&(scale + 1.0)?)? + shift)?;
    out.to_dtype(dtype)
}

fn modulate_prompt_context(
    context: &Tensor,
    prompt_table: &Tensor,
    prompt_timestep: &Tensor,
    batch_size: usize,
) -> Result<Tensor> {
    let dtype = context.dtype();
    let dim = prompt_table.dim(1)?;
    let n = prompt_timestep.elem_count() / (batch_size * 2 * dim);
    let table = prompt_table
        .unsqueeze(0)?
        .unsqueeze(0)?
        .to_dtype(DType::F32)?;
    let ts = prompt_timestep
        .reshape(&[batch_size, n, 2, dim])?
        .to_dtype(DType::F32)?;
    let combined = ts.broadcast_add(&table)?;
    let shift = combined.narrow(2, 0, 1)?.squeeze(2)?;
    let scale = combined.narrow(2, 1, 1)?.squeeze(2)?;
    let out = context
        .to_dtype(DType::F32)?
        .broadcast_mul(&(scale + 1.0)?)?
        .broadcast_add(&shift)?;
    out.to_dtype(dtype)
}

/// Extract 3 adaptive normalization values (shift, scale, gate) from a
/// scale_shift_table at the given offset.
///
/// table: (num_params, dim), e.g. (6, 4096)
/// timestep: (B, T, num_params * dim)
/// offset: start index (0 for self-attn, 3 for FFN)
///
/// Returns 3 tensors each of shape (B, T, dim).
fn get_ada_values_3(
    table: &Tensor,
    batch_size: usize,
    timestep: &Tensor,
    offset: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let num_params = table.dims()[0]; // 6
    let dim = table.dims()[1];
    let n_tokens = timestep.dims()[1];

    // table[offset..offset+3] -> (3, dim) -> (1, 1, 3, dim)
    let sub_table = table
        .narrow(0, offset, 3)?
        .unsqueeze(0)?
        .unsqueeze(0)?
        .to_dtype(timestep.dtype())?;

    // timestep: (B, T, 6*dim) -> (B, T, 6, dim) -> narrow -> (B, T, 3, dim)
    let ts_4d = timestep.reshape(&[batch_size, n_tokens, num_params, dim])?;
    let ts_slice = ts_4d.narrow(2, offset, 3)?;

    let combined = ts_slice.broadcast_add(&sub_table)?;

    let v0 = combined.narrow(2, 0, 1)?.squeeze(2)?;
    let v1 = combined.narrow(2, 1, 1)?.squeeze(2)?;
    let v2 = combined.narrow(2, 2, 1)?.squeeze(2)?;

    Ok((v0, v1, v2))
}

/// AV cross-attention adaptive values.
///
/// 5-row table: 4 scale/shift + 1 gate.
/// Returns scale_a2v, shift_a2v, scale_v2a, shift_v2a, gate.
struct AvCaAdaValues {
    scale_a2v: Tensor,
    shift_a2v: Tensor,
    scale_v2a: Tensor,
    shift_v2a: Tensor,
    gate: Tensor,
}

fn get_av_ca_ada_values(
    table: &Tensor,
    batch_size: usize,
    scale_shift_timestep: &Tensor,
    gate_timestep: &Tensor,
) -> Result<AvCaAdaValues> {
    let dim = table.dims()[1];

    // Scale-shift part: table[:4] with scale_shift_timestep
    let ss_table = table.narrow(0, 0, 4)?;
    let ss_n_tokens = scale_shift_timestep.dims()[1];
    let ss_sub = ss_table
        .unsqueeze(0)?
        .unsqueeze(0)?
        .to_dtype(scale_shift_timestep.dtype())?;
    let ss_ts = scale_shift_timestep.reshape(&[batch_size, ss_n_tokens, 4, dim])?;
    let ss_combined = ss_ts.broadcast_add(&ss_sub)?;

    let scale_a2v = ss_combined.narrow(2, 0, 1)?.squeeze(2)?;
    let shift_a2v = ss_combined.narrow(2, 1, 1)?.squeeze(2)?;
    let scale_v2a = ss_combined.narrow(2, 2, 1)?.squeeze(2)?;
    let shift_v2a = ss_combined.narrow(2, 3, 1)?.squeeze(2)?;

    // Gate part: table[4:] with gate_timestep
    let gate_table = table.narrow(0, 4, 1)?;
    let gate_n_tokens = gate_timestep.dims()[1];
    let gate_sub = gate_table
        .unsqueeze(0)?
        .unsqueeze(0)?
        .to_dtype(gate_timestep.dtype())?;
    let gate_ts = gate_timestep.reshape(&[batch_size, gate_n_tokens, 1, dim])?;
    let gate_combined = gate_ts.broadcast_add(&gate_sub)?;
    let gate = gate_combined.narrow(2, 0, 1)?.squeeze(2)?;

    Ok(AvCaAdaValues {
        scale_a2v,
        shift_a2v,
        scale_v2a,
        shift_v2a,
        gate,
    })
}
