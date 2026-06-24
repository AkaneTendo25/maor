// Rust-only LTX-2.3 prompt-conditioned inference.
//
// prompt -> Gemma3 -> dual text projection/connectors -> transformer denoise
// -> VAE/audio decode -> MP4.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::{Parser, ValueEnum};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use serde_json::Value;

use maor_audio_vae::causal_conv2d::CausalityAxis;
use maor_audio_vae::decoder::AudioDecoder;
use maor_audio_vae::vocoder::VocoderWithBwe;
use maor_core::config::{LtxModelType, SchedulerConfig, TransformerConfig, VaeConfig};
use maor_core::ops::to_denoised;
use maor_core::patchify::{get_pixel_coords, AudioPatchifier, VideoLatentPatchifier};
use maor_core::statistics::PerChannelStatistics;
use maor_core::types::{AudioLatentShape, VideoLatentShape, VIDEO_SCALE_FACTORS};
use maor_nn::conv3d::SpatialPaddingMode;
use maor_nn::lora::LoraConfig;
use maor_scheduler::diffusion_step::{euler_step, res2s_midpoint, res2s_step};
use maor_scheduler::guiders::multi_modal_guide;
use maor_scheduler::schedule::maor_schedule;
use maor_text_encoder::encoder::{AVGemmaEncoderOutput, AVGemmaTextEncoder};
use maor_transformer::modality::Modality;
use maor_transformer::model::LTXModel;
use maor_video_vae::decoder::VideoDecoder;
use maor_video_vae::upsampler::LatentUpsampler;

const DEFAULT_NEGATIVE_PROMPT: &str = concat!(
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, ",
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, ",
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, ",
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of ",
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent ",
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny ",
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, ",
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, ",
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward ",
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, ",
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
);

const STAGE2_SIGMAS: [f32; 4] = [0.909375, 0.725, 0.421875, 0.0];
const DISTILLED_TWO_STAGE_SIGMAS: [f32; 9] = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0,
];
const DISTILLED_STAGE2_SIGMA_START: usize = 5;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum GenerationMode {
    Video,
    Av,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum SamplerKind {
    Euler,
    Res2s,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum StgMode {
    Video,
    Audio,
    Both,
}

#[derive(Parser, Debug)]
#[command(
    name = "ner",
    about = "Rust-only LTX-2.3 prompt-conditioned video inference"
)]
struct Args {
    /// LTX-2.3 single-file checkpoint.
    #[arg(long)]
    checkpoint: PathBuf,

    /// Directory containing Gemma3 safetensors shards.
    #[arg(long)]
    gemma_dir: PathBuf,

    /// Gemma tokenizer.json path.
    #[arg(long)]
    tokenizer: PathBuf,

    /// Text prompt.
    #[arg(long)]
    prompt: String,

    /// Negative prompt for CFG.
    #[arg(long)]
    negative_prompt: Option<String>,

    /// Disable the reference LTX-2.3 default negative prompt when CFG is enabled.
    #[arg(long)]
    no_default_negative_prompt: bool,

    /// Generation modality. Reference standalone generation defaults to video.
    #[arg(long, value_enum, default_value_t = GenerationMode::Video)]
    mode: GenerationMode,

    /// Sampler to use for the video latent. Reference video mode uses RES_2S.
    #[arg(long, value_enum, default_value_t = SamplerKind::Res2s)]
    sampler: SamplerKind,

    /// Output MP4 path.
    #[arg(long, default_value = "outputs/ner.mp4")]
    output: PathBuf,

    /// Output video width in pixels.
    #[arg(long, default_value_t = 960)]
    width: usize,

    /// Output video height in pixels.
    #[arg(long, default_value_t = 544)]
    height: usize,

    /// Number of frames. LTX expects 1 + 8*k.
    #[arg(long, default_value_t = 121)]
    frames: usize,

    /// Frames per second.
    #[arg(long, default_value_t = 24.0)]
    fps: f64,

    /// Denoising steps.
    #[arg(long, default_value_t = 15)]
    steps: usize,

    /// CFG scale. Use 1.0 to disable CFG.
    #[arg(long, default_value_t = 3.0)]
    cfg_scale: f64,

    /// CFG rescale for video x0 after guidance.
    #[arg(long, default_value_t = 0.9)]
    video_rescale_scale: f64,

    /// Random seed for initial video/audio latents.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Use f32 instead of bf16 for model weights/activations.
    #[arg(long)]
    f32: bool,

    /// Disable temporal tiling during VAE decode.
    #[arg(long)]
    no_vae_temporal_tiling: bool,

    /// Maximum latent frames per VAE temporal decode tile.
    #[arg(long, default_value_t = 5)]
    vae_temporal_tile_latents: usize,

    /// Overlapping latent frames between VAE temporal decode tiles.
    #[arg(long, default_value_t = 2)]
    vae_temporal_overlap_latents: usize,

    /// Spatio-Temporal Guidance scale (LTX-2.3 default = 1.0; 0 disables).
    #[arg(long, default_value_t = 1.0)]
    stg_scale: f64,

    /// Transformer block index whose self-attention STG skips (reference = 28).
    #[arg(long, default_value_t = 28)]
    stg_block: usize,

    /// Which modality STG perturbs.
    #[arg(long, value_enum, default_value_t = StgMode::Video)]
    stg_mode: StgMode,

    /// Enable two-stage generation with latent spatial upscaling and refinement.
    #[arg(long)]
    two_stage: bool,

    /// LTX-2.3 spatial upscaler checkpoint used by two-stage generation.
    #[arg(long)]
    spatial_upscaler: Option<PathBuf>,

    /// Number of second-stage refinement steps.
    #[arg(long, default_value_t = 3)]
    stage2_steps: usize,

    /// Optional distilled LoRA checkpoint for HQ two-stage inference.
    #[arg(long)]
    distilled_lora: Option<PathBuf>,

    /// LoRA multiplier for first-stage generation.
    #[arg(long, default_value_t = 0.0)]
    stage1_lora_scale: f64,

    /// LoRA multiplier for second-stage refinement.
    #[arg(long, default_value_t = 1.0)]
    stage2_lora_scale: f64,
}

fn seeded_randn(shape: &[usize], seed: u64, dtype: DType, device: &Device) -> Result<Tensor> {
    let mut rng = StdRng::seed_from_u64(seed);
    let numel: usize = shape.iter().product();
    let vals: Vec<f32> = (0..numel)
        .map(|_| StandardNormal.sample(&mut rng))
        .collect();
    Ok(Tensor::from_vec(vals, shape, device)?.to_dtype(dtype)?)
}

fn read_safetensors_header(path: &Path) -> Result<Value> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .with_context(|| format!("read safetensors header length from {}", path.display()))?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    if header_len > 128 * 1024 * 1024 {
        bail!("safetensors header is unexpectedly large: {header_len} bytes");
    }
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)
        .with_context(|| format!("read safetensors header from {}", path.display()))?;
    Ok(serde_json::from_slice(&header)?)
}

fn load_configs(path: &Path) -> Result<(TransformerConfig, VaeConfig)> {
    let header = read_safetensors_header(path)?;
    let metadata = header
        .get("__metadata__")
        .and_then(|v| v.as_object())
        .context("safetensors header has no __metadata__ object")?;
    let config_json = metadata
        .get("config")
        .and_then(|v| v.as_str())
        .context("safetensors metadata has no config string")?;
    let config: Value = serde_json::from_str(config_json).context("parse metadata config JSON")?;
    let transformer = serde_json::from_value(
        config
            .get("transformer")
            .context("metadata config has no transformer section")?
            .clone(),
    )
    .context("parse TransformerConfig")?;
    let vae = serde_json::from_value(
        config
            .get("vae")
            .context("metadata config has no vae section")?
            .clone(),
    )
    .context("parse VaeConfig")?;
    Ok((transformer, vae))
}

fn find_safetensors(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("safetensors") {
            paths.push(path);
        }
    }
    paths.sort();
    if paths.is_empty() {
        bail!("no .safetensors files in {}", dir.display());
    }
    Ok(paths)
}

fn load_vb(paths: &[PathBuf], dtype: DType, device: &Device) -> Result<VarBuilder<'static>> {
    let path_refs: Vec<&Path> = paths.iter().map(|p| p.as_path()).collect();
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&path_refs, dtype, device)? })
}

fn select_gemma_root(vb: VarBuilder) -> Result<VarBuilder> {
    let hf_root = vb.pp("language_model").pp("model");
    if hf_root.pp("embed_tokens").contains_tensor("weight") {
        return Ok(hf_root);
    }

    let model_root = vb.pp("model");
    if model_root.pp("embed_tokens").contains_tensor("weight") {
        return Ok(model_root);
    }

    if vb.pp("embed_tokens").contains_tensor("weight") {
        return Ok(vb);
    }

    bail!("could not find Gemma embed_tokens.weight under known roots")
}

fn stats(name: &str, tensor: &Tensor) -> Result<()> {
    let values: Vec<f32> = tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let finite = values.iter().all(|x| x.is_finite());
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let min = values.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
    let max = values.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    println!(
        "{name}: shape={:?} finite={finite} mean={mean:.5} std={:.5} min={min:.5} max={max:.5}",
        tensor.dims(),
        var.sqrt()
    );
    if !finite {
        bail!("{name} contains non-finite values");
    }
    Ok(())
}

fn encode_prompt(
    args: &Args,
    cfg: &TransformerConfig,
    dtype: DType,
    device: &Device,
) -> Result<(AVGemmaEncoderOutput, Option<AVGemmaEncoderOutput>)> {
    println!("loading Gemma + 2.3 text projection/connectors");
    let gemma_files = find_safetensors(&args.gemma_dir)?;
    let gemma_vb = load_vb(&gemma_files, dtype, device)?;
    let gemma_root = select_gemma_root(gemma_vb)?;
    let cond_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[args.checkpoint.as_path()], dtype, device)?
    };
    let encoder = AVGemmaTextEncoder::new_with_transformer_config(
        args.tokenizer
            .to_str()
            .context("tokenizer path contains invalid UTF-8")?,
        cfg,
        gemma_root,
        cond_vb.pp("text_embedding_projection"),
        cond_vb
            .pp("model")
            .pp("diffusion_model")
            .pp("video_embeddings_connector"),
        cond_vb
            .pp("model")
            .pp("diffusion_model")
            .pp("audio_embeddings_connector"),
    )?;

    let t = Instant::now();
    let pos = encoder.forward(&args.prompt)?;
    let neg = if args.cfg_scale > 1.0 {
        let negative_prompt =
            args.negative_prompt
                .as_deref()
                .unwrap_or(if args.no_default_negative_prompt {
                    ""
                } else {
                    DEFAULT_NEGATIVE_PROMPT
                });
        Some(encoder.forward(negative_prompt)?)
    } else {
        None
    };
    println!("text encoded in {:.1}s", t.elapsed().as_secs_f64());
    stats("pos.video_encoding", &pos.video_encoding)?;
    stats("pos.audio_encoding", &pos.audio_encoding)?;
    if let Some(neg) = &neg {
        stats("neg.video_encoding", &neg.video_encoding)?;
        stats("neg.audio_encoding", &neg.audio_encoding)?;
        let video_delta = (&pos.video_encoding - &neg.video_encoding)?;
        let audio_delta = (&pos.audio_encoding - &neg.audio_encoding)?;
        stats("text_delta.video_encoding", &video_delta)?;
        stats("text_delta.audio_encoding", &audio_delta)?;
    }
    Ok((pos, neg))
}

fn tensor_to_rgb_frames(video: &Tensor) -> Result<(Vec<Vec<u8>>, usize, usize)> {
    let (_b, _c, f, h, w) = video.dims5()?;
    let video = video.squeeze(0)?;
    let video = ((video + 1.0)? * 0.5)?;
    let video = video.clamp(0.0, 1.0)?;
    let video = (video * 255.0)?;
    let video = video.to_dtype(DType::U8)?;
    let video = video.permute((1, 2, 3, 0))?;

    let mut frames = Vec::with_capacity(f);
    for i in 0..f {
        let frame = video.narrow(0, i, 1)?.squeeze(0)?;
        frames.push(frame.flatten_all()?.to_vec1()?);
    }
    Ok((frames, h, w))
}

fn write_video_ffmpeg(
    frames: &[Vec<u8>],
    width: usize,
    height: usize,
    fps: f64,
    output_path: &Path,
) -> Result<()> {
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let mut child = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            &format!("{width}x{height}"),
            "-framerate",
            &format!("{fps}"),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "fast",
        ])
        .arg(output_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .context("start ffmpeg")?;

    {
        let stdin = child.stdin.as_mut().context("open ffmpeg stdin")?;
        for frame in frames {
            stdin.write_all(frame)?;
        }
    }

    let status = child.wait()?;
    if !status.success() {
        bail!("ffmpeg exited with {status}");
    }
    Ok(())
}

fn sidecar_path(output_path: &Path, suffix: &str, extension: &str) -> PathBuf {
    let parent = output_path.parent().unwrap_or_else(|| Path::new(""));
    let stem = output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("maor_output");
    parent.join(format!("{stem}.{suffix}.{extension}"))
}

fn write_wav(audio: &Tensor, sample_rate: u32, output_path: &Path) -> Result<()> {
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let audio = audio.to_dtype(DType::F32)?;
    let (channels, samples) = audio.dims2()?;
    let spec = hound::WavSpec {
        channels: channels as u16,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(output_path, spec)?;
    for s in 0..samples {
        for c in 0..channels {
            let val: f32 = audio.get(c)?.get(s)?.to_scalar()?;
            writer.write_sample((val.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
    }
    writer.finalize()?;
    Ok(())
}

fn mux_av(video_path: &Path, audio_path: &Path, output_path: &Path) -> Result<()> {
    let status = Command::new("ffmpeg")
        .args(["-y", "-i"])
        .arg(video_path)
        .arg("-i")
        .arg(audio_path)
        .args([
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        ])
        .arg(output_path)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .status()
        .context("start ffmpeg mux")?;

    if !status.success() {
        bail!("ffmpeg mux exited with {status}");
    }
    Ok(())
}

fn video_positions_in_seconds(positions: &Tensor, fps: f64) -> Result<Tensor> {
    let time = (positions.narrow(1, 0, 1)? * (1.0 / fps))?;
    let spatial = positions.narrow(1, 1, 2)?;
    Ok(Tensor::cat(&[&time, &spatial], 1)?)
}

fn make_video_positions(
    patchifier: &VideoLatentPatchifier,
    shape: &VideoLatentShape,
    fps: f64,
    device: &Device,
) -> Result<Tensor> {
    let positions = get_pixel_coords(
        &patchifier.get_patch_grid_bounds(shape, device)?,
        &VIDEO_SCALE_FACTORS,
        true,
    )?;
    video_positions_in_seconds(&positions, fps)
}

fn add_noise_at_sigma(clean: &Tensor, noise: Tensor, sigma: f32) -> Result<Tensor> {
    let clean_part = (clean * (1.0 - sigma as f64))?;
    let noise_part = (noise * sigma as f64)?;
    Ok((clean_part + noise_part)?)
}

#[allow(clippy::too_many_arguments)]
fn predict_denoised(
    model: &LTXModel,
    pos: &AVGemmaEncoderOutput,
    neg: Option<&AVGemmaEncoderOutput>,
    video_latent: &Tensor,
    audio_latent: Option<&Tensor>,
    video_positions: &Tensor,
    audio_positions: Option<&Tensor>,
    video_denoise_mask: &Tensor,
    audio_denoise_mask: Option<&Tensor>,
    sigma: f32,
    cfg_scale: f64,
    video_rescale_scale: f64,
    stg_scale: f64,
    stg_blocks: &[usize],
    stg_mode: StgMode,
    dtype: DType,
    device: &Device,
    stats_prefix: Option<&str>,
) -> Result<(Tensor, Option<Tensor>, usize)> {
    let sigma_t = Tensor::new(&[sigma], device)?.to_dtype(dtype)?;
    let video_timesteps = (video_denoise_mask * sigma as f64)?;
    let audio_timesteps = match audio_denoise_mask {
        Some(mask) => Some((mask * sigma as f64)?),
        None => None,
    };

    let v_modality = Modality {
        latent: video_latent.clone(),
        timesteps: video_timesteps.clone(),
        positions: video_positions.clone(),
        context: pos.video_encoding.clone(),
        enabled: true,
        context_mask: Some(pos.attention_mask.clone()),
        sigma: Some(sigma_t.clone()),
    };
    let a_modality = match (audio_latent, audio_positions, audio_timesteps.as_ref()) {
        (Some(latent), Some(positions), Some(timesteps)) => Some(Modality {
            latent: latent.clone(),
            timesteps: timesteps.clone(),
            positions: positions.clone(),
            context: pos.audio_encoding.clone(),
            enabled: true,
            context_mask: Some(pos.attention_mask.clone()),
            sigma: Some(sigma_t.clone()),
        }),
        _ => None,
    };

    let (v_vel_cond, a_vel_cond) = model.forward(Some(&v_modality), a_modality.as_ref())?;
    let v_denoised_cond = to_denoised(
        video_latent,
        &v_vel_cond.context("missing conditioned video velocity")?,
        &sigma_t,
    )?;
    let a_denoised_cond = match (audio_latent, a_vel_cond) {
        (Some(latent), Some(vel)) => Some(to_denoised(latent, &vel, &sigma_t)?),
        (Some(_), None) => bail!("missing conditioned audio velocity"),
        _ => None,
    };

    let stg_video_active = stg_scale != 0.0 && matches!(stg_mode, StgMode::Video | StgMode::Both);
    let stg_audio_active = stg_scale != 0.0
        && audio_latent.is_some()
        && matches!(stg_mode, StgMode::Audio | StgMode::Both);
    let mut v_denoised_stg = None;
    let mut a_denoised_stg = None;
    let mut passes = 1;

    if stg_video_active || stg_audio_active {
        let skip_video = if stg_video_active { stg_blocks } else { &[] };
        let skip_audio = if stg_audio_active { stg_blocks } else { &[] };
        let (v_vel_stg, a_vel_stg) = model.forward_perturbed(
            Some(&v_modality),
            a_modality.as_ref(),
            skip_video,
            skip_audio,
        )?;
        if stg_video_active {
            v_denoised_stg = Some(to_denoised(
                video_latent,
                &v_vel_stg.context("missing STG video velocity")?,
                &sigma_t,
            )?);
        }
        if stg_audio_active {
            let audio_latent = audio_latent.context("missing STG audio latent")?;
            let vel = a_vel_stg.context("missing STG audio velocity")?;
            a_denoised_stg = Some(to_denoised(audio_latent, &vel, &sigma_t)?);
        }
        passes += 1;
    }

    if let Some(neg) = neg {
        let v_modality_neg = Modality {
            latent: video_latent.clone(),
            timesteps: video_timesteps,
            positions: video_positions.clone(),
            context: neg.video_encoding.clone(),
            enabled: true,
            context_mask: Some(neg.attention_mask.clone()),
            sigma: Some(sigma_t.clone()),
        };
        let a_modality_neg = match (audio_latent, audio_positions, audio_timesteps.as_ref()) {
            (Some(latent), Some(positions), Some(timesteps)) => Some(Modality {
                latent: latent.clone(),
                timesteps: timesteps.clone(),
                positions: positions.clone(),
                context: neg.audio_encoding.clone(),
                enabled: true,
                context_mask: Some(neg.attention_mask.clone()),
                sigma: Some(sigma_t.clone()),
            }),
            _ => None,
        };

        let (v_vel_uncond, a_vel_uncond) =
            model.forward(Some(&v_modality_neg), a_modality_neg.as_ref())?;
        let v_denoised_uncond = to_denoised(
            video_latent,
            &v_vel_uncond.context("missing uncond video velocity")?,
            &sigma_t,
        )?;
        let a_denoised_uncond = match (audio_latent, a_vel_uncond) {
            (Some(latent), Some(vel)) => Some(to_denoised(latent, &vel, &sigma_t)?),
            (Some(_), None) => bail!("missing uncond audio velocity"),
            _ => None,
        };

        let v_denoised = multi_modal_guide(
            &v_denoised_cond,
            Some(&v_denoised_uncond),
            v_denoised_stg.as_ref(),
            None,
            cfg_scale,
            stg_scale,
            1.0,
            video_rescale_scale,
        )?;
        let a_denoised = match (a_denoised_cond.as_ref(), a_denoised_uncond.as_ref()) {
            (Some(cond), Some(uncond)) => Some(multi_modal_guide(
                cond,
                Some(uncond),
                a_denoised_stg.as_ref(),
                None,
                cfg_scale,
                stg_scale,
                1.0,
                0.0,
            )?),
            _ => None,
        };

        if let Some(prefix) = stats_prefix {
            let v_delta = (&v_denoised - &v_denoised_cond)?;
            stats(&format!("{prefix}.video_denoised_cond"), &v_denoised_cond)?;
            stats(
                &format!("{prefix}.video_denoised_uncond"),
                &v_denoised_uncond,
            )?;
            stats(&format!("{prefix}.video_guidance_delta"), &v_delta)?;
            if let Some(stg) = &v_denoised_stg {
                let stg_delta = (&v_denoised_cond - stg)?;
                stats(&format!("{prefix}.video_stg_delta"), &stg_delta)?;
            }
            if let (Some(cond), Some(uncond), Some(guided)) = (
                a_denoised_cond.as_ref(),
                a_denoised_uncond.as_ref(),
                a_denoised.as_ref(),
            ) {
                let a_delta = (guided - cond)?;
                stats(&format!("{prefix}.audio_denoised_cond"), cond)?;
                stats(&format!("{prefix}.audio_denoised_uncond"), uncond)?;
                stats(&format!("{prefix}.audio_guidance_delta"), &a_delta)?;
                if let Some(stg) = &a_denoised_stg {
                    let stg_delta = (cond - stg)?;
                    stats(&format!("{prefix}.audio_stg_delta"), &stg_delta)?;
                }
            }
        }

        Ok((v_denoised, a_denoised, passes + 1))
    } else {
        let v_denoised = if stg_scale != 0.0 && v_denoised_stg.is_some() {
            multi_modal_guide(
                &v_denoised_cond,
                None,
                v_denoised_stg.as_ref(),
                None,
                1.0,
                stg_scale,
                1.0,
                video_rescale_scale,
            )?
        } else {
            v_denoised_cond
        };
        let a_denoised = match (a_denoised_cond.as_ref(), a_denoised_stg.as_ref()) {
            (Some(cond), Some(stg)) if stg_scale != 0.0 => Some(multi_modal_guide(
                cond,
                None,
                Some(stg),
                None,
                1.0,
                stg_scale,
                1.0,
                0.0,
            )?),
            _ => a_denoised_cond,
        };
        Ok((v_denoised, a_denoised, passes))
    }
}

#[allow(clippy::too_many_arguments)]
fn run_denoise_loop(
    model: &LTXModel,
    pos: &AVGemmaEncoderOutput,
    neg: Option<&AVGemmaEncoderOutput>,
    mut video_latent: Tensor,
    mut audio_latent: Option<Tensor>,
    video_positions: &Tensor,
    audio_positions: Option<&Tensor>,
    video_denoise_mask: &Tensor,
    audio_denoise_mask: Option<&Tensor>,
    sigmas: &[f32],
    sampler: SamplerKind,
    cfg_scale: f64,
    video_rescale_scale: f64,
    stg_scale: f64,
    stg_blocks: &[usize],
    stg_mode: StgMode,
    dtype: DType,
    device: &Device,
    label: &str,
) -> Result<(Tensor, Option<Tensor>)> {
    if sigmas.len() < 2 {
        bail!("denoise loop requires at least two sigma values");
    }
    if sampler == SamplerKind::Res2s && audio_latent.is_some() {
        bail!("RES_2S sampling is only supported for video-only refinement");
    }

    let total_steps = sigmas.len() - 1;
    for step_idx in 0..total_steps {
        let step_start = Instant::now();
        let sigma = sigmas[step_idx];
        let stats_prefix = (step_idx == 0).then(|| format!("{label}.step1"));
        let (v_denoised, a_denoised, mut passes) = predict_denoised(
            model,
            pos,
            neg,
            &video_latent,
            audio_latent.as_ref(),
            video_positions,
            audio_positions,
            video_denoise_mask,
            audio_denoise_mask,
            sigma,
            cfg_scale,
            video_rescale_scale,
            stg_scale,
            stg_blocks,
            stg_mode,
            dtype,
            device,
            stats_prefix.as_deref(),
        )?;

        match sampler {
            SamplerKind::Euler => {
                video_latent = euler_step(&video_latent, &v_denoised, sigmas, step_idx)?;
                let next_audio = match (audio_latent.as_ref(), a_denoised.as_ref()) {
                    (Some(latent), Some(denoised)) => {
                        Some(euler_step(latent, denoised, sigmas, step_idx)?)
                    }
                    _ => None,
                };
                if next_audio.is_some() {
                    audio_latent = next_audio;
                }
            }
            SamplerKind::Res2s => {
                if let Some((midpoint_latent, midpoint_sigma)) =
                    res2s_midpoint(&video_latent, &v_denoised, sigma, sigmas[step_idx + 1])?
                {
                    let (midpoint_denoised, _, midpoint_passes) = predict_denoised(
                        model,
                        pos,
                        neg,
                        &midpoint_latent,
                        None,
                        video_positions,
                        None,
                        video_denoise_mask,
                        None,
                        midpoint_sigma,
                        cfg_scale,
                        video_rescale_scale,
                        stg_scale,
                        stg_blocks,
                        stg_mode,
                        dtype,
                        device,
                        None,
                    )?;
                    passes += midpoint_passes;
                    video_latent = res2s_step(
                        &video_latent,
                        &v_denoised,
                        &midpoint_denoised,
                        sigmas,
                        step_idx,
                    )?;
                } else {
                    video_latent = v_denoised;
                }
            }
        }
        println!(
            "{label} step {}/{} sigma={sigma:.4} {}x fwd elapsed={:.2}s",
            step_idx + 1,
            total_steps,
            passes,
            step_start.elapsed().as_secs_f64()
        );
    }

    Ok((video_latent, audio_latent))
}

fn load_transformer_model(
    args: &Args,
    cfg: &TransformerConfig,
    dtype: DType,
    device: &Device,
    lora_scale: f64,
    label: &str,
) -> Result<LTXModel> {
    let transformer_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[args.checkpoint.as_path()], dtype, device)?
    };
    let lora_config = if let Some(path) = &args.distilled_lora {
        if lora_scale == 0.0 {
            None
        } else {
            println!("loading {label} transformer with LoRA scale={lora_scale}");
            let lora_vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&[path.as_path()], dtype, device)? };
            Some(LoraConfig::new(lora_vb.pp("diffusion_model"), lora_scale))
        }
    } else {
        println!("loading {label} transformer");
        None
    };

    LTXModel::new_with_lora(
        cfg,
        LtxModelType::AudioVideo,
        transformer_vb.pp("model").pp("diffusion_model"),
        lora_config.as_ref(),
    )
    .context("LTXModel::new_with_lora")
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.width % 32 != 0 || args.height % 32 != 0 {
        bail!("width and height must be divisible by 32");
    }
    if (args.frames - 1) % 8 != 0 {
        bail!("frames must be 1 + 8*k");
    }
    if args.steps == 0 {
        bail!("steps must be >= 1");
    }
    if args.two_stage {
        if args.spatial_upscaler.is_none() {
            bail!("--spatial-upscaler is required with --two-stage");
        }
        if args.width % 64 != 0 || args.height % 64 != 0 {
            bail!("two-stage output width and height must be divisible by 64");
        }
        if args.stage2_steps == 0 || args.stage2_steps > 3 {
            bail!("--stage2-steps must be in 1..=3");
        }
        if args.distilled_lora.is_some() {
            let expected_steps = DISTILLED_TWO_STAGE_SIGMAS.len() - 1;
            if args.steps != expected_steps {
                bail!("distilled two-stage inference requires --steps {expected_steps}");
            }
            if args.stage2_steps != 3 {
                bail!("distilled two-stage inference requires --stage2-steps 3");
            }
        }
    }

    let start = Instant::now();
    let device = Device::new_cuda(0).context("Device::new_cuda(0)")?;
    let dtype = if args.f32 { DType::F32 } else { DType::BF16 };
    println!("device=cuda:0 dtype={dtype:?}");

    let (transformer_cfg, vae_cfg) = load_configs(&args.checkpoint)?;
    if !transformer_cfg.cross_attention_adaln || !transformer_cfg.caption_proj_before_connector {
        bail!("checkpoint config does not look like LTX-2.3");
    }
    if vae_cfg.decoder_blocks.is_empty() {
        bail!("VAE config decoder_blocks is empty");
    }
    println!(
        "2.3 config: layers={} video_dim={} audio_dim={} connector_layers={} mode={:?} sampler={:?} cfg_scale={} rescale={}",
        transformer_cfg.num_layers,
        transformer_cfg.video_inner_dim(),
        transformer_cfg.audio_inner_dim(),
        transformer_cfg.connector_num_layers,
        args.mode,
        args.sampler,
        args.cfg_scale,
        args.video_rescale_scale
    );

    let (pos, neg) = encode_prompt(&args, &transformer_cfg, dtype, &device)?;

    let stage_width = if args.two_stage {
        args.width / 2
    } else {
        args.width
    };
    let stage_height = if args.two_stage {
        args.height / 2
    } else {
        args.height
    };
    let stage_video_shape = VideoLatentShape::from_pixel_shape(
        1,
        args.frames,
        stage_height,
        stage_width,
        128,
        &VIDEO_SCALE_FACTORS,
    );
    let final_video_shape = VideoLatentShape::from_pixel_shape(
        1,
        args.frames,
        args.height,
        args.width,
        128,
        &VIDEO_SCALE_FACTORS,
    );
    let video_patchifier = VideoLatentPatchifier::new(1);
    let stage_num_video_tokens = video_patchifier.get_token_count(&stage_video_shape);
    let final_num_video_tokens = video_patchifier.get_token_count(&final_video_shape);
    let use_audio = args.mode == GenerationMode::Av;
    let audio_patchifier = AudioPatchifier::new(1);
    let audio_shape = if use_audio {
        Some(AudioLatentShape::from_video_frames(
            1,
            args.frames,
            args.fps,
            8,
            16,
            16000,
            160,
            4,
        ))
    } else {
        None
    };
    let num_audio_tokens = audio_shape
        .as_ref()
        .map(|shape| audio_patchifier.get_token_count(shape))
        .unwrap_or(0);

    let video_latent = video_patchifier.patchify(&seeded_randn(
        &stage_video_shape.to_vec(),
        args.seed,
        dtype,
        &device,
    )?)?;
    let audio_latent = if let Some(shape) = &audio_shape {
        Some(audio_patchifier.patchify(&seeded_randn(
            &shape.to_vec(),
            args.seed.wrapping_add(1),
            dtype,
            &device,
        )?)?)
    } else {
        None
    };
    let stage_video_positions =
        make_video_positions(&video_patchifier, &stage_video_shape, args.fps, &device)?;
    let audio_positions = if let Some(shape) = &audio_shape {
        Some(audio_patchifier.get_patch_grid_bounds(shape, &device)?)
    } else {
        None
    };

    let model = load_transformer_model(
        &args,
        &transformer_cfg,
        dtype,
        &device,
        args.stage1_lora_scale,
        "stage1",
    )?;

    let scheduler_config = SchedulerConfig::default();
    let sigmas = if args.two_stage && args.distilled_lora.is_some() {
        DISTILLED_TWO_STAGE_SIGMAS.to_vec()
    } else {
        maor_schedule(
            args.steps,
            Some(stage_num_video_tokens),
            scheduler_config.max_shift,
            scheduler_config.base_shift,
            scheduler_config.stretch,
            scheduler_config.terminal,
        )
    };
    let stage1_steps = sigmas.len() - 1;
    let effective_sampler = if args.sampler == SamplerKind::Res2s && use_audio {
        println!("RES_2S is video-only in the reference path when audio latents are present; using Euler for AV mode");
        SamplerKind::Euler
    } else {
        args.sampler
    };
    println!(
        "stage1 denoising {} steps: sigma {:.4} -> {:.4}, video_tokens={}, audio_tokens={}, sampler={:?}",
        stage1_steps,
        sigmas[0],
        sigmas[sigmas.len() - 1],
        stage_num_video_tokens,
        num_audio_tokens,
        effective_sampler
    );

    let stage_video_denoise_mask = Tensor::ones((1, stage_num_video_tokens), dtype, &device)?;
    let audio_denoise_mask = if use_audio {
        Some(Tensor::ones((1, num_audio_tokens), dtype, &device)?)
    } else {
        None
    };
    let neg_ref = neg.as_ref();
    let stg_blocks = if args.stg_scale > 0.0 {
        vec![args.stg_block]
    } else {
        Vec::new()
    };

    let (stage_video_latent, stage_audio_latent) = run_denoise_loop(
        &model,
        &pos,
        neg_ref,
        video_latent,
        audio_latent,
        &stage_video_positions,
        audio_positions.as_ref(),
        &stage_video_denoise_mask,
        audio_denoise_mask.as_ref(),
        &sigmas,
        effective_sampler,
        args.cfg_scale,
        args.video_rescale_scale,
        args.stg_scale,
        &stg_blocks,
        args.stg_mode,
        dtype,
        &device,
        "stage1",
    )?;
    drop(model);

    let (video_latent, audio_latent, decode_video_shape) = if args.two_stage {
        stats("stage1.video_latent_tokens", &stage_video_latent)?;
        println!("loading latent spatial upscaler");
        let stage_video_latent_5d = video_patchifier
            .unpatchify(&stage_video_latent, &stage_video_shape)?
            .to_dtype(dtype)?;
        stats("stage1.video_latent_5d", &stage_video_latent_5d)?;

        let stats_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[args.checkpoint.as_path()], dtype, &device)?
        };
        let latent_stats =
            PerChannelStatistics::from_vb(128, stats_vb.pp("vae").pp("per_channel_statistics"))?;

        let upscaler_path = args
            .spatial_upscaler
            .as_ref()
            .context("--spatial-upscaler is required with --two-stage")?;
        let upscaler_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[upscaler_path.as_path()], dtype, &device)?
        };
        let upscaler = LatentUpsampler::new_x2(128, 1024, 4, upscaler_vb)
            .context("LatentUpsampler::new_x2")?;

        let unnormalized = latent_stats.denormalize(&stage_video_latent_5d)?;
        let upsampled = upscaler
            .forward(&unnormalized)
            .context("LatentUpsampler::forward")?;
        let upsampled = latent_stats.normalize(&upsampled)?;
        drop(upscaler);
        stats("stage2.upsampled_video_latent_5d", &upsampled)?;

        let mut stage2_audio_latent = stage_audio_latent;
        let mut stage2_video_latent = video_patchifier.patchify(&upsampled)?;

        let sigma0 = STAGE2_SIGMAS[0];
        let video_noise = video_patchifier.patchify(&seeded_randn(
            &final_video_shape.to_vec(),
            args.seed.wrapping_add(2),
            dtype,
            &device,
        )?)?;
        stage2_video_latent = add_noise_at_sigma(&stage2_video_latent, video_noise, sigma0)?;

        if let (Some(audio_latent), Some(audio_shape)) =
            (stage2_audio_latent.as_ref(), audio_shape.as_ref())
        {
            let audio_noise = audio_patchifier.patchify(&seeded_randn(
                &audio_shape.to_vec(),
                args.seed.wrapping_add(3),
                dtype,
                &device,
            )?)?;
            stage2_audio_latent = Some(add_noise_at_sigma(audio_latent, audio_noise, sigma0)?);
        }

        let stage2_sigmas = if args.distilled_lora.is_some() {
            DISTILLED_TWO_STAGE_SIGMAS[DISTILLED_STAGE2_SIGMA_START..].to_vec()
        } else {
            STAGE2_SIGMAS[..args.stage2_steps + 1].to_vec()
        };
        let final_video_positions =
            make_video_positions(&video_patchifier, &final_video_shape, args.fps, &device)?;
        let final_video_denoise_mask = Tensor::ones((1, final_num_video_tokens), dtype, &device)?;
        println!(
            "stage2 refinement {} steps: sigma {:.4} -> {:.4}, video_tokens={}, audio_tokens={}, sampler={:?}",
            args.stage2_steps,
            stage2_sigmas[0],
            stage2_sigmas[stage2_sigmas.len() - 1],
            final_num_video_tokens,
            num_audio_tokens,
            effective_sampler
        );

        let stage2_model = load_transformer_model(
            &args,
            &transformer_cfg,
            dtype,
            &device,
            args.stage2_lora_scale,
            "stage2",
        )?;
        let (video_latent, audio_latent) = run_denoise_loop(
            &stage2_model,
            &pos,
            None,
            stage2_video_latent,
            stage2_audio_latent,
            &final_video_positions,
            audio_positions.as_ref(),
            &final_video_denoise_mask,
            audio_denoise_mask.as_ref(),
            &stage2_sigmas,
            effective_sampler,
            1.0,
            0.0,
            args.stg_scale,
            &stg_blocks,
            args.stg_mode,
            dtype,
            &device,
            "stage2",
        )?;
        drop(stage2_model);
        (video_latent, audio_latent, final_video_shape)
    } else {
        (stage_video_latent, stage_audio_latent, stage_video_shape)
    };
    stats("video_latent_tokens", &video_latent)?;

    println!("loading video VAE");
    // F32 decode avoids BF16 overflow in the high-resolution VAE path.
    let vae_dtype = DType::F32;
    let vae_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[args.checkpoint.as_path()], vae_dtype, &device)?
    };
    let video_decoder = VideoDecoder::new_with_roots(
        vae_cfg.latent_channels,
        vae_cfg.out_channels,
        &vae_cfg.decoder_blocks,
        vae_cfg.patch_size,
        vae_cfg.norm_layer == "pixel_norm",
        vae_cfg.causal_decoder,
        vae_cfg.timestep_conditioning,
        32,
        SpatialPaddingMode::parse(&vae_cfg.decoder_spatial_padding_mode)?,
        vae_vb.pp("vae").pp("per_channel_statistics"),
        vae_vb.pp("vae").pp("decoder"),
    )
    .context("VideoDecoder::new_with_roots")?;

    let video_latent_5d = video_patchifier
        .unpatchify(&video_latent, &decode_video_shape)?
        .to_dtype(vae_dtype)?;
    stats("video_latent_5d", &video_latent_5d)?;
    let latent_frames = video_latent_5d.dim(2)?;
    let decoded = if !args.no_vae_temporal_tiling && latent_frames > args.vae_temporal_tile_latents
    {
        println!(
            "decoding video with temporal VAE tiles: latent_frames={}, tile={}, overlap={}",
            latent_frames, args.vae_temporal_tile_latents, args.vae_temporal_overlap_latents
        );
        video_decoder
            .forward_temporal_tiled(
                &video_latent_5d,
                None,
                args.vae_temporal_tile_latents,
                args.vae_temporal_overlap_latents,
            )
            .context("VideoDecoder::forward_temporal_tiled")?
    } else {
        println!("decoding video");
        video_decoder
            .forward(&video_latent_5d, None)
            .context("VideoDecoder::forward")?
    };
    stats("decoded_video", &decoded)?;

    let (frames, height, width) = tensor_to_rgb_frames(&decoded)?;
    let video_output_path = if use_audio {
        sidecar_path(&args.output, "video", "mp4")
    } else {
        args.output.clone()
    };
    println!(
        "writing {} frames at {}x{} to {}",
        frames.len(),
        width,
        height,
        video_output_path.display()
    );
    write_video_ffmpeg(&frames, width, height, args.fps, &video_output_path)?;

    if let (Some(audio_latent), Some(audio_shape)) = (audio_latent.as_ref(), audio_shape.as_ref()) {
        println!("decoding audio");
        let audio_dtype = DType::F32;
        let audio_latent_4d = audio_patchifier
            .unpatchify(audio_latent, audio_shape)?
            .to_dtype(audio_dtype)?;
        stats("audio_latent_4d", &audio_latent_4d)?;

        let audio_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[args.checkpoint.as_path()], audio_dtype, &device)?
        };
        let audio_decoder = AudioDecoder::new_with_roots(
            128,
            2,
            &[1, 2, 4],
            2,
            8,
            CausalityAxis::Height,
            Some(64),
            audio_vb.pp("audio_vae").pp("per_channel_statistics"),
            audio_vb.pp("audio_vae").pp("decoder"),
        )
        .context("AudioDecoder::new_with_roots")?;
        let decoded_spectrogram = audio_decoder.forward(&audio_latent_4d)?;
        stats("decoded_spectrogram", &decoded_spectrogram)?;
        drop(audio_decoder);

        let vocoder_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[args.checkpoint.as_path()], audio_dtype, &device)?
        };
        let vocoder =
            VocoderWithBwe::new(vocoder_vb.pp("vocoder")).context("VocoderWithBwe::new")?;
        let audio_waveform = vocoder.forward(&decoded_spectrogram)?;
        stats("audio_waveform", &audio_waveform)?;
        let sample_rate = vocoder.output_sampling_rate();
        drop(vocoder);

        let wav_path = sidecar_path(&args.output, "audio", "wav");
        write_wav(&audio_waveform.squeeze(0)?, sample_rate, &wav_path)?;
        println!(
            "muxing {} + {}",
            video_output_path.display(),
            wav_path.display()
        );
        mux_av(&video_output_path, &wav_path, &args.output)?;
    }

    let size = std::fs::metadata(&args.output)?.len();
    println!(
        "DONE wrote {} ({} bytes) in {:.1}s",
        args.output.display(),
        size,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
