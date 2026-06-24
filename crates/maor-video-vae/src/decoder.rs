use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

use maor_core::statistics::PerChannelStatistics;
use maor_nn::activation::silu;
use maor_nn::conv3d::{CausalConv3d, SpatialPaddingMode};
use maor_nn::timestep_embedding::PixArtAlphaCombinedTimestepSizeEmbeddings;

use crate::resnet::{NormLayer, ResnetBlock3D, UNetMidBlock3D};
use crate::upsample::DepthToSpaceUpsample;

type TraceTensors = Vec<(String, Tensor)>;

/// A single decoder up-block (polymorphic).
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum DecoderBlock {
    ResX(UNetMidBlock3D),
    ResXY(ResnetBlock3D),
    CompressTime(DepthToSpaceUpsample),
    CompressSpace(DepthToSpaceUpsample),
    CompressAll(DepthToSpaceUpsample),
}

impl DecoderBlock {
    pub fn forward(&self, x: &Tensor, causal: bool, timestep: Option<&Tensor>) -> Result<Tensor> {
        match self {
            Self::ResX(block) => block.forward(x, causal, timestep),
            Self::ResXY(block) => block.forward(x, causal, None),
            Self::CompressTime(block) => block.forward(x, causal),
            Self::CompressSpace(block) => block.forward(x, causal),
            Self::CompressAll(block) => block.forward(x, causal),
        }
    }

    /// Output channels after this block (for tracking feature channel width).
    pub fn out_channels(block_name: &str, in_channels: usize, config: &serde_json::Value) -> usize {
        match block_name {
            "res_x_y" | "compress_time" | "compress_space" | "compress_all" => {
                in_channels / Self::channel_multiplier(block_name, config)
            }
            _ => in_channels,
        }
    }

    /// Inverse channel multiplier used to infer the decoder's initial feature width
    /// from an encoder-order block list.
    pub fn inverse_channel_multiplier(block_name: &str, config: &serde_json::Value) -> usize {
        match block_name {
            "res_x_y" | "compress_time" | "compress_space" | "compress_all" => {
                Self::channel_multiplier(block_name, config)
            }
            _ => 1,
        }
    }

    fn channel_multiplier(block_name: &str, config: &serde_json::Value) -> usize {
        let default = if block_name == "res_x_y" { 2 } else { 1 };
        config
            .get("multiplier")
            .and_then(|v| v.as_u64())
            .unwrap_or(default) as usize
    }
}

/// Video VAE Decoder.
///
/// Decodes latent representation (B, 128, F', H', W') into video frames (B, 3, F, H, W).
/// Architecture: denormalize → conv_in → up_blocks → norm → [timestep modulation] → SiLU → conv_out → unpatchify
#[derive(Debug)]
pub struct VideoDecoder {
    per_channel_statistics: PerChannelStatistics,
    conv_in: CausalConv3d,
    up_blocks: Vec<DecoderBlock>,
    conv_norm_out: NormLayer,
    conv_out: CausalConv3d,
    causal: bool,
    patch_size: usize,
    // Timestep conditioning
    timestep_conditioning: bool,
    timestep_scale_multiplier: Option<Tensor>,
    last_time_embedder: Option<PixArtAlphaCombinedTimestepSizeEmbeddings>,
    last_scale_shift_table: Option<Tensor>,
    decode_noise_scale: f64,
    decode_timestep: f64,
}

impl VideoDecoder {
    /// Construct the decoder from config and safetensors weights.
    ///
    /// `decoder_blocks`: list of (block_name, block_config) tuples, in **encoder order**
    /// (will be reversed internally to create the decoder).
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        decoder_blocks: &[(String, serde_json::Value)],
        patch_size: usize,
        use_pixel_norm: bool,
        causal: bool,
        timestep_conditioning: bool,
        norm_num_groups: usize,
        spatial_padding_mode: SpatialPaddingMode,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_with_roots(
            in_channels,
            out_channels,
            decoder_blocks,
            patch_size,
            use_pixel_norm,
            causal,
            timestep_conditioning,
            norm_num_groups,
            spatial_padding_mode,
            vb.pp("per_channel_statistics"),
            vb,
        )
    }

    /// Construct the decoder when statistics and decoder modules live under
    /// different checkpoint prefixes:
    /// `vae.per_channel_statistics.*` and `vae.decoder.*`.
    pub fn new_with_roots(
        in_channels: usize,
        out_channels: usize,
        decoder_blocks: &[(String, serde_json::Value)],
        patch_size: usize,
        use_pixel_norm: bool,
        causal: bool,
        timestep_conditioning: bool,
        norm_num_groups: usize,
        spatial_padding_mode: SpatialPaddingMode,
        stats_vb: VarBuilder,
        decoder_vb: VarBuilder,
    ) -> Result<Self> {
        let per_channel_statistics = PerChannelStatistics::from_vb(in_channels, stats_vb)?;

        // Effective output channels includes patch_size² spatial expansion
        let effective_out_channels = out_channels * patch_size * patch_size;

        // Compute initial feature_channels by going through blocks in reverse
        // (reversed encoder order = forward decoder order)
        let reversed_blocks: Vec<_> = decoder_blocks.iter().rev().collect();
        let mut feature_channels = in_channels;
        for (name, config) in &reversed_blocks {
            feature_channels *= DecoderBlock::inverse_channel_multiplier(name, config);
        }

        let conv_in = CausalConv3d::new_with_padding_mode(
            in_channels,
            feature_channels,
            3,
            (1, 1, 1),
            1,
            true,
            spatial_padding_mode,
            decoder_vb.pp("conv_in"),
        )?;

        // Build decoder blocks (in reversed encoder order)
        let mut up_blocks = Vec::new();
        let blocks_vb = decoder_vb.pp("up_blocks");

        for (idx, (block_name, block_config)) in reversed_blocks.iter().enumerate() {
            let block = make_decoder_block(
                block_name,
                block_config,
                feature_channels,
                use_pixel_norm,
                timestep_conditioning,
                norm_num_groups,
                spatial_padding_mode,
                blocks_vb.pp(idx),
            )?;

            feature_channels =
                DecoderBlock::out_channels(block_name, feature_channels, block_config);
            up_blocks.push(block);
        }

        let conv_norm_out = if use_pixel_norm {
            NormLayer::new_pixel_norm()
        } else {
            NormLayer::new_group_norm(
                norm_num_groups,
                feature_channels,
                1e-6,
                decoder_vb.pp("conv_norm_out"),
            )?
        };

        let conv_out = CausalConv3d::new_with_padding_mode(
            feature_channels,
            effective_out_channels,
            3,
            (1, 1, 1),
            1,
            true,
            spatial_padding_mode,
            decoder_vb.pp("conv_out"),
        )?;

        let (timestep_scale_multiplier, last_time_embedder, last_scale_shift_table) =
            if timestep_conditioning {
                let scale = Some(decoder_vb.get(&[1], "timestep_scale_multiplier")?);
                let embedder = Some(PixArtAlphaCombinedTimestepSizeEmbeddings::new(
                    feature_channels * 2,
                    decoder_vb.pp("last_time_embedder"),
                )?);
                let table = Some(decoder_vb.get(&[2, feature_channels], "last_scale_shift_table")?);
                (scale, embedder, table)
            } else {
                (None, None, None)
            };

        Ok(Self {
            per_channel_statistics,
            conv_in,
            up_blocks,
            conv_norm_out,
            conv_out,
            causal,
            patch_size,
            timestep_conditioning,
            timestep_scale_multiplier,
            last_time_embedder,
            last_scale_shift_table,
            // Default LTX-2.3 decoder settings:
            // Small noise injection during decode for perceptual quality
            decode_noise_scale: 0.025,
            // Default timestep for decoder conditioning (matches training)
            decode_timestep: 0.05,
        })
    }

    /// Decode latent to video frames.
    ///
    /// Input: (B, 128, F', H', W') latent tensor.
    /// Output: (B, 3, F, H, W) decoded video where F=8(F'-1)+1, H=32H', W=32W'.
    pub fn forward(&self, sample: &Tensor, timestep: Option<&Tensor>) -> Result<Tensor> {
        Ok(self.forward_impl(sample, timestep, false)?.0)
    }

    /// Decode long videos in overlapping latent-time chunks.
    ///
    /// The plain full decode path is numerically fragile on long LTX-2.3 clips
    /// (for example 16 latent frames -> 121 video frames). Each chunk is decoded
    /// with overlapping latent context, then the already-covered video frames are
    /// dropped from subsequent chunks before concatenation.
    pub fn forward_temporal_tiled(
        &self,
        sample: &Tensor,
        timestep: Option<&Tensor>,
        max_latent_frames: usize,
        overlap_latent_frames: usize,
    ) -> Result<Tensor> {
        let (_, _, latent_frames, _, _) = sample.dims5()?;
        if latent_frames <= max_latent_frames {
            return self.forward(sample, timestep);
        }
        if max_latent_frames < 2 {
            candle_core::bail!("max_latent_frames must be at least 2");
        }
        if overlap_latent_frames == 0 || overlap_latent_frames >= max_latent_frames {
            candle_core::bail!(
                "overlap_latent_frames must be in 1..max_latent_frames, got {overlap_latent_frames}"
            );
        }

        let step = max_latent_frames - overlap_latent_frames;
        let mut start = 0usize;
        let mut chunks = Vec::new();

        loop {
            let len = (latent_frames - start).min(max_latent_frames);
            let latent_chunk = sample.narrow(2, start, len)?;
            let decoded_chunk = self.forward(&latent_chunk, timestep)?;

            let decoded_chunk = if start == 0 {
                decoded_chunk
            } else {
                let chunk_video_frames = decoded_chunk.dim(2)?;
                let temporal_scale = if len > 1 {
                    (chunk_video_frames - 1) / (len - 1)
                } else {
                    1
                };
                let skip_frames = (overlap_latent_frames - 1) * temporal_scale + 1;
                if skip_frames >= chunk_video_frames {
                    candle_core::bail!(
                        "temporal tile overlap consumes the whole decoded chunk: skip={skip_frames}, frames={chunk_video_frames}"
                    );
                }
                decoded_chunk.narrow(2, skip_frames, chunk_video_frames - skip_frames)?
            };
            chunks.push(decoded_chunk);

            if start + len >= latent_frames {
                break;
            }
            start += step;
        }

        let chunk_refs: Vec<&Tensor> = chunks.iter().collect();
        Tensor::cat(&chunk_refs, 2)
    }

    /// Decode latent and return named intermediate tensors for diagnostics.
    pub fn forward_with_trace(
        &self,
        sample: &Tensor,
        timestep: Option<&Tensor>,
    ) -> Result<(Tensor, TraceTensors)> {
        let (decoded, trace) = self.forward_impl(sample, timestep, true)?;
        Ok((decoded, trace.unwrap_or_default()))
    }

    fn forward_impl(
        &self,
        sample: &Tensor,
        timestep: Option<&Tensor>,
        trace_enabled: bool,
    ) -> Result<(Tensor, Option<TraceTensors>)> {
        let mut trace = if trace_enabled {
            Some(Vec::new())
        } else {
            None
        };
        let batch_size = sample.dims()[0];

        // Add noise if timestep conditioning
        let sample = if self.timestep_conditioning {
            let noise = sample.randn_like(0.0, 1.0)?;
            let noise = (noise * self.decode_noise_scale)?;
            (noise + (sample * (1.0 - self.decode_noise_scale))?)?
        } else {
            sample.clone()
        };

        // Denormalize latents
        let sample = self.per_channel_statistics.denormalize(&sample)?;
        record_trace(&mut trace, "denormalized", &sample);

        // Default timestep if not provided
        let default_ts;
        let timestep = if timestep.is_none() && self.timestep_conditioning {
            default_ts = Tensor::full(self.decode_timestep as f32, (batch_size,), sample.device())?;
            Some(&default_ts)
        } else {
            timestep
        };

        let mut sample = self.conv_in.forward(&sample, self.causal)?;
        record_trace(&mut trace, "conv_in", &sample);

        // Scale timestep
        let scaled_timestep = if self.timestep_conditioning {
            if let (Some(ts), Some(scale)) = (timestep, &self.timestep_scale_multiplier) {
                let scale = scale.to_dtype(ts.dtype())?;
                Some(ts.broadcast_mul(&scale)?)
            } else {
                None
            }
        } else {
            None
        };

        // Run through decoder blocks
        for (idx, block) in self.up_blocks.iter().enumerate() {
            sample = block.forward(&sample, self.causal, scaled_timestep.as_ref())?;
            record_trace(&mut trace, &format!("up_blocks.{idx}"), &sample);
        }

        // Output normalization
        sample = self.conv_norm_out.forward(&sample)?;
        record_trace(&mut trace, "conv_norm_out", &sample);

        // Final timestep modulation
        if self.timestep_conditioning {
            if let (Some(embedder), Some(table), Some(ts)) = (
                &self.last_time_embedder,
                &self.last_scale_shift_table,
                &scaled_timestep,
            ) {
                let emb = embedder.forward(&ts.flatten_all()?)?;
                let emb_dim = emb.dim(emb.rank() - 1)?;
                let emb = emb.reshape(&[batch_size, emb_dim, 1, 1, 1])?;

                // table: (2, feat_ch) + emb reshaped to (B, 2, feat_ch, 1, 1, 1)
                let feat_ch = table.dim(1)?;
                let table_expanded = table
                    .unsqueeze(0)?
                    .reshape((1, 2, feat_ch, 1, 1, 1))?
                    .to_dtype(sample.dtype())?;
                let emb_reshaped = emb.reshape(&[batch_size, 2, feat_ch, 1, 1, 1])?;
                let ada_values = (table_expanded + emb_reshaped)?;

                let shift = ada_values.narrow(1, 0, 1)?.squeeze(1)?;
                let scale = ada_values.narrow(1, 1, 1)?.squeeze(1)?;
                sample = (sample.broadcast_mul(&(scale + 1.0)?)? + shift)?;
            }
        }

        sample = silu(&sample)?;
        record_trace(&mut trace, "conv_act", &sample);
        sample = self.conv_out.forward(&sample, self.causal)?;
        record_trace(&mut trace, "conv_out", &sample);

        // Unpatchify: (B, C*p², F, H', W') → (B, C, F, H'*p, W'*p)
        let sample = unpatchify_spatial(&sample, self.patch_size)?;
        record_trace(&mut trace, "decoded", &sample);
        Ok((sample, trace))
    }
}

fn record_trace(trace: &mut Option<TraceTensors>, name: &str, tensor: &Tensor) {
    if let Some(trace) = trace {
        trace.push((name.to_string(), tensor.clone()));
    }
}

/// Build a single decoder block from its name and config.
fn make_decoder_block(
    block_name: &str,
    block_config: &serde_json::Value,
    in_channels: usize,
    use_pixel_norm: bool,
    timestep_conditioning: bool,
    norm_num_groups: usize,
    spatial_padding_mode: SpatialPaddingMode,
    vb: VarBuilder,
) -> Result<DecoderBlock> {
    match block_name {
        "res_x" => {
            let num_layers = block_config
                .get("num_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize;
            Ok(DecoderBlock::ResX(UNetMidBlock3D::new(
                in_channels,
                num_layers,
                use_pixel_norm,
                timestep_conditioning,
                norm_num_groups,
                1e-6,
                spatial_padding_mode,
                vb,
            )?))
        }
        "attn_res_x" => {
            // For now, same as res_x (attention in mid-block not yet implemented)
            let num_layers = block_config
                .get("num_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize;
            Ok(DecoderBlock::ResX(UNetMidBlock3D::new(
                in_channels,
                num_layers,
                use_pixel_norm,
                timestep_conditioning,
                norm_num_groups,
                1e-6,
                spatial_padding_mode,
                vb,
            )?))
        }
        "res_x_y" => {
            let mult = block_config
                .get("multiplier")
                .and_then(|v| v.as_u64())
                .unwrap_or(2) as usize;
            let out_channels = in_channels / mult;
            Ok(DecoderBlock::ResXY(ResnetBlock3D::new(
                in_channels,
                out_channels,
                use_pixel_norm,
                false, // res_x_y never uses timestep conditioning
                norm_num_groups,
                1e-6,
                spatial_padding_mode,
                vb,
            )?))
        }
        "compress_time" => {
            let mult = block_config
                .get("multiplier")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize;
            Ok(DecoderBlock::CompressTime(DepthToSpaceUpsample::new(
                in_channels,
                (2, 1, 1),
                mult,
                spatial_padding_mode,
                vb,
            )?))
        }
        "compress_space" => {
            let mult = block_config
                .get("multiplier")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize;
            Ok(DecoderBlock::CompressSpace(DepthToSpaceUpsample::new(
                in_channels,
                (1, 2, 2),
                mult,
                spatial_padding_mode,
                vb,
            )?))
        }
        "compress_all" => {
            let mult = block_config
                .get("multiplier")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize;
            Ok(DecoderBlock::CompressAll(DepthToSpaceUpsample::new(
                in_channels,
                (2, 2, 2),
                mult,
                spatial_padding_mode,
                vb,
            )?))
        }
        _ => candle_core::bail!("unknown decoder block: {block_name}"),
    }
}

/// Spatial unpatchify: (B, C*p², F, H', W') → (B, C, F, H'*p, W'*p)
///
/// With patch_size_t=1, this is: "b (c r q) f h w -> b c f (h q) (w r)"
fn unpatchify_spatial(x: &Tensor, patch_size: usize) -> Result<Tensor> {
    if patch_size == 1 {
        return Ok(x.clone());
    }

    let (b, c_packed, f, h, w) = x.dims5()?;
    let p = patch_size;
    let c = c_packed / (p * p);

    // Unpack spatial patches:
    //   "b (c r q) f h w -> b c f (h q) (w r)"
    // The first packed patch axis is width (r), the second is height (q).
    let x = x.reshape(&[b, c, p, p, f, h, w])?;

    // Permute to (B, C, F, H, q, W, r).
    let x = x.permute([0usize, 1, 4, 5, 3, 6, 2].as_slice())?;

    // Reshape to (B, C, F, H*p, W*p)
    x.contiguous()?.reshape((b, c, f, h * p, w * p))
}
