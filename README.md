# Maor

Rust inference for LTX-2.3 audio-video generation.

This repository implements the LTX-2.3 inference stack in Rust:

```text
prompt
  -> Gemma3 text encoder
  -> LTX-2.3 dual text projection/connectors
  -> 48-layer audio-video DiT transformer
  -> video VAE decoder
  -> audio VAE decoder
  -> AMP vocoder + BWE
  -> ffmpeg MP4 mux
```

Model execution is Rust/Candle. `ffmpeg` is used only for video encoding and
final audio/video muxing.

## Status

The main supported binary is `maor`.

Validated inference target:

- Model: LTX-2.3 22B single-file safetensors checkpoint.
- Output: `960x544`, `121` frames, `24 fps`.
- HQ output: two-stage `1024x576`, `121` frames, `24 fps`.
- Audio: stereo `48 kHz`, generated through audio VAE + AMP vocoder + BWE.
- Hardware class: NVIDIA data-center GPU with enough VRAM for the 22B model.

The repository is focused on LTX-2.3 inference.

## Showcase

The clips below were generated with the Rust/Candle HQ two-stage path at
`24 fps`. They intentionally cover durations from 5 to 20 seconds; longer
clips use lower spatial resolutions to keep transformer token counts practical.

| Clip | Duration | Resolution | Demonstrates | Video |
| --- | --- | --- | --- | --- |
| Wildlife motion | 5.04 s | 1024x576 | Animal motion, fur, snowy outdoor lighting | <video src="https://github.com/user-attachments/assets/c3df0825-6761-460a-b481-267952acdfad" controls width="320"></video> |
| Macro rain flower | 6.04 s | 1024x576 | Macro detail, droplets, shallow depth of field | <video src="https://github.com/user-attachments/assets/3124e1a2-31a7-466f-955a-bc7a1ea22431" controls width="320"></video> |
| Drone coastline | 8.04 s | 896x512 | Landscape scale, waves, aerial motion | <video src="https://github.com/user-attachments/assets/02861957-bfe4-4ea8-86f6-a35c64f38bfb" controls width="320"></video> |
| Product turntable | 9.04 s | 768x448 | Reflective materials, studio lighting, object rotation | <video src="https://github.com/user-attachments/assets/3ef16937-8696-4b28-937b-9b8406664734" controls width="320"></video> |
| Steam food closeup | 10.04 s | 768x448 | Food texture, steam, warm indoor lighting | <video src="https://github.com/user-attachments/assets/aaed526f-c04a-4442-9867-eeb7cdc4d8f8" controls width="320"></video> |
| Robot materials | 12.04 s | 640x384 | Metallic surfaces, small character motion, workshop scene | <video src="https://github.com/user-attachments/assets/332b5167-41f0-4fb8-97b4-6a9e4e6fec70" controls width="320"></video> |
| Underwater reef | 14.04 s | 640x384 | Water volume, particles, marine scene motion | <video src="https://github.com/user-attachments/assets/1c54ec8e-a2c1-4c42-9af9-41ad7624f01a" controls width="320"></video> |
| Fabric motion | 16.04 s | 512x320 | Cloth dynamics, slow-motion folds, studio lighting | <video src="https://github.com/user-attachments/assets/830eb306-13fc-4eea-9e2e-30e87ea4248d" controls width="320"></video> |

## Weights

Weights are not included in this repository.

No weight conversion is required for the supported LTX-2.3 path:

- Pass the original LTX-2.3 single-file `.safetensors` checkpoint directly with
  `--checkpoint`.
- Pass the Gemma3 safetensors directory directly with `--gemma-dir`.
- Pass `tokenizer.json` directly with `--tokenizer`.
- For two-stage HQ generation, pass the original spatial upscaler checkpoint
  with `--spatial-upscaler`.
- Optionally pass the distilled LoRA checkpoint with `--distilled-lora`; it is
  merged into diffusion weights at runtime.

The checkpoint must contain the LTX-2.3 layout used by this code: transformer,
video VAE, audio VAE, vocoder, and metadata config in one safetensors file.

## Requirements

- Rust toolchain with Cargo.
- CUDA-capable NVIDIA GPU for practical inference.
- CUDA toolkit compatible with Candle CUDA builds.
- `ffmpeg` and `ffprobe` on `PATH`.
- LTX-2.3 checkpoint, Gemma3 text model shards, and tokenizer.

## Install From Source

```bash
export PATH=/usr/local/cuda-12.8/bin:$HOME/.cargo/bin:$PATH
export CUDA_COMPUTE_CAP=90
cargo build --release --features cuda --bin maor
```

Set `CUDA_COMPUTE_CAP` for your GPU architecture, for example `86`, `89`, or
`90`. The executable is written to `target/release/maor`.

On Windows, run CUDA builds from a Visual Studio Developer PowerShell or another
shell where the MSVC compiler `cl.exe` is on `PATH`; `nvcc` requires it.

## Run LTX-2.3 Inference

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

./target/release/maor \
  --checkpoint /path/to/ltx-2.3-22b-dev.safetensors \
  --gemma-dir /path/to/gemma-3-12b-it-qat-q4_0-unquantized \
  --tokenizer /path/to/tokenizer.json \
  --prompt "a red fox walking through a snowy pine forest" \
  --output outputs/maor_fox.mp4 \
  --width 960 \
  --height 544 \
  --frames 121 \
  --fps 24 \
  --steps 15 \
  --cfg-scale 3.0 \
  --video-rescale-scale 0.9 \
  --seed 42 \
  --mode av \
  --sampler euler \
  --stg-scale 1.0 \
  --stg-block 28 \
  --stg-mode video
```

For HQ two-stage video generation:

```bash
./target/release/maor \
  --checkpoint /path/to/ltx-2.3-22b-dev.safetensors \
  --gemma-dir /path/to/gemma-3-12b-it-qat-q4_0-unquantized \
  --tokenizer /path/to/tokenizer.json \
  --prompt "a red fox walking through a snowy pine forest" \
  --output outputs/maor_fox_hq.mp4 \
  --width 1024 \
  --height 576 \
  --frames 121 \
  --fps 24 \
  --steps 8 \
  --cfg-scale 3.0 \
  --video-rescale-scale 0.9 \
  --seed 42 \
  --mode video \
  --sampler euler \
  --stg-scale 0.0 \
  --two-stage \
  --stage2-steps 3 \
  --spatial-upscaler /path/to/ltx-2.3-spatial-upscaler-x2.safetensors \
  --distilled-lora /path/to/ltx-2.3-22b-distilled-lora.safetensors \
  --stage1-lora-scale 0.0 \
  --stage2-lora-scale 1.0
```

For a smaller smoke run:

```bash
./target/release/maor \
  --checkpoint /path/to/ltx-2.3-22b-dev.safetensors \
  --gemma-dir /path/to/gemma-3-12b-it-qat-q4_0-unquantized \
  --tokenizer /path/to/tokenizer.json \
  --prompt "a red fox walking through a snowy pine forest" \
  --output outputs/smoke.mp4 \
  --width 256 \
  --height 256 \
  --frames 9 \
  --steps 4 \
  --mode av \
  --sampler euler
```

## Main CLI Options

| Flag | Default | Description |
| --- | --- | --- |
| `--checkpoint` | required | LTX-2.3 single-file safetensors checkpoint |
| `--gemma-dir` | required | Directory containing Gemma3 safetensors shards |
| `--tokenizer` | required | Gemma tokenizer JSON |
| `--prompt` | required | Positive generation prompt |
| `--negative-prompt` | built-in default | Optional negative prompt for CFG |
| `--output` | `outputs/ner.mp4` | Output MP4 path |
| `--mode` | `video` | `video` or `av` |
| `--sampler` | `res2s` | `res2s` or `euler` |
| `--width` | `960` | Output width, divisible by 32 |
| `--height` | `544` | Output height, divisible by 32 |
| `--frames` | `121` | Frame count, expected as `1 + 8*k` |
| `--fps` | `24` | Output video FPS |
| `--steps` | `15` | Denoising steps |
| `--cfg-scale` | `3.0` | Classifier-free guidance scale |
| `--video-rescale-scale` | `0.9` | CFG rescale for video latent |
| `--stg-scale` | `1.0` | Spatio-temporal guidance scale; `0` disables |
| `--stg-block` | `28` | Transformer block used for STG perturbation |
| `--stg-mode` | `video` | `video`, `audio`, or `both` |
| `--two-stage` | false | Generate at half spatial size, upscale latents, then refine |
| `--spatial-upscaler` | optional | LTX-2.3 latent x2 spatial upscaler checkpoint |
| `--stage2-steps` | `3` | Number of second-stage refinement steps |
| `--distilled-lora` | optional | Runtime-merged distilled LoRA checkpoint |
| `--stage1-lora-scale` | `0.0` | First-stage LoRA multiplier |
| `--stage2-lora-scale` | `1.0` | Second-stage LoRA multiplier |
| `--f32` | false | Use f32 instead of bf16 |
| `--no-vae-temporal-tiling` | false | Disable temporal tiled VAE decode |

When `--two-stage` and `--distilled-lora` are used together, the binary uses
the exact distilled two-stage sigma chain and requires `--steps 8` plus
`--stage2-steps 3`.

## TODO

- Rust-native training pipeline.

## Project Structure

```text
crates/
  maor-core/           Core config, latent shapes, patchification, statistics
  maor-nn/             Attention, RoPE, AdaLN, convs, projections
  maor-transformer/    LTX-2.3 audio-video DiT transformer
  maor-video-vae/      LTX-2.3 video VAE decoder
  maor-audio-vae/      LTX-2.3 audio VAE, AMP vocoder, BWE vocoder
  maor-text-encoder/   Gemma3 text encoder and AV connectors
  maor-scheduler/      Sigma schedule, Euler, RES_2S, guidance
  maor-pipeline/       CLI inference binary
```

## Troubleshooting

**CUDA build cannot find nvcc or CUDA headers** - ensure the CUDA toolkit bin
directory is on `PATH` and set `CUDA_COMPUTE_CAP`.

**Runtime cannot load CUDA libraries** - set `LD_LIBRARY_PATH` to include the CUDA
`lib64` directory.

**ffmpeg not found** - install ffmpeg and make sure it is on `PATH`.

**Out of memory** - reduce `--width`, `--height`, `--frames`, or run video-only
mode first.

**Output is video-only** - use `--mode av`; the default mode is `video`.

## License

This is an unofficial implementation. Model weights are subject to their own
licenses and must be obtained separately from the original providers.
