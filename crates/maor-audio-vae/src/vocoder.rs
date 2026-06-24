use candle_core::{DType, Result, Tensor};
use candle_nn::{Conv1dConfig, ConvTranspose1dConfig, Module, VarBuilder};

use crate::causal_conv2d::leaky_relu;
use crate::resnet::ResBlock1;

const LRELU_SLOPE: f64 = 0.1;

/// HiFi-GAN vocoder for converting mel spectrograms to audio waveforms.
///
/// Architecture: conv_pre → 5×(LeakyReLU → ConvTranspose1d → mean(3×ResBlock1)) → conv_post → tanh.
///
/// Default LTX-2.3 config: upsample_rates=[6,5,2,2,2], kernel_sizes=[16,15,8,4,4],
/// resblock_kernel_sizes=[3,7,11], dilations=[[1,3,5],...], stereo=true.
#[derive(Debug)]
pub struct Vocoder {
    conv_pre: candle_nn::Conv1d,
    ups: Vec<candle_nn::ConvTranspose1d>,
    resblocks: Vec<VocoderResBlock>,
    act_post: Option<Activation1d>,
    conv_post: candle_nn::Conv1d,
    num_upsamples: usize,
    num_kernels: usize,
    apply_final_activation: bool,
    use_tanh_at_final: bool,
    is_amp: bool,
}

impl Vocoder {
    pub fn new(
        upsample_rates: &[usize],
        upsample_kernel_sizes: &[usize],
        resblock_kernel_sizes: &[usize],
        resblock_dilation_sizes: &[Vec<usize>],
        upsample_initial_channel: usize,
        stereo: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_with_options(
            upsample_rates,
            upsample_kernel_sizes,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_initial_channel,
            stereo,
            false,
            true,
            true,
            true,
            vb,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_options(
        upsample_rates: &[usize],
        upsample_kernel_sizes: &[usize],
        resblock_kernel_sizes: &[usize],
        resblock_dilation_sizes: &[Vec<usize>],
        upsample_initial_channel: usize,
        stereo: bool,
        amp_resblock: bool,
        use_tanh_at_final: bool,
        apply_final_activation: bool,
        use_bias_at_final: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let num_upsamples = upsample_rates.len();
        let num_kernels = resblock_kernel_sizes.len();
        let in_channels = if stereo { 128 } else { 64 };

        let conv_pre_cfg = Conv1dConfig {
            padding: 3,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let conv_pre = candle_nn::conv1d(
            in_channels,
            upsample_initial_channel,
            7,
            conv_pre_cfg,
            vb.pp("conv_pre"),
        )?;

        let mut ups = Vec::new();
        let ups_vb = vb.pp("ups");
        for (i, (&rate, &kernel_size)) in upsample_rates
            .iter()
            .zip(upsample_kernel_sizes.iter())
            .enumerate()
        {
            let in_ch = upsample_initial_channel / (1 << i);
            let out_ch = upsample_initial_channel / (1 << (i + 1));
            let padding = (kernel_size - rate) / 2;
            let cfg = ConvTranspose1dConfig {
                padding,
                output_padding: 0,
                stride: rate,
                dilation: 1,
                groups: 1,
            };
            ups.push(candle_nn::conv_transpose1d(
                in_ch,
                out_ch,
                kernel_size,
                cfg,
                ups_vb.pp(i),
            )?);
        }

        let mut resblocks = Vec::new();
        let rb_vb = vb.pp("resblocks");
        for i in 0..num_upsamples {
            let ch = upsample_initial_channel / (1 << (i + 1));
            for (j, (&kernel_size, dilations)) in resblock_kernel_sizes
                .iter()
                .zip(resblock_dilation_sizes.iter())
                .enumerate()
            {
                let idx = i * num_kernels + j;
                let block = if amp_resblock {
                    VocoderResBlock::Amp1(AmpBlock1::new(
                        ch,
                        kernel_size,
                        dilations,
                        rb_vb.pp(idx),
                    )?)
                } else {
                    VocoderResBlock::Res1(ResBlock1::new(
                        ch,
                        kernel_size,
                        dilations,
                        rb_vb.pp(idx),
                    )?)
                };
                resblocks.push(block);
            }
        }

        let out_channels = if stereo { 2 } else { 1 };
        let final_channels = upsample_initial_channel / (1 << num_upsamples);
        let conv_post_cfg = Conv1dConfig {
            padding: 3,
            stride: 1,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let conv_post = if use_bias_at_final {
            candle_nn::conv1d(
                final_channels,
                out_channels,
                7,
                conv_post_cfg,
                vb.pp("conv_post"),
            )?
        } else {
            candle_nn::conv1d_no_bias(
                final_channels,
                out_channels,
                7,
                conv_post_cfg,
                vb.pp("conv_post"),
            )?
        };
        let act_post = if amp_resblock {
            Some(Activation1d::new(final_channels, vb.pp("act_post"))?)
        } else {
            None
        };

        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            act_post,
            conv_post,
            num_upsamples,
            num_kernels,
            apply_final_activation,
            use_tanh_at_final,
            is_amp: amp_resblock,
        })
    }

    /// Convert mel spectrogram to audio waveform.
    ///
    /// Input: (B, 2, time, mel_bins) stereo spectrogram.
    /// Output: (B, 2, audio_length) waveform with tanh activation.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // (B, channels, time, mel_bins) → (B, channels, mel_bins, time)
        let x = x.transpose(2, 3)?;

        // Stereo: merge channels with mel bins → (B, 2*mel, time)
        let x = if x.rank() == 4 {
            let (b, s, c, t) = x.dims4()?;
            x.reshape((b, s * c, t))?
        } else {
            x
        };

        let mut x = self.conv_pre.forward(&x)?;

        for i in 0..self.num_upsamples {
            if !self.is_amp {
                x = leaky_relu(&x, LRELU_SLOPE)?;
            }
            x = self.ups[i].forward(&x)?;

            // Average outputs of all resblocks at this stage
            let start = i * self.num_kernels;
            let end = start + self.num_kernels;
            let mut sum = self.resblocks[start].forward(&x)?;
            for idx in (start + 1)..end {
                sum = (sum + self.resblocks[idx].forward(&x)?)?;
            }
            x = (sum / self.num_kernels as f64)?;
        }

        // Final activation → conv_post → optional final clamp/tanh.
        x = if let Some(act_post) = &self.act_post {
            act_post.forward(&x)?
        } else {
            leaky_relu(&x, 0.01)?
        };
        x = self.conv_post.forward(&x)?;
        if self.apply_final_activation {
            if self.use_tanh_at_final {
                x.tanh()
            } else {
                x.clamp(-1.0, 1.0)
            }
        } else {
            Ok(x)
        }
    }
}

/// LTX-2.3 vocoder with bandwidth extension.
///
/// The base vocoder produces 16 kHz stereo audio. BWE computes a causal log-mel
/// spectrogram from that waveform, predicts a high-rate residual with a second
/// vocoder, and adds a 3x sinc-resampled skip connection to produce 48 kHz.
#[derive(Debug)]
pub struct VocoderWithBwe {
    vocoder: Vocoder,
    bwe_generator: Vocoder,
    mel_stft: MelStft,
    resampler: UpSample1d,
    input_sampling_rate: usize,
    output_sampling_rate: usize,
    hop_length: usize,
}

impl VocoderWithBwe {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let vocoder = Vocoder::new_with_options(
            &[5, 2, 2, 2, 2, 2],
            &[11, 4, 4, 4, 4, 4],
            &[3, 7, 11],
            &[vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            1536,
            true,
            true,
            false,
            true,
            false,
            vb.pp("vocoder"),
        )?;
        let bwe_generator = Vocoder::new_with_options(
            &[6, 5, 2, 2, 2],
            &[12, 11, 4, 4, 4],
            &[3, 7, 11],
            &[vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            512,
            true,
            true,
            false,
            false,
            false,
            vb.pp("bwe_generator"),
        )?;
        let mel_stft = MelStft::new(512, 80, 512, vb.pp("mel_stft"))?;
        Ok(Self {
            vocoder,
            bwe_generator,
            mel_stft,
            resampler: UpSample1d::new_hann(3),
            input_sampling_rate: 16000,
            output_sampling_rate: 48000,
            hop_length: 80,
        })
    }

    pub fn output_sampling_rate(&self) -> u32 {
        self.output_sampling_rate as u32
    }

    pub fn forward(&self, mel_spec: &Tensor) -> Result<Tensor> {
        let mut x = self.vocoder.forward(mel_spec)?;
        let (_, _, length_low_rate) = x.dims3()?;
        let output_length = length_low_rate * self.output_sampling_rate / self.input_sampling_rate;

        let remainder = length_low_rate % self.hop_length;
        if remainder != 0 {
            x = x.pad_with_zeros(2, 0, self.hop_length - remainder)?;
        }

        let mel = self.compute_mel(&x)?;
        let mel_for_bwe = mel.transpose(2, 3)?;
        let residual = self.bwe_generator.forward(&mel_for_bwe)?;
        let skip = self.resampler.forward(&x)?;
        if residual.dims() != skip.dims() {
            candle_core::bail!(
                "BWE residual/skip shape mismatch: {:?} != {:?}",
                residual.dims(),
                skip.dims()
            );
        }
        let out = (residual + skip)?.clamp(-1.0, 1.0)?;
        out.narrow(2, 0, output_length)
    }

    fn compute_mel(&self, audio: &Tensor) -> Result<Tensor> {
        let (batch, channels, samples) = audio.dims3()?;
        let flat = audio.reshape((batch * channels, samples))?;
        let mel = self.mel_stft.mel_spectrogram(&flat)?;
        let (_, n_mels, frames) = mel.dims3()?;
        mel.reshape((batch, channels, n_mels, frames))
    }
}

#[derive(Debug)]
struct MelStft {
    forward_basis: Tensor,
    mel_basis: Tensor,
    hop_length: usize,
    win_length: usize,
}

impl MelStft {
    fn new(
        filter_length: usize,
        hop_length: usize,
        win_length: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            forward_basis: vb
                .pp("stft_fn")
                .get((filter_length + 2, 1, filter_length), "forward_basis")?,
            mel_basis: vb.get((64, filter_length / 2 + 1), "mel_basis")?,
            hop_length,
            win_length,
        })
    }

    fn mel_spectrogram(&self, y: &Tensor) -> Result<Tensor> {
        let y = if y.rank() == 2 {
            y.unsqueeze(1)?
        } else {
            y.clone()
        };
        let left_pad = self.win_length.saturating_sub(self.hop_length);
        let y = y.pad_with_zeros(2, left_pad, 0)?;
        let basis = self.forward_basis.to_dtype(y.dtype())?;
        let spec = y.conv1d(&basis, 0, self.hop_length, 1, 1)?;
        let n_freqs = spec.dim(1)? / 2;
        let real = spec.narrow(1, 0, n_freqs)?;
        let imag = spec.narrow(1, n_freqs, n_freqs)?;
        let magnitude = (real.sqr()? + imag.sqr()?)?.sqrt()?;
        let batch = magnitude.dim(0)?;
        let mel_basis = self
            .mel_basis
            .to_dtype(magnitude.dtype())?
            .unsqueeze(0)?
            .expand((batch, self.mel_basis.dim(0)?, n_freqs))?
            .contiguous()?;
        let mel = mel_basis.matmul(&magnitude.contiguous()?)?;
        mel.clamp(1e-5, f64::INFINITY)?.log()
    }
}

#[derive(Debug)]
enum VocoderResBlock {
    Res1(ResBlock1),
    Amp1(AmpBlock1),
}

impl VocoderResBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Res1(block) => block.forward(x),
            Self::Amp1(block) => block.forward(x),
        }
    }
}

#[derive(Debug)]
struct SnakeBeta {
    alpha: Tensor,
    beta: Tensor,
}

impl SnakeBeta {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            alpha: vb.get(channels, "alpha")?,
            beta: vb.get(channels, "beta")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let alpha = self
            .alpha
            .to_dtype(candle_core::DType::F32)?
            .exp()?
            .reshape((1, self.alpha.elem_count(), 1))?;
        let beta = self
            .beta
            .to_dtype(candle_core::DType::F32)?
            .exp()?
            .reshape((1, self.beta.elem_count(), 1))?;
        let x_f32 = x.to_dtype(candle_core::DType::F32)?;
        let periodic = x_f32.broadcast_mul(&alpha)?.sin()?.sqr()?;
        let scaled = periodic.broadcast_div(&(beta + 1e-9)?)?;
        (x_f32 + scaled)?.to_dtype(dtype)
    }
}

#[derive(Debug)]
struct Activation1d {
    act: SnakeBeta,
    upsample: UpSample1d,
    downsample: DownSample1d,
}

impl Activation1d {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            act: SnakeBeta::new(channels, vb.pp("act"))?,
            upsample: UpSample1d::new_checkpointed(2, 12, vb.pp("upsample"))?,
            downsample: DownSample1d::new_checkpointed(2, 12, vb.pp("downsample"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.upsample.forward(x)?;
        let x = self.act.forward(&x)?;
        self.downsample.forward(&x)
    }
}

#[derive(Debug)]
struct UpSample1d {
    ratio: usize,
    stride: usize,
    pad: usize,
    pad_left: usize,
    pad_right: usize,
    filter: ResampleFilter,
}

impl UpSample1d {
    fn new(ratio: usize, kernel_size: usize) -> Self {
        let pad = kernel_size / ratio - 1;
        Self {
            ratio,
            stride: ratio,
            pad,
            pad_left: pad * ratio + (kernel_size - ratio) / 2,
            pad_right: pad * ratio + (kernel_size - ratio).div_ceil(2),
            filter: ResampleFilter::Values(kaiser_sinc_filter1d(
                0.5 / ratio as f64,
                0.6 / ratio as f64,
                kernel_size,
            )),
        }
    }

    fn new_checkpointed(ratio: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        let mut out = Self::new(ratio, kernel_size);
        if vb.contains_tensor("filter") {
            out.filter = ResampleFilter::Tensor(vb.get((1, 1, kernel_size), "filter")?);
        }
        Ok(out)
    }

    fn new_hann(ratio: usize) -> Self {
        let rolloff: f64 = 0.99;
        let lowpass_filter_width: f64 = 6.0;
        let width = (lowpass_filter_width / rolloff).ceil() as usize;
        let kernel_size = 2 * width * ratio + 1;
        let pad = width;
        let pad_left = 2 * width * ratio;
        let pad_right = kernel_size - ratio;
        let mut filter = Vec::with_capacity(kernel_size);
        for n in 0..kernel_size {
            let time_axis = (n as f64 / ratio as f64 - width as f64) * rolloff;
            let time_clamped = time_axis.clamp(-lowpass_filter_width, lowpass_filter_width);
            let window = (time_clamped * std::f64::consts::PI / lowpass_filter_width / 2.0)
                .cos()
                .powi(2);
            filter.push((sinc(time_axis) * window * rolloff / ratio as f64) as f32);
        }
        Self {
            ratio,
            stride: ratio,
            pad,
            pad_left,
            pad_right,
            filter: ResampleFilter::Values(filter),
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = replicate_pad1d(x, self.pad, self.pad)?;
        let filter = filter_tensor(&self.filter, x.dtype(), x.device())?;
        let x = depthwise_conv_transpose1d(&x, &filter, self.stride)?;
        let x = (x * self.ratio as f64)?;
        let out_len = x.dim(2)? - self.pad_left - self.pad_right;
        x.narrow(2, self.pad_left, out_len)
    }
}

#[derive(Debug)]
struct DownSample1d {
    lowpass: LowPassFilter1d,
}

impl DownSample1d {
    fn new_checkpointed(ratio: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            lowpass: LowPassFilter1d::new_checkpointed(
                0.5 / ratio as f64,
                0.6 / ratio as f64,
                ratio,
                kernel_size,
                vb.pp("lowpass"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.lowpass.forward(x)
    }
}

#[derive(Debug)]
struct LowPassFilter1d {
    pad_left: usize,
    pad_right: usize,
    stride: usize,
    filter: ResampleFilter,
}

impl LowPassFilter1d {
    fn new(cutoff: f64, half_width: f64, stride: usize, kernel_size: usize) -> Self {
        let even = kernel_size.is_multiple_of(2);
        Self {
            pad_left: kernel_size / 2 - usize::from(even),
            pad_right: kernel_size / 2,
            stride,
            filter: ResampleFilter::Values(kaiser_sinc_filter1d(cutoff, half_width, kernel_size)),
        }
    }

    fn new_checkpointed(
        cutoff: f64,
        half_width: f64,
        stride: usize,
        kernel_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut out = Self::new(cutoff, half_width, stride, kernel_size);
        if vb.contains_tensor("filter") {
            out.filter = ResampleFilter::Tensor(vb.get((1, 1, kernel_size), "filter")?);
        }
        Ok(out)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = replicate_pad1d(x, self.pad_left, self.pad_right)?;
        let filter = filter_tensor(&self.filter, x.dtype(), x.device())?;
        depthwise_conv1d(&x, &filter, self.stride)
    }
}

#[derive(Debug)]
enum ResampleFilter {
    Tensor(Tensor),
    Values(Vec<f32>),
}

fn filter_tensor(
    filter: &ResampleFilter,
    dtype: DType,
    device: &candle_core::Device,
) -> Result<Tensor> {
    match filter {
        ResampleFilter::Tensor(tensor) => tensor.to_dtype(dtype),
        ResampleFilter::Values(values) => {
            Tensor::from_vec(values.clone(), (1, 1, values.len()), device)?.to_dtype(dtype)
        }
    }
}

fn depthwise_conv1d(x: &Tensor, filter: &Tensor, stride: usize) -> Result<Tensor> {
    let (b, c, t) = x.dims3()?;
    let folded = x.reshape((b * c, 1, t))?.contiguous()?;
    let y = folded.conv1d(filter, 0, stride, 1, 1)?;
    let out_t = y.dim(2)?;
    y.reshape((b, c, out_t))
}

fn depthwise_conv_transpose1d(x: &Tensor, filter: &Tensor, stride: usize) -> Result<Tensor> {
    let (b, c, t) = x.dims3()?;
    let folded = x.reshape((b * c, 1, t))?.contiguous()?;
    let y = folded.conv_transpose1d(filter, 0, 0, stride, 1, 1)?;
    let out_t = y.dim(2)?;
    y.reshape((b, c, out_t))
}

fn replicate_pad1d(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let t = x.dim(2)?;
    let first = x.narrow(2, 0, 1)?;
    let last = x.narrow(2, t - 1, 1)?;
    let mut parts = Vec::with_capacity(left + 1 + right);
    for _ in 0..left {
        parts.push(first.clone());
    }
    parts.push(x.clone());
    for _ in 0..right {
        parts.push(last.clone());
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 2)
}

fn kaiser_sinc_filter1d(cutoff: f64, half_width: f64, kernel_size: usize) -> Vec<f32> {
    let even = kernel_size.is_multiple_of(2);
    let half_size = kernel_size / 2;
    let delta_f = 4.0 * half_width;
    let amplitude = 2.285 * (half_size as f64 - 1.0) * std::f64::consts::PI * delta_f + 7.95;
    let beta = if amplitude > 50.0 {
        0.1102 * (amplitude - 8.7)
    } else if amplitude >= 21.0 {
        0.5842 * (amplitude - 21.0).powf(0.4) + 0.07886 * (amplitude - 21.0)
    } else {
        0.0
    };
    let beta_i0 = bessel_i0(beta);

    let mut values = Vec::with_capacity(kernel_size);
    for n in 0..kernel_size {
        let r = if kernel_size > 1 {
            2.0 * n as f64 / (kernel_size - 1) as f64 - 1.0
        } else {
            0.0
        };
        let window = bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / beta_i0;
        let time = if even {
            n as f64 - half_size as f64 + 0.5
        } else {
            n as f64 - half_size as f64
        };
        let val = if cutoff == 0.0 {
            0.0
        } else {
            2.0 * cutoff * window * sinc(2.0 * cutoff * time)
        };
        values.push(val);
    }

    let sum: f64 = values.iter().sum();
    values.into_iter().map(|v| (v / sum) as f32).collect()
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let pix = std::f64::consts::PI * x;
        pix.sin() / pix
    }
}

fn bessel_i0(x: f64) -> f64 {
    let y = x * x / 4.0;
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..=50 {
        let k = k as f64;
        term *= y / (k * k);
        sum += term;
        if term.abs() < sum.abs() * 1e-14 {
            break;
        }
    }
    sum
}

#[derive(Debug)]
struct AmpBlock1 {
    convs1: Vec<candle_nn::Conv1d>,
    convs2: Vec<candle_nn::Conv1d>,
    acts1: Vec<Activation1d>,
    acts2: Vec<Activation1d>,
}

impl AmpBlock1 {
    fn new(
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        let mut acts1 = Vec::new();
        let mut acts2 = Vec::new();
        let vb1 = vb.pp("convs1");
        let vb2 = vb.pp("convs2");
        let a1_vb = vb.pp("acts1");
        let a2_vb = vb.pp("acts2");

        for (i, &dilation) in dilations.iter().enumerate() {
            let padding1 = dilation * (kernel_size - 1) / 2;
            let cfg1 = Conv1dConfig {
                padding: padding1,
                stride: 1,
                dilation,
                groups: 1,
                ..Default::default()
            };
            convs1.push(candle_nn::conv1d(
                channels,
                channels,
                kernel_size,
                cfg1,
                vb1.pp(i),
            )?);

            let padding2 = (kernel_size - 1) / 2;
            let cfg2 = Conv1dConfig {
                padding: padding2,
                stride: 1,
                dilation: 1,
                groups: 1,
                ..Default::default()
            };
            convs2.push(candle_nn::conv1d(
                channels,
                channels,
                kernel_size,
                cfg2,
                vb2.pp(i),
            )?);
            acts1.push(Activation1d::new(channels, a1_vb.pp(i))?);
            acts2.push(Activation1d::new(channels, a2_vb.pp(i))?);
        }

        Ok(Self {
            convs1,
            convs2,
            acts1,
            acts2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (((conv1, conv2), act1), act2) in self
            .convs1
            .iter()
            .zip(self.convs2.iter())
            .zip(self.acts1.iter())
            .zip(self.acts2.iter())
        {
            let xt = act1.forward(&x)?;
            let xt = conv1.forward(&xt)?;
            let xt = act2.forward(&xt)?;
            let xt = conv2.forward(&xt)?;
            x = (x + xt)?;
        }
        Ok(x)
    }
}
