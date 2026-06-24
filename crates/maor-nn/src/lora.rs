use candle_core::{bail, DType, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

/// Runtime LoRA adapter used for inference-time weight merging.
#[derive(Clone)]
pub struct LoraConfig {
    vb: VarBuilder<'static>,
    scale: f64,
}

impl std::fmt::Debug for LoraConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoraConfig")
            .field("scale", &self.scale)
            .finish_non_exhaustive()
    }
}

impl LoraConfig {
    pub fn new(vb: VarBuilder<'static>, scale: f64) -> Self {
        Self { vb, scale }
    }

    pub fn pp<S: ToString>(&self, name: S) -> Self {
        Self {
            vb: self.vb.pp(name),
            scale: self.scale,
        }
    }
}

pub fn linear(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    lora: Option<&LoraConfig>,
) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let weight = merge_lora_weight(weight, in_dim, out_dim, lora)?;
    let bias = Some(vb.get(out_dim, "bias")?);
    Ok(Linear::new(weight, bias))
}

pub fn linear_no_bias(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    lora: Option<&LoraConfig>,
) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let weight = merge_lora_weight(weight, in_dim, out_dim, lora)?;
    Ok(Linear::new(weight, None))
}

fn merge_lora_weight(
    weight: Tensor,
    in_dim: usize,
    out_dim: usize,
    lora: Option<&LoraConfig>,
) -> Result<Tensor> {
    let Some(lora) = lora else {
        return Ok(weight);
    };
    if lora.scale == 0.0 || !lora.vb.contains_tensor("lora_A.weight") {
        return Ok(weight);
    }

    let a = lora.vb.get_unchecked("lora_A.weight")?;
    let b = lora.vb.get_unchecked("lora_B.weight")?;
    let (rank, lora_in) = a.dims2()?;
    let (lora_out, rank_b) = b.dims2()?;
    if lora_in != in_dim || lora_out != out_dim || rank_b != rank {
        bail!(
            "LoRA shape mismatch: expected A=({rank},{in_dim}) B=({out_dim},{rank}), got A=({rank},{lora_in}) B=({lora_out},{rank_b})"
        );
    }

    let dtype = weight.dtype();
    let delta = b.to_dtype(DType::F32)?.matmul(&a.to_dtype(DType::F32)?)?;
    let delta = (delta * lora.scale)?;
    (weight.to_dtype(DType::F32)? + delta)?.to_dtype(dtype)
}
