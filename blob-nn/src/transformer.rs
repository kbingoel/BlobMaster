//! Session 3.2 — pre-norm Transformer encoder.
//!
//! Stacks 8 identical blocks of the form
//! `x + MHA(LN(x))` then `x + FFN(LN(x))`
//! over the `[B, S, 128]` tensor produced by the input projection
//! (Session 3.1).
//!
//! Multi-head self-attention is implemented manually rather than via
//! `tch::nn::MultiheadAttention` so that the key-padding mask can be
//! applied cleanly: padded positions receive `-inf` attention scores and
//! therefore contribute zero attention weight after softmax.

use tch::{nn, nn::Module, Kind, Tensor};

pub const D_MODEL: i64 = 128;
pub const N_HEADS: i64 = 8;
pub const HEAD_DIM: i64 = D_MODEL / N_HEADS; // 16
pub const FFN_DIM: i64 = 512;
pub const N_LAYERS: usize = 8;
pub const DROPOUT: f64 = 0.1;
pub const LN_EPS: f64 = 1e-5;

#[derive(Debug)]
struct MultiHeadSelfAttention {
    qkv: nn::Linear,
    out: nn::Linear,
}

impl MultiHeadSelfAttention {
    fn new(vs: &nn::Path) -> Self {
        let lc = nn::LinearConfig::default();
        Self {
            qkv: nn::linear(vs / "qkv", D_MODEL, 3 * D_MODEL, lc),
            out: nn::linear(vs / "out", D_MODEL, D_MODEL, lc),
        }
    }

    /// `x: [B, S, D]`, `attention_mask: [B, S]` bool (true = real token).
    fn forward(&self, x: &Tensor, attention_mask: &Tensor, train: bool) -> Tensor {
        let sz = x.size();
        let b = sz[0];
        let s = sz[1];

        let qkv = self.qkv.forward(x); // [B, S, 3D]
        let qkv = qkv.view([b, s, 3, N_HEADS, HEAD_DIM]);
        // [3, B, H, S, Dh]
        let qkv = qkv.permute([2, 0, 3, 1, 4]).contiguous();
        let q = qkv.get(0);
        let k = qkv.get(1);
        let v = qkv.get(2);

        // Scores: [B, H, S, S]
        let scale = (HEAD_DIM as f64).sqrt();
        let scores = q.matmul(&k.transpose(-2, -1)) / scale;

        // Key-padding mask: [B, 1, 1, S], true where key is padding (to set -inf).
        let key_pad = attention_mask
            .logical_not()
            .view([b, 1, 1, s]);
        let neg_inf = Tensor::from(f64::NEG_INFINITY).to_device(x.device());
        let masked = scores.where_self(&key_pad.logical_not(), &neg_inf);

        // Softmax; rows that are entirely padding on the query side will be
        // zeroed by the mask applied at the block level (and at input).
        let attn = masked.softmax(-1, Kind::Float);
        // Replace any NaN introduced by all-`-inf` rows (all-padding queries)
        // with zeros so they do not contaminate gradients.
        let attn = attn.nan_to_num(0.0, 0.0, 0.0);
        let attn = attn.dropout(DROPOUT, train);

        let ctx = attn.matmul(&v); // [B, H, S, Dh]
        let ctx = ctx.transpose(1, 2).contiguous().view([b, s, D_MODEL]);
        self.out.forward(&ctx)
    }
}

#[derive(Debug)]
struct Ffn {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Ffn {
    fn new(vs: &nn::Path) -> Self {
        let lc = nn::LinearConfig::default();
        Self {
            fc1: nn::linear(vs / "fc1", D_MODEL, FFN_DIM, lc),
            fc2: nn::linear(vs / "fc2", FFN_DIM, D_MODEL, lc),
        }
    }

    fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let h = self.fc1.forward(x).gelu("none").dropout(DROPOUT, train);
        self.fc2.forward(&h).dropout(DROPOUT, train)
    }
}

#[derive(Debug)]
struct EncoderBlock {
    ln1: nn::LayerNorm,
    attn: MultiHeadSelfAttention,
    ln2: nn::LayerNorm,
    ffn: Ffn,
}

impl EncoderBlock {
    fn new(vs: &nn::Path) -> Self {
        let ln_cfg = nn::LayerNormConfig {
            eps: LN_EPS,
            elementwise_affine: true,
            ..Default::default()
        };
        Self {
            ln1: nn::layer_norm(vs / "ln1", vec![D_MODEL], ln_cfg),
            attn: MultiHeadSelfAttention::new(&(vs / "attn")),
            ln2: nn::layer_norm(vs / "ln2", vec![D_MODEL], ln_cfg),
            ffn: Ffn::new(&(vs / "ffn")),
        }
    }

    fn forward(&self, x: &Tensor, attention_mask: &Tensor, train: bool) -> Tensor {
        let h = x + self.attn.forward(&self.ln1.forward(x), attention_mask, train);
        &h + self.ffn.forward(&self.ln2.forward(&h), train)
    }
}

/// Stack of `N_LAYERS` pre-norm Transformer encoder blocks.
#[derive(Debug)]
pub struct TransformerEncoder {
    layers: Vec<EncoderBlock>,
}

impl TransformerEncoder {
    pub fn new(vs: &nn::Path) -> Self {
        let layers = (0..N_LAYERS)
            .map(|i| EncoderBlock::new(&(vs / format!("layer{i}"))))
            .collect();
        Self { layers }
    }

    /// `x: [B, S, 128]`, `attention_mask: [B, S]` bool (true = real token).
    /// Returns `[B, S, 128]`. Padding rows are re-zeroed on exit.
    pub fn forward(&self, x: &Tensor, attention_mask: &Tensor, train: bool) -> Tensor {
        let mut h = x.shallow_clone();
        for layer in &self.layers {
            h = layer.forward(&h, attention_mask, train);
        }
        &h * attention_mask.to_kind(h.kind()).unsqueeze(-1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn::VarStore, Device, Kind};

    fn make_enc() -> (VarStore, TransformerEncoder) {
        let vs = VarStore::new(Device::Cpu);
        let enc = TransformerEncoder::new(&vs.root());
        (vs, enc)
    }

    #[test]
    fn output_shape_preserved() {
        let (_vs, enc) = make_enc();
        let (b, s) = (2i64, 10i64);
        let x = Tensor::randn([b, s, D_MODEL], (Kind::Float, Device::Cpu));
        let mask = Tensor::ones([b, s], (Kind::Bool, Device::Cpu));
        let out = enc.forward(&x, &mask, false);
        assert_eq!(out.size(), vec![b, s, D_MODEL]);
    }

    #[test]
    fn padded_positions_are_zero_on_exit() {
        let (_vs, enc) = make_enc();
        let (b, s) = (1i64, 5i64);
        let x = Tensor::randn([b, s, D_MODEL], (Kind::Float, Device::Cpu));
        let mask = Tensor::from_slice(&[true, true, true, false, false]).view([b, s]);
        let out = enc.forward(&x, &mask, false);
        let pad_sum: f64 = out
            .narrow(1, 3, 2)
            .abs()
            .sum(Kind::Float)
            .double_value(&[]);
        assert_eq!(pad_sum, 0.0);
    }

    #[test]
    fn gradient_flows_through_all_layers() {
        let (vs, enc) = make_enc();
        let (b, s) = (2i64, 6i64);
        let x = Tensor::randn([b, s, D_MODEL], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let mask = Tensor::ones([b, s], (Kind::Bool, Device::Cpu));
        let out = enc.forward(&x, &mask, true);
        let loss = out.sum(Kind::Float);
        loss.backward();

        // Every trainable variable should have a non-NaN, non-all-zero gradient.
        for (name, var) in vs.variables() {
            let g = var.grad();
            assert!(g.defined(), "no grad for {name}");
            let any_nan: bool = g.isnan().any().int64_value(&[]) != 0;
            assert!(!any_nan, "NaN grad in {name}");
            let abs_sum: f64 = g.abs().sum(Kind::Float).double_value(&[]);
            assert!(abs_sum > 0.0, "zero grad for {name}");
        }
    }

    #[test]
    fn parameter_count_matches_spec() {
        let (vs, _enc) = make_enc();
        let total: i64 = vs
            .variables()
            .values()
            .map(|t| t.numel() as i64)
            .sum();
        // Spec: ~198K × 8 = ~1,585K. Allow a tight ±5% band.
        assert!(
            (1_500_000..=1_700_000).contains(&total),
            "transformer param count {total} outside expected band"
        );
    }

    #[test]
    fn padded_keys_do_not_leak_into_real_queries() {
        // Two batches: first with all-real tokens, second identical but with
        // an extra padded position appended. Outputs on the real positions
        // should match (padding must not influence attention).
        let (_vs, enc) = make_enc();
        let s_real = 4i64;
        let x_short = Tensor::randn([1, s_real, D_MODEL], (Kind::Float, Device::Cpu));
        let mask_short = Tensor::ones([1, s_real], (Kind::Bool, Device::Cpu));
        let out_short = enc.forward(&x_short, &mask_short, false);

        let pad_row = Tensor::randn([1, 1, D_MODEL], (Kind::Float, Device::Cpu));
        let x_long = Tensor::cat(&[&x_short, &pad_row], 1);
        let mask_long = Tensor::from_slice(&[true, true, true, true, false]).view([1, 5]);
        let out_long = enc.forward(&x_long, &mask_long, false);

        let diff: f64 = (out_short - out_long.narrow(1, 0, s_real))
            .abs()
            .sum(Kind::Float)
            .double_value(&[]);
        // Small float-accumulation across 8 LN+MHA layers is expected; the
        // guarantee is that `-inf` masked keys contribute zero attention
        // weight, not exact bitwise equality.
        assert!(diff < 1e-2, "padding leaked into real positions (diff={diff})");
    }
}
