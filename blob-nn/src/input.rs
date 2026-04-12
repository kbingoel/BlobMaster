//! Session 3.1 — per-token-type input projections.
//!
//! Projects the variable-length token sequence produced by
//! `blob_engine::encoder::encode` into a uniform `[batch, seq, 128]` tensor
//! that the Transformer encoder (Session 3.2) consumes.
//!
//! Input tensors (constructed by callers; see [`pad_batch`]):
//! - `features: [B, S, 48]` f32 — per-token features, right-padded to 48
//! - `token_types: [B, S]` i64 — values 0..=4 (see encoder constants)
//! - `chrono_indices: [B, S]` i64 — 0..52 for played tokens, 0 elsewhere
//! - `attention_mask: [B, S]` bool — true for real tokens, false for padding
//!
//! Output: `[B, S, 128]` f32 with padding rows zeroed.

use blob_engine::encoder::{
    EncodedState, CONTEXT_DIM as ENC_CONTEXT_DIM, HAND_CARD_DIM, PLAYED_CARD_DIM,
    PLAYER_STATE_DIM, TOKEN_TYPE_CLS, TOKEN_TYPE_CONTEXT, TOKEN_TYPE_HAND,
    TOKEN_TYPE_PLAYED, TOKEN_TYPE_PLAYER,
};
use tch::{nn, nn::Module, Tensor};

pub const D_MODEL: i64 = 128;
/// Max per-token feature width (played card = 48). All features are
/// right-padded to this width before being stacked into a batched tensor.
pub const FEAT_DIM: i64 = PLAYED_CARD_DIM as i64;
/// Chronological-embedding table size — max 52 plays in a 4P×13C round.
pub const MAX_CHRONO: i64 = 52;

const HAND_DIM: i64 = HAND_CARD_DIM as i64;
const PLAYED_DIM: i64 = PLAYED_CARD_DIM as i64;
const PLAYER_DIM: i64 = PLAYER_STATE_DIM as i64;
const CTX_DIM: i64 = ENC_CONTEXT_DIM as i64;

/// Per-token-type input projections + CLS parameter + chronological embedding.
#[derive(Debug)]
pub struct InputProjection {
    hand: nn::Linear,
    played: nn::Linear,
    player: nn::Linear,
    context: nn::Linear,
    cls: Tensor,
    chrono: nn::Embedding,
}

impl InputProjection {
    pub fn new(vs: &nn::Path) -> Self {
        let lc = nn::LinearConfig::default();
        let ec = nn::EmbeddingConfig::default();
        Self {
            hand: nn::linear(vs / "hand_proj", HAND_DIM, D_MODEL, lc),
            played: nn::linear(vs / "played_proj", PLAYED_DIM, D_MODEL, lc),
            player: nn::linear(vs / "player_proj", PLAYER_DIM, D_MODEL, lc),
            context: nn::linear(vs / "context_proj", CTX_DIM, D_MODEL, lc),
            cls: vs.randn("cls", &[D_MODEL], 0.0, 0.02),
            chrono: nn::embedding(vs / "chrono_embed", MAX_CHRONO, D_MODEL, ec),
        }
    }

    /// Project the batched padded features into `[B, S, 128]`.
    pub fn forward(
        &self,
        features: &Tensor,
        token_types: &Tensor,
        chrono_indices: &Tensor,
        attention_mask: &Tensor,
    ) -> Tensor {
        let sz = features.size();
        assert_eq!(sz.len(), 3, "features must be [B, S, FEAT_DIM]");
        assert_eq!(sz[2], FEAT_DIM, "last dim must equal FEAT_DIM (48)");
        let b = sz[0];
        let s = sz[1];

        // Per-type projections over the full [B, S, ...] tensor; each consumes
        // only its meaningful prefix of the 48-dim feature vector.
        let hand_out = self.hand.forward(&features.narrow(-1, 0, HAND_DIM));
        let played_out = self.played.forward(&features.narrow(-1, 0, PLAYED_DIM));
        let player_out = self.player.forward(&features.narrow(-1, 0, PLAYER_DIM));
        let context_out = self.context.forward(&features.narrow(-1, 0, CTX_DIM));
        let cls_out = self.cls.view([1, 1, D_MODEL]).expand([b, s, D_MODEL], true);

        let kind = features.kind();
        let mask_of = |tt: i64| -> Tensor {
            token_types.eq(tt).to_kind(kind).unsqueeze(-1)
        };

        let mut out = cls_out * mask_of(TOKEN_TYPE_CLS as i64)
            + context_out * mask_of(TOKEN_TYPE_CONTEXT as i64)
            + player_out * mask_of(TOKEN_TYPE_PLAYER as i64)
            + hand_out * mask_of(TOKEN_TYPE_HAND as i64)
            + played_out * mask_of(TOKEN_TYPE_PLAYED as i64);

        // Chronological embedding added for played card tokens only.
        let chrono_emb = self.chrono.forward(chrono_indices);
        out = out + chrono_emb * mask_of(TOKEN_TYPE_PLAYED as i64);

        // Zero out padding rows so they contribute nothing downstream.
        out * attention_mask.to_kind(kind).unsqueeze(-1)
    }
}

/// Batched tensor bundle produced from a slice of `EncodedState`s.
pub struct InputBatch {
    pub features: Tensor,       // [B, S, 48] f32
    pub token_types: Tensor,    // [B, S] i64
    pub chrono_indices: Tensor, // [B, S] i64
    pub attention_mask: Tensor, // [B, S] bool
}

/// Pad and stack a batch of encoded states into a single set of tensors.
///
/// Sequence length is the max `num_tokens` across the batch; shorter
/// sequences are right-padded. Per-token feature vectors are right-padded to
/// `FEAT_DIM` (48).
pub fn pad_batch(states: &[EncodedState], device: tch::Device) -> InputBatch {
    let b = states.len();
    assert!(b > 0, "batch must be non-empty");
    let max_s = states.iter().map(|e| e.num_tokens).max().unwrap_or(0);
    let feat_dim = FEAT_DIM as usize;

    let mut feat_buf = vec![0.0f32; b * max_s * feat_dim];
    let mut tt_buf = vec![0i64; b * max_s];
    let mut chrono_buf = vec![0i64; b * max_s];
    let mut mask_buf = vec![false; b * max_s];

    for (bi, enc) in states.iter().enumerate() {
        for si in 0..enc.num_tokens {
            let row = (bi * max_s + si) * feat_dim;
            let f = &enc.features[si];
            for (j, v) in f.iter().enumerate() {
                feat_buf[row + j] = *v;
            }
            tt_buf[bi * max_s + si] = enc.token_types[si] as i64;
            chrono_buf[bi * max_s + si] = enc.chronological_indices[si] as i64;
            mask_buf[bi * max_s + si] = true;
        }
    }

    let b64 = b as i64;
    let s64 = max_s as i64;
    let features = Tensor::from_slice(&feat_buf)
        .view([b64, s64, FEAT_DIM])
        .to_device(device);
    let token_types = Tensor::from_slice(&tt_buf).view([b64, s64]).to_device(device);
    let chrono_indices = Tensor::from_slice(&chrono_buf).view([b64, s64]).to_device(device);
    let attention_mask = Tensor::from_slice(&mask_buf).view([b64, s64]).to_device(device);

    InputBatch {
        features,
        token_types,
        chrono_indices,
        attention_mask,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn::VarStore, Device, Kind};

    fn make_proj() -> (VarStore, InputProjection) {
        let vs = VarStore::new(Device::Cpu);
        let p = InputProjection::new(&vs.root());
        (vs, p)
    }

    #[test]
    fn output_shape_is_batch_seq_dmodel() {
        let (_vs, proj) = make_proj();
        let (b, s) = (2i64, 10i64);
        let features = Tensor::randn([b, s, FEAT_DIM], (Kind::Float, Device::Cpu));
        // Token types cycle through 0..=4.
        let tt_vec: Vec<i64> = (0..(b * s)).map(|i| i % 5).collect();
        let token_types = Tensor::from_slice(&tt_vec).view([b, s]);
        let chrono_vec: Vec<i64> = (0..(b * s)).map(|i| i % MAX_CHRONO).collect();
        let chrono_indices = Tensor::from_slice(&chrono_vec).view([b, s]);
        let mask = Tensor::ones([b, s], (Kind::Bool, Device::Cpu));

        let out = proj.forward(&features, &token_types, &chrono_indices, &mask);
        assert_eq!(out.size(), vec![b, s, D_MODEL]);
        assert_eq!(out.kind(), Kind::Float);
    }

    #[test]
    fn padding_rows_are_zero() {
        let (_vs, proj) = make_proj();
        let (b, s) = (1i64, 4i64);
        let features = Tensor::randn([b, s, FEAT_DIM], (Kind::Float, Device::Cpu));
        let token_types = Tensor::from_slice(&[0i64, 1, 3, 4]).view([b, s]);
        let chrono_indices = Tensor::from_slice(&[0i64, 0, 0, 7]).view([b, s]);
        // Last position is padding.
        let mask = Tensor::from_slice(&[true, true, true, false]).view([b, s]);

        let out = proj.forward(&features, &token_types, &chrono_indices, &mask);
        let pad_row = out.get(0).get(3); // [128]
        let sum_abs: f64 = pad_row.abs().sum(Kind::Float).double_value(&[]);
        assert_eq!(sum_abs, 0.0);
    }

    #[test]
    fn chrono_embedding_only_affects_played_tokens() {
        let (_vs, proj) = make_proj();
        let (b, s) = (1i64, 2i64);
        let features = Tensor::zeros([b, s, FEAT_DIM], (Kind::Float, Device::Cpu));
        // Pos 0: hand (type 3). Pos 1: played (type 4).
        let token_types = Tensor::from_slice(&[TOKEN_TYPE_HAND as i64, TOKEN_TYPE_PLAYED as i64])
            .view([b, s]);
        let mask = Tensor::ones([b, s], (Kind::Bool, Device::Cpu));

        let out_a = proj.forward(
            &features,
            &token_types,
            &Tensor::from_slice(&[0i64, 5]).view([b, s]),
            &mask,
        );
        let out_b = proj.forward(
            &features,
            &token_types,
            &Tensor::from_slice(&[0i64, 17]).view([b, s]),
            &mask,
        );

        // Hand row unaffected by chrono index; played row differs.
        let diff_hand: f64 = (out_a.get(0).get(0) - out_b.get(0).get(0))
            .abs()
            .sum(Kind::Float)
            .double_value(&[]);
        let diff_played: f64 = (out_a.get(0).get(1) - out_b.get(0).get(1))
            .abs()
            .sum(Kind::Float)
            .double_value(&[]);
        assert_eq!(diff_hand, 0.0);
        assert!(diff_played > 0.0);
    }

    #[test]
    fn pad_batch_from_encoded_state_roundtrip() {
        use blob_engine::{dealing::deal, encoder::encode, game::new_game};
        use rand_xoshiro::rand_core::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut s1 = new_game(4, 5).unwrap();
        deal(&mut s1, &mut rng);
        let mut s2 = new_game(5, 5).unwrap();
        deal(&mut s2, &mut rng);

        let enc1 = encode(&s1, s1.current_player);
        let enc2 = encode(&s2, s2.current_player);
        let batch = pad_batch(&[enc1.clone(), enc2.clone()], Device::Cpu);

        let max_s = enc1.num_tokens.max(enc2.num_tokens) as i64;
        assert_eq!(batch.features.size(), vec![2, max_s, FEAT_DIM]);
        assert_eq!(batch.token_types.size(), vec![2, max_s]);
        assert_eq!(batch.attention_mask.size(), vec![2, max_s]);

        let (_vs, proj) = make_proj();
        let out = proj.forward(
            &batch.features,
            &batch.token_types,
            &batch.chrono_indices,
            &batch.attention_mask,
        );
        assert_eq!(out.size(), vec![2, max_s, D_MODEL]);
    }
}
