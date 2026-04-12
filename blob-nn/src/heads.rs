//! Session 3.3 — output heads.
//!
//! Three independent heads read from the Transformer's `[B, S, 128]` output:
//!
//! - **Playing head** (entity-based): a shared MLP is applied to every hand
//!   card token position to produce a scalar score. The caller passes a
//!   `legal_mask: [B, S]` which is `true` only at hand card token positions
//!   whose card is a legal play; illegal positions are set to `-inf` before
//!   softmax.
//! - **Bidding head**: reads the CLS token (position 0 in the encoder's
//!   emit order) and produces logits over bids 0..=13. The caller supplies
//!   a legal-bid mask of shape `[B, 14]`.
//! - **Value head**: reads the CLS token and produces a scalar in `[-1, 1]`.
//!
//! Phase dispatch is the caller's responsibility: the NN always computes a
//! value, plus one of (bid policy, play policy). This module intentionally
//! does not inspect `GamePhase` — the Transformer forward pass is identical
//! regardless of phase, and the phase-specific head is chosen by the owner
//! of the trained model.

use tch::{nn, nn::Module, Kind, Tensor};

use crate::transformer::D_MODEL;

pub const PLAY_MLP_HIDDEN: i64 = 32;
pub const HEAD_HIDDEN: i64 = 64;
pub const NUM_BIDS: i64 = 14; // bids 0..=13
pub const HEAD_DROPOUT: f64 = 0.1;

/// Entity-based playing head. One shared `Linear→GeLU→Linear` MLP is
/// applied to every sequence position; the caller masks non-hand and
/// illegal positions via `legal_mask` before softmax.
#[derive(Debug)]
pub struct PlayingHead {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl PlayingHead {
    pub fn new(vs: &nn::Path) -> Self {
        let lc = nn::LinearConfig::default();
        Self {
            fc1: nn::linear(vs / "fc1", D_MODEL, PLAY_MLP_HIDDEN, lc),
            fc2: nn::linear(vs / "fc2", PLAY_MLP_HIDDEN, 1, lc),
        }
    }

    /// Raw per-position scores `[B, S]`. No masking applied yet.
    pub fn scores(&self, x: &Tensor) -> Tensor {
        let h = self.fc1.forward(x).gelu("none");
        self.fc2.forward(&h).squeeze_dim(-1)
    }

    /// Full forward: scores with `legal_mask` applied, softmaxed over
    /// sequence positions. Padding and non-hand tokens must be `false` in
    /// `legal_mask`; hand-card positions with illegal moves must also be
    /// `false`.
    ///
    /// Returns `[B, S]` probability distribution. For any batch row with
    /// zero legal actions (should not occur in a valid game state), the
    /// returned row is all zeros rather than NaN.
    pub fn forward(&self, x: &Tensor, legal_mask: &Tensor) -> Tensor {
        let scores = self.scores(x);
        let neg_inf = Tensor::from(f64::NEG_INFINITY).to_device(x.device());
        let masked = scores.where_self(legal_mask, &neg_inf);
        let probs = masked.softmax(-1, Kind::Float);
        probs.nan_to_num(0.0, 0.0, 0.0)
    }
}

/// Bidding head: CLS → MLP → logits over 14 bids.
#[derive(Debug)]
pub struct BiddingHead {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl BiddingHead {
    pub fn new(vs: &nn::Path) -> Self {
        let lc = nn::LinearConfig::default();
        Self {
            fc1: nn::linear(vs / "fc1", D_MODEL, HEAD_HIDDEN, lc),
            fc2: nn::linear(vs / "fc2", HEAD_HIDDEN, NUM_BIDS, lc),
        }
    }

    /// Raw logits `[B, 14]` from the CLS token.
    pub fn logits(&self, x: &Tensor, train: bool) -> Tensor {
        let cls = x.select(1, 0); // [B, D]
        let h = self
            .fc1
            .forward(&cls)
            .gelu("none")
            .dropout(HEAD_DROPOUT, train);
        self.fc2.forward(&h)
    }

    /// Masked softmax over legal bids. `legal_bid_mask: [B, 14]` bool.
    pub fn forward(&self, x: &Tensor, legal_bid_mask: &Tensor, train: bool) -> Tensor {
        let logits = self.logits(x, train);
        let neg_inf = Tensor::from(f64::NEG_INFINITY).to_device(x.device());
        let masked = logits.where_self(legal_bid_mask, &neg_inf);
        let probs = masked.softmax(-1, Kind::Float);
        probs.nan_to_num(0.0, 0.0, 0.0)
    }
}

/// Value head: CLS → MLP → scalar ∈ \[-1, 1\].
#[derive(Debug)]
pub struct ValueHead {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl ValueHead {
    pub fn new(vs: &nn::Path) -> Self {
        let lc = nn::LinearConfig::default();
        Self {
            fc1: nn::linear(vs / "fc1", D_MODEL, HEAD_HIDDEN, lc),
            fc2: nn::linear(vs / "fc2", HEAD_HIDDEN, 1, lc),
        }
    }

    /// Scalar value `[B]` ∈ [-1, 1].
    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let cls = x.select(1, 0); // [B, D]
        let h = self
            .fc1
            .forward(&cls)
            .gelu("none")
            .dropout(HEAD_DROPOUT, train);
        self.fc2.forward(&h).tanh().squeeze_dim(-1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{pad_batch, InputProjection, FEAT_DIM};
    use crate::transformer::TransformerEncoder;
    use tch::{nn::VarStore, Device, Kind};

    fn new_vs() -> VarStore {
        VarStore::new(Device::Cpu)
    }

    #[test]
    fn playing_head_masks_and_normalizes() {
        let vs = new_vs();
        let head = PlayingHead::new(&vs.root());
        let (b, s) = (2i64, 6i64);
        let x = Tensor::randn([b, s, D_MODEL], (Kind::Float, Device::Cpu));
        // Legal at positions 1, 3 for batch 0; positions 0, 2, 4 for batch 1.
        let mut m = vec![false; (b * s) as usize];
        m[1] = true;
        m[3] = true;
        m[(s + 0) as usize] = true;
        m[(s + 2) as usize] = true;
        m[(s + 4) as usize] = true;
        let mask = Tensor::from_slice(&m).view([b, s]);
        let probs = head.forward(&x, &mask);

        assert_eq!(probs.size(), vec![b, s]);
        // Illegal positions are exactly zero.
        let illegal = mask.logical_not();
        let illegal_sum: f64 = probs
            .where_self(&illegal, &Tensor::from(0.0))
            .sum(Kind::Float)
            .double_value(&[]);
        assert_eq!(illegal_sum, 0.0);
        // Each row sums to 1.
        let row_sums: Vec<f64> = (0..b)
            .map(|i| probs.get(i).sum(Kind::Float).double_value(&[]))
            .collect();
        for r in row_sums {
            assert!((r - 1.0).abs() < 1e-5, "row sum {r} != 1");
        }
    }

    #[test]
    fn bidding_head_shape_and_softmax() {
        let vs = new_vs();
        let head = BiddingHead::new(&vs.root());
        let (b, s) = (3i64, 4i64);
        let x = Tensor::randn([b, s, D_MODEL], (Kind::Float, Device::Cpu));
        // All bids legal.
        let mask = Tensor::ones([b, NUM_BIDS], (Kind::Bool, Device::Cpu));
        let probs = head.forward(&x, &mask, false);
        assert_eq!(probs.size(), vec![b, NUM_BIDS]);
        for i in 0..b {
            let r: f64 = probs.get(i).sum(Kind::Float).double_value(&[]);
            assert!((r - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn bidding_head_respects_legal_mask() {
        let vs = new_vs();
        let head = BiddingHead::new(&vs.root());
        let b = 1i64;
        let x = Tensor::randn([b, 3, D_MODEL], (Kind::Float, Device::Cpu));
        // Only bids 2, 5, 9 legal.
        let mut m = vec![false; NUM_BIDS as usize];
        m[2] = true;
        m[5] = true;
        m[9] = true;
        let mask = Tensor::from_slice(&m).view([b, NUM_BIDS]);
        let probs = head.forward(&x, &mask, false);
        let p0: Vec<f64> = (0..NUM_BIDS).map(|i| probs.get(0).get(i).double_value(&[])).collect();
        for (i, p) in p0.iter().enumerate() {
            if m[i] {
                assert!(*p > 0.0, "legal bid {i} has zero prob");
            } else {
                assert_eq!(*p, 0.0, "illegal bid {i} has non-zero prob {p}");
            }
        }
    }

    #[test]
    fn value_head_in_tanh_range() {
        let vs = new_vs();
        let head = ValueHead::new(&vs.root());
        let (b, s) = (4i64, 5i64);
        // Use large magnitudes to push toward saturation.
        let x = Tensor::randn([b, s, D_MODEL], (Kind::Float, Device::Cpu)) * 100.0;
        let v = head.forward(&x, false);
        assert_eq!(v.size(), vec![b]);
        let min = v.min().double_value(&[]);
        let max = v.max().double_value(&[]);
        assert!(min >= -1.0 && max <= 1.0, "value out of [-1,1]: min={min} max={max}");
    }

    #[test]
    fn total_param_count_matches_spec() {
        // Full model (input + transformer + 3 heads) ≈ 1.63M params.
        let vs = new_vs();
        let root = vs.root();
        let _ip = InputProjection::new(&(&root / "input"));
        let _tr = TransformerEncoder::new(&(&root / "transformer"));
        let _ph = PlayingHead::new(&(&root / "play_head"));
        let _bh = BiddingHead::new(&(&root / "bid_head"));
        let _vh = ValueHead::new(&(&root / "value_head"));

        let total: i64 = vs.variables().values().map(|t| t.numel() as i64).sum();
        // Spec: ~1.63M. Accept ±5%.
        assert!(
            (1_550_000..=1_710_000).contains(&total),
            "full model param count {total} outside spec band (~1.63M)"
        );
    }

    #[test]
    fn end_to_end_forward_from_encoded_state() {
        use blob_engine::encoder::{encode, TOKEN_TYPE_HAND};
        use blob_engine::{dealing::deal, game::new_game};
        use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256PlusPlus};

        let vs = new_vs();
        let root = vs.root();
        let ip = InputProjection::new(&(&root / "input"));
        let tr = TransformerEncoder::new(&(&root / "transformer"));
        let ph = PlayingHead::new(&(&root / "play_head"));
        let bh = BiddingHead::new(&(&root / "bid_head"));
        let vh = ValueHead::new(&(&root / "value_head"));

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let mut s1 = new_game(4, 5).unwrap();
        deal(&mut s1, &mut rng);
        let mut s2 = new_game(5, 5).unwrap();
        deal(&mut s2, &mut rng);

        let enc1 = encode(&s1, s1.current_player);
        let enc2 = encode(&s2, s2.current_player);
        let batch = pad_batch(&[enc1.clone(), enc2.clone()], Device::Cpu);
        assert_eq!(batch.features.size()[2], FEAT_DIM);

        let embedded = ip.forward(
            &batch.features,
            &batch.token_types,
            &batch.chrono_indices,
            &batch.attention_mask,
        );
        let encoded = tr.forward(&embedded, &batch.attention_mask, false);

        // Build a play legal_mask that marks every hand card token as legal.
        let hand_mask = batch.token_types.eq(TOKEN_TYPE_HAND as i64);
        let play_probs = ph.forward(&encoded, &hand_mask);
        let s = encoded.size()[1];
        assert_eq!(play_probs.size(), vec![2, s]);
        for i in 0..2i64 {
            let r: f64 = play_probs.get(i).sum(Kind::Float).double_value(&[]);
            assert!((r - 1.0).abs() < 1e-5);
        }

        let bid_mask = Tensor::ones([2, NUM_BIDS], (Kind::Bool, Device::Cpu));
        let bid_probs = bh.forward(&encoded, &bid_mask, false);
        assert_eq!(bid_probs.size(), vec![2, NUM_BIDS]);

        let v = vh.forward(&encoded, false);
        assert_eq!(v.size(), vec![2]);
        let vmin = v.min().double_value(&[]);
        let vmax = v.max().double_value(&[]);
        assert!(vmin >= -1.0 && vmax <= 1.0);
    }
}
