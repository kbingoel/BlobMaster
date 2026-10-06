//! The two gen-2 networks (gen-2.md §5.3), each a thin owning wrapper
//! around the input, transformer and head building blocks:
//!
//! - [`PolicyNet`] (P): the seat to move's own view → bid policy and
//!   per-hand-card play policy. 8 layers.
//! - [`ValueNet`] (V): the whole deal → expected ŝ for every seat, read at
//!   the player tokens. 4 layers, plus a projection for opponents' cards.
//!
//! Each network registers its parameters in its own `VarStore`, which the
//! caller owns and hands to the optimizer. Sub-modules sit under `input/`,
//! `transformer/` and the head names, so parameter names are stable and
//! match `scripts/export_onnx.py`'s `PolicyNet` / `ValueNet`.

use tch::{nn, Tensor};

use crate::heads::{BiddingHead, PlayingHead, SeatValueHead};
use crate::input::{InputBatch, InputProjection};
use crate::transformer::TransformerEncoder;

/// P's depth.
pub const P_LAYERS: usize = 8;
/// V's depth.
pub const V_LAYERS: usize = 4;
/// Sequence position of the first player token: both encoder modes start
/// `[CLS, context, players…]`, players in relative-seat order.
pub const FIRST_PLAYER_TOKEN: i64 = 2;

fn encode(input: &InputProjection, transformer: &TransformerEncoder, batch: &InputBatch, train: bool) -> Tensor {
    let x = input.forward(&batch.features, &batch.token_types, &batch.chrono_indices, &batch.attention_mask);
    transformer.forward(&x, &batch.attention_mask, train)
}

/// The policy net P.
#[derive(Debug)]
pub struct PolicyNet {
    pub input: InputProjection,
    pub transformer: TransformerEncoder,
    pub play_head: PlayingHead,
    pub bid_head: BiddingHead,
}

impl PolicyNet {
    pub fn new(vs: &nn::Path) -> Self {
        Self {
            input: InputProjection::new(&(vs / "input"), false),
            transformer: TransformerEncoder::new(&(vs / "transformer"), P_LAYERS),
            play_head: PlayingHead::new(&(vs / "play_head")),
            bid_head: BiddingHead::new(&(vs / "bid_head")),
        }
    }

    /// Playing: the policy `[B, S]` over sequence positions.
    /// `play_legal_mask: [B, S]` is true only at hand-card tokens of legal plays.
    pub fn forward_play(&self, batch: &InputBatch, play_legal_mask: &Tensor, train: bool) -> Tensor {
        let h = encode(&self.input, &self.transformer, batch, train);
        self.play_head.forward(&h, play_legal_mask)
    }

    /// Bidding: the policy `[B, 14]` over bids.
    pub fn forward_bid(&self, batch: &InputBatch, legal_bid_mask: &Tensor, train: bool) -> Tensor {
        let h = encode(&self.input, &self.transformer, batch, train);
        self.bid_head.forward(&h, legal_bid_mask, train)
    }
}

/// The value net V. Its input is V mode (`encoder::encode_value`).
#[derive(Debug)]
pub struct ValueNet {
    pub input: InputProjection,
    pub transformer: TransformerEncoder,
    pub value_head: SeatValueHead,
}

impl ValueNet {
    pub fn new(vs: &nn::Path) -> Self {
        Self {
            input: InputProjection::new(&(vs / "input"), true),
            transformer: TransformerEncoder::new(&(vs / "transformer"), V_LAYERS),
            value_head: SeatValueHead::new(&(vs / "value_head")),
        }
    }

    /// Per-token logits `[B, S]`; [`ValueNet::forward`] is their sigmoid.
    pub fn logits(&self, batch: &InputBatch, train: bool) -> Tensor {
        self.value_head.logits(&encode(&self.input, &self.transformer, batch, train), train)
    }

    /// Per-token values `[B, S]`, as the exported graph outputs them.
    pub fn forward(&self, batch: &InputBatch, train: bool) -> Tensor {
        self.logits(batch, train).sigmoid()
    }

    /// Logits of the first `seats` relative seats, `[B, seats]`. Rows with
    /// fewer players have hand-card entries past their last seat; mask those.
    pub fn seat_logits(&self, batch: &InputBatch, seats: i64, train: bool) -> Tensor {
        self.logits(batch, train).narrow(1, FIRST_PLAYER_TOKEN, seats)
    }

    /// Expected ŝ of the first `seats` relative seats, `[B, seats]`.
    pub fn seat_values(&self, batch: &InputBatch, seats: i64, train: bool) -> Tensor {
        self.seat_logits(batch, seats, train).sigmoid()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blob_engine::encoder::{encode, encode_value, TOKEN_TYPE_PLAYER};
    use blob_engine::{new_round, RoundParams};
    use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256PlusPlus};
    use tch::{nn::VarStore, Device};

    fn params(vs: &VarStore) -> i64 {
        vs.variables().values().map(|t| t.numel() as i64).sum()
    }

    #[test]
    fn parameter_counts() {
        // P is gen 1's 1.63M net without its value head; V half its depth.
        let p = VarStore::new(Device::Cpu);
        let _ = PolicyNet::new(&p.root());
        assert!((1_550_000..=1_700_000).contains(&params(&p)), "P has {}", params(&p));
        let v = VarStore::new(Device::Cpu);
        let _ = ValueNet::new(&v.root());
        assert!((780_000..=900_000).contains(&params(&v)), "V has {}", params(&v));
    }

    /// Player tokens start at `FIRST_PLAYER_TOKEN`, one per seat, in both modes.
    #[test]
    fn player_tokens_follow_cls_and_context() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(3);
        for n in 3..=8u8 {
            let s = new_round(RoundParams { num_players: n, cards_dealt: 4, trump: 0, dealer: 0 }, &mut rng).unwrap();
            for enc in [encode(&s, 1), encode_value(&s, 1)] {
                let players: Vec<usize> =
                    (0..enc.num_tokens).filter(|&i| enc.token_types[i] == TOKEN_TYPE_PLAYER).collect();
                let first = FIRST_PLAYER_TOKEN as usize;
                assert_eq!(players, (first..first + n as usize).collect::<Vec<_>>());
            }
        }
    }
}
