//! Composite BlobNet model — input projection + Transformer + three heads.
//!
//! This is a thin owning wrapper around the building blocks from Sessions
//! 3.1–3.3 so that training code (Session 3.4) has a single object to pass
//! around. Parameters are registered into one `VarStore`, which the caller
//! owns and passes to the optimizer.

use tch::{nn, Tensor};

use crate::heads::{BiddingHead, PlayingHead, ValueHead};
use crate::input::{InputBatch, InputProjection};
use crate::transformer::TransformerEncoder;

/// Full network. Hold this alongside the `VarStore` it was built against.
#[derive(Debug)]
pub struct BlobNet {
    pub input: InputProjection,
    pub transformer: TransformerEncoder,
    pub play_head: PlayingHead,
    pub bid_head: BiddingHead,
    pub value_head: ValueHead,
}

impl BlobNet {
    /// Build a fresh model under `vs`. Convention: sub-modules are placed
    /// under `input/`, `transformer/`, `play_head/`, `bid_head/`,
    /// `value_head/` so parameter names are stable across runs and
    /// checkpoints round-trip.
    pub fn new(vs: &nn::Path) -> Self {
        Self {
            input: InputProjection::new(&(vs / "input")),
            transformer: TransformerEncoder::new(&(vs / "transformer")),
            play_head: PlayingHead::new(&(vs / "play_head")),
            bid_head: BiddingHead::new(&(vs / "bid_head")),
            value_head: ValueHead::new(&(vs / "value_head")),
        }
    }

    /// Run input + transformer stages, returning `[B, S, 128]`.
    pub fn encode(&self, batch: &InputBatch, train: bool) -> Tensor {
        let x = self.input.forward(
            &batch.features,
            &batch.token_types,
            &batch.chrono_indices,
            &batch.attention_mask,
        );
        self.transformer.forward(&x, &batch.attention_mask, train)
    }

    /// Playing forward: `(policy_probs [B, S], value [B])`.
    /// `play_legal_mask: [B, S]` — true only at hand-card tokens with legal plays.
    pub fn forward_play(
        &self,
        batch: &InputBatch,
        play_legal_mask: &Tensor,
        train: bool,
    ) -> (Tensor, Tensor) {
        let h = self.encode(batch, train);
        let policy = self.play_head.forward(&h, play_legal_mask);
        let value = self.value_head.forward(&h, train);
        (policy, value)
    }

    /// Bidding forward: `(policy_probs [B, 14], value [B])`.
    pub fn forward_bid(
        &self,
        batch: &InputBatch,
        legal_bid_mask: &Tensor,
        train: bool,
    ) -> (Tensor, Tensor) {
        let h = self.encode(batch, train);
        let policy = self.bid_head.forward(&h, legal_bid_mask, train);
        let value = self.value_head.forward(&h, train);
        (policy, value)
    }
}
