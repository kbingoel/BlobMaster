//! `Evaluator` trait and the dummy implementation (the ONNX one is in
//! `onnx.rs`).
//!
//! MCTS consumes `Evaluator` trait objects so that the
//! tree search code is agnostic to the inference backend. Implementations
//! own their encoder calls; callers only supply a `BlobState`.
//!
//! Policy vector semantics depend on `state.game_phase`:
//! - `Bidding`: length `NUM_BIDS` (14), probabilities over bids 0..=13.
//! - `Playing`: length `hand_card_indices.len()`, per-hand-card-position
//!   scores in `Hand::iter()` order (same mapping used by
//!   `EncodedState::hand_card_indices`). **NOT** indexed by card index.

use crate::bidding::legal_bids;
use crate::encoder::hand_card_indices;
use crate::playing::legal_plays;
use crate::state::{BlobState, GamePhase};

/// Number of possible bid values (0..=13 inclusive).
pub const NUM_BIDS: usize = 14;

/// Shared inference interface. Returns `(policy, value)`.
///
/// - `policy`: masked, renormalized probability distribution. Illegal
///   actions are zero. See module docs for length semantics.
/// - `value`: scalar in `[-1, 1]` from the perspective of
///   `state.current_player`.
pub trait Evaluator: Send + Sync {
    fn evaluate(&self, state: &BlobState) -> (Vec<f32>, f32);

    /// Batched inference. Returns one `(policy, value)` per input state, in
    /// the same order. Default implementation loops `evaluate`, which is
    /// correct but does not amortize per-call overhead — production
    /// implementations (`OnnxEvaluator`) override this with a single
    /// `sess.run` over a `[B, S_max, FEAT_DIM]` zero-padded tensor.
    ///
    /// Lockstep MCTS across determinizations calls
    /// this once per "step" with B = number of dets currently producing a
    /// non-terminal leaf. Terminal leaves are filtered by the caller; this
    /// method must not be called with a state in `Scoring`/`Complete`.
    fn evaluate_batch(&self, states: &[&BlobState]) -> Vec<(Vec<f32>, f32)> {
        states.iter().map(|s| self.evaluate(s)).collect()
    }
}

/// Dummy evaluator: uniform over legal actions, value = 0.0.
///
/// Used for MCTS correctness tests.
#[derive(Debug, Clone, Copy, Default)]
pub struct DummyEvaluator;

impl Evaluator for DummyEvaluator {
    fn evaluate(&self, state: &BlobState) -> (Vec<f32>, f32) {
        let phase = GamePhase::from_u8(state.game_phase).unwrap_or(GamePhase::Scoring);
        match phase {
            GamePhase::Bidding => {
                let mask = legal_bids(state);
                let n = mask.count_ones();
                let mut policy = vec![0.0f32; NUM_BIDS];
                if n > 0 {
                    let p = 1.0 / n as f32;
                    for b in 0..NUM_BIDS {
                        if (mask >> b) & 1 == 1 {
                            policy[b] = p;
                        }
                    }
                }
                (policy, 0.0)
            }
            GamePhase::Playing => {
                let hand = hand_card_indices(state, state.current_player);
                let legal = legal_plays(state);
                let mut policy = vec![0.0f32; hand.len()];
                let mut n_legal = 0u32;
                for (i, card_idx) in hand.iter().enumerate() {
                    if (legal >> *card_idx) & 1 == 1 {
                        policy[i] = 1.0;
                        n_legal += 1;
                    }
                }
                if n_legal > 0 {
                    let p = 1.0 / n_legal as f32;
                    for v in policy.iter_mut() {
                        if *v > 0.0 {
                            *v = p;
                        }
                    }
                }
                (policy, 0.0)
            }
            GamePhase::Scoring | GamePhase::Complete => (Vec::new(), 0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dealing::deal;
    use crate::encoder::encode;
    use crate::game::new_game;
    use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256PlusPlus};

    #[test]
    fn dummy_bidding_uniform_over_legal() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let mut s = new_game(4, 5).unwrap();
        deal(&mut s, &mut rng);
        assert_eq!(s.game_phase, GamePhase::Bidding as u8);

        let (policy, value) = DummyEvaluator.evaluate(&s);
        assert_eq!(policy.len(), NUM_BIDS);
        assert_eq!(value, 0.0);
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");

        let mask = legal_bids(&s);
        for b in 0..NUM_BIDS {
            let legal = (mask >> b) & 1 == 1;
            if legal {
                assert!(policy[b] > 0.0);
            } else {
                assert_eq!(policy[b], 0.0);
            }
        }
    }

    #[test]
    fn dummy_playing_uniform_over_hand_positions() {
        use crate::bidding::apply_bid;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2);
        let mut s = new_game(4, 5).unwrap();
        deal(&mut s, &mut rng);
        // Drive through bidding to reach playing phase.
        while s.game_phase == GamePhase::Bidding as u8 {
            let mask = legal_bids(&s);
            let b = mask.trailing_zeros() as u8;
            apply_bid(&mut s, b);
        }
        assert_eq!(s.game_phase, GamePhase::Playing as u8);

        let enc = encode(&s, s.current_player);
        let (policy, value) = DummyEvaluator.evaluate(&s);
        assert_eq!(policy.len(), enc.hand_card_indices.len());
        assert_eq!(value, 0.0);
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
    }
}
