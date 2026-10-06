//! The two network interfaces search consumes (gen-2.md §5.3), and a dummy
//! implementation of both for tests. The ONNX implementations are in
//! `onnx.rs`.
//!
//! - [`PolicyEvaluator`] (P): move priors for the seat to move, from that
//!   seat's own view. Used at every expanded node and as the fast no-search
//!   player.
//! - [`ValueEvaluator`] (V): the expected normalized round score ŝ of every
//!   seat (`scoring.rs`), from a fully known deal. Search calls it on
//!   sampled deals, never on the real hidden cards.
//!
//! Implementations own their encoder calls; callers only supply a
//! `BlobState`. Both take batches: lockstep search evaluates one leaf per
//! sampled deal per step.
//!
//! Policy vector semantics depend on `state.game_phase`:
//! - `Bidding`: length `NUM_BIDS` (14), probabilities over bids 0..=13.
//! - `Playing`: length `hand_card_indices.len()`, per-hand-card-position
//!   probabilities in `Hand::iter()` order (same mapping as
//!   `EncodedState::hand_card_indices`). **NOT** indexed by card index.

use crate::bidding::legal_bids;
use crate::encoder::hand_card_indices;
use crate::playing::legal_plays;
use crate::state::{BlobState, GamePhase, MAX_PLAYERS};

/// Number of possible bid values (0..=13 inclusive).
pub const NUM_BIDS: usize = 14;

/// Move priors for `state.current_player`.
///
/// The policy is masked and renormalized: illegal actions are zero. See the
/// module docs for its length and indexing. Must not be called in
/// `Scoring` or `Complete`.
pub trait PolicyEvaluator: Send + Sync {
    fn policy(&self, state: &BlobState) -> Vec<f32>;

    /// One policy per state, in order. The default loops [`Self::policy`];
    /// the ONNX implementation runs one padded batch.
    fn policy_batch(&self, states: &[&BlobState]) -> Vec<Vec<f32>> {
        states.iter().map(|s| self.policy(s)).collect()
    }
}

/// Largest batch [`policy_in_chunks`] sends at once.
const POLICY_CHUNK: usize = 256;

/// [`PolicyEvaluator::policy_batch`] over any number of states, in batches
/// of at most 256. For the bid likelihoods of sampled deals, which can run
/// to a thousand states per decision.
pub fn policy_in_chunks<P: PolicyEvaluator + ?Sized>(policy: &P, states: &[BlobState]) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(states.len());
    for chunk in states.chunks(POLICY_CHUNK) {
        let refs: Vec<&BlobState> = chunk.iter().collect();
        out.extend(policy.policy_batch(&refs));
    }
    out
}

/// Expected ŝ (round points / (10 + cards dealt), in [0, 1]) of every seat,
/// indexed by absolute seat; slots `>= num_players` are 0.
///
/// Reads every hand of `state`. Must not be called in `Scoring` or
/// `Complete`: search scores finished rounds exactly.
pub trait ValueEvaluator: Send + Sync {
    fn values(&self, state: &BlobState) -> [f32; MAX_PLAYERS];

    /// One value vector per state, in order. The default loops
    /// [`Self::values`]; the ONNX implementation runs one padded batch.
    fn values_batch(&self, states: &[&BlobState]) -> Vec<[f32; MAX_PLAYERS]> {
        states.iter().map(|s| self.values(s)).collect()
    }
}

/// Dummy evaluator for search tests: uniform over legal actions, and ŝ = 0
/// for every seat (utility 0).
#[derive(Debug, Clone, Copy, Default)]
pub struct DummyEvaluator;

/// Uniform distribution over the legal actions of `state`, in the policy
/// layout of the module docs. Empty outside bidding and playing.
pub fn uniform_policy(state: &BlobState) -> Vec<f32> {
    match state.phase() {
        GamePhase::Bidding => {
            let mask = legal_bids(state);
            let p = 1.0 / mask.count_ones().max(1) as f32;
            (0..NUM_BIDS).map(|b| if (mask >> b) & 1 == 1 { p } else { 0.0 }).collect()
        }
        GamePhase::Playing => {
            let legal = legal_plays(state);
            let p = 1.0 / legal.count_ones().max(1) as f32;
            hand_card_indices(state, state.current_player)
                .iter()
                .map(|&c| if (legal >> c) & 1 == 1 { p } else { 0.0 })
                .collect()
        }
        GamePhase::Scoring | GamePhase::Complete => Vec::new(),
    }
}

impl PolicyEvaluator for DummyEvaluator {
    fn policy(&self, state: &BlobState) -> Vec<f32> {
        uniform_policy(state)
    }
}

impl ValueEvaluator for DummyEvaluator {
    fn values(&self, _state: &BlobState) -> [f32; MAX_PLAYERS] {
        [0.0; MAX_PLAYERS]
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

        let policy = DummyEvaluator.policy(&s);
        assert_eq!(policy.len(), NUM_BIDS);
        assert_eq!(DummyEvaluator.values(&s), [0.0; MAX_PLAYERS]);
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
        let policy = DummyEvaluator.policy(&s);
        assert_eq!(policy.len(), enc.hand_card_indices.len());
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
        let batch = DummyEvaluator.policy_batch(&[&s, &s]);
        assert_eq!(batch, vec![policy.clone(), policy]);
    }
}
