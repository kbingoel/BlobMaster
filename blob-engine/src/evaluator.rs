//! Session 3.5 — `Evaluator` trait and dummy/ONNX implementations.
//!
//! MCTS (Sections 4.2 / 5.2) consumes `Evaluator` trait objects so that the
//! tree search code is agnostic to the inference backend. Implementations
//! own their encoder calls; callers only supply a `BlobState`.
//!
//! Policy vector semantics depend on `state.game_phase`:
//! - `Bidding`: length `NUM_BIDS` (14), probabilities over bids 0..=13.
//! - `Playing`: length `hand_card_indices.len()`, per-hand-card-position
//!   scores in `Hand::iter()` order (same mapping used by
//!   `EncodedState::hand_card_indices`). **NOT** indexed by card index.

use crate::bidding::legal_bids;
use crate::card::{NUM_RANKS, NUM_SUITS};
use crate::encoder::encode;
use crate::hand::Hand;
use crate::playing::legal_plays;
use crate::round::NO_TRUMP;
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
}

/// Dummy evaluator: uniform over legal actions, value = 0.0.
///
/// Used for MCTS correctness tests (Section 4) before a trained model
/// exists, and as a baseline in comparison eval (Section 6).
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
                let enc = encode(state, state.current_player);
                let hand = Hand::new(state.hands[state.current_player as usize]);
                let legal = legal_plays(state);
                let mut policy = vec![0.0f32; enc.hand_card_indices.len()];
                let mut n_legal = 0u32;
                for (i, card_idx) in enc.hand_card_indices.iter().enumerate() {
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
                let _ = hand;
                (policy, 0.0)
            }
            GamePhase::Scoring | GamePhase::Complete => (Vec::new(), 0.0),
        }
    }
}

/// Heuristic baseline evaluator (Session 6.1).
///
/// Bid rule: `raw = count(aces in trump suit) + count(kings in trump suit)
/// + count(aces in non-trump suits)`. In NoTrump rounds the trump-suit
/// terms vanish, collapsing to `count(aces in hand)`. The raw value is
/// projected onto the legal-bid mask (nearest legal, rounding down on
/// ties). Policy output is one-hot over `NUM_BIDS` at the chosen bid.
///
/// Play rule: if no card has been played yet, or no legal card can beat
/// the currently-winning card in the trick, play the lowest legal card.
/// Otherwise play the lowest legal card that beats the current winner.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeuristicEvaluator;

fn current_trick_best(state: &BlobState) -> Option<(u8, bool, u8)> {
    // Returns (best_rank, best_is_trump, best_suit) across already-played
    // cards in the in-progress trick, or None if no cards played yet.
    if state.trick_cards_played == 0 {
        return None;
    }
    let trump = state.trump_suit;
    let trump_active = trump != NO_TRUMP;
    let lead = state.trick_play_order[0];
    let suit_led = lead / NUM_RANKS;
    let mut best_rank = lead % NUM_RANKS;
    let mut best_is_trump = trump_active && suit_led == trump;
    let mut best_suit = suit_led;
    for i in 1..state.trick_cards_played as usize {
        let c = state.trick_play_order[i];
        let c_suit = c / NUM_RANKS;
        let c_rank = c % NUM_RANKS;
        let c_is_trump = trump_active && c_suit == trump;
        let takes = if best_is_trump {
            c_is_trump && c_rank > best_rank
        } else if c_is_trump {
            true
        } else {
            c_suit == suit_led && c_rank > best_rank
        };
        if takes {
            best_rank = c_rank;
            best_is_trump = c_is_trump;
            best_suit = c_suit;
        }
    }
    Some((best_rank, best_is_trump, best_suit))
}

impl Evaluator for HeuristicEvaluator {
    fn evaluate(&self, state: &BlobState) -> (Vec<f32>, f32) {
        let phase = GamePhase::from_u8(state.game_phase).unwrap_or(GamePhase::Scoring);
        match phase {
            GamePhase::Bidding => {
                let hand = state.hands[state.current_player as usize];
                let trump = state.trump_suit;
                let trump_active = trump != NO_TRUMP;
                let mut raw: i32 = 0;
                for s in 0..NUM_SUITS {
                    let ace_bit = 1u64 << (s * NUM_RANKS + NUM_RANKS - 1);
                    if (hand & ace_bit) != 0 {
                        // Aces always count (trump or not; NoTrump formula collapses cleanly).
                        raw += 1;
                    }
                }
                if trump_active {
                    let king_bit = 1u64 << (trump * NUM_RANKS + NUM_RANKS - 2);
                    if (hand & king_bit) != 0 {
                        raw += 1;
                    }
                }
                let mask = legal_bids(state);
                let mut policy = vec![0.0f32; NUM_BIDS];
                // Nearest legal, rounding down on ties: iterate ascending
                // and keep strict-less-than on distance.
                let mut best_bid: Option<u8> = None;
                let mut best_dist = i32::MAX;
                for b in 0..NUM_BIDS as u8 {
                    if (mask >> b) & 1 == 1 {
                        let d = (raw - b as i32).abs();
                        if d < best_dist {
                            best_dist = d;
                            best_bid = Some(b);
                        }
                    }
                }
                if let Some(b) = best_bid {
                    policy[b as usize] = 1.0;
                }
                (policy, 0.0)
            }
            GamePhase::Playing => {
                let enc = encode(state, state.current_player);
                let legal = legal_plays(state);
                let best = current_trick_best(state);
                let trump = state.trump_suit;
                let trump_active = trump != NO_TRUMP;

                let beats = |c: u8| -> bool {
                    let Some((br, bt, bs)) = best else {
                        return false;
                    };
                    let suit = c / NUM_RANKS;
                    let rank = c % NUM_RANKS;
                    let c_is_trump = trump_active && suit == trump;
                    if bt {
                        c_is_trump && rank > br
                    } else {
                        c_is_trump || (suit == bs && rank > br)
                    }
                };

                let mut chosen: Option<usize> = None;
                let mut chosen_rank: i32 = i32::MAX;
                // Prefer lowest legal that beats current winner.
                for (pos, &c) in enc.hand_card_indices.iter().enumerate() {
                    if (legal >> c) & 1 != 1 {
                        continue;
                    }
                    if beats(c) {
                        let r = (c % NUM_RANKS) as i32;
                        if r < chosen_rank {
                            chosen_rank = r;
                            chosen = Some(pos);
                        }
                    }
                }
                // Fallback: lowest legal card overall.
                if chosen.is_none() {
                    let mut lo: i32 = i32::MAX;
                    for (pos, &c) in enc.hand_card_indices.iter().enumerate() {
                        if (legal >> c) & 1 != 1 {
                            continue;
                        }
                        let r = (c % NUM_RANKS) as i32;
                        if r < lo {
                            lo = r;
                            chosen = Some(pos);
                        }
                    }
                }
                let mut policy = vec![0.0f32; enc.hand_card_indices.len()];
                if let Some(p) = chosen {
                    policy[p] = 1.0;
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

    #[test]
    fn heuristic_bid_is_onehot_over_legal() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let mut s = new_game(4, 7).unwrap();
        deal(&mut s, &mut rng);
        let (policy, value) = HeuristicEvaluator.evaluate(&s);
        assert_eq!(policy.len(), NUM_BIDS);
        assert_eq!(value, 0.0);
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
        let mask = legal_bids(&s);
        for (b, &p) in policy.iter().enumerate() {
            if p > 0.0 {
                assert_eq!((mask >> b) & 1, 1, "bid {b} not legal");
            }
        }
    }

    #[test]
    fn heuristic_play_chooses_legal_card() {
        use crate::bidding::apply_bid;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(11);
        let mut s = new_game(4, 7).unwrap();
        deal(&mut s, &mut rng);
        while s.game_phase == GamePhase::Bidding as u8 {
            let mask = legal_bids(&s);
            let b = mask.trailing_zeros() as u8;
            apply_bid(&mut s, b);
        }
        let enc = encode(&s, s.current_player);
        let (policy, _) = HeuristicEvaluator.evaluate(&s);
        assert_eq!(policy.len(), enc.hand_card_indices.len());
        let nonzero: Vec<usize> = policy
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(nonzero.len(), 1);
        let legal = legal_plays(&s);
        let card = enc.hand_card_indices[nonzero[0]];
        assert_eq!((legal >> card) & 1, 1, "chosen card illegal");
    }
}
