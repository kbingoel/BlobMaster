//! Bid-aware rule bot — a fixed, search-free baseline opponent.
//!
//! Added 2026-10-02 as an **absolute** strength reference. Gen 1's in-loop
//! evals compared a checkpoint against an earlier checkpoint, which cannot
//! tell "stronger than last week" apart from "strong". This bot never
//! changes, so "score vs 4× rule bot" is comparable across runs.
//!
//! Unlike gen 1's heuristic baseline (which tried to win every trick
//! regardless of its own bid), this bot plays *towards its bid*:
//!
//! - **Bidding**: sum a per-card chance of taking a trick (high trumps, side
//!   aces, guarded side kings/queens, plus a ruffing bonus for a void side
//!   suit backed by ≥ 2 trumps), round to the nearest integer, and snap to
//!   the nearest legal bid (lower bid on ties — respects the dealer rule).
//! - **Playing**: while short of its bid it tries to win (lead its strongest
//!   card; when following, the strongest winning card, or the cheapest one if
//!   it plays last). Once the bid is met it ducks (lead its weakest card;
//!   follow with the strongest card that still loses). When it cannot do
//!   what it wants it sheds its weakest card (short of bid) or, if every
//!   card wins, takes with its strongest card when last and its weakest
//!   otherwise (hoping to be overtaken).
//!
//! "Strength" orders trumps above every side-suit card, then by rank.
//!
//! Measured at 5 players / 7 cards, one focal seat (rotated) vs 4 identical
//! opponents, fresh deals per game:
//!
//! | focal vs 4× …                           | games | pts/game diff | bids made |
//! |-----------------------------------------|-------|---------------|-----------|
//! | rule bot vs random                      | 2000  | +70.4 ± 1.0   | 0.679     |
//! | rule bot vs gen-1 heuristic baseline    | 2000  | +19.2 ± 1.0   | 0.659     |
//! | run-2026-05-14 iter 167, 5×100 MCTS vs rule bot | 320 | −9.2 ± 2.6 | 0.600 |
//!
//! The per-card weights are hand-picked, not tuned, and have only been
//! measured at 5p/7c.
//!
//! Use the functions here directly. Wrapping the bot in a
//! [`PolicyEvaluator`] and running it through `mcts_search` turns it into a
//! different player (one-hot priors plus search), which is what gen 1's
//! eval "heuristic" seats did (gen-2.md §2.8).
//!
//! [`PolicyEvaluator`]: crate::evaluator::PolicyEvaluator

use smallvec::SmallVec;

use crate::bidding::legal_bids;
use crate::card::{NUM_RANKS, NUM_SUITS};
use crate::playing::legal_plays;
use crate::round::NO_TRUMP;
use crate::state::{BlobState, GamePhase};

const ACE: u8 = 12;
const KING: u8 = 11;
const QUEEN: u8 = 10;
const JACK: u8 = 9;

/// Extra expected tricks for a void side suit when holding ≥ 2 trumps.
const VOID_RUFF_BONUS: f32 = 0.4;

/// Chance that a trump of `rank` takes a trick.
fn trump_trick_chance(rank: u8) -> f32 {
    match rank {
        ACE => 1.0,
        KING => 0.85,
        QUEEN => 0.65,
        JACK => 0.45,
        _ => 0.25,
    }
}

/// Chance that a side-suit card of `rank` takes a trick, given how many
/// cards of that suit we hold (long suits get trumped before a king or
/// queen cashes).
fn side_trick_chance(rank: u8, suit_len: u32) -> f32 {
    match rank {
        ACE => 0.8,
        KING if suit_len <= 4 => 0.45,
        KING => 0.2,
        QUEEN if suit_len <= 3 => 0.15,
        QUEEN => 0.05,
        _ => 0.0,
    }
}

/// 13-bit rank mask of `suit` within `hand` (bit `r` ⇔ rank `r` held).
#[inline]
fn suit_bits(hand: u64, suit: u8) -> u64 {
    (hand >> (suit * NUM_RANKS)) & 0x1FFF
}

/// Ordering key: every trump beats every side-suit card, then rank.
#[inline]
fn strength(card: u8, trump: u8) -> u8 {
    let rank = card % NUM_RANKS;
    if trump != NO_TRUMP && card / NUM_RANKS == trump {
        100 + rank
    } else {
        rank
    }
}

/// Estimated tricks for the current player's hand (unrounded).
pub fn expected_tricks(state: &BlobState) -> f32 {
    let hand = state.hands[state.current_player as usize];
    let trump = state.trump_suit;
    let trump_active = trump != NO_TRUMP;
    let trump_len = if trump_active {
        suit_bits(hand, trump).count_ones()
    } else {
        0
    };

    let mut est = 0.0f32;
    for suit in 0..NUM_SUITS {
        let bits = suit_bits(hand, suit);
        let len = bits.count_ones();
        let is_trump = trump_active && suit == trump;
        for rank in 0..NUM_RANKS {
            if (bits >> rank) & 1 == 1 {
                est += if is_trump {
                    trump_trick_chance(rank)
                } else {
                    side_trick_chance(rank, len)
                };
            }
        }
        if !is_trump && len == 0 && trump_len >= 2 {
            est += VOID_RUFF_BONUS;
        }
    }
    est
}

/// Bid for the current player: [`expected_tricks`] rounded, clamped to
/// `0..=cards_dealt`, then snapped to the nearest legal bid (ties → lower).
pub fn rule_bot_bid(state: &BlobState) -> u8 {
    debug_assert_eq!(state.phase(), GamePhase::Bidding);
    let target = expected_tricks(state)
        .round()
        .clamp(0.0, state.cards_dealt as f32) as i32;
    let mask = legal_bids(state);
    let mut best = 0u8;
    let mut best_dist = i32::MAX;
    for b in 0..=state.cards_dealt {
        if (mask >> b) & 1 == 1 {
            let d = (b as i32 - target).abs();
            if d < best_dist {
                best_dist = d;
                best = b;
            }
        }
    }
    best
}

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

/// Card index to play for the current player. See module docs for the rule.
pub fn rule_bot_play(state: &BlobState) -> u8 {
    debug_assert_eq!(state.phase(), GamePhase::Playing);
    let p = state.current_player as usize;
    let trump = state.trump_suit;
    let trump_active = trump != NO_TRUMP;
    let legal = legal_plays(state);

    // Ascending card index, then a stable sort by strength — weakest first.
    let mut cards: SmallVec<[u8; 13]> = (0..52u8).filter(|&c| (legal >> c) & 1 == 1).collect();
    cards.sort_by_key(|&c| strength(c, trump));

    let wants_tricks = state.bids[p] > state.tricks_won[p];
    let plays_last = state.trick_cards_played + 1 == state.num_players;

    let Some((best_rank, best_is_trump, best_suit)) = current_trick_best(state) else {
        // Leading.
        return if wants_tricks { cards[cards.len() - 1] } else { cards[0] };
    };

    let beats = |c: u8| -> bool {
        let suit = c / NUM_RANKS;
        let rank = c % NUM_RANKS;
        let c_is_trump = trump_active && suit == trump;
        if best_is_trump {
            c_is_trump && rank > best_rank
        } else {
            c_is_trump || (suit == best_suit && rank > best_rank)
        }
    };
    let winners: SmallVec<[u8; 13]> = cards.iter().copied().filter(|&c| beats(c)).collect();
    let losers: SmallVec<[u8; 13]> = cards.iter().copied().filter(|&c| !beats(c)).collect();

    if wants_tricks {
        if winners.is_empty() {
            losers[0]
        } else if plays_last {
            winners[0]
        } else {
            winners[winners.len() - 1]
        }
    } else if let Some(&strongest_loser) = losers.last() {
        strongest_loser
    } else if plays_last {
        winners[winners.len() - 1]
    } else {
        winners[0]
    }
}

/// Phase-stable action label for the current player — the bid value in
/// `Bidding`, the card index in `Playing` (same labels as
/// [`crate::mcts::apply_action`]).
///
/// Panics outside a decision phase (`Scoring` / `Complete`).
pub fn rule_bot_action(state: &BlobState) -> u8 {
    match state.phase() {
        GamePhase::Bidding => rule_bot_bid(state),
        GamePhase::Playing => rule_bot_play(state),
        phase => panic!("rule bot asked to act in {phase:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bidding::apply_bid;
    use crate::dealing::start_round;
    use crate::game::{advance_round, is_game_over, new_game};
    use crate::playing::apply_play;
    use crate::state::MAX_PLAYERS;
    use rand::Rng;
    use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256PlusPlus};

    const SPADES: u8 = 0;
    const HEARTS: u8 = 1;
    const CLUBS: u8 = 2;

    fn card(suit: u8, rank: u8) -> u8 {
        suit * NUM_RANKS + rank
    }

    fn bits(cards: &[u8]) -> u64 {
        cards.iter().fold(0u64, |m, &c| m | (1u64 << c))
    }

    /// 5-player playing-phase state with seat 0 to act after `trick` has
    /// been played by the seats before it.
    fn playing_state(trump: u8, hand: &[u8], trick: &[u8], bid: u8, won: u8) -> BlobState {
        let mut s = BlobState::empty();
        s.num_players = 5;
        s.cards_dealt = 7;
        s.trump_suit = trump;
        s.game_phase = GamePhase::Playing as u8;
        s.current_player = 0;
        s.trick_leader = (5 - trick.len() as u8) % 5;
        for (i, &c) in trick.iter().enumerate() {
            s.trick_play_order[i] = c;
        }
        s.trick_cards_played = trick.len() as u8;
        s.hands[0] = bits(hand);
        s.bids[0] = bid;
        s.tricks_won[0] = won;
        s
    }

    #[test]
    fn expected_tricks_counts_honours_and_ruff() {
        // Hearts trump. ♥A ♥K ♠A ♣2, no diamonds → 1.0 + 0.85 + 0.8 + 0 +
        // 0.4 ruff bonus (diamond void with 2 trumps).
        let mut s = BlobState::empty();
        s.num_players = 5;
        s.cards_dealt = 4;
        s.trump_suit = HEARTS;
        s.hands[0] = bits(&[card(HEARTS, ACE), card(HEARTS, KING), card(SPADES, ACE), card(CLUBS, 0)]);
        assert!((expected_tricks(&s) - 3.05).abs() < 1e-5);

        // Same cards in a no-trump round: two side aces + a guarded side
        // king (0.8 + 0.8 + 0.45), no ruff bonus.
        s.trump_suit = NO_TRUMP;
        assert!((expected_tricks(&s) - 2.05).abs() < 1e-5);
    }

    #[test]
    fn leads_strongest_when_short_and_weakest_when_met() {
        let hand = [card(SPADES, ACE), card(HEARTS, 0), card(CLUBS, 5)];
        let s = playing_state(HEARTS, &hand, &[], 2, 0);
        assert_eq!(rule_bot_play(&s), card(HEARTS, 0), "lowest trump outranks a side ace");
        let s = playing_state(HEARTS, &hand, &[], 1, 1);
        assert_eq!(rule_bot_play(&s), card(CLUBS, 5), "weakest side card ducks");
    }

    #[test]
    fn follows_to_win_only_while_short_of_bid() {
        // ♠5 led, no trump. We hold ♠K ♠9 ♠3.
        let hand = [card(SPADES, KING), card(SPADES, 7), card(SPADES, 1)];
        let led = [card(SPADES, 3)];
        // Short of bid, not last: strongest winner.
        let s = playing_state(NO_TRUMP, &hand, &led, 1, 0);
        assert_eq!(rule_bot_play(&s), card(SPADES, KING));
        // Short of bid, last to play: cheapest winner.
        let full = [card(SPADES, 3), card(CLUBS, 0), card(CLUBS, 1), card(CLUBS, 2)];
        let s = playing_state(NO_TRUMP, &hand, &full, 1, 0);
        assert_eq!(rule_bot_play(&s), card(SPADES, 7));
        // Bid met: strongest card that still loses.
        let s = playing_state(NO_TRUMP, &hand, &led, 0, 0);
        assert_eq!(rule_bot_play(&s), card(SPADES, 1));
    }

    fn play_full_game(num_players: u8, start_cards: u8, seed: u64) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut s = new_game(num_players, start_cards).unwrap();
        start_round(&mut s, &mut rng);
        while !is_game_over(&s) {
            match s.phase() {
                GamePhase::Bidding => {
                    let b = rule_bot_action(&s);
                    assert_eq!((legal_bids(&s) >> b) & 1, 1, "illegal bid {b}");
                    apply_bid(&mut s, b);
                }
                GamePhase::Playing => {
                    let c = rule_bot_action(&s);
                    assert_eq!((legal_plays(&s) >> c) & 1, 1, "illegal card {c}");
                    apply_play(&mut s, c);
                }
                GamePhase::Scoring => advance_round(&mut s, &mut rng),
                GamePhase::Complete => unreachable!(),
            }
        }
    }

    #[test]
    fn plays_legal_full_games_at_every_table_size() {
        for n in 3..=8u8 {
            let c = (52 / n).min(8);
            for seed in 0..5 {
                play_full_game(n, c, seed);
            }
        }
    }

    fn random_action(s: &BlobState, rng: &mut Xoshiro256PlusPlus) -> u8 {
        let mask = if s.phase() == GamePhase::Bidding {
            legal_bids(s) as u64
        } else {
            legal_plays(s)
        };
        let legal: SmallVec<[u8; 14]> = (0..64u8).filter(|&a| (mask >> a) & 1 == 1).collect();
        legal[rng.gen_range(0..legal.len())]
    }

    /// Mean (rule-bot score − mean opponent score) over `games` 5p/7c games
    /// against random opponents, rule bot in a rotating seat.
    fn mean_score_edge_vs_random(games: u64) -> f64 {
        let mut total = 0.0;
        for g in 0..games {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB07 + g);
            let seat = (g % 5) as u8;
            let mut s = new_game(5, 7).unwrap();
            start_round(&mut s, &mut rng);
            while !is_game_over(&s) {
                match s.phase() {
                    GamePhase::Bidding | GamePhase::Playing => {
                        let a = if s.current_player == seat {
                            rule_bot_action(&s)
                        } else {
                            random_action(&s, &mut rng)
                        };
                        if s.phase() == GamePhase::Bidding {
                            apply_bid(&mut s, a);
                        } else {
                            apply_play(&mut s, a);
                        }
                    }
                    GamePhase::Scoring => advance_round(&mut s, &mut rng),
                    GamePhase::Complete => unreachable!(),
                }
            }
            let scores: [f64; MAX_PLAYERS] = core::array::from_fn(|i| s.cumulative_scores[i] as f64);
            let others: f64 = (0..5).filter(|&i| i != seat as usize).map(|i| scores[i]).sum::<f64>() / 4.0;
            total += scores[seat as usize] - others;
        }
        total / games as f64
    }

    /// Guards against a regression that silently weakens the baseline.
    /// Full-scale number (2000 games): +70 vs random. `bench rulebot2`
    /// checks the bot against a stronger opponent.
    #[test]
    fn beats_random_baseline() {
        let vs_random = mean_score_edge_vs_random(200);
        assert!(vs_random > 50.0, "edge vs random only {vs_random:.1}");
    }
}
