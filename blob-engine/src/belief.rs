//! Session 4.3 — Belief tracking and determinization for imperfect
//! information MCTS.
//!
//! Blob is imperfect information: each player sees only their own hand.
//! MCTS operates on fully-observable states, so we sample plausible
//! opponent hands ("determinizations") and run an independent tree on
//! each, aggregating the root visit counts at the end (Section 4.3 of
//! the development plan).
//!
//! Belief information is conservative: the only public inference rule
//! used is the suit-void signal (an opponent who didn't follow the led
//! suit is provably void in that suit). Rejection sampling then
//! enforces void constraints when dealing opponent hands.

use rand::seq::SliceRandom;
use rand::Rng;

use crate::card::{NUM_RANKS, NUM_SUITS};
use crate::state::{BlobState, MAX_PLAYERS};

/// Cap on rejection-sampling retries before falling back to an
/// unconstrained deal. With only void constraints (at most
/// `NUM_SUITS * MAX_PLAYERS = 32` boolean flags) 32 attempts is
/// generous — pathological failures tend to indicate contradictory
/// beliefs, in which case the fallback is deliberately permissive.
pub const DEFAULT_DETERMINIZE_ATTEMPTS: u32 = 32;

/// Per-player boolean flags: `void_suits[p][s] == true` iff seat `p`
/// has been observed to be void in suit `s`.
pub type VoidTable = [[bool; NUM_SUITS as usize]; MAX_PLAYERS];

/// Derive each player's void suits from the completed trick history.
///
/// Rule: in slot `i ∈ 1..num_played` of a completed trick, if the card
/// played is not of the led suit, the player is provably void in the
/// led suit (they would have been forced to follow otherwise). Slot 0
/// is the lead itself and reveals nothing.
pub fn void_suits(state: &BlobState) -> VoidTable {
    let mut voids: VoidTable = [[false; NUM_SUITS as usize]; MAX_PLAYERS];
    for t in 0..state.tricks_completed as usize {
        let rec = &state.trick_history[t];
        let led = rec.suit_led as usize;
        for i in 1..rec.num_played as usize {
            let (player, card) = rec.cards[i];
            let card_suit = (card / NUM_RANKS) as usize;
            if card_suit != led {
                voids[player as usize][led] = true;
            }
        }
    }
    voids
}

/// Opponent hand sizes for a determinization: `cards_dealt - tricks_completed`
/// minus 1 for each opponent who has already contributed to the in-progress
/// trick. The perspective seat's slot is set to zero (they keep their real
/// hand untouched by `determinize`).
fn required_hand_sizes(state: &BlobState, perspective: u8) -> [usize; MAX_PLAYERS] {
    let mut out = [0usize; MAX_PLAYERS];
    let base = state.cards_dealt as usize - state.tricks_completed as usize;
    for p in 0..state.num_players as usize {
        if p as u8 == perspective {
            continue;
        }
        let mut contributed = 0usize;
        for j in 0..state.trick_cards_played as usize {
            let player = (state.trick_leader + j as u8) % state.num_players;
            if player as usize == p {
                contributed = 1;
                break;
            }
        }
        out[p] = base.saturating_sub(contributed);
    }
    out
}

/// Sample a determinized `BlobState`: opponents' hands are replaced with
/// a uniformly-random consistent deal. `perspective`'s hand,
/// `played_this_round`, trick history, and all other fields are
/// preserved — this is a cloned state with reshuffled *unseen* cards.
///
/// Void constraints are enforced by rejection sampling; on failure
/// after `max_attempts` the function falls back to an unconstrained
/// deal (preferring a sub-optimal sample over blocking the search).
pub fn determinize<R: Rng + ?Sized>(
    state: &BlobState,
    perspective: u8,
    voids: &VoidTable,
    rng: &mut R,
    max_attempts: u32,
) -> BlobState {
    let mut out = *state;
    let num_players = state.num_players as usize;
    let my_hand = state.hands[perspective as usize];

    // Unseen cards = deck − perspective's hand − cards that have hit the
    // table this round (completed + in-progress). `played_this_round` is
    // maintained incrementally by `apply_play`, so this is exact.
    let deck_mask: u64 = (1u64 << 52) - 1;
    let unseen_mask = deck_mask & !my_hand & !state.played_this_round;
    let unseen_cards: Vec<u8> = (0..52u8).filter(|&c| (unseen_mask >> c) & 1 == 1).collect();

    let required = required_hand_sizes(state, perspective);
    let total_required: usize = required.iter().sum();
    debug_assert!(
        unseen_cards.len() >= total_required,
        "unseen pool ({}) smaller than total required opponent hand size ({total_required})",
        unseen_cards.len()
    );
    // The unseen pool may be larger than the opponents' combined hand
    // size when `num_players * cards_dealt < 52` (undealt cards remain
    // "unseen" from perspective's point of view). We shuffle the whole
    // pool and use only the first `total_required` slots each attempt;
    // the tail represents cards that weren't dealt to anyone.

    // Most-constrained-first ordering: opponents with more voids get
    // first pick to reduce the rejection probability.
    let mut order: Vec<usize> = (0..num_players).filter(|&p| p as u8 != perspective).collect();
    order.sort_by_key(|&p| {
        let voided = voids[p].iter().filter(|v| **v).count();
        std::cmp::Reverse(voided)
    });

    let mut deck = unseen_cards.clone();
    for _ in 0..max_attempts.max(1) {
        deck.shuffle(rng);
        if let Some(new_hands) = try_deal(&deck, &order, &required, voids) {
            out.hands = new_hands_merged(state.hands, &new_hands, perspective);
            return out;
        }
    }

    // Fallback: unconstrained deal in `order` (still excludes perspective).
    deck.shuffle(rng);
    let mut cursor = 0usize;
    let mut fallback = [0u64; MAX_PLAYERS];
    for &p in &order {
        let need = required[p];
        let mut h: u64 = 0;
        for &c in &deck[cursor..cursor + need] {
            h |= 1u64 << c;
        }
        cursor += need;
        fallback[p] = h;
    }
    out.hands = new_hands_merged(state.hands, &fallback, perspective);
    out
}

fn try_deal(
    deck: &[u8],
    order: &[usize],
    required: &[usize; MAX_PLAYERS],
    voids: &VoidTable,
) -> Option<[u64; MAX_PLAYERS]> {
    let mut hands = [0u64; MAX_PLAYERS];
    let mut cursor = 0usize;
    for &p in order {
        let need = required[p];
        let slice = &deck[cursor..cursor + need];
        cursor += need;
        let mut h: u64 = 0;
        for &c in slice {
            let suit = (c / NUM_RANKS) as usize;
            if voids[p][suit] {
                return None;
            }
            h |= 1u64 << c;
        }
        hands[p] = h;
    }
    Some(hands)
}

fn new_hands_merged(
    original: [u64; MAX_PLAYERS],
    opponents: &[u64; MAX_PLAYERS],
    perspective: u8,
) -> [u64; MAX_PLAYERS] {
    let mut out = [0u64; MAX_PLAYERS];
    for i in 0..MAX_PLAYERS {
        out[i] = if i == perspective as usize {
            original[i]
        } else {
            opponents[i]
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bidding::{apply_bid, legal_bids};
    use crate::dealing::deal;
    use crate::game::new_game;
    use crate::playing::apply_play;
    use crate::state::GamePhase;
    use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256PlusPlus};

    fn mid_playing_state(seed: u64) -> (BlobState, Xoshiro256PlusPlus) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut s = new_game(4, 5).unwrap();
        deal(&mut s, &mut rng);
        while s.game_phase == GamePhase::Bidding as u8 {
            let mask = legal_bids(&s);
            let b = mask.trailing_zeros() as u8;
            apply_bid(&mut s, b);
        }
        (s, rng)
    }

    #[test]
    fn void_suits_detected_when_player_skips_led_suit() {
        // Construct: 3 players, 1 card each. Player 0 leads hearts (suit 2),
        // player 1 discards a club (suit 0) — proves void in hearts.
        let mut s = BlobState::empty();
        s.num_players = 3;
        s.cards_dealt = 1;
        s.tricks_completed = 1;
        s.trick_history[0] = crate::state::TrickRecord {
            // (player, card): card = suit * 13 + rank
            cards: [
                (0, 2 * NUM_RANKS + 5), // P0 leads hearts (suit 2)
                (1, 0 * NUM_RANKS + 3), // P1 plays clubs → void in hearts
                (2, 2 * NUM_RANKS + 7), // P2 follows hearts
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
            ],
            num_played: 3,
            winner: 2,
            suit_led: 2,
        };

        let voids = void_suits(&s);
        assert!(voids[1][2], "player 1 must be void in hearts");
        assert!(!voids[1][0], "player 1 not void in clubs");
        assert!(!voids[2][2], "player 2 followed → no void");
        assert!(!voids[0][2], "player 0 led → no void inferred");
    }

    #[test]
    fn determinize_preserves_perspective_hand_and_card_totals() {
        let (s, mut rng) = mid_playing_state(17);
        let perspective = s.current_player;
        let voids = void_suits(&s);
        let d = determinize(&s, perspective, &voids, &mut rng, DEFAULT_DETERMINIZE_ATTEMPTS);

        // Perspective untouched.
        assert_eq!(d.hands[perspective as usize], s.hands[perspective as usize]);
        // Hand sizes correct.
        let required = required_hand_sizes(&s, perspective);
        for p in 0..s.num_players as usize {
            if p as u8 == perspective {
                continue;
            }
            assert_eq!(d.hands[p].count_ones() as usize, required[p]);
        }
        // No duplicate cards across all hands.
        let mut union: u64 = 0;
        let mut popsum = 0u32;
        for p in 0..s.num_players as usize {
            union |= d.hands[p];
            popsum += d.hands[p].count_ones();
        }
        assert_eq!(union.count_ones(), popsum, "no card appears in two hands");
        // Hands disjoint from played_this_round.
        assert_eq!(union & s.played_this_round, 0);
    }

    #[test]
    fn determinize_respects_void_constraints() {
        let (s, mut rng) = mid_playing_state(23);
        let perspective = s.current_player;
        // Forge a void: opponent `t` is void in suit 0.
        let opp = (perspective + 1) % s.num_players;
        let mut voids = void_suits(&s);
        voids[opp as usize][0] = true;

        let suit0_mask: u64 = 0x1FFFu64; // ranks 0..13 of suit 0

        for _ in 0..50 {
            let d = determinize(&s, perspective, &voids, &mut rng, DEFAULT_DETERMINIZE_ATTEMPTS);
            assert_eq!(
                d.hands[opp as usize] & suit0_mask,
                0,
                "voided player holds no suit-0 card"
            );
        }
    }

    #[test]
    fn determinize_bidding_phase_distributes_full_deck() {
        // In bidding phase played_this_round == 0, so all 52 cards are
        // distributed (num_players * cards_dealt must fit; 4×5 = 20).
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let mut s = new_game(4, 5).unwrap();
        deal(&mut s, &mut rng);
        let perspective = s.current_player;
        let voids = void_suits(&s);

        let d = determinize(&s, perspective, &voids, &mut rng, 8);
        for p in 0..s.num_players as usize {
            assert_eq!(d.hands[p].count_ones(), 5);
        }
    }

    #[test]
    fn determinize_handles_in_progress_trick() {
        // Advance one player's play so the in-progress trick has 1 card.
        let (mut s, mut rng) = mid_playing_state(29);
        let first_legal = crate::playing::legal_plays(&s).trailing_zeros() as u8;
        apply_play(&mut s, first_legal);
        assert_eq!(s.trick_cards_played, 1);

        let perspective = s.current_player;
        let voids = void_suits(&s);
        let d = determinize(&s, perspective, &voids, &mut rng, DEFAULT_DETERMINIZE_ATTEMPTS);

        // The seat that just played should now have cards_dealt - 1 cards
        // (they've committed one to the in-progress trick).
        let leader = s.trick_leader as usize;
        assert_eq!(
            d.hands[leader].count_ones() as usize,
            s.cards_dealt as usize - s.tricks_completed as usize - 1
        );
    }
}
