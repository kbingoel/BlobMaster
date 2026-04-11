//! Round dealing primitives.
//!
//! [`deal`] shuffles a 52-card deck (Fisher-Yates via `rand::seq::SliceRandom`)
//! and writes per-player `u64` bitmasks into `BlobState.hands`. [`start_round`]
//! is the higher-level helper used by `new_game`/`advance_round` (Section 1.4):
//! it clears all per-round state, deals fresh hands, and primes the bidding
//! phase with `current_player = (dealer + 1) % num_players`.
//!
//! RNG ownership lives **outside** `BlobState` (see `state.rs`) so the state
//! itself stays `Copy` and per-thread self-play seeding remains deterministic.

use rand::seq::SliceRandom;
use rand::Rng;

use crate::card::NUM_CARDS;
use crate::state::{BlobState, GamePhase, TrickRecord, MAX_PLAYERS};

/// Shuffle a 52-card deck and distribute `state.cards_dealt` cards to each
/// of the first `state.num_players` players. Hands are written as `u64`
/// bitmasks; previous hand contents are overwritten.
///
/// Other round state (`bids`, `played_this_round`, etc.) is **not** touched —
/// see [`start_round`] for the all-in-one reset+deal helper.
///
/// Panics in debug if `num_players × cards_dealt > 52`.
pub fn deal<R: Rng + ?Sized>(state: &mut BlobState, rng: &mut R) {
    let n = state.num_players as usize;
    let cd = state.cards_dealt as usize;
    debug_assert!(
        n * cd <= NUM_CARDS as usize,
        "deck exceeded: {n} players × {cd} cards > 52"
    );

    // Shuffled deck of card indices (0..52).
    let mut deck = [0u8; NUM_CARDS as usize];
    for (i, slot) in deck.iter_mut().enumerate() {
        *slot = i as u8;
    }
    deck.shuffle(rng);

    // Reset all hand slots, then fill the first `n`. Block layout: player p
    // gets `deck[p*cd..(p+1)*cd]`. Pattern is uniform after shuffling, so the
    // exact stride doesn't affect randomness — block is simpler than the
    // legacy round-robin pop-from-front.
    state.hands = [0; MAX_PLAYERS];
    for p in 0..n {
        let mut h: u64 = 0;
        for k in 0..cd {
            h |= 1u64 << deck[p * cd + k];
        }
        state.hands[p] = h;
    }
}

/// Reset per-round state, deal new hands, and prime the bidding phase.
///
/// The caller must have set `num_players`, `cards_dealt`, `dealer`, and
/// `trump_suit` on `state` before calling. After this returns:
///
/// - all `bids` and `tricks_won` are zero,
/// - `played_this_round` is empty and `trick_history` is cleared,
/// - `current_player == trick_leader == (dealer + 1) % num_players`,
/// - `game_phase == GamePhase::Bidding`,
/// - each of the first `num_players` players holds `cards_dealt` cards.
pub fn start_round<R: Rng + ?Sized>(state: &mut BlobState, rng: &mut R) {
    state.bids = [0; MAX_PLAYERS];
    state.tricks_won = [0; MAX_PLAYERS];
    state.played_this_round = 0;
    state.trick_history.fill(TrickRecord::default());
    state.trick_play_order = [0; MAX_PLAYERS];
    state.trick_cards_played = 0;
    state.tricks_completed = 0;

    let first = (state.dealer + 1) % state.num_players;
    state.trick_leader = first;
    state.current_player = first;
    state.game_phase = GamePhase::Bidding as u8;

    deal(state, rng);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::NUM_CARDS;
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn fresh_state(num_players: u8, cards_dealt: u8, dealer: u8) -> BlobState {
        let mut s = BlobState::empty();
        s.num_players = num_players;
        s.cards_dealt = cards_dealt;
        s.dealer = dealer;
        s
    }

    #[test]
    fn deal_distributes_correct_card_counts() {
        let mut s = fresh_state(4, 5, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        deal(&mut s, &mut rng);
        for p in 0..4 {
            assert_eq!(s.hands[p].count_ones(), 5, "player {p}");
        }
        // Other slots untouched (zero).
        for p in 4..MAX_PLAYERS {
            assert_eq!(s.hands[p], 0);
        }
    }

    #[test]
    fn deal_uses_disjoint_cards_no_duplicates() {
        let mut s = fresh_state(5, 7, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        deal(&mut s, &mut rng);
        let union: u64 = s.hands.iter().fold(0, |acc, &h| acc | h);
        let intersection_total: u32 = s.hands.iter().map(|h| h.count_ones()).sum();
        // No card appears twice ⇒ union popcount equals sum of popcounts.
        assert_eq!(union.count_ones(), intersection_total);
        // Total dealt = 5 × 7 = 35.
        assert_eq!(union.count_ones(), 35);
        // All dealt cards are within the 52-card range.
        assert_eq!(union & !((1u64 << NUM_CARDS) - 1), 0);
    }

    #[test]
    fn deal_full_deck_4p_13c() {
        let mut s = fresh_state(4, 13, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        deal(&mut s, &mut rng);
        let union: u64 = s.hands.iter().fold(0, |acc, &h| acc | h);
        assert_eq!(union, (1u64 << 52) - 1, "full deck must be dealt");
        for p in 0..4 {
            assert_eq!(s.hands[p].count_ones(), 13);
        }
    }

    #[test]
    fn deal_is_deterministic_for_same_seed() {
        let mut a = fresh_state(5, 7, 0);
        let mut b = fresh_state(5, 7, 0);
        let mut rng_a = Xoshiro256PlusPlus::seed_from_u64(12345);
        let mut rng_b = Xoshiro256PlusPlus::seed_from_u64(12345);
        deal(&mut a, &mut rng_a);
        deal(&mut b, &mut rng_b);
        assert_eq!(a.hands, b.hands);
    }

    #[test]
    fn deal_overwrites_previous_hands() {
        let mut s = fresh_state(4, 5, 0);
        // Pre-seed garbage.
        s.hands[0] = 0xFFFF_FFFF_FFFF_FFFF;
        s.hands[7] = 0xDEAD_BEEF;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(99);
        deal(&mut s, &mut rng);
        // Player 0's count is the dealt count, not the garbage popcount.
        assert_eq!(s.hands[0].count_ones(), 5);
        // Slot beyond num_players is cleared too.
        assert_eq!(s.hands[7], 0);
    }

    #[test]
    fn start_round_resets_state_and_advances_to_bidding() {
        let mut s = fresh_state(4, 5, 1);
        // Dirty up state to make sure start_round clears it.
        s.bids = [3; MAX_PLAYERS];
        s.tricks_won = [2; MAX_PLAYERS];
        s.played_this_round = 0xDEAD_BEEF;
        s.trick_cards_played = 4;
        s.tricks_completed = 7;
        s.game_phase = GamePhase::Scoring as u8;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        start_round(&mut s, &mut rng);

        assert_eq!(s.bids, [0; MAX_PLAYERS]);
        assert_eq!(s.tricks_won, [0; MAX_PLAYERS]);
        assert_eq!(s.played_this_round, 0);
        assert_eq!(s.trick_cards_played, 0);
        assert_eq!(s.tricks_completed, 0);
        assert_eq!(s.phase(), GamePhase::Bidding);
        assert_eq!(s.current_player, 2, "player left of dealer (1+1) bids first");
        assert_eq!(s.trick_leader, 2);
        for p in 0..4 {
            assert_eq!(s.hands[p].count_ones(), 5);
        }
    }

    #[test]
    fn start_round_first_bidder_wraps_around() {
        let mut s = fresh_state(4, 5, 3);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        start_round(&mut s, &mut rng);
        // dealer = 3 (last seat), so first bidder wraps to 0.
        assert_eq!(s.current_player, 0);
        assert_eq!(s.trick_leader, 0);
    }
}
