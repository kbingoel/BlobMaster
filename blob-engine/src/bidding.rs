//! Bidding-phase rules: legal-bid generation and bid application.
//!
//! Bidding order is `(dealer + 1) → (dealer + 2) → … → dealer`, so the dealer
//! is **always** last to bid. The dealer is forbidden from a bid that would
//! make the running total of bids equal `cards_dealt` — at least one player
//! must miss their bid each round.
//!
//! `legal_bids` returns a `u16` bitmask (bit `b` set ⇔ bid `b` is legal),
//! avoiding the heap allocation that a `Vec<u8>` return would force.

use crate::card::MAX_CARDS_DEALT;
use crate::state::{BlobState, GamePhase};

/// Bitmask of legal bid values for the player whose turn it is.
///
/// Returns `0` outside the bidding phase. Otherwise returns bits `0..=cards_dealt`,
/// with the dealer's forbidden value cleared (when one exists in range).
pub fn legal_bids(state: &BlobState) -> u16 {
    if state.phase() != GamePhase::Bidding {
        return 0;
    }
    let cards = state.cards_dealt as u16;
    debug_assert!((cards as usize) <= MAX_CARDS_DEALT);

    // Bits 0..=cards_dealt set.
    let mut mask: u16 = ((1u32 << (cards + 1)) - 1) as u16;

    if state.current_player == state.dealer {
        if let Some(forbidden) = forbidden_bid(state) {
            mask &= !(1u16 << forbidden);
        }
    }
    mask
}

/// The dealer's forbidden bid value, or `None` if it falls outside `[0, cards_dealt]`.
///
/// Computed as `cards_dealt − sum(bids of all non-dealer players)`. By the time
/// the dealer is up, every other seat has already bid (bidding order puts the
/// dealer last), so `state.bids[i]` for `i != dealer` is authoritative.
pub fn forbidden_bid(state: &BlobState) -> Option<u8> {
    let cards = state.cards_dealt as i32;
    let total_others: i32 = (0..state.num_players as usize)
        .filter(|&i| i != state.dealer as usize)
        .map(|i| state.bids[i] as i32)
        .sum();
    let forbidden = cards - total_others;
    if (0..=cards).contains(&forbidden) {
        Some(forbidden as u8)
    } else {
        None
    }
}

/// Apply `bid` for the current player and advance bidding state.
///
/// When the dealer bids (last in order), transitions to the playing phase
/// with `current_player == trick_leader == (dealer + 1) % num_players` and
/// `trick_cards_played == 0`. Otherwise advances `current_player` by one
/// seat (modulo `num_players`).
///
/// Panics in debug if `bid` is not in [`legal_bids`] for the current player.
pub fn apply_bid(state: &mut BlobState, bid: u8) {
    debug_assert_eq!(state.phase(), GamePhase::Bidding);
    debug_assert!(
        bid as usize <= MAX_CARDS_DEALT && (legal_bids(state) >> bid) & 1 == 1,
        "illegal bid {bid} for current player {} (legal mask = {:b})",
        state.current_player,
        legal_bids(state)
    );

    let p = state.current_player as usize;
    state.bids[p] = bid;

    if state.current_player == state.dealer {
        let first = (state.dealer + 1) % state.num_players;
        state.game_phase = GamePhase::Playing as u8;
        state.current_player = first;
        state.trick_leader = first;
        state.trick_cards_played = 0;
    } else {
        state.current_player = (state.current_player + 1) % state.num_players;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dealing::start_round;
    use crate::state::{BlobState, MAX_PLAYERS};
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn bidding_state(num_players: u8, cards_dealt: u8, dealer: u8) -> BlobState {
        let mut s = BlobState::empty();
        s.num_players = num_players;
        s.cards_dealt = cards_dealt;
        s.dealer = dealer;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB1D);
        start_round(&mut s, &mut rng);
        s
    }

    fn legal_set(state: &BlobState) -> Vec<u8> {
        let mask = legal_bids(state);
        (0..=13u8).filter(|b| (mask >> b) & 1 == 1).collect()
    }

    // -- forbidden_bid -------------------------------------------------------

    #[test]
    fn forbidden_bid_basic() {
        // 3 players, 5 cards, dealer = 0. Others bid 2 + 1 = 3 → forbidden = 2.
        let mut s = bidding_state(3, 5, 0);
        s.bids[1] = 2;
        s.bids[2] = 1;
        s.current_player = 0;
        assert_eq!(forbidden_bid(&s), Some(2));
    }

    #[test]
    fn forbidden_bid_zero_when_others_full() {
        // 3 players, 5 cards. Others bid 5 → forbidden = 0.
        let mut s = bidding_state(3, 5, 0);
        s.bids[1] = 3;
        s.bids[2] = 2;
        assert_eq!(forbidden_bid(&s), Some(0));
    }

    #[test]
    fn forbidden_bid_max_when_others_zero() {
        // Others bid 0 → forbidden = cards_dealt.
        let s = bidding_state(3, 5, 0);
        assert_eq!(forbidden_bid(&s), Some(5));
    }

    #[test]
    fn forbidden_bid_none_when_negative() {
        // Others bid more than cards dealt ⇒ forbidden would be negative.
        let mut s = bidding_state(3, 5, 0);
        s.bids[1] = 4;
        s.bids[2] = 3;
        assert_eq!(forbidden_bid(&s), None);
    }

    #[test]
    fn forbidden_bid_one_card_round_edge_cases() {
        // 4 players, 1 card.
        let mut s = bidding_state(4, 1, 0);
        // Others all bid 0 → forbidden = 1, dealer must bid 0.
        assert_eq!(forbidden_bid(&s), Some(1));
        // Others sum to 1 → forbidden = 0, dealer must bid 1.
        s.bids[1] = 1;
        assert_eq!(forbidden_bid(&s), Some(0));
        // Others sum to 2 → forbidden = -1 ⇒ None.
        s.bids[2] = 1;
        assert_eq!(forbidden_bid(&s), None);
    }

    // -- legal_bids ----------------------------------------------------------

    #[test]
    fn legal_bids_non_dealer_full_range() {
        // 4 players, 5 cards, dealer = 0. First bidder is player 1 (non-dealer).
        let s = bidding_state(4, 5, 0);
        assert_eq!(s.current_player, 1);
        assert_eq!(legal_set(&s), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn legal_bids_dealer_excludes_forbidden() {
        // 3 players, 5 cards, dealer = 0. Others bid 2 + 1 → forbidden = 2.
        let mut s = bidding_state(3, 5, 0);
        s.bids[1] = 2;
        s.bids[2] = 1;
        s.current_player = 0;
        assert_eq!(legal_set(&s), vec![0, 1, 3, 4, 5]);
    }

    #[test]
    fn legal_bids_dealer_excludes_zero_when_forbidden() {
        let mut s = bidding_state(3, 5, 0);
        s.bids[1] = 3;
        s.bids[2] = 2;
        s.current_player = 0;
        assert_eq!(legal_set(&s), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn legal_bids_dealer_excludes_max_when_forbidden() {
        let s = bidding_state(3, 5, 0);
        let mut s2 = s;
        s2.current_player = 0;
        // Others bid 0+0 → forbidden = 5.
        assert_eq!(legal_set(&s2), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn legal_bids_dealer_no_forbidden_full_range() {
        // Others bid sum to > cards_dealt ⇒ no forbidden.
        let mut s = bidding_state(3, 5, 0);
        s.bids[1] = 4;
        s.bids[2] = 3;
        s.current_player = 0;
        assert_eq!(legal_set(&s), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn legal_bids_returns_zero_outside_bidding() {
        let mut s = bidding_state(4, 5, 0);
        s.game_phase = GamePhase::Playing as u8;
        assert_eq!(legal_bids(&s), 0);
        s.game_phase = GamePhase::Scoring as u8;
        assert_eq!(legal_bids(&s), 0);
    }

    #[test]
    fn legal_bids_one_card_round_dealer_first() {
        let mut s = bidding_state(4, 1, 0);
        s.current_player = 0;
        // Others all 0 → forbidden 1 → dealer must bid 0.
        assert_eq!(legal_set(&s), vec![0]);
    }

    // -- apply_bid -----------------------------------------------------------

    #[test]
    fn apply_bid_advances_turn_to_next_seat() {
        let mut s = bidding_state(4, 5, 0);
        assert_eq!(s.current_player, 1);
        apply_bid(&mut s, 2);
        assert_eq!(s.bids[1], 2);
        assert_eq!(s.current_player, 2);
        assert_eq!(s.phase(), GamePhase::Bidding);
    }

    #[test]
    fn apply_bid_full_sequence_transitions_to_playing() {
        let mut s = bidding_state(4, 5, 0);
        // Order: 1, 2, 3, 0 (dealer last).
        let bidders = [1u8, 2, 3, 0];
        let bids = [1u8, 2, 0, 1]; // sum = 4 ≠ 5 ⇒ dealer's bid is legal
        for (i, (&p, &b)) in bidders.iter().zip(bids.iter()).enumerate() {
            assert_eq!(s.current_player, p, "step {i}");
            assert_eq!(s.phase(), GamePhase::Bidding);
            apply_bid(&mut s, b);
        }
        // After dealer bids: in playing phase, leader = first bidder.
        assert_eq!(s.phase(), GamePhase::Playing);
        assert_eq!(s.current_player, 1);
        assert_eq!(s.trick_leader, 1);
        assert_eq!(s.trick_cards_played, 0);
        assert_eq!(s.bids[..4], [1, 1, 2, 0]);
    }

    #[test]
    fn apply_bid_dealer_wraps_with_high_dealer_index() {
        // dealer = 3 ⇒ first bidder wraps to 0.
        let mut s = bidding_state(4, 5, 3);
        assert_eq!(s.current_player, 0);
        for (p, b) in [(0u8, 0), (1, 1), (2, 1)] {
            assert_eq!(s.current_player, p);
            apply_bid(&mut s, b);
        }
        // Now dealer is up; sum of others = 2, forbidden = 3.
        assert_eq!(s.current_player, 3);
        assert_eq!(forbidden_bid(&s), Some(3));
        apply_bid(&mut s, 4);
        assert_eq!(s.phase(), GamePhase::Playing);
        assert_eq!(s.current_player, 0);
        assert_eq!(s.trick_leader, 0);
    }

    #[test]
    fn apply_bid_one_card_round_dealer_constraint_enforced() {
        // 4 players, 1 card. Others bid 0,0,0 → forbidden = 1, dealer must bid 0.
        let mut s = bidding_state(4, 1, 0);
        for p in [1u8, 2, 3] {
            assert_eq!(s.current_player, p);
            apply_bid(&mut s, 0);
        }
        // Now dealer up; legal_bids excludes 1.
        assert_eq!(s.current_player, 0);
        assert_eq!(legal_set(&s), vec![0]);
        apply_bid(&mut s, 0);
        assert_eq!(s.phase(), GamePhase::Playing);
        assert_eq!(s.bids[..4], [0, 0, 0, 0]);
    }

    #[test]
    fn apply_bid_one_card_round_dealer_must_bid_one() {
        // 4 players, 1 card. Others bid 1,0,0 → forbidden = 0, dealer must bid 1.
        let mut s = bidding_state(4, 1, 0);
        apply_bid(&mut s, 1); // player 1
        apply_bid(&mut s, 0); // player 2
        apply_bid(&mut s, 0); // player 3
        assert_eq!(s.current_player, 0);
        assert_eq!(legal_set(&s), vec![1]);
        apply_bid(&mut s, 1);
        assert_eq!(s.phase(), GamePhase::Playing);
        assert_eq!(s.bids[..4], [1, 1, 0, 0]);
    }

    #[test]
    #[should_panic(expected = "illegal bid")]
    fn apply_bid_panics_on_illegal_dealer_value_in_debug() {
        let mut s = bidding_state(3, 5, 0);
        s.bids[1] = 2;
        s.bids[2] = 1;
        s.current_player = 0;
        // Forbidden = 2.
        apply_bid(&mut s, 2);
    }

    #[test]
    #[should_panic(expected = "illegal bid")]
    fn apply_bid_panics_on_out_of_range_in_debug() {
        let mut s = bidding_state(4, 5, 0);
        apply_bid(&mut s, 6);
    }

    // -- TestGetCurrentPlayer ports ------------------------------------------

    #[test]
    fn first_bidder_is_left_of_dealer() {
        for dealer in 0..4u8 {
            let s = bidding_state(4, 5, dealer);
            assert_eq!(s.current_player, (dealer + 1) % 4);
        }
    }

    #[test]
    fn bidding_sequence_visits_each_player_once() {
        let mut s = bidding_state(4, 5, 0);
        let mut visited = Vec::new();
        for _ in 0..4 {
            visited.push(s.current_player);
            // Pick any legal bid (avoid the dealer's forbidden one).
            let mask = legal_bids(&s);
            let bid = (0..=13u8).find(|b| (mask >> b) & 1 == 1).unwrap();
            apply_bid(&mut s, bid);
        }
        assert_eq!(visited.len(), 4);
        let mut sorted = visited.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
        assert_eq!(visited[0], 1, "starts left of dealer");
        assert_eq!(s.phase(), GamePhase::Playing);
    }

    #[test]
    fn unused_player_slots_remain_zero() {
        // 3 players: bids[3..8] should remain zero through bidding.
        let mut s = bidding_state(3, 5, 0);
        for _ in 0..3 {
            let mask = legal_bids(&s);
            let bid = (0..=13u8).find(|b| (mask >> b) & 1 == 1).unwrap();
            apply_bid(&mut s, bid);
        }
        for slot in &s.bids[3..MAX_PLAYERS] {
            assert_eq!(*slot, 0);
        }
    }
}
