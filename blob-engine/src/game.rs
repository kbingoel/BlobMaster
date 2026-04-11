//! Multi-round game lifecycle: `new_game`, `advance_round`, and `is_game_over`.
//!
//! These wire together the per-phase primitives from sessions 1.1–1.3
//! (`dealing`, `bidding`, `playing`, `round`) into the full game loop:
//!
//! ```text
//! new_game → start_round → bidding → playing → advance_round → … → Complete
//! ```
//!
//! Round-level state (cards dealt, dealer, trump) is derived from
//! `BlobState::round_idx` and `BlobState::start_cards`. Per-game RNG ownership
//! lives outside `BlobState` (see `state.rs`), so `Copy` semantics survive
//! intact for MCTS determinization.

use rand::Rng;

use crate::dealing::start_round;
use crate::playing::score_round;
use crate::round::{
    cards_dealt_for_round, total_rounds, trump_for_round, validate_round_params, RoundParamsError,
};
use crate::state::{BlobState, GamePhase};

/// Initialize a fresh game with the given parameters.
///
/// Sets meta-state (`num_players`, `start_cards`, `round_idx = 0`,
/// `dealer = 0`, `cards_dealt = start_cards`, `trump_suit = ♠`) but does
/// **not** deal — call [`start_round`] next with your RNG of choice. Keeping
/// dealing as a separate step lets MCTS construct states without consuming
/// randomness, which matters for determinization.
///
/// Returns an error if `(start_cards, num_players)` would be rejected by
/// [`validate_round_params`].
pub fn new_game(num_players: u8, start_cards: u8) -> Result<BlobState, RoundParamsError> {
    validate_round_params(start_cards, num_players)?;
    let mut s = BlobState::empty();
    s.num_players = num_players;
    s.start_cards = start_cards;
    s.cards_dealt = start_cards;
    s.round_idx = 0;
    s.dealer = 0;
    s.trump_suit = trump_for_round(0);
    s.game_phase = GamePhase::Bidding as u8;
    s.current_player = (s.dealer + 1) % num_players;
    s.trick_leader = s.current_player;
    Ok(s)
}

/// Score the just-finished round and transition the game forward.
///
/// Precondition: `state.phase() == Scoring`. The trick-playing phase
/// auto-transitions to `Scoring` after the last `apply_play`, so the typical
/// caller does not set the phase manually.
///
/// On call:
/// 1. [`score_round`] is invoked once to fold this round's per-player scores
///    into `cumulative_scores`.
/// 2. If this was the **last** round of the game, `game_phase` becomes
///    `Complete` and the function returns — no further state changes.
/// 3. Otherwise: `round_idx` advances by one, `cards_dealt` is recomputed
///    from the symmetric round structure, the dealer rotates one seat, the
///    trump suit cycles, and [`start_round`] is called to deal fresh hands
///    and prime the bidding phase.
///
/// **Do not** also call `score_round` directly when using `advance_round` —
/// `cumulative_scores` would be double-counted.
pub fn advance_round<R: Rng + ?Sized>(state: &mut BlobState, rng: &mut R) {
    debug_assert_eq!(
        state.phase(),
        GamePhase::Scoring,
        "advance_round must be called from the scoring phase"
    );

    score_round(state);

    let next = state.round_idx + 1;
    if next >= total_rounds(state.start_cards, state.num_players) {
        state.game_phase = GamePhase::Complete as u8;
        return;
    }

    state.round_idx = next;
    state.cards_dealt = cards_dealt_for_round(next, state.start_cards, state.num_players);
    state.dealer = (state.dealer + 1) % state.num_players;
    state.trump_suit = trump_for_round(next as u32);
    start_round(state, rng);
}

/// True once the final round has been scored and `advance_round` has marked
/// the state `Complete`.
#[inline]
pub fn is_game_over(state: &BlobState) -> bool {
    state.phase() == GamePhase::Complete
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bidding::{apply_bid, legal_bids};
    use crate::card::Suit;
    use crate::playing::{apply_play, legal_plays};
    use crate::round::{NO_TRUMP, TRUMP_CYCLE_LEN};
    use crate::state::MAX_PLAYERS;
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn pick_first_legal_bid(state: &BlobState) -> u8 {
        let mask = legal_bids(state);
        (0..=13u8).find(|b| (mask >> b) & 1 == 1).expect("legal bid")
    }

    fn pick_first_legal_play(state: &BlobState) -> u8 {
        let mask = legal_plays(state);
        debug_assert!(mask != 0);
        mask.trailing_zeros() as u8
    }

    fn play_one_round<R: Rng + ?Sized>(state: &mut BlobState, rng: &mut R) {
        // Bid out.
        while state.phase() == GamePhase::Bidding {
            let bid = pick_first_legal_bid(state);
            apply_bid(state, bid);
        }
        // Play out.
        while state.phase() == GamePhase::Playing {
            let card = pick_first_legal_play(state);
            apply_play(state, card);
        }
        // Now in scoring phase. Caller drives advance_round.
        let _ = rng;
    }

    #[test]
    fn new_game_validates_params() {
        // 8 × 7 = 56 > 52 — should fail.
        assert_eq!(new_game(8, 7), Err(RoundParamsError::DeckExceeded));
        // 2 players is below the minimum.
        assert_eq!(new_game(2, 5), Err(RoundParamsError::PlayerCountOutOfRange));
        // start_cards = 0 fails.
        assert_eq!(new_game(4, 0), Err(RoundParamsError::StartCardsZero));
    }

    #[test]
    fn new_game_initializes_meta_state() {
        let s = new_game(5, 7).unwrap();
        assert_eq!(s.num_players, 5);
        assert_eq!(s.start_cards, 7);
        assert_eq!(s.cards_dealt, 7);
        assert_eq!(s.round_idx, 0);
        assert_eq!(s.dealer, 0);
        assert_eq!(s.trump_suit, Suit::Spades as u8);
        assert_eq!(s.phase(), GamePhase::Bidding);
        assert_eq!(s.current_player, 1, "left of dealer 0");
        assert_eq!(s.trick_leader, 1);
        // No cards dealt yet — caller must run start_round.
        for h in s.hands.iter() {
            assert_eq!(*h, 0);
        }
    }

    #[test]
    fn advance_round_to_round_two_rotates_dealer_and_trump() {
        let mut s = new_game(4, 5).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(11);
        start_round(&mut s, &mut rng);
        play_one_round(&mut s, &mut rng);
        assert_eq!(s.phase(), GamePhase::Scoring);

        advance_round(&mut s, &mut rng);

        assert_eq!(s.round_idx, 1);
        assert_eq!(s.dealer, 1, "dealer rotates one seat");
        assert_eq!(s.trump_suit, Suit::Hearts as u8, "round 1 trump = ♥");
        // Cards dealt for round 1 with start_cards=5: descending [5,4,…] → 4.
        assert_eq!(s.cards_dealt, 4);
        assert_eq!(s.phase(), GamePhase::Bidding);
        assert_eq!(s.current_player, 2, "left of new dealer 1");
        // Hands re-dealt to the new size.
        for p in 0..4 {
            assert_eq!(s.hands[p].count_ones(), 4);
        }
        // Per-round state was reset.
        assert_eq!(s.bids, [0; MAX_PLAYERS]);
        assert_eq!(s.tricks_won, [0; MAX_PLAYERS]);
        assert_eq!(s.tricks_completed, 0);
        assert_eq!(s.played_this_round, 0);
    }

    #[test]
    fn advance_round_accumulates_scores_into_cumulative_scores() {
        // Hand-craft a Scoring state and call advance_round directly.
        let mut s = new_game(4, 5).unwrap();
        s.game_phase = GamePhase::Scoring as u8;
        s.bids[..4].copy_from_slice(&[2, 1, 0, 3]);
        s.tricks_won[..4].copy_from_slice(&[2, 0, 0, 3]); // p0,p2,p3 made bid

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        advance_round(&mut s, &mut rng);

        // p0: 12, p1: 0, p2: 10, p3: 13
        assert_eq!(s.cumulative_scores[..4], [12, 0, 10, 13]);
        // Game continued to round 1.
        assert_eq!(s.round_idx, 1);
        assert_eq!(s.phase(), GamePhase::Bidding);
    }

    #[test]
    fn full_game_4p_3c_terminates_with_complete_phase() {
        let mut s = new_game(4, 3).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2024);
        start_round(&mut s, &mut rng);

        let total = total_rounds(3, 4); // 2*3 + 4 - 2 = 8
        let mut rounds_played = 0;
        loop {
            play_one_round(&mut s, &mut rng);
            advance_round(&mut s, &mut rng);
            rounds_played += 1;
            if is_game_over(&s) {
                break;
            }
            assert!(rounds_played < total + 1, "loop ran too long");
        }
        assert_eq!(rounds_played, total);
        assert!(is_game_over(&s));
        assert_eq!(s.phase(), GamePhase::Complete);
        assert_eq!(s.round_idx, total - 1, "round_idx pinned at last index");
    }

    #[test]
    fn full_game_5p_7c_dealer_rotates_correctly() {
        let mut s = new_game(5, 7).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        start_round(&mut s, &mut rng);

        let total = total_rounds(7, 5); // 17
        for r in 0..total {
            assert_eq!(s.round_idx, r);
            assert_eq!(s.dealer, (r % 5));
            play_one_round(&mut s, &mut rng);
            advance_round(&mut s, &mut rng);
        }
        assert!(is_game_over(&s));
    }

    #[test]
    fn full_game_5p_7c_trump_cycles_through_all_five_values() {
        let mut s = new_game(5, 7).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(8);
        start_round(&mut s, &mut rng);

        let total = total_rounds(7, 5);
        let mut trumps = Vec::new();
        for _ in 0..total {
            trumps.push(s.trump_suit);
            play_one_round(&mut s, &mut rng);
            advance_round(&mut s, &mut rng);
        }
        // Sequence is [trump_for_round(i)] for i = 0..total.
        let expected: Vec<u8> = (0..total as u32).map(trump_for_round).collect();
        assert_eq!(trumps, expected);
        // Sanity: a 17-round game must hit every trump value at least once
        // (ceil(17/5) = 4 cycles of the 5-element rotation).
        for t in 0..TRUMP_CYCLE_LEN {
            assert!(trumps.contains(&t), "trump value {t} should appear");
        }
        assert!(trumps.contains(&NO_TRUMP));
    }

    #[test]
    fn cards_dealt_per_round_matches_round_structure() {
        let mut s = new_game(4, 5).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(99);
        start_round(&mut s, &mut rng);

        let expected = [5u8, 4, 3, 2, 1, 1, 1, 1, 2, 3, 4, 5];
        let mut got = Vec::new();
        for _ in 0..expected.len() {
            got.push(s.cards_dealt);
            play_one_round(&mut s, &mut rng);
            advance_round(&mut s, &mut rng);
        }
        assert_eq!(got, expected);
        assert!(is_game_over(&s));
    }

    #[test]
    fn is_game_over_only_true_after_final_advance() {
        let mut s = new_game(3, 2).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        start_round(&mut s, &mut rng);

        let total = total_rounds(2, 3); // 2*2 + 3 - 2 = 5
        for _ in 0..total - 1 {
            assert!(!is_game_over(&s));
            play_one_round(&mut s, &mut rng);
            advance_round(&mut s, &mut rng);
            assert!(!is_game_over(&s));
        }
        // Last round.
        play_one_round(&mut s, &mut rng);
        assert_eq!(s.phase(), GamePhase::Scoring);
        assert!(!is_game_over(&s));
        advance_round(&mut s, &mut rng);
        assert!(is_game_over(&s));
    }

    #[test]
    fn cumulative_scores_grow_monotonically_across_rounds() {
        let mut s = new_game(4, 4).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
        start_round(&mut s, &mut rng);

        let mut prev = [0u16; MAX_PLAYERS];
        loop {
            play_one_round(&mut s, &mut rng);
            advance_round(&mut s, &mut rng);
            for (i, &p) in prev.iter().enumerate().take(s.num_players as usize) {
                assert!(
                    s.cumulative_scores[i] >= p,
                    "cumulative_scores[{i}] decreased: {} → {}",
                    p,
                    s.cumulative_scores[i]
                );
            }
            prev = s.cumulative_scores;
            if is_game_over(&s) {
                break;
            }
        }
    }

    #[test]
    fn one_card_round_in_middle_of_game_handled() {
        // 4p × 5c game has a one-card plateau in rounds 4..=7. Step through
        // the plateau and confirm cards_dealt = 1 and bidding/play still work.
        let mut s = new_game(4, 5).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(555);
        start_round(&mut s, &mut rng);

        // Run rounds 0..4 (descending C..=2), reaching the plateau.
        for _ in 0..4 {
            play_one_round(&mut s, &mut rng);
            advance_round(&mut s, &mut rng);
        }
        // Now at round 4 (first one-card round).
        assert_eq!(s.round_idx, 4);
        assert_eq!(s.cards_dealt, 1);
        // Each player should hold exactly one card.
        for p in 0..4 {
            assert_eq!(s.hands[p].count_ones(), 1);
        }
        play_one_round(&mut s, &mut rng);
        // After one trick the round ends → Scoring.
        assert_eq!(s.phase(), GamePhase::Scoring);
        assert_eq!(s.tricks_completed, 1);
        let total: u32 = s.tricks_won[..4].iter().map(|&t| t as u32).sum();
        assert_eq!(total, 1);
    }

    #[test]
    fn deal_full_deck_4p_13c_smoke() {
        // 4 × 13 = 52 (full deck) is the largest legal deal.
        let s = new_game(4, 13).unwrap();
        assert_eq!(s.start_cards, 13);
        assert_eq!(total_rounds(13, 4), 2 * 13 + 4 - 2);
    }

    #[test]
    #[should_panic(expected = "scoring phase")]
    fn advance_round_panics_outside_scoring_phase_in_debug() {
        let mut s = new_game(4, 5).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        start_round(&mut s, &mut rng);
        // Phase is Bidding.
        advance_round(&mut s, &mut rng);
    }
}
