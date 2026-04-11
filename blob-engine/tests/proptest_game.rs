//! Property-based tests for the full Blob game loop.
//!
//! These probe the engine with random `(num_players, start_cards, seed)`
//! triples and assert game-wide invariants:
//!
//! - **Termination**: a randomly-played game ends in the `Complete` phase
//!   in exactly `total_rounds` rounds.
//! - **Card conservation**: the 52-card deck is never lost or duplicated.
//!   At the end of every round, hands are empty and `played_this_round`
//!   contains exactly `num_players * cards_dealt` distinct cards.
//! - **Tricks_won sums to cards_dealt**: per round, the sum across players
//!   equals the number of tricks (which equals `cards_dealt`).
//! - **No illegal plays**: every card chosen during proptest playouts is
//!   drawn from `legal_plays`, so `apply_play` never panics in debug.
//! - **Hand membership**: a card played by a player must have been in their
//!   hand the moment before they played it.

use blob_engine::{
    advance_round, apply_bid, apply_play, is_game_over, legal_bids, legal_plays, new_game,
    start_round, total_rounds, BlobState, GamePhase,
};
use proptest::prelude::*;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Pick a random *legal* bid for the current player using the provided RNG.
fn random_legal_bid(state: &BlobState, rng: &mut Xoshiro256PlusPlus) -> u8 {
    let mask = legal_bids(state);
    debug_assert!(mask != 0);
    let bids: Vec<u8> = (0..=13u8).filter(|b| (mask >> b) & 1 == 1).collect();
    *bids.choose(rng).expect("legal bids non-empty")
}

/// Pick a random *legal* card for the current player using the provided RNG.
fn random_legal_card(state: &BlobState, rng: &mut Xoshiro256PlusPlus) -> u8 {
    let mask = legal_plays(state);
    debug_assert!(mask != 0);
    let cards: Vec<u8> = (0..52u8).filter(|c| (mask >> c) & 1 == 1).collect();
    *cards.choose(rng).expect("legal plays non-empty")
}

/// Drive a single round forward through bid → play → scoring, asserting
/// per-step invariants. Returns the cards-dealt count for that round so the
/// caller can check totals.
fn play_one_round_with_invariants(
    state: &mut BlobState,
    rng: &mut Xoshiro256PlusPlus,
) -> (u8, u32) {
    let cards_dealt = state.cards_dealt;
    let num_players = state.num_players;
    let expected_dealt: u32 = cards_dealt as u32 * num_players as u32;

    // Hand-card-count invariant at start of round.
    let total_in_hands: u32 = state.hands[..num_players as usize]
        .iter()
        .map(|h| h.count_ones())
        .sum();
    assert_eq!(total_in_hands, expected_dealt, "deal must hand out exactly cards_dealt × num_players cards");
    // Hands are pairwise disjoint (no card duplicated across seats).
    let mut union: u64 = 0;
    for h in &state.hands[..num_players as usize] {
        assert_eq!(union & *h, 0, "duplicated card across hands");
        union |= *h;
    }
    assert_eq!(union.count_ones(), expected_dealt);

    // Bidding phase: pick a random legal bid each turn.
    while state.phase() == GamePhase::Bidding {
        let bid = random_legal_bid(state, rng);
        apply_bid(state, bid);
    }

    // Playing phase: every play must be a card from the current player's hand.
    while state.phase() == GamePhase::Playing {
        let card = random_legal_card(state, rng);
        let p = state.current_player as usize;
        let hand_before = state.hands[p];
        assert!(
            (hand_before >> card) & 1 == 1,
            "selected card not in hand at play time"
        );
        apply_play(state, card);
    }

    // After all tricks: scoring phase, hands empty.
    assert_eq!(state.phase(), GamePhase::Scoring);
    for i in 0..num_players as usize {
        assert_eq!(state.hands[i], 0, "hand {i} not empty after round");
    }
    // played_this_round contains exactly the dealt cards.
    assert_eq!(state.played_this_round.count_ones(), expected_dealt);
    // tricks_won sums to cards_dealt.
    let trick_sum: u32 = state.tricks_won[..num_players as usize]
        .iter()
        .map(|&t| t as u32)
        .sum();
    assert_eq!(
        trick_sum, cards_dealt as u32,
        "tricks_won must sum to cards_dealt"
    );

    (cards_dealt, expected_dealt)
}

/// Strategy: legal `(num_players, start_cards)` pair plus a seed.
///
/// `start_cards` is bounded by `52 / num_players` so we never overflow the
/// deck. Range is intentionally narrow on the high end (start_cards ≤ 6)
/// to keep proptest runtime predictable; full-deck cases are exercised by
/// dedicated unit tests in `game.rs`.
fn game_params() -> impl Strategy<Value = (u8, u8, u64)> {
    (3u8..=6u8)
        .prop_flat_map(|n| {
            let max_cards = (52u8 / n).min(6);
            (Just(n), 1u8..=max_cards, any::<u64>())
        })
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        .. ProptestConfig::default()
    })]

    #[test]
    fn random_games_terminate_in_complete_phase((num_players, start_cards, seed) in game_params()) {
        let mut state = new_game(num_players, start_cards).expect("valid params");
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        start_round(&mut state, &mut rng);

        let total = total_rounds(start_cards, num_players);
        let mut rounds_played = 0u8;
        let max_iters = (total as u32).saturating_mul(2);
        for _ in 0..max_iters {
            play_one_round_with_invariants(&mut state, &mut rng);
            advance_round(&mut state, &mut rng);
            rounds_played += 1;
            if is_game_over(&state) {
                break;
            }
        }
        prop_assert!(is_game_over(&state), "game must reach Complete phase");
        prop_assert_eq!(rounds_played, total, "exactly total_rounds rounds played");
    }

    #[test]
    fn card_conservation_across_full_game((num_players, start_cards, seed) in game_params()) {
        let mut state = new_game(num_players, start_cards).expect("valid params");
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        start_round(&mut state, &mut rng);

        let mut total_cards_played: u64 = 0;
        loop {
            let (cards_dealt, expected) = play_one_round_with_invariants(&mut state, &mut rng);
            // No card was lost or duplicated within the round.
            prop_assert_eq!(state.played_this_round.count_ones(), expected);
            // Every dealt card index is within [0, 52).
            prop_assert_eq!(state.played_this_round & !((1u64 << 52) - 1), 0);
            total_cards_played += cards_dealt as u64 * num_players as u64;
            advance_round(&mut state, &mut rng);
            if is_game_over(&state) {
                break;
            }
            // After advance_round, played_this_round was reset.
            prop_assert_eq!(state.played_this_round, 0);
        }
        // Sanity: total cards played across the game equals the sum of the
        // round structure × num_players.
        let expected_total: u64 = (0..total_rounds(start_cards, num_players))
            .map(|i| {
                blob_engine::cards_dealt_for_round(i, start_cards, num_players) as u64
                    * num_players as u64
            })
            .sum();
        prop_assert_eq!(total_cards_played, expected_total);
    }

    #[test]
    fn cumulative_scores_in_legal_range((num_players, start_cards, seed) in game_params()) {
        let mut state = new_game(num_players, start_cards).expect("valid params");
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        start_round(&mut state, &mut rng);

        loop {
            play_one_round_with_invariants(&mut state, &mut rng);
            advance_round(&mut state, &mut rng);
            if is_game_over(&state) {
                break;
            }
        }
        // Each round contributes at most 10 + cards_dealt to a player's
        // total. Upper bound: sum_round (10 + cards_dealt[round]) per player.
        let max_per_player: u32 = (0..total_rounds(start_cards, num_players))
            .map(|i| {
                10u32 + blob_engine::cards_dealt_for_round(i, start_cards, num_players) as u32
            })
            .sum();
        for i in 0..num_players as usize {
            prop_assert!(
                (state.cumulative_scores[i] as u32) <= max_per_player,
                "player {} score {} exceeds max {}",
                i,
                state.cumulative_scores[i],
                max_per_player
            );
        }
    }

    #[test]
    fn legal_plays_subset_of_hand((num_players, start_cards, seed) in game_params()) {
        // While a round is in progress, legal_plays(state) is always a
        // subset of the current player's hand (and never empty).
        let mut state = new_game(num_players, start_cards).expect("valid params");
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        start_round(&mut state, &mut rng);

        // Walk through bids without further checks.
        while state.phase() == GamePhase::Bidding {
            let bid = random_legal_bid(&state, &mut rng);
            apply_bid(&mut state, bid);
        }

        while state.phase() == GamePhase::Playing {
            let mask = legal_plays(&state);
            let hand = state.hands[state.current_player as usize];
            prop_assert!(mask != 0, "legal_plays must be non-empty during Playing");
            prop_assert_eq!(mask & !hand, 0, "legal_plays must be subset of current hand");
            let card = random_legal_card(&state, &mut rng);
            apply_play(&mut state, card);
        }
        prop_assert_eq!(state.phase(), GamePhase::Scoring);
    }
}
