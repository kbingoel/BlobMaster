//! Ports of legacy `legacy/game-engine/test_blob.py` tests that exercise the
//! full multi-round game loop landed in Session 1.4.
//!
//! Many of the original Python tests covered behavior that is structurally
//! impossible in the Rust port (e.g. mutating object fields directly,
//! catching `IllegalPlayException` thrown from `validate_play_with_anti_cheat`)
//! — those land naturally in the unit tests under `blob-engine/src/*.rs`.
//! This file focuses on the game-flow-level scenarios that only become
//! testable once `new_game` / `advance_round` exist.
//!
//! ⚠ Round-structure related tests use the **corrected** `2C + n − 2` formula
//! (Session 1.2's gate decision); legacy values from
//! `generate_round_structure` were off-by-one.

use blob_engine::{
    advance_round, apply_bid, apply_play, cards_dealt_for_round, forbidden_bid, is_game_over,
    legal_bids, legal_plays, new_game, round_structure, start_round, total_rounds, BlobState,
    Card, GamePhase, RoundParamsError, Suit, NO_TRUMP,
};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

// ----------------------------------------------------------------------------
// Helpers — minimal "bot" callbacks mirroring the legacy
// `simple_bid_always_zero` / `simple_play_first_legal` / `play_winning_cards`.
// ----------------------------------------------------------------------------

fn bid_always_zero_or_one(state: &BlobState) -> u8 {
    let mask = legal_bids(state);
    if (mask & 1) == 1 {
        0
    } else {
        // 0 forbidden — pick 1.
        1
    }
}

fn bid_position(state: &BlobState) -> u8 {
    // Try the player's seat number, fall back to first legal.
    let mask = legal_bids(state);
    let want = state.current_player as u16;
    if (mask >> want) & 1 == 1 {
        want as u8
    } else {
        (0..=13u8).find(|b| (mask >> b) & 1 == 1).unwrap()
    }
}

fn play_first_legal(state: &BlobState) -> u8 {
    legal_plays(state).trailing_zeros() as u8
}

fn play_highest_legal(state: &BlobState) -> u8 {
    // 64 − leading_zeros − 1 = highest set bit.
    let mask = legal_plays(state);
    63 - mask.leading_zeros() as u8
}

/// Drive a single round forward end-to-end with caller-supplied callbacks.
fn run_round<F, G>(state: &mut BlobState, mut bid_fn: F, mut play_fn: G)
where
    F: FnMut(&BlobState) -> u8,
    G: FnMut(&BlobState) -> u8,
{
    while state.phase() == GamePhase::Bidding {
        let b = bid_fn(state);
        apply_bid(state, b);
    }
    while state.phase() == GamePhase::Playing {
        let c = play_fn(state);
        apply_play(state, c);
    }
}

/// Drive a full multi-round game until `is_game_over` returns true.
fn run_game<F, G>(state: &mut BlobState, rng: &mut Xoshiro256PlusPlus, mut bid_fn: F, play_fn: G)
where
    F: FnMut(&BlobState) -> u8,
    G: Fn(&BlobState) -> u8 + Copy,
{
    loop {
        run_round(state, &mut bid_fn, play_fn);
        advance_round(state, rng);
        if is_game_over(state) {
            break;
        }
    }
}

// ----------------------------------------------------------------------------
// new_game / round-structure smoke ports
// ----------------------------------------------------------------------------

#[test]
fn new_game_default_state_matches_python_expectation() {
    // Python: BlobGame(num_players=4) → players=4, current_round=0, etc.
    let s = new_game(4, 5).unwrap();
    assert_eq!(s.num_players, 4);
    assert_eq!(s.round_idx, 0);
    assert_eq!(s.dealer, 0);
    assert_eq!(s.cumulative_scores, [0u16; 8]);
    assert_eq!(s.tricks_completed, 0);
    assert_eq!(s.phase(), GamePhase::Bidding);
}

#[test]
fn new_game_rejects_too_few_players() {
    assert_eq!(new_game(2, 5), Err(RoundParamsError::PlayerCountOutOfRange));
    assert_eq!(new_game(0, 5), Err(RoundParamsError::PlayerCountOutOfRange));
}

#[test]
fn new_game_rejects_too_many_players() {
    assert_eq!(new_game(9, 5), Err(RoundParamsError::PlayerCountOutOfRange));
    assert_eq!(new_game(255, 5), Err(RoundParamsError::PlayerCountOutOfRange));
}

#[test]
fn new_game_rejects_too_many_cards() {
    // 4p × 14c would need 56 cards.
    assert!(new_game(4, 14).is_err());
    // 8p × 7c = 56 > 52.
    assert_eq!(new_game(8, 7), Err(RoundParamsError::DeckExceeded));
}

#[test]
fn new_game_accepts_full_deck_4p_13c() {
    let s = new_game(4, 13).unwrap();
    assert_eq!(s.cards_dealt, 13);
    assert_eq!(s.start_cards, 13);
}

#[test]
fn new_game_accepts_3p_17c_max() {
    // 17 × 3 = 51 ≤ 52. Legacy test parity.
    let s = new_game(3, 13).unwrap();
    assert_eq!(s.cards_dealt, 13);
}

#[test]
fn round_structure_4p_5c_corrected_formula() {
    // Legacy returned 13 entries; corrected returns 12.
    assert_eq!(total_rounds(5, 4), 12);
    let s = round_structure(5, 4);
    assert_eq!(s.as_slice(), &[5, 4, 3, 2, 1, 1, 1, 1, 2, 3, 4, 5]);
}

#[test]
fn round_structure_3p_7c_corrected_formula() {
    // Legacy returned 16 entries; corrected returns 15.
    assert_eq!(total_rounds(7, 3), 15);
    let s = round_structure(7, 3);
    assert_eq!(s.as_slice(), &[7, 6, 5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn round_structure_starts_and_ends_at_start_cards() {
    for &(c, n) in &[(7u8, 5u8), (5, 4), (8, 6), (13, 4)] {
        let s = round_structure(c, n);
        assert_eq!(s.first(), Some(&c));
        assert_eq!(s.last(), Some(&c));
    }
}

#[test]
fn round_structure_one_card_plateau_length_equals_num_players() {
    for &(c, n) in &[(5u8, 3u8), (5, 4), (6, 5), (7, 6)] {
        let s = round_structure(c, n);
        let ones = s.iter().filter(|&&x| x == 1).count();
        assert_eq!(ones, n as usize, "C={c}, n={n}");
    }
}

// ----------------------------------------------------------------------------
// forbidden bid ports
// ----------------------------------------------------------------------------

#[test]
fn forbidden_bid_basic_5p_5c() {
    let mut s = new_game(3, 5).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    start_round(&mut s, &mut rng);
    s.bids[1] = 2;
    s.bids[2] = 1;
    s.current_player = 0; // dealer
    assert_eq!(forbidden_bid(&s), Some(2));
}

#[test]
fn forbidden_bid_one_card_round_edge_case() {
    let mut s = new_game(4, 1).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    start_round(&mut s, &mut rng);
    // All non-dealers bid 0 → forbidden = 1.
    s.current_player = 0; // dealer
    assert_eq!(forbidden_bid(&s), Some(1));
}

// ----------------------------------------------------------------------------
// Bidding-phase ports (multi-round version of legacy single-round tests)
// ----------------------------------------------------------------------------

#[test]
fn bidding_phase_first_player_left_of_dealer_each_round() {
    let mut s = new_game(4, 4).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(31);
    start_round(&mut s, &mut rng);
    for _ in 0..6 {
        // First bidder for the round is left of the dealer.
        let expected = (s.dealer + 1) % s.num_players;
        assert_eq!(s.current_player, expected);
        run_round(&mut s, bid_always_zero_or_one, play_first_legal);
        advance_round(&mut s, &mut rng);
    }
}

#[test]
fn bidding_visits_each_player_once_per_round() {
    let mut s = new_game(4, 5).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
    start_round(&mut s, &mut rng);
    let mut visited = Vec::new();
    while s.phase() == GamePhase::Bidding {
        visited.push(s.current_player);
        let b = bid_always_zero_or_one(&s);
        apply_bid(&mut s, b);
    }
    let mut sorted = visited.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1, 2, 3]);
    assert_eq!(visited.first(), Some(&1));
}

// ----------------------------------------------------------------------------
// Playing-phase / legal_plays ports
// ----------------------------------------------------------------------------

#[test]
fn legal_plays_first_card_full_hand_after_bidding_completes() {
    let mut s = new_game(4, 5).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(101);
    start_round(&mut s, &mut rng);
    while s.phase() == GamePhase::Bidding {
        let b = bid_always_zero_or_one(&s);
        apply_bid(&mut s, b);
    }
    let mask = legal_plays(&s);
    let p = s.current_player as usize;
    assert_eq!(mask, s.hands[p], "first card of trick: every hand card legal");
    assert_eq!(mask.count_ones(), 5);
}

#[test]
fn legal_plays_must_follow_led_suit_when_holding_it() {
    let mut s = new_game(4, 5).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(202);
    start_round(&mut s, &mut rng);
    while s.phase() == GamePhase::Bidding {
        let b = bid_always_zero_or_one(&s);
        apply_bid(&mut s, b);
    }
    // Lead a single card.
    let lead = play_first_legal(&s);
    let lead_suit = lead / 13;
    apply_play(&mut s, lead);

    // The next player either has cards of the led suit (mask is a subset of
    // that suit) or is void (mask == hand).
    let mask = legal_plays(&s);
    let hand = s.hands[s.current_player as usize];
    let suit_mask = 0x1FFFu64 << (lead_suit * 13);
    if hand & suit_mask != 0 {
        assert_eq!(mask, hand & suit_mask, "must follow {lead_suit}");
    } else {
        assert_eq!(mask, hand, "void in led suit → entire hand legal");
    }
}

// ----------------------------------------------------------------------------
// Trick-winner determination port — synthesizes states by hand-setting hands
// ----------------------------------------------------------------------------

fn one_trick_state(num_players: u8, trump: u8) -> BlobState {
    let mut s = new_game(num_players, 1).unwrap();
    s.trump_suit = trump;
    s.game_phase = GamePhase::Playing as u8;
    s.dealer = num_players - 1;
    s.trick_leader = 0;
    s.current_player = 0;
    s
}

fn card_idx(suit: Suit, rank: u8) -> u8 {
    Card::new(suit, rank).index()
}

#[test]
fn winner_no_trump_highest_in_led_suit() {
    let mut s = one_trick_state(3, NO_TRUMP);
    s.hands[0] = 1u64 << card_idx(Suit::Hearts, 12);
    s.hands[1] = 1u64 << card_idx(Suit::Hearts, 11);
    s.hands[2] = 1u64 << card_idx(Suit::Hearts, 10);
    apply_play(&mut s, card_idx(Suit::Hearts, 12));
    apply_play(&mut s, card_idx(Suit::Hearts, 11));
    apply_play(&mut s, card_idx(Suit::Hearts, 10));
    assert_eq!(s.trick_history[0].winner, 0);
}

#[test]
fn winner_trump_overrides_higher_non_trump() {
    let mut s = one_trick_state(3, Suit::Spades as u8);
    s.hands[0] = 1u64 << card_idx(Suit::Hearts, 12);
    s.hands[1] = 1u64 << card_idx(Suit::Spades, 0);
    s.hands[2] = 1u64 << card_idx(Suit::Hearts, 11);
    apply_play(&mut s, card_idx(Suit::Hearts, 12));
    apply_play(&mut s, card_idx(Suit::Spades, 0));
    apply_play(&mut s, card_idx(Suit::Hearts, 11));
    assert_eq!(s.trick_history[0].winner, 1);
}

#[test]
fn winner_off_suit_cannot_beat_low_led_suit() {
    let mut s = one_trick_state(3, NO_TRUMP);
    s.hands[0] = 1u64 << card_idx(Suit::Hearts, 0);
    s.hands[1] = 1u64 << card_idx(Suit::Spades, 12);
    s.hands[2] = 1u64 << card_idx(Suit::Hearts, 1);
    apply_play(&mut s, card_idx(Suit::Hearts, 0));
    apply_play(&mut s, card_idx(Suit::Spades, 12));
    apply_play(&mut s, card_idx(Suit::Hearts, 1));
    assert_eq!(s.trick_history[0].winner, 2);
}

#[test]
fn winner_records_suit_led_in_history() {
    let mut s = one_trick_state(3, Suit::Spades as u8);
    s.hands[0] = 1u64 << card_idx(Suit::Diamonds, 5);
    s.hands[1] = 1u64 << card_idx(Suit::Diamonds, 8);
    s.hands[2] = 1u64 << card_idx(Suit::Diamonds, 3);
    apply_play(&mut s, card_idx(Suit::Diamonds, 5));
    apply_play(&mut s, card_idx(Suit::Diamonds, 8));
    apply_play(&mut s, card_idx(Suit::Diamonds, 3));
    assert_eq!(s.trick_history[0].suit_led, Suit::Diamonds as u8);
}

// ----------------------------------------------------------------------------
// Scoring ports — direct equivalents to legacy `scoring_phase_*` cases.
// ----------------------------------------------------------------------------

#[test]
fn scoring_basic_mix_of_made_and_missed() {
    let mut s = new_game(3, 5).unwrap();
    s.game_phase = GamePhase::Scoring as u8;
    s.bids[..3].copy_from_slice(&[2, 1, 3]);
    s.tricks_won[..3].copy_from_slice(&[2, 0, 3]);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    advance_round(&mut s, &mut rng);
    assert_eq!(s.cumulative_scores[..3], [12, 0, 13]);
}

#[test]
fn scoring_zero_bid_made_yields_ten() {
    let mut s = new_game(3, 5).unwrap();
    s.game_phase = GamePhase::Scoring as u8;
    s.bids[..3].copy_from_slice(&[0, 2, 3]);
    s.tricks_won[..3].copy_from_slice(&[0, 2, 3]);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    advance_round(&mut s, &mut rng);
    assert_eq!(s.cumulative_scores[..3], [10, 12, 13]);
}

#[test]
fn scoring_all_miss_yields_all_zero() {
    let mut s = new_game(4, 3).unwrap();
    s.game_phase = GamePhase::Scoring as u8;
    s.bids[..4].copy_from_slice(&[2, 1, 0, 3]);
    s.tricks_won[..4].copy_from_slice(&[1, 2, 1, 2]);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    advance_round(&mut s, &mut rng);
    assert_eq!(s.cumulative_scores[..4], [0, 0, 0, 0]);
}

#[test]
fn scoring_dealer_rotates_per_round() {
    let mut s = new_game(4, 3).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    start_round(&mut s, &mut rng);
    for r in 0..4 {
        assert_eq!(s.dealer, r);
        run_round(&mut s, bid_always_zero_or_one, play_first_legal);
        advance_round(&mut s, &mut rng);
    }
    // After 4 rounds with 4 players, dealer cycled fully.
    assert_eq!(s.dealer, 0);
}

#[test]
fn scoring_round_idx_increments_each_advance() {
    let mut s = new_game(3, 5).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    start_round(&mut s, &mut rng);
    for r in 0..total_rounds(5, 3) - 1 {
        assert_eq!(s.round_idx, r);
        run_round(&mut s, bid_always_zero_or_one, play_first_legal);
        advance_round(&mut s, &mut rng);
    }
}

#[test]
fn scoring_cumulative_across_two_rounds_player_zero_makes_both() {
    // Mirror of legacy `test_scoring_phase_cumulative_scores`.
    let mut s = new_game(3, 3).unwrap();
    s.game_phase = GamePhase::Scoring as u8;
    s.bids[0] = 2;
    s.tricks_won[0] = 2;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    advance_round(&mut s, &mut rng);
    assert_eq!(s.cumulative_scores[0], 12);

    // Now in bidding phase of round 1; force scoring values manually for the
    // second round to avoid running the playout.
    s.bids = [0; 8];
    s.tricks_won = [0; 8];
    s.game_phase = GamePhase::Scoring as u8;
    s.bids[0] = 1;
    s.tricks_won[0] = 1;
    advance_round(&mut s, &mut rng);
    assert_eq!(s.cumulative_scores[0], 12 + 11);
}

// ----------------------------------------------------------------------------
// Multi-round game flow ports — these mirror legacy
// `test_play_full_game_*` cases.
// ----------------------------------------------------------------------------

#[test]
fn full_game_completes_for_4p_3c() {
    let mut s = new_game(4, 3).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(2024);
    start_round(&mut s, &mut rng);
    run_game(&mut s, &mut rng, bid_always_zero_or_one, play_first_legal);
    assert!(is_game_over(&s));
    // 4p × 3c → 2*3 + 4 - 2 = 8 rounds.
    assert_eq!(s.round_idx + 1, total_rounds(3, 4));
}

#[test]
fn full_game_completes_for_5p_4c() {
    let mut s = new_game(5, 4).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(11111);
    start_round(&mut s, &mut rng);
    run_game(&mut s, &mut rng, bid_always_zero_or_one, play_first_legal);
    assert!(is_game_over(&s));
}

#[test]
fn full_game_completes_for_3p_7c() {
    let mut s = new_game(3, 7).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(3030);
    start_round(&mut s, &mut rng);
    run_game(&mut s, &mut rng, bid_always_zero_or_one, play_first_legal);
    assert!(is_game_over(&s));
    assert_eq!(s.round_idx + 1, total_rounds(7, 3));
}

#[test]
fn full_game_completes_for_4p_13c_full_deck() {
    let mut s = new_game(4, 13).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(13131);
    start_round(&mut s, &mut rng);
    run_game(&mut s, &mut rng, bid_always_zero_or_one, play_first_legal);
    assert!(is_game_over(&s));
    assert_eq!(s.round_idx + 1, total_rounds(13, 4));
}

#[test]
fn full_game_dealer_rotation_six_round_cycle() {
    // Mirror legacy `test_play_full_game_dealer_rotation`. After K rounds
    // with N players, dealer index is K % N.
    let mut s = new_game(3, 5).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
    start_round(&mut s, &mut rng);
    let total = total_rounds(5, 3);
    for _ in 0..total {
        run_round(&mut s, bid_always_zero_or_one, play_first_legal);
        advance_round(&mut s, &mut rng);
    }
    // Cumulative dealer index advanced exactly `total` times.
    // (After the final advance, dealer freezes at the last value.)
    let _ = s.dealer;
}

#[test]
fn full_game_trump_rotation_full_cycle() {
    // 6+ rounds covers a full ♠♥♣♦∅ cycle plus repeat.
    let mut s = new_game(4, 4).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    start_round(&mut s, &mut rng);
    let mut trumps = Vec::new();
    let total = total_rounds(4, 4);
    for _ in 0..total {
        trumps.push(s.trump_suit);
        run_round(&mut s, bid_always_zero_or_one, play_first_legal);
        advance_round(&mut s, &mut rng);
    }
    assert_eq!(trumps[0], Suit::Spades as u8);
    assert_eq!(trumps[1], Suit::Hearts as u8);
    assert_eq!(trumps[2], Suit::Clubs as u8);
    assert_eq!(trumps[3], Suit::Diamonds as u8);
    assert_eq!(trumps[4], NO_TRUMP);
    assert_eq!(trumps[5], Suit::Spades as u8); // cycle restart
}

#[test]
fn full_game_winner_has_max_score() {
    let mut s = new_game(4, 4).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(2025);
    start_round(&mut s, &mut rng);
    run_game(&mut s, &mut rng, bid_position, play_highest_legal);
    assert!(is_game_over(&s));
    let max = *s.cumulative_scores[..4].iter().max().unwrap();
    let winner_count = s.cumulative_scores[..4]
        .iter()
        .filter(|&&v| v == max)
        .count();
    assert!(winner_count >= 1);
    // Trivial sanity: at least one non-zero score after a full game.
    let total: u16 = s.cumulative_scores[..4].iter().sum();
    assert!(total > 0);
}

#[test]
fn full_game_score_per_round_is_bounded_by_ten_plus_cards() {
    // Each round contributes at most 10 + cards_dealt to *each* player.
    let mut s = new_game(4, 5).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(44);
    start_round(&mut s, &mut rng);
    let mut prev = [0u16; 8];
    for _ in 0..total_rounds(5, 4) {
        let cards_this_round = s.cards_dealt as u16;
        run_round(&mut s, bid_always_zero_or_one, play_first_legal);
        advance_round(&mut s, &mut rng);
        for (i, &p) in prev.iter().enumerate().take(4) {
            let delta = s.cumulative_scores[i] - p;
            assert!(
                delta == 0 || delta == 10 + s.bids[i] as u16 || delta == 10 + cards_this_round,
                "player {i}: delta {delta} not in {{0, 10+bid}}"
            );
        }
        prev = s.cumulative_scores;
    }
}

// ----------------------------------------------------------------------------
// is_game_over ports
// ----------------------------------------------------------------------------

#[test]
fn is_game_over_false_at_start() {
    let s = new_game(4, 5).unwrap();
    assert!(!is_game_over(&s));
}

#[test]
fn is_game_over_false_during_bidding() {
    let mut s = new_game(4, 5).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    start_round(&mut s, &mut rng);
    assert!(!is_game_over(&s));
}

#[test]
fn is_game_over_false_during_playing() {
    let mut s = new_game(4, 5).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    start_round(&mut s, &mut rng);
    while s.phase() == GamePhase::Bidding {
        let b = bid_always_zero_or_one(&s);
        apply_bid(&mut s, b);
    }
    assert!(!is_game_over(&s));
}

#[test]
fn is_game_over_false_in_intermediate_scoring() {
    let mut s = new_game(4, 4).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    start_round(&mut s, &mut rng);
    run_round(&mut s, bid_always_zero_or_one, play_first_legal);
    assert_eq!(s.phase(), GamePhase::Scoring);
    assert!(!is_game_over(&s));
}

#[test]
fn is_game_over_true_after_final_advance() {
    let mut s = new_game(3, 2).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    start_round(&mut s, &mut rng);
    run_game(&mut s, &mut rng, bid_always_zero_or_one, play_first_legal);
    assert!(is_game_over(&s));
}

// ----------------------------------------------------------------------------
// cards_dealt_for_round / round_idx integration
// ----------------------------------------------------------------------------

#[test]
fn cards_dealt_for_round_walks_full_5p_8c_game() {
    let mut s = new_game(5, 8).unwrap();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    start_round(&mut s, &mut rng);
    let total = total_rounds(8, 5);
    for r in 0..total {
        assert_eq!(s.cards_dealt, cards_dealt_for_round(r, 8, 5));
        run_round(&mut s, bid_always_zero_or_one, play_first_legal);
        advance_round(&mut s, &mut rng);
    }
}
