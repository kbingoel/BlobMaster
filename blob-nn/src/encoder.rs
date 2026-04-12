//! Entity encoder — Section 2: raw feature vectors for neural network input.
//!
//! Session 2.1: Hand card token encoder.
//! Produces a `Vec<[f32; 30]>` of per-card feature vectors for the
//! perspective player's hand, emitted in `Hand::iter()` order (ascending
//! card index — the canonical action order).

use blob_engine::card::NUM_SUITS;
use blob_engine::hand::Hand;
use blob_engine::state::BlobState;

/// Dimensionality of a hand-card token.
pub const HAND_CARD_DIM: usize = 30;

/// Encode the perspective player's hand cards into feature vectors.
///
/// Each card in the hand produces a 30-dimensional feature vector:
/// - `[0..16)`: rank one-hot (13 values + 3 padding)
/// - `[16..24)`: suit one-hot (4 values + 4 padding)
/// - `[24]`: is_trump
/// - `[25]`: suit_count_in_hand
/// - `[26]`: is_highest_in_suit (among cards of that suit still in play)
/// - `[27]`: is_lowest_in_suit (among cards of that suit still in play)
/// - `[28]`: cards_above_remaining (same suit, higher rank, not in hand, not played)
/// - `[29]`: cards_below_remaining (same suit, lower rank, not in hand, not played)
///
/// Tokens are emitted in `Hand::iter()` order (ascending card index).
/// This is the canonical action order used by the playing head and MCTS.
pub fn encode_hand_cards(state: &BlobState, perspective: u8) -> Vec<[f32; HAND_CARD_DIM]> {
    let hand = Hand::new(state.hands[perspective as usize]);
    let trump = state.trump_suit;
    let played = state.played_this_round;
    let deck_mask: u64 = (1u64 << 52) - 1;

    // Cards not in own hand and not yet played — "unknown remaining" for
    // cards_above_remaining / cards_below_remaining.
    let unknown_remaining = !hand.bits() & !played & deck_mask;

    // Cards still alive (not yet played) — includes our hand.
    // Used for is_highest_in_suit / is_lowest_in_suit.
    let alive = !played & deck_mask;

    let mut tokens = Vec::with_capacity(hand.count() as usize);

    for card in hand.iter() {
        let mut feat = [0.0f32; HAND_CARD_DIM];
        let suit = card.suit();
        let rank = card.rank();
        let idx = card.index();

        // Rank one-hot: [0..16), 13 values + 3 padding slots.
        feat[rank as usize] = 1.0;

        // Suit one-hot: [16..24), 4 values + 4 padding slots.
        feat[16 + suit.index() as usize] = 1.0;

        // is_trump: [24].
        let is_trump = trump < NUM_SUITS && suit.index() == trump;
        feat[24] = if is_trump { 1.0 } else { 0.0 };

        // suit_count_in_hand: [25].
        feat[25] = hand.cards_of_suit(suit).count_ones() as f32;

        // Precompute masks for cards of the same suit strictly above/below.
        let suit_mask = suit.mask();
        let above_in_suit = suit_mask & !((1u64 << (idx + 1)) - 1);
        let below_in_suit = if idx == 0 { 0 } else { suit_mask & ((1u64 << idx) - 1) };
        let alive_of_suit = alive & suit_mask;

        // is_highest_in_suit: [26]. No alive card of same suit outranks this.
        feat[26] = if (alive_of_suit & above_in_suit) == 0 {
            1.0
        } else {
            0.0
        };

        // is_lowest_in_suit: [27]. No alive card of same suit is lower-ranked.
        feat[27] = if (alive_of_suit & below_in_suit) == 0 {
            1.0
        } else {
            0.0
        };

        // cards_above_remaining: [28]. Unknown remaining in same suit, higher rank.
        feat[28] = (unknown_remaining & above_in_suit).count_ones() as f32;

        // cards_below_remaining: [29]. Unknown remaining in same suit, lower rank.
        feat[29] = (unknown_remaining & below_in_suit).count_ones() as f32;

        tokens.push(feat);
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use blob_engine::card::{Card, Suit};
    use blob_engine::round::NO_TRUMP;
    use blob_engine::state::GamePhase;

    /// Helper: make a Card from suit and rank.
    fn c(suit: Suit, rank: u8) -> Card {
        Card::new(suit, rank)
    }

    /// Helper: build a hand from a slice of cards.
    fn make_hand(cards: &[Card]) -> Hand {
        let mut h = Hand::EMPTY;
        for &card in cards {
            h.add(card);
        }
        h
    }

    /// Helper: build a minimal BlobState with the given hand, trump, and
    /// played_this_round bitmask.
    fn test_state(hand: Hand, trump: u8, played: u64, num_players: u8) -> BlobState {
        let mut s = BlobState::empty();
        s.hands[0] = hand.bits();
        s.trump_suit = trump;
        s.played_this_round = played;
        s.num_players = num_players;
        s.game_phase = GamePhase::Playing as u8;
        s
    }

    // ---------------------------------------------------------------
    // Basic structure tests
    // ---------------------------------------------------------------

    #[test]
    fn empty_hand_produces_no_tokens() {
        let s = test_state(Hand::EMPTY, Suit::Spades as u8, 0, 4);
        let tokens = encode_hand_cards(&s, 0);
        assert!(tokens.is_empty());
    }

    #[test]
    fn token_count_matches_hand_size() {
        let hand = make_hand(&[
            c(Suit::Spades, 0),
            c(Suit::Hearts, 5),
            c(Suit::Clubs, 12),
        ]);
        let s = test_state(hand, Suit::Spades as u8, 0, 4);
        let tokens = encode_hand_cards(&s, 0);
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn emit_order_is_ascending_card_index() {
        // Hand with cards in non-ascending insertion order.
        let hand = make_hand(&[
            c(Suit::Diamonds, 12), // idx 51
            c(Suit::Spades, 0),    // idx 0
            c(Suit::Hearts, 6),    // idx 19
        ]);
        let s = test_state(hand, Suit::Spades as u8, 0, 4);
        let tokens = encode_hand_cards(&s, 0);

        // Token 0: Spades 0 (rank 0)
        assert_eq!(tokens[0][0], 1.0, "first token should be rank 0");
        assert_eq!(tokens[0][16], 1.0, "first token should be Spades");

        // Token 1: Hearts 6 (rank 6)
        assert_eq!(tokens[1][6], 1.0, "second token should be rank 6");
        assert_eq!(tokens[1][17], 1.0, "second token should be Hearts");

        // Token 2: Diamonds 12 (rank 12)
        assert_eq!(tokens[2][12], 1.0, "third token should be rank 12");
        assert_eq!(tokens[2][19], 1.0, "third token should be Diamonds");
    }

    // ---------------------------------------------------------------
    // One-hot encoding tests
    // ---------------------------------------------------------------

    #[test]
    fn rank_one_hot_has_exactly_one_bit() {
        let hand = make_hand(&[c(Suit::Clubs, 7)]); // rank 7
        let s = test_state(hand, NO_TRUMP, 0, 4);
        let feat = &encode_hand_cards(&s, 0)[0];

        // Exactly one 1.0 in [0..16), at position 7.
        for i in 0..16 {
            let expected = if i == 7 { 1.0 } else { 0.0 };
            assert_eq!(feat[i], expected, "rank one-hot[{i}]");
        }
    }

    #[test]
    fn suit_one_hot_has_exactly_one_bit() {
        let hand = make_hand(&[c(Suit::Clubs, 3)]); // suit index 2
        let s = test_state(hand, NO_TRUMP, 0, 4);
        let feat = &encode_hand_cards(&s, 0)[0];

        // Exactly one 1.0 in [16..24), at position 18.
        for i in 16..24 {
            let expected = if i == 18 { 1.0 } else { 0.0 };
            assert_eq!(feat[i], expected, "suit one-hot[{i}]");
        }
    }

    #[test]
    fn padding_slots_are_zero() {
        let hand = make_hand(&[c(Suit::Diamonds, 12)]); // rank 12, suit 3
        let s = test_state(hand, NO_TRUMP, 0, 4);
        let feat = &encode_hand_cards(&s, 0)[0];

        // Rank padding: [13..16)
        assert_eq!(feat[13], 0.0);
        assert_eq!(feat[14], 0.0);
        assert_eq!(feat[15], 0.0);
        // Suit padding: [20..24)
        assert_eq!(feat[20], 0.0);
        assert_eq!(feat[21], 0.0);
        assert_eq!(feat[22], 0.0);
        assert_eq!(feat[23], 0.0);
    }

    // ---------------------------------------------------------------
    // Trump detection
    // ---------------------------------------------------------------

    #[test]
    fn is_trump_when_suit_matches() {
        let hand = make_hand(&[c(Suit::Hearts, 5)]);
        let s = test_state(hand, Suit::Hearts as u8, 0, 4);
        assert_eq!(encode_hand_cards(&s, 0)[0][24], 1.0);
    }

    #[test]
    fn is_not_trump_when_suit_differs() {
        let hand = make_hand(&[c(Suit::Hearts, 5)]);
        let s = test_state(hand, Suit::Spades as u8, 0, 4);
        assert_eq!(encode_hand_cards(&s, 0)[0][24], 0.0);
    }

    #[test]
    fn is_not_trump_in_no_trump_round() {
        let hand = make_hand(&[c(Suit::Spades, 12)]);
        let s = test_state(hand, NO_TRUMP, 0, 4);
        assert_eq!(encode_hand_cards(&s, 0)[0][24], 0.0);
    }

    // ---------------------------------------------------------------
    // suit_count_in_hand
    // ---------------------------------------------------------------

    #[test]
    fn suit_count_reflects_all_cards_of_suit_in_hand() {
        let hand = make_hand(&[
            c(Suit::Spades, 0),
            c(Suit::Spades, 5),
            c(Suit::Spades, 12),
            c(Suit::Hearts, 3),
        ]);
        let s = test_state(hand, NO_TRUMP, 0, 4);
        let tokens = encode_hand_cards(&s, 0);

        // All three spades tokens should report suit_count = 3.
        assert_eq!(tokens[0][25], 3.0); // ♠0
        assert_eq!(tokens[1][25], 3.0); // ♠5
        assert_eq!(tokens[2][25], 3.0); // ♠12
        // Hearts token should report suit_count = 1.
        assert_eq!(tokens[3][25], 1.0); // ♥3
    }

    // ---------------------------------------------------------------
    // is_highest / is_lowest in suit (no cards played)
    // ---------------------------------------------------------------

    #[test]
    fn ace_is_highest_when_no_cards_played() {
        // Ace of Spades (rank 12) — no cards played, so all 13 spades alive.
        // It's the highest rank in spades.
        let hand = make_hand(&[c(Suit::Spades, 12)]);
        let s = test_state(hand, NO_TRUMP, 0, 4);
        let feat = &encode_hand_cards(&s, 0)[0];
        assert_eq!(feat[26], 1.0, "Ace should be highest");
        assert_eq!(feat[27], 0.0, "Ace should not be lowest (all 13 alive)");
    }

    #[test]
    fn two_is_lowest_when_no_cards_played() {
        let hand = make_hand(&[c(Suit::Spades, 0)]);
        let s = test_state(hand, NO_TRUMP, 0, 4);
        let feat = &encode_hand_cards(&s, 0)[0];
        assert_eq!(feat[26], 0.0, "2 should not be highest (all 13 alive)");
        assert_eq!(feat[27], 1.0, "2 should be lowest");
    }

    #[test]
    fn mid_rank_is_neither_extreme_in_full_deck() {
        let hand = make_hand(&[c(Suit::Hearts, 6)]); // 8 of Hearts
        let s = test_state(hand, NO_TRUMP, 0, 4);
        let feat = &encode_hand_cards(&s, 0)[0];
        assert_eq!(feat[26], 0.0, "mid-rank should not be highest");
        assert_eq!(feat[27], 0.0, "mid-rank should not be lowest");
    }

    // ---------------------------------------------------------------
    // is_highest / is_lowest after cards played
    // ---------------------------------------------------------------

    #[test]
    fn becomes_highest_after_higher_cards_played() {
        // Hand: 10 of Spades (rank 8). Play all spades ranks 9..12.
        let hand = make_hand(&[c(Suit::Spades, 8)]);
        let played = c(Suit::Spades, 9).bit()
            | c(Suit::Spades, 10).bit()
            | c(Suit::Spades, 11).bit()
            | c(Suit::Spades, 12).bit();
        let s = test_state(hand, NO_TRUMP, played, 4);
        let feat = &encode_hand_cards(&s, 0)[0];
        assert_eq!(feat[26], 1.0, "should be highest after higher cards played");
    }

    #[test]
    fn becomes_lowest_after_lower_cards_played() {
        // Hand: 5 of Clubs (rank 3). Play all clubs ranks 0..2.
        let hand = make_hand(&[c(Suit::Clubs, 3)]);
        let played =
            c(Suit::Clubs, 0).bit() | c(Suit::Clubs, 1).bit() | c(Suit::Clubs, 2).bit();
        let s = test_state(hand, NO_TRUMP, played, 4);
        let feat = &encode_hand_cards(&s, 0)[0];
        assert_eq!(feat[27], 1.0, "should be lowest after lower cards played");
    }

    #[test]
    fn sole_survivor_is_both_highest_and_lowest() {
        // Only one spade alive (the rest played). That card is both extremes.
        let the_card = c(Suit::Spades, 6);
        let hand = make_hand(&[the_card]);
        let mut played = 0u64;
        for r in 0..13u8 {
            if r != 6 {
                played |= Card::new(Suit::Spades, r).bit();
            }
        }
        let s = test_state(hand, NO_TRUMP, played, 4);
        let feat = &encode_hand_cards(&s, 0)[0];
        assert_eq!(feat[26], 1.0, "sole survivor is highest");
        assert_eq!(feat[27], 1.0, "sole survivor is lowest");
    }

    // ---------------------------------------------------------------
    // cards_above_remaining / cards_below_remaining
    // ---------------------------------------------------------------

    #[test]
    fn remaining_counts_with_no_cards_played() {
        // Hand: 7 of Spades (rank 5). No cards played.
        // cards_above_remaining = cards with rank > 5 in spades, not in hand = 7 (ranks 6..12)
        // cards_below_remaining = cards with rank < 5 in spades, not in hand = 5 (ranks 0..4)
        let hand = make_hand(&[c(Suit::Spades, 5)]);
        let s = test_state(hand, NO_TRUMP, 0, 4);
        let feat = &encode_hand_cards(&s, 0)[0];
        assert_eq!(feat[28], 7.0, "cards_above_remaining");
        assert_eq!(feat[29], 5.0, "cards_below_remaining");
    }

    #[test]
    fn remaining_counts_exclude_played_cards() {
        // Hand: 7 of Spades (rank 5). Play ranks 6 and 7 (two cards above).
        let hand = make_hand(&[c(Suit::Spades, 5)]);
        let played = c(Suit::Spades, 6).bit() | c(Suit::Spades, 7).bit();
        let s = test_state(hand, NO_TRUMP, played, 4);
        let feat = &encode_hand_cards(&s, 0)[0];
        // Originally 7 above (ranks 6..12); 2 played → 5 remaining.
        assert_eq!(feat[28], 5.0, "cards_above_remaining after 2 played");
        // Below unchanged (no below-rank cards played).
        assert_eq!(feat[29], 5.0, "cards_below_remaining unchanged");
    }

    #[test]
    fn remaining_counts_exclude_own_hand() {
        // Hand: ranks 5, 8, 12 of Spades. Perspective on rank 8:
        // cards_above_remaining = ranks above 8, not in hand = {9, 10, 11} = 3
        //   (rank 12 is in hand, so excluded)
        // cards_below_remaining = ranks below 8, not in hand = {0,1,2,3,4,6,7} = 7
        //   (rank 5 is in hand, so excluded)
        let hand = make_hand(&[
            c(Suit::Spades, 5),
            c(Suit::Spades, 8),
            c(Suit::Spades, 12),
        ]);
        let s = test_state(hand, NO_TRUMP, 0, 4);
        let tokens = encode_hand_cards(&s, 0);
        // tokens[1] is rank 8 (middle card in ascending order: 5, 8, 12).
        let feat = &tokens[1];
        assert_eq!(feat[28], 3.0, "above remaining excludes hand cards");
        assert_eq!(feat[29], 7.0, "below remaining excludes hand cards");
    }

    #[test]
    fn ace_has_zero_above_remaining() {
        let hand = make_hand(&[c(Suit::Diamonds, 12)]);
        let s = test_state(hand, NO_TRUMP, 0, 4);
        let feat = &encode_hand_cards(&s, 0)[0];
        assert_eq!(feat[28], 0.0, "Ace has no cards above");
        assert_eq!(feat[29], 12.0, "Ace has 12 cards below (not in hand)");
    }

    #[test]
    fn two_has_zero_below_remaining() {
        let hand = make_hand(&[c(Suit::Clubs, 0)]);
        let s = test_state(hand, NO_TRUMP, 0, 4);
        let feat = &encode_hand_cards(&s, 0)[0];
        assert_eq!(feat[28], 12.0, "2 has 12 cards above (not in hand)");
        assert_eq!(feat[29], 0.0, "2 has no cards below");
    }

    // ---------------------------------------------------------------
    // Integrated scenario: mid-game state after several tricks
    // ---------------------------------------------------------------

    #[test]
    fn mid_game_features_after_tricks() {
        // Scenario: 4 players, trump = Hearts. Player 0 holds:
        //   ♠5 (rank 3), ♥J (rank 9), ♦A (rank 12)
        //
        // Played this round (by various players):
        //   ♠2 (rank 0), ♠K (rank 11), ♠A (rank 12), ♥3 (rank 1), ♦2 (rank 0)
        let hand = make_hand(&[
            c(Suit::Spades, 3),   // idx 3
            c(Suit::Hearts, 9),   // idx 22
            c(Suit::Diamonds, 12), // idx 51
        ]);
        let played = c(Suit::Spades, 0).bit()
            | c(Suit::Spades, 11).bit()
            | c(Suit::Spades, 12).bit()
            | c(Suit::Hearts, 1).bit()
            | c(Suit::Diamonds, 0).bit();
        let s = test_state(hand, Suit::Hearts as u8, played, 4);
        let tokens = encode_hand_cards(&s, 0);
        assert_eq!(tokens.len(), 3);

        // Token 0: ♠5 (rank 3). Spades alive: {1,2,3,4,5,6,7,8,9,10} (0,11,12 played).
        // is_trump = false (suit is Spades, trump is Hearts).
        // suit_count_in_hand = 1 (only ♠5 in hand).
        // is_highest = 0 (ranks 4..10 alive above rank 3).
        // is_lowest = 0 (ranks 1, 2 alive below rank 3).
        // cards_above_remaining: ranks above 3, in spades, not in hand, not played.
        //   Ranks 4..10 not in hand, not played → 7 cards.
        // cards_below_remaining: ranks below 3, in spades, not in hand, not played.
        //   Ranks 1, 2 not in hand, not played → 2 cards.
        let f0 = &tokens[0];
        assert_eq!(f0[3], 1.0, "rank 3 one-hot");
        assert_eq!(f0[16], 1.0, "Spades one-hot");
        assert_eq!(f0[24], 0.0, "not trump");
        assert_eq!(f0[25], 1.0, "suit_count_in_hand");
        assert_eq!(f0[26], 0.0, "not highest");
        assert_eq!(f0[27], 0.0, "not lowest");
        assert_eq!(f0[28], 7.0, "cards_above_remaining");
        assert_eq!(f0[29], 2.0, "cards_below_remaining");

        // Token 1: ♥J (rank 9). Hearts alive: all except rank 1 (played).
        // is_trump = true.
        // suit_count_in_hand = 1.
        // is_highest = 0 (ranks 10, 11, 12 alive above).
        // is_lowest = 0 (rank 0 alive below, and 2..8).
        // cards_above_remaining: ranks 10, 11, 12 in hearts, not in hand, not played → 3.
        // cards_below_remaining: ranks 0, 2..8 in hearts, not in hand, not played → 9.
        //   (rank 1 played, so 11 total hearts - 1 played - 1 in hand = 11 not in hand, but
        //    ranks below 9: {0,2,3,4,5,6,7,8} = 8 ranks. rank 1 is played, so 8 remaining below.)
        //   Wait: ranks below 9 = {0,1,2,3,4,5,6,7,8} = 9 ranks. rank 1 is played. Not in hand: all 9.
        //   Not played: 9 - 1 = 8. So cards_below_remaining = 8.
        let f1 = &tokens[1];
        assert_eq!(f1[9], 1.0, "rank 9");
        assert_eq!(f1[17], 1.0, "Hearts");
        assert_eq!(f1[24], 1.0, "is trump");
        assert_eq!(f1[25], 1.0, "suit_count_in_hand");
        assert_eq!(f1[26], 0.0, "not highest (10,11,12 alive)");
        assert_eq!(f1[27], 0.0, "not lowest");
        assert_eq!(f1[28], 3.0, "cards_above_remaining");
        assert_eq!(f1[29], 8.0, "cards_below_remaining");

        // Token 2: ♦A (rank 12). Diamonds alive: all except rank 0 (played).
        // is_trump = false.
        // suit_count_in_hand = 1.
        // is_highest = 1 (rank 12 is highest, no higher alive).
        // is_lowest = 0 (ranks 1..11 alive below).
        // cards_above_remaining = 0 (no rank above 12).
        // cards_below_remaining: ranks 1..11, not in hand, not played → 11.
        let f2 = &tokens[2];
        assert_eq!(f2[12], 1.0, "rank 12");
        assert_eq!(f2[19], 1.0, "Diamonds");
        assert_eq!(f2[24], 0.0, "not trump");
        assert_eq!(f2[25], 1.0, "suit_count_in_hand");
        assert_eq!(f2[26], 1.0, "highest in suit");
        assert_eq!(f2[27], 0.0, "not lowest");
        assert_eq!(f2[28], 0.0, "cards_above_remaining");
        assert_eq!(f2[29], 11.0, "cards_below_remaining");
    }

    // ---------------------------------------------------------------
    // Perspective parameter
    // ---------------------------------------------------------------

    #[test]
    fn encodes_correct_player_hand() {
        let mut s = BlobState::empty();
        s.num_players = 3;
        s.game_phase = GamePhase::Playing as u8;
        // Player 0: ♠A
        s.hands[0] = c(Suit::Spades, 12).bit();
        // Player 1: ♥2, ♥3
        s.hands[1] = c(Suit::Hearts, 0).bit() | c(Suit::Hearts, 1).bit();
        // Player 2: ♦K
        s.hands[2] = c(Suit::Diamonds, 11).bit();
        s.trump_suit = NO_TRUMP;

        assert_eq!(encode_hand_cards(&s, 0).len(), 1);
        assert_eq!(encode_hand_cards(&s, 1).len(), 2);
        assert_eq!(encode_hand_cards(&s, 2).len(), 1);

        // Player 1's first token should be ♥2.
        let p1 = encode_hand_cards(&s, 1);
        assert_eq!(p1[0][0], 1.0, "rank 0 for ♥2");
        assert_eq!(p1[0][17], 1.0, "Hearts suit");
    }

    // ---------------------------------------------------------------
    // Feature dimension sanity
    // ---------------------------------------------------------------

    #[test]
    fn all_features_within_expected_ranges() {
        // 7-card hand, some cards played.
        let hand = make_hand(&[
            c(Suit::Spades, 0),
            c(Suit::Spades, 6),
            c(Suit::Spades, 12),
            c(Suit::Hearts, 3),
            c(Suit::Hearts, 10),
            c(Suit::Clubs, 7),
            c(Suit::Diamonds, 1),
        ]);
        let played = c(Suit::Spades, 3).bit()
            | c(Suit::Hearts, 12).bit()
            | c(Suit::Clubs, 0).bit()
            | c(Suit::Diamonds, 8).bit();
        let s = test_state(hand, Suit::Clubs as u8, played, 5);
        let tokens = encode_hand_cards(&s, 0);
        assert_eq!(tokens.len(), 7);

        for (i, feat) in tokens.iter().enumerate() {
            // Exactly one rank bit set in [0..13).
            let rank_sum: f32 = feat[0..13].iter().sum();
            assert_eq!(rank_sum, 1.0, "token {i}: exactly one rank bit");
            // Padding bits zero.
            assert_eq!(feat[13], 0.0, "token {i}: rank padding[13]");
            assert_eq!(feat[14], 0.0, "token {i}: rank padding[14]");
            assert_eq!(feat[15], 0.0, "token {i}: rank padding[15]");

            // Exactly one suit bit set in [16..20).
            let suit_sum: f32 = feat[16..20].iter().sum();
            assert_eq!(suit_sum, 1.0, "token {i}: exactly one suit bit");
            // Suit padding zero.
            for j in 20..24 {
                assert_eq!(feat[j], 0.0, "token {i}: suit padding[{j}]");
            }

            // Binary features are 0 or 1.
            assert!(
                feat[24] == 0.0 || feat[24] == 1.0,
                "token {i}: is_trump binary"
            );
            assert!(
                feat[26] == 0.0 || feat[26] == 1.0,
                "token {i}: is_highest binary"
            );
            assert!(
                feat[27] == 0.0 || feat[27] == 1.0,
                "token {i}: is_lowest binary"
            );

            // Count features are non-negative integers ≤ 12.
            assert!(feat[25] >= 1.0, "token {i}: suit_count ≥ 1");
            assert!(feat[25] <= 13.0, "token {i}: suit_count ≤ 13");
            assert!(feat[28] >= 0.0, "token {i}: above_remaining ≥ 0");
            assert!(feat[28] <= 12.0, "token {i}: above_remaining ≤ 12");
            assert!(feat[29] >= 0.0, "token {i}: below_remaining ≥ 0");
            assert!(feat[29] <= 12.0, "token {i}: below_remaining ≤ 12");
        }
    }

    // ---------------------------------------------------------------
    // Full-game integration: use engine to set up a real mid-game state
    // ---------------------------------------------------------------

    #[test]
    fn encode_after_real_game_play() {
        use blob_engine::bidding::{apply_bid, legal_bids};
        use blob_engine::dealing::start_round;
        use blob_engine::game::new_game;
        use blob_engine::playing::{apply_play, legal_plays};
        use rand_xoshiro::rand_core::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;

        let mut s = new_game(4, 5).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        start_round(&mut s, &mut rng);

        // Complete bidding.
        while s.phase() == GamePhase::Bidding {
            let mask = legal_bids(&s);
            let bid = (0..=13u8).find(|b| (mask >> b) & 1 == 1).unwrap();
            apply_bid(&mut s, bid);
        }

        // Play two tricks.
        for _ in 0..2 {
            for _ in 0..s.num_players {
                let mask = legal_plays(&s);
                let card = mask.trailing_zeros() as u8;
                apply_play(&mut s, card);
            }
        }

        // Now encode from current player's perspective.
        let perspective = s.current_player;
        let tokens = encode_hand_cards(&s, perspective);
        let hand = Hand::new(s.hands[perspective as usize]);

        // Token count should match remaining hand size (5 - 2 = 3).
        assert_eq!(tokens.len(), hand.count() as usize);
        assert_eq!(tokens.len(), 3, "5 dealt - 2 tricks played = 3 cards");

        // Verify emit order matches Hand::iter().
        let hand_cards: Vec<Card> = hand.iter().collect();
        for (i, card) in hand_cards.iter().enumerate() {
            let feat = &tokens[i];
            assert_eq!(
                feat[card.rank() as usize],
                1.0,
                "token {i} rank mismatch"
            );
            assert_eq!(
                feat[16 + card.suit().index() as usize],
                1.0,
                "token {i} suit mismatch"
            );
        }

        // played_this_round should have 8 bits set (2 tricks × 4 players).
        assert_eq!(s.played_this_round.count_ones(), 8);
    }
}
