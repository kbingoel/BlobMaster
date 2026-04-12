//! Entity encoder — Section 2: raw feature vectors for neural network input.
//!
//! Session 2.1: Hand card token encoder.
//! Produces a `Vec<[f32; 30]>` of per-card feature vectors for the
//! perspective player's hand, emitted in `Hand::iter()` order (ascending
//! card index — the canonical action order).
//!
//! Session 2.2: Played card tokens and player state tokens.
//! Produces chronologically-ordered 48-dim played card tokens and
//! per-player 29-dim state tokens with void detection.

use blob_engine::card::{Card, NUM_SUITS};
use blob_engine::hand::Hand;
use blob_engine::round::total_rounds;
use blob_engine::state::{BlobState, MAX_PLAYERS};

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

/// Dimensionality of a played-card token.
pub const PLAYED_CARD_DIM: usize = 48;

/// Dimensionality of a player-state token.
pub const PLAYER_STATE_DIM: usize = 29;

/// A played-card token with its chronological position index.
///
/// The `chrono_index` (0–51) is used by the neural network to look up a
/// learned chronological embedding (Session 3.1, 52×128 table).
#[derive(Debug, Clone)]
pub struct PlayedCardToken {
    pub features: [f32; PLAYED_CARD_DIM],
    pub chrono_index: u8,
}

/// Encode all played cards into 48-dim feature vectors in chronological order.
///
/// Each played card token:
/// - `[0..16)`: rank one-hot (13 values + 3 padding)
/// - `[16..24)`: suit one-hot (4 values + 4 padding)
/// - `[24..40)`: player one-hot (up to 8 values + 8 padding)
/// - `[40]`: trick_number (normalized by cards_dealt)
/// - `[41]`: position_in_trick (normalized to \[0, 1\])
/// - `[42]`: was_lead
/// - `[43]`: followed_suit (card suit == led suit)
/// - `[44]`: is_trump_play
/// - `[45]`: trick_complete
/// - `[46]`: won_trick (only for the winning card of a completed trick)
/// - `[47]`: is_current_trick
///
/// Iterates `trick_history[0..tricks_completed]` then current trick's
/// `trick_play_order[0..trick_cards_played]` in strict chronological order.
pub fn encode_played_cards(state: &BlobState) -> Vec<PlayedCardToken> {
    let np = state.num_players as usize;
    let trump = state.trump_suit;
    let cd = state.cards_dealt.max(1) as f32;
    let pos_norm = (state.num_players.saturating_sub(1)).max(1) as f32;

    let total_played = state.tricks_completed as usize * np + state.trick_cards_played as usize;
    let mut tokens = Vec::with_capacity(total_played);
    let mut chrono: u8 = 0;

    // Completed tricks.
    for t in 0..state.tricks_completed as usize {
        let rec = &state.trick_history[t];
        for i in 0..rec.num_played as usize {
            let (player, card_idx) = rec.cards[i];
            let card = Card::from_index_unchecked(card_idx);
            let mut feat = [0.0f32; PLAYED_CARD_DIM];

            feat[card.rank() as usize] = 1.0;
            feat[16 + card.suit().index() as usize] = 1.0;
            feat[24 + player as usize] = 1.0;

            feat[40] = t as f32 / cd;
            feat[41] = i as f32 / pos_norm;
            feat[42] = if i == 0 { 1.0 } else { 0.0 };
            feat[43] = if card.suit().index() == rec.suit_led {
                1.0
            } else {
                0.0
            };
            feat[44] = if trump < NUM_SUITS && card.suit().index() == trump {
                1.0
            } else {
                0.0
            };
            feat[45] = 1.0; // trick_complete
            feat[46] = if player == rec.winner { 1.0 } else { 0.0 };
            feat[47] = 0.0; // is_current_trick

            tokens.push(PlayedCardToken {
                features: feat,
                chrono_index: chrono,
            });
            chrono += 1;
        }
    }

    // Current (in-progress) trick.
    if state.trick_cards_played > 0 {
        let led_card = Card::from_index_unchecked(state.trick_play_order[0]);
        let led_suit = led_card.suit().index();
        let trick_num = state.tricks_completed as usize;

        for i in 0..state.trick_cards_played as usize {
            let card_idx = state.trick_play_order[i];
            let card = Card::from_index_unchecked(card_idx);
            let player = (state.trick_leader + i as u8) % state.num_players;
            let mut feat = [0.0f32; PLAYED_CARD_DIM];

            feat[card.rank() as usize] = 1.0;
            feat[16 + card.suit().index() as usize] = 1.0;
            feat[24 + player as usize] = 1.0;

            feat[40] = trick_num as f32 / cd;
            feat[41] = i as f32 / pos_norm;
            feat[42] = if i == 0 { 1.0 } else { 0.0 };
            feat[43] = if card.suit().index() == led_suit {
                1.0
            } else {
                0.0
            };
            feat[44] = if trump < NUM_SUITS && card.suit().index() == trump {
                1.0
            } else {
                0.0
            };
            feat[45] = 0.0; // trick_complete (in progress)
            feat[46] = 0.0; // won_trick (not complete)
            feat[47] = 1.0; // is_current_trick

            tokens.push(PlayedCardToken {
                features: feat,
                chrono_index: chrono,
            });
            chrono += 1;
        }
    }

    tokens
}

/// Encode per-player state tokens (29 dims each, one per player).
///
/// Each player state token:
/// - `[0..16)`: player one-hot (up to 8 values + 8 padding)
/// - `[16]`: bid (normalized by cards_dealt)
/// - `[17]`: tricks_won (normalized by cards_dealt)
/// - `[18]`: tricks_needed (max(0, bid − tricks_won), normalized by cards_dealt)
/// - `[19]`: bid_status (−1.0 busted, 0.0 live, +1.0 met)
/// - `[20]`: is_dealer
/// - `[21]`: is_me (1.0 for perspective player)
/// - `[22]`: relative_position ((p − current_player) mod N, normalized to \[0, 1\])
/// - `[23]`: cumulative_score (normalized by theoretical ceiling)
/// - `[24]`: cards_in_hand (normalized by cards_dealt)
/// - `[25]`: void_spades
/// - `[26]`: void_hearts
/// - `[27]`: void_clubs
/// - `[28]`: void_diamonds
///
/// Void flags are precomputed by scanning played cards where
/// `followed_suit == 0 && was_lead == 0`, marking that player as void in
/// the led suit.
pub fn encode_player_states(
    state: &BlobState,
    perspective: u8,
) -> Vec<[f32; PLAYER_STATE_DIM]> {
    let np = state.num_players as usize;
    let cd = state.cards_dealt.max(1) as f32;
    let tricks_remaining = state.cards_dealt.saturating_sub(state.tricks_completed);

    // Void detection: voids[player][suit] = true when observed.
    let mut voids = [[false; NUM_SUITS as usize]; MAX_PLAYERS];

    // Scan completed tricks.
    for t in 0..state.tricks_completed as usize {
        let rec = &state.trick_history[t];
        let led_suit = rec.suit_led as usize;
        for i in 1..rec.num_played as usize {
            let (player, card_idx) = rec.cards[i];
            let card = Card::from_index_unchecked(card_idx);
            if card.suit().index() as usize != led_suit {
                voids[player as usize][led_suit] = true;
            }
        }
    }

    // Scan current in-progress trick.
    if state.trick_cards_played > 1 {
        let led_suit =
            Card::from_index_unchecked(state.trick_play_order[0]).suit().index() as usize;
        for i in 1..state.trick_cards_played as usize {
            let card = Card::from_index_unchecked(state.trick_play_order[i]);
            let player = (state.trick_leader + i as u8) % state.num_players;
            if card.suit().index() as usize != led_suit {
                voids[player as usize][led_suit] = true;
            }
        }
    }

    // Cumulative score normalization: theoretical ceiling.
    let total_r = total_rounds(state.start_cards.max(1), state.num_players.max(3)) as f32;
    let score_ceiling = total_r * (10.0 + state.start_cards as f32);

    let mut tokens = Vec::with_capacity(np);

    for p in 0..np {
        let mut feat = [0.0f32; PLAYER_STATE_DIM];
        let player_idx = p as u8;

        // Player one-hot [0..16).
        feat[p] = 1.0;

        // bid [16].
        feat[16] = state.bids[p] as f32 / cd;

        // tricks_won [17].
        feat[17] = state.tricks_won[p] as f32 / cd;

        // tricks_needed [18].
        let needed = state.bids[p].saturating_sub(state.tricks_won[p]);
        feat[18] = needed as f32 / cd;

        // bid_status [19]: -1 busted, 0 live, +1 met.
        feat[19] = if state.tricks_won[p] > state.bids[p] {
            -1.0
        } else if state.tricks_won[p] == state.bids[p] {
            1.0
        } else if tricks_remaining >= needed {
            0.0
        } else {
            -1.0
        };

        // is_dealer [20].
        feat[20] = if player_idx == state.dealer { 1.0 } else { 0.0 };

        // is_me [21].
        feat[21] = if player_idx == perspective { 1.0 } else { 0.0 };

        // relative_position [22].
        let rel = (player_idx + state.num_players - state.current_player) % state.num_players;
        feat[22] = rel as f32 / state.num_players as f32;

        // cumulative_score [23].
        feat[23] = if score_ceiling > 0.0 {
            state.cumulative_scores[p] as f32 / score_ceiling
        } else {
            0.0
        };

        // cards_in_hand [24].
        feat[24] = Hand::new(state.hands[p]).count() as f32 / cd;

        // void_spades [25].
        feat[25] = if voids[p][0] { 1.0 } else { 0.0 };
        // void_hearts [26].
        feat[26] = if voids[p][1] { 1.0 } else { 0.0 };
        // void_clubs [27].
        feat[27] = if voids[p][2] { 1.0 } else { 0.0 };
        // void_diamonds [28].
        feat[28] = if voids[p][3] { 1.0 } else { 0.0 };

        tokens.push(feat);
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use blob_engine::card::{Card, Suit};
    use blob_engine::round::NO_TRUMP;
    use blob_engine::state::{GamePhase, TrickRecord};

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

    // ===============================================================
    // Session 2.2 — Played card token tests
    // ===============================================================

    /// Helper: make a TrickRecord from play sequence.
    fn make_trick(cards: &[(u8, u8)], winner: u8, suit_led: u8) -> TrickRecord {
        let mut rec = TrickRecord::default();
        rec.num_played = cards.len() as u8;
        rec.winner = winner;
        rec.suit_led = suit_led;
        for (i, &(player, card_idx)) in cards.iter().enumerate() {
            rec.cards[i] = (player, card_idx);
        }
        rec
    }

    /// Helper: build a playing-phase state with completed tricks and an
    /// optional in-progress trick.
    fn state_with_tricks(
        num_players: u8,
        cards_dealt: u8,
        start_cards: u8,
        trump: u8,
        dealer: u8,
        tricks: &[TrickRecord],
        current_trick: &[(u8, u8)], // (player, card_idx)
    ) -> BlobState {
        let mut s = BlobState::empty();
        s.num_players = num_players;
        s.cards_dealt = cards_dealt;
        s.start_cards = start_cards;
        s.trump_suit = trump;
        s.dealer = dealer;
        s.game_phase = GamePhase::Playing as u8;
        s.tricks_completed = tricks.len() as u8;
        for (i, rec) in tricks.iter().enumerate() {
            s.trick_history[i] = *rec;
            s.tricks_won[rec.winner as usize] += 1;
        }
        if !current_trick.is_empty() {
            s.trick_leader = current_trick[0].0;
            s.trick_cards_played = current_trick.len() as u8;
            for (i, &(_player, card_idx)) in current_trick.iter().enumerate() {
                s.trick_play_order[i] = card_idx;
            }
            s.current_player = (s.trick_leader + current_trick.len() as u8) % num_players;
        } else if !tricks.is_empty() {
            let winner = tricks.last().unwrap().winner;
            s.trick_leader = winner;
            s.current_player = winner;
        } else {
            s.trick_leader = (dealer + 1) % num_players;
            s.current_player = (dealer + 1) % num_players;
        }
        // Compute played_this_round.
        for t in 0..s.tricks_completed as usize {
            let rec = &s.trick_history[t];
            for i in 0..rec.num_played as usize {
                s.played_this_round |= 1u64 << rec.cards[i].1;
            }
        }
        for i in 0..s.trick_cards_played as usize {
            s.played_this_round |= 1u64 << s.trick_play_order[i];
        }
        s
    }

    #[test]
    fn no_plays_produces_empty_played_tokens() {
        let s = state_with_tricks(4, 5, 5, Suit::Spades as u8, 0, &[], &[]);
        let tokens = encode_played_cards(&s);
        assert!(tokens.is_empty());
    }

    #[test]
    fn one_completed_trick_produces_correct_tokens() {
        // 4 players, trump=Spades, cards_dealt=5.
        // Trick 0: P1 leads ♥5(idx 16), P2 plays ♥7(idx 18),
        //          P3 plays ♣2(idx 26), P0 plays ♥K(idx 24).
        // Led suit: Hearts(1). Winner: P0 (♥K highest heart).
        let trick = make_trick(
            &[(1, 16), (2, 18), (3, 26), (0, 24)],
            0,                    // winner = player 0
            Suit::Hearts as u8,   // suit_led = Hearts
        );
        let s = state_with_tricks(4, 5, 5, Suit::Spades as u8, 0, &[trick], &[]);
        let tokens = encode_played_cards(&s);
        assert_eq!(tokens.len(), 4);

        // Token 0: P1, ♥5 (rank 3, suit Hearts=1).
        let f = &tokens[0].features;
        assert_eq!(f[3], 1.0, "rank 3 one-hot");
        assert_eq!(f[17], 1.0, "Hearts suit one-hot");
        assert_eq!(f[25], 1.0, "player 1 one-hot");
        assert_eq!(f[40], 0.0, "trick_number = 0/5");
        assert_eq!(f[41], 0.0, "position_in_trick = 0/3");
        assert_eq!(f[42], 1.0, "was_lead");
        assert_eq!(f[43], 1.0, "followed_suit (Hearts==Hearts)");
        assert_eq!(f[44], 0.0, "not trump (Hearts!=Spades)");
        assert_eq!(f[45], 1.0, "trick_complete");
        assert_eq!(f[46], 0.0, "not winner (P1!=P0)");
        assert_eq!(f[47], 0.0, "not current trick");
        assert_eq!(tokens[0].chrono_index, 0);

        // Token 2: P3, ♣2 — did NOT follow suit (Clubs != Hearts).
        let f2 = &tokens[2].features;
        assert_eq!(f2[0], 1.0, "rank 0");
        assert_eq!(f2[18], 1.0, "Clubs suit");
        assert_eq!(f2[27], 1.0, "player 3");
        assert_eq!(f2[42], 0.0, "not lead");
        assert_eq!(f2[43], 0.0, "did NOT follow suit");
        assert_eq!(f2[44], 0.0, "not trump");
        assert_eq!(f2[46], 0.0, "not winner");

        // Token 3: P0, ♥K — winner.
        let f3 = &tokens[3].features;
        assert_eq!(f3[11], 1.0, "rank 11 (King)");
        assert_eq!(f3[17], 1.0, "Hearts");
        assert_eq!(f3[24], 1.0, "player 0");
        assert_eq!(f3[43], 1.0, "followed suit");
        assert_eq!(f3[46], 1.0, "won_trick");
    }

    #[test]
    fn current_trick_tokens_are_marked_correctly() {
        // No completed tricks, 2 cards played in current trick.
        // P1 leads ♠A(idx 12), P2 plays ♦3(idx 40).
        let s = state_with_tricks(
            4, 5, 5, Suit::Spades as u8, 0,
            &[],
            &[(1, 12), (2, 40)],
        );
        let tokens = encode_played_cards(&s);
        assert_eq!(tokens.len(), 2);

        // Both should be is_current_trick=1, trick_complete=0, won_trick=0.
        for tok in &tokens {
            assert_eq!(tok.features[45], 0.0, "trick not complete");
            assert_eq!(tok.features[46], 0.0, "no won_trick for in-progress");
            assert_eq!(tok.features[47], 1.0, "is_current_trick");
        }

        // Token 0: P1, ♠A — lead, followed suit (Spades==Spades), is trump.
        assert_eq!(tokens[0].features[42], 1.0, "was_lead");
        assert_eq!(tokens[0].features[43], 1.0, "followed suit");
        assert_eq!(tokens[0].features[44], 1.0, "is_trump_play (♠ is trump)");

        // Token 1: P2, ♦3 — not lead, didn't follow (Diamonds!=Spades), not trump.
        assert_eq!(tokens[1].features[42], 0.0, "not lead");
        assert_eq!(tokens[1].features[43], 0.0, "did not follow suit");
        assert_eq!(tokens[1].features[44], 0.0, "not trump");
    }

    #[test]
    fn chrono_indices_are_sequential_across_tricks() {
        // 2 completed tricks of 3 players + 1 card in current trick = 7 tokens.
        let t0 = make_trick(&[(0, 0), (1, 13), (2, 26)], 0, 0);
        let t1 = make_trick(&[(0, 1), (1, 14), (2, 27)], 1, 0);
        let s = state_with_tricks(3, 5, 5, NO_TRUMP, 2, &[t0, t1], &[(1, 15)]);
        let tokens = encode_played_cards(&s);
        assert_eq!(tokens.len(), 7);
        for (i, tok) in tokens.iter().enumerate() {
            assert_eq!(tok.chrono_index, i as u8, "chrono_index mismatch at {i}");
        }
    }

    #[test]
    fn won_trick_set_for_exactly_one_card_per_trick() {
        let t0 = make_trick(&[(0, 0), (1, 1), (2, 2), (3, 3)], 2, 0);
        let t1 = make_trick(&[(2, 4), (3, 5), (0, 6), (1, 7)], 0, 0);
        let s = state_with_tricks(4, 5, 5, NO_TRUMP, 3, &[t0, t1], &[]);
        let tokens = encode_played_cards(&s);
        assert_eq!(tokens.len(), 8);

        // Trick 0: winner=P2, which is cards[2] = (2, 2).
        let trick0_winners: Vec<usize> = (0..4)
            .filter(|&i| tokens[i].features[46] == 1.0)
            .collect();
        assert_eq!(trick0_winners, vec![2], "trick 0 winner at slot 2 (P2)");

        // Trick 1: winner=P0, which is cards[2] = (0, 6) (P0 is 3rd to play).
        let trick1_winners: Vec<usize> = (4..8)
            .filter(|&i| tokens[i].features[46] == 1.0)
            .collect();
        assert_eq!(trick1_winners, vec![6], "trick 1 winner at slot 6 (P0)");
    }

    #[test]
    fn trump_play_in_no_trump_round() {
        // No-trump round: no card should have is_trump_play = 1.
        let t = make_trick(&[(0, 0), (1, 13), (2, 26)], 0, 0);
        let s = state_with_tricks(3, 5, 5, NO_TRUMP, 2, &[t], &[]);
        let tokens = encode_played_cards(&s);
        for tok in &tokens {
            assert_eq!(tok.features[44], 0.0, "no trump plays in no-trump round");
        }
    }

    #[test]
    fn position_in_trick_normalization() {
        // 4 players, position_in_trick should be 0/3, 1/3, 2/3, 3/3.
        let t = make_trick(&[(0, 0), (1, 13), (2, 26), (3, 39)], 0, 0);
        let s = state_with_tricks(4, 5, 5, NO_TRUMP, 3, &[t], &[]);
        let tokens = encode_played_cards(&s);
        let expected = [0.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 3.0 / 3.0];
        for (i, tok) in tokens.iter().enumerate() {
            assert!(
                (tok.features[41] - expected[i]).abs() < 1e-6,
                "position_in_trick[{i}]: {} != {}",
                tok.features[41],
                expected[i]
            );
        }
    }

    #[test]
    fn played_card_padding_slots_are_zero() {
        let t = make_trick(&[(0, 0), (1, 13), (2, 26)], 0, 0);
        let s = state_with_tricks(3, 5, 5, NO_TRUMP, 2, &[t], &[]);
        let tokens = encode_played_cards(&s);
        for (i, tok) in tokens.iter().enumerate() {
            // Rank padding [13..16).
            for j in 13..16 {
                assert_eq!(tok.features[j], 0.0, "tok {i}: rank padding[{j}]");
            }
            // Suit padding [20..24).
            for j in 20..24 {
                assert_eq!(tok.features[j], 0.0, "tok {i}: suit padding[{j}]");
            }
            // Player padding [27..40) for 3 players (only 0,1,2 used).
            for j in 27..40 {
                assert_eq!(tok.features[j], 0.0, "tok {i}: player padding[{j}]");
            }
        }
    }

    // ===============================================================
    // Session 2.2 — Player state token tests
    // ===============================================================

    #[test]
    fn player_state_token_count_matches_num_players() {
        for np in 3..=6u8 {
            let s = state_with_tricks(np, 5, 5, NO_TRUMP, 0, &[], &[]);
            let tokens = encode_player_states(&s, 0);
            assert_eq!(tokens.len(), np as usize);
        }
    }

    #[test]
    fn player_state_one_hot_correct() {
        let s = state_with_tricks(4, 5, 5, NO_TRUMP, 0, &[], &[]);
        let tokens = encode_player_states(&s, 0);
        for (p, feat) in tokens.iter().enumerate() {
            // Exactly one bit in [0..8), at position p.
            for i in 0..8 {
                let expected = if i == p { 1.0 } else { 0.0 };
                assert_eq!(feat[i], expected, "player {p}: one-hot[{i}]");
            }
            // Padding [8..16) all zero.
            for i in 8..16 {
                assert_eq!(feat[i], 0.0, "player {p}: padding[{i}]");
            }
        }
    }

    #[test]
    fn bid_status_busted() {
        // Player 0 bid 1, won 2 → busted.
        let mut s = state_with_tricks(4, 5, 5, NO_TRUMP, 0, &[], &[]);
        s.bids[0] = 1;
        s.tricks_won[0] = 2;
        let tokens = encode_player_states(&s, 0);
        assert_eq!(tokens[0][19], -1.0, "busted: tricks_won > bid");
    }

    #[test]
    fn bid_status_met() {
        // Player 0 bid 2, won 2 → met.
        let mut s = state_with_tricks(4, 5, 5, NO_TRUMP, 0, &[], &[]);
        s.bids[0] = 2;
        s.tricks_won[0] = 2;
        let tokens = encode_player_states(&s, 0);
        assert_eq!(tokens[0][19], 1.0, "met: tricks_won == bid");
    }

    #[test]
    fn bid_status_live() {
        // Player 0 bid 3, won 1, tricks_remaining=4 → live.
        let mut s = state_with_tricks(4, 5, 5, NO_TRUMP, 0, &[], &[]);
        s.bids[0] = 3;
        s.tricks_won[0] = 1;
        s.tricks_completed = 1;
        let tokens = encode_player_states(&s, 0);
        assert_eq!(tokens[0][19], 0.0, "live: needed=2, remaining=4");
    }

    #[test]
    fn bid_status_cannot_meet() {
        // Player 0 bid 4, won 1, only 2 tricks remaining → can't meet (needs 3).
        let mut s = state_with_tricks(4, 5, 5, NO_TRUMP, 0, &[], &[]);
        s.bids[0] = 4;
        s.tricks_won[0] = 1;
        s.tricks_completed = 3;
        let tokens = encode_player_states(&s, 0);
        assert_eq!(tokens[0][19], -1.0, "busted: needed=3 > remaining=2");
    }

    #[test]
    fn is_dealer_flag() {
        let s = state_with_tricks(4, 5, 5, NO_TRUMP, 2, &[], &[]);
        let tokens = encode_player_states(&s, 0);
        for (p, feat) in tokens.iter().enumerate() {
            let expected = if p == 2 { 1.0 } else { 0.0 };
            assert_eq!(feat[20], expected, "player {p}: is_dealer");
        }
    }

    #[test]
    fn is_me_flag() {
        let s = state_with_tricks(4, 5, 5, NO_TRUMP, 0, &[], &[]);
        let tokens = encode_player_states(&s, 2);
        for (p, feat) in tokens.iter().enumerate() {
            let expected = if p == 2 { 1.0 } else { 0.0 };
            assert_eq!(feat[21], expected, "player {p}: is_me (perspective=2)");
        }
    }

    #[test]
    fn relative_position_values() {
        // 4 players, current_player = 1.
        let mut s = state_with_tricks(4, 5, 5, NO_TRUMP, 0, &[], &[]);
        s.current_player = 1;
        let tokens = encode_player_states(&s, 0);
        // rel = (p + 4 - 1) % 4 / 4
        // P0: (0+4-1)%4=3, 3/4=0.75
        // P1: (1+4-1)%4=0, 0/4=0.0
        // P2: (2+4-1)%4=1, 1/4=0.25
        // P3: (3+4-1)%4=2, 2/4=0.5
        assert!((tokens[0][22] - 0.75).abs() < 1e-6, "P0 rel_pos");
        assert!((tokens[1][22] - 0.0).abs() < 1e-6, "P1 rel_pos (current)");
        assert!((tokens[2][22] - 0.25).abs() < 1e-6, "P2 rel_pos");
        assert!((tokens[3][22] - 0.5).abs() < 1e-6, "P3 rel_pos");
    }

    #[test]
    fn cumulative_score_normalization() {
        // 4 players, start_cards=5. total_rounds = 2*5+4-2 = 12.
        // Ceiling = 12 * (10 + 5) = 180.
        let mut s = state_with_tricks(4, 5, 5, NO_TRUMP, 0, &[], &[]);
        s.cumulative_scores[0] = 90; // 90/180 = 0.5
        s.cumulative_scores[1] = 0;
        s.cumulative_scores[2] = 180; // 180/180 = 1.0
        let tokens = encode_player_states(&s, 0);
        assert!((tokens[0][23] - 0.5).abs() < 1e-6, "P0 score");
        assert!((tokens[1][23] - 0.0).abs() < 1e-6, "P1 score");
        assert!((tokens[2][23] - 1.0).abs() < 1e-6, "P2 score");
    }

    #[test]
    fn void_detection_from_completed_trick() {
        // P1 leads ♥5(idx 16), P2 plays ♣2(idx 26) — P2 void in Hearts.
        // P3 plays ♥K(idx 24) — P3 followed suit, NOT void.
        let trick = make_trick(
            &[(1, 16), (2, 26), (3, 24)],
            3,                   // winner
            Suit::Hearts as u8,  // suit_led
        );
        let s = state_with_tricks(4, 5, 5, NO_TRUMP, 0, &[trick], &[]);
        let tokens = encode_player_states(&s, 0);

        // P2: void_hearts = 1.0 (didn't follow Hearts).
        assert_eq!(tokens[2][26], 1.0, "P2 void in Hearts");
        // P2: other voids should be 0.
        assert_eq!(tokens[2][25], 0.0, "P2 not void Spades");
        assert_eq!(tokens[2][27], 0.0, "P2 not void Clubs");
        assert_eq!(tokens[2][28], 0.0, "P2 not void Diamonds");

        // P3: followed suit, no void.
        assert_eq!(tokens[3][26], 0.0, "P3 NOT void in Hearts");

        // P1: leader, not checked for void.
        assert_eq!(tokens[1][26], 0.0, "P1 (leader) not flagged void");
    }

    #[test]
    fn void_detection_from_current_trick() {
        // Current trick: P0 leads ♠3(idx 3), P1 plays ♦7(idx 44) — P1 void in Spades.
        let s = state_with_tricks(
            4, 5, 5, Suit::Spades as u8, 3,
            &[],
            &[(0, 3), (1, 44)],
        );
        let tokens = encode_player_states(&s, 0);
        assert_eq!(tokens[1][25], 1.0, "P1 void in Spades from current trick");
        // P0 is leader, no void.
        assert_eq!(tokens[0][25], 0.0, "P0 (leader) not void");
    }

    #[test]
    fn cards_in_hand_normalization() {
        let mut s = state_with_tricks(4, 5, 5, NO_TRUMP, 0, &[], &[]);
        s.hands[0] = c(Suit::Spades, 0).bit()
            | c(Suit::Spades, 1).bit()
            | c(Suit::Hearts, 5).bit();
        s.hands[1] = c(Suit::Clubs, 3).bit()
            | c(Suit::Clubs, 4).bit()
            | c(Suit::Clubs, 5).bit()
            | c(Suit::Clubs, 6).bit()
            | c(Suit::Clubs, 7).bit();
        let tokens = encode_player_states(&s, 0);
        assert!((tokens[0][24] - 3.0 / 5.0).abs() < 1e-6, "P0: 3/5");
        assert!((tokens[1][24] - 5.0 / 5.0).abs() < 1e-6, "P1: 5/5");
        assert!((tokens[2][24] - 0.0).abs() < 1e-6, "P2: 0/5");
    }

    #[test]
    fn bid_and_tricks_normalization() {
        let mut s = state_with_tricks(4, 5, 5, NO_TRUMP, 0, &[], &[]);
        s.bids[0] = 3;
        s.tricks_won[0] = 1;
        let tokens = encode_player_states(&s, 0);
        assert!((tokens[0][16] - 3.0 / 5.0).abs() < 1e-6, "bid normalized");
        assert!((tokens[0][17] - 1.0 / 5.0).abs() < 1e-6, "tricks_won normalized");
        assert!((tokens[0][18] - 2.0 / 5.0).abs() < 1e-6, "tricks_needed normalized");
    }

    // ---------------------------------------------------------------
    // Integration: real game + played cards + player states
    // ---------------------------------------------------------------

    #[test]
    fn played_cards_integration_with_real_game() {
        use blob_engine::bidding::{apply_bid, legal_bids};
        use blob_engine::dealing::start_round;
        use blob_engine::game::new_game;
        use blob_engine::playing::{apply_play, legal_plays};
        use rand_xoshiro::rand_core::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;

        let mut s = new_game(4, 5).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(99);
        start_round(&mut s, &mut rng);

        // Bidding.
        while s.phase() == GamePhase::Bidding {
            let mask = legal_bids(&s);
            let bid = (0..=13u8).find(|b| (mask >> b) & 1 == 1).unwrap();
            apply_bid(&mut s, bid);
        }

        // Play 3 tricks.
        for _ in 0..3 {
            for _ in 0..s.num_players {
                let mask = legal_plays(&s);
                let card = mask.trailing_zeros() as u8;
                apply_play(&mut s, card);
            }
        }

        // Play 1 card of the 4th trick.
        {
            let mask = legal_plays(&s);
            let card = mask.trailing_zeros() as u8;
            apply_play(&mut s, card);
        }

        let tokens = encode_played_cards(&s);
        // 3 completed tricks × 4 players + 1 current trick card = 13 tokens.
        assert_eq!(tokens.len(), 13);

        // First 12 tokens: trick_complete=1, is_current_trick=0.
        for tok in &tokens[..12] {
            assert_eq!(tok.features[45], 1.0, "completed trick");
            assert_eq!(tok.features[47], 0.0, "not current trick");
        }
        // Last token: trick_complete=0, is_current_trick=1, was_lead=1.
        assert_eq!(tokens[12].features[45], 0.0, "current trick not complete");
        assert_eq!(tokens[12].features[47], 1.0, "is current trick");
        assert_eq!(tokens[12].features[42], 1.0, "current trick leader");

        // Chrono indices 0..12.
        for (i, tok) in tokens.iter().enumerate() {
            assert_eq!(tok.chrono_index, i as u8);
        }

        // Each completed trick has exactly one won_trick flag.
        for t in 0..3 {
            let trick_tokens = &tokens[t * 4..(t + 1) * 4];
            let winner_count: usize = trick_tokens
                .iter()
                .filter(|tok| tok.features[46] == 1.0)
                .count();
            assert_eq!(winner_count, 1, "trick {t} has exactly one winner");
        }
    }

    #[test]
    fn player_states_integration_with_real_game() {
        use blob_engine::bidding::{apply_bid, legal_bids};
        use blob_engine::dealing::start_round;
        use blob_engine::game::new_game;
        use blob_engine::playing::{apply_play, legal_plays};
        use rand_xoshiro::rand_core::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;

        let mut s = new_game(5, 7).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
        start_round(&mut s, &mut rng);

        // Bidding.
        while s.phase() == GamePhase::Bidding {
            let mask = legal_bids(&s);
            let bid = (0..=13u8).find(|b| (mask >> b) & 1 == 1).unwrap();
            apply_bid(&mut s, bid);
        }

        // Play 4 tricks.
        for _ in 0..4 {
            for _ in 0..s.num_players {
                let mask = legal_plays(&s);
                let card = mask.trailing_zeros() as u8;
                apply_play(&mut s, card);
            }
        }

        let perspective = s.current_player;
        let tokens = encode_player_states(&s, perspective);
        assert_eq!(tokens.len(), 5);

        // Exactly one is_me flag.
        let me_count: usize = tokens.iter().filter(|f| f[21] == 1.0).count();
        assert_eq!(me_count, 1);
        assert_eq!(tokens[perspective as usize][21], 1.0);

        // Exactly one is_dealer flag.
        let dealer_count: usize = tokens.iter().filter(|f| f[20] == 1.0).count();
        assert_eq!(dealer_count, 1);
        assert_eq!(tokens[s.dealer as usize][20], 1.0);

        // All features within expected ranges.
        for (p, feat) in tokens.iter().enumerate() {
            assert!(feat[16] >= 0.0 && feat[16] <= 1.0, "P{p} bid in [0,1]");
            assert!(feat[17] >= 0.0 && feat[17] <= 1.0, "P{p} tricks_won in [0,1]");
            assert!(feat[18] >= 0.0 && feat[18] <= 1.0, "P{p} tricks_needed in [0,1]");
            assert!(
                feat[19] == -1.0 || feat[19] == 0.0 || feat[19] == 1.0,
                "P{p} bid_status in {{-1,0,1}}"
            );
            assert!(feat[22] >= 0.0 && feat[22] < 1.0, "P{p} rel_pos in [0,1)");
            assert!(feat[23] >= 0.0 && feat[23] <= 1.0, "P{p} cum_score in [0,1]");
            assert!(feat[24] >= 0.0 && feat[24] <= 1.0, "P{p} cards_in_hand in [0,1]");
            for v in 25..29 {
                assert!(
                    feat[v] == 0.0 || feat[v] == 1.0,
                    "P{p} void[{v}] binary"
                );
            }
        }

        // Current player's relative_position is 0.
        assert_eq!(tokens[s.current_player as usize][22], 0.0);
    }
}
