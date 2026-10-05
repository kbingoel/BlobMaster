//! Frozen gen-1 encoder (feature layout v1).
//!
//! Gen-1 ONNX checkpoints (`checkpoints/run-2026-05-14`) were trained on this
//! layout: features right-padded to 48 (`FEAT_DIM`). It is kept only so they
//! still run as the reference and as `bench` / `play` opponents;
//! [`crate::onnx::OnnxEvaluator`] selects it from the model's input width.
//! **Never change it**: `golden_hash_is_frozen` pins its output bit-for-bit.
//! New models use the parent module's encoder (gen-2.md §5.5).
//!
//! Its known defects are listed in gen-2.md §2.8: absolute seats, no
//! `has_bid`, unscaled counts, cumulative scores as input.

use crate::bidding::forbidden_bid;
use crate::card::{Card, NUM_SUITS};
use crate::hand::Hand;
use crate::round::total_rounds;
use crate::state::{BlobState, GamePhase, MAX_PLAYERS};
use smallvec::SmallVec;

use super::{
    EncodedState, TOKEN_TYPE_CLS, TOKEN_TYPE_CONTEXT, TOKEN_TYPE_HAND, TOKEN_TYPE_PLAYED,
    TOKEN_TYPE_PLAYER,
};

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

/// Per-token feature width a v1 model expects (the played-card width).
pub const FEAT_DIM: usize = PLAYED_CARD_DIM;

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

/// Dimensionality of the context token.
pub const CONTEXT_DIM: usize = 13;

/// Encode the 13-dim context token for the current game state.
///
/// Layout:
/// - `[0..5)`: trump_suit one-hot (♠=0, ♥=1, ♣=2, ♦=3, NoTrump=4)
/// - `[5]`: cards_dealt (normalized by 13, the hard cap)
/// - `[6]`: current_trick (tricks_completed / cards_dealt)
/// - `[7]`: tricks_remaining ((cards_dealt − tricks_completed) / cards_dealt)
/// - `[8]`: num_players (normalized by 8, the max)
/// - `[9]`: round_number (round_idx / total_rounds(start_cards, num_players))
/// - `[10..12)`: game_phase one-hot: \[is_bidding, is_playing\]
/// - `[12]`: bidding_constraint_active (1.0 iff bidding, current player is
///   dealer, and the forbidden-bid constraint applies)
pub fn encode_context(state: &BlobState) -> [f32; CONTEXT_DIM] {
    let mut feat = [0.0f32; CONTEXT_DIM];

    // Trump suit one-hot [0..5): value 0–3 for suits, 4 for NoTrump.
    feat[state.trump_suit.min(4) as usize] = 1.0;

    // cards_dealt [5]: normalized by MAX_CARDS_DEALT (13).
    feat[5] = state.cards_dealt as f32 / 13.0;

    // current_trick [6]: tricks_completed / cards_dealt.
    let cd = state.cards_dealt.max(1) as f32;
    feat[6] = state.tricks_completed as f32 / cd;

    // tricks_remaining [7]: (cards_dealt − tricks_completed) / cards_dealt.
    let remaining = state.cards_dealt.saturating_sub(state.tricks_completed);
    feat[7] = remaining as f32 / cd;

    // num_players [8]: normalized by MAX_PLAYERS (8).
    feat[8] = state.num_players as f32 / MAX_PLAYERS as f32;

    // round_number [9]: round_idx / total_rounds.
    let total_r = total_rounds(state.start_cards.max(1), state.num_players.max(3));
    feat[9] = state.round_idx as f32 / total_r as f32;

    // game_phase one-hot [10..12): [is_bidding, is_playing].
    match state.phase() {
        GamePhase::Bidding => feat[10] = 1.0,
        GamePhase::Playing => feat[11] = 1.0,
        _ => {} // debug_assert in encode() prevents this path
    }

    // bidding_constraint_active [12].
    if state.phase() == GamePhase::Bidding
        && state.current_player == state.dealer
        && forbidden_bid(state).is_some()
    {
        feat[12] = 1.0;
    }

    feat
}

/// Full encoder entry point. Assembles the variable-length token sequence
/// from the perspective of the given player.
///
/// Sequence: `[CLS, context, player_states…, hand_cards…, played_cards…]`.
///
/// The `perspective` argument is the player whose viewpoint the encoding
/// represents. MCTS always passes `state.current_player`; other call sites
/// (e.g. eval tooling) may pass a fixed seat.
///
/// Panics in debug if called from `Scoring` or `Complete` phase.
pub fn encode(state: &BlobState, perspective: u8) -> EncodedState {
    crate::profiling::time(&crate::profiling::ENCODE, || {
        debug_assert!(
            matches!(state.phase(), GamePhase::Bidding | GamePhase::Playing),
            "encoder called in {:?} phase — only Bidding/Playing are valid",
            state.phase()
        );

        let hand_cards = encode_hand_cards(state, perspective);
        let played_cards = encode_played_cards(state);
        let player_states = encode_player_states(state, perspective);
        let context = encode_context(state);

        let np = state.num_players as usize;
        let num_hand = hand_cards.len();
        let num_played = played_cards.len();
        let num_tokens = 1 + 1 + np + num_hand + num_played;

        let mut features = Vec::with_capacity(num_tokens);
        let mut token_types = Vec::with_capacity(num_tokens);
        let mut chrono_indices = Vec::with_capacity(num_tokens);
        let mut hand_card_indices = SmallVec::with_capacity(num_hand);

        // CLS token: zero-length feature vector (NN uses a learned 128-dim parameter).
        features.push(Vec::new());
        token_types.push(TOKEN_TYPE_CLS);
        chrono_indices.push(0);

        // Context token.
        features.push(context.to_vec());
        token_types.push(TOKEN_TYPE_CONTEXT);
        chrono_indices.push(0);

        // Player state tokens.
        for ps in &player_states {
            features.push(ps.to_vec());
            token_types.push(TOKEN_TYPE_PLAYER);
            chrono_indices.push(0);
        }

        // Hand card tokens — record card indices for MCTS action mapping.
        let hand = Hand::new(state.hands[perspective as usize]);
        for (i, card) in hand.iter().enumerate() {
            features.push(hand_cards[i].to_vec());
            token_types.push(TOKEN_TYPE_HAND);
            chrono_indices.push(0);
            hand_card_indices.push(card.index());
        }

        // Played card tokens.
        for pc in &played_cards {
            features.push(pc.features.to_vec());
            token_types.push(TOKEN_TYPE_PLAYED);
            chrono_indices.push(pc.chrono_index);
        }

        EncodedState {
            features,
            token_types,
            chronological_indices: chrono_indices,
            hand_card_indices,
            num_tokens,
        }
    })
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::random_game_states;

    fn fnv(hash: &mut u64, bytes: &[u8]) {
        for &b in bytes {
            *hash ^= b as u64;
            *hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }

    /// FNV-1a over every output of `encode` for the states of three random
    /// games, from the acting seat and from the next seat.
    fn golden_hash() -> (u64, usize) {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut n = 0;
        for s in random_game_states() {
            for perspective in [s.current_player, (s.current_player + 1) % s.num_players] {
                let enc = encode(&s, perspective);
                for row in &enc.features {
                    for v in row {
                        fnv(&mut h, &v.to_bits().to_le_bytes());
                    }
                    fnv(&mut h, &[0xff]);
                }
                fnv(&mut h, &enc.token_types);
                fnv(&mut h, &enc.chronological_indices);
                fnv(&mut h, &enc.hand_card_indices);
                n += 1;
            }
        }
        (h, n)
    }

    /// Pins v1 bit-for-bit to the encoder gen 1 was trained with (the
    /// pre-gen-2 `encoder.rs`, commit 210b82f). If this fails, v1 changed
    /// and gen-1 checkpoints no longer see the inputs they were trained on.
    #[test]
    fn golden_hash_is_frozen() {
        assert_eq!(golden_hash(), (GOLDEN_HASH, GOLDEN_COUNT));
    }

    #[test]
    fn feature_widths_match_gen1_models() {
        let widths = (HAND_CARD_DIM, PLAYED_CARD_DIM, PLAYER_STATE_DIM, CONTEXT_DIM);
        assert_eq!(widths, (30, 48, 29, 13));
        assert_eq!(FEAT_DIM, 48);
    }

    const GOLDEN_HASH: u64 = 18_037_513_820_615_112_858;
    const GOLDEN_COUNT: usize = 2264;
}
