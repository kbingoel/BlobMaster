//! Entity encoder: raw per-token feature vectors for the network.
//!
//! `encode(state, perspective)` → [`EncodedState`]: the variable-length
//! sequence `[CLS, context, players…, hand cards…, played cards…]` with
//! token type IDs, chronological indices for played cards, and the
//! hand-card index mapping.
//!
//! Feature layout v2 (gen-2.md §5.5). Everything is seen from
//! `perspective`:
//! - **Seats are relative.** "Me" is seat 0, then the seats after me in
//!   play order. Player tokens are emitted in that order.
//! - **`has_bid`** on player tokens, so "not yet bid" differs from "bid 0".
//! - **Bid context:** bids so far, seats still to bid, over/under-bid, and
//!   my place in bidding order.
//! - **Trick features:** "winning so far" on current-trick cards; "legal"
//!   and "beats the current winner" on hand cards.
//! - Counts are scaled to [0, 1]; "highest/lowest in suit" ignore my own
//!   cards; cumulative scores are not an input.
//!
//! Gen-1 checkpoints were trained on the frozen layout in [`v1`];
//! [`EncoderVersion`] selects between them.

use crate::belief::void_suits;
use crate::bidding::{bid_order_position, forbidden_bid, has_bid};
use crate::card::{Card, NUM_RANKS, NUM_SUITS};
use crate::hand::Hand;
use crate::playing::{beats, current_trick_winner};
use crate::round::total_rounds;
use crate::state::{BlobState, GamePhase, MAX_PLAYERS};
use smallvec::SmallVec;

pub mod v1;

/// Dimensionality of a hand-card token.
pub const HAND_CARD_DIM: usize = 32;

/// Dimensionality of a played-card token.
pub const PLAYED_CARD_DIM: usize = 49;

/// Dimensionality of a player-state token.
pub const PLAYER_STATE_DIM: usize = 28;

/// Dimensionality of the context token.
pub const CONTEXT_DIM: usize = 17;

const fn max(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}

/// Per-token feature width of the network input: every token is
/// right-padded to the widest token type.
pub const FEAT_DIM: usize = max(
    max(HAND_CARD_DIM, PLAYED_CARD_DIM),
    max(PLAYER_STATE_DIM, CONTEXT_DIM),
);

/// Token type IDs (0–4) for each position in the assembled sequence.
pub const TOKEN_TYPE_CLS: u8 = 0;
pub const TOKEN_TYPE_CONTEXT: u8 = 1;
pub const TOKEN_TYPE_PLAYER: u8 = 2;
pub const TOKEN_TYPE_HAND: u8 = 3;
pub const TOKEN_TYPE_PLAYED: u8 = 4;

/// Feature layout a model was trained on. The token sequence is the same in
/// both; only the per-token features differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderVersion {
    /// Gen 1 ([`v1`], 48 features per token).
    V1,
    /// Gen 2 (this module, [`FEAT_DIM`] features per token).
    V2,
}

impl EncoderVersion {
    /// The layout new models are trained on.
    pub const CURRENT: Self = Self::V2;

    /// The version whose padded token width is `feat_dim`, as read from a
    /// model's `features` input. `None` for an unknown width.
    pub fn from_feat_dim(feat_dim: usize) -> Option<Self> {
        match feat_dim {
            v1::FEAT_DIM => Some(Self::V1),
            FEAT_DIM => Some(Self::V2),
            _ => None,
        }
    }

    /// Padded per-token feature width.
    pub fn feat_dim(self) -> usize {
        match self {
            Self::V1 => v1::FEAT_DIM,
            Self::V2 => FEAT_DIM,
        }
    }

    /// Encode `state` from `perspective` in this layout.
    pub fn encode(self, state: &BlobState, perspective: u8) -> EncodedState {
        match self {
            Self::V1 => v1::encode(state, perspective),
            Self::V2 => encode(state, perspective),
        }
    }
}

/// Seat of `player` relative to `perspective`: 0 for `perspective`, then 1,
/// 2, … for the seats after it in play order.
#[inline]
pub fn relative_seat(state: &BlobState, perspective: u8, player: u8) -> u8 {
    (player + state.num_players - perspective) % state.num_players
}

/// Card indices of `perspective`'s hand in `Hand::iter()` order: the
/// play-policy action order, identical to `EncodedState::hand_card_indices`
/// without encoding the rest of the state.
#[inline]
pub fn hand_card_indices(state: &BlobState, perspective: u8) -> SmallVec<[u8; 13]> {
    Hand::new(state.hands[perspective as usize]).iter().map(|c| c.index()).collect()
}

#[inline]
fn flag(b: bool) -> f32 {
    if b {
        1.0
    } else {
        0.0
    }
}

/// Encode the perspective player's hand cards into feature vectors.
///
/// Each card in the hand produces a 32-dimensional feature vector:
/// - `[0..16)`: rank one-hot (13 values + 3 padding)
/// - `[16..24)`: suit one-hot (4 values + 4 padding)
/// - `[24]`: is_trump
/// - `[25]`: suit_count_in_hand / 13
/// - `[26]`: is_highest_in_suit: no unseen card of the suit outranks it
/// - `[27]`: is_lowest_in_suit: no unseen card of the suit ranks below it
/// - `[28]`: cards_above_unseen / 13 (same suit, higher, not mine, not played)
/// - `[29]`: cards_below_unseen / 13 (same suit, lower, not mine, not played)
/// - `[30]`: is_legal: playable into the current trick (follow suit if able);
///   0 during bidding
/// - `[31]`: beats_current_winner: would take the trick in progress from the
///   card winning it so far; 0 when no card has been played
///
/// "Unseen" cards are neither in my hand nor played this round, so they may
/// be in an opponent's hand (or undealt). My own cards don't count against
/// `is_highest` / `is_lowest`: holding A♠ doesn't stop my K♠ from being the
/// best spade an opponent can face.
///
/// Tokens are emitted in `Hand::iter()` order (ascending card index).
/// This is the canonical action order used by the playing head and MCTS.
pub fn encode_hand_cards(state: &BlobState, perspective: u8) -> Vec<[f32; HAND_CARD_DIM]> {
    let hand = Hand::new(state.hands[perspective as usize]);
    let trump = state.trump_suit;
    let deck_mask: u64 = (1u64 << 52) - 1;
    let unseen = !hand.bits() & !state.played_this_round & deck_mask;
    let counts = NUM_RANKS as f32;

    // Follow-suit rule and current winner for the trick in progress.
    let playing = state.phase() == GamePhase::Playing;
    let trick = current_trick_winner(state).filter(|_| playing).map(|slot| {
        let led = state.trick_play_order[0] / NUM_RANKS;
        (led, state.trick_play_order[slot as usize])
    });
    let legal = match trick {
        _ if !playing => 0,
        None => hand.bits(),
        Some((led, _)) => {
            let of_led = hand.bits() & (0x1FFFu64 << (led * NUM_RANKS));
            if of_led != 0 {
                of_led
            } else {
                hand.bits()
            }
        }
    };

    let mut tokens = Vec::with_capacity(hand.count() as usize);
    for card in hand.iter() {
        let mut feat = [0.0f32; HAND_CARD_DIM];
        let suit = card.suit();
        let idx = card.index();

        feat[card.rank() as usize] = 1.0;
        feat[16 + suit.index() as usize] = 1.0;
        feat[24] = flag(trump < NUM_SUITS && suit.index() == trump);
        feat[25] = hand.cards_of_suit(suit).count_ones() as f32 / counts;

        let suit_mask = suit.mask();
        let above_in_suit = suit_mask & !((1u64 << (idx + 1)) - 1);
        let below_in_suit = suit_mask & ((1u64 << idx) - 1);
        let above = (unseen & above_in_suit).count_ones();
        let below = (unseen & below_in_suit).count_ones();
        feat[26] = flag(above == 0);
        feat[27] = flag(below == 0);
        feat[28] = above as f32 / counts;
        feat[29] = below as f32 / counts;

        feat[30] = flag((legal >> idx) & 1 == 1);
        feat[31] = flag(trick.is_some_and(|(led, best)| beats(idx, best, led, trump)));

        tokens.push(feat);
    }
    tokens
}

/// A played-card token with its chronological position index.
///
/// The `chrono_index` (0–51) is used by the neural network to look up a
/// learned chronological embedding (Session 3.1, 52×128 table).
#[derive(Debug, Clone)]
pub struct PlayedCardToken {
    pub features: [f32; PLAYED_CARD_DIM],
    pub chrono_index: u8,
}

/// Encode all played cards into 49-dim feature vectors in chronological order.
///
/// Each played card token:
/// - `[0..16)`: rank one-hot (13 values + 3 padding)
/// - `[16..24)`: suit one-hot (4 values + 4 padding)
/// - `[24..40)`: relative seat of the player one-hot (up to 8 + 8 padding)
/// - `[40]`: trick_number (normalized by cards_dealt)
/// - `[41]`: position_in_trick (normalized to \[0, 1\])
/// - `[42]`: was_lead
/// - `[43]`: followed_suit (card suit == led suit)
/// - `[44]`: is_trump_play
/// - `[45]`: trick_complete
/// - `[46]`: won_trick (only for the winning card of a completed trick)
/// - `[47]`: is_current_trick
/// - `[48]`: winning_so_far (only for the card currently winning the trick
///   in progress)
///
/// Iterates `trick_history[0..tricks_completed]` then current trick's
/// `trick_play_order[0..trick_cards_played]` in strict chronological order.
pub fn encode_played_cards(state: &BlobState, perspective: u8) -> Vec<PlayedCardToken> {
    let np = state.num_players as usize;
    let trump = state.trump_suit;
    let cd = state.cards_dealt.max(1) as f32;
    let pos_norm = (state.num_players.saturating_sub(1)).max(1) as f32;

    let total_played = state.tricks_completed as usize * np + state.trick_cards_played as usize;
    let mut tokens = Vec::with_capacity(total_played);
    let mut chrono: u8 = 0;

    // Features shared by both kinds of trick; `slot` is the card's
    // position in its trick.
    let base = |player: u8, card_idx: u8, trick: usize, slot: usize, led: u8| {
        let card = Card::from_index_unchecked(card_idx);
        let mut feat = [0.0f32; PLAYED_CARD_DIM];
        feat[card.rank() as usize] = 1.0;
        feat[16 + card.suit().index() as usize] = 1.0;
        feat[24 + relative_seat(state, perspective, player) as usize] = 1.0;
        feat[40] = trick as f32 / cd;
        feat[41] = slot as f32 / pos_norm;
        feat[42] = flag(slot == 0);
        feat[43] = flag(card.suit().index() == led);
        feat[44] = flag(trump < NUM_SUITS && card.suit().index() == trump);
        feat
    };

    // Completed tricks.
    for t in 0..state.tricks_completed as usize {
        let rec = &state.trick_history[t];
        for i in 0..rec.num_played as usize {
            let (player, card_idx) = rec.cards[i];
            let mut feat = base(player, card_idx, t, i, rec.suit_led);
            feat[45] = 1.0; // trick_complete
            feat[46] = flag(player == rec.winner);
            tokens.push(PlayedCardToken {
                features: feat,
                chrono_index: chrono,
            });
            chrono += 1;
        }
    }

    // Current (in-progress) trick.
    if let Some(winning_slot) = current_trick_winner(state) {
        let led = state.trick_play_order[0] / NUM_RANKS;
        let trick = state.tricks_completed as usize;
        for i in 0..state.trick_cards_played as usize {
            let player = (state.trick_leader + i as u8) % state.num_players;
            let mut feat = base(player, state.trick_play_order[i], trick, i, led);
            feat[47] = 1.0; // is_current_trick
            feat[48] = flag(i == winning_slot as usize);
            tokens.push(PlayedCardToken {
                features: feat,
                chrono_index: chrono,
            });
            chrono += 1;
        }
    }

    tokens
}

/// Encode per-player state tokens (28 dims each, one per player), in
/// relative-seat order: `perspective` first, then the seats after it.
///
/// Each player state token:
/// - `[0..16)`: relative seat one-hot (up to 8 values + 8 padding)
/// - `[16]`: bid (normalized by cards_dealt; 0 until has_bid)
/// - `[17]`: tricks_won (normalized by cards_dealt)
/// - `[18]`: tricks_needed (max(0, bid − tricks_won), normalized by
///   cards_dealt; 0 until has_bid)
/// - `[19]`: bid_status (−1.0 busted, 0.0 live, +1.0 met; 0 until has_bid)
/// - `[20]`: is_dealer
/// - `[21]`: has_bid
/// - `[22]`: is_to_move (the seat that acts now)
/// - `[23]`: cards_in_hand (normalized by cards_dealt)
/// - `[24..28)`: void in ♠ ♥ ♣ ♦, from [`void_suits`] (completed tricks and
///   the trick in progress)
pub fn encode_player_states(
    state: &BlobState,
    perspective: u8,
) -> Vec<[f32; PLAYER_STATE_DIM]> {
    let np = state.num_players as usize;
    let cd = state.cards_dealt.max(1) as f32;
    let tricks_remaining = state.cards_dealt.saturating_sub(state.tricks_completed);
    let voids = void_suits(state);

    let mut tokens = Vec::with_capacity(np);
    for rel in 0..np {
        let player = ((perspective as usize + rel) % np) as u8;
        let p = player as usize;
        let mut feat = [0.0f32; PLAYER_STATE_DIM];

        feat[rel] = 1.0;
        let bid_made = has_bid(state, player);
        if bid_made {
            let needed = state.bids[p].saturating_sub(state.tricks_won[p]);
            feat[16] = state.bids[p] as f32 / cd;
            feat[18] = needed as f32 / cd;
            feat[19] = if state.tricks_won[p] > state.bids[p] {
                -1.0
            } else if state.tricks_won[p] == state.bids[p] {
                1.0
            } else if tricks_remaining >= needed {
                0.0
            } else {
                -1.0
            };
        }
        feat[17] = state.tricks_won[p] as f32 / cd;
        feat[20] = flag(player == state.dealer);
        feat[21] = flag(bid_made);
        feat[22] = flag(player == state.current_player);
        feat[23] = Hand::new(state.hands[p]).count() as f32 / cd;
        for s in 0..NUM_SUITS as usize {
            feat[24 + s] = flag(voids[p][s]);
        }

        tokens.push(feat);
    }
    tokens
}

/// Assembled variable-length sequence output from the encoder.
///
/// Sequence order: `[CLS, context, player_states…, hand_cards…, played_cards…]`.
///
/// `hand_card_indices` maps hand-card token positions back to card indices
/// so MCTS can translate playing-head scores to actions without re-iterating
/// the hand. Entry `i` is the card index emitted at hand-card token slot `i`,
/// matching `Hand::iter()` order exactly.
#[derive(Debug, Clone)]
pub struct EncodedState {
    pub features: Vec<Vec<f32>>,
    pub token_types: Vec<u8>,
    pub chronological_indices: Vec<u8>,
    pub hand_card_indices: SmallVec<[u8; 13]>,
    pub num_tokens: usize,
}

/// Encode the 17-dim context token from `perspective`.
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
/// - `[13]`: bid_sum / 13: total of the bids made so far
/// - `[14]`: seats_to_bid / num_players: seats that haven't bid yet
/// - `[15]`: (bid_sum − cards_dealt) / cards_dealt: over- (> 0) or under-bid
/// - `[16]`: my bidding position / (num_players − 1): 0 bids first, 1 is
///   the dealer
pub fn encode_context(state: &BlobState, perspective: u8) -> [f32; CONTEXT_DIM] {
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

    // Bid context [13..17).
    let np = state.num_players;
    let (mut bid_sum, mut to_bid) = (0u32, 0u32);
    for p in 0..np {
        if has_bid(state, p) {
            bid_sum += state.bids[p as usize] as u32;
        } else {
            to_bid += 1;
        }
    }
    feat[13] = bid_sum as f32 / 13.0;
    feat[14] = to_bid as f32 / np.max(1) as f32;
    feat[15] = (bid_sum as f32 - state.cards_dealt as f32) / cd;
    feat[16] = bid_order_position(state, perspective) as f32 / np.saturating_sub(1).max(1) as f32;

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
        let played_cards = encode_played_cards(state, perspective);
        let player_states = encode_player_states(state, perspective);
        let context = encode_context(state, perspective);

        let np = state.num_players as usize;
        let num_hand = hand_cards.len();
        let num_played = played_cards.len();
        let num_tokens = 1 + 1 + np + num_hand + num_played;

        let mut features = Vec::with_capacity(num_tokens);
        let mut token_types = Vec::with_capacity(num_tokens);
        let mut chrono_indices = Vec::with_capacity(num_tokens);

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

        // Hand card tokens.
        for hc in &hand_cards {
            features.push(hc.to_vec());
            token_types.push(TOKEN_TYPE_HAND);
            chrono_indices.push(0);
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
            hand_card_indices: hand_card_indices(state, perspective),
            num_tokens,
        }
    })
}

/// Every decision state of three random-move games (4p5c, 5p7c, 6p8c),
/// shared by the encoder test suites.
#[cfg(test)]
pub(crate) fn random_game_states() -> Vec<BlobState> {
    use crate::bidding::{apply_bid, legal_bids};
    use crate::dealing::start_round;
    use crate::game::{advance_round, new_game};
    use crate::playing::{apply_play, legal_plays};
    use rand::Rng;
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn pick<R: Rng>(mask: u64, rng: &mut R) -> u8 {
        let k = rng.gen_range(0..mask.count_ones());
        let mut m = mask;
        for _ in 0..k {
            m &= m - 1;
        }
        m.trailing_zeros() as u8
    }

    let mut out = Vec::new();
    for (i, &(np, sc)) in [(4u8, 5u8), (5, 7), (6, 8)].iter().enumerate() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xE1C0_DE00 + i as u64);
        let mut s = new_game(np, sc).unwrap();
        start_round(&mut s, &mut rng);
        loop {
            match s.phase() {
                GamePhase::Bidding => {
                    out.push(s);
                    let b = pick(legal_bids(&s) as u64, &mut rng);
                    apply_bid(&mut s, b);
                }
                GamePhase::Playing => {
                    out.push(s);
                    let c = pick(legal_plays(&s), &mut rng);
                    apply_play(&mut s, c);
                }
                GamePhase::Scoring => advance_round(&mut s, &mut rng),
                GamePhase::Complete => break,
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{Card, Suit};
    use crate::round::NO_TRUMP;
    use crate::state::{GamePhase, TrickRecord};

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

        // All three spades tokens should report suit_count = 3 (scaled by 13).
        assert_eq!(tokens[0][25], 3.0 / 13.0); // ♠0
        assert_eq!(tokens[1][25], 3.0 / 13.0); // ♠5
        assert_eq!(tokens[2][25], 3.0 / 13.0); // ♠12
        // Hearts token should report suit_count = 1.
        assert_eq!(tokens[3][25], 1.0 / 13.0); // ♥3
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
        assert_eq!(feat[27], 0.0, "Ace should not be lowest (12 unseen below)");
    }

    #[test]
    fn two_is_lowest_when_no_cards_played() {
        let hand = make_hand(&[c(Suit::Spades, 0)]);
        let s = test_state(hand, NO_TRUMP, 0, 4);
        let feat = &encode_hand_cards(&s, 0)[0];
        assert_eq!(feat[26], 0.0, "2 should not be highest (12 unseen above)");
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
        assert_eq!(feat[28], 7.0 / 13.0, "cards_above_unseen");
        assert_eq!(feat[29], 5.0 / 13.0, "cards_below_unseen");
    }

    #[test]
    fn remaining_counts_exclude_played_cards() {
        // Hand: 7 of Spades (rank 5). Play ranks 6 and 7 (two cards above).
        let hand = make_hand(&[c(Suit::Spades, 5)]);
        let played = c(Suit::Spades, 6).bit() | c(Suit::Spades, 7).bit();
        let s = test_state(hand, NO_TRUMP, played, 4);
        let feat = &encode_hand_cards(&s, 0)[0];
        // Originally 7 above (ranks 6..12); 2 played → 5 remaining.
        assert_eq!(feat[28], 5.0 / 13.0, "cards_above_unseen after 2 played");
        // Below unchanged (no below-rank cards played).
        assert_eq!(feat[29], 5.0 / 13.0, "cards_below_unseen unchanged");
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
        assert_eq!(feat[28], 3.0 / 13.0, "above unseen excludes hand cards");
        assert_eq!(feat[29], 7.0 / 13.0, "below unseen excludes hand cards");
    }

    #[test]
    fn highest_and_lowest_ignore_my_own_cards() {
        // ♠K with my own ♠A above it: no opponent spade beats the king.
        // ♠3 with my own ♠2 below it: no opponent spade ducks under it.
        let hand = make_hand(&[
            c(Suit::Spades, 0),
            c(Suit::Spades, 1),
            c(Suit::Spades, 11),
            c(Suit::Spades, 12),
        ]);
        let s = test_state(hand, NO_TRUMP, 0, 4);
        let tokens = encode_hand_cards(&s, 0);
        assert_eq!(tokens[2][26], 1.0, "K is highest: only my A is above");
        assert_eq!(tokens[1][27], 1.0, "3 is lowest: only my 2 is below");
        assert_eq!(tokens[1][26], 0.0, "3 is not highest");
        assert_eq!(tokens[2][27], 0.0, "K is not lowest");
    }

    #[test]
    fn ace_has_zero_above_remaining() {
        let hand = make_hand(&[c(Suit::Diamonds, 12)]);
        let s = test_state(hand, NO_TRUMP, 0, 4);
        let feat = &encode_hand_cards(&s, 0)[0];
        assert_eq!(feat[28], 0.0, "Ace has no cards above");
        assert_eq!(feat[29], 12.0 / 13.0, "Ace has 12 cards below (not in hand)");
    }

    #[test]
    fn two_has_zero_below_remaining() {
        let hand = make_hand(&[c(Suit::Clubs, 0)]);
        let s = test_state(hand, NO_TRUMP, 0, 4);
        let feat = &encode_hand_cards(&s, 0)[0];
        assert_eq!(feat[28], 12.0 / 13.0, "2 has 12 cards above (not in hand)");
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
        assert_eq!(f0[25], 1.0 / 13.0, "suit_count_in_hand");
        assert_eq!(f0[26], 0.0, "not highest");
        assert_eq!(f0[27], 0.0, "not lowest");
        assert_eq!(f0[28], 7.0 / 13.0, "cards_above_unseen");
        assert_eq!(f0[29], 2.0 / 13.0, "cards_below_unseen");

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
        assert_eq!(f1[25], 1.0 / 13.0, "suit_count_in_hand");
        assert_eq!(f1[26], 0.0, "not highest (10,11,12 alive)");
        assert_eq!(f1[27], 0.0, "not lowest");
        assert_eq!(f1[28], 3.0 / 13.0, "cards_above_unseen");
        assert_eq!(f1[29], 8.0 / 13.0, "cards_below_unseen");

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
        assert_eq!(f2[25], 1.0 / 13.0, "suit_count_in_hand");
        assert_eq!(f2[26], 1.0, "highest in suit");
        assert_eq!(f2[27], 0.0, "not lowest");
        assert_eq!(f2[28], 0.0, "cards_above_unseen");
        assert_eq!(f2[29], 11.0 / 13.0, "cards_below_unseen");
        // No trick in progress: every card is legal, none beats anything.
        for f in &tokens {
            assert_eq!(f[30], 1.0, "legal to lead");
            assert_eq!(f[31], 0.0, "nothing to beat");
        }
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
            for j in [30, 31] {
                assert!(feat[j] == 0.0 || feat[j] == 1.0, "token {i}: [{j}] binary");
            }

            // Count features are scaled into [0, 1].
            assert!(feat[25] >= 1.0 / 13.0, "token {i}: suit_count ≥ 1/13");
            assert!(feat[25] <= 1.0, "token {i}: suit_count ≤ 1");
            for j in [28, 29] {
                assert!((0.0..=12.0 / 13.0).contains(&feat[j]), "token {i}: [{j}] in [0, 12/13]");
            }
        }
    }

    // ---------------------------------------------------------------
    // Full-game integration: use engine to set up a real mid-game state
    // ---------------------------------------------------------------

    #[test]
    fn encode_after_real_game_play() {
        use crate::bidding::{apply_bid, legal_bids};
        use crate::dealing::start_round;
        use crate::game::new_game;
        use crate::playing::{apply_play, legal_plays};
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
        let tokens = encode_played_cards(&s, 0);
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
        let tokens = encode_played_cards(&s, 0);
        assert_eq!(tokens.len(), 4);

        // Token 0: P1, ♥5 (rank 3, suit Hearts=1).
        let f = &tokens[0].features;
        assert_eq!(f[3], 1.0, "rank 3 one-hot");
        assert_eq!(f[17], 1.0, "Hearts suit one-hot");
        assert_eq!(f[25], 1.0, "relative seat 1 one-hot (perspective 0)");
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
        let tokens = encode_played_cards(&s, 0);
        assert_eq!(tokens.len(), 2);

        // Both should be is_current_trick=1, trick_complete=0, won_trick=0.
        for tok in &tokens {
            assert_eq!(tok.features[45], 0.0, "trick not complete");
            assert_eq!(tok.features[46], 0.0, "no won_trick for in-progress");
            assert_eq!(tok.features[47], 1.0, "is_current_trick");
        }
        // The trump ace leads the trick so far.
        assert_eq!(tokens[0].features[48], 1.0, "♠A winning so far");
        assert_eq!(tokens[1].features[48], 0.0, "♦3 not winning");

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
        let tokens = encode_played_cards(&s, 0);
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
        let tokens = encode_played_cards(&s, 0);
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
        let tokens = encode_played_cards(&s, 0);
        for tok in &tokens {
            assert_eq!(tok.features[44], 0.0, "no trump plays in no-trump round");
        }
    }

    #[test]
    fn position_in_trick_normalization() {
        // 4 players, position_in_trick should be 0/3, 1/3, 2/3, 3/3.
        let t = make_trick(&[(0, 0), (1, 13), (2, 26), (3, 39)], 0, 0);
        let s = state_with_tricks(4, 5, 5, NO_TRUMP, 3, &[t], &[]);
        let tokens = encode_played_cards(&s, 0);
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
        let tokens = encode_played_cards(&s, 0);
        for (i, tok) in tokens.iter().enumerate() {
            // Rank padding [13..16).
            for j in 13..16 {
                assert_eq!(tok.features[j], 0.0, "tok {i}: rank padding[{j}]");
            }
            // Suit padding [20..24).
            for j in 20..24 {
                assert_eq!(tok.features[j], 0.0, "tok {i}: suit padding[{j}]");
            }
            // Seat padding [27..40) for 3 players (only 0,1,2 used).
            for j in 27..40 {
                assert_eq!(tok.features[j], 0.0, "tok {i}: seat padding[{j}]");
            }
            assert_eq!(tok.features[48], 0.0, "tok {i}: completed trick has no winning_so_far");
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
    fn player_tokens_start_at_perspective_in_play_order() {
        // Perspective 2 at a 4-seat table: tokens are seats 2, 3, 0, 1, and
        // each carries its relative seat one-hot.
        let mut s = state_with_tricks(4, 5, 5, NO_TRUMP, 0, &[], &[]);
        s.tricks_won = [1, 2, 3, 4, 0, 0, 0, 0];
        let tokens = encode_player_states(&s, 2);
        for (rel, feat) in tokens.iter().enumerate() {
            for j in 0..16 {
                assert_eq!(feat[j], if j == rel { 1.0 } else { 0.0 }, "token {rel}: seat[{j}]");
            }
        }
        let won: Vec<f32> = tokens.iter().map(|f| f[17]).collect();
        let expected: Vec<f32> = [3.0, 4.0, 1.0, 2.0].iter().map(|w| w / 5.0).collect();
        assert_eq!(won, expected, "tokens follow seats 2, 3, 0, 1");
        // The dealer (seat 0) is relative seat 2.
        assert_eq!(tokens[2][20], 1.0);
    }

    #[test]
    fn is_to_move_marks_the_current_player() {
        let mut s = state_with_tricks(4, 5, 5, NO_TRUMP, 0, &[], &[]);
        s.current_player = 1;
        let tokens = encode_player_states(&s, 0);
        let to_move: Vec<f32> = tokens.iter().map(|f| f[22]).collect();
        assert_eq!(to_move, vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn not_yet_bid_differs_from_bid_zero() {
        // 4 players, dealer 3: seats 0 and 1 have bid (0 and 2), seat 2 is
        // to bid, the dealer bids last.
        let mut s = BlobState::empty();
        s.num_players = 4;
        s.cards_dealt = 5;
        s.start_cards = 5;
        s.game_phase = GamePhase::Bidding as u8;
        s.dealer = 3;
        s.bids[1] = 2;
        s.current_player = 2;
        let tokens = encode_player_states(&s, 2);
        // Relative order: seats 2, 3, 0, 1.
        let has: Vec<f32> = tokens.iter().map(|f| f[21]).collect();
        assert_eq!(has, vec![0.0, 0.0, 1.0, 1.0], "has_bid");
        // Seat 0 bid 0 and has 0 tricks: "met" so far. Seats 2 and 3 have
        // not bid: no status yet.
        assert_eq!(tokens[2][19], 1.0, "seat 0: bid 0, met");
        assert_eq!(tokens[0][19], 0.0, "seat 2: no bid yet");
        assert_eq!(tokens[1][19], 0.0, "seat 3: no bid yet");
        assert!((tokens[3][16] - 2.0 / 5.0).abs() < 1e-6, "seat 1 bid 2");
        assert!((tokens[3][18] - 2.0 / 5.0).abs() < 1e-6, "seat 1 needs 2");
    }

    #[test]
    fn cumulative_scores_are_not_an_input() {
        let t0 = make_trick(&[(0, 0), (1, 13), (2, 26), (3, 39)], 0, 0);
        let mut s = state_with_tricks(4, 5, 5, NO_TRUMP, 3, &[t0], &[(0, 1)]);
        s.hands[0] = c(Suit::Hearts, 0).bit() | c(Suit::Clubs, 10).bit();
        let before = encode(&s, 0);
        s.cumulative_scores = [90, 0, 180, 33, 0, 0, 0, 0];
        let after = encode(&s, 0);
        assert_eq!(before.features, after.features);
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
        assert_eq!(tokens[2][25], 1.0, "P2 void in Hearts");
        // P2: other voids should be 0.
        assert_eq!(tokens[2][24], 0.0, "P2 not void Spades");
        assert_eq!(tokens[2][26], 0.0, "P2 not void Clubs");
        assert_eq!(tokens[2][27], 0.0, "P2 not void Diamonds");

        // P3: followed suit, no void.
        assert_eq!(tokens[3][25], 0.0, "P3 NOT void in Hearts");

        // P1: leader, not checked for void.
        assert_eq!(tokens[1][25], 0.0, "P1 (leader) not flagged void");
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
        assert_eq!(tokens[1][24], 1.0, "P1 void in Spades from current trick");
        // P0 is leader, no void.
        assert_eq!(tokens[0][24], 0.0, "P0 (leader) not void");
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
        assert!((tokens[0][23] - 3.0 / 5.0).abs() < 1e-6, "P0: 3/5");
        assert!((tokens[1][23] - 5.0 / 5.0).abs() < 1e-6, "P1: 5/5");
        assert!((tokens[2][23] - 0.0).abs() < 1e-6, "P2: 0/5");
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
        use crate::bidding::{apply_bid, legal_bids};
        use crate::dealing::start_round;
        use crate::game::new_game;
        use crate::playing::{apply_play, legal_plays};
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

        let tokens = encode_played_cards(&s, 0);
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
        use crate::bidding::{apply_bid, legal_bids};
        use crate::dealing::start_round;
        use crate::game::new_game;
        use crate::playing::{apply_play, legal_plays};
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

        // Exactly one is_dealer flag, at the dealer's relative seat.
        let dealer_count: usize = tokens.iter().filter(|f| f[20] == 1.0).count();
        assert_eq!(dealer_count, 1);
        let dealer_rel = relative_seat(&s, perspective, s.dealer);
        assert_eq!(tokens[dealer_rel as usize][20], 1.0);

        // All features within expected ranges.
        for (p, feat) in tokens.iter().enumerate() {
            assert!(feat[16] >= 0.0 && feat[16] <= 1.0, "P{p} bid in [0,1]");
            assert!(feat[17] >= 0.0 && feat[17] <= 1.0, "P{p} tricks_won in [0,1]");
            assert!(feat[18] >= 0.0 && feat[18] <= 1.0, "P{p} tricks_needed in [0,1]");
            assert!(
                feat[19] == -1.0 || feat[19] == 0.0 || feat[19] == 1.0,
                "P{p} bid_status in {{-1,0,1}}"
            );
            assert_eq!(feat[21], 1.0, "P{p} has bid during play");
            assert!(feat[23] >= 0.0 && feat[23] <= 1.0, "P{p} cards_in_hand in [0,1]");
            for v in [20, 22, 24, 25, 26, 27] {
                assert!(
                    feat[v] == 0.0 || feat[v] == 1.0,
                    "P{p} [{v}] binary"
                );
            }
        }

        // The perspective is the current player: relative seat 0, to move.
        assert_eq!(tokens[0][0], 1.0);
        assert_eq!(tokens[0][22], 1.0);
        assert_eq!(tokens.iter().filter(|f| f[22] == 1.0).count(), 1);
    }

    // ===============================================================
    // Session 2.3 — Context token tests
    // ===============================================================

    #[test]
    fn context_token_trump_one_hot_spades() {
        let s = state_with_tricks(4, 5, 5, Suit::Spades as u8, 0, &[], &[]);
        let ctx = encode_context(&s, 0);
        assert_eq!(ctx[0], 1.0, "Spades");
        assert_eq!(ctx[1], 0.0);
        assert_eq!(ctx[2], 0.0);
        assert_eq!(ctx[3], 0.0);
        assert_eq!(ctx[4], 0.0);
    }

    #[test]
    fn context_token_trump_one_hot_no_trump() {
        let s = state_with_tricks(4, 5, 5, NO_TRUMP, 0, &[], &[]);
        let ctx = encode_context(&s, 0);
        assert_eq!(ctx[0], 0.0);
        assert_eq!(ctx[1], 0.0);
        assert_eq!(ctx[2], 0.0);
        assert_eq!(ctx[3], 0.0);
        assert_eq!(ctx[4], 1.0, "NoTrump");
    }

    #[test]
    fn context_token_trump_one_hot_diamonds() {
        let s = state_with_tricks(4, 5, 5, Suit::Diamonds as u8, 0, &[], &[]);
        let ctx = encode_context(&s, 0);
        assert_eq!(ctx[3], 1.0, "Diamonds");
        let sum: f32 = ctx[0..5].iter().sum();
        assert_eq!(sum, 1.0, "exactly one trump bit");
    }

    #[test]
    fn context_token_cards_dealt_normalization() {
        let s = state_with_tricks(4, 7, 7, NO_TRUMP, 0, &[], &[]);
        let ctx = encode_context(&s, 0);
        assert!((ctx[5] - 7.0 / 13.0).abs() < 1e-6, "cards_dealt=7/13");
    }

    #[test]
    fn context_token_current_trick_and_remaining() {
        // 5 cards dealt, 2 tricks completed → current=2/5, remaining=3/5.
        let t0 = make_trick(&[(0, 0), (1, 13), (2, 26), (3, 39)], 0, 0);
        let t1 = make_trick(&[(0, 1), (1, 14), (2, 27), (3, 40)], 1, 0);
        let s = state_with_tricks(4, 5, 5, NO_TRUMP, 0, &[t0, t1], &[]);
        let ctx = encode_context(&s, 0);
        assert!((ctx[6] - 2.0 / 5.0).abs() < 1e-6, "current_trick=2/5");
        assert!((ctx[7] - 3.0 / 5.0).abs() < 1e-6, "tricks_remaining=3/5");
    }

    #[test]
    fn context_token_no_tricks_played() {
        let s = state_with_tricks(4, 5, 5, NO_TRUMP, 0, &[], &[]);
        let ctx = encode_context(&s, 0);
        assert_eq!(ctx[6], 0.0, "current_trick=0");
        assert!((ctx[7] - 1.0).abs() < 1e-6, "tricks_remaining=5/5=1.0");
    }

    #[test]
    fn context_token_num_players_normalization() {
        for np in 3..=6u8 {
            let s = state_with_tricks(np, 5, 5, NO_TRUMP, 0, &[], &[]);
            let ctx = encode_context(&s, 0);
            assert!(
                (ctx[8] - np as f32 / 8.0).abs() < 1e-6,
                "num_players={np}/8"
            );
        }
    }

    #[test]
    fn context_token_round_number_normalization() {
        // 4 players, start_cards=5 → total_rounds = 2*5+4-2 = 12.
        let mut s = state_with_tricks(4, 5, 5, NO_TRUMP, 0, &[], &[]);
        s.round_idx = 3;
        let ctx = encode_context(&s, 0);
        assert!((ctx[9] - 3.0 / 12.0).abs() < 1e-6, "round_idx=3/12");
    }

    #[test]
    fn context_token_phase_bidding() {
        let mut s = BlobState::empty();
        s.num_players = 4;
        s.cards_dealt = 5;
        s.start_cards = 5;
        s.game_phase = GamePhase::Bidding as u8;
        let ctx = encode_context(&s, 0);
        assert_eq!(ctx[10], 1.0, "is_bidding");
        assert_eq!(ctx[11], 0.0, "not is_playing");
    }

    #[test]
    fn context_token_phase_playing() {
        let s = state_with_tricks(4, 5, 5, NO_TRUMP, 0, &[], &[]);
        let ctx = encode_context(&s, 0);
        assert_eq!(ctx[10], 0.0, "not is_bidding");
        assert_eq!(ctx[11], 1.0, "is_playing");
    }

    #[test]
    fn context_token_bidding_constraint_active() {
        // 4 players, 1 card dealt. Players 1,2,3 bid 0. Dealer=0.
        // Sum of others' bids = 0. Forbidden bid = 1-0 = 1. Constraint active.
        let mut s = BlobState::empty();
        s.num_players = 4;
        s.cards_dealt = 1;
        s.start_cards = 1;
        s.game_phase = GamePhase::Bidding as u8;
        s.dealer = 0;
        s.current_player = 0; // dealer's turn
        s.bids[1] = 0;
        s.bids[2] = 0;
        s.bids[3] = 0;
        let ctx = encode_context(&s, 0);
        assert_eq!(ctx[12], 1.0, "bidding constraint active for dealer");
    }

    #[test]
    fn context_token_bidding_constraint_not_active_non_dealer() {
        // Current player is not dealer → constraint not active.
        let mut s = BlobState::empty();
        s.num_players = 4;
        s.cards_dealt = 5;
        s.start_cards = 5;
        s.game_phase = GamePhase::Bidding as u8;
        s.dealer = 3;
        s.current_player = 1; // not dealer
        let ctx = encode_context(&s, 0);
        assert_eq!(ctx[12], 0.0, "non-dealer has no constraint");
    }

    #[test]
    fn context_token_bidding_constraint_not_active_playing_phase() {
        // Playing phase → constraint not active regardless.
        let s = state_with_tricks(4, 5, 5, NO_TRUMP, 0, &[], &[]);
        let ctx = encode_context(&s, 0);
        assert_eq!(ctx[12], 0.0, "no constraint in playing phase");
    }

    #[test]
    fn context_token_all_features_in_range() {
        let t = make_trick(&[(0, 0), (1, 13), (2, 26), (3, 39)], 0, 0);
        let s = state_with_tricks(4, 7, 7, Suit::Hearts as u8, 2, &[t], &[]);
        let ctx = encode_context(&s, 0);

        // Trump one-hot: exactly one 1.0 in [0..5).
        let trump_sum: f32 = ctx[0..5].iter().sum();
        assert_eq!(trump_sum, 1.0);

        // Normalized features in [0, 1].
        assert!(ctx[5] >= 0.0 && ctx[5] <= 1.0, "cards_dealt in [0,1]");
        assert!(ctx[6] >= 0.0 && ctx[6] <= 1.0, "current_trick in [0,1]");
        assert!(ctx[7] >= 0.0 && ctx[7] <= 1.0, "tricks_remaining in [0,1]");
        assert!(ctx[8] > 0.0 && ctx[8] <= 1.0, "num_players in (0,1]");
        assert!(ctx[9] >= 0.0 && ctx[9] < 1.0, "round_number in [0,1)");

        // Phase one-hot: exactly one 1.0 in [10..12).
        let phase_sum: f32 = ctx[10..12].iter().sum();
        assert_eq!(phase_sum, 1.0);

        // Bidding constraint is binary.
        assert!(ctx[12] == 0.0 || ctx[12] == 1.0);

        // Bid context: everyone has bid during play.
        assert_eq!(ctx[14], 0.0, "no seats left to bid");
        assert!(ctx[16] >= 0.0 && ctx[16] <= 1.0, "bidding position in [0,1]");
    }

    #[test]
    fn context_bid_totals_mid_bidding() {
        // 5 players, 7 cards, dealer 3: seats 4 and 0 bid 2 and 1, seat 1
        // is to bid.
        let mut s = BlobState::empty();
        s.num_players = 5;
        s.cards_dealt = 7;
        s.start_cards = 7;
        s.game_phase = GamePhase::Bidding as u8;
        s.dealer = 3;
        s.bids[4] = 2;
        s.bids[0] = 1;
        s.current_player = 1;
        let ctx = encode_context(&s, 1);
        assert!((ctx[13] - 3.0 / 13.0).abs() < 1e-6, "bid_sum 3");
        assert!((ctx[14] - 3.0 / 5.0).abs() < 1e-6, "seats 1, 2, 3 still to bid");
        assert!((ctx[15] - (3.0 - 7.0) / 7.0).abs() < 1e-6, "4 tricks unclaimed");
        assert!((ctx[16] - 2.0 / 4.0).abs() < 1e-6, "seat 1 bids third of five");
        // The dealer bids last.
        assert_eq!(encode_context(&s, 3)[16], 1.0);
        assert_eq!(encode_context(&s, 4)[16], 0.0);
    }

    #[test]
    fn context_bid_totals_during_play() {
        // Everyone has bid; the table overbid 5 cards by 2.
        let mut s = state_with_tricks(4, 5, 5, NO_TRUMP, 0, &[], &[]);
        s.bids = [2, 0, 3, 2, 0, 0, 0, 0];
        let ctx = encode_context(&s, 0);
        assert!((ctx[13] - 7.0 / 13.0).abs() < 1e-6);
        assert_eq!(ctx[14], 0.0);
        assert!((ctx[15] - 2.0 / 5.0).abs() < 1e-6);
    }

    // ===============================================================
    // Session 2.3 — Full encode() tests
    // ===============================================================

    #[test]
    fn encode_sequence_structure() {
        // 4 players, 5 cards dealt, 1 trick completed, 1 card in current trick.
        let t0 = make_trick(&[(0, 0), (1, 13), (2, 26), (3, 39)], 0, 0);
        let mut s = state_with_tricks(4, 5, 5, Suit::Spades as u8, 3, &[t0], &[(0, 1)]);
        // Give perspective player (P0) a hand of 4 cards (5 dealt - 1 played).
        s.hands[0] = c(Suit::Hearts, 0).bit()
            | c(Suit::Hearts, 5).bit()
            | c(Suit::Clubs, 10).bit()
            | c(Suit::Diamonds, 3).bit();

        let enc = encode(&s, 0);

        // Total tokens: 1 CLS + 1 context + 4 players + 4 hand + 5 played = 15.
        assert_eq!(enc.num_tokens, 15);
        assert_eq!(enc.features.len(), 15);
        assert_eq!(enc.token_types.len(), 15);
        assert_eq!(enc.chronological_indices.len(), 15);
    }

    #[test]
    fn encode_token_types_correct() {
        let mut s = state_with_tricks(3, 3, 3, NO_TRUMP, 0, &[], &[]);
        s.hands[1] = c(Suit::Spades, 0).bit()
            | c(Suit::Spades, 1).bit()
            | c(Suit::Spades, 2).bit();

        let enc = encode(&s, 1);

        // [CLS, context, P0, P1, P2, hand0, hand1, hand2]
        assert_eq!(enc.token_types[0], TOKEN_TYPE_CLS);
        assert_eq!(enc.token_types[1], TOKEN_TYPE_CONTEXT);
        assert_eq!(enc.token_types[2], TOKEN_TYPE_PLAYER);
        assert_eq!(enc.token_types[3], TOKEN_TYPE_PLAYER);
        assert_eq!(enc.token_types[4], TOKEN_TYPE_PLAYER);
        assert_eq!(enc.token_types[5], TOKEN_TYPE_HAND);
        assert_eq!(enc.token_types[6], TOKEN_TYPE_HAND);
        assert_eq!(enc.token_types[7], TOKEN_TYPE_HAND);
        assert_eq!(enc.num_tokens, 8); // no played cards
    }

    #[test]
    fn encode_token_types_with_played_cards() {
        let t0 = make_trick(&[(0, 0), (1, 13), (2, 26)], 0, 0);
        let mut s = state_with_tricks(3, 3, 3, NO_TRUMP, 2, &[t0], &[]);
        s.hands[0] = c(Suit::Spades, 1).bit() | c(Suit::Spades, 2).bit();

        let enc = encode(&s, 0);

        // [CLS, context, P0, P1, P2, hand0, hand1, played0, played1, played2]
        assert_eq!(enc.num_tokens, 10);
        for i in 7..10 {
            assert_eq!(enc.token_types[i], TOKEN_TYPE_PLAYED, "slot {i} is played");
        }
    }

    #[test]
    fn encode_cls_token_empty_features() {
        let s = state_with_tricks(3, 3, 3, NO_TRUMP, 0, &[], &[]);
        let enc = encode(&s, 0);
        assert!(enc.features[0].is_empty(), "CLS has no encoder features");
    }

    #[test]
    fn encode_context_token_has_correct_dim() {
        let s = state_with_tricks(4, 5, 5, Suit::Hearts as u8, 0, &[], &[]);
        let enc = encode(&s, 0);
        assert_eq!(enc.features[1].len(), CONTEXT_DIM, "context is 17-dim");
    }

    #[test]
    fn encode_player_state_tokens_have_correct_dim() {
        let s = state_with_tricks(5, 7, 7, NO_TRUMP, 0, &[], &[]);
        let enc = encode(&s, 0);
        for i in 2..7 {
            assert_eq!(
                enc.features[i].len(),
                PLAYER_STATE_DIM,
                "player token {i} is 28-dim"
            );
        }
    }

    #[test]
    fn encode_hand_card_tokens_have_correct_dim() {
        let mut s = state_with_tricks(3, 3, 3, NO_TRUMP, 0, &[], &[]);
        s.hands[0] = c(Suit::Spades, 0).bit()
            | c(Suit::Hearts, 5).bit()
            | c(Suit::Clubs, 12).bit();

        let enc = encode(&s, 0);
        // Hand tokens start at offset 1+1+3 = 5.
        for i in 5..8 {
            assert_eq!(
                enc.features[i].len(),
                HAND_CARD_DIM,
                "hand token {i} is 32-dim"
            );
        }
    }

    #[test]
    fn encode_played_card_tokens_have_correct_dim() {
        let t0 = make_trick(&[(0, 0), (1, 13), (2, 26)], 0, 0);
        let s = state_with_tricks(3, 3, 3, NO_TRUMP, 2, &[t0], &[]);
        let enc = encode(&s, 0);
        let played_start = 1 + 1 + 3 + Hand::new(s.hands[0]).count() as usize;
        for i in played_start..enc.num_tokens {
            assert_eq!(
                enc.features[i].len(),
                PLAYED_CARD_DIM,
                "played token {i} is 49-dim"
            );
        }
    }

    #[test]
    fn encode_hand_card_indices_match_hand_iter() {
        let mut s = state_with_tricks(4, 5, 5, NO_TRUMP, 0, &[], &[]);
        s.hands[2] = c(Suit::Diamonds, 12).bit()  // idx 51
            | c(Suit::Spades, 0).bit()             // idx 0
            | c(Suit::Hearts, 6).bit()             // idx 19
            | c(Suit::Clubs, 3).bit()              // idx 29
            | c(Suit::Spades, 11).bit();           // idx 11

        let enc = encode(&s, 2);
        let hand = Hand::new(s.hands[2]);
        let expected: Vec<u8> = hand.iter().map(|c| c.index()).collect();
        let actual: Vec<u8> = enc.hand_card_indices.iter().copied().collect();
        assert_eq!(actual, expected, "hand_card_indices matches Hand::iter()");
    }

    #[test]
    fn encode_hand_card_indices_ascending_order() {
        let mut s = state_with_tricks(3, 3, 3, NO_TRUMP, 0, &[], &[]);
        s.hands[0] = c(Suit::Clubs, 12).bit()     // idx 38
            | c(Suit::Spades, 5).bit()             // idx 5
            | c(Suit::Hearts, 0).bit();            // idx 13

        let enc = encode(&s, 0);
        // Hand::iter() yields ascending card index: 5, 13, 38.
        assert_eq!(enc.hand_card_indices[0], 5);
        assert_eq!(enc.hand_card_indices[1], 13);
        assert_eq!(enc.hand_card_indices[2], 38);
    }

    #[test]
    fn encode_chrono_indices_only_for_played() {
        let t0 = make_trick(&[(0, 0), (1, 13), (2, 26)], 0, 0);
        let mut s = state_with_tricks(3, 3, 3, NO_TRUMP, 2, &[t0], &[(0, 1)]);
        s.hands[0] = c(Suit::Spades, 2).bit() | c(Suit::Hearts, 5).bit();

        let enc = encode(&s, 0);
        let played_start = 1 + 1 + 3 + 2; // CLS + ctx + 3 players + 2 hand

        // Non-played tokens have chrono_index 0.
        for i in 0..played_start {
            assert_eq!(enc.chronological_indices[i], 0, "non-played slot {i}");
        }

        // Played tokens have sequential chrono indices.
        for (j, i) in (played_start..enc.num_tokens).enumerate() {
            assert_eq!(
                enc.chronological_indices[i], j as u8,
                "played slot {i} has chrono {j}"
            );
        }
    }

    #[test]
    fn encode_num_tokens_formula() {
        // 1 + 1 + num_players + hand_size + cards_played
        let t0 = make_trick(&[(0, 0), (1, 13), (2, 26), (3, 39)], 0, 0);
        let mut s = state_with_tricks(4, 5, 5, NO_TRUMP, 3, &[t0], &[(0, 1)]);
        s.hands[0] = c(Suit::Spades, 2).bit()
            | c(Suit::Hearts, 5).bit()
            | c(Suit::Clubs, 10).bit()
            | c(Suit::Diamonds, 3).bit();

        let enc = encode(&s, 0);
        let expected = 1 + 1 + 4 + 4 + 5; // CLS + ctx + players + hand + played
        assert_eq!(enc.num_tokens, expected);
    }

    #[test]
    fn encode_minimal_state_3p_1card_no_plays() {
        // Lower bound scenario: 3 players, 1 card, no plays yet.
        let mut s = state_with_tricks(3, 1, 1, NO_TRUMP, 0, &[], &[]);
        s.hands[0] = c(Suit::Spades, 0).bit();

        let enc = encode(&s, 0);
        // 1 + 1 + 3 + 1 + 0 = 6
        assert_eq!(enc.num_tokens, 6, "minimal state: 6 tokens");
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "only Bidding/Playing are valid")]
    fn encode_rejects_scoring_phase() {
        let mut s = BlobState::empty();
        s.num_players = 4;
        s.cards_dealt = 5;
        s.start_cards = 5;
        s.game_phase = GamePhase::Scoring as u8;
        let _ = encode(&s, 0);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "only Bidding/Playing are valid")]
    fn encode_rejects_complete_phase() {
        let mut s = BlobState::empty();
        s.num_players = 4;
        s.cards_dealt = 5;
        s.start_cards = 5;
        s.game_phase = GamePhase::Complete as u8;
        let _ = encode(&s, 0);
    }

    #[test]
    fn encode_bidding_phase_state() {
        // During bidding: no played cards, hands are full.
        let mut s = BlobState::empty();
        s.num_players = 4;
        s.cards_dealt = 5;
        s.start_cards = 5;
        s.game_phase = GamePhase::Bidding as u8;
        s.dealer = 3;
        s.current_player = 0;
        s.trump_suit = Suit::Hearts as u8;
        // Give player 0 a hand of 5 cards.
        s.hands[0] = c(Suit::Spades, 0).bit()
            | c(Suit::Spades, 1).bit()
            | c(Suit::Hearts, 5).bit()
            | c(Suit::Clubs, 10).bit()
            | c(Suit::Diamonds, 3).bit();

        let enc = encode(&s, 0);
        // 1 + 1 + 4 + 5 + 0 = 11
        assert_eq!(enc.num_tokens, 11);
        assert_eq!(enc.hand_card_indices.len(), 5);

        // Context token should show bidding phase.
        assert_eq!(enc.features[1][10], 1.0, "is_bidding");
        assert_eq!(enc.features[1][11], 0.0, "not is_playing");
    }

    // ---------------------------------------------------------------
    // Full encode integration with real game engine
    // ---------------------------------------------------------------

    #[test]
    fn encode_integration_early_game() {
        use crate::bidding::{apply_bid, legal_bids};
        use crate::dealing::start_round;
        use crate::game::new_game;
        use crate::playing::{apply_play, legal_plays};
        use rand_xoshiro::rand_core::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;

        let mut s = new_game(4, 5).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        start_round(&mut s, &mut rng);

        // Bidding: encode during bidding phase.
        let perspective = s.current_player;
        let enc_bid = encode(&s, perspective);
        assert_eq!(enc_bid.features[1][10], 1.0, "bidding phase");
        assert_eq!(enc_bid.features[1][11], 0.0, "not playing");

        // Complete bidding.
        while s.phase() == GamePhase::Bidding {
            let mask = legal_bids(&s);
            let bid = (0..=13u8).find(|b| (mask >> b) & 1 == 1).unwrap();
            apply_bid(&mut s, bid);
        }

        // Play 1 trick.
        for _ in 0..s.num_players {
            let mask = legal_plays(&s);
            let card = mask.trailing_zeros() as u8;
            apply_play(&mut s, card);
        }

        let perspective = s.current_player;
        let enc = encode(&s, perspective);

        // Verify structure.
        let hand = Hand::new(s.hands[perspective as usize]);
        let expected_tokens = 1 + 1 + 4 + hand.count() as usize
            + (s.tricks_completed as usize * 4 + s.trick_cards_played as usize);
        assert_eq!(enc.num_tokens, expected_tokens);

        // hand_card_indices matches Hand::iter().
        let hand_indices: Vec<u8> = hand.iter().map(|c| c.index()).collect();
        let enc_indices: Vec<u8> = enc.hand_card_indices.iter().copied().collect();
        assert_eq!(enc_indices, hand_indices);

        // Token types are in correct order.
        assert_eq!(enc.token_types[0], TOKEN_TYPE_CLS);
        assert_eq!(enc.token_types[1], TOKEN_TYPE_CONTEXT);
        for i in 2..6 {
            assert_eq!(enc.token_types[i], TOKEN_TYPE_PLAYER);
        }
        let hand_end = 6 + hand.count() as usize;
        for i in 6..hand_end {
            assert_eq!(enc.token_types[i], TOKEN_TYPE_HAND);
        }
        for i in hand_end..enc.num_tokens {
            assert_eq!(enc.token_types[i], TOKEN_TYPE_PLAYED);
        }
    }

    #[test]
    fn encode_integration_mid_game() {
        use crate::bidding::{apply_bid, legal_bids};
        use crate::dealing::start_round;
        use crate::game::new_game;
        use crate::playing::{apply_play, legal_plays};
        use rand_xoshiro::rand_core::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;

        let mut s = new_game(5, 7).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(77);
        start_round(&mut s, &mut rng);

        // Complete bidding.
        while s.phase() == GamePhase::Bidding {
            let mask = legal_bids(&s);
            let bid = (0..=13u8).find(|b| (mask >> b) & 1 == 1).unwrap();
            apply_bid(&mut s, bid);
        }

        // Play 4 full tricks + 2 cards of 5th trick.
        for _ in 0..4 {
            for _ in 0..s.num_players {
                let mask = legal_plays(&s);
                let card = mask.trailing_zeros() as u8;
                apply_play(&mut s, card);
            }
        }
        for _ in 0..2 {
            let mask = legal_plays(&s);
            let card = mask.trailing_zeros() as u8;
            apply_play(&mut s, card);
        }

        let perspective = s.current_player;
        let enc = encode(&s, perspective);

        let hand = Hand::new(s.hands[perspective as usize]);
        let total_played =
            s.tricks_completed as usize * 5 + s.trick_cards_played as usize;
        let expected = 1 + 1 + 5 + hand.count() as usize + total_played;
        assert_eq!(enc.num_tokens, expected);

        // Context features: playing phase, mid-round.
        assert_eq!(enc.features[1][11], 1.0, "playing phase");
        assert!((enc.features[1][6] - s.tricks_completed as f32 / 7.0).abs() < 1e-6);

        // All features are finite and non-NaN.
        for (i, feat) in enc.features.iter().enumerate() {
            for (j, &v) in feat.iter().enumerate() {
                assert!(v.is_finite(), "features[{i}][{j}] is finite");
            }
        }
    }

    #[test]
    fn encode_integration_late_game() {
        use crate::bidding::{apply_bid, legal_bids};
        use crate::dealing::start_round;
        use crate::game::new_game;
        use crate::playing::{apply_play, legal_plays};
        use rand_xoshiro::rand_core::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;

        let mut s = new_game(4, 5).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(99);
        start_round(&mut s, &mut rng);

        // Complete bidding.
        while s.phase() == GamePhase::Bidding {
            let mask = legal_bids(&s);
            let bid = (0..=13u8).find(|b| (mask >> b) & 1 == 1).unwrap();
            apply_bid(&mut s, bid);
        }

        // Play 4 tricks (1 trick remaining, 1 card each).
        for _ in 0..4 {
            for _ in 0..s.num_players {
                let mask = legal_plays(&s);
                let card = mask.trailing_zeros() as u8;
                apply_play(&mut s, card);
            }
        }

        let perspective = s.current_player;
        let enc = encode(&s, perspective);
        let hand = Hand::new(s.hands[perspective as usize]);

        // Each player has 1 card left.
        assert_eq!(hand.count(), 1, "1 card remaining");
        assert_eq!(enc.hand_card_indices.len(), 1);

        // 16 played cards (4 tricks × 4 players).
        let played_count: usize = enc
            .token_types
            .iter()
            .filter(|&&t| t == TOKEN_TYPE_PLAYED)
            .count();
        assert_eq!(played_count, 16);

        // Context: tricks_remaining = 1/5 = 0.2.
        assert!((enc.features[1][7] - 1.0 / 5.0).abs() < 1e-6);
    }

    // ===============================================================
    // Layout v2 — properties over real game states
    // ===============================================================

    #[test]
    fn legal_flag_matches_legal_plays() {
        use crate::playing::legal_plays;
        for s in random_game_states() {
            let me = s.current_player;
            let legal = if s.phase() == GamePhase::Playing { legal_plays(&s) } else { 0 };
            let hand = hand_card_indices(&s, me);
            for (feat, &card) in encode_hand_cards(&s, me).iter().zip(&hand) {
                assert_eq!(feat[30], flag((legal >> card) & 1 == 1), "card {card}");
            }
        }
    }

    #[test]
    fn beats_flag_matches_playing_the_card() {
        use crate::playing::{apply_play, legal_plays};
        let mut checked = 0;
        for s in random_game_states() {
            if s.phase() != GamePhase::Playing || s.trick_cards_played == 0 {
                continue;
            }
            let me = s.current_player;
            let my_slot = s.trick_cards_played;
            let legal = legal_plays(&s);
            let hand = hand_card_indices(&s, me);
            for (feat, &card) in encode_hand_cards(&s, me).iter().zip(&hand) {
                if (legal >> card) & 1 == 0 {
                    continue;
                }
                let mut after = s;
                apply_play(&mut after, card);
                let wins = if after.trick_cards_played == 0 {
                    after.trick_history[after.tricks_completed as usize - 1].winner == me
                } else {
                    current_trick_winner(&after) == Some(my_slot)
                };
                assert_eq!(feat[31], flag(wins), "card {card}");
                checked += 1;
            }
        }
        assert!(checked > 500, "only {checked} cards checked");
    }

    #[test]
    fn winning_so_far_marks_the_current_winner_only() {
        for s in random_game_states() {
            let tokens = encode_played_cards(&s, s.current_player);
            let marked: Vec<usize> =
                (0..tokens.len()).filter(|&i| tokens[i].features[48] == 1.0).collect();
            match current_trick_winner(&s) {
                None => assert!(marked.is_empty()),
                Some(slot) => {
                    let first = tokens.len() - s.trick_cards_played as usize;
                    assert_eq!(marked, vec![first + slot as usize]);
                }
            }
        }
    }

    /// `s` with every seat moved `k` places on.
    fn rotate_seats(s: &BlobState, k: u8) -> BlobState {
        let n = s.num_players;
        let r = |p: u8| (p + k) % n;
        let mut out = *s;
        for p in 0..n {
            let (from, to) = (p as usize, r(p) as usize);
            out.hands[to] = s.hands[from];
            out.bids[to] = s.bids[from];
            out.tricks_won[to] = s.tricks_won[from];
            out.cumulative_scores[to] = s.cumulative_scores[from];
        }
        out.current_player = r(s.current_player);
        out.dealer = r(s.dealer);
        out.trick_leader = r(s.trick_leader);
        for t in 0..s.tricks_completed as usize {
            let rec = &mut out.trick_history[t];
            for i in 0..rec.num_played as usize {
                rec.cards[i].0 = r(rec.cards[i].0);
            }
            rec.winner = r(rec.winner);
        }
        out
    }

    #[test]
    fn encoding_is_invariant_to_seat_rotation() {
        for (i, s) in random_game_states().iter().enumerate().step_by(7) {
            let k = 1 + (i as u8 % (s.num_players - 1));
            let rotated = rotate_seats(s, k);
            for p in [s.current_player, (s.dealer + 1) % s.num_players] {
                let a = encode(s, p);
                let b = encode(&rotated, (p + k) % s.num_players);
                assert_eq!(a.features, b.features, "state {i}, seat {p}, shift {k}");
            }
        }
    }

    #[test]
    fn features_are_finite_and_one_hots_are_exact() {
        for s in random_game_states() {
            let enc = encode(&s, s.current_player);
            for (i, feat) in enc.features.iter().enumerate() {
                assert!(feat.iter().all(|v| v.is_finite()), "token {i}");
                let one_hot = |r: std::ops::Range<usize>| feat[r].iter().sum::<f32>();
                match enc.token_types[i] {
                    TOKEN_TYPE_HAND => {
                        assert_eq!((one_hot(0..16), one_hot(16..24)), (1.0, 1.0));
                        assert!(feat.iter().all(|v| (0.0..=1.0).contains(v)));
                    }
                    TOKEN_TYPE_PLAYED => {
                        let hots = (one_hot(0..16), one_hot(16..24), one_hot(24..40));
                        assert_eq!(hots, (1.0, 1.0, 1.0));
                    }
                    TOKEN_TYPE_PLAYER => {
                        assert_eq!(one_hot(0..16), 1.0);
                        assert!(feat.iter().all(|v| (-1.0..=1.0).contains(v)));
                    }
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn encoder_version_follows_feature_width() {
        assert_eq!(FEAT_DIM, 49);
        assert_eq!(EncoderVersion::from_feat_dim(48), Some(EncoderVersion::V1));
        assert_eq!(EncoderVersion::from_feat_dim(FEAT_DIM), Some(EncoderVersion::V2));
        assert_eq!(EncoderVersion::from_feat_dim(64), None);
        assert_eq!(EncoderVersion::CURRENT.feat_dim(), FEAT_DIM);
        for v in [EncoderVersion::V1, EncoderVersion::V2] {
            let s = random_game_states()[40];
            let enc = v.encode(&s, s.current_player);
            assert!(enc.features.iter().all(|f| f.len() <= v.feat_dim()));
        }
    }

    /// `scripts/export_onnx.py` rebuilds the network in PyTorch; its token
    /// widths must equal the encoder's or the exported model can't load
    /// the trained weights.
    #[test]
    fn export_script_mirrors_feature_widths() {
        let script = include_str!("../../scripts/export_onnx.py");
        let value = |name: &str| -> usize {
            let prefix = format!("{name} = ");
            let line = script
                .lines()
                .find(|l| l.starts_with(&prefix))
                .unwrap_or_else(|| panic!("{name} not defined in export_onnx.py"));
            let rest = line[prefix.len()..].split('#').next().unwrap();
            rest.trim().parse().unwrap_or_else(|_| panic!("not an integer: {line}"))
        };
        assert_eq!(value("HAND_DIM"), HAND_CARD_DIM);
        assert_eq!(value("PLAYED_DIM"), PLAYED_CARD_DIM);
        assert_eq!(value("PLAYER_DIM"), PLAYER_STATE_DIM);
        assert_eq!(value("CONTEXT_DIM"), CONTEXT_DIM);
        assert_eq!(value("FEAT_DIM"), FEAT_DIM);
    }
}
