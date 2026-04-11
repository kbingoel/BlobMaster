//! `BlobState`, the stack-allocated game state struct.
//!
//! Sized for `C ≤ 13` and up to 8 players. The [`Copy`] derive is load-bearing:
//! MCTS relies on zero-cost cloning for determinization, and Section 5.3 pins
//! the shuffling RNG *outside* the state so copies stay bit-for-bit identical.

use serde::{Deserialize, Serialize};

use crate::card::MAX_CARDS_DEALT;

/// Minimum players supported by the engine.
pub const MIN_PLAYERS: usize = 3;
/// Hard cap on players supported by the engine.
pub const MAX_PLAYERS: usize = 8;

/// Game phase discriminant. `u8` repr so `BlobState` can hold it directly.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GamePhase {
    Bidding = 0,
    Playing = 1,
    Scoring = 2,
}

impl GamePhase {
    #[inline]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(GamePhase::Bidding),
            1 => Some(GamePhase::Playing),
            2 => Some(GamePhase::Scoring),
            _ => None,
        }
    }
}

/// Record of a single completed trick.
///
/// `cards[i] = (player, card)`, ordered by play sequence (index 0 = trick
/// leader). `num_played` equals `num_players` for every completed trick;
/// it is kept for validation/invariant checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrickRecord {
    pub cards: [(u8, u8); MAX_PLAYERS],
    pub num_played: u8,
    pub winner: u8,
    pub suit_led: u8,
}

impl Default for TrickRecord {
    fn default() -> Self {
        TrickRecord {
            cards: [(0, 0); MAX_PLAYERS],
            num_played: 0,
            winner: 0,
            suit_led: 0,
        }
    }
}

/// Full game state. Stack-only (~410 bytes), `Copy` for MCTS cloning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobState {
    /// Per-player hand bitmasks.
    pub hands: [u64; MAX_PLAYERS],
    /// Bitmask of cards played during the current round (any player).
    /// Maintained incrementally — read by the encoder for
    /// `cards_above_remaining` and `is_highest_in_suit`.
    pub played_this_round: u64,

    pub bids: [u8; MAX_PLAYERS],
    pub tricks_won: [u8; MAX_PLAYERS],
    pub trump_suit: u8,
    pub current_player: u8,
    pub dealer: u8,
    pub num_players: u8,
    pub cards_dealt: u8,
    /// Stored as `GamePhase as u8`.
    pub game_phase: u8,

    pub trick_leader: u8,
    /// **Card indices** (not players) played in the in-progress trick,
    /// slot `i` = card from player `(trick_leader + i) % num_players`.
    pub trick_play_order: [u8; MAX_PLAYERS],
    pub trick_cards_played: u8,

    /// 13 = hard max tricks given `C ≤ 13`. Training distributions never
    /// exceed 8 tricks/round, but the cap sizes fixed arrays uniformly.
    pub trick_history: [TrickRecord; MAX_CARDS_DEALT],
    pub tricks_completed: u8,

    /// Multi-round running totals, referenced by player state tokens.
    pub cumulative_scores: [u16; MAX_PLAYERS],
}

impl BlobState {
    /// Fresh empty state. Fields that will be initialized by `new_game`
    /// (Section 1.4) are zeroed here.
    pub const fn empty() -> Self {
        BlobState {
            hands: [0; MAX_PLAYERS],
            played_this_round: 0,
            bids: [0; MAX_PLAYERS],
            tricks_won: [0; MAX_PLAYERS],
            trump_suit: 0,
            current_player: 0,
            dealer: 0,
            num_players: 0,
            cards_dealt: 0,
            game_phase: GamePhase::Bidding as u8,
            trick_leader: 0,
            trick_play_order: [0; MAX_PLAYERS],
            trick_cards_played: 0,
            trick_history: [TrickRecord {
                cards: [(0, 0); MAX_PLAYERS],
                num_played: 0,
                winner: 0,
                suit_led: 0,
            }; MAX_CARDS_DEALT],
            tricks_completed: 0,
            cumulative_scores: [0; MAX_PLAYERS],
        }
    }

    /// Typed accessor for the phase field.
    #[inline]
    pub fn phase(&self) -> GamePhase {
        GamePhase::from_u8(self.game_phase).expect("valid game_phase")
    }
}

impl Default for BlobState {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn blobstate_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<BlobState>();
    }

    #[test]
    fn blobstate_size_within_budget() {
        // Plan target: ~410 bytes. Leave headroom for alignment padding,
        // but fail if we blow past the "~6 cache lines" (~416 B) estimate
        // by more than a cache line.
        let n = size_of::<BlobState>();
        assert!(
            n <= 512,
            "BlobState grew to {n} bytes; expected ≤ 512"
        );
    }

    #[test]
    fn trick_record_default_is_zeroed() {
        let tr = TrickRecord::default();
        assert_eq!(tr.num_played, 0);
        assert_eq!(tr.winner, 0);
        assert_eq!(tr.suit_led, 0);
        assert!(tr.cards.iter().all(|&(p, c)| p == 0 && c == 0));
    }

    #[test]
    fn empty_state_phase_roundtrips() {
        let s = BlobState::empty();
        assert_eq!(s.phase(), GamePhase::Bidding);
        assert_eq!(s.played_this_round, 0);
        assert_eq!(s.tricks_completed, 0);
    }

    #[test]
    fn copy_preserves_fields() {
        let mut a = BlobState::empty();
        a.num_players = 5;
        a.cards_dealt = 7;
        a.hands[0] = 0x0000_0000_0000_1FFF;
        a.cumulative_scores[2] = 42;

        let b = a; // Copy
        assert_eq!(b.num_players, 5);
        assert_eq!(b.cards_dealt, 7);
        assert_eq!(b.hands[0], 0x1FFF);
        assert_eq!(b.cumulative_scores[2], 42);
    }
}
