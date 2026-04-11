//! Round structure and trump rotation helpers.
//!
//! Trump cycles every 5 rounds (♠ → ♥ → ♣ → ♦ → no-trump). The per-game
//! round structure is symmetric: descending from `C` to 2, a plateau of
//! `num_players` one-card rounds, then ascending back to `C`.
//!
//! The "no-trump" round encodes as [`NO_TRUMP`] = 4 in `BlobState.trump_suit`,
//! sitting just past the four [`crate::card::Suit`] values (0..=3).

use smallvec::SmallVec;

use crate::card::{MAX_CARDS_DEALT, NUM_CARDS};
use crate::state::{MAX_PLAYERS, MIN_PLAYERS};

/// Sentinel value stored in `BlobState.trump_suit` for no-trump rounds.
pub const NO_TRUMP: u8 = 4;

/// Length of the trump rotation cycle: ♠, ♥, ♣, ♦, no-trump.
pub const TRUMP_CYCLE_LEN: u8 = 5;

/// Trump suit for a given round index. Returns 0..=3 for the four
/// [`crate::card::Suit`] values, or [`NO_TRUMP`] for no-trump rounds.
#[inline]
pub const fn trump_for_round(round_idx: u32) -> u8 {
    (round_idx % TRUMP_CYCLE_LEN as u32) as u8
}

/// Total rounds in a game: `2C + num_players − 2`.
///
/// **Note**: this differs from `legacy/game-engine/constants.py`, which
/// produces one extra 1-card round. See `development-plan.md` Session 1.2.
#[inline]
pub const fn total_rounds(start_cards: u8, num_players: u8) -> u8 {
    2 * start_cards + num_players - 2
}

/// Validation errors for round-structure parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundParamsError {
    PlayerCountOutOfRange,
    StartCardsZero,
    StartCardsExceedsCap,
    DeckExceeded,
}

/// Validate `(start_cards, num_players)` per game rules:
/// `num_players ∈ [MIN_PLAYERS, MAX_PLAYERS]`, `1 ≤ start_cards ≤ MAX_CARDS_DEALT`,
/// and `start_cards × num_players ≤ 52`.
pub const fn validate_round_params(
    start_cards: u8,
    num_players: u8,
) -> Result<(), RoundParamsError> {
    if (num_players as usize) < MIN_PLAYERS || (num_players as usize) > MAX_PLAYERS {
        return Err(RoundParamsError::PlayerCountOutOfRange);
    }
    if start_cards == 0 {
        return Err(RoundParamsError::StartCardsZero);
    }
    if start_cards as usize > MAX_CARDS_DEALT {
        return Err(RoundParamsError::StartCardsExceedsCap);
    }
    if (start_cards as usize) * (num_players as usize) > NUM_CARDS as usize {
        return Err(RoundParamsError::DeckExceeded);
    }
    Ok(())
}

/// Symmetric round structure as a stack-allocated `SmallVec`.
///
/// Pattern: descending `[C, C-1, …, 2]` (`C-1` entries), then `num_players`
/// rounds of 1 card, then ascending `[2, 3, …, C]` (`C-1` entries). Total
/// length matches [`total_rounds`].
///
/// Panics in debug if [`validate_round_params`] would reject the inputs.
pub fn round_structure(start_cards: u8, num_players: u8) -> SmallVec<[u8; 32]> {
    debug_assert!(validate_round_params(start_cards, num_players).is_ok());
    let total = total_rounds(start_cards, num_players) as usize;
    let mut out = SmallVec::with_capacity(total);
    // Descending C, C-1, …, 2.
    for c in (2..=start_cards).rev() {
        out.push(c);
    }
    // One-card plateau.
    for _ in 0..num_players {
        out.push(1);
    }
    // Ascending 2, 3, …, C.
    for c in 2..=start_cards {
        out.push(c);
    }
    debug_assert_eq!(out.len(), total);
    out
}

/// O(1) lookup of cards dealt for a specific round index, equivalent to
/// `round_structure(start_cards, num_players)[round_idx]` without allocating.
pub fn cards_dealt_for_round(round_idx: u8, start_cards: u8, num_players: u8) -> u8 {
    debug_assert!(validate_round_params(start_cards, num_players).is_ok());
    debug_assert!(round_idx < total_rounds(start_cards, num_players));
    let descending_len = start_cards - 1; // [C..=2]
    let plateau_end = descending_len + num_players;
    if round_idx < descending_len {
        start_cards - round_idx
    } else if round_idx < plateau_end {
        1
    } else {
        // Ascending segment: round_idx - plateau_end ∈ [0..C-1)
        (round_idx - plateau_end) + 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::Suit;

    #[test]
    fn trump_cycles_through_five() {
        assert_eq!(trump_for_round(0), Suit::Spades as u8);
        assert_eq!(trump_for_round(1), Suit::Hearts as u8);
        assert_eq!(trump_for_round(2), Suit::Clubs as u8);
        assert_eq!(trump_for_round(3), Suit::Diamonds as u8);
        assert_eq!(trump_for_round(4), NO_TRUMP);
        // Cycle repeats.
        assert_eq!(trump_for_round(5), Suit::Spades as u8);
        assert_eq!(trump_for_round(10), Suit::Spades as u8);
        assert_eq!(trump_for_round(11), Suit::Hearts as u8);
        assert_eq!(trump_for_round(14), NO_TRUMP);
    }

    #[test]
    fn total_rounds_matches_corrected_formula() {
        // 5 players, C=7 → 17 rounds (README example).
        assert_eq!(total_rounds(7, 5), 17);
        // 5 players, C=8 → 19 rounds.
        assert_eq!(total_rounds(8, 5), 19);
        // 4 players, C=5 → 12 rounds (corrected; legacy gave 13).
        assert_eq!(total_rounds(5, 4), 12);
        // 3 players, C=7 → 15 rounds (corrected; legacy gave 16).
        assert_eq!(total_rounds(7, 3), 15);
    }

    #[test]
    fn round_structure_5p_7c_matches_readme_example() {
        let s = round_structure(7, 5);
        assert_eq!(
            s.as_slice(),
            &[7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn round_structure_5p_8c() {
        let s = round_structure(8, 5);
        assert_eq!(
            s.as_slice(),
            &[8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8]
        );
    }

    #[test]
    fn round_structure_4p_5c_corrected() {
        // Corrected: 4 ones (= num_players), 12 rounds total.
        let s = round_structure(5, 4);
        assert_eq!(s.as_slice(), &[5, 4, 3, 2, 1, 1, 1, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn round_structure_3p_7c_corrected() {
        // Corrected: 3 ones, 15 rounds total.
        let s = round_structure(7, 3);
        assert_eq!(
            s.as_slice(),
            &[7, 6, 5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn cards_dealt_for_round_matches_round_structure() {
        for &(c, n) in &[(7u8, 5u8), (8, 5), (5, 4), (7, 3), (13, 4), (8, 6)] {
            let s = round_structure(c, n);
            for (i, &cards) in s.iter().enumerate() {
                assert_eq!(
                    cards_dealt_for_round(i as u8, c, n),
                    cards,
                    "mismatch at round {i} for C={c}, n={n}"
                );
            }
        }
    }

    #[test]
    fn validate_rejects_player_count() {
        assert_eq!(
            validate_round_params(5, 2),
            Err(RoundParamsError::PlayerCountOutOfRange)
        );
        assert_eq!(
            validate_round_params(5, 9),
            Err(RoundParamsError::PlayerCountOutOfRange)
        );
    }

    #[test]
    fn validate_rejects_zero_or_oversized_start_cards() {
        assert_eq!(
            validate_round_params(0, 4),
            Err(RoundParamsError::StartCardsZero)
        );
        assert_eq!(
            validate_round_params(14, 4),
            Err(RoundParamsError::StartCardsExceedsCap)
        );
    }

    #[test]
    fn validate_rejects_deck_overflow() {
        // 8 players × 7 cards = 56 > 52
        assert_eq!(
            validate_round_params(7, 8),
            Err(RoundParamsError::DeckExceeded)
        );
    }

    #[test]
    fn validate_accepts_max_valid() {
        // 4 × 13 = 52 (full deck) is allowed.
        assert!(validate_round_params(13, 4).is_ok());
        // 6 × 8 = 48
        assert!(validate_round_params(8, 6).is_ok());
        // 8 × 6 = 48
        assert!(validate_round_params(6, 8).is_ok());
    }
}
