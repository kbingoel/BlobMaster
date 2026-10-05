//! Round structure and trump rotation helpers.
//!
//! Trump cycles every 5 rounds (♠ → ♥ → ♣ → ♦ → no-trump). The per-game
//! round structure is symmetric: descending from `C` to 2, a plateau of
//! `num_players` one-card rounds, then ascending back to `C`.
//!
//! The "no-trump" round encodes as [`NO_TRUMP`] = 4 in `BlobState.trump_suit`,
//! sitting just past the four [`crate::card::Suit`] values (0..=3).
//!
//! Self-play plays single rounds (gen-2.md §5.2); [`RoundMix`] draws their
//! parameters from the rounds of real games.

use rand::Rng;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::card::{MAX_CARDS_DEALT, NUM_CARDS};
use crate::dealing::RoundParams;
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
/// produces one extra 1-card round.
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
    /// Trump above [`NO_TRUMP`].
    TrumpOutOfRange,
    /// Dealer seat not below the player count.
    DealerOutOfRange,
    /// A [`RoundMix`] with no table sizes or a non-finite exponent.
    InvalidMix,
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

/// The rounds single-round self-play draws from (gen-2.md §5.2).
///
/// - **Table size:** uniform over `players`.
/// - **Cards dealt:** one of the rounds of a game starting at `start_cards`
///   at that table, so 1-card rounds come up once per player and the others
///   twice. Each round is weighted by `cards_dealt ^ large_round_exponent`:
///   0 keeps real games' mix, larger values oversample the larger rounds,
///   where bidding matters most.
/// - **Trump:** uniform over the four suits and no-trump, the rotation's
///   share over a game.
/// - **Dealer:** uniform.
///
/// Unknown keys are an error, so a stale config can't half-load.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoundMix {
    pub players: Vec<u8>,
    pub start_cards: u8,
    #[serde(default)]
    pub large_round_exponent: f32,
}

impl Default for RoundMix {
    /// 5 players starting at 7 cards, real games' mix.
    fn default() -> Self {
        Self { players: vec![5], start_cards: 7, large_round_exponent: 0.0 }
    }
}

impl RoundMix {
    /// Every table size must make a valid game at `start_cards`.
    pub fn validate(&self) -> Result<(), RoundParamsError> {
        if self.players.is_empty() || !self.large_round_exponent.is_finite() {
            return Err(RoundParamsError::InvalidMix);
        }
        for &n in &self.players {
            validate_round_params(self.start_cards, n)?;
        }
        Ok(())
    }

    /// Draw one round's parameters. The mix must pass [`RoundMix::validate`].
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> RoundParams {
        let num_players = self.players[rng.gen_range(0..self.players.len())];
        let rounds = round_structure(self.start_cards, num_players);
        let weight = |c: u8| (c as f64).powf(self.large_round_exponent as f64);
        let total: f64 = rounds.iter().map(|&c| weight(c)).sum();
        let mut x = rng.gen::<f64>() * total;
        let mut cards_dealt = rounds[rounds.len() - 1];
        for &c in &rounds {
            if x < weight(c) {
                cards_dealt = c;
                break;
            }
            x -= weight(c);
        }
        RoundParams {
            num_players,
            cards_dealt,
            trump: rng.gen_range(0..TRUMP_CYCLE_LEN),
            dealer: rng.gen_range(0..num_players),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::Suit;
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn shares(mix: &RoundMix, draws: usize) -> [f64; 14] {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5A3);
        let mut count = [0usize; 14];
        for _ in 0..draws {
            count[mix.sample(&mut rng).cards_dealt as usize] += 1;
        }
        count.map(|c| c as f64 / draws as f64)
    }

    #[test]
    fn round_mix_reproduces_a_games_rounds() {
        // 5p/7c: 17 rounds, five of them 1-card, two of every other size.
        let p = shares(&RoundMix::default(), 100_000);
        assert!((p[1] - 5.0 / 17.0).abs() < 0.01, "1-card share {}", p[1]);
        for (c, share) in p.iter().enumerate().take(8).skip(2) {
            assert!((share - 2.0 / 17.0).abs() < 0.01, "{c}-card share {share}");
        }
        assert_eq!(p[8..].iter().sum::<f64>(), 0.0);
    }

    #[test]
    fn round_mix_exponent_oversamples_large_rounds() {
        // Weight c per round: 1-card rounds 5 × 1, others 2 × c; total 59.
        let mix = RoundMix { large_round_exponent: 1.0, ..RoundMix::default() };
        let p = shares(&mix, 100_000);
        assert!((p[1] - 5.0 / 59.0).abs() < 0.01, "1-card share {}", p[1]);
        assert!((p[7] - 14.0 / 59.0).abs() < 0.01, "7-card share {}", p[7]);
    }

    #[test]
    fn round_mix_draws_valid_rounds() {
        let mix = RoundMix { players: vec![3, 4, 6], start_cards: 8, large_round_exponent: 0.5 };
        mix.validate().unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let mut trumps = [0usize; 5];
        for _ in 0..5_000 {
            let r = mix.sample(&mut rng);
            r.validate().unwrap();
            assert!([3, 4, 6].contains(&r.num_players));
            assert!((1..=8).contains(&r.cards_dealt));
            trumps[r.trump as usize] += 1;
        }
        assert!(trumps.iter().all(|&t| t > 800), "trumps {trumps:?}");
    }

    #[test]
    fn round_mix_rejects_bad_configs() {
        assert_eq!(RoundMix { players: vec![], ..RoundMix::default() }.validate(), Err(RoundParamsError::InvalidMix));
        let nan = RoundMix { large_round_exponent: f32::NAN, ..RoundMix::default() };
        assert_eq!(nan.validate(), Err(RoundParamsError::InvalidMix));
        let big = RoundMix { players: vec![8], start_cards: 7, large_round_exponent: 0.0 };
        assert_eq!(big.validate(), Err(RoundParamsError::DeckExceeded));
        let toml_ok = "players = [4, 5]\nstart_cards = 7\n";
        let mix: RoundMix = toml::from_str(toml_ok).unwrap();
        assert_eq!(mix.large_round_exponent, 0.0);
        assert!(toml::from_str::<RoundMix>(&format!("{toml_ok}round_number = 3\n")).is_err());
    }

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
