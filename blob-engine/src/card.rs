//! Card encoding.
//!
//! `card_index = suit * 13 + rank`. Suits: ♠=0 ♥=1 ♣=2 ♦=3. Ranks: 2=0 … A=12.

use serde::{Deserialize, Serialize};

/// Number of ranks (2..=A).
pub const NUM_RANKS: u8 = 13;
/// Number of suits (♠ ♥ ♣ ♦).
pub const NUM_SUITS: u8 = 4;
/// Total cards in a standard deck.
pub const NUM_CARDS: u8 = NUM_RANKS * NUM_SUITS;
/// Hard cap on `cards_dealt` per round.
///
/// Never binding in training (C ∈ {7, 8}); keeping the cap explicit lets
/// every fixed-size array in [`crate::state`] use 13 as its upper bound.
pub const MAX_CARDS_DEALT: usize = 13;

/// Suit enum mirroring the `suit * 13 + rank` layout.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Suit {
    Spades = 0,
    Hearts = 1,
    Clubs = 2,
    Diamonds = 3,
}

impl Suit {
    /// Rebuild from the raw `u8` encoding. Returns `None` for `>= 4`.
    #[inline]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Suit::Spades),
            1 => Some(Suit::Hearts),
            2 => Some(Suit::Clubs),
            3 => Some(Suit::Diamonds),
            _ => None,
        }
    }

    #[inline]
    pub const fn index(self) -> u8 {
        self as u8
    }

    /// 13-bit mask covering every card of this suit within a `u64` hand.
    #[inline]
    pub const fn mask(self) -> u64 {
        0x1FFF << (self as u8 * NUM_RANKS)
    }
}

/// Lightweight card identifier. Stored as the flat `u8` index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Card(u8);

impl Card {
    /// Build from `suit` and `rank`. Panics in debug if out of range.
    #[inline]
    pub const fn new(suit: Suit, rank: u8) -> Self {
        debug_assert!(rank < NUM_RANKS);
        Card(suit as u8 * NUM_RANKS + rank)
    }

    /// Wrap a raw `u8` index. Returns `None` for `>= 52`.
    #[inline]
    pub const fn from_index(idx: u8) -> Option<Self> {
        if idx < NUM_CARDS {
            Some(Card(idx))
        } else {
            None
        }
    }

    /// Wrap an index without bounds check. Caller must ensure `idx < 52`.
    #[inline]
    pub const fn from_index_unchecked(idx: u8) -> Self {
        debug_assert!(idx < NUM_CARDS);
        Card(idx)
    }

    #[inline]
    pub const fn index(self) -> u8 {
        self.0
    }

    #[inline]
    pub const fn suit(self) -> Suit {
        // `self.0 / 13` is always < 4 for valid cards.
        match self.0 / NUM_RANKS {
            0 => Suit::Spades,
            1 => Suit::Hearts,
            2 => Suit::Clubs,
            _ => Suit::Diamonds,
        }
    }

    #[inline]
    pub const fn rank(self) -> u8 {
        self.0 % NUM_RANKS
    }

    /// Single-bit `u64` mask for this card.
    #[inline]
    pub const fn bit(self) -> u64 {
        1u64 << self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_round_trip() {
        for i in 0..NUM_CARDS {
            let c = Card::from_index(i).unwrap();
            assert_eq!(c.index(), i);
            assert_eq!(Card::new(c.suit(), c.rank()).index(), i);
        }
    }

    #[test]
    fn from_index_rejects_out_of_range() {
        assert!(Card::from_index(52).is_none());
        assert!(Card::from_index(255).is_none());
    }

    #[test]
    fn suit_mask_covers_13_bits() {
        for s in [Suit::Spades, Suit::Hearts, Suit::Clubs, Suit::Diamonds] {
            assert_eq!(s.mask().count_ones(), 13);
        }
        // Masks are disjoint and together cover the full deck.
        let all = Suit::Spades.mask()
            | Suit::Hearts.mask()
            | Suit::Clubs.mask()
            | Suit::Diamonds.mask();
        assert_eq!(all, (1u64 << 52) - 1);
    }

    #[test]
    fn suit_of_card() {
        assert_eq!(Card::new(Suit::Spades, 0).suit(), Suit::Spades);
        assert_eq!(Card::new(Suit::Hearts, 12).suit(), Suit::Hearts);
        assert_eq!(Card::new(Suit::Clubs, 6).suit(), Suit::Clubs);
        assert_eq!(Card::new(Suit::Diamonds, 12).index(), 51);
    }
}
