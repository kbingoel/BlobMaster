//! `u64` bitmask hand. Bit *N* set ⇔ card index *N* present.

use serde::{Deserialize, Serialize};

use crate::card::{Card, Suit, NUM_CARDS};

/// A set of cards stored as a `u64` bitmask.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Hand(pub u64);

impl Hand {
    pub const EMPTY: Hand = Hand(0);

    #[inline]
    pub const fn new(bits: u64) -> Self {
        Hand(bits)
    }

    #[inline]
    pub const fn bits(self) -> u64 {
        self.0
    }

    #[inline]
    pub fn add(&mut self, card: Card) {
        self.0 |= card.bit();
    }

    #[inline]
    pub fn remove(&mut self, card: Card) {
        self.0 &= !card.bit();
    }

    #[inline]
    pub const fn contains(self, card: Card) -> bool {
        (self.0 & card.bit()) != 0
    }

    #[inline]
    pub const fn count(self) -> u8 {
        self.0.count_ones() as u8
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Bitmask of cards of a given suit, still expressed in the full 52-bit layout.
    #[inline]
    pub const fn cards_of_suit(self, suit: Suit) -> u64 {
        self.0 & suit.mask()
    }

    /// Iterator over cards in ascending index order. Non-destructive.
    #[inline]
    pub fn iter(self) -> HandIter {
        HandIter(self.0)
    }
}

impl IntoIterator for Hand {
    type Item = Card;
    type IntoIter = HandIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator that yields each card once via `trailing_zeros`/`blsr`.
pub struct HandIter(u64);

impl Iterator for HandIter {
    type Item = Card;

    #[inline]
    fn next(&mut self) -> Option<Card> {
        if self.0 == 0 {
            return None;
        }
        let idx = self.0.trailing_zeros() as u8;
        // Clear lowest set bit.
        self.0 &= self.0 - 1;
        debug_assert!(idx < NUM_CARDS);
        Some(Card::from_index_unchecked(idx))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.0.count_ones() as usize;
        (n, Some(n))
    }
}

impl ExactSizeIterator for HandIter {}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(s: Suit, r: u8) -> Card {
        Card::new(s, r)
    }

    #[test]
    fn add_remove_contains() {
        let mut h = Hand::EMPTY;
        let ace_spades = c(Suit::Spades, 12);
        let two_hearts = c(Suit::Hearts, 0);

        assert!(!h.contains(ace_spades));
        h.add(ace_spades);
        assert!(h.contains(ace_spades));
        assert_eq!(h.count(), 1);

        h.add(two_hearts);
        assert_eq!(h.count(), 2);
        assert!(h.contains(two_hearts));

        h.remove(ace_spades);
        assert!(!h.contains(ace_spades));
        assert!(h.contains(two_hearts));
        assert_eq!(h.count(), 1);
    }

    #[test]
    fn double_add_is_idempotent() {
        let mut h = Hand::EMPTY;
        let k = c(Suit::Clubs, 11);
        h.add(k);
        h.add(k);
        assert_eq!(h.count(), 1);
    }

    #[test]
    fn remove_missing_is_noop() {
        let mut h = Hand::EMPTY;
        h.add(c(Suit::Diamonds, 5));
        h.remove(c(Suit::Spades, 5));
        assert_eq!(h.count(), 1);
    }

    #[test]
    fn iter_visits_each_card_once_ascending() {
        let mut h = Hand::EMPTY;
        let cards = [
            c(Suit::Spades, 0),
            c(Suit::Spades, 12),
            c(Suit::Hearts, 4),
            c(Suit::Clubs, 7),
            c(Suit::Diamonds, 3),
        ];
        for &card in &cards {
            h.add(card);
        }

        let collected: Vec<u8> = h.iter().map(|c| c.index()).collect();
        let mut expected: Vec<u8> = cards.iter().map(|c| c.index()).collect();
        expected.sort_unstable();
        assert_eq!(collected, expected);
        assert_eq!(h.iter().count(), cards.len());
        // Iter is non-destructive.
        assert_eq!(h.count(), cards.len() as u8);
    }

    #[test]
    fn cards_of_suit_extracts_only_that_suit() {
        let mut h = Hand::EMPTY;
        h.add(c(Suit::Spades, 0));
        h.add(c(Suit::Spades, 12));
        h.add(c(Suit::Hearts, 4));
        h.add(c(Suit::Clubs, 7));

        let spades = h.cards_of_suit(Suit::Spades);
        assert_eq!(spades.count_ones(), 2);
        assert_eq!(spades & Suit::Hearts.mask(), 0);
        assert_eq!(spades & Suit::Clubs.mask(), 0);

        let diamonds = h.cards_of_suit(Suit::Diamonds);
        assert_eq!(diamonds, 0);
    }

    #[test]
    fn full_deck_count() {
        let mut h = Hand::EMPTY;
        for i in 0..NUM_CARDS {
            h.add(Card::from_index(i).unwrap());
        }
        assert_eq!(h.count(), 52);
        assert_eq!(h.bits(), (1u64 << 52) - 1);
        assert_eq!(h.iter().count(), 52);
    }
}
