//! Suit-permutation augmentation (gen-2.md §5.5 item 9).
//!
//! Relabelling the four suits consistently, trump included, turns a position
//! into another position of the same game: legal moves, trick winners and
//! scores carry over card for card. The replay buffer gives each example it
//! samples one of the 24 relabellings at random, which multiplies the
//! variety the networks see without touching the targets.
//!
//! No-trump stays no-trump. Bid policies are unchanged; play policies are
//! indexed by hand position, which a relabelling reorders
//! ([`hand_position_map`]).

use rand::seq::SliceRandom;
use rand::Rng;
use smallvec::SmallVec;

use crate::card::{NUM_RANKS, NUM_SUITS};
use crate::hand::Hand;
use crate::state::{BlobState, MAX_PLAYERS};

/// Suit `s` becomes suit `perm[s]`.
pub type SuitPerm = [u8; NUM_SUITS as usize];

pub const IDENTITY: SuitPerm = [0, 1, 2, 3];

/// All 24 relabellings, the identity first.
pub fn all_suit_perms() -> Vec<SuitPerm> {
    let mut out = Vec::with_capacity(24);
    for a in 0..4u8 {
        for b in (0..4u8).filter(|&b| b != a) {
            for c in (0..4u8).filter(|&c| c != a && c != b) {
                out.push([a, b, c, 6 - a - b - c]);
            }
        }
    }
    out
}

/// A uniformly random relabelling.
pub fn random_suit_perm<R: Rng + ?Sized>(rng: &mut R) -> SuitPerm {
    let mut p = IDENTITY;
    p.shuffle(rng);
    p
}

pub fn inverse(perm: &SuitPerm) -> SuitPerm {
    let mut inv = IDENTITY;
    for (s, &t) in perm.iter().enumerate() {
        inv[t as usize] = s as u8;
    }
    inv
}

#[inline]
pub fn permute_card(card: u8, perm: &SuitPerm) -> u8 {
    perm[(card / NUM_RANKS) as usize] * NUM_RANKS + card % NUM_RANKS
}

/// A card bitmask with every card relabelled.
#[inline]
pub fn permute_mask(mask: u64, perm: &SuitPerm) -> u64 {
    let mut out = 0u64;
    for (s, &t) in perm.iter().enumerate() {
        let suit = (mask >> (s as u8 * NUM_RANKS)) & 0x1FFF;
        out |= suit << (t * NUM_RANKS);
    }
    out
}

/// `state` with every suit relabelled by `perm`: hands, cards played, the
/// trick in progress, the trick history and trump.
pub fn permute_suits(state: &BlobState, perm: &SuitPerm) -> BlobState {
    let mut out = *state;
    for p in 0..MAX_PLAYERS {
        out.hands[p] = permute_mask(state.hands[p], perm);
    }
    out.played_this_round = permute_mask(state.played_this_round, perm);
    if state.trump_suit < NUM_SUITS {
        out.trump_suit = perm[state.trump_suit as usize];
    }
    for i in 0..state.trick_cards_played as usize {
        out.trick_play_order[i] = permute_card(state.trick_play_order[i], perm);
    }
    for rec in out.trick_history[..state.tricks_completed as usize].iter_mut() {
        for i in 0..rec.num_played as usize {
            rec.cards[i].1 = permute_card(rec.cards[i].1, perm);
        }
        rec.suit_led = perm[rec.suit_led as usize];
    }
    out
}

/// Where each card of `hand` lands after relabelling: entry `i` is the
/// position, in the relabelled hand's `Hand::iter` order, of the card at
/// position `i` today.
pub fn hand_position_map(hand: u64, perm: &SuitPerm) -> SmallVec<[u8; 13]> {
    let permuted = permute_mask(hand, perm);
    Hand::new(hand)
        .iter()
        .map(|c| {
            let below = (1u64 << permute_card(c.index(), perm)) - 1;
            (permuted & below).count_ones() as u8
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bidding::{apply_bid, legal_bids};
    use crate::encoder::hand_card_indices;
    use crate::playing::{apply_play, legal_plays};
    use crate::scoring::round_points;
    use crate::state::GamePhase;
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn nth_bit(mask: u64, k: u32) -> u8 {
        let mut m = mask;
        for _ in 0..k {
            m &= m - 1;
        }
        m.trailing_zeros() as u8
    }

    #[test]
    fn the_24_relabellings_are_distinct_bijections() {
        let all = all_suit_perms();
        assert_eq!(all.len(), 24);
        assert_eq!(all[0], IDENTITY);
        for (i, p) in all.iter().enumerate() {
            let mut sorted = *p;
            sorted.sort();
            assert_eq!(sorted, IDENTITY);
            assert!(all[..i].iter().all(|q| q != p));
            assert_eq!(inverse(&inverse(p)), *p);
            for card in 0..52u8 {
                assert_eq!(permute_card(permute_card(card, p), &inverse(p)), card);
            }
        }
    }

    #[test]
    fn permute_mask_moves_each_card() {
        let perm = [2, 0, 3, 1];
        let mask = (1u64 << 0) | (1 << 18) | (1 << 51);
        let want = (1u64 << permute_card(0, &perm))
            | (1 << permute_card(18, &perm))
            | (1 << permute_card(51, &perm));
        assert_eq!(permute_mask(mask, &perm), want);
        assert_eq!(permute_mask(u64::MAX >> 12, &perm), u64::MAX >> 12);
    }

    /// Exit criterion (gen-2.md §6 Phase 3): a suit permutation leaves legal
    /// moves, trick winners and scores unchanged. Plays random rounds on a
    /// state and its relabelled twin, move for move.
    #[test]
    fn relabelling_keeps_legal_moves_trick_winners_and_scores() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA06);
        let perms = all_suit_perms();
        let mut tricks_checked = 0;
        for game in 0..96 {
            let mut s = crate::dealing::new_round(
                crate::dealing::RoundParams {
                    num_players: 3 + (game % 4) as u8,
                    cards_dealt: 1 + (game % 8) as u8,
                    trump: (game % 5) as u8,
                    dealer: (game % 3) as u8,
                },
                &mut rng,
            )
            .unwrap();
            let perm = perms[game % 24];
            let mut t = permute_suits(&s, &perm);
            loop {
                assert_eq!(t, permute_suits(&s, &perm), "game {game}: twins drifted");
                assert_eq!(t.current_player, s.current_player);
                match s.phase() {
                    GamePhase::Bidding => {
                        assert_eq!(legal_bids(&t), legal_bids(&s));
                        let mask = legal_bids(&s) as u64;
                        let b = nth_bit(mask, rng.gen_range(0..mask.count_ones()));
                        apply_bid(&mut s, b);
                        apply_bid(&mut t, b);
                    }
                    GamePhase::Playing => {
                        let legal = legal_plays(&s);
                        assert_eq!(legal_plays(&t), permute_mask(legal, &perm));
                        let c = nth_bit(legal, rng.gen_range(0..legal.count_ones()));
                        let done = s.tricks_completed;
                        apply_play(&mut s, c);
                        apply_play(&mut t, permute_card(c, &perm));
                        if s.tricks_completed > done {
                            let i = done as usize;
                            assert_eq!(t.trick_history[i].winner, s.trick_history[i].winner);
                            tricks_checked += 1;
                        }
                    }
                    _ => break,
                }
            }
            assert_eq!(t.tricks_won, s.tricks_won);
            assert_eq!(round_points(&t), round_points(&s));
        }
        assert!(tricks_checked > 300, "only {tricks_checked} tricks");
    }

    #[test]
    fn hand_position_map_follows_each_card() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(9);
        for _ in 0..200 {
            let mut s = BlobState::empty();
            s.num_players = 4;
            s.cards_dealt = 9;
            crate::dealing::deal(&mut s, &mut rng);
            let perm = random_suit_perm(&mut rng);
            let before = hand_card_indices(&s, 0);
            let after = hand_card_indices(&permute_suits(&s, &perm), 0);
            for (i, &pos) in hand_position_map(s.hands[0], &perm).iter().enumerate() {
                assert_eq!(after[pos as usize], permute_card(before[i], &perm));
            }
        }
    }
}
