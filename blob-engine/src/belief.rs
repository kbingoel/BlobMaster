//! Session 4.3 — Belief tracking and determinization for imperfect
//! information MCTS.
//!
//! Blob is imperfect information: each player sees only their own hand.
//! MCTS operates on fully-observable states, so we sample plausible
//! opponent hands ("determinizations") and run an independent tree on
//! each, aggregating the root visit counts at the end (Section 4.3 of
//! the development plan).
//!
//! Belief information is conservative: the only public inference rule
//! used is the suit-void signal (an opponent who didn't follow the led
//! suit is provably void in that suit). Rejection sampling then
//! enforces void constraints when dealing opponent hands; when it keeps
//! failing, a constrained deal relaxes only the seats whose voids can't
//! all be met (none, for a real game state).

use rand::seq::SliceRandom;
use rand::Rng;
use smallvec::SmallVec;

use crate::card::{NUM_RANKS, NUM_SUITS};
use crate::state::{BlobState, MAX_PLAYERS};

/// Cap on rejection-sampling retries before falling back to a constrained
/// sequential deal. Rejection sampling is exactly uniform over consistent
/// deals; the fallback only approximates that, so it is reserved for tight
/// constraints (late in a round, several voids).
pub const DEFAULT_DETERMINIZE_ATTEMPTS: u32 = 32;

/// Per-player boolean flags: `void_suits[p][s] == true` iff seat `p`
/// has been observed to be void in suit `s`.
pub type VoidTable = [[bool; NUM_SUITS as usize]; MAX_PLAYERS];

/// Derive each player's void suits from this round's play so far.
///
/// Rule: a card that is not of the led suit, played after the lead of a
/// completed or in-progress trick, proves its player void in the led suit
/// (they would have been forced to follow otherwise). The lead itself
/// reveals nothing. The encoder's void flags read this table, so sampled
/// deals and network inputs agree.
pub fn void_suits(state: &BlobState) -> VoidTable {
    let mut voids: VoidTable = [[false; NUM_SUITS as usize]; MAX_PLAYERS];
    for t in 0..state.tricks_completed as usize {
        let rec = &state.trick_history[t];
        let led = rec.suit_led as usize;
        for i in 1..rec.num_played as usize {
            let (player, card) = rec.cards[i];
            let card_suit = (card / NUM_RANKS) as usize;
            if card_suit != led {
                voids[player as usize][led] = true;
            }
        }
    }
    if state.trick_cards_played > 1 {
        let led = (state.trick_play_order[0] / NUM_RANKS) as usize;
        for i in 1..state.trick_cards_played {
            let card_suit = (state.trick_play_order[i as usize] / NUM_RANKS) as usize;
            if card_suit != led {
                let player = (state.trick_leader + i) % state.num_players;
                voids[player as usize][led] = true;
            }
        }
    }
    voids
}

/// Opponent hand sizes for a determinization: `cards_dealt - tricks_completed`
/// minus 1 for each opponent who has already contributed to the in-progress
/// trick. The perspective seat's slot is set to zero (they keep their real
/// hand untouched by `determinize`).
fn required_hand_sizes(state: &BlobState, perspective: u8) -> [usize; MAX_PLAYERS] {
    let mut out = [0usize; MAX_PLAYERS];
    let base = state.cards_dealt as usize - state.tricks_completed as usize;
    for p in 0..state.num_players as usize {
        if p as u8 == perspective {
            continue;
        }
        let mut contributed = 0usize;
        for j in 0..state.trick_cards_played as usize {
            let player = (state.trick_leader + j as u8) % state.num_players;
            if player as usize == p {
                contributed = 1;
                break;
            }
        }
        out[p] = base.saturating_sub(contributed);
    }
    out
}

/// Sample a determinized `BlobState`: opponents' hands are replaced with
/// a uniformly-random consistent deal. `perspective`'s hand,
/// `played_this_round`, trick history, and all other fields are
/// preserved — this is a cloned state with reshuffled *unseen* cards.
///
/// Void constraints are enforced by rejection sampling. After
/// `max_attempts` failures it falls back to [`constrained_deal`], which
/// keeps every void it can and relaxes only seats whose voids can't all
/// be met together — for a real game state that is none, since the true
/// deal satisfies every observed void.
pub fn determinize<R: Rng + ?Sized>(
    state: &BlobState,
    perspective: u8,
    voids: &VoidTable,
    rng: &mut R,
    max_attempts: u32,
) -> BlobState {
    crate::profiling::time(&crate::profiling::DETERMINIZE, || {
        let mut out = *state;
        let num_players = state.num_players as usize;
        let my_hand = state.hands[perspective as usize];

        // Unseen cards = deck − perspective's hand − cards that have hit the
        // table this round (completed + in-progress). `played_this_round` is
        // maintained incrementally by `apply_play`, so this is exact.
        let deck_mask: u64 = (1u64 << 52) - 1;
        let unseen_mask = deck_mask & !my_hand & !state.played_this_round;
        let unseen_cards: Vec<u8> = (0..52u8).filter(|&c| (unseen_mask >> c) & 1 == 1).collect();

        let required = required_hand_sizes(state, perspective);
        let total_required: usize = required.iter().sum();
        debug_assert!(
            unseen_cards.len() >= total_required,
            "unseen pool ({}) smaller than total required opponent hand size ({total_required})",
            unseen_cards.len()
        );
        // The unseen pool may be larger than the opponents' combined hand
        // size when `num_players * cards_dealt < 52` (undealt cards remain
        // "unseen" from perspective's point of view). We shuffle the whole
        // pool and use only the first `total_required` slots each attempt;
        // the tail represents cards that weren't dealt to anyone.

        // Most-constrained-first ordering: opponents with more voids get
        // first pick to reduce the rejection probability.
        let mut order: Vec<usize> =
            (0..num_players).filter(|&p| p as u8 != perspective).collect();
        order.sort_by_key(|&p| {
            let voided = voids[p].iter().filter(|v| **v).count();
            std::cmp::Reverse(voided)
        });

        let mut deck = unseen_cards.clone();
        for _ in 0..max_attempts.max(1) {
            deck.shuffle(rng);
            if let Some(new_hands) = try_deal(&deck, &order, &required, voids) {
                out.hands = new_hands_merged(state.hands, &new_hands, perspective);
                return out;
            }
        }

        deck.shuffle(rng);
        let (fallback, _relaxed) = constrained_deal(&deck, &order, &required, voids);
        out.hands = new_hands_merged(state.hands, &fallback, perspective);
        out
    })
}

/// Suit bitmask (bit `s` = suit `s`) a seat may still hold.
fn allowed_suits(voids: &[bool; NUM_SUITS as usize]) -> u8 {
    (0..NUM_SUITS).filter(|&s| !voids[s as usize]).fold(0, |m, s| m | (1 << s))
}

/// Whether the seats in `order` can each be dealt `need[p]` cards of their
/// `allowed[p]` suits from a pool holding `pool[s]` cards of suit `s`.
/// Hall's condition: for every suit set `U`, the seats confined to `U`
/// need no more cards than `U` holds.
fn feasible(
    pool: &[usize; NUM_SUITS as usize],
    order: &[usize],
    need: &[usize; MAX_PLAYERS],
    allowed: &[u8; MAX_PLAYERS],
) -> bool {
    (0u8..1 << NUM_SUITS).all(|u| {
        let demand: usize =
            order.iter().filter(|&&p| allowed[p] & !u == 0).map(|&p| need[p]).sum();
        let supply: usize =
            (0..NUM_SUITS as usize).filter(|&s| u >> s & 1 == 1).map(|s| pool[s]).sum();
        demand <= supply
    })
}

/// Deal `required[p]` cards from the shuffled `deck` to each seat in
/// `order`, honouring as many voids as can be met together. Returns the
/// hands and the seats whose voids were dropped.
///
/// If the voids can't all be met, seats are relaxed one at a time (the
/// first in `order` whose relaxation restores feasibility, else the first
/// still constrained) until they can. Seats then draw in `order`, each
/// card being the next one in `deck` whose suit the seat may hold and that
/// leaves the remaining deal feasible. Close to uniform, not exact: that
/// is what rejection sampling is for.
fn constrained_deal(
    deck: &[u8],
    order: &[usize],
    required: &[usize; MAX_PLAYERS],
    voids: &VoidTable,
) -> ([u64; MAX_PLAYERS], SmallVec<[usize; MAX_PLAYERS]>) {
    let mut allowed = [0u8; MAX_PLAYERS];
    for &p in order {
        allowed[p] = allowed_suits(&voids[p]);
    }
    let mut pool = [0usize; NUM_SUITS as usize];
    for &c in deck {
        pool[(c / NUM_RANKS) as usize] += 1;
    }
    let mut need = *required;

    let all_suits: u8 = (1 << NUM_SUITS) - 1;
    let mut relaxed: SmallVec<[usize; MAX_PLAYERS]> = SmallVec::new();
    while !feasible(&pool, order, &need, &allowed) {
        let constrained = || order.iter().copied().filter(|&p| allowed[p] != all_suits);
        let fixes = |p: usize| {
            let mut a = allowed;
            a[p] = all_suits;
            feasible(&pool, order, &need, &a)
        };
        let Some(p) = constrained().find(|&p| fixes(p)).or_else(|| constrained().next()) else {
            break; // Only reachable if the deck is short of cards.
        };
        allowed[p] = all_suits;
        relaxed.push(p);
    }

    let mut hands = [0u64; MAX_PLAYERS];
    let mut taken = [false; 52];
    for (i, &p) in order.iter().enumerate() {
        let rest = &order[i..];
        while need[p] > 0 {
            // Suits this seat can draw now without stranding a later seat.
            let mut ok = 0u8;
            for s in 0..NUM_SUITS as usize {
                if allowed[p] >> s & 1 == 0 || pool[s] == 0 {
                    continue;
                }
                pool[s] -= 1;
                need[p] -= 1;
                if feasible(&pool, rest, &need, &allowed) {
                    ok |= 1 << s;
                }
                pool[s] += 1;
                need[p] += 1;
            }
            // `ok` is empty only if the deck itself is short; take any card then.
            let pos = (0..deck.len())
                .find(|&k| !taken[k] && (ok == 0 || ok >> (deck[k] / NUM_RANKS) & 1 == 1))
                .expect("deck holds every required card");
            taken[pos] = true;
            let c = deck[pos];
            pool[(c / NUM_RANKS) as usize] -= 1;
            need[p] -= 1;
            hands[p] |= 1u64 << c;
        }
    }
    (hands, relaxed)
}

fn try_deal(
    deck: &[u8],
    order: &[usize],
    required: &[usize; MAX_PLAYERS],
    voids: &VoidTable,
) -> Option<[u64; MAX_PLAYERS]> {
    let mut hands = [0u64; MAX_PLAYERS];
    let mut cursor = 0usize;
    for &p in order {
        let need = required[p];
        let slice = &deck[cursor..cursor + need];
        cursor += need;
        let mut h: u64 = 0;
        for &c in slice {
            let suit = (c / NUM_RANKS) as usize;
            if voids[p][suit] {
                return None;
            }
            h |= 1u64 << c;
        }
        hands[p] = h;
    }
    Some(hands)
}

fn new_hands_merged(
    original: [u64; MAX_PLAYERS],
    opponents: &[u64; MAX_PLAYERS],
    perspective: u8,
) -> [u64; MAX_PLAYERS] {
    let mut out = [0u64; MAX_PLAYERS];
    for i in 0..MAX_PLAYERS {
        out[i] = if i == perspective as usize {
            original[i]
        } else {
            opponents[i]
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bidding::{apply_bid, legal_bids};
    use crate::dealing::deal;
    use crate::game::new_game;
    use crate::playing::apply_play;
    use crate::state::GamePhase;
    use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256PlusPlus};

    fn mid_playing_state(seed: u64) -> (BlobState, Xoshiro256PlusPlus) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut s = new_game(4, 5).unwrap();
        deal(&mut s, &mut rng);
        while s.game_phase == GamePhase::Bidding as u8 {
            let mask = legal_bids(&s);
            let b = mask.trailing_zeros() as u8;
            apply_bid(&mut s, b);
        }
        (s, rng)
    }

    #[test]
    fn void_suits_detected_when_player_skips_led_suit() {
        // Construct: 3 players, 1 card each. Player 0 leads hearts (suit 2),
        // player 1 discards a club (suit 0) — proves void in hearts.
        let mut s = BlobState::empty();
        s.num_players = 3;
        s.cards_dealt = 1;
        s.tricks_completed = 1;
        s.trick_history[0] = crate::state::TrickRecord {
            // (player, card): card = suit * 13 + rank
            cards: [
                (0, 2 * NUM_RANKS + 5), // P0 leads hearts (suit 2)
                (1, 0 * NUM_RANKS + 3), // P1 plays clubs → void in hearts
                (2, 2 * NUM_RANKS + 7), // P2 follows hearts
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
            ],
            num_played: 3,
            winner: 2,
            suit_led: 2,
        };

        let voids = void_suits(&s);
        assert!(voids[1][2], "player 1 must be void in hearts");
        assert!(!voids[1][0], "player 1 not void in clubs");
        assert!(!voids[2][2], "player 2 followed → no void");
        assert!(!voids[0][2], "player 0 led → no void inferred");
    }

    #[test]
    fn determinize_preserves_perspective_hand_and_card_totals() {
        let (s, mut rng) = mid_playing_state(17);
        let perspective = s.current_player;
        let voids = void_suits(&s);
        let d = determinize(&s, perspective, &voids, &mut rng, DEFAULT_DETERMINIZE_ATTEMPTS);

        // Perspective untouched.
        assert_eq!(d.hands[perspective as usize], s.hands[perspective as usize]);
        // Hand sizes correct.
        let required = required_hand_sizes(&s, perspective);
        for p in 0..s.num_players as usize {
            if p as u8 == perspective {
                continue;
            }
            assert_eq!(d.hands[p].count_ones() as usize, required[p]);
        }
        // No duplicate cards across all hands.
        let mut union: u64 = 0;
        let mut popsum = 0u32;
        for p in 0..s.num_players as usize {
            union |= d.hands[p];
            popsum += d.hands[p].count_ones();
        }
        assert_eq!(union.count_ones(), popsum, "no card appears in two hands");
        // Hands disjoint from played_this_round.
        assert_eq!(union & s.played_this_round, 0);
    }

    #[test]
    fn determinize_respects_void_constraints() {
        let (s, mut rng) = mid_playing_state(23);
        let perspective = s.current_player;
        // Forge a void: opponent `t` is void in suit 0.
        let opp = (perspective + 1) % s.num_players;
        let mut voids = void_suits(&s);
        voids[opp as usize][0] = true;

        let suit0_mask: u64 = 0x1FFFu64; // ranks 0..13 of suit 0

        for _ in 0..50 {
            let d = determinize(&s, perspective, &voids, &mut rng, DEFAULT_DETERMINIZE_ATTEMPTS);
            assert_eq!(
                d.hands[opp as usize] & suit0_mask,
                0,
                "voided player holds no suit-0 card"
            );
        }
    }

    #[test]
    fn determinize_bidding_phase_distributes_full_deck() {
        // In bidding phase played_this_round == 0, so all 52 cards are
        // distributed (num_players * cards_dealt must fit; 4×5 = 20).
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let mut s = new_game(4, 5).unwrap();
        deal(&mut s, &mut rng);
        let perspective = s.current_player;
        let voids = void_suits(&s);

        let d = determinize(&s, perspective, &voids, &mut rng, 8);
        for p in 0..s.num_players as usize {
            assert_eq!(d.hands[p].count_ones(), 5);
        }
    }

    #[test]
    fn determinize_handles_in_progress_trick() {
        // Advance one player's play so the in-progress trick has 1 card.
        let (mut s, mut rng) = mid_playing_state(29);
        let first_legal = crate::playing::legal_plays(&s).trailing_zeros() as u8;
        apply_play(&mut s, first_legal);
        assert_eq!(s.trick_cards_played, 1);

        let perspective = s.current_player;
        let voids = void_suits(&s);
        let d = determinize(&s, perspective, &voids, &mut rng, DEFAULT_DETERMINIZE_ATTEMPTS);

        // The seat that just played should now have cards_dealt - 1 cards
        // (they've committed one to the in-progress trick).
        let leader = s.trick_leader as usize;
        assert_eq!(
            d.hands[leader].count_ones() as usize,
            s.cards_dealt as usize - s.tricks_completed as usize - 1
        );
    }

    #[test]
    fn void_suits_include_the_trick_in_progress() {
        // 4 players, spades led by P1, P2 follows, P3 discards a club: P3 is
        // void in spades before the trick completes.
        let mut s = BlobState::empty();
        s.num_players = 4;
        s.cards_dealt = 3;
        s.game_phase = GamePhase::Playing as u8;
        s.trick_leader = 1;
        s.trick_play_order[0] = 5; // ♠7
        s.trick_play_order[1] = 9; // ♠J
        s.trick_play_order[2] = 2 * NUM_RANKS + 4; // ♣6
        s.trick_cards_played = 3;
        s.current_player = 0;
        let voids = void_suits(&s);
        assert!(voids[3][0], "P3 discarded on a spade lead");
        assert!(!voids[1][0] && !voids[2][0], "leader and follower are not void");
    }

    fn suit_mask(suit: u8) -> u64 {
        0x1FFFu64 << (suit * NUM_RANKS)
    }

    fn cards_of(suits: &[(u8, u8)]) -> Vec<u8> {
        // (suit, how many) → the lowest `how many` ranks of that suit.
        suits.iter().flat_map(|&(s, n)| (0..n).map(move |r| s * NUM_RANKS + r)).collect()
    }

    #[test]
    fn constrained_deal_meets_voids_that_rejection_rarely_hits() {
        // 3♠ 3♥ 2♣ for seats needing 3/3/2. Seat 1 may hold only spades,
        // seat 3 only clubs, seat 2 no spades: the only consistent deal is
        // ♠→1, ♥→2, ♣→3 (1 in 560 random deals). Seat 2 draws first, so an
        // unguarded draw would often take a club and strand seat 3.
        let mut voids: VoidTable = [[false; 4]; MAX_PLAYERS];
        voids[1] = [false, true, true, true];
        voids[2] = [true, false, false, false];
        voids[3] = [true, true, false, true];
        let mut required = [0usize; MAX_PLAYERS];
        required[1] = 3;
        required[2] = 3;
        required[3] = 2;
        let mut deck = cards_of(&[(0, 3), (1, 3), (2, 2)]);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        for _ in 0..200 {
            deck.shuffle(&mut rng);
            let (hands, relaxed) = constrained_deal(&deck, &[2, 1, 3], &required, &voids);
            assert!(relaxed.is_empty());
            assert_eq!(hands[1], suit_mask(0) & 0b111);
            assert_eq!(hands[2], suit_mask(1) & (0b111 << NUM_RANKS));
            assert_eq!(hands[3], suit_mask(2) & (0b11 << (2 * NUM_RANKS)));
        }
    }

    #[test]
    fn constrained_deal_relaxes_only_the_unsatisfiable_seat() {
        // 2♠ 2♥. Seat 1 may hold only clubs/diamonds (none left): it must be
        // relaxed. Seat 2 (no hearts) still gets exactly the spades.
        let mut voids: VoidTable = [[false; 4]; MAX_PLAYERS];
        voids[1] = [true, true, false, false];
        voids[2] = [false, true, false, false];
        let mut required = [0usize; MAX_PLAYERS];
        required[1] = 2;
        required[2] = 2;
        let mut deck = cards_of(&[(0, 2), (1, 2)]);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(11);
        for _ in 0..50 {
            deck.shuffle(&mut rng);
            let (hands, relaxed) = constrained_deal(&deck, &[1, 2], &required, &voids);
            assert_eq!(relaxed.as_slice(), &[1]);
            assert_eq!(hands[2], 0b11, "seat 2 keeps its void");
            assert_eq!(hands[1], 0b11 << NUM_RANKS);
        }
    }

    #[test]
    fn determinize_fallback_keeps_satisfiable_voids() {
        // 4 players × 13 cards; the perspective holds every diamond, and the
        // (forged) voids leave exactly one consistent deal: ♠→1, ♥→2, ♣→3.
        // Rejection sampling never finds it, so this exercises the fallback,
        // which used to drop every void.
        let mut s = BlobState::empty();
        s.num_players = 4;
        s.cards_dealt = 13;
        s.game_phase = GamePhase::Playing as u8;
        for p in 0..4u8 {
            s.hands[p as usize] = suit_mask(p);
        }
        s.hands[0] = suit_mask(3);
        s.hands[3] = suit_mask(0);
        let mut voids: VoidTable = [[false; 4]; MAX_PLAYERS];
        voids[1] = [false, true, true, true];
        voids[2] = [true, false, true, true];
        voids[3] = [true, true, false, true];
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(3);
        let d = determinize(&s, 0, &voids, &mut rng, DEFAULT_DETERMINIZE_ATTEMPTS);
        assert_eq!(d.hands[0], suit_mask(3));
        assert_eq!(d.hands[1], suit_mask(0));
        assert_eq!(d.hands[2], suit_mask(1));
        assert_eq!(d.hands[3], suit_mask(2));
    }

    #[test]
    fn determinize_respects_real_voids_late_in_rounds() {
        // Random 5p7c rounds: every sampled deal must honour every void the
        // play so far revealed (including the trick in progress).
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(99);
        let mut checked = 0;
        for _ in 0..20 {
            let mut s = new_game(5, 7).unwrap();
            deal(&mut s, &mut rng);
            while s.game_phase == GamePhase::Bidding as u8 {
                let b = legal_bids(&s).trailing_zeros() as u8;
                apply_bid(&mut s, b);
            }
            while s.game_phase == GamePhase::Playing as u8 {
                let me = s.current_player;
                let voids = void_suits(&s);
                let d = determinize(&s, me, &voids, &mut rng, 2);
                for p in 0..5usize {
                    for suit in 0..4u8 {
                        if voids[p][suit as usize] {
                            assert_eq!(d.hands[p] & suit_mask(suit), 0, "seat {p} void in {suit}");
                            checked += 1;
                        }
                    }
                }
                let legal = crate::playing::legal_plays(&s);
                apply_play(&mut s, (63 - legal.leading_zeros()) as u8);
            }
        }
        assert!(checked > 100, "only {checked} void checks ran");
    }
}
