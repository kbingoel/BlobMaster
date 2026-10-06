//! Belief tracking and determinization for imperfect information MCTS.
//!
//! Blob is imperfect information: each player sees only their own hand.
//! MCTS operates on fully-observable states, so we sample plausible
//! opponent hands ("determinizations") and run an independent tree on
//! each, aggregating the root visit counts at the end.
//!
//! Two kinds of evidence shape the sampled deals:
//! - **Voids, as hard constraints.** An opponent who didn't follow the led
//!   suit is provably void in it. Rejection sampling enforces the voids;
//!   when it keeps failing, a constrained deal relaxes only the seats whose
//!   voids can't all be met (none, for a real game state).
//! - **Bids, as weights** ([`sample_deals`], gen-2.md §6 Phase 4b). A deal
//!   that explains the bids already made is likelier than one that
//!   doesn't. Candidate deals are weighted by how likely each earlier bid
//!   was under the policy net, from that bidder's view of the candidate
//!   when it bid, and the kept deals are resampled from them.

use rand::seq::SliceRandom;
use rand::Rng;
use smallvec::SmallVec;

use crate::bidding::{bid_order_position, has_bid, legal_bids};
use crate::card::{NUM_RANKS, NUM_SUITS};
use crate::evaluator::{policy_in_chunks, PolicyEvaluator};
use crate::state::{BlobState, GamePhase, TrickRecord, MAX_PLAYERS};

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

/// The round as `seat` saw it when it bid: every seat holds its starting
/// hand (its cards now plus those it has played this round), the seats
/// before `seat` in bidding order have their bids, and no card is played.
///
/// On a real state this is the state `seat` bid from; on a sampled deal,
/// the one it would have bid from with those cards. `seat` must have bid
/// already (or be about to).
pub fn rewind_to_bid(state: &BlobState, seat: u8) -> BlobState {
    let n = state.num_players;
    let mut s = *state;
    for rec in &state.trick_history[..state.tricks_completed as usize] {
        for &(p, c) in &rec.cards[..rec.num_played as usize] {
            s.hands[p as usize] |= 1u64 << c;
        }
    }
    for i in 0..state.trick_cards_played {
        let p = (state.trick_leader + i) % n;
        s.hands[p as usize] |= 1u64 << state.trick_play_order[i as usize];
    }
    s.tricks_won = [0; MAX_PLAYERS];
    s.played_this_round = 0;
    s.trick_history = [TrickRecord::default(); crate::card::MAX_CARDS_DEALT];
    s.trick_play_order = [0; MAX_PLAYERS];
    s.trick_cards_played = 0;
    s.tricks_completed = 0;
    s.trick_leader = (s.dealer + 1) % n;
    s.game_phase = GamePhase::Bidding as u8;
    s.current_player = seat;
    let pos = bid_order_position(&s, seat);
    for p in 0..n {
        if bid_order_position(&s, p) >= pos {
            s.bids[p as usize] = 0;
        }
    }
    s
}

/// How the bids already made weight the sampled deals (gen-2.md §6
/// Phase 4b). Unknown keys are an error.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BidWeighting {
    /// Candidate deals drawn per deal kept. 0 or 1 turns weighting off:
    /// every kept deal is a uniform consistent one.
    pub candidates: u32,
    /// The bid model's noise floor: a seat bids from P's policy with
    /// probability `1 − noise` and uniformly over its legal bids otherwise.
    /// It keeps a bid P finds unlikely from ruling a deal out, and stands
    /// for players who bid unlike P. Also used by exact 1-card bids.
    pub noise: f32,
}

/// Default candidates per kept deal.
pub const DEFAULT_BID_CANDIDATES: u32 = 8;
/// Default noise floor of the bid model.
pub const DEFAULT_BID_NOISE: f32 = 0.1;

impl Default for BidWeighting {
    fn default() -> Self {
        Self { candidates: DEFAULT_BID_CANDIDATES, noise: DEFAULT_BID_NOISE }
    }
}

impl BidWeighting {
    /// No weighting: uniform consistent deals, as before Phase 4b.
    pub const OFF: Self = Self { candidates: 0, noise: DEFAULT_BID_NOISE };

    /// Whether candidate deals are weighted at all.
    pub fn is_on(&self) -> bool {
        self.candidates > 1
    }

    /// The bid model's probability of a bid that P gives `p`, for a seat
    /// with `legal` legal bids.
    #[inline]
    pub fn likelihood(&self, p: f32, legal: u32) -> f32 {
        (1.0 - self.noise) * p + self.noise / legal.max(1) as f32
    }
}

impl std::fmt::Display for BidWeighting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_on() {
            write!(f, "{}x candidates, noise {}", self.candidates, self.noise)
        } else {
            write!(f, "off")
        }
    }
}

/// `n` deals of the hidden cards for `perspective`, consistent with the
/// known voids and weighted by the bids already made.
///
/// With weighting on and at least one opponent's bid made, this draws
/// `n × weighting.candidates` uniform consistent deals ([`determinize`]),
/// scores each by the product over those bidders of
/// [`BidWeighting::likelihood`] of the actual bid, P's policy read from the
/// bidder's view of the candidate when it bid ([`rewind_to_bid`]), and
/// keeps `n` by systematic resampling. A deal may be kept more than once.
/// Otherwise it is `n` uniform consistent deals.
pub fn sample_deals<P, R>(
    state: &BlobState,
    perspective: u8,
    policy: &P,
    n: usize,
    weighting: BidWeighting,
    rng: &mut R,
) -> Vec<BlobState>
where
    P: PolicyEvaluator + ?Sized,
    R: Rng + ?Sized,
{
    let voids = void_suits(state);
    let bidders: SmallVec<[u8; MAX_PLAYERS]> =
        (0..state.num_players).filter(|&p| p != perspective && has_bid(state, p)).collect();
    let draw = |rng: &mut R| determinize(state, perspective, &voids, rng, DEFAULT_DETERMINIZE_ATTEMPTS);
    if !weighting.is_on() || bidders.is_empty() || n == 0 {
        return (0..n).map(|_| draw(rng)).collect();
    }

    let candidates: Vec<BlobState> = (0..n * weighting.candidates as usize).map(|_| draw(rng)).collect();
    let log_w = bid_log_weights(state, &bidders, &candidates, policy, weighting);
    systematic_resample(&log_w, n, rng).into_iter().map(|i| candidates[i]).collect()
}

/// Log weight of each candidate deal: the sum over `bidders` of the log
/// [`BidWeighting::likelihood`] of the bid each made in `state`, P's policy
/// read from its view of the candidate when it bid.
pub fn bid_log_weights<P: PolicyEvaluator + ?Sized>(
    state: &BlobState,
    bidders: &[u8],
    candidates: &[BlobState],
    policy: &P,
    weighting: BidWeighting,
) -> Vec<f64> {
    crate::profiling::time(&crate::profiling::BID_WEIGHTS, || {
        let views: Vec<BlobState> =
            candidates.iter().flat_map(|c| bidders.iter().map(|&j| rewind_to_bid(c, j))).collect();
        let mut log_w = vec![0.0f64; candidates.len()];
        for (i, (view, pol)) in views.iter().zip(policy_in_chunks(policy, &views)).enumerate() {
            let j = view.current_player as usize;
            let l = weighting.likelihood(pol[state.bids[j] as usize], legal_bids(view).count_ones());
            log_w[i / bidders.len()] += (l.max(f32::MIN_POSITIVE) as f64).ln();
        }
        log_w
    })
}

/// `n` indices drawn in proportion to `exp(log_w)` by systematic
/// resampling: one uniform offset, then evenly spaced points along the
/// cumulative weights. Index `i` appears `⌊n·w_i⌋` or `⌈n·w_i⌉` times
/// (normalized weights), the lowest-variance unbiased scheme.
fn systematic_resample<R: Rng + ?Sized>(log_w: &[f64], n: usize, rng: &mut R) -> Vec<usize> {
    let max = log_w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = log_w.iter().map(|&l| (l - max).exp()).collect();
    let step = w.iter().sum::<f64>() / n as f64;
    let mut point = rng.gen::<f64>() * step;
    let (mut i, mut cum) = (0usize, w[0]);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        while cum < point && i + 1 < w.len() {
            i += 1;
            cum += w[i];
        }
        out.push(i);
        point += step;
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

    /// Rewinding any later state of a round to a seat's bid gives exactly
    /// the state that seat bid from.
    #[test]
    fn rewind_to_bid_recovers_each_bidders_state() {
        use crate::dealing::{new_round, RoundParams};
        use crate::rule_bot::rule_bot_action;
        for seed in 0..20u64 {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
            let params = RoundParams { num_players: 3 + (seed % 4) as u8, cards_dealt: 1 + (seed % 7) as u8, trump: (seed % 5) as u8, dealer: (seed % 3) as u8 };
            let mut s = new_round(params, &mut rng).unwrap();
            let mut at_bid = Vec::new();
            while s.phase() != GamePhase::Scoring {
                if s.phase() == GamePhase::Bidding {
                    at_bid.push(s);
                }
                let a = rule_bot_action(&s);
                crate::mcts::apply_action(&mut s, a);
                if s.phase() == GamePhase::Playing {
                    for b in &at_bid {
                        assert_eq!(rewind_to_bid(&s, b.current_player), *b, "seed {seed}");
                    }
                }
            }
        }
    }

    #[test]
    fn systematic_resample_keeps_each_index_in_proportion() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(11);
        let log_w: Vec<f64> = [0.5f64, 0.25, 0.25, 0.0].iter().map(|w| w.ln()).collect();
        for _ in 0..50 {
            let picks = systematic_resample(&log_w, 8, &mut rng);
            let count = |i| picks.iter().filter(|&&p| p == i).count();
            assert_eq!((count(0), count(1), count(2), count(3)), (4, 2, 2, 0), "{picks:?}");
        }
        let flat = systematic_resample(&[0.0; 40], 5, &mut rng);
        assert_eq!(flat.len(), 5);
        assert!(flat.windows(2).all(|w| w[1] - w[0] == 8), "one per block of 8: {flat:?}");
    }

    /// A policy that bids 1 exactly with the ♠A: after such a bid every
    /// kept deal gives that seat the ♠A (no noise); without weighting most
    /// don't.
    #[test]
    fn sample_deals_follow_the_bids() {
        use crate::dealing::{new_round, RoundParams};
        use crate::evaluator::uniform_policy;
        struct AceBidder;
        impl PolicyEvaluator for AceBidder {
            fn policy(&self, s: &BlobState) -> Vec<f32> {
                let mut p = uniform_policy(s);
                let one = s.hands[s.current_player as usize] >> 12 & 1 == 1;
                let target = if one { 1 } else { 0 };
                if legal_bids(s) >> target & 1 == 1 {
                    p.iter_mut().enumerate().for_each(|(b, x)| *x = if b == target { 1.0 } else { 0.0 });
                }
                p
            }
        }
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(12);
        let params = RoundParams { num_players: 4, cards_dealt: 3, trump: 1, dealer: 3 };
        let mut s = new_round(params, &mut rng).unwrap();
        let ace = 1u64 << 12;
        let hand = |cards: [u8; 3]| cards.iter().fold(0u64, |h, &c| h | 1u64 << c);
        s.hands[..4].copy_from_slice(&[hand([12, 13, 26]), hand([14, 27, 40]), hand([15, 28, 41]), hand([16, 29, 42])]);
        crate::bidding::apply_bid(&mut s, 1); // seat 0
        let sharp = BidWeighting { candidates: 16, noise: 0.0 };
        let deals = sample_deals(&s, 1, &AceBidder, 40, sharp, &mut rng);
        assert_eq!(deals.len(), 40);
        assert!(deals.iter().all(|d| d.hands[0] & ace != 0), "every deal explains the bid");
        assert!(deals.iter().all(|d| d.hands[1] == s.hands[1]), "my hand is kept");
        let uniform = sample_deals(&s, 1, &AceBidder, 40, BidWeighting::OFF, &mut rng);
        assert!(uniform.iter().filter(|d| d.hands[0] & ace != 0).count() < 20);
    }

    /// Effective sample size of the bid weights with a real P, at every
    /// decision of a few rounds played by rule bot 2. Needs `BLOB_MODEL_DIR`;
    /// run with `--ignored --nocapture` in release.
    #[test]
    #[ignore]
    fn bid_weight_ess_with_a_model() {
        use crate::dealing::{new_round, RoundParams};
        use crate::rule_bot_2::rule_bot_2_action;
        let Ok(dir) = std::env::var("BLOB_MODEL_DIR") else { return };
        let p = crate::onnx::OnnxPolicy::from_dir(&dir).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(21);
        let k = 80usize;
        for noise in [0.05f32, 0.1, 0.2] {
            let w = BidWeighting { candidates: 8, noise };
            let (mut bid_ess, mut play_ess) = (Vec::new(), Vec::new());
            for r in 0..6u8 {
                let params = RoundParams { num_players: 5, cards_dealt: 7, trump: r % 5, dealer: r % 5 };
                let mut s = new_round(params, &mut rng).unwrap();
                while s.phase() != GamePhase::Scoring {
                    let me = s.current_player;
                    let bidders: Vec<u8> = (0..5).filter(|&j| j != me && has_bid(&s, j)).collect();
                    if !bidders.is_empty() {
                        let voids = void_suits(&s);
                        let cands: Vec<BlobState> =
                            (0..k).map(|_| determinize(&s, me, &voids, &mut rng, DEFAULT_DETERMINIZE_ATTEMPTS)).collect();
                        let lw = bid_log_weights(&s, &bidders, &cands, &p, w);
                        let max = lw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let ws: Vec<f64> = lw.iter().map(|l| (l - max).exp()).collect();
                        let ess = ws.iter().sum::<f64>().powi(2) / ws.iter().map(|x| x * x).sum::<f64>();
                        if s.phase() == GamePhase::Bidding { bid_ess.push(ess) } else { play_ess.push(ess) }
                    }
                    let a = rule_bot_2_action(&s);
                    crate::mcts::apply_action(&mut s, a);
                }
            }
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let low = |v: &[f64]| { let mut v = v.to_vec(); v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 10] };
            println!(
                "noise {noise}: ESS of {k} candidates — bids mean {:.1} (10th pct {:.1}), plays mean {:.1} (10th pct {:.1})",
                mean(&bid_ess), low(&bid_ess), mean(&play_ess), low(&play_ess)
            );
        }
    }
}
