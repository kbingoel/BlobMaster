//! Exact 1-card bids (gen-2.md §6 Phase 4b, §8).
//!
//! In a 1-card round every play is forced, so the bid is each seat's only
//! decision and the deal decides who takes the trick. The bid is computed
//! instead of searched:
//! - **Earlier bidders' cards** are drawn in proportion to how likely each
//!   one's actual bid was with that card under P, with the noise floor of
//!   [`BidWeighting`]. The later bidders' cards are uniform over the rest.
//! - **Later bidders bid from P's policy**, given their card and the bids
//!   before them, mine included. Every line of their bids counts with its
//!   probability; lines below [`MIN_LINE_PROB`] within a sample are dropped.
//! - **The trick plays out**, and each of my legal bids gets its expected
//!   `u` (`scoring.rs`) over the samples and the later bids' lines.
//!
//! No tree and no value net. Search on sampled deals lost to P alone here
//! (gen-2.md §6 Phase 4): its deals ignored the bids, and inside a deal the
//! later bidders bid as if they saw every card.
//!
//! **Cost:** one P call per unseen card per earlier bidder (51 each), plus
//! one per distinct (later bidder, bids before it, card) that the samples
//! reach, all batched.

use std::collections::HashMap;

use rand::Rng;
use smallvec::SmallVec;

use crate::belief::BidWeighting;
use crate::bidding::{bid_order_position, legal_bids};
use crate::card::NUM_RANKS;
use crate::evaluator::{policy_in_chunks, PolicyEvaluator};
use crate::playing::beats;
use crate::scoring::{score_scale, utilities};
use crate::state::{BlobState, GamePhase, MAX_PLAYERS};

/// Default number of sampled deals per 1-card bid.
pub const DEFAULT_ONE_CARD_SAMPLES: usize = 2048;

/// Lines of later bids less likely than this, within one sample, are
/// dropped. The likeliest line is always kept: its probability is at least
/// `0.5^7`.
pub const MIN_LINE_PROB: f64 = 1e-4;

const DECK: u64 = (1u64 << 52) - 1;

/// An exact 1-card bid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneCardBid {
    /// Expected `u` of the seat to bid, for bid 0 and bid 1;
    /// `NEG_INFINITY` for an illegal bid.
    pub values: [f32; 2],
    /// The legal bid with the higher value; ties go to the higher prior.
    pub bid: u8,
}

/// `seat`'s view when it bids in a 1-card round: it holds `card`, the seats
/// before it in bidding order have their entries of `bids`, the rest none.
/// Every other seat holds one placeholder card: P's input reads only its
/// own hand and each seat's hand size, so the real hidden cards never reach
/// it.
fn bid_view(state: &BlobState, seat: u8, card: u8, bids: &[u8; MAX_PLAYERS]) -> BlobState {
    let mut s = *state;
    let placeholder = 1u64 << (if card == 0 { 1 } else { 0 });
    for h in s.hands[..s.num_players as usize].iter_mut() {
        *h = placeholder;
    }
    s.hands[seat as usize] = 1u64 << card;
    s.current_player = seat;
    let pos = bid_order_position(&s, seat);
    for p in 0..s.num_players {
        s.bids[p as usize] = if bid_order_position(&s, p) < pos { bids[p as usize] } else { 0 };
    }
    s
}

/// Card indices in `mask`, ascending.
fn cards(mut mask: u64) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        (mask != 0).then(|| {
            let c = mask.trailing_zeros() as u8;
            mask &= mask - 1;
            c
        })
    })
}

/// The seat whose card takes the trick; `order` is the playing order.
fn trick_winner(held: &[u8; MAX_PLAYERS], order: &[u8], trump: u8) -> u8 {
    let led = held[order[0] as usize] / NUM_RANKS;
    let mut best = order[0];
    for &p in &order[1..] {
        if beats(held[p as usize], held[best as usize], led, trump) {
            best = p;
        }
    }
    best
}

/// One sampled deal: every seat's card, who takes the trick, and the
/// importance weight of the earlier bidders' draws.
struct Sample {
    held: [u8; MAX_PLAYERS],
    winner: u8,
    weight: f64,
}

/// The exact bid of `state.current_player` in a 1-card round (module docs).
/// `prior` is P's bid policy at `state`, for ties; `lambda` is the utility's
/// λ, and `weighting.noise` the bid model's noise floor.
pub fn one_card_bid<P, R>(
    state: &BlobState,
    policy: &P,
    prior: &[f32],
    lambda: f32,
    weighting: BidWeighting,
    samples: usize,
    rng: &mut R,
) -> OneCardBid
where
    P: PolicyEvaluator + ?Sized,
    R: Rng + ?Sized,
{
    debug_assert!(state.phase() == GamePhase::Bidding && state.cards_dealt == 1);
    crate::profiling::time(&crate::profiling::ONE_CARD_BID, || {
        let n = state.num_players;
        let me = state.current_player;
        let mine = state.hands[me as usize].trailing_zeros() as u8;
        let first = (state.dealer + 1) % n;
        let order: SmallVec<[u8; MAX_PLAYERS]> = (0..n).map(|i| (first + i) % n).collect();
        let my_pos = bid_order_position(state, me) as usize;
        let (earlier, later) = (&order[..my_pos], &order[my_pos + 1..]);
        let unseen = DECK & !(1u64 << mine);

        // How likely each earlier bid was, per card its bidder might hold.
        let mut like = vec![[0.0f32; 52]; earlier.len()];
        let views: Vec<BlobState> = earlier
            .iter()
            .flat_map(|&j| cards(unseen).map(move |x| bid_view(state, j, x, &state.bids)))
            .collect();
        for (view, pol) in views.iter().zip(policy_in_chunks(policy, &views)) {
            let j = view.current_player;
            let e = earlier.iter().position(|&s| s == j).expect("an earlier bidder");
            let x = view.hands[j as usize].trailing_zeros() as usize;
            like[e][x] = weighting.likelihood(pol[state.bids[j as usize] as usize], legal_bids(view).count_ones());
        }
        // A bid no card explains (possible only without noise) says nothing.
        for l in like.iter_mut() {
            if cards(unseen).all(|x| l[x as usize] <= 0.0) {
                cards(unseen).for_each(|x| l[x as usize] = 1.0);
            }
        }

        // Sampled deals. Each earlier bidder in turn draws from the cards
        // left in proportion to its likelihood; the weight multiplies the
        // normalizers, which makes the deals exact draws from "uniform deal,
        // weighted by every earlier bid's likelihood".
        let draws: Vec<Sample> = (0..samples.max(1))
            .map(|_| {
                let mut left = unseen;
                let mut held = [0u8; MAX_PLAYERS];
                held[me as usize] = mine;
                let mut weight = 1.0f64;
                for (e, &j) in earlier.iter().enumerate() {
                    let z: f32 = cards(left).map(|x| like[e][x as usize]).sum();
                    let mut t = rng.gen::<f32>() * z;
                    let mut pick = 0;
                    for x in cards(left) {
                        pick = x;
                        t -= like[e][x as usize];
                        if t <= 0.0 {
                            break;
                        }
                    }
                    weight *= z as f64;
                    held[j as usize] = pick;
                    left &= !(1u64 << pick);
                }
                for &k in later {
                    let pick = cards(left).nth(rng.gen_range(0..left.count_ones() as usize)).expect("cards left");
                    held[k as usize] = pick;
                    left &= !(1u64 << pick);
                }
                Sample { held, winner: trick_winner(&held, &order, state.trump_suit), weight }
            })
            .collect();
        let total_weight: f64 = draws.iter().map(|d| d.weight).sum();

        // P(bid 1) of later bidder `t`, keyed by (my bid, t, the later bids
        // before it as bits, its card).
        let mut p_one: HashMap<(u8, usize, u8, u8), f32> = HashMap::new();
        let mut values = [f32::NEG_INFINITY; 2];
        let my_legal = legal_bids(state);
        let scale = score_scale(1);
        for b in (0..2u8).filter(|&b| (my_legal >> b) & 1 == 1) {
            let mut bids = state.bids;
            bids[me as usize] = b;
            // Lines of later bids: (sample, their bids as bits, probability).
            let mut lines: Vec<(usize, u8, f64)> = (0..draws.len()).map(|s| (s, 0, 1.0)).collect();
            for (t, &k) in later.iter().enumerate() {
                let mut missing: Vec<(u8, usize, u8, u8)> = Vec::new();
                for &(s, bits, _) in &lines {
                    let key = (b, t, bits, draws[s].held[k as usize]);
                    if !p_one.contains_key(&key) {
                        p_one.insert(key, f32::NAN);
                        missing.push(key);
                    }
                }
                let views: Vec<BlobState> = missing
                    .iter()
                    .map(|&(_, _, bits, x)| {
                        let mut line_bids = bids;
                        for (i, &l) in later[..t].iter().enumerate() {
                            line_bids[l as usize] = (bits >> i) & 1;
                        }
                        bid_view(state, k, x, &line_bids)
                    })
                    .collect();
                for (key, pol) in missing.into_iter().zip(policy_in_chunks(policy, &views)) {
                    p_one.insert(key, pol[1]);
                }
                let mut next = Vec::with_capacity(lines.len());
                for (s, bits, p) in lines {
                    let one = p_one[&(b, t, bits, draws[s].held[k as usize])] as f64;
                    for (bid, q) in [(0u8, 1.0 - one), (1, one)] {
                        if p * q >= MIN_LINE_PROB {
                            next.push((s, bits | (bid << t), p * q));
                        }
                    }
                }
                lines = next;
            }

            // Expected u over each sample's lines, then over the samples.
            let mut num = vec![0.0f64; draws.len()];
            let mut den = vec![0.0f64; draws.len()];
            for (s, bits, p) in lines {
                let mut line_bids = bids;
                for (i, &l) in later.iter().enumerate() {
                    line_bids[l as usize] = (bits >> i) & 1;
                }
                let mut s_hat = [0.0f32; MAX_PLAYERS];
                for seat in 0..n {
                    let won = (seat == draws[s].winner) as u8;
                    if won == line_bids[seat as usize] {
                        s_hat[seat as usize] = (10 + won) as f32 / scale;
                    }
                }
                num[s] += p * utilities(&s_hat, n, lambda)[me as usize] as f64;
                den[s] += p;
            }
            let value: f64 = draws.iter().enumerate().map(|(s, d)| d.weight * num[s] / den[s]).sum();
            values[b as usize] = (value / total_weight) as f32;
        }

        let prior_of = |b: usize| prior.get(b).copied().unwrap_or(0.0);
        let bid = if values[1] > values[0] + 1e-6 || ((values[1] - values[0]).abs() <= 1e-6 && prior_of(1) > prior_of(0)) {
            1
        } else {
            0
        };
        OneCardBid { values, bid }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bidding::apply_bid;
    use crate::dealing::{new_round, RoundParams};
    use crate::evaluator::{uniform_policy, DummyEvaluator};
    use crate::round::NO_TRUMP;
    use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256PlusPlus};

    fn card(suit: u8, rank: u8) -> u8 {
        suit * NUM_RANKS + rank
    }

    /// A 1-card round of `n` seats with `dealer`, seat `p` holding
    /// `held[p]`, ready for the first bid.
    fn round(n: u8, dealer: u8, trump: u8, held: &[u8]) -> BlobState {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let mut s = new_round(RoundParams { num_players: n, cards_dealt: 1, trump, dealer }, &mut rng).unwrap();
        for (p, &c) in held.iter().enumerate() {
            s.hands[p] = 1u64 << c;
        }
        s
    }

    /// Bids a fixed amount whatever it holds: its bids say nothing.
    struct Fixed(f32);
    impl PolicyEvaluator for Fixed {
        fn policy(&self, s: &BlobState) -> Vec<f32> {
            let mask = legal_bids(s);
            match mask {
                0b11 => vec![1.0 - self.0, self.0],
                _ => uniform_policy(s),
            }
        }
    }

    /// Bids 1 exactly with a trump (♠ is trump in these tests).
    struct TrumpBidder;
    impl PolicyEvaluator for TrumpBidder {
        fn policy(&self, s: &BlobState) -> Vec<f32> {
            let mask = legal_bids(s);
            if mask != 0b11 {
                return uniform_policy(s);
            }
            let c = s.hands[s.current_player as usize].trailing_zeros() as u8;
            if c / NUM_RANKS == 0 { vec![0.0, 1.0] } else { vec![1.0, 0.0] }
        }
    }

    #[test]
    fn view_hides_the_other_cards_and_keeps_hand_sizes() {
        let s = round(4, 3, 0, &[card(0, 12), card(1, 5), card(2, 7), card(3, 9)]);
        let mut bids = [0u8; MAX_PLAYERS];
        bids[0] = 1;
        bids[1] = 1;
        let v = bid_view(&s, 2, card(1, 3), &bids);
        assert_eq!(v.current_player, 2);
        assert_eq!(v.hands[2], 1u64 << card(1, 3));
        for p in [0usize, 1, 3] {
            assert_eq!(v.hands[p].count_ones(), 1);
            assert_eq!(v.hands[p] & (s.hands[p] | s.hands[2]), 0, "a real card leaked");
        }
        assert_eq!(&v.bids[..4], &[1, 1, 0, 0], "only the seats before 2 have bid");
    }

    #[test]
    fn trick_winner_follows_trump_then_led_suit() {
        let mut held = [0u8; MAX_PLAYERS];
        held[..3].copy_from_slice(&[card(1, 5), card(1, 9), card(2, 12)]);
        assert_eq!(trick_winner(&held, &[0, 1, 2], NO_TRUMP), 1, "highest of the led suit");
        assert_eq!(trick_winner(&held, &[0, 1, 2], 2), 2, "the trump");
        assert_eq!(trick_winner(&held, &[2, 0, 1], NO_TRUMP), 2, "the lead holds");
    }

    /// The ace of trumps always takes the trick, so bid 1 is worth exactly
    /// the made score, whatever the others hold.
    #[test]
    fn a_sure_winner_bids_one_with_its_exact_value() {
        let s = round(3, 2, 0, &[card(0, 12), card(1, 0), card(2, 0)]);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let prior = uniform_policy(&s);
        let r = one_card_bid(&s, &DummyEvaluator, &prior, 0.0, BidWeighting::default(), 512, &mut rng);
        assert_eq!(r.bid, 1);
        assert!((r.values[1] - 1.0).abs() < 1e-6, "{:?}", r.values);
        assert_eq!(r.values[0], 0.0);
    }

    /// At λ = 0 bid 1 is worth P(win) and bid 0 is worth (10/11)(1 − P(win)).
    /// Seat 0 leads the ♥2 at no trump into two random cards: it wins iff
    /// neither is one of the 12 higher hearts, (39·38)/(51·50).
    #[test]
    fn values_match_the_win_chance_at_lambda_zero() {
        let s = round(3, 2, NO_TRUMP, &[card(1, 0), card(2, 0), card(3, 0)]);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2);
        let prior = uniform_policy(&s);
        let r = one_card_bid(&s, &DummyEvaluator, &prior, 0.0, BidWeighting::default(), 20_000, &mut rng);
        let win = (39.0 * 38.0) / (51.0 * 50.0);
        assert!((r.values[1] - win).abs() < 0.01, "{:?} vs {win}", r.values);
        assert!((r.values[0] - (1.0 - win) * 10.0 / 11.0).abs() < 0.01, "{:?}", r.values);
        assert_eq!(r.bid, 1, "P(win) {win:.3} > 10/21");
    }

    /// An earlier bid of 1 from a seat that bids 1 exactly with a trump
    /// tells me it holds one: 9 of the 12 unseen spades beat my ♠5, so I
    /// bid 0. A bidder whose bids say nothing leaves my ♠5 a likely winner
    /// (no higher spade in two random cards: (42·41)/(51·50)), so I bid 1.
    #[test]
    fn an_earlier_bid_reveals_the_bidders_card() {
        let mut s = round(3, 2, 0, &[card(0, 4), card(0, 3), card(2, 0)]);
        apply_bid(&mut s, 1); // seat 0
        let prior = uniform_policy(&s);
        let sharp = BidWeighting { noise: 0.0, ..BidWeighting::default() };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(3);
        let informed = one_card_bid(&s, &TrumpBidder, &prior, 0.0, sharp, 8192, &mut rng);
        assert_eq!(informed.bid, 0, "{:?}", informed.values);
        assert!(informed.values[1] < 0.25, "seat 0 holds a spade: {:?}", informed.values);
        let blind = one_card_bid(&s, &Fixed(0.5), &prior, 0.0, sharp, 8192, &mut rng);
        assert_eq!(blind.bid, 1, "{:?}", blind.values);
        let win = (42.0 * 41.0) / (51.0 * 50.0);
        assert!((blind.values[1] - win).abs() < 0.02, "{:?} vs {win}", blind.values);

        // The noise floor keeps the cards that contradict the bid possible.
        let noisy = BidWeighting { noise: 0.5, ..BidWeighting::default() };
        let r = one_card_bid(&s, &TrumpBidder, &prior, 0.0, noisy, 8192, &mut rng);
        assert!(r.values[1] > informed.values[1] + 0.1, "{:?} vs {:?}", r.values, informed.values);
    }

    /// The dealer's forbidden bid gets no value.
    #[test]
    fn an_illegal_bid_is_never_chosen() {
        let mut s = round(3, 2, 0, &[card(1, 0), card(2, 0), card(0, 12)]);
        apply_bid(&mut s, 0);
        apply_bid(&mut s, 0); // the dealer, seat 2, may not bid 1
        assert_eq!(s.current_player, 2);
        let prior = uniform_policy(&s);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(5);
        let r = one_card_bid(&s, &DummyEvaluator, &prior, 1.0, BidWeighting::default(), 256, &mut rng);
        assert_eq!(r.bid, 0);
        assert_eq!(r.values[1], f32::NEG_INFINITY);
    }

    /// At λ = 1 the later bidders' scores count against mine, and my bid
    /// moves the dealer's. I lead the ♠A (trumps): I take the trick. Seat 1
    /// bids 0/1 at 50/50, the dealer too unless the rule forces it.
    /// - Bid 1: seat 1 expects 5/11; the dealer must bid 1 after (1, 0) and
    ///   expects 0, else 5/11. u = 1 − (5/11 + 2.5/11) / 2.
    /// - Bid 0: seat 1 expects 5/11; the dealer must bid 0 after (0, 0)
    ///   (10/11) and 1 after (0, 1) (0). u = 0 − (5/11 + 5/11) / 2.
    #[test]
    fn later_bidders_lines_follow_their_policy_and_the_dealer_rule() {
        let s = round(3, 2, 0, &[card(0, 12), card(1, 0), card(2, 0)]);
        let prior = uniform_policy(&s);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(6);
        let r = one_card_bid(&s, &Fixed(0.5), &prior, 1.0, BidWeighting::default(), 64, &mut rng);
        assert!((r.values[1] - (1.0 - 3.75 / 11.0)).abs() < 1e-5, "{:?}", r.values);
        assert!((r.values[0] + 5.0 / 11.0).abs() < 1e-5, "{:?}", r.values);
        assert_eq!(r.bid, 1);
    }

    /// Cost of one exact bid with a real P, from every bidding position at
    /// 5 players. Needs `BLOB_MODEL_DIR`; run with `--ignored --nocapture`
    /// in release.
    #[test]
    #[ignore]
    fn exact_bid_cost_with_a_model() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let Ok(dir) = std::env::var("BLOB_MODEL_DIR") else { return };
        let p = crate::onnx::OnnxPolicy::from_dir(&dir).unwrap();
        struct Counted<'a>(&'a crate::onnx::OnnxPolicy, AtomicUsize, AtomicUsize);
        impl PolicyEvaluator for Counted<'_> {
            fn policy(&self, s: &BlobState) -> Vec<f32> {
                self.1.fetch_add(1, Ordering::Relaxed);
                self.0.policy(s)
            }
            fn policy_batch(&self, states: &[&BlobState]) -> Vec<Vec<f32>> {
                self.1.fetch_add(states.len(), Ordering::Relaxed);
                self.2.fetch_add(1, Ordering::Relaxed);
                self.0.policy_batch(states)
            }
        }
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        for samples in [256usize, 2048] {
            for pos in 0..5 {
                let mut s = new_round(RoundParams { num_players: 5, cards_dealt: 1, trump: 1, dealer: 4 }, &mut rng).unwrap();
                for _ in 0..pos {
                    let prior = p.policy(&s);
                    let b = if prior[1] > prior[0] && legal_bids(&s) >> 1 & 1 == 1 { 1 } else { 0 };
                    let b = if legal_bids(&s) >> b & 1 == 1 { b } else { 1 - b };
                    apply_bid(&mut s, b);
                }
                let c = Counted(&p, AtomicUsize::new(0), AtomicUsize::new(0));
                let prior = p.policy(&s);
                let t = std::time::Instant::now();
                let r = one_card_bid(&s, &c, &prior, 1.0, BidWeighting::default(), samples, &mut rng);
                println!(
                    "samples {samples:5} position {pos}: {:7.1} ms, {:5} P states in {:3} calls, values {:?}",
                    t.elapsed().as_secs_f64() * 1e3,
                    c.1.load(Ordering::Relaxed),
                    c.2.load(Ordering::Relaxed),
                    r.values
                );
            }
        }
    }
}
