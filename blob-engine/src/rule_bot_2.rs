//! Rule bot 2 — a card-counting, bid-aware rule bot. No search, no RNG,
//! and it reads only what the seat to act can see.
//!
//! Added 2026-10-05 as a stronger fixed opponent and a candidate teacher
//! for the gen-2 warm start (gen-2.md Phase 3). [`crate::rule_bot`] stays
//! frozen: it is the yardstick every gen-1 / gen-2 number is measured
//! against.
//!
//! # What it knows
//!
//! `Knowledge` is rebuilt from the state at every decision:
//! - **Card counting.** Unseen cards = deck − my hand − cards played this
//!   round. Not every unseen card is in play: `52 − players × cards` are
//!   undealt, so "the ace is still out" is a probability, not a fact.
//! - **Voids.** A seat that didn't follow suit has none of it left
//!   ([`crate::belief::void_suits`]).
//! - **Who holds what.** For each opponent and suit, the chance that a
//!   given unseen card sits in that opponent's hand. Hand sizes and voids
//!   are the constraints and the undealt stock takes the rest (iterative
//!   proportional fitting over seats × suits).
//! - **Goals.** A seat that has bid is hungry (short of its bid) or done
//!   (at or past it). A hungry seat takes every trick it can; a done seat
//!   takes one when every card it may play is higher, and otherwise with
//!   weight `WILL_DONE`.
//!
//! # How it decides
//!
//! Both decisions maximise the chance of making the bid exactly.
//! - **Outlook of a card** = (sure, possible): the chance it takes a later
//!   trick if I try to lose with it, and if I try to win with it. Built
//!   from the higher cards still unseen and how likely a willing opponent
//!   holds them, my lower cards of the suit that let me wait, the ruff
//!   risk (opponents short in the suit with trumps left) and the chance
//!   the suit is led often enough. Low trumps get credit for ruffing my
//!   short side suits.
//! - **Exact-count chance.** Each card is a sure winner, flexible, or a
//!   sure loser; I can take exactly `k` more tricks iff
//!   sure ≤ k ≤ sure + flexible. Only `CONTROL` of the flexible share
//!   counts as real control; the rest is a coin flip.
//! - **Bidding**: the legal bid with the best `(10 + b) · P(make b)`.
//!   Earlier bids above their share of the tricks shrink my uncertain
//!   cards, bids below it grow them (`BID_PRESSURE`). 1-card rounds are
//!   computed directly, weighting each earlier bidder's possible cards by
//!   whether they explain that bid.
//! - **Playing**: for each legal card, P(it takes this trick) from the
//!   seats still to play, then P(make) over the rest of the hand. Ties
//!   (including every card of a doomed seat) go to the weaker card.
//!
//! # Measured (`blobmaster bench`, duplicate deals, 2026-10-05)
//!
//! All at 5p/7c; gen 1 = `run-2026-05-14/iter_000167`.
//!
//! | focal vs 4× …                       | games  | pts/game diff | bids made |
//! |-------------------------------------|--------|---------------|-----------|
//! | rule bot 2 vs rule bot              | 20 000 | +14.5 ± 0.3   | 0.725 (opp 0.639) |
//! | rule bot vs rule bot 2              | 20 000 | −10.5 ± 0.3   | 0.630 |
//! | rule bot 2 vs gen-1 network         | 1 280  | +29.7 ± 1.2   | 0.701 |
//! | rule bot vs gen-1 network           | 1 280  | +17.6 ± 1.2   | 0.649 |
//! | gen-1 network vs rule bot 2         | 640    | −32.2 ± 1.6   | 0.525 |
//! | gen-1 5×100 search vs rule bot 2    | 320    | −30.4 ± 2.5   | 0.532 |
//!
//! For scale, gen 1 scores −12.1 (network) and −10.2 (search) against
//! the rule bot. A table of four rule bot 2s makes 0.680 of its bids (four
//! rule bots: 0.638).
//!
//! Against the rule bot at other tables (2000 deals each, CI ±0.3–0.7):
//! 3p/8c +18.9, 4p/7c +13.3, 4p/13c +23.4, 6p/7c +18.9, 7p/7c +25.2,
//! 8p/6c +24.0. Bids made by hand size at 5p/7c: 1 card 0.814, 2–4 cards
//! 0.719, 5–8 cards 0.657 (rule bot: 0.753 / 0.634 / 0.547).
//!
//! # Tuning notes
//!
//! The constants were swept one at a time at 5p/7c against the rule bot,
//! so they fit that opponent best. Bidding and play each carried about
//! half of the gain. The biggest single gains came from two changes: not
//! assuming a side suit gets led (side cards had been overrated), and
//! weighting the higher cards out by whether their holder wants tricks.
//! Bidding higher across the board made things worse, even though
//! positive bids go over more often than under. Spoiling when doomed
//! (taking tricks from hungry seats) measured zero and was dropped.
//!
//! Use [`bid_chances`] / [`play_chances`] for per-action scores (soft
//! teacher targets, `blobmaster play --bot rulebot2 --show`).
//!
//! # v2r: with rollouts
//!
//! [`rule_bot_2r_action`] ("v2r", `blobmaster bench rulebot2r`) tries each
//! legal action on deals sampled from what the seat has seen, plays every
//! seat forward with rule bot 2 and keeps the best mean of "my round score
//! minus the table's mean" ([`Rollouts`]). Measured vs 4× rule bot 2 at
//! 5p/7c, 10 000 games per row (2026-10-05); CPU time is averaged over all
//! moves, forced ones included:
//!
//! | samples | depth | rolls out    | pts/game vs v2 | CPU ms/move |
//! |---------|-------|--------------|----------------|-------------|
//! | 32      | 1     | bids + plays | +0.8 ± 0.3     | 1.6 |
//! | 32      | 2     | bids + plays | +4.8 ± 0.4     | 2.5 |
//! | 32      | 3     | bids + plays | +6.6 ± 0.4     | 3.0 |
//! | 32      | 4–6   | bids + plays | +6.3 … +6.4    | 3.3 |
//! | 32      | full  | bids + plays | +6.7 ± 0.4     | 3.3 |
//! | 32      | full  | plays only   | +3.5 ± 0.2     | 1.3 |
//! | 8       | full  | bids + plays | +0.0 ± 0.4     | 0.8 |
//! | 16      | full  | bids + plays | +3.8 ± 0.4     | 1.6 |
//! | 64      | full  | bids + plays | +8.6 ± 0.4     | 6.6 |
//! | 128     | full  | bids + plays | +9.6 ± 0.3     | 13.1 |
//! | 256     | full  | bids + plays | +10.1 ± 0.5    | 27 (5000 games) |
//! | 512     | full  | bids + plays | +10.3 ± 0.5    | 54 (5000 games) |
//! | 1024    | full  | bids + plays | +10.6 ± 0.7    | 109 (2500 games) |
//! | 128     | 2     | bids + plays | +5.9 ± 0.4     | 10.1 |
//!
//! - **Depth saturates at 3 tricks**, and playing the round out costs no
//!   more: scoring a cut-off position with the static estimate for every
//!   seat costs about as much as playing the last tricks. So the default is
//!   full depth, 128 samples: the knee of the samples curve (+9.6 of the
//!   ~+10.5 plateau, 13 ms per move).
//! - **Samples are the lever, up to ~128.** Below 16 the noise in
//!   comparing actions costs more than the rollouts gain. Each doubling
//!   then adds less (+2.9, +1.9, +1.0), and past 128 only a few tenths:
//!   the plateau is about +10.5, the value of a noise-free one-step
//!   improvement over rule bot 2 with uniform deal sampling.
//! - **Rolled-out bids carry half the gain** but are worse in 1-card
//!   rounds (bids made 0.773 vs 0.783): the sampled deals ignore what
//!   earlier bids reveal, which rule bot 2's 1-card formula uses.
//! - **Best case for rollouts:** the opponents here *are* the rollout
//!   policy. Against other players the gain will be smaller.

use rand::Rng;
use smallvec::{smallvec, SmallVec};

use crate::belief::{determinize, void_suits, DEFAULT_DETERMINIZE_ATTEMPTS};
use crate::bidding::{apply_bid, has_bid, legal_bids};
use crate::card::{NUM_RANKS, NUM_SUITS};
use crate::mcts::apply_action;
use crate::playing::{apply_play, beats, current_trick_winner, legal_plays};
use crate::round::NO_TRUMP;
use crate::state::{BlobState, GamePhase, MAX_PLAYERS};

/// How readily a seat at or past its bid takes a trick it could duck.
const WILL_DONE: f32 = 0.3;
/// How readily a seat that hasn't bid yet takes a trick it can take.
const WILL_UNKNOWN: f32 = 0.7;
/// How much the danger from higher cards out is weighted by their likely
/// holder's will (the rest counts every dealt card), in play and when
/// bidding. Lower when bidding, where most wills are still unknown.
const RIVAL_PLAY: f32 = 0.75;
const RIVAL_BID: f32 = 0.25;
/// Chance a ruff with a low trump is not over-ruffed.
const RUFF_WIN: f32 = 0.8;
/// Chance a side-suit winner can't be shed when I'd rather lose with it.
const STUCK: f32 = 0.5;
/// Share of a card's flexible probability treated as real control.
const CONTROL: f32 = 0.2;
/// How far earlier bids above their share shrink my uncertain cards.
const BID_PRESSURE: f32 = 0.5;
/// 1-card rounds: weight of a card that contradicts its holder's bid.
const BID_NOISE: f32 = 0.15;
/// P(make) differences below this count as ties.
const MAKE_TOL: f32 = 1e-3;
/// Iterations of the seats × suits fit in `Knowledge::new`.
const FIT_ITERS: usize = 25;
/// v2r: weight of the other seats' mean round score in a rollout's value
/// (λ of gen-2.md §5.1; 1 = "my points minus the table's", as `bench`
/// scores it).
const SPITE: f32 = 1.0;

const DECK: u64 = (1u64 << 52) - 1;
const SUITS: usize = NUM_SUITS as usize;
/// Per-card (sure, possible) outlooks.
type Outlooks = SmallVec<[(f32, f32); 13]>;
/// `spread()[s][f]`: chance that `s` cards are sure winners and `f` flexible.
type Spread = [[f32; 14]; 14];

#[inline]
fn suit_of(c: u8) -> u8 {
    c / NUM_RANKS
}

#[inline]
fn rank_of(c: u8) -> u8 {
    c % NUM_RANKS
}

#[inline]
fn suit_mask(suit: u8) -> u64 {
    0x1FFFu64 << (suit * NUM_RANKS)
}

/// Cards of `c`'s suit ranked above `c`.
#[inline]
fn above(c: u8) -> u64 {
    suit_mask(suit_of(c)) & !((2u64 << c) - 1)
}

/// Cards of `c`'s suit ranked below `c`.
#[inline]
fn below(c: u8) -> u64 {
    suit_mask(suit_of(c)) & ((1u64 << c) - 1)
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

/// Ordering key: every trump beats every side-suit card, then rank.
#[inline]
fn strength(card: u8, trump: u8) -> u8 {
    if trump != NO_TRUMP && suit_of(card) == trump {
        100 + rank_of(card)
    } else {
        rank_of(card)
    }
}

/// Σ_{k=0}^{kmax} P(Binomial(n, p) = k) · f(k).
fn binom_sum(n: u32, p: f32, kmax: u32, mut f: impl FnMut(u32) -> f32) -> f32 {
    let p = p.clamp(0.0, 1.0);
    if p == 0.0 {
        return f(0);
    }
    if p == 1.0 {
        return if n <= kmax { f(n) } else { 0.0 };
    }
    let odds = p / (1.0 - p);
    let mut pmf = (1.0 - p).powi(n as i32);
    let mut total = 0.0;
    for k in 0..=kmax.min(n) {
        total += pmf * f(k);
        pmf *= odds * (n - k) as f32 / (k + 1) as f32;
    }
    total
}

/// P(Binomial(n, p) ≥ t).
fn binom_at_least(n: u32, p: f32, t: u32) -> f32 {
    if t == 0 {
        1.0
    } else {
        (1.0 - binom_sum(n, p, t - 1, |_| 1.0)).max(0.0)
    }
}

/// Chance that none of `bad` marked cards is among `draws` cards drawn
/// without replacement from `pool`.
fn none_drawn(pool: u32, bad: u32, draws: u32) -> f32 {
    (0..draws)
        .map(|i| pool.saturating_sub(bad + i) as f32 / pool.saturating_sub(i).max(1) as f32)
        .product()
}

/// Where a seat stands against its bid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Goal {
    /// Hasn't bid yet.
    Unknown,
    /// Short of its bid.
    Hungry,
    /// At or past its bid.
    Done,
}

fn goal(s: &BlobState, seat: usize) -> Goal {
    if !has_bid(s, seat as u8) {
        Goal::Unknown
    } else if s.tricks_won[seat] < s.bids[seat] {
        Goal::Hungry
    } else {
        Goal::Done
    }
}

/// How readily a seat takes a trick it can take but doesn't have to.
fn will(g: Goal) -> f32 {
    match g {
        Goal::Hungry => 1.0,
        Goal::Unknown => WILL_UNKNOWN,
        Goal::Done => WILL_DONE,
    }
}

/// What one seat knows, rebuilt at every decision. Reads only that seat's
/// hand and public information.
struct Knowledge {
    me: usize,
    n: usize,
    trump: u8,
    hand: u64,
    /// Neither in my hand nor played this round: in an opponent's hand or
    /// undealt.
    unseen: u64,
    /// Cards still in each opponent's hand (0 for me).
    held: [u32; MAX_PLAYERS],
    /// `q[j][s]`: chance that a given unseen card of suit `s` is in seat
    /// `j`'s hand (0 for me).
    q: [[f32; SUITS]; MAX_PLAYERS],
    will: [f32; MAX_PLAYERS],
    /// [`RIVAL_PLAY`] or [`RIVAL_BID`].
    rival: f32,
}

impl Knowledge {
    /// What seat `me` knows in `s`.
    fn new(s: &BlobState, me: usize) -> Self {
        let n = s.num_players as usize;
        let hand = s.hands[me];
        let unseen = DECK & !hand & !s.played_this_round;
        let voids = void_suits(s);
        let left = (s.cards_dealt - s.tricks_completed) as u32;
        let mut held = [0u32; MAX_PLAYERS];
        let mut wills = [0f32; MAX_PLAYERS];
        for j in (0..n).filter(|&j| j != me) {
            let slot = (j + n - s.trick_leader as usize) % n;
            held[j] = left - (slot < s.trick_cards_played as usize) as u32;
            wills[j] = will(goal(s, j));
        }

        // Expected unseen cards per (holder, suit), holders = seats plus
        // the undealt stock in row `n`. Rows sum to hand sizes, columns to
        // the unseen cards of each suit, and a void seat holds none.
        let per_suit: [f32; SUITS] =
            std::array::from_fn(|x| (unseen & suit_mask(x as u8)).count_ones() as f32);
        let total: f32 = per_suit.iter().sum();
        let mut cap = [0f32; MAX_PLAYERS + 1];
        for j in 0..n {
            cap[j] = held[j] as f32;
        }
        cap[n] = (total - held.iter().sum::<u32>() as f32).max(0.0);
        let mut m = [[0f32; SUITS]; MAX_PLAYERS + 1];
        for k in 0..=n {
            for x in 0..SUITS {
                let open = k == n || !voids[k][x];
                if open && cap[k] > 0.0 && per_suit[x] > 0.0 {
                    m[k][x] = cap[k] * per_suit[x] / total;
                }
            }
        }
        for _ in 0..FIT_ITERS {
            for k in 0..=n {
                let r: f32 = m[k].iter().sum();
                if r > 0.0 {
                    let f = cap[k] / r;
                    m[k].iter_mut().for_each(|v| *v *= f);
                }
            }
            for x in 0..SUITS {
                let c: f32 = (0..=n).map(|k| m[k][x]).sum();
                if c > 0.0 {
                    let f = per_suit[x] / c;
                    (0..=n).for_each(|k| m[k][x] *= f);
                }
            }
        }
        let mut q = [[0f32; SUITS]; MAX_PLAYERS];
        for j in 0..n {
            for x in 0..SUITS {
                if per_suit[x] > 0.0 {
                    q[j][x] = (m[j][x] / per_suit[x]).min(1.0);
                }
            }
        }
        let rival = if s.phase() == GamePhase::Bidding { RIVAL_BID } else { RIVAL_PLAY };
        Knowledge { me, n, trump: s.trump_suit, hand, unseen, held, q, will: wills, rival }
    }

    #[inline]
    fn trump_on(&self) -> bool {
        self.trump != NO_TRUMP
    }

    /// Opponents with cards left.
    fn opponents(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.n).filter(move |&j| j != self.me && self.held[j] > 0)
    }

    fn unseen_in(&self, suit: u8) -> u32 {
        (self.unseen & suit_mask(suit)).count_ones()
    }

    /// Chance seat `j` holds at least one of the unseen cards in `set`.
    fn holds_any(&self, j: usize, set: u64) -> f32 {
        let set = set & self.unseen;
        let none: f32 = (0..NUM_SUITS)
            .map(|x| (1.0 - self.q[j][x as usize]).powi((set & suit_mask(x)).count_ones() as i32))
            .product();
        1.0 - none
    }

    /// Chance seat `j` has no card of `suit` left.
    fn void_in(&self, j: usize, suit: u8) -> f32 {
        1.0 - self.holds_any(j, suit_mask(suit))
    }

    /// Chance a given unseen card of `suit` is held by an opponent who
    /// will use it against me: a blend of "is dealt to anyone" and "is in
    /// a hand that wants tricks" (a card in a done hand mostly ducks).
    fn rival_share(&self, suit: u8) -> f32 {
        let (dealt, willing) = self.opponents().fold((0.0, 0.0), |(d, w), j| {
            let q = self.q[j][suit as usize];
            (d + q, w + q * self.will[j])
        });
        (self.rival * willing + (1.0 - self.rival) * dealt).min(1.0)
    }

    /// Chance that playing `c` now takes the current trick.
    fn trick_win(&self, s: &BlobState, c: u8) -> f32 {
        let led = if s.trick_cards_played == 0 { suit_of(c) } else { suit_of(s.trick_play_order[0]) };
        if let Some(slot) = current_trick_winner(s) {
            if !beats(c, s.trick_play_order[slot as usize], led, self.trump) {
                return 0.0;
            }
        }
        (s.trick_cards_played as usize + 1..self.n)
            .map(|slot| (s.trick_leader as usize + slot) % self.n)
            .map(|j| {
                let can = self.can_beat(j, c, led);
                let forced = self.forced_beat(j, c, led).min(can);
                1.0 - (forced + self.will[j] * (can - forced))
            })
            .product()
    }

    /// Chance seat `j`, still to play, holds a card that may beat `c`
    /// (the trick's best card) in a trick led in `led`.
    fn can_beat(&self, j: usize, c: u8, led: u8) -> f32 {
        let over = self.holds_any(j, above(c));
        if self.trump_on() && suit_of(c) == self.trump {
            if led == self.trump {
                over
            } else {
                self.void_in(j, led) * over
            }
        } else if self.trump_on() {
            (over + self.void_in(j, led) * self.holds_any(j, suit_mask(self.trump))).min(1.0)
        } else {
            over
        }
    }

    /// Chance seat `j` must beat `c` (the trick's best card) because every
    /// card it holds of the led suit is higher. Zero when `c` ruffed: a
    /// seat that follows suit can't take a ruff, and an over-ruff is a
    /// choice.
    fn forced_beat(&self, j: usize, c: u8, led: u8) -> f32 {
        let in_suit = self.unseen_in(led);
        if suit_of(c) != led || in_suit == 0 {
            return 0.0;
        }
        let q = self.q[j][led as usize];
        let higher = (above(c) & self.unseen).count_ones() as f32 / in_suit as f32;
        // E[higher^M] − P(M = 0) for M ~ Binomial(in_suit, q) cards held.
        ((1.0 - q * (1.0 - higher)).powi(in_suit as i32) - (1.0 - q).powi(in_suit as i32)).max(0.0)
    }

    /// (sure, possible): the chance card `x` of `hand` takes one of the
    /// tricks after this one if I try to lose with it, and if I try to
    /// win with it.
    fn outlook(&self, x: u8, hand: u64) -> (f32, f32) {
        let suit = suit_of(x);
        let mine = hand & suit_mask(suit);
        let higher = (above(x) & self.unseen).count_ones();
        let rivals = self.rival_share(suit);
        let guards = (mine & below(x)).count_ones();
        // No higher card of the suit is held by a rival.
        let top = binom_sum(higher, rivals, 0, |_| 1.0);
        if self.trump_on() && suit == self.trump {
            // Every card gets played, so a trump with no higher one against
            // it wins whenever it goes; with k higher ones out it needs k
            // lower trumps of mine to wait them out.
            return (top, binom_sum(higher, rivals, guards, |_| 1.0));
        }
        let ahead = (mine & above(x)).count_ones();
        // `x` is the best card left on the `round`-th lead of its suit.
        let reach = |round: u32| self.no_ruff(suit, round) * self.led(suit, round, hand);
        let hi = binom_sum(higher, rivals, guards, |k| reach(ahead + k + 1));
        let lo = if hand.count_ones() <= 1 { hi } else { STUCK * top * reach(ahead + 1) };
        (lo.min(hi), hi)
    }

    /// Chance no opponent ruffs the `round`-th lead of `suit` from now.
    fn no_ruff(&self, suit: u8, round: u32) -> f32 {
        if !self.trump_on() || suit == self.trump {
            return 1.0;
        }
        let trumps = suit_mask(self.trump);
        let in_suit = self.unseen_in(suit);
        self.opponents()
            .map(|j| {
                let short = 1.0 - binom_at_least(in_suit, self.q[j][suit as usize], round);
                1.0 - self.will[j] * short * self.holds_any(j, trumps)
            })
            .product()
    }

    /// Chance `suit` is led at least `times` in the tricks after this one,
    /// if each lead picks a suit in proportion to the cards held.
    fn led(&self, suit: u8, times: u32, hand: u64) -> f32 {
        let rounds = hand.count_ones();
        let theirs: f32 = self.opponents().map(|j| self.q[j][suit as usize]).sum();
        let in_suit = (hand & suit_mask(suit)).count_ones() as f32 + theirs * self.unseen_in(suit) as f32;
        let in_all = rounds as f32 + self.opponents().map(|j| self.held[j] as f32).sum::<f32>();
        binom_at_least(rounds, (in_suit / in_all.max(1.0)).min(1.0), times)
    }

    /// Outlooks of every card of `hand` (ascending card index), with my
    /// lowest trumps credited for ruffing short side suits.
    fn hand_outlook(&self, hand: u64) -> Outlooks {
        let mut outs: Outlooks = cards(hand).map(|x| self.outlook(x, hand)).collect();
        if !self.trump_on() {
            return outs;
        }
        let mut ruffs: f32 = (0..NUM_SUITS)
            .filter(|&x| x != self.trump && self.unseen_in(x) > 0)
            .map(|x| match (hand & suit_mask(x)).count_ones() {
                0 => self.led(x, 1, hand),
                1 => 0.5 * self.led(x, 2, hand),
                _ => 0.0,
            })
            .sum();
        for (i, x) in cards(hand).enumerate() {
            if ruffs <= 0.0 {
                break;
            }
            if suit_of(x) == self.trump {
                let used = ruffs.min(1.0);
                outs[i].1 += (1.0 - outs[i].1) * used * RUFF_WIN;
                ruffs -= used;
            }
        }
        outs
    }

    /// Shrink (grow) my uncertain cards when the seats that already bid
    /// claimed more (fewer) tricks than their share of the ones I don't
    /// expect to take.
    fn bid_pressure(&self, s: &BlobState, outs: &mut [(f32, f32)]) {
        let bidders: SmallVec<[usize; MAX_PLAYERS]> =
            (0..self.n).filter(|&j| j != self.me && has_bid(s, j as u8)).collect();
        if bidders.is_empty() {
            return;
        }
        let c = s.cards_dealt as f32;
        let claimed: f32 = bidders.iter().map(|&j| s.bids[j] as f32).sum();
        let mine: f32 = outs.iter().map(|&(lo, hi)| 0.5 * (lo + hi)).sum();
        let share = bidders.len() as f32 * (c - mine).max(0.0) / (self.n - 1) as f32;
        let f = (1.0 - BID_PRESSURE * (claimed - share) / c).clamp(0.5, 1.5);
        for o in outs.iter_mut().filter(|o| o.1 < 0.98) {
            o.1 = (o.1 * f).min(1.0);
            o.0 = (o.0 * f).min(o.1);
        }
    }
}

/// Distribution of (sure winners, flexible cards) over `outs`.
fn spread(outs: &[(f32, f32)]) -> Spread {
    let mut dp: Spread = [[0.0; 14]; 14];
    dp[0][0] = 1.0;
    for (i, &(lo, hi)) in outs.iter().enumerate() {
        let free = (hi - lo).max(0.0);
        let win = lo + 0.5 * (1.0 - CONTROL) * free;
        let flex = CONTROL * free;
        let lose = (1.0 - win - flex).max(0.0);
        // Descending, so every cell is updated before mass is added to it.
        for s in (0..=i).rev() {
            for f in (0..=i - s).rev() {
                let v = dp[s][f];
                if v == 0.0 {
                    continue;
                }
                dp[s][f] = v * lose;
                dp[s + 1][f] += v * win;
                dp[s][f + 1] += v * flex;
            }
        }
    }
    dp
}

/// Chance of being able to take exactly `need` more tricks.
fn exact(dp: &Spread, need: i32) -> f32 {
    if !(0..14).contains(&need) {
        return 0.0;
    }
    let k = need as usize;
    (0..=k).map(|s| dp[s][k - s..14 - s].iter().sum::<f32>()).sum()
}

/// Chance a card wins a 1-card round for a holder who knows nothing else:
/// the other `n − 1` cards are a random draw from the 51 unseen.
fn naive_one_card_win(x: u8, leads: bool, n: usize, trump: u8) -> f32 {
    let x_trump = trump != NO_TRUMP && suit_of(x) == trump;
    let higher = (NUM_RANKS - 1 - rank_of(x)) as u32;
    let beaters = higher + if trump != NO_TRUMP && !x_trump { NUM_RANKS as u32 } else { 0 };
    let others = n as u32 - 1;
    if leads || x_trump {
        none_drawn(51, beaters, others)
    } else {
        // The leader must lead this suit, lower.
        rank_of(x) as f32 / 51.0 * none_drawn(50, beaters, others - 1)
    }
}

/// Chance my only card takes the trick of a 1-card round. Each earlier
/// bidder's possible cards are down-weighted ([`BID_NOISE`]) where a
/// simple player would have bid otherwise with them.
fn one_card_win(s: &BlobState) -> f32 {
    let n = s.num_players as usize;
    let me = s.current_player as usize;
    let mine = s.hands[me].trailing_zeros() as u8;
    let trump = s.trump_suit;
    let leader = (s.dealer as usize + 1) % n;
    let unseen = DECK & !s.hands[me];
    // Chance seat `j`'s card satisfies `pred`.
    let chance = |j: usize, pred: &dyn Fn(u8) -> bool| -> f32 {
        let (mut hit, mut all) = (0.0f32, 0.0f32);
        for x in cards(unseen) {
            let bids_one = naive_one_card_win(x, j == leader, n, trump) > 10.0 / 21.0;
            let w = if has_bid(s, j as u8) && (s.bids[j] == 1) != bids_one { BID_NOISE } else { 1.0 };
            all += w;
            if pred(x) {
                hit += w;
            }
        }
        hit / all
    };
    let beats_mine = |x: u8| beats(x, mine, suit_of(mine), trump);
    let others = (0..n).filter(|&j| j != me);
    if me == leader || (trump != NO_TRUMP && suit_of(mine) == trump) {
        others.map(|j| 1.0 - chance(j, &beats_mine)).product()
    } else {
        let led_under = |x: u8| suit_of(x) == suit_of(mine) && x < mine;
        chance(leader, &led_under)
            * others.filter(|&j| j != leader).map(|j| 1.0 - chance(j, &beats_mine)).product::<f32>()
    }
}

/// Chance of making each bid `0..=cards_dealt` for the seat to bid. The
/// dealer's forbidden bid is included; [`rule_bot_2_bid`] masks it.
pub fn bid_chances(state: &BlobState) -> SmallVec<[f32; 14]> {
    debug_assert_eq!(state.phase(), GamePhase::Bidding);
    if state.cards_dealt == 1 {
        let p = one_card_win(state);
        return smallvec![1.0 - p, p];
    }
    let kn = Knowledge::new(state, state.current_player as usize);
    let mut outs = kn.hand_outlook(kn.hand);
    kn.bid_pressure(state, &mut outs);
    let dp = spread(&outs);
    (0..=state.cards_dealt as i32).map(|b| exact(&dp, b)).collect()
}

/// Bid for the current player: the legal bid with the best expected
/// score `(10 + b) · P(make b)`, ties to the lower bid.
pub fn rule_bot_2_bid(state: &BlobState) -> u8 {
    let mask = legal_bids(state);
    let mut best = (0u8, f32::NEG_INFINITY);
    for (b, &p) in bid_chances(state).iter().enumerate() {
        let ev = (10 + b) as f32 * p;
        if (mask >> b) & 1 == 1 && ev > best.1 {
            best = (b as u8, ev);
        }
    }
    best.0
}

/// Chance of making the bid after each legal card, as `(card, chance)` in
/// ascending card index. All zero for a doomed seat.
pub fn play_chances(state: &BlobState) -> SmallVec<[(u8, f32); 13]> {
    debug_assert_eq!(state.phase(), GamePhase::Playing);
    let kn = Knowledge::new(state, state.current_player as usize);
    let need = state.bids[kn.me] as i32 - state.tricks_won[kn.me] as i32;
    cards(legal_plays(state))
        .map(|card| {
            let win = kn.trick_win(state, card);
            let dp = spread(&kn.hand_outlook(kn.hand & !(1u64 << card)));
            (card, win * exact(&dp, need - 1) + (1.0 - win) * exact(&dp, need))
        })
        .collect()
}

/// Card index to play for the current player: the best [`play_chances`],
/// ties to the weaker card.
pub fn rule_bot_2_play(state: &BlobState) -> u8 {
    debug_assert_eq!(state.phase(), GamePhase::Playing);
    let legal = legal_plays(state);
    if legal.count_ones() == 1 {
        return legal.trailing_zeros() as u8;
    }
    let trump = state.trump_suit;
    let chances = play_chances(state);
    let mut best = chances[0];
    for &(card, make) in &chances[1..] {
        let better = if (make - best.1).abs() > MAKE_TOL {
            make > best.1
        } else {
            strength(card, trump) < strength(best.0, trump)
        };
        if better {
            best = (card, make);
        }
    }
    best.0
}

/// Phase-stable action label for the current player — the bid value in
/// `Bidding`, the card index in `Playing` (same labels as
/// [`crate::mcts::apply_action`]).
///
/// Panics outside a decision phase (`Scoring` / `Complete`).
pub fn rule_bot_2_action(state: &BlobState) -> u8 {
    match state.phase() {
        GamePhase::Bidding => rule_bot_2_bid(state),
        GamePhase::Playing => rule_bot_2_play(state),
        phase => panic!("rule bot 2 asked to act in {phase:?}"),
    }
}

// ---------------------------------------------------------------------------
// v2r: rule bot 2 with rollouts
// ---------------------------------------------------------------------------

/// Rollout settings for v2r: rule bot 2 that, at every unforced decision,
/// tries each legal action on deals sampled from what it has seen and plays
/// the rest of the round forward with rule bot 2 at every seat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rollouts {
    /// Sampled deals per decision. Every action is rolled out on the same
    /// deals, so their comparison doesn't depend on which deals came up.
    pub samples: u32,
    /// Tricks each rollout looks ahead before the static estimate scores
    /// the position, counting the current trick (for a bid: the first
    /// trick). A trick in progress is always finished. `None` plays the
    /// round out and scores it exactly.
    pub depth: Option<u8>,
    /// Use rollouts for bids too; otherwise bids are rule bot 2's.
    pub bids: bool,
}

impl Default for Rollouts {
    fn default() -> Self {
        Rollouts { samples: 128, depth: None, bids: true }
    }
}

impl std::fmt::Display for Rollouts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.depth {
            Some(d) => write!(f, "{} samples, depth {d}", self.samples)?,
            None => write!(f, "{} samples, full depth", self.samples)?,
        }
        if !self.bids {
            write!(f, ", plays only")?;
        }
        Ok(())
    }
}

/// Expected round score of `seat` from its own view: exact once the round
/// is over, else `(10 + bid) · P(make)` from rule bot 2's estimate. `s`
/// must be between tricks.
fn expected_score(s: &BlobState, seat: usize) -> f32 {
    let bid = s.bids[seat];
    if s.phase() != GamePhase::Playing {
        return if s.tricks_won[seat] == bid { 10.0 + bid as f32 } else { 0.0 };
    }
    let kn = Knowledge::new(s, seat);
    let need = bid as i32 - s.tricks_won[seat] as i32;
    (10.0 + bid as f32) * exact(&spread(&kn.hand_outlook(kn.hand)), need)
}

/// Plays `w` forward with rule bot 2 at every seat until `stop` tricks are
/// complete or the round ends, and returns its value for `me`.
fn rollout(mut w: BlobState, me: usize, stop: u8) -> f32 {
    loop {
        match w.phase() {
            GamePhase::Bidding => {
                let b = rule_bot_2_bid(&w);
                apply_bid(&mut w, b);
            }
            GamePhase::Playing if w.trick_cards_played == 0 && w.tricks_completed >= stop => break,
            GamePhase::Playing => {
                let c = rule_bot_2_play(&w);
                apply_play(&mut w, c);
            }
            GamePhase::Scoring | GamePhase::Complete => break,
        }
    }
    let n = w.num_players as usize;
    let others: f32 = (0..n).filter(|&j| j != me).map(|j| expected_score(&w, j)).sum();
    expected_score(&w, me) - SPITE * others / (n - 1) as f32
}

/// Mean rollout value of each legal action of the current player (bid
/// value or card index): its expected round score minus [`SPITE`] × the
/// other seats' mean, over `cfg.samples` deals consistent with its hand,
/// the cards played and the known voids.
pub fn rollout_values<R: Rng + ?Sized>(state: &BlobState, cfg: &Rollouts, rng: &mut R) -> SmallVec<[(u8, f32); 14]> {
    let me = state.current_player as usize;
    let actions: SmallVec<[u8; 14]> = match state.phase() {
        GamePhase::Bidding => {
            let mask = legal_bids(state);
            (0..=state.cards_dealt).filter(|&b| (mask >> b) & 1 == 1).collect()
        }
        GamePhase::Playing => cards(legal_plays(state)).collect(),
        phase => panic!("rule bot 2r asked to act in {phase:?}"),
    };
    let stop = cfg.depth.map_or(u8::MAX, |d| state.tricks_completed.saturating_add(d));
    let voids = void_suits(state);
    let mut totals: SmallVec<[f32; 14]> = smallvec![0.0; actions.len()];
    for _ in 0..cfg.samples {
        let world = determinize(state, me as u8, &voids, rng, DEFAULT_DETERMINIZE_ATTEMPTS);
        for (total, &a) in totals.iter_mut().zip(&actions) {
            let mut w = world;
            apply_action(&mut w, a);
            *total += rollout(w, me, stop);
        }
    }
    let k = cfg.samples.max(1) as f32;
    actions.iter().zip(totals).map(|(&a, t)| (a, t / k)).collect()
}

/// v2r: the action with the best [`rollout_values`], ties to rule bot 2's
/// own choice. Forced moves (and bids, unless `cfg.bids`) skip the
/// rollouts.
pub fn rule_bot_2r_action<R: Rng + ?Sized>(state: &BlobState, cfg: &Rollouts, rng: &mut R) -> u8 {
    let plain = rule_bot_2_action(state);
    let options = match state.phase() {
        GamePhase::Bidding if !cfg.bids => 1,
        GamePhase::Bidding => legal_bids(state).count_ones(),
        _ => legal_plays(state).count_ones(),
    };
    if options <= 1 || cfg.samples == 0 {
        return plain;
    }
    let values = rollout_values(state, cfg, rng);
    let base = values.iter().find(|v| v.0 == plain).map_or(f32::NEG_INFINITY, |v| v.1);
    values.iter().fold((plain, base), |best, &(a, v)| if v > best.1 { (a, v) } else { best }).0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::{determinize, DEFAULT_DETERMINIZE_ATTEMPTS};
    use crate::bench::{run_bench, Agent, BenchConfig};
    use crate::bidding::apply_bid;
    use crate::dealing::start_round;
    use crate::game::{advance_round, is_game_over, new_game};
    use crate::playing::apply_play;
    use crate::rule_bot::rule_bot_action;
    use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256PlusPlus};

    const SPADES: u8 = 0;
    const HEARTS: u8 = 1;
    const CLUBS: u8 = 2;
    const KING: u8 = 11;
    const QUEEN: u8 = 10;
    const ACE: u8 = 12;

    fn card(suit: u8, rank: u8) -> u8 {
        suit * NUM_RANKS + rank
    }

    fn bits(cards: &[u8]) -> u64 {
        cards.iter().fold(0u64, |m, &c| m | (1u64 << c))
    }

    /// 5-player playing-phase state, no trump, every seat has bid; seat 0
    /// acts after `trick` was played by the seats before it.
    fn playing_state(cards_dealt: u8, hand: &[u8], trick: &[u8], bids: [u8; 5], won: u8) -> BlobState {
        let mut s = BlobState::empty();
        s.num_players = 5;
        s.cards_dealt = cards_dealt;
        s.trump_suit = NO_TRUMP;
        s.game_phase = GamePhase::Playing as u8;
        s.current_player = 0;
        s.trick_leader = (5 - trick.len() as u8) % 5;
        for (i, &c) in trick.iter().enumerate() {
            s.trick_play_order[i] = c;
        }
        s.trick_cards_played = trick.len() as u8;
        s.played_this_round = bits(trick);
        s.hands[0] = bits(hand);
        s.bids[..5].copy_from_slice(&bids);
        s.tricks_won[0] = won;
        s
    }

    /// Plays one game with every seat on rule bot 2, calling `check` on
    /// each decision state before acting.
    fn play_game(num_players: u8, start_cards: u8, seed: u64, mut check: impl FnMut(&BlobState, u8)) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut s = new_game(num_players, start_cards).unwrap();
        start_round(&mut s, &mut rng);
        while !is_game_over(&s) {
            match s.phase() {
                GamePhase::Bidding => {
                    let b = rule_bot_2_action(&s);
                    check(&s, b);
                    apply_bid(&mut s, b);
                }
                GamePhase::Playing => {
                    let c = rule_bot_2_action(&s);
                    check(&s, c);
                    apply_play(&mut s, c);
                }
                GamePhase::Scoring => advance_round(&mut s, &mut rng),
                GamePhase::Complete => unreachable!(),
            }
        }
    }

    #[test]
    fn plays_legal_full_games_at_every_table_size() {
        let mut tables: Vec<(u8, u8)> = (3..=8u8).map(|n| (n, (52 / n).min(8))).collect();
        tables.push((4, 13)); // whole deck dealt: no undealt stock
        for (n, c) in tables {
            for seed in 0..3 {
                play_game(n, c, seed, |s, a| {
                    let legal = if s.phase() == GamePhase::Bidding {
                        (legal_bids(s) >> a) & 1
                    } else {
                        ((legal_plays(s) >> a) & 1) as u16
                    };
                    assert_eq!(legal, 1, "illegal action {a} at {n}p/{c}c");
                });
            }
        }
    }

    /// Re-dealing the hidden cards (consistently with known voids) must
    /// never change a decision: the bot only reads its own hand and public
    /// information.
    #[test]
    fn decisions_never_read_hidden_hands() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5EE);
        for seed in 0..2 {
            play_game(5, 7, seed, |s, a| {
                let voids = void_suits(s);
                for _ in 0..2 {
                    let other = determinize(s, s.current_player, &voids, &mut rng, DEFAULT_DETERMINIZE_ATTEMPTS);
                    assert_eq!(rule_bot_2_action(&other), a, "decision changed with the hidden cards");
                }
            });
        }
    }

    #[test]
    fn fit_respects_hand_sizes_and_voids() {
        let mut checked = 0;
        play_game(5, 7, 3, |s, _| {
            if s.phase() != GamePhase::Playing {
                return;
            }
            let kn = Knowledge::new(s, s.current_player as usize);
            let voids = void_suits(s);
            for j in kn.opponents() {
                let expected: f32 = (0..NUM_SUITS).map(|x| kn.q[j][x as usize] * kn.unseen_in(x) as f32).sum();
                assert!((expected - kn.held[j] as f32).abs() < 0.05, "seat {j}: {expected} vs {}", kn.held[j]);
                for x in 0..SUITS {
                    if voids[j][x] {
                        assert_eq!(kn.q[j][x], 0.0, "void seat {j} holds suit {x}");
                    }
                }
                checked += 1;
            }
        });
        assert!(checked > 0);
    }

    #[test]
    fn one_card_round_bids_on_its_chance() {
        let mut s = BlobState::empty();
        s.num_players = 5;
        s.cards_dealt = 1;
        s.trump_suit = HEARTS;
        s.dealer = 0;
        s.current_player = 1; // first to bid, leads the trick
        s.hands[1] = bits(&[card(HEARTS, ACE)]);
        assert_eq!(rule_bot_2_bid(&s), 1, "the trump ace always wins");
        // Seat 2 after a bid of 1, holding a low side card it can't lead.
        apply_bid(&mut s, 1);
        s.hands[2] = bits(&[card(CLUBS, 3)]);
        assert_eq!(rule_bot_2_bid(&s), 0);
        assert!(bid_chances(&s)[1] < 0.05);
    }

    #[test]
    fn full_seat_sheds_its_highest_loser() {
        // ♠K led, ♠A on it; I'm at my bid of 0 and hold ♠Q ♠5 ♣3. Both
        // spades lose now, and ♠Q would be the top spade later.
        let hand = [card(SPADES, QUEEN), card(SPADES, 3), card(CLUBS, 1)];
        let trick = [card(SPADES, KING), card(SPADES, ACE)];
        let s = playing_state(3, &hand, &trick, [0, 1, 0, 1, 1], 0);
        assert_eq!(rule_bot_2_play(&s), card(SPADES, QUEEN));
    }

    #[test]
    fn takes_a_needed_last_trick_with_its_strongest_winner() {
        // ♠5 led, three clubs on it, I play last needing one trick with
        // ♠K ♠9 ♠3. Win with the king: the 9 is an easier card to lose
        // with later (the rule bot keeps the king and wins with the 9).
        let hand = [card(SPADES, KING), card(SPADES, 7), card(SPADES, 1)];
        let trick = [card(SPADES, 3), card(CLUBS, 0), card(CLUBS, 1), card(CLUBS, 2)];
        let s = playing_state(3, &hand, &trick, [1, 0, 1, 0, 1], 0);
        assert_eq!(rule_bot_2_play(&s), card(SPADES, KING));
        assert_eq!(rule_bot_action(&s), card(SPADES, 7));
    }

    const QUICK: Rollouts = Rollouts { samples: 4, depth: Some(2), bids: true };

    /// Seat 0 plays v2r with `cfg`, the others rule bot 2; `check` sees each
    /// of seat 0's decisions.
    fn play_v2r_game(n: u8, c: u8, cfg: Rollouts, seed: u64, mut check: impl FnMut(&BlobState, u8)) {
        let mut cards_rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut decide = Xoshiro256PlusPlus::seed_from_u64(seed ^ 0xD1CE);
        let mut s = new_game(n, c).unwrap();
        start_round(&mut s, &mut cards_rng);
        while !is_game_over(&s) {
            match s.phase() {
                GamePhase::Bidding | GamePhase::Playing => {
                    let a = if s.current_player == 0 {
                        let a = rule_bot_2r_action(&s, &cfg, &mut decide);
                        check(&s, a);
                        a
                    } else {
                        rule_bot_2_action(&s)
                    };
                    apply_action(&mut s, a);
                }
                GamePhase::Scoring => advance_round(&mut s, &mut cards_rng),
                GamePhase::Complete => unreachable!(),
            }
        }
    }

    #[test]
    fn rollouts_play_legal_games() {
        for (n, c) in [(3, 8), (5, 7), (8, 6)] {
            for cfg in [QUICK, Rollouts { samples: 2, depth: None, bids: true }] {
                play_v2r_game(n, c, cfg, n as u64, |s, a| {
                    let legal = match s.phase() {
                        GamePhase::Bidding => (legal_bids(s) >> a) & 1 == 1,
                        _ => (legal_plays(s) >> a) & 1 == 1,
                    };
                    assert!(legal, "illegal action {a} at {n}p/{c}c");
                });
            }
        }
    }

    /// With the same RNG, re-dealing the hidden cards never changes a v2r
    /// decision: it samples deals from its own view only.
    #[test]
    fn rollouts_never_read_hidden_hands() {
        let mut deal = Xoshiro256PlusPlus::seed_from_u64(0x5EE);
        play_v2r_game(5, 4, QUICK, 1, |s, _| {
            let voids = void_suits(s);
            let other = determinize(s, s.current_player, &voids, &mut deal, DEFAULT_DETERMINIZE_ATTEMPTS);
            let pick = |st: &BlobState| rule_bot_2r_action(st, &QUICK, &mut Xoshiro256PlusPlus::seed_from_u64(9));
            assert_eq!(pick(s), pick(&other), "decision changed with the hidden cards");
        });
    }

    #[test]
    fn depth_past_the_round_end_plays_it_out() {
        let mut checked = 0;
        play_v2r_game(5, 4, QUICK, 2, |s, _| {
            let deep = Rollouts { samples: 3, depth: Some(13), bids: true };
            let full = Rollouts { depth: None, ..deep };
            let values = |cfg: &Rollouts| rollout_values(s, cfg, &mut Xoshiro256PlusPlus::seed_from_u64(3));
            assert_eq!(values(&deep), values(&full));
            checked += 1;
        });
        assert!(checked > 0);
    }

    #[test]
    fn expected_score_is_exact_once_the_round_is_over() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(4);
        let mut s = new_game(5, 7).unwrap();
        start_round(&mut s, &mut rng);
        while s.phase() != GamePhase::Scoring {
            let a = rule_bot_2_action(&s);
            apply_action(&mut s, a);
        }
        for seat in 0..5 {
            let (bid, won) = (s.bids[seat], s.tricks_won[seat]);
            let want = if bid == won { 10.0 + bid as f32 } else { 0.0 };
            assert_eq!(expected_score(&s, seat), want);
        }
    }

    /// Guards against a change that silently weakens the bot. Full scale
    /// (4000 deals): +14.5 ± 0.3 points per game vs 4× the rule bot.
    #[test]
    fn beats_the_rule_bot() {
        let cfg = BenchConfig { deals: 60, threads: 4, ..BenchConfig::default() };
        let r = run_bench(&Agent::RuleBot2, &Agent::RuleBot, &cfg, &|_, _| {});
        assert!(r.diff > 8.0, "edge vs the rule bot only {:.1} ± {:.1}", r.diff, r.diff_ci95);
    }
}
