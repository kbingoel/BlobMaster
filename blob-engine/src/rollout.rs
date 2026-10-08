//! Policy iteration by rollouts (gen-2.md §6 Phase 5, day 3): P improves
//! on itself without search.
//!
//! A round is played by P ([`rollout_round`]). At a few of its decisions
//! with a choice, every legal move is played out on the real deal: the
//! move, then P's top move at every seat, each from its own view, to the
//! round's end. That is each move's exact value under P's play, on that
//! deal. The learner trains P on the decision as P saw it, with a loss
//! linear in those values (`blob_nn::train::pi_loss`), so over many deals P
//! follows each move's expected value given what it sees: the hidden cards
//! averaged as they really fall in self-play, inference from the bids and
//! plays included, with no sampled deals, no V and no tree. A single deal's
//! values are noisy; the average over every similar decision is not.
//!
//! - **Samples** ([`RolloutSample`]): the true state, P's policy there, and
//!   each legal move with every seat's points after its play-out.
//! - **Opponents** (`RolloutConfig::rule_bot_2_share`): seats of a round
//!   can be played by rule bot 2 instead of P, in the round and in every
//!   play-out, so P learns a best response to a mixed table rather than to
//!   itself. Only P's decisions are valued.
//! - **More deals for bids** (`RolloutConfig::bid_deals`): a bid can also be
//!   played out on deals drawn from the bidder's view (`belief::sample_deals`:
//!   the bids made so far weight them, and at bid time they are all there is
//!   to infer from). The training target then averages them with the real
//!   one: less noise where it is largest. Plays keep the real deal alone: the
//!   drawn deals ignore what the play so far reveals. Measurements read the
//!   real deal ([`RolloutSample::deal_utilities`]).
//! - **Children for V** ([`RolloutSample::children`]): the state after each
//!   move with the same points, what search asks V about.
//! - **Batches** ([`PiReplay::sample_batch`]): per phase, P's policy and
//!   the mover's utility of each move (`scoring::utilities`), centred over
//!   the legal moves; suits relabelled at random.

use rand::seq::index;
use rand::Rng;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::augment::{hand_position_map, permute_suits, random_suit_perm};
use crate::belief::{sample_deals, BidWeighting};
use crate::bidding::legal_bids;
use crate::dealing::{new_round, RoundParams};
use crate::encoder::hand_card_indices;
use crate::evaluator::{PolicyEvaluator, NUM_BIDS};
use crate::mcts::{apply_action, forced_action, is_terminal, play_out_greedy_with};
use crate::rule_bot_2::rule_bot_2_action;
use crate::playing::legal_plays;
use crate::replay::{SparsePolicy, MAX_BID_ACTIONS};
use crate::scoring::{normalized_scores, round_points, utilities, DEFAULT_LAMBDA};
use crate::selfplay::{at_temperature, sample_index, top};
use crate::state::{BlobState, GamePhase, MAX_PLAYERS};

/// How rollout rounds are played. Unknown keys are an error.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RolloutConfig {
    /// Decisions with a choice valued per round (all of them if the round
    /// has fewer), drawn uniformly.
    pub samples_per_round: usize,
    /// τ of the bids P plays in the round; 0 = its top bid, so the hidden
    /// cards fall as they do behind P's own play (what P must infer).
    pub bid_temperature: f32,
    /// τ of the cards P plays in the round.
    pub play_temperature: f32,
    /// Deals each valued bid is played out on: the real one and
    /// `bid_deals − 1` drawn from the bidder's view. Plays: the real one.
    pub bid_deals: usize,
    /// Bid weighting of the drawn deals.
    pub bid_weighting: BidWeighting,
    /// Chance that each seat of a round is played by rule bot 2 (in the
    /// round and its play-outs); at least one seat stays P's.
    pub rule_bot_2_share: f32,
}

impl Default for RolloutConfig {
    fn default() -> Self {
        Self {
            samples_per_round: 2,
            bid_temperature: 0.0,
            play_temperature: 0.0,
            bid_deals: 1,
            bid_weighting: BidWeighting { candidates: 2, noise: 0.1 },
            rule_bot_2_share: 0.0,
        }
    }
}

impl RolloutConfig {
    pub fn validate(&self) -> Result<(), String> {
        let ok = |x: f32| x.is_finite() && x >= 0.0;
        if self.samples_per_round == 0 || self.bid_deals == 0 || !ok(self.bid_temperature) || !ok(self.play_temperature) {
            return Err("rollout: samples_per_round and bid_deals must be > 0, temperatures >= 0".into());
        }
        if !(0.0..1.0).contains(&self.rule_bot_2_share) {
            return Err("rollout: rule_bot_2_share must be in [0, 1)".into());
        }
        Ok(())
    }
}

/// Each legal move (policy index: the bid, or the card's hand position) and
/// the points every seat (absolute) scored once the round was played out
/// from it.
pub type Outcomes = SmallVec<[(u8, [u8; MAX_PLAYERS]); MAX_BID_ACTIONS]>;

/// A decision with every legal move valued on the real deal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutSample {
    /// The true state before the move, every hand included.
    pub state: BlobState,
    /// P's policy here: the policy of the model that played the round.
    pub prior: SparsePolicy,
    /// On the real deal.
    pub outcomes: Outcomes,
    /// The same moves on each drawn deal (`RolloutConfig::bid_deals`).
    pub sampled: Vec<Outcomes>,
}

/// The action of policy index `i` in `s`: the bid, or the card at hand
/// position `i`.
fn action_of(s: &BlobState, i: u8) -> u8 {
    match s.phase() {
        GamePhase::Bidding => i,
        _ => hand_card_indices(s, s.current_player)[i as usize],
    }
}

/// The legal moves of `s` as policy indices, ascending.
fn legal_indices(s: &BlobState) -> SmallVec<[u8; MAX_BID_ACTIONS]> {
    match s.phase() {
        GamePhase::Bidding => (0..NUM_BIDS as u8).filter(|&b| (legal_bids(s) >> b) & 1 == 1).collect(),
        _ => {
            let legal = legal_plays(s);
            let hand = hand_card_indices(s, s.current_player);
            (0..hand.len() as u8).filter(|&i| (legal >> hand[i as usize]) & 1 == 1).collect()
        }
    }
}

impl RolloutSample {
    /// The mover's utility (`scoring::utilities` at `lambda`) of points.
    fn mover_utility(&self, pts: &[u8; MAX_PLAYERS], lambda: f32) -> f32 {
        let s_hat = normalized_scores(pts, self.state.cards_dealt);
        utilities(&s_hat, self.state.num_players, lambda)[self.state.current_player as usize]
    }

    /// The mover's utility of each move: the training target, averaged over
    /// the real deal and the drawn ones.
    pub fn utilities(&self, lambda: f32) -> SmallVec<[(u8, f32); MAX_BID_ACTIONS]> {
        let k = 1.0 + self.sampled.len() as f32;
        self.outcomes
            .iter()
            .enumerate()
            .map(|(j, (i, pts))| {
                let drawn: f32 = self.sampled.iter().map(|o| self.mover_utility(&o[j].1, lambda)).sum();
                (*i, (self.mover_utility(pts, lambda) + drawn) / k)
            })
            .collect()
    }

    /// The mover's utility of each move on the real deal alone: unbiased,
    /// for measurements.
    pub fn deal_utilities(&self, lambda: f32) -> SmallVec<[(u8, f32); MAX_BID_ACTIONS]> {
        self.outcomes.iter().map(|(i, pts)| (*i, self.mover_utility(pts, lambda))).collect()
    }

    /// The state after the move of policy index `i`.
    pub fn child(&self, i: u8) -> BlobState {
        let mut c = self.state;
        apply_action(&mut c, action_of(&self.state, i));
        c
    }

    /// The state after each move with the points it led to; states that end
    /// the round are left out.
    pub fn children(&self) -> Vec<(BlobState, [u8; MAX_PLAYERS])> {
        self.outcomes
            .iter()
            .map(|&(i, pts)| (self.child(i), pts))
            .filter(|(c, _)| !is_terminal(c))
            .collect()
    }

    /// Policy index of P's top move: its highest prior, ties to the lower
    /// index (as P alone plays).
    pub fn prior_top(&self) -> u8 {
        let mut best = (0u8, f32::NEG_INFINITY);
        for &(i, p) in &self.prior {
            if p > best.1 || (p == best.1 && i < best.0) {
                best = (i, p);
            }
        }
        best.0
    }

    /// The sample under suit relabelling `perm`: the state relabelled, play
    /// indices moved to their cards' new hand positions.
    fn relabelled(&self, perm: &crate::augment::SuitPerm) -> Self {
        if self.state.phase() != GamePhase::Playing {
            return Self { state: permute_suits(&self.state, perm), ..self.clone() };
        }
        let to = hand_position_map(self.state.hands[self.state.current_player as usize], perm);
        let moved = |o: &Outcomes| -> Outcomes { o.iter().map(|&(i, pts)| (to[i as usize], pts)).collect() };
        Self {
            state: permute_suits(&self.state, perm),
            prior: self.prior.iter().map(|&(i, p)| (to[i as usize], p)).collect(),
            outcomes: moved(&self.outcomes),
            sampled: self.sampled.iter().map(moved).collect(),
        }
    }
}

/// What rollout rounds looked like. Per phase (bids, plays): samples, moves
/// valued, samples whose best move on the deal is not P's top one, and the
/// sum of best minus P's top utility on the deal (hindsight: every card
/// seen, so no policy reaches it).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct RolloutStats {
    pub rounds: u64,
    pub samples: [u64; 2],
    pub moves: [u64; 2],
    pub top_not_best: [u64; 2],
    pub hindsight_sum: [f64; 2],
}

impl RolloutStats {
    pub fn merge(&mut self, o: &Self) {
        self.rounds += o.rounds;
        for k in 0..2 {
            self.samples[k] += o.samples[k];
            self.moves[k] += o.moves[k];
            self.top_not_best[k] += o.top_not_best[k];
            self.hindsight_sum[k] += o.hindsight_sum[k];
        }
    }

    fn record(&mut self, s: &RolloutSample) {
        let k = (s.state.phase() != GamePhase::Bidding) as usize;
        let u = s.deal_utilities(DEFAULT_LAMBDA);
        let top = s.prior_top();
        let at_top = u.iter().find(|&&(i, _)| i == top).map_or(f32::NEG_INFINITY, |&(_, x)| x);
        let best = u.iter().map(|&(_, x)| x).fold(f32::NEG_INFINITY, f32::max);
        self.samples[k] += 1;
        self.moves[k] += u.len() as u64;
        self.top_not_best[k] += (at_top < best - 1e-6) as u64;
        if at_top.is_finite() {
            self.hindsight_sum[k] += (best - at_top) as f64;
        }
    }
}

/// One round played by P, every seat (moves drawn at the configured
/// temperatures), and `cfg.samples_per_round` of its decisions with a
/// choice valued by playing out every legal move on the real deal with P's
/// top move at every seat. The play-outs of all the round's samples run
/// together, so P runs in batches.
pub fn rollout_round<P, R>(params: RoundParams, policy: &P, cfg: &RolloutConfig, rng: &mut R) -> (Vec<RolloutSample>, RolloutStats)
where
    P: PolicyEvaluator + ?Sized,
    R: Rng + ?Sized,
{
    // Rule bot 2's seats, at least one seat left to P.
    let mut bots = 0u8;
    if cfg.rule_bot_2_share > 0.0 {
        for seat in 0..params.num_players {
            if rng.gen::<f32>() < cfg.rule_bot_2_share {
                bots |= 1 << seat;
            }
        }
        if bots.count_ones() == params.num_players as u32 {
            bots &= !(1 << rng.gen_range(0..params.num_players));
        }
    }
    rollout_round_with(params, policy, cfg, bots, rng)
}

/// [`rollout_round`] with rule bot 2 at the seats in the `bots` bitmask.
pub fn rollout_round_with<P, R>(params: RoundParams, policy: &P, cfg: &RolloutConfig, bots: u8, rng: &mut R) -> (Vec<RolloutSample>, RolloutStats)
where
    P: PolicyEvaluator + ?Sized,
    R: Rng + ?Sized,
{
    let mut s = new_round(params, rng).expect("valid round parameters");
    let mut choices: Vec<(BlobState, Vec<f32>)> = Vec::with_capacity(params.num_players as usize * (1 + params.cards_dealt as usize));
    while matches!(s.phase(), GamePhase::Bidding | GamePhase::Playing) {
        let action = match forced_action(&s) {
            Some(a) => a,
            None if (bots >> s.current_player) & 1 == 1 => rule_bot_2_action(&s),
            None => {
                let p = policy.policy(&s);
                let tau = if s.phase() == GamePhase::Bidding { cfg.bid_temperature } else { cfg.play_temperature };
                let i = sample_index(&at_temperature(&p, tau, top(&p, &[])), rng);
                let a = action_of(&s, i as u8);
                choices.push((s, p));
                a
            }
        };
        apply_action(&mut s, action);
    }
    let mut stats = RolloutStats { rounds: 1, ..Default::default() };
    let mut picked = index::sample(rng, choices.len(), cfg.samples_per_round.min(choices.len())).into_vec();
    picked.sort_unstable();
    let moves: Vec<SmallVec<[u8; MAX_BID_ACTIONS]>> = picked.iter().map(|&c| legal_indices(&choices[c].0)).collect();
    // Each picked decision on the real deal; a bid also on `bid_deals − 1`
    // drawn ones.
    let deals: Vec<Vec<BlobState>> = picked
        .iter()
        .map(|&c| {
            let st = choices[c].0;
            let mut d = vec![st];
            if cfg.bid_deals > 1 && st.phase() == GamePhase::Bidding {
                d.extend(sample_deals(&st, st.current_player, policy, cfg.bid_deals - 1, cfg.bid_weighting, rng));
            }
            d
        })
        .collect();
    let mut games: Vec<BlobState> = Vec::with_capacity(moves.iter().zip(&deals).map(|(m, d)| m.len() * d.len()).sum());
    for ((&c, m), ds) in picked.iter().zip(&moves).zip(&deals) {
        let st = &choices[c].0;
        for d in ds {
            for &i in m {
                let mut g = *d;
                apply_action(&mut g, action_of(st, i));
                games.push(g);
            }
        }
    }
    play_out_greedy_with(&mut games, policy, bots);
    let mut ends = games.iter();
    let mut samples = Vec::with_capacity(picked.len());
    for ((&c, m), ds) in picked.iter().zip(&moves).zip(&deals) {
        let (st, p) = &choices[c];
        let mut on_deals: Vec<Outcomes> = (0..ds.len())
            .map(|_| m.iter().map(|&i| (i, round_points(ends.next().expect("one game per move and deal")))).collect())
            .collect();
        let outcomes = on_deals.remove(0);
        let prior = p.iter().enumerate().filter(|(_, &x)| x > 0.0).map(|(i, &x)| (i as u8, x)).collect();
        let sample = RolloutSample { state: *st, prior, outcomes, sampled: on_deals };
        stats.record(&sample);
        samples.push(sample);
    }
    (samples, stats)
}

/// One phase's rollout samples as dense rows: `cols` columns (14 for bids,
/// the largest hand position + 1 for plays), P's policy and the mover's
/// utility of each legal move, centred over the legal moves (0 elsewhere).
#[derive(Debug, Clone, Default)]
pub struct PiBatch {
    pub states: Vec<BlobState>,
    pub cols: usize,
    pub prior: Vec<f32>,
    pub utility: Vec<f32>,
}

impl PiBatch {
    /// The rows of `samples`, all of one phase.
    pub fn from_samples(samples: &[&RolloutSample], lambda: f32) -> Self {
        let Some(first) = samples.first() else { return Self::default() };
        let bids = first.state.phase() == GamePhase::Bidding;
        debug_assert!(samples.iter().all(|s| (s.state.phase() == GamePhase::Bidding) == bids), "one phase per batch");
        let cols = if bids {
            MAX_BID_ACTIONS
        } else {
            samples
                .iter()
                .flat_map(|s| s.outcomes.iter().map(|&(i, _)| i).chain(s.prior.iter().map(|&(i, _)| i)))
                .max()
                .map_or(1, |i| i as usize + 1)
        };
        let mut b = Self { states: Vec::with_capacity(samples.len()), cols, prior: vec![0.0; samples.len() * cols], utility: vec![0.0; samples.len() * cols] };
        for (row, s) in samples.iter().enumerate() {
            b.states.push(s.state);
            for &(i, p) in &s.prior {
                b.prior[row * cols + i as usize] = p;
            }
            let u = s.utilities(lambda);
            let mean = u.iter().map(|&(_, x)| x).sum::<f32>() / u.len().max(1) as f32;
            for &(i, x) in &u {
                b.utility[row * cols + i as usize] = x - mean;
            }
        }
        b
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

/// Rollout samples (FIFO), with the round id each came from (the learner's
/// validation split hashes it).
#[derive(Debug)]
pub struct PiReplay {
    samples: Vec<RolloutSample>,
    ids: Vec<u64>,
    capacity: usize,
    write_idx: usize,
}

impl PiReplay {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "rollout buffer capacity must be > 0");
        Self { samples: Vec::new(), ids: Vec::new(), capacity, write_idx: 0 }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn get(&self, i: usize) -> &RolloutSample {
        &self.samples[i]
    }

    pub fn round_id(&self, i: usize) -> u64 {
        self.ids[i]
    }

    /// Slots of the newest `n` samples (all of them if fewer), newest first.
    pub fn recent(&self, n: usize) -> Vec<usize> {
        let len = self.len();
        (1..=n.min(len)).map(|k| (self.write_idx + len - k) % len).collect()
    }

    pub fn push(&mut self, id: u64, sample: RolloutSample) {
        if self.samples.len() < self.capacity {
            self.samples.push(sample);
            self.ids.push(id);
        } else {
            self.samples[self.write_idx] = sample;
            self.ids[self.write_idx] = id;
        }
        self.write_idx = (self.write_idx + 1) % self.capacity;
    }

    /// `n` samples drawn uniformly (without replacement when `n <= len`),
    /// each relabelled at random with `augment`, as (bids, plays).
    pub fn sample_batch<R: Rng + ?Sized>(&self, n: usize, rng: &mut R, augment: bool, lambda: f32) -> (PiBatch, PiBatch) {
        assert!(!self.is_empty(), "cannot sample from an empty rollout buffer");
        let picked: Vec<usize> = if n <= self.len() {
            index::sample(rng, self.len(), n).into_vec()
        } else {
            (0..n).map(|_| rng.gen_range(0..self.len())).collect()
        };
        let owned: Vec<RolloutSample> = picked
            .iter()
            .map(|&i| if augment { self.samples[i].relabelled(&random_suit_perm(rng)) } else { self.samples[i].clone() })
            .collect();
        split_batches(owned.iter(), lambda)
    }

    /// The samples at `indices`, in order, not relabelled, as (bids, plays).
    pub fn batch_from_indices(&self, indices: &[usize], lambda: f32) -> (PiBatch, PiBatch) {
        split_batches(indices.iter().map(|&i| &self.samples[i]), lambda)
    }
}

fn split_batches<'a>(samples: impl Iterator<Item = &'a RolloutSample>, lambda: f32) -> (PiBatch, PiBatch) {
    let (bids, plays): (Vec<&RolloutSample>, Vec<&RolloutSample>) = samples.partition(|s| s.state.phase() == GamePhase::Bidding);
    (PiBatch::from_samples(&bids, lambda), PiBatch::from_samples(&plays, lambda))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::augment::all_suit_perms;
    use crate::evaluator::{uniform_policy, DummyEvaluator};
    use crate::scoring::terminal_utilities;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    /// P that puts 0.9 on its first legal move and spreads the rest: a
    /// deterministic top move that differs from the lowest card or bid in
    /// some states.
    struct Skewed;

    impl PolicyEvaluator for Skewed {
        fn policy(&self, s: &BlobState) -> Vec<f32> {
            let mut p = uniform_policy(s);
            let legal: Vec<usize> = (0..p.len()).filter(|&i| p[i] > 0.0).collect();
            if legal.len() > 1 {
                let pick = legal[(s.played_this_round.count_ones() as usize + s.current_player as usize) % legal.len()];
                for &i in &legal {
                    p[i] = if i == pick { 0.9 } else { 0.1 / (legal.len() - 1) as f32 };
                }
            }
            p
        }
    }

    fn greedy_play_out(mut g: BlobState, p: &impl PolicyEvaluator) -> BlobState {
        while !is_terminal(&g) {
            let a = forced_action(&g).unwrap_or_else(|| {
                let pr = p.policy(&g);
                action_of(&g, top(&pr, &[]) as u8)
            });
            apply_action(&mut g, a);
        }
        g
    }

    /// Every legal move of every sample is valued by its own greedy
    /// play-out on the real deal, one game at a time equal to the lockstep
    /// batches; the prior is P's policy at the state.
    #[test]
    fn samples_value_every_legal_move_by_its_play_out() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(4);
        let cfg = RolloutConfig { samples_per_round: 3, ..Default::default() };
        let mut n = 0;
        for k in 0..40u8 {
            let params = RoundParams { num_players: 5, cards_dealt: 1 + k % 7, trump: k % 5, dealer: k % 5 };
            let (samples, stats) = rollout_round(params, &Skewed, &cfg, &mut rng);
            assert_eq!(stats.rounds, 1);
            assert_eq!(stats.samples.iter().sum::<u64>(), samples.len() as u64);
            assert!(samples.len() <= 3);
            for s in &samples {
                n += 1;
                assert!(forced_action(&s.state).is_none(), "a decision with a choice");
                let legal = legal_indices(&s.state);
                assert_eq!(s.outcomes.iter().map(|&(i, _)| i).collect::<Vec<_>>(), legal.to_vec());
                let p = Skewed.policy(&s.state);
                for &(i, x) in &s.prior {
                    assert_eq!(p[i as usize], x);
                }
                for &(i, pts) in &s.outcomes {
                    let mut g = s.state;
                    apply_action(&mut g, action_of(&s.state, i));
                    let end = greedy_play_out(g, &Skewed);
                    assert_eq!(round_points(&end), pts, "move {i}");
                }
                // The mover's utility is the play-out's terminal utility.
                let me = s.state.current_player as usize;
                for (&(i, u), &(j, _)) in s.utilities(1.0).iter().zip(&s.outcomes) {
                    let mut g = s.state;
                    apply_action(&mut g, action_of(&s.state, j));
                    let end = greedy_play_out(g, &Skewed);
                    assert!((u - terminal_utilities(&end, 1.0)[me]).abs() < 1e-6, "move {i}");
                }
            }
        }
        assert!(n > 60, "{n} samples");
    }

    /// With more deals, every bid is also played out on each drawn deal,
    /// the target the mean over all of them, measurements on the real deal;
    /// plays keep the real deal alone.
    #[test]
    fn drawn_deals_average_into_the_bid_target() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(14);
        let cfg = RolloutConfig { samples_per_round: 3, bid_deals: 4, ..Default::default() };
        let (mut n, mut plays) = (0, 0);
        for k in 0..20u8 {
            let params = RoundParams { num_players: 5, cards_dealt: 3 + k % 5, trump: k % 5, dealer: k % 5 };
            for s in rollout_round(params, &Skewed, &cfg, &mut rng).0 {
                if s.state.phase() != GamePhase::Bidding {
                    plays += 1;
                    assert!(s.sampled.is_empty());
                    continue;
                }
                n += 1;
                assert_eq!(s.sampled.len(), 3);
                let me = s.state.current_player as usize;
                for o in &s.sampled {
                    assert_eq!(o.iter().map(|&(i, _)| i).collect::<Vec<_>>(), s.outcomes.iter().map(|&(i, _)| i).collect::<Vec<_>>());
                }
                let (avg, real) = (s.utilities(1.0), s.deal_utilities(1.0));
                for (j, (&(i, a), &(_, r))) in avg.iter().zip(&real).enumerate() {
                    let all: f32 = std::iter::once(&s.outcomes)
                        .chain(&s.sampled)
                        .map(|o| utilities(&normalized_scores(&o[j].1, s.state.cards_dealt), s.state.num_players, 1.0)[me])
                        .sum();
                    assert!((a - all / 4.0).abs() < 1e-6, "move {i}");
                    let mut g = s.state;
                    apply_action(&mut g, action_of(&s.state, i));
                    assert!((r - terminal_utilities(&greedy_play_out(g, &Skewed), 1.0)[me]).abs() < 1e-6, "move {i} on the real deal");
                }
            }
        }
        assert!(n > 15 && plays > 0, "{n} bids, {plays} plays");
        assert!(RolloutConfig { bid_deals: 0, ..Default::default() }.validate().is_err());
    }

    /// Rule bot 2's seats play its moves in the round and in the
    /// play-outs, and their decisions are not valued.
    #[test]
    fn rule_bot_seats_play_its_moves_and_are_not_valued() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(31);
        let cfg = RolloutConfig { samples_per_round: 100, ..Default::default() };
        let bots = 0b00110u8;
        let mut n = 0;
        for k in 0..20u8 {
            let params = RoundParams { num_players: 5, cards_dealt: 2 + k % 5, trump: k % 5, dealer: k % 5 };
            for s in rollout_round_with(params, &Skewed, &cfg, bots, &mut rng).0 {
                n += 1;
                assert_eq!((bots >> s.state.current_player) & 1, 0, "a bot's decision was valued");
                for &(i, pts) in &s.outcomes {
                    let mut g = s.child(i);
                    while !is_terminal(&g) {
                        let a = forced_action(&g).unwrap_or_else(|| {
                            if (bots >> g.current_player) & 1 == 1 {
                                rule_bot_2_action(&g)
                            } else {
                                action_of(&g, top(&Skewed.policy(&g), &[]) as u8)
                            }
                        });
                        apply_action(&mut g, a);
                    }
                    assert_eq!(round_points(&g), pts, "move {i}");
                }
            }
        }
        assert!(n > 40, "{n} samples");
        // Drawn seats: some rounds get bots, never all five seats.
        let share = RolloutConfig { samples_per_round: 100, rule_bot_2_share: 0.9, ..Default::default() };
        for k in 0..20u8 {
            let params = RoundParams { num_players: 5, cards_dealt: 3, trump: k % 5, dealer: k % 5 };
            let movers: std::collections::BTreeSet<u8> =
                rollout_round(params, &Skewed, &share, &mut rng).0.iter().map(|s| s.state.current_player).collect();
            assert!(movers.len() <= 4);
        }
        assert!(RolloutConfig { rule_bot_2_share: 1.0, ..Default::default() }.validate().is_err());
    }

    #[test]
    fn rounds_with_fewer_choices_value_them_all() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(6);
        let params = RoundParams { num_players: 4, cards_dealt: 1, trump: 0, dealer: 0 };
        let cfg = RolloutConfig { samples_per_round: 50, ..Default::default() };
        let (samples, _) = rollout_round(params, &DummyEvaluator, &cfg, &mut rng);
        // A 1-card round: every play is forced, and so is the dealer's bid
        // unless the others bid 2 or more.
        assert!((3..=4).contains(&samples.len()), "{}", samples.len());
        assert!(samples.iter().all(|s| s.state.phase() == GamePhase::Bidding));
    }

    /// Children are the states after each move with the move's points; a
    /// move that ends the round has none.
    #[test]
    fn children_follow_each_move() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(9);
        let cfg = RolloutConfig { samples_per_round: 100, ..Default::default() };
        let params = RoundParams { num_players: 3, cards_dealt: 2, trump: 1, dealer: 2 };
        let (samples, _) = rollout_round(params, &Skewed, &cfg, &mut rng);
        for s in &samples {
            let ch = s.children();
            let ending = s.outcomes.iter().filter(|&&(i, _)| {
                let mut c = s.state;
                apply_action(&mut c, action_of(&s.state, i));
                is_terminal(&c)
            });
            assert_eq!(ch.len() + ending.count(), s.outcomes.len());
            for (c, pts) in ch {
                assert!(matches!(c.phase(), GamePhase::Bidding | GamePhase::Playing));
                assert!(s.outcomes.iter().any(|&(_, p)| p == pts));
            }
        }
    }

    /// Batches: bids and plays apart, the prior and the centred utilities
    /// in their columns, and a relabelling moves both with the cards.
    #[test]
    fn batches_carry_prior_and_centred_utility() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(12);
        let cfg = RolloutConfig { samples_per_round: 4, ..Default::default() };
        let mut buf = PiReplay::new(10_000);
        for k in 0..30u64 {
            let params = RoundParams { num_players: 5, cards_dealt: 2 + (k % 6) as u8, trump: (k % 5) as u8, dealer: (k % 5) as u8 };
            for s in rollout_round(params, &Skewed, &cfg, &mut rng).0 {
                buf.push(k, s);
            }
        }
        let all: Vec<usize> = (0..buf.len()).collect();
        let (bid, play) = buf.batch_from_indices(&all, 1.0);
        assert_eq!(bid.len() + play.len(), buf.len());
        assert_eq!(bid.cols, MAX_BID_ACTIONS);
        let bid_samples: Vec<&RolloutSample> = all.iter().map(|&i| buf.get(i)).filter(|s| s.state.phase() == GamePhase::Bidding).collect();
        for (row, s) in bid_samples.iter().enumerate() {
            let u = s.utilities(1.0);
            let mean = u.iter().map(|&(_, x)| x).sum::<f32>() / u.len() as f32;
            for &(i, x) in &u {
                assert!((bid.utility[row * bid.cols + i as usize] - (x - mean)).abs() < 1e-6);
            }
            let row_sum: f32 = bid.utility[row * bid.cols..(row + 1) * bid.cols].iter().sum();
            assert!(row_sum.abs() < 1e-5);
            let prior_sum: f32 = bid.prior[row * bid.cols..(row + 1) * bid.cols].iter().sum();
            assert!((prior_sum - 1.0).abs() < 1e-5);
        }
        // A relabelled play sample keeps each card's prior and outcome.
        let s = all.iter().map(|&i| buf.get(i)).find(|s| s.state.phase() == GamePhase::Playing).unwrap();
        for perm in all_suit_perms() {
            let r = s.relabelled(&perm);
            let hand = hand_card_indices(&s.state, s.state.current_player);
            let rhand = hand_card_indices(&r.state, r.state.current_player);
            for (&(i, p), &(j, q)) in s.prior.iter().zip(&r.prior) {
                assert_eq!(p, q);
                assert_eq!(crate::augment::permute_card(hand[i as usize], &perm), rhand[j as usize]);
            }
            for (&(i, a), &(j, b)) in s.outcomes.iter().zip(&r.outcomes) {
                assert_eq!(a, b);
                assert_eq!(crate::augment::permute_card(hand[i as usize], &perm), rhand[j as usize]);
            }
        }
        let (b2, p2) = buf.sample_batch(64, &mut rng, true, 1.0);
        assert_eq!(b2.len() + p2.len(), 64);
        // FIFO: the oldest sample goes first.
        let mut small = PiReplay::new(2);
        for (k, s) in [buf.get(0), buf.get(1), buf.get(2)].into_iter().enumerate() {
            small.push(k as u64, s.clone());
        }
        assert_eq!((small.len(), small.round_id(0), small.round_id(1)), (2, 2, 1));
        assert_eq!(small.recent(5), vec![0, 1], "newest first");
        assert_eq!(buf.recent(2), vec![buf.len() - 1, buf.len() - 2]);
    }

    #[test]
    fn stats_count_hindsight_gaps() {
        let mut st = RolloutStats::default();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2);
        let cfg = RolloutConfig { samples_per_round: 5, ..Default::default() };
        for k in 0..20u8 {
            let params = RoundParams { num_players: 5, cards_dealt: 5, trump: k % 5, dealer: k % 5 };
            st.merge(&rollout_round(params, &Skewed, &cfg, &mut rng).1);
        }
        assert_eq!(st.rounds, 20);
        assert!(st.samples[0] + st.samples[1] == 100);
        for k in 0..2 {
            assert!(st.hindsight_sum[k] >= 0.0 && st.top_not_best[k] <= st.samples[k]);
            assert!(st.moves[k] >= 2 * st.samples[k]);
        }
        assert!(st.top_not_best.iter().sum::<u64>() > 0, "a skewed P is not always best");
        assert!(RolloutConfig { samples_per_round: 0, ..Default::default() }.validate().is_err());
    }
}
