//! Self-play rounds for the Phase-5 actors (gen-2.md §5.2, §5.6): every
//! seat searches with the same P and V, and every decision is recorded with
//! its search target.
//!
//! - **Targets:** the root visits at τ = 1 (`MctsResult::policy_target`),
//!   less the one visit every legal move gets in every tree
//!   (`prune_forced_visits`): an unvisited child scores +∞, so a bid
//!   searched 20 × 25 would otherwise put ≥ 4% on every legal bid, however
//!   bad (KataGo's policy-target pruning). 1-card bids are one-hot on the
//!   computed bid. Forced moves are recorded too (V trains on every state;
//!   P's learner skips them). With root rule `q` the target is π' instead
//!   (`mcts::RootRule::Q`: P reweighted by each move's mean value over the
//!   sampled deals), and nothing is pruned.
//! - **Moves played:** drawn from the same visits at the phase's
//!   temperature (`bid_temperature`, `play_temperature`; 0 = most visits),
//!   with root Dirichlet noise in every tree.
//! - **Search health** ([`SelfPlayStats`]): how far the targets depart from
//!   P's priors (KL, top-move disagreement), per phase. At c_puct 1.5 the
//!   targets were P's own priors (gen-2.md §6 Phase 4), so RL would have
//!   nothing to learn; these show whether that holds in self-play.
//!
//! **Rounds for V** ([`policy_round`]): P alone plays every seat, with one
//! random move per round, and the states after it are recorded with the
//! round's outcome. Search asks V about the root's children, one move off
//! P's policy and P's play after it; these rounds are that, at about 1/100
//! of a searched round's cost (AlphaGo trained its value net the same way,
//! on games of its policy net). The actor process plays them on
//! `value_actors` threads into `<run>/replay-v/`, which the learner reads
//! and deletes.
//!
//! **Rollout rounds** (`crate::rollout::rollout_round`, policy iteration
//! by rollouts): P alone plays every seat, and a few decisions per round
//! are valued by playing out every legal move on the real deal. The actor
//! process plays them on `rollout_actors` threads into `<run>/replay-pi/`,
//! which the learner reads and deletes.
//!
//! **The actor process** ([`run_actors`], `blobmaster selfplay`): ONNX only,
//! apart from the learner's libtorch process (an in-process mix crashed in
//! ONNX Runtime). It reads its settings from `<run>/selfplay.json`
//! ([`ActorsConfig`]) and its model from `<run>/model.json`
//! ([`ModelPointer`], re-read every few seconds), and writes the rounds to
//! `<run>/replay/` in chunks ([`Chunk`]), which the learner tails.

use std::io::{BufReader, BufWriter};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::SeqCst};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use crate::belief::BidWeighting;
use crate::dealing::{new_round, RoundParams};
use crate::encoder::hand_card_indices;
use crate::evaluator::{PolicyEvaluator, ValueEvaluator};
use crate::onnx::{OnnxPolicy, OnnxValue};
use crate::mcts::{
    apply_action, forced_action, mcts_search, MctsConfig, OneCardBids, RootRule, SearchBudget, DEFAULT_BID_BUDGET,
    DEFAULT_C_PUCT, DEFAULT_ONE_CARD_BIDS, DEFAULT_PLAY_BUDGET, DEFAULT_Q_TEMPERATURE,
};
use crate::replay::{Decision, SparsePolicy};
use crate::rollout::{rollout_round, RolloutConfig, RolloutSample, RolloutStats};
use crate::round::RoundMix;
use crate::scoring::DEFAULT_LAMBDA;
use crate::state::{BlobState, GamePhase};

/// Self-play search settings. Unknown keys are an error.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SelfPlayConfig {
    pub c_puct: f32,
    pub lambda: f32,
    pub bid_budget: SearchBudget,
    pub play_budget: SearchBudget,
    /// τ of the move played at a bid, over the visits; 0 = most visits.
    pub bid_temperature: f32,
    /// τ of the card played.
    pub play_temperature: f32,
    /// Root Dirichlet noise weight ε; 0 = off.
    pub dirichlet_epsilon: f32,
    /// Dirichlet α; 0 = 10 / legal moves.
    pub dirichlet_alpha: f32,
    pub one_card_bids: OneCardBids,
    pub bid_weighting: BidWeighting,
    /// Take the one visit per tree every legal move gets out of the target
    /// and the move drawn (root rule `visits` only).
    pub prune_forced_visits: bool,
    /// How the root's move and target are formed (`MctsConfig::root_rule`):
    /// `visits` or `q`.
    pub root_rule: RootRule,
    /// T of the `q` rule; 0 = the best Q̄.
    pub q_temperature: f32,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            c_puct: DEFAULT_C_PUCT,
            lambda: DEFAULT_LAMBDA,
            bid_budget: DEFAULT_BID_BUDGET,
            play_budget: DEFAULT_PLAY_BUDGET,
            bid_temperature: 1.0,
            play_temperature: 0.0,
            dirichlet_epsilon: 0.25,
            dirichlet_alpha: 0.0,
            one_card_bids: DEFAULT_ONE_CARD_BIDS,
            bid_weighting: BidWeighting::default(),
            prune_forced_visits: true,
            root_rule: RootRule::Visits,
            q_temperature: DEFAULT_Q_TEMPERATURE,
        }
    }
}

impl SelfPlayConfig {
    pub fn validate(&self) -> Result<(), String> {
        let ok = |x: f32| x.is_finite() && x >= 0.0;
        if !(self.c_puct > 0.0 && ok(self.lambda) && ok(self.bid_temperature) && ok(self.play_temperature)) {
            return Err("selfplay: c_puct must be > 0; lambda and temperatures >= 0".into());
        }
        if !ok(self.q_temperature) {
            return Err("selfplay: q_temperature must be >= 0".into());
        }
        if !(ok(self.dirichlet_epsilon) && self.dirichlet_epsilon <= 1.0 && ok(self.dirichlet_alpha)) {
            return Err("selfplay: dirichlet_epsilon must be in [0, 1], dirichlet_alpha >= 0".into());
        }
        let budgets = [self.bid_budget, self.play_budget];
        if budgets.iter().any(|b| b.determinizations == 0 || b.sims_per_determinization == 0) {
            return Err("selfplay: search budgets must be > 0".into());
        }
        Ok(())
    }

    /// The search of a decision in `phase`.
    pub fn mcts(&self, phase: GamePhase) -> MctsConfig {
        MctsConfig {
            c_puct: self.c_puct,
            lambda: self.lambda,
            bid_budget: self.bid_budget,
            play_budget: self.play_budget,
            temperature: if phase == GamePhase::Bidding { self.bid_temperature } else { self.play_temperature },
            root_dirichlet_epsilon: self.dirichlet_epsilon,
            root_dirichlet_alpha: self.dirichlet_alpha,
            one_card_bids: self.one_card_bids,
            bid_weighting: self.bid_weighting,
            root_rule: self.root_rule,
            q_temperature: self.q_temperature,
            arena_capacity: 4096,
            ..MctsConfig::default()
        }
    }
}

/// Search health of one phase's decisions with a choice.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct PhaseStats {
    pub decisions: u64,
    /// Σ KL(target ‖ P's prior), nats.
    pub kl_sum: f64,
    /// Decisions whose target's top move is not P's.
    pub top_differs: u64,
    /// Σ entropy of the target and of P's prior, nats.
    pub target_entropy_sum: f64,
    pub prior_entropy_sum: f64,
    /// Decisions where the move played is not the target's top move.
    pub off_top_played: u64,
}

impl PhaseStats {
    fn merge(&mut self, o: &Self) {
        self.decisions += o.decisions;
        self.kl_sum += o.kl_sum;
        self.top_differs += o.top_differs;
        self.target_entropy_sum += o.target_entropy_sum;
        self.prior_entropy_sum += o.prior_entropy_sum;
        self.off_top_played += o.off_top_played;
    }
}

/// Bid statistics buckets by cards dealt: 1, 2–4, 5+.
pub const SELFPLAY_BUCKETS: [&str; 3] = ["1", "2-4", "5+"];

fn bucket(cards: u8) -> usize {
    match cards {
        1 => 0,
        2..=4 => 1,
        _ => 2,
    }
}

/// What a batch of self-play rounds looked like.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct SelfPlayStats {
    pub rounds: u64,
    pub decisions: u64,
    pub forced: u64,
    pub bid: PhaseStats,
    pub play: PhaseStats,
    /// Seat-rounds and bids made, by [`SELFPLAY_BUCKETS`].
    pub seat_rounds: [u64; 3],
    pub made: [u64; 3],
    pub zero_bids: [u64; 3],
}

impl SelfPlayStats {
    pub fn merge(&mut self, o: &Self) {
        self.rounds += o.rounds;
        self.decisions += o.decisions;
        self.forced += o.forced;
        self.bid.merge(&o.bid);
        self.play.merge(&o.play);
        for i in 0..3 {
            self.seat_rounds[i] += o.seat_rounds[i];
            self.made[i] += o.made[i];
            self.zero_bids[i] += o.zero_bids[i];
        }
    }

    fn record_round(&mut self, end: &BlobState) {
        self.rounds += 1;
        let b = bucket(end.cards_dealt);
        for seat in 0..end.num_players as usize {
            self.seat_rounds[b] += 1;
            self.made[b] += (end.bids[seat] == end.tricks_won[seat]) as u64;
            self.zero_bids[b] += (end.bids[seat] == 0) as u64;
        }
    }
}

fn entropy(p: &[f32]) -> f64 {
    p.iter().filter(|&&x| x > 0.0).map(|&x| -(x as f64) * (x as f64).ln()).sum()
}

/// Index of the largest entry, ties to the higher `tiebreak` entry, then
/// the lowest index.
pub(crate) fn top(p: &[f32], tiebreak: &[f32]) -> usize {
    let t = |i: usize| tiebreak.get(i).copied().unwrap_or(0.0);
    (1..p.len()).fold(0, |b, i| if p[i] > p[b] || (p[i] == p[b] && t(i) > t(b)) { i } else { b })
}

/// An index drawn in proportion to `p`; the top entry if `p` sums to 0.
pub(crate) fn sample_index<R: Rng + ?Sized>(p: &[f32], rng: &mut R) -> usize {
    let z: f32 = p.iter().sum();
    if z <= 0.0 {
        return top(p, &[]);
    }
    let mut x = rng.gen::<f32>() * z;
    for (i, &w) in p.iter().enumerate() {
        if x < w {
            return i;
        }
        x -= w;
    }
    p.iter().rposition(|&w| w > 0.0).unwrap_or(0)
}

/// The visit distribution less one visit per tree per move (`trees` trees,
/// `total` visits in all), or `target` itself when nothing would be left or
/// there was no tree (forced moves, computed 1-card bids).
fn prune(target: &[f32], total: u32, trees: u32) -> Vec<f32> {
    if total == 0 {
        return target.to_vec();
    }
    let counts: Vec<f32> = target.iter().map(|&p| (p * total as f32 - trees as f32).max(0.0)).collect();
    let z: f32 = counts.iter().sum();
    if z <= 0.5 {
        return target.to_vec();
    }
    counts.iter().map(|&c| c / z).collect()
}

/// `p` at temperature `tau`; one-hot on index `best` below 1e-3.
pub(crate) fn at_temperature(p: &[f32], tau: f32, best: usize) -> Vec<f32> {
    if tau < 1e-3 {
        let mut out = vec![0.0; p.len()];
        out[best] = 1.0;
        return out;
    }
    p.iter().map(|&x| if x > 0.0 { x.powf(1.0 / tau) } else { 0.0 }).collect()
}

/// A finished self-play round.
pub struct SelfPlayRound {
    /// Every decision with its search target (bids by value, plays by hand
    /// position), forced ones included.
    pub decisions: Vec<Decision>,
    /// The finished round, in `Scoring`.
    pub end: BlobState,
    pub stats: SelfPlayStats,
}

/// Play one round with `params`, every seat searching with `policy` and
/// `value`.
pub fn selfplay_round<P, V, R>(params: RoundParams, policy: &P, value: &V, cfg: &SelfPlayConfig, rng: &mut R) -> SelfPlayRound
where
    P: PolicyEvaluator + ?Sized,
    V: ValueEvaluator + ?Sized,
    R: Rng + ?Sized,
{
    let mut s = new_round(params, rng).expect("valid round parameters");
    let mut decisions = Vec::with_capacity(params.num_players as usize * (1 + params.cards_dealt as usize));
    let mut stats = SelfPlayStats::default();
    let mut i = 0;
    while matches!(s.phase(), GamePhase::Bidding | GamePhase::Playing) {
        let phase = s.phase();
        let forced = forced_action(&s).is_some();
        let mcts = cfg.mcts(phase);
        let r = mcts_search(&s, policy, value, &mcts, rng, i);
        let visits = if cfg.prune_forced_visits && cfg.root_rule == RootRule::Visits {
            prune(&r.policy_target, r.total_visits, mcts.budget(phase).determinizations)
        } else {
            r.policy_target.clone()
        };
        let best = top(&visits, &r.root_prior);
        let chosen = sample_index(&at_temperature(&visits, mcts.temperature, best), rng);
        let action = match phase {
            GamePhase::Bidding => chosen as u8,
            _ => hand_card_indices(&s, s.current_player)[chosen],
        };
        let mut target: SparsePolicy =
            visits.iter().enumerate().filter(|(_, &p)| p > 0.0).map(|(a, &p)| (a as u8, p)).collect();
        if target.is_empty() {
            target.push((chosen as u8, 1.0));
        }
        stats.decisions += 1;
        if forced {
            stats.forced += 1;
        } else {
            let prior = policy.policy(&s);
            let ph = if phase == GamePhase::Bidding { &mut stats.bid } else { &mut stats.play };
            ph.decisions += 1;
            ph.kl_sum += visits
                .iter()
                .zip(&prior)
                .filter(|(&t, _)| t > 0.0)
                .map(|(&t, &p)| t as f64 * ((t as f64).ln() - (p.max(1e-8) as f64).ln()))
                .sum::<f64>();
            ph.top_differs += (best != top(&prior, &[])) as u64;
            ph.target_entropy_sum += entropy(&visits);
            ph.prior_entropy_sum += entropy(&prior);
            ph.off_top_played += (chosen != best) as u64;
        }
        decisions.push(Decision { state: s, policy: target });
        apply_action(&mut s, action);
        i += 1;
    }
    stats.record_round(&s);
    SelfPlayRound { decisions, end: s, stats }
}

/// A round of P alone ([`policy_round`]).
pub struct PolicyRound {
    /// The states after the round's random move, each with the move then
    /// played (V reads only the state; P never trains on these).
    pub decisions: Vec<Decision>,
    pub end: BlobState,
}

/// The index of `action` in `s`'s policy layout: the bid, or the card's
/// hand position.
fn action_index(s: &BlobState, action: u8) -> u8 {
    match s.phase() {
        GamePhase::Bidding => action,
        _ => hand_card_indices(s, s.current_player).iter().position(|&c| c == action).unwrap_or(0) as u8,
    }
}

/// One round played by P alone, for V's targets (AlphaGo's value-net
/// recipe). A decision index U is drawn uniformly. Before U every seat
/// samples P's policy (τ = 1); the first decision with a choice at or after
/// U is a uniformly random legal move; after it every seat plays P's top
/// move. The states after the random move are recorded: their outcome is
/// that of greedy P after one move off P's policy, which is what search
/// asks V at its root's children. If every decision from U on is forced,
/// the states after U are recorded (their outcome is fixed by the deal).
pub fn policy_round<P, R>(params: RoundParams, policy: &P, rng: &mut R) -> PolicyRound
where
    P: PolicyEvaluator + ?Sized,
    R: Rng + ?Sized,
{
    let mut s = new_round(params, rng).expect("valid round parameters");
    let n = params.num_players as usize * (1 + params.cards_dealt as usize);
    let u = rng.gen_range(0..n);
    let mut explored: Option<usize> = None;
    let mut kept: Vec<(usize, Decision)> = Vec::with_capacity(n);
    let mut i = 0;
    while matches!(s.phase(), GamePhase::Bidding | GamePhase::Playing) {
        let index = match forced_action(&s) {
            Some(a) => action_index(&s, a) as usize,
            None if i >= u && explored.is_none() => {
                explored = Some(i);
                kept.clear();
                sample_index(&crate::evaluator::uniform_policy(&s), rng)
            }
            None => {
                let p = policy.policy(&s);
                if i < u { sample_index(&p, rng) } else { top(&p, &[]) }
            }
        };
        if i > u && explored.is_none_or(|e| i > e) {
            let mut target = SparsePolicy::new();
            target.push((index as u8, 1.0));
            kept.push((i, Decision { state: s, policy: target }));
        }
        let action = match s.phase() {
            GamePhase::Bidding => index as u8,
            _ => hand_card_indices(&s, s.current_player)[index],
        };
        apply_action(&mut s, action);
        i += 1;
    }
    PolicyRound { decisions: kept.into_iter().map(|(_, d)| d).collect(), end: s }
}

// ---- the actor process ----------------------------------------------------------

/// The actor process's settings: `<run>/selfplay.json`, written by the
/// learner's driver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActorsConfig {
    pub actors: usize,
    pub seed: u64,
    pub mix: RoundMix,
    pub search: SelfPlayConfig,
    /// Rounds per chunk file.
    pub chunk_rounds: usize,
    /// Write a partial chunk after this many seconds without one.
    pub chunk_secs: f64,
    /// Threads playing rounds of P alone for V ([`policy_round`]) into
    /// `replay-v/`; 0 = none.
    #[serde(default)]
    pub value_actors: usize,
    /// A file of `replay-v/` every this many seconds.
    #[serde(default = "default_value_chunk_secs")]
    pub value_chunk_secs: f64,
    /// Threads playing rollout rounds (`crate::rollout::rollout_round`)
    /// into `replay-pi/`; 0 = none.
    #[serde(default)]
    pub rollout_actors: usize,
    #[serde(default)]
    pub rollout: RolloutConfig,
    /// A file of `replay-pi/` every this many seconds.
    #[serde(default = "default_value_chunk_secs")]
    pub rollout_chunk_secs: f64,
}

fn default_value_chunk_secs() -> f64 {
    30.0
}

pub const ACTORS_CONFIG_FILE: &str = "selfplay.json";
pub const MODEL_POINTER_FILE: &str = "model.json";
pub const REPLAY_DIR: &str = "replay";
/// Rounds of P alone for V ([`policy_round`]); the learner deletes each
/// file once read.
pub const VALUE_REPLAY_DIR: &str = "replay-v";
/// Rollout rounds ([`RolloutChunk`]); the learner deletes each file once
/// read.
pub const ROLLOUT_REPLAY_DIR: &str = "replay-pi";

/// The model the actors play: `<run>/model.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPointer {
    pub model: PathBuf,
    /// The learner step it was exported at.
    pub step: u64,
}

fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// Write `value` as JSON to `path`, atomically.
pub fn write_json<T: Serialize>(path: &Path, value: &T) -> std::io::Result<()> {
    write_atomic(path, (serde_json::to_string_pretty(value).expect("serializes") + "\n").as_bytes())
}

pub fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// One finished round as stored in `replay/`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRound {
    /// Round id: consecutive across chunks; the learner's validation split
    /// hashes it.
    pub id: u64,
    /// The learner step of the model that played it.
    pub model_step: u64,
    pub decisions: Vec<Decision>,
    pub end: BlobState,
}

/// A file of `replay/`: consecutive rounds and their statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Chunk {
    pub stats: SelfPlayStats,
    pub rounds: Vec<ChunkRound>,
}

/// One rollout round's samples as stored in `replay-pi/`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutChunkRound {
    /// Round id; the learner's validation split hashes it.
    pub id: u64,
    /// The learner step of the model that played it.
    pub model_step: u64,
    pub samples: Vec<RolloutSample>,
}

/// A file of `replay-pi/`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RolloutChunk {
    pub stats: RolloutStats,
    pub rounds: Vec<RolloutChunkRound>,
}

/// `run/<dir>/chunk-<first id>.bin`, written beside and renamed in, so a
/// reader never sees a partial file.
fn write_bin_in<T: Serialize>(run: &Path, dir: &str, first: u64, value: &T) -> Result<PathBuf, String> {
    let path = run.join(dir).join(format!("chunk-{first:09}.bin"));
    let tmp = path.with_extension("tmp");
    let f = std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    bincode::serialize_into(BufWriter::new(f), value).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}

fn read_bin<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    bincode::deserialize_from(BufReader::new(f)).map_err(|e| format!("{}: {e}", path.display()))
}

/// `replay/chunk-<first id>.bin`, written beside and renamed in, so a
/// reader never sees a partial file.
pub fn write_chunk(run: &Path, chunk: &Chunk) -> Result<PathBuf, String> {
    write_chunk_in(run, REPLAY_DIR, chunk)
}

/// [`write_chunk`] into `run/<dir>/`.
pub fn write_chunk_in(run: &Path, dir: &str, chunk: &Chunk) -> Result<PathBuf, String> {
    write_bin_in(run, dir, chunk.rounds.first().ok_or("empty chunk")?.id, chunk)
}

pub fn read_chunk(path: &Path) -> Result<Chunk, String> {
    read_bin(path)
}

/// A file of `replay-pi/`, written like [`write_chunk`].
pub fn write_rollout_chunk(run: &Path, chunk: &RolloutChunk) -> Result<PathBuf, String> {
    write_bin_in(run, ROLLOUT_REPLAY_DIR, chunk.rounds.first().ok_or("empty chunk")?.id, chunk)
}

pub fn read_rollout_chunk(path: &Path) -> Result<RolloutChunk, String> {
    read_bin(path)
}

/// The chunk files of `run`, by first round id.
pub fn chunk_files(run: &Path) -> Vec<(u64, PathBuf)> {
    chunk_files_in(run, REPLAY_DIR)
}

/// [`chunk_files`] of `run/<dir>/`.
pub fn chunk_files_in(run: &Path, dir: &str) -> Vec<(u64, PathBuf)> {
    let mut out: Vec<(u64, PathBuf)> = std::fs::read_dir(run.join(dir))
        .map(|d| {
            d.filter_map(|e| e.ok().map(|e| e.path()))
                .filter_map(|p| {
                    let id = p.file_name()?.to_str()?.strip_prefix("chunk-")?.strip_suffix(".bin")?.parse().ok()?;
                    Some((id, p))
                })
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

/// The id after the last round in `replay/` (0 if none). Reads the last
/// readable file.
pub fn next_round_id(run: &Path) -> u64 {
    chunk_files(run)
        .iter()
        .rev()
        .find_map(|(_, p)| read_chunk(p).ok().and_then(|c| c.rounds.last().map(|r| r.id + 1)))
        .unwrap_or(0)
}

/// The model pointer and a counter that changes when it does.
struct Current {
    pointer: Mutex<Option<ModelPointer>>,
    version: AtomicU64,
}

/// Play rounds on `cfg.actors` threads into `run/replay/` until `stop` is
/// set: the rounds in flight finish and the last partial chunk is written.
/// A round that panics (e.g. in ONNX Runtime) is dropped and the actor
/// reloads its networks.
pub fn run_actors(run: &Path, cfg: &ActorsConfig, stop: &AtomicBool) -> Result<(), String> {
    std::fs::create_dir_all(run.join(REPLAY_DIR)).map_err(|e| e.to_string())?;
    let first_id = next_round_id(run);
    let current = Arc::new(Current { pointer: Mutex::new(None), version: AtomicU64::new(0) });
    let pointer_path = run.join(MODEL_POINTER_FILE);
    let (tx, rx) = channel::<(SelfPlayRound, u64)>();
    let (vtx, vrx) = channel::<(PolicyRound, u64)>();
    let (rtx, rrx) = channel::<(Vec<RolloutSample>, RolloutStats, u64)>();
    let log = |msg: String| eprintln!("[selfplay] {msg}");
    log(format!(
        "{} actors, first round id {first_id}; {} actors for V; {} rollout actors",
        cfg.actors, cfg.value_actors, cfg.rollout_actors
    ));
    if cfg.value_actors > 0 {
        std::fs::create_dir_all(run.join(VALUE_REPLAY_DIR)).map_err(|e| e.to_string())?;
    }
    if cfg.rollout_actors > 0 {
        std::fs::create_dir_all(run.join(ROLLOUT_REPLAY_DIR)).map_err(|e| e.to_string())?;
    }
    std::thread::scope(|sc| {
        // Writer of the rollout rounds: a chunk every `rollout_chunk_secs`,
        // ids from the clock as for the rounds for V.
        sc.spawn(move || {
            let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
            let mut next = secs << 24;
            let mut chunk = RolloutChunk::default();
            let mut last = Instant::now();
            let flush = |chunk: &mut RolloutChunk, last: &mut Instant| {
                if !chunk.rounds.is_empty() {
                    if let Err(e) = write_rollout_chunk(run, chunk) {
                        eprintln!("[selfplay] {e}");
                    }
                }
                *chunk = RolloutChunk::default();
                *last = Instant::now();
            };
            loop {
                match rrx.recv_timeout(Duration::from_millis(500)) {
                    Ok((samples, stats, model_step)) => {
                        chunk.stats.merge(&stats);
                        if !samples.is_empty() {
                            chunk.rounds.push(RolloutChunkRound { id: next, model_step, samples });
                            next += 1;
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                }
                if last.elapsed().as_secs_f64() >= cfg.rollout_chunk_secs {
                    flush(&mut chunk, &mut last);
                }
            }
            flush(&mut chunk, &mut last);
        });
        // Writer of the rounds for V: a chunk every `value_chunk_secs`. The
        // learner deletes what it read, so ids start from the clock, unique
        // across restarts of this process.
        sc.spawn(move || {
            let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
            let mut next = secs << 24;
            let mut chunk = Chunk::default();
            let mut last = Instant::now();
            let flush = |chunk: &mut Chunk, last: &mut Instant| {
                if !chunk.rounds.is_empty() {
                    if let Err(e) = write_chunk_in(run, VALUE_REPLAY_DIR, chunk) {
                        eprintln!("[selfplay] {e}");
                    }
                }
                *chunk = Chunk::default();
                *last = Instant::now();
            };
            loop {
                match vrx.recv_timeout(Duration::from_millis(500)) {
                    Ok((round, model_step)) => {
                        chunk.rounds.push(ChunkRound { id: next, model_step, decisions: round.decisions, end: round.end });
                        next += 1;
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                }
                if last.elapsed().as_secs_f64() >= cfg.value_chunk_secs {
                    flush(&mut chunk, &mut last);
                }
            }
            flush(&mut chunk, &mut last);
        });
        // Writer: ids in arrival order, a chunk every `chunk_rounds` rounds
        // or `chunk_secs`.
        sc.spawn(move || {
            let mut next = first_id;
            let mut chunk = Chunk::default();
            let mut last = Instant::now();
            let flush = |chunk: &mut Chunk, last: &mut Instant| {
                if !chunk.rounds.is_empty() {
                    if let Err(e) = write_chunk(run, chunk) {
                        eprintln!("[selfplay] {e}");
                    }
                }
                *chunk = Chunk::default();
                *last = Instant::now();
            };
            loop {
                match rx.recv_timeout(Duration::from_millis(500)) {
                    Ok((round, model_step)) => {
                        chunk.stats.merge(&round.stats);
                        chunk.rounds.push(ChunkRound { id: next, model_step, decisions: round.decisions, end: round.end });
                        next += 1;
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                }
                if chunk.rounds.len() >= cfg.chunk_rounds || last.elapsed().as_secs_f64() >= cfg.chunk_secs {
                    flush(&mut chunk, &mut last);
                }
            }
            flush(&mut chunk, &mut last);
        });
        // Model pointer poller.
        {
            let current = current.clone();
            let pointer_path = pointer_path.clone();
            sc.spawn(move || {
                while !stop.load(SeqCst) {
                    if let Ok(p) = read_json::<ModelPointer>(&pointer_path) {
                        let mut cur = current.pointer.lock().unwrap();
                        if cur.as_ref() != Some(&p) {
                            eprintln!("[selfplay] model: step {} ({})", p.step, p.model.display());
                            *cur = Some(p);
                            current.version.fetch_add(1, SeqCst);
                        }
                    }
                    std::thread::sleep(Duration::from_secs(2));
                }
            });
        }
        for i in 0..cfg.actors {
            let (tx, current) = (tx.clone(), current.clone());
            let seed = cfg.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((first_id << 8) + i as u64);
            sc.spawn(move || {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
                let mut version = 0u64;
                let mut nets: Option<(OnnxPolicy, OnnxValue, u64)> = None;
                while !stop.load(SeqCst) {
                    let v = current.version.load(SeqCst);
                    if v != version || nets.is_none() {
                        let Some(p) = current.pointer.lock().unwrap().clone() else {
                            std::thread::sleep(Duration::from_millis(500));
                            continue;
                        };
                        match OnnxPolicy::from_dir(&p.model).and_then(|pn| OnnxValue::from_dir(&p.model).map(|vn| (pn, vn))) {
                            Ok((pn, vn)) => nets = Some((pn, vn, p.step)),
                            Err(e) => {
                                if i == 0 {
                                    eprintln!("[selfplay] can't load {}: {e}", p.model.display());
                                }
                                if nets.is_none() {
                                    std::thread::sleep(Duration::from_secs(5));
                                    continue;
                                }
                            }
                        }
                        version = v;
                    }
                    let (pn, vn, step) = nets.as_ref().expect("nets loaded");
                    let params = cfg.mix.sample(&mut rng);
                    match catch_unwind(AssertUnwindSafe(|| selfplay_round(params, pn, vn, &cfg.search, &mut rng))) {
                        Ok(round) => {
                            if tx.send((round, *step)).is_err() {
                                return;
                            }
                        }
                        Err(_) => {
                            eprintln!("[selfplay] actor {i}: a round panicked; reloading the networks");
                            nets = None;
                        }
                    }
                }
            });
        }
        for i in 0..cfg.value_actors {
            let (vtx, current) = (vtx.clone(), current.clone());
            let seed = cfg.seed.wrapping_mul(0xD1B5_4A32_D192_ED03) ^ ((first_id << 8) + 0x80 + i as u64);
            sc.spawn(move || {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
                let mut version = 0u64;
                let mut net: Option<(OnnxPolicy, u64)> = None;
                while !stop.load(SeqCst) {
                    let v = current.version.load(SeqCst);
                    if v != version || net.is_none() {
                        let Some(p) = current.pointer.lock().unwrap().clone() else {
                            std::thread::sleep(Duration::from_millis(500));
                            continue;
                        };
                        match OnnxPolicy::from_dir(&p.model) {
                            Ok(pn) => net = Some((pn, p.step)),
                            Err(e) => {
                                if i == 0 {
                                    eprintln!("[selfplay] V actors can't load {}: {e}", p.model.display());
                                }
                                if net.is_none() {
                                    std::thread::sleep(Duration::from_secs(5));
                                    continue;
                                }
                            }
                        }
                        version = v;
                    }
                    let (pn, step) = net.as_ref().expect("net loaded");
                    let params = cfg.mix.sample(&mut rng);
                    match catch_unwind(AssertUnwindSafe(|| policy_round(params, pn, &mut rng))) {
                        Ok(round) => {
                            if vtx.send((round, *step)).is_err() {
                                return;
                            }
                        }
                        Err(_) => {
                            eprintln!("[selfplay] V actor {i}: a round panicked; reloading the network");
                            net = None;
                        }
                    }
                }
            });
        }
        for i in 0..cfg.rollout_actors {
            let (rtx, current) = (rtx.clone(), current.clone());
            let seed = cfg.seed.wrapping_mul(0xA24B_AED4_963E_E407) ^ ((first_id << 8) + 0xC0 + i as u64);
            sc.spawn(move || {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
                let mut version = 0u64;
                let mut net: Option<(OnnxPolicy, u64)> = None;
                while !stop.load(SeqCst) {
                    let v = current.version.load(SeqCst);
                    if v != version || net.is_none() {
                        let Some(p) = current.pointer.lock().unwrap().clone() else {
                            std::thread::sleep(Duration::from_millis(500));
                            continue;
                        };
                        match OnnxPolicy::from_dir(&p.model) {
                            Ok(pn) => net = Some((pn, p.step)),
                            Err(e) => {
                                if i == 0 {
                                    eprintln!("[selfplay] rollout actors can't load {}: {e}", p.model.display());
                                }
                                if net.is_none() {
                                    std::thread::sleep(Duration::from_secs(5));
                                    continue;
                                }
                            }
                        }
                        version = v;
                    }
                    let (pn, step) = net.as_ref().expect("net loaded");
                    let params = cfg.mix.sample(&mut rng);
                    match catch_unwind(AssertUnwindSafe(|| rollout_round(params, pn, &cfg.rollout, &mut rng))) {
                        Ok((samples, stats)) => {
                            if rtx.send((samples, stats, *step)).is_err() {
                                return;
                            }
                        }
                        Err(_) => {
                            eprintln!("[selfplay] rollout actor {i}: a round panicked; reloading the network");
                            net = None;
                        }
                    }
                }
            });
        }
        drop(tx);
        drop(vtx);
        drop(rtx);
    });
    log("stopped".into());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::DummyEvaluator;
    use crate::scoring::round_points;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn small() -> SelfPlayConfig {
        SelfPlayConfig {
            bid_budget: SearchBudget::new(2, 8),
            play_budget: SearchBudget::new(2, 8),
            bid_weighting: BidWeighting::OFF,
            ..Default::default()
        }
    }

    /// Every decision is recorded with a distribution over legal moves, and
    /// replaying the moves that match the recorded states reaches the end.
    #[test]
    fn round_records_every_decision_with_a_legal_target() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(3);
        let ev = DummyEvaluator;
        for cards in [1u8, 3, 7] {
            let params = RoundParams { num_players: 5, cards_dealt: cards, trump: 1, dealer: 2 };
            let r = selfplay_round(params, &ev, &ev, &small(), &mut rng);
            assert_eq!(r.decisions.len(), 5 * (1 + cards as usize));
            assert_eq!(r.end.phase(), GamePhase::Scoring);
            assert_eq!(r.stats.decisions, r.decisions.len() as u64);
            assert_eq!(r.stats.rounds, 1);
            assert_eq!(r.stats.seat_rounds.iter().sum::<u64>(), 5);
            let _ = round_points(&r.end);
            for d in &r.decisions {
                let sum: f32 = d.policy.iter().map(|&(_, p)| p).sum();
                assert!((sum - 1.0).abs() < 1e-4, "target sums to {sum}");
                match d.state.phase() {
                    GamePhase::Bidding => {
                        let legal = crate::bidding::legal_bids(&d.state);
                        assert!(d.policy.iter().all(|&(b, _)| (legal >> b) & 1 == 1));
                    }
                    _ => {
                        let hand = hand_card_indices(&d.state, d.state.current_player);
                        let legal = crate::playing::legal_plays(&d.state);
                        assert!(d.policy.iter().all(|&(pos, _)| (legal >> hand[pos as usize]) & 1 == 1));
                    }
                }
            }
            for w in r.decisions.windows(2) {
                assert_eq!(w[0].state.cards_dealt, w[1].state.cards_dealt);
            }
            let ph = r.stats.bid.decisions + r.stats.play.decisions;
            assert_eq!(ph + r.stats.forced, r.stats.decisions);
        }
    }

    /// Root rule `q`: targets are π' over the legal moves, unpruned.
    #[test]
    fn q_rule_rounds_record_legal_targets() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(4);
        let ev = DummyEvaluator;
        let cfg = SelfPlayConfig { root_rule: RootRule::Q, play_budget: SearchBudget::new(4, 8), ..small() };
        let params = RoundParams { num_players: 5, cards_dealt: 4, trump: 0, dealer: 0 };
        let r = selfplay_round(params, &ev, &ev, &cfg, &mut rng);
        assert_eq!(r.decisions.len(), 25);
        for d in &r.decisions {
            let sum: f32 = d.policy.iter().map(|&(_, p)| p).sum();
            assert!((sum - 1.0).abs() < 1e-4, "target sums to {sum}");
        }
    }

    /// Rounds of P alone: the recorded states come from one round, in
    /// order, after the random move; there every unforced move is P's top
    /// (the uniform P's lowest legal index), and the forced ones are legal.
    #[test]
    fn policy_rounds_record_greedy_play_after_the_random_move() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(12);
        let ev = DummyEvaluator;
        let mut recorded = 0;
        for k in 0..200 {
            let cards = 1 + (k % 7) as u8;
            let params = RoundParams { num_players: 5, cards_dealt: cards, trump: (k % 5) as u8, dealer: (k % 5) as u8 };
            let r = policy_round(params, &ev, &mut rng);
            assert_eq!(r.end.phase(), GamePhase::Scoring);
            assert!(r.decisions.len() < 5 * (1 + cards as usize));
            recorded += r.decisions.len();
            for w in r.decisions.windows(2) {
                let progress = |s: &BlobState| (s.phase() == GamePhase::Playing, s.played_this_round.count_ones(), (0..s.num_players).filter(|&p| crate::bidding::has_bid(s, p)).count());
                assert!(progress(&w[0].state) < progress(&w[1].state), "states out of order");
            }
            for d in &r.decisions {
                assert_eq!(d.policy.len(), 1);
                let first_legal = crate::evaluator::uniform_policy(&d.state).iter().position(|&p| p > 0.0).unwrap();
                if forced_action(&d.state).is_none() {
                    assert_eq!(d.policy[0].0 as usize, first_legal, "not P's top after the random move");
                }
            }
        }
        assert!(recorded > 1000, "{recorded} states recorded");
    }

    #[test]
    fn rollout_chunks_round_trip() {
        let dir = std::env::temp_dir().join(format!("blob-rollout-chunks-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join(ROLLOUT_REPLAY_DIR)).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(5);
        let params = RoundParams { num_players: 5, cards_dealt: 4, trump: 2, dealer: 1 };
        let mut chunk = RolloutChunk::default();
        for id in 40..42u64 {
            let (samples, stats) = rollout_round(params, &DummyEvaluator, &RolloutConfig::default(), &mut rng);
            chunk.stats.merge(&stats);
            chunk.rounds.push(RolloutChunkRound { id, model_step: 3, samples });
        }
        let path = write_rollout_chunk(&dir, &chunk).unwrap();
        assert!(path.ends_with("chunk-000000040.bin"));
        let back = read_rollout_chunk(&path).unwrap();
        assert_eq!(back.stats, chunk.stats);
        assert_eq!(back.rounds[1].samples.len(), chunk.rounds[1].samples.len());
        assert_eq!(back.rounds[1].samples[0].outcomes, chunk.rounds[1].samples[0].outcomes);
        assert_eq!(chunk_files_in(&dir, ROLLOUT_REPLAY_DIR).len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn chunks_round_trip_and_ids_continue() {
        let dir = std::env::temp_dir().join(format!("blob-chunks-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join(REPLAY_DIR)).unwrap();
        assert_eq!(next_round_id(&dir), 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let ev = DummyEvaluator;
        let mut chunk = Chunk::default();
        for id in 0..3u64 {
            let params = RoundParams { num_players: 4, cards_dealt: 2, trump: 0, dealer: 1 };
            let r = selfplay_round(params, &ev, &ev, &small(), &mut rng);
            chunk.stats.merge(&r.stats);
            chunk.rounds.push(ChunkRound { id, model_step: 7, decisions: r.decisions, end: r.end });
        }
        let path = write_chunk(&dir, &chunk).unwrap();
        assert!(path.ends_with("chunk-000000000.bin"));
        let back = read_chunk(&path).unwrap();
        assert_eq!(back.stats, chunk.stats);
        assert_eq!(back.rounds.len(), 3);
        assert_eq!(back.rounds[2].decisions.len(), chunk.rounds[2].decisions.len());
        assert_eq!(next_round_id(&dir), 3);
        std::fs::write(dir.join(REPLAY_DIR).join("chunk-000000003.tmp"), b"partial").unwrap();
        assert_eq!(chunk_files(&dir).len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pruning_takes_one_visit_per_tree_out() {
        // 2 trees, 20 visits: counts 12 / 6 / 2 → 10 / 4 / 0.
        let t = [0.6f32, 0.3, 0.1];
        let p = prune(&t, 20, 2);
        assert!((p[0] - 10.0 / 14.0).abs() < 1e-6 && (p[1] - 4.0 / 14.0).abs() < 1e-6 && p[2] == 0.0, "{p:?}");
        assert_eq!(prune(&t, 0, 2), t.to_vec());
        assert_eq!(prune(&[0.5, 0.5], 2, 1), vec![0.5, 0.5]);
        assert_eq!(at_temperature(&[0.2, 0.8], 0.0, 1), vec![0.0, 1.0]);
    }

    #[test]
    fn sampling_follows_the_weights() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(9);
        let p = [0.0f32, 0.25, 0.0, 0.75];
        let mut n = [0usize; 4];
        for _ in 0..20_000 {
            n[sample_index(&p, &mut rng)] += 1;
        }
        assert_eq!((n[0], n[2]), (0, 0));
        assert!((n[3] as f64 / 20_000.0 - 0.75).abs() < 0.02, "{n:?}");
        assert_eq!(sample_index(&[0.0, 0.0], &mut rng), 0);
        assert_eq!(top(&[0.5, 0.5], &[0.1, 0.9]), 1);
    }

    #[test]
    fn config_rejects_unknown_keys_and_bad_values() {
        let cfg: SelfPlayConfig = toml::from_str("c_puct = 0.5\n").unwrap();
        assert_eq!(cfg.c_puct, 0.5);
        assert_eq!(cfg.play_budget, DEFAULT_PLAY_BUDGET);
        assert!(toml::from_str::<SelfPlayConfig>("cpuct = 0.5\n").is_err());
        assert!(SelfPlayConfig { dirichlet_epsilon: 2.0, ..Default::default() }.validate().is_err());
        assert!(SelfPlayConfig::default().validate().is_ok());
        let m = SelfPlayConfig::default().mcts(GamePhase::Bidding);
        assert_eq!((m.temperature, m.root_dirichlet_epsilon), (1.0, 0.25));
        assert_eq!(SelfPlayConfig::default().mcts(GamePhase::Playing).temperature, 0.0);
    }
}
