//! Absolute-strength benchmark (gen-2.md §5.7): one focal player against
//! identical opponents, on duplicate deals.
//!
//! - **Duplicate deals.** A deal seed fixes the cards of every round of a
//!   game. Each seed is played once with the focal player in every seat,
//!   so across a seed's games the focal player holds every hand once and
//!   card luck cancels. Dealing uses its own RNG; decisions never consume
//!   it, so all games on one seed see the same cards whatever is played.
//! - **Confidence intervals** treat each seed's games as one sample, because
//!   games on the same cards are correlated.
//! - **Bots never search.** Opponents are the rule bot or a network playing
//!   its greedy raw policy; only the focal player may search.
//!
//! Every deal and every game is seeded from [`BenchConfig::seed`], so two
//! models benched with the same config play the same cards.
//!
//! Gen-1 reference (`run-2026-05-14/iter_000167`, 5 players / 7 cards):
//! about −9 points per game with 5×100 search and −13.5 network-only, both
//! vs the rule bot (gen-2.md §2.1).

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::bidding::{apply_bid, legal_bids};
use crate::dealing::start_round;
use crate::evaluator::Evaluator;
use crate::game::{advance_round, new_game};
use crate::hand::Hand;
use crate::mcts::{mcts_search, MctsConfig};
use crate::onnx::OnnxEvaluator;
use crate::playing::{apply_play, legal_plays};
use crate::rule_bot::rule_bot_action;
use crate::state::{BlobState, GamePhase};

/// Who sits in a seat.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Agent {
    /// The fixed rule bot (`rule_bot.rs`).
    RuleBot,
    /// A network's greedy raw policy, no search.
    Network(PathBuf),
    /// A network driving greedy MCTS with [`BenchConfig::mcts`].
    Search(PathBuf),
}

impl Agent {
    pub fn model(&self) -> Option<&Path> {
        match self {
            Agent::RuleBot => None,
            Agent::Network(p) | Agent::Search(p) => Some(p),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub num_players: u8,
    pub start_cards: u8,
    /// Number of deal seeds; games = `deals × num_players`.
    pub deals: usize,
    /// Base seed for the deal list and the search RNGs.
    pub seed: u64,
    /// Worker threads, one ONNX session per model per thread.
    pub threads: usize,
    /// Search settings for [`Agent::Search`]. Use [`eval_mcts_config`].
    pub mcts: MctsConfig,
}

pub const DEFAULT_SEED: u64 = 0xBE7C_5EED;

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            num_players: 5,
            start_cards: 7,
            deals: 64,
            seed: DEFAULT_SEED,
            threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8),
            mcts: eval_mcts_config(5, 100),
        }
    }
}

/// The gen-1 search recipe (`run-2026-05-14`) without root noise or a
/// temperature schedule; the caller plays the most-visited move.
pub fn eval_mcts_config(num_determinizations: u32, sims_per_determinization: u32) -> MctsConfig {
    MctsConfig {
        c_puct: 1.5,
        num_determinizations,
        sims_per_determinization,
        min_sims_floor: 60,
        temperature: 1.0,
        temperature_schedule: None,
        arena_capacity: 4096,
        target_batch: 5,
        root_dirichlet_alpha: 0.0,
        root_dirichlet_epsilon: 0.0,
    }
}

/// Labels of [`BidStats::buckets`], by cards dealt in the round.
pub const BUCKET_LABELS: [&str; 4] = ["1 card", "2-4 cards", "5-8 cards", "9+ cards"];

fn bucket(cards_dealt: u8) -> usize {
    match cards_dealt {
        0 | 1 => 0,
        2..=4 => 1,
        5..=8 => 2,
        _ => 3,
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BucketStats {
    pub rounds: u64,
    pub made: u64,
    pub zero_bids: u64,
}

/// Bid outcomes for one side of the table, one entry per (seat, round).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BidStats {
    pub buckets: [BucketStats; 4],
    /// Histogram of `tricks_won − bid`: ≤−3, −2, −1, 0, +1, +2, ≥+3.
    pub error_hist: [u64; 7],
}

impl BidStats {
    fn record(&mut self, cards_dealt: u8, bid: u8, tricks: u8) {
        let b = &mut self.buckets[bucket(cards_dealt)];
        b.rounds += 1;
        b.made += (bid == tricks) as u64;
        b.zero_bids += (bid == 0) as u64;
        let err = (tricks as i32 - bid as i32).clamp(-3, 3);
        self.error_hist[(err + 3) as usize] += 1;
    }

    fn merge(&mut self, o: &BidStats) {
        for (a, b) in self.buckets.iter_mut().zip(&o.buckets) {
            a.rounds += b.rounds;
            a.made += b.made;
            a.zero_bids += b.zero_bids;
        }
        for (a, b) in self.error_hist.iter_mut().zip(&o.error_hist) {
            *a += b;
        }
    }

    pub fn total(&self) -> BucketStats {
        self.buckets.iter().fold(BucketStats::default(), |a, b| BucketStats {
            rounds: a.rounds + b.rounds,
            made: a.made + b.made,
            zero_bids: a.zero_bids + b.zero_bids,
        })
    }
}

/// One finished game.
#[derive(Debug, Clone)]
pub struct GameRecord {
    pub deal: usize,
    pub focal_seat: u8,
    pub focal_score: f64,
    pub opponent_mean_score: f64,
    /// 1 for an outright win, 1/k when tied for first with k−1 others.
    pub win_share: f64,
    pub focal_bids: BidStats,
    pub opponent_bids: BidStats,
}

#[derive(Debug, Clone)]
pub struct BenchReport {
    pub focal: Agent,
    pub opponent: Agent,
    pub num_players: u8,
    pub start_cards: u8,
    /// `(dets, sims)` used when either side searches.
    pub search_budget: (u32, u32),
    pub deals: usize,
    pub games: usize,
    pub focal_points: f64,
    pub opponent_points: f64,
    /// Mean of focal score − opponents' mean score, per game.
    pub diff: f64,
    /// 95% half-width of `diff`, over deals (NaN with fewer than 2 deals).
    pub diff_ci95: f64,
    pub win_share: f64,
    pub win_share_ci95: f64,
    pub focal_bids: BidStats,
    pub opponent_bids: BidStats,
    pub secs: f64,
}

/// SplitMix64 finalizer over two words, for well-spread derived seeds.
fn mix(a: u64, b: u64) -> u64 {
    let mut x = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// RNG that deals every round of the game on deal `deal`.
pub fn deal_rng(seed: u64, deal: usize) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(mix(seed, deal as u64))
}

/// Most probable legal action of a dense policy (bids by value, plays by
/// hand position in `Hand::iter` order), as an action label: the bid, or
/// the card index. Ties go to the lowest index.
pub fn greedy_action(state: &BlobState, policy: &[f32]) -> u8 {
    let bidding = state.phase() == GamePhase::Bidding;
    let mut best: Option<(u8, f32)> = None;
    let mut consider = |action: u8, p: f32| {
        if best.is_none_or(|(_, bp)| p > bp) {
            best = Some((action, p));
        }
    };
    if bidding {
        let mask = legal_bids(state);
        for (b, &p) in policy.iter().enumerate() {
            if (mask >> b) & 1 == 1 {
                consider(b as u8, p);
            }
        }
    } else {
        let legal = legal_plays(state);
        let hand = Hand::new(state.hands[state.current_player as usize]);
        for (card, &p) in hand.iter().zip(policy) {
            if (legal >> card.index()) & 1 == 1 {
                consider(card.index(), p);
            }
        }
    }
    best.expect("decision state has a legal action").0
}

/// The action `agent` takes in `state`. `eval` must be loaded from the
/// agent's model when it has one.
pub fn agent_action(
    agent: &Agent,
    eval: Option<&OnnxEvaluator>,
    state: &BlobState,
    mcts: &MctsConfig,
    rng: &mut Xoshiro256PlusPlus,
) -> u8 {
    match agent {
        Agent::RuleBot => rule_bot_action(state),
        Agent::Network(_) => {
            let ev = eval.expect("network agent needs an evaluator");
            greedy_action(state, &ev.evaluate(state).0)
        }
        Agent::Search(_) => {
            let ev = eval.expect("search agent needs an evaluator");
            greedy_action(state, &mcts_search(state, ev, mcts, rng, 0).policy_target)
        }
    }
}

fn play_game(
    cfg: &BenchConfig,
    focal: (&Agent, Option<&OnnxEvaluator>),
    opponent: (&Agent, Option<&OnnxEvaluator>),
    deal: usize,
    focal_seat: u8,
) -> GameRecord {
    let n = cfg.num_players;
    let mut cards = deal_rng(cfg.seed, deal);
    let mut search_rng =
        Xoshiro256PlusPlus::seed_from_u64(mix(mix(cfg.seed, deal as u64), 1 + focal_seat as u64));
    let mut s = new_game(n, cfg.start_cards).expect("valid bench table");
    start_round(&mut s, &mut cards);
    let mut focal_bids = BidStats::default();
    let mut opponent_bids = BidStats::default();
    loop {
        match s.phase() {
            GamePhase::Bidding | GamePhase::Playing => {
                let (agent, ev) = if s.current_player == focal_seat { focal } else { opponent };
                let a = agent_action(agent, ev, &s, &cfg.mcts, &mut search_rng);
                if s.phase() == GamePhase::Bidding {
                    apply_bid(&mut s, a);
                } else {
                    apply_play(&mut s, a);
                }
            }
            GamePhase::Scoring => {
                for i in 0..n {
                    let side = if i == focal_seat { &mut focal_bids } else { &mut opponent_bids };
                    side.record(s.cards_dealt, s.bids[i as usize], s.tricks_won[i as usize]);
                }
                advance_round(&mut s, &mut cards);
            }
            GamePhase::Complete => break,
        }
    }
    let score = |i: u8| s.cumulative_scores[i as usize] as f64;
    let focal_score = score(focal_seat);
    let others: Vec<f64> = (0..n).filter(|&i| i != focal_seat).map(score).collect();
    let top = others.iter().copied().fold(focal_score, f64::max);
    let tied_at_top = 1 + others.iter().filter(|&&x| x == top).count();
    GameRecord {
        deal,
        focal_seat,
        focal_score,
        opponent_mean_score: others.iter().sum::<f64>() / others.len() as f64,
        win_share: if focal_score == top { 1.0 / tied_at_top as f64 } else { 0.0 },
        focal_bids,
        opponent_bids,
    }
}

fn load(agent: &Agent) -> Option<OnnxEvaluator> {
    agent.model().map(|p| {
        OnnxEvaluator::from_file(p).unwrap_or_else(|e| panic!("load ONNX model {}: {e}", p.display()))
    })
}

/// Play every game of the benchmark on `cfg.threads` threads and return
/// them in (deal, seat) order. `progress(done, total)` is called after
/// each finished game.
pub fn play_games(
    focal: &Agent,
    opponent: &Agent,
    cfg: &BenchConfig,
    progress: &(dyn Fn(usize, usize) + Sync),
) -> Vec<GameRecord> {
    let n = cfg.num_players as usize;
    let total = cfg.deals * n;
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let out = Mutex::new(Vec::with_capacity(total));
    std::thread::scope(|sc| {
        for _ in 0..cfg.threads.clamp(1, total.max(1)) {
            sc.spawn(|| {
                let focal_ev = load(focal);
                let opponent_ev = load(opponent);
                loop {
                    let g = next.fetch_add(1, Ordering::Relaxed);
                    if g >= total {
                        break;
                    }
                    let rec = play_game(
                        cfg,
                        (focal, focal_ev.as_ref()),
                        (opponent, opponent_ev.as_ref()),
                        g / n,
                        (g % n) as u8,
                    );
                    out.lock().unwrap().push(rec);
                    progress(done.fetch_add(1, Ordering::Relaxed) + 1, total);
                }
            });
        }
    });
    let mut games = out.into_inner().unwrap();
    games.sort_by_key(|g| (g.deal, g.focal_seat));
    games
}

/// Mean and 95% half-width of the per-deal means of `f`.
fn per_deal_mean_ci(games: &[GameRecord], deals: usize, f: impl Fn(&GameRecord) -> f64) -> (f64, f64) {
    let mut sums = vec![(0.0f64, 0usize); deals];
    for g in games {
        sums[g.deal].0 += f(g);
        sums[g.deal].1 += 1;
    }
    let means: Vec<f64> = sums.iter().filter(|s| s.1 > 0).map(|s| s.0 / s.1 as f64).collect();
    let k = means.len() as f64;
    let mean = means.iter().sum::<f64>() / k;
    if means.len() < 2 {
        return (mean, f64::NAN);
    }
    let var = means.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / (k - 1.0);
    (mean, 1.96 * (var / k).sqrt())
}

/// Aggregate finished games into a report.
pub fn summarize(focal: &Agent, opponent: &Agent, cfg: &BenchConfig, games: &[GameRecord], secs: f64) -> BenchReport {
    assert!(!games.is_empty(), "no games to summarize");
    let (diff, diff_ci95) = per_deal_mean_ci(games, cfg.deals, |g| g.focal_score - g.opponent_mean_score);
    let (win_share, win_share_ci95) = per_deal_mean_ci(games, cfg.deals, |g| g.win_share);
    let n = games.len() as f64;
    let mut focal_bids = BidStats::default();
    let mut opponent_bids = BidStats::default();
    for g in games {
        focal_bids.merge(&g.focal_bids);
        opponent_bids.merge(&g.opponent_bids);
    }
    BenchReport {
        focal: focal.clone(),
        opponent: opponent.clone(),
        num_players: cfg.num_players,
        start_cards: cfg.start_cards,
        search_budget: (cfg.mcts.num_determinizations, cfg.mcts.sims_per_determinization),
        deals: cfg.deals,
        games: games.len(),
        focal_points: games.iter().map(|g| g.focal_score).sum::<f64>() / n,
        opponent_points: games.iter().map(|g| g.opponent_mean_score).sum::<f64>() / n,
        diff,
        diff_ci95,
        win_share,
        win_share_ci95,
        focal_bids,
        opponent_bids,
        secs,
    }
}

/// Run the benchmark: [`play_games`] then [`summarize`].
pub fn run_bench(
    focal: &Agent,
    opponent: &Agent,
    cfg: &BenchConfig,
    progress: &(dyn Fn(usize, usize) + Sync),
) -> BenchReport {
    let started = Instant::now();
    let games = play_games(focal, opponent, cfg, progress);
    summarize(focal, opponent, cfg, &games, started.elapsed().as_secs_f64())
}

fn agent_label(a: &Agent, budget: (u32, u32)) -> String {
    match a {
        Agent::RuleBot => "rule bot".to_string(),
        Agent::Network(p) => format!("network {}", p.display()),
        Agent::Search(p) => format!("search {}x{} {}", budget.0, budget.1, p.display()),
    }
}

fn share(num: u64, den: u64) -> String {
    if den == 0 {
        "    -".to_string()
    } else {
        format!("{:.3}", num as f64 / den as f64)
    }
}

impl fmt::Display for BenchReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let opp = self.num_players - 1;
        writeln!(f, "focal     {}", agent_label(&self.focal, self.search_budget))?;
        writeln!(f, "opponents {opp}x {}", agent_label(&self.opponent, self.search_budget))?;
        writeln!(
            f,
            "table     {} players, {} start cards; {} deals x {} seats = {} games in {:.0} s",
            self.num_players, self.start_cards, self.deals, self.num_players, self.games, self.secs
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "points/game  focal {:.1}  opponents {:.1}  diff {:+.1} ± {:.1} (95% CI over deals)",
            self.focal_points, self.opponent_points, self.diff, self.diff_ci95
        )?;
        writeln!(
            f,
            "win share    {:.3} ± {:.3} (fair {:.3})",
            self.win_share,
            self.win_share_ci95,
            1.0 / self.num_players as f64
        )?;
        writeln!(f)?;
        writeln!(f, "{:<12}{:>8}{:>8}{:>9}{:>11}{:>12}", "bids", "rounds", "made", "0-bids", "opp made", "opp 0-bids")?;
        let rows = self.focal_bids.buckets.iter().zip(&self.opponent_bids.buckets).enumerate();
        for (i, (fb, ob)) in rows {
            if fb.rounds == 0 {
                continue;
            }
            writeln!(
                f,
                "{:<12}{:>8}{:>8}{:>9}{:>11}{:>12}",
                BUCKET_LABELS[i],
                fb.rounds,
                share(fb.made, fb.rounds),
                share(fb.zero_bids, fb.rounds),
                share(ob.made, ob.rounds),
                share(ob.zero_bids, ob.rounds)
            )?;
        }
        let (ft, ot) = (self.focal_bids.total(), self.opponent_bids.total());
        writeln!(
            f,
            "{:<12}{:>8}{:>8}{:>9}{:>11}{:>12}",
            "all",
            ft.rounds,
            share(ft.made, ft.rounds),
            share(ft.zero_bids, ft.rounds),
            share(ot.made, ot.rounds),
            share(ot.zero_bids, ot.rounds)
        )?;
        writeln!(f)?;
        writeln!(f, "{:<12}{:>7}{:>7}{:>7}{:>7}{:>7}{:>7}{:>7}", "tricks-bid", "<=-3", "-2", "-1", "0", "+1", "+2", ">=+3")?;
        for (label, st) in [("focal", &self.focal_bids), ("opponents", &self.opponent_bids)] {
            let total: u64 = st.error_hist.iter().sum();
            write!(f, "{label:<12}")?;
            for &c in &st.error_hist {
                write!(f, "{:>7}", share(c, total))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(deals: usize) -> BenchConfig {
        BenchConfig { deals, threads: 4, ..BenchConfig::default() }
    }

    #[test]
    fn deals_repeat_across_seats() {
        // The cards of every round depend only on the deal seed.
        let mut a = deal_rng(DEFAULT_SEED, 3);
        let mut b = deal_rng(DEFAULT_SEED, 3);
        let mut sa = new_game(5, 7).unwrap();
        let mut sb = sa;
        start_round(&mut sa, &mut a);
        start_round(&mut sb, &mut b);
        assert_eq!(sa.hands, sb.hands);
        let mut c = deal_rng(DEFAULT_SEED, 4);
        let mut sc = new_game(5, 7).unwrap();
        start_round(&mut sc, &mut c);
        assert_ne!(sa.hands, sc.hands);
    }

    #[test]
    fn rule_bot_mirror_is_exactly_even() {
        // Identical deterministic players on duplicate deals: every seat
        // rotation replays the same game, so the focal seat's edge is 0.
        let c = cfg(4);
        let r = run_bench(&Agent::RuleBot, &Agent::RuleBot, &c, &|_, _| {});
        assert_eq!(r.games, 20);
        assert!(r.diff.abs() < 1e-9, "diff {}", r.diff);
        assert!(r.diff_ci95.abs() < 1e-9);
        assert_eq!(r.focal_bids.total().rounds, 20 * 17);
        assert_eq!(r.opponent_bids.total().rounds, 20 * 17 * 4);
        // Same bids on the same cards → same made share either side.
        let (ft, ot) = (r.focal_bids.total(), r.opponent_bids.total());
        assert_eq!(ft.made * 4, ot.made);
        // Five 1-card rounds per 5p/7c game.
        assert_eq!(r.focal_bids.buckets[0].rounds, 20 * 5);
    }

    #[test]
    fn games_come_back_in_order_and_cover_every_seat() {
        let c = cfg(3);
        let games = play_games(&Agent::RuleBot, &Agent::RuleBot, &c, &|_, _| {});
        let keys: Vec<(usize, u8)> = games.iter().map(|g| (g.deal, g.focal_seat)).collect();
        let want: Vec<(usize, u8)> = (0..3).flat_map(|d| (0..5).map(move |s| (d, s))).collect();
        assert_eq!(keys, want);
    }

    #[test]
    fn bid_stats_bucket_and_histogram() {
        let mut s = BidStats::default();
        s.record(1, 0, 0); // made 0-bid, 1 card
        s.record(3, 2, 1); // −1, 2-4 cards
        s.record(7, 1, 5); // +4 → clamps to ≥+3, 5-8 cards
        s.record(9, 0, 0);
        assert_eq!(s.buckets[0], BucketStats { rounds: 1, made: 1, zero_bids: 1 });
        assert_eq!(s.buckets[1], BucketStats { rounds: 1, made: 0, zero_bids: 0 });
        assert_eq!(s.buckets[2].rounds, 1);
        assert_eq!(s.buckets[3].zero_bids, 1);
        assert_eq!(s.error_hist, [0, 0, 1, 2, 0, 0, 1]);
        assert_eq!(s.total().rounds, 4);
    }

    #[test]
    fn greedy_action_skips_illegal_and_breaks_ties_low() {
        let mut rng = deal_rng(1, 0);
        let mut s = new_game(5, 7).unwrap();
        start_round(&mut s, &mut rng);
        // Bidding: all-zero policy except two equal legal maxima.
        let mut p = vec![0.0f32; 14];
        p[3] = 0.4;
        p[5] = 0.4;
        p[12] = 0.9; // illegal with 7 cards
        assert_eq!(greedy_action(&s, &p), 3);
    }

    #[test]
    fn win_share_splits_ties() {
        let c = cfg(1);
        let games = play_games(&Agent::RuleBot, &Agent::RuleBot, &c, &|_, _| {});
        // Mirror games: each seat rotation is the same game, so the shares
        // over the five rotations sum to exactly one winner.
        let total: f64 = games.iter().map(|g| g.win_share).sum();
        assert!((total - 1.0).abs() < 1e-9, "total {total}");
    }
}
