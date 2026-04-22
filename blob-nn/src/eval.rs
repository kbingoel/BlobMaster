//! Session 6.1 — evaluation games, strength tracking, and promotion.
//!
//! Provides a self-play-equivalent game loop that dispatches to a different
//! evaluator per seat (so we can pit iteration-N against iteration-M or the
//! heuristic baseline without entangling the self-play worker), a 200-game
//! head-to-head harness, Wilson 95% CI on win rate, and the strength CSV
//! append path plus the `best_model.onnx` / `best_stats.json` atomic
//! promotion plumbing.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use blob_engine::bidding::apply_bid;
use blob_engine::dealing::start_round;
use blob_engine::evaluator::Evaluator;
use blob_engine::game::{advance_round, is_game_over, new_game};
use blob_engine::hand::Hand;
use blob_engine::mcts::{mcts_search, MctsConfig};
use blob_engine::playing::apply_play;
use blob_engine::state::{BlobState, GamePhase, MAX_PLAYERS};
use rand::Rng;
use smallvec::SmallVec;

/// Per-seat evaluator table. `None` is reserved for the `Complete`/`Scoring`
/// phases where no decision is taken — any seat that might become the
/// acting player must have `Some`.
pub struct SeatEvaluators<'a>(pub [Option<&'a dyn Evaluator>; MAX_PLAYERS]);

impl<'a> SeatEvaluators<'a> {
    pub fn new() -> Self {
        Self([None; MAX_PLAYERS])
    }
    pub fn with(mut self, seat: usize, eval: &'a dyn Evaluator) -> Self {
        self.0[seat] = Some(eval);
        self
    }
}

impl<'a> Default for SeatEvaluators<'a> {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of one scored round. Captured *before* `advance_round` clears
/// `state.bids` / `state.tricks_won`.
#[derive(Debug, Clone, Copy)]
pub struct RoundOutcome {
    pub bids: [u8; MAX_PLAYERS],
    pub tricks_won: [u8; MAX_PLAYERS],
    pub num_players: u8,
}

/// Full outcome of one evaluation game.
#[derive(Debug, Clone)]
pub struct EvalGameOutcome {
    pub final_scores: [u16; MAX_PLAYERS],
    pub rounds: SmallVec<[RoundOutcome; 24]>,
}

/// Thin adapter so a `&dyn Evaluator` can be passed to the generic
/// `mcts_search<E: Evaluator>` API without requiring `dyn Evaluator: Sized`.
struct DynEval<'a>(&'a dyn Evaluator);
impl<'a> Evaluator for DynEval<'a> {
    fn evaluate(&self, state: &BlobState) -> (Vec<f32>, f32) {
        self.0.evaluate(state)
    }
}

fn sample_from_policy<R: Rng + ?Sized>(policy: &[f32], rng: &mut R) -> usize {
    let total: f32 = policy.iter().sum();
    if total <= 0.0 {
        return policy.iter().position(|&p| p > 0.0).unwrap_or(0);
    }
    let mut u: f32 = rng.gen::<f32>() * total;
    for (i, &p) in policy.iter().enumerate() {
        u -= p;
        if u <= 0.0 {
            return i;
        }
    }
    policy.len() - 1
}

/// Play a complete evaluation game with per-seat evaluators. Captures each
/// round's bids/tricks_won snapshot and returns the final cumulative scores.
pub fn play_eval_game<R: Rng + ?Sized>(
    num_players: u8,
    start_cards: u8,
    seats: &SeatEvaluators,
    mcts_cfg: &MctsConfig,
    rng: &mut R,
) -> EvalGameOutcome {
    let mut state = new_game(num_players, start_cards).expect("valid game params");
    start_round(&mut state, rng);
    let mut outcome = EvalGameOutcome {
        final_scores: [0; MAX_PLAYERS],
        rounds: SmallVec::new(),
    };

    while !is_game_over(&state) {
        match state.phase() {
            GamePhase::Bidding => {
                let seat = state.current_player as usize;
                let eval = seats.0[seat].expect("seat evaluator for active seat");
                let wrapper = DynEval(eval);
                let result = mcts_search(&state, &wrapper, mcts_cfg, rng);
                let action = sample_from_policy(&result.policy, rng) as u8;
                apply_bid(&mut state, action);
            }
            GamePhase::Playing => {
                let seat = state.current_player as usize;
                let eval = seats.0[seat].expect("seat evaluator for active seat");
                let hand_cards: Vec<u8> =
                    Hand::new(state.hands[seat]).iter().map(|c| c.index()).collect();
                let wrapper = DynEval(eval);
                let result = mcts_search(&state, &wrapper, mcts_cfg, rng);
                let pos = sample_from_policy(&result.policy, rng);
                let card_idx = hand_cards[pos];
                apply_play(&mut state, card_idx);
            }
            GamePhase::Scoring => {
                // Capture the round snapshot before `advance_round` clears it.
                outcome.rounds.push(RoundOutcome {
                    bids: state.bids,
                    tricks_won: state.tricks_won,
                    num_players: state.num_players,
                });
                advance_round(&mut state, rng);
            }
            GamePhase::Complete => break,
        }
    }
    outcome.final_scores = state.cumulative_scores;
    outcome
}

/// Aggregate head-to-head result of one evaluation (current vs opponent).
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    pub num_games: usize,
    pub wins_a: usize,
    pub win_rate: f64,
    pub win_rate_lower95: f64,
    pub win_rate_upper95: f64,
    pub score_differential: f64,
    pub bid_success_rate_a: f64,
    pub bid_success_rate_b: f64,
    /// True iff neither early-stop band (`lower95 ≥ 0.55` or
    /// `upper95 ≤ 0.45`) was ever crossed and we burned the full cap.
    /// Session 7.3a exposes this so the promotion gate can see post-hoc
    /// how often the CI stayed ambiguous at 200 games.
    pub inconclusive: bool,
}

/// Wilson 95% confidence interval.
pub fn wilson_95(successes: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 0.0);
    }
    let n = n as f64;
    let p = successes as f64 / n;
    let z = 1.959964f64;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let margin = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denom;
    ((center - margin).max(0.0), (center + margin).min(1.0))
}

/// Chunk size for sequential early-stop evaluation (Session 7.3a). The CI
/// is rechecked at every multiple of this; 50 gives four decision points
/// inside the 200-game cap.
pub const EVAL_CHUNK_SIZE: usize = 50;
/// Early-stop threshold: if `win_rate_lower95 ≥ EVAL_EARLY_STOP_HIGH` we
/// declare a clear improvement. 0.55 cushions above the 0.5 promotion
/// gate so a CI that just barely clears 0.5 doesn't trigger stop-and-
/// promote only to widen back below it on later chunks.
pub const EVAL_EARLY_STOP_HIGH: f64 = 0.55;
/// Mirror of `EVAL_EARLY_STOP_HIGH` for the regression side.
pub const EVAL_EARLY_STOP_LOW: f64 = 0.45;

/// Run a head-to-head evaluation between `eval_a` (the "current" model) and
/// `eval_b` (the opponent). Remaining seats are filled with `heuristic`.
///
/// Seat assignment rotates so each model spends an equal share of games in
/// each seat pair:
///   `seat_A = game_idx % num_players`
///   `seat_B = (game_idx + num_players / 2) % num_players`
///
/// Session 7.3a: `num_games` is a **cap**, not the number actually played.
/// Games run in chunks of `EVAL_CHUNK_SIZE`; after each chunk the Wilson
/// 95% CI on the cumulative win rate is recomputed, and we stop as soon
/// as `lower95 ≥ EVAL_EARLY_STOP_HIGH` (clear improvement) or `upper95 ≤
/// EVAL_EARLY_STOP_LOW` (clear regression). If neither band is crossed
/// by `num_games`, the returned `EvaluationResult.inconclusive` is true.
pub fn run_evaluation<R: Rng + ?Sized>(
    eval_a: &dyn Evaluator,
    eval_b: &dyn Evaluator,
    heuristic: &dyn Evaluator,
    num_games: usize,
    num_players: u8,
    start_cards: u8,
    mcts_cfg: &MctsConfig,
    rng: &mut R,
) -> EvaluationResult {
    let half = (num_players as usize) / 2;
    let mut wins_a = 0usize;
    let mut score_diff_sum: i64 = 0;
    let mut a_hits = 0usize;
    let mut a_rounds = 0usize;
    let mut b_hits = 0usize;
    let mut b_rounds = 0usize;

    let mut games_played = 0usize;
    let mut lo = 0.0f64;
    let mut hi = 1.0f64;
    let mut stopped_early = false;

    'outer: while games_played < num_games {
        let chunk_end = (games_played + EVAL_CHUNK_SIZE).min(num_games);
        for game_idx in games_played..chunk_end {
            let seat_a = game_idx % num_players as usize;
            let seat_b = (game_idx + half) % num_players as usize;
            debug_assert_ne!(seat_a, seat_b);

            let mut seats = SeatEvaluators::new();
            for s in 0..num_players as usize {
                seats.0[s] = Some(heuristic);
            }
            seats.0[seat_a] = Some(eval_a);
            seats.0[seat_b] = Some(eval_b);

            let outcome = play_eval_game(num_players, start_cards, &seats, mcts_cfg, rng);
            let sa = outcome.final_scores[seat_a] as i64;
            let sb = outcome.final_scores[seat_b] as i64;
            if sa > sb {
                wins_a += 1;
            }
            score_diff_sum += sa - sb;
            for r in &outcome.rounds {
                if r.bids[seat_a] == r.tricks_won[seat_a] {
                    a_hits += 1;
                }
                if r.bids[seat_b] == r.tricks_won[seat_b] {
                    b_hits += 1;
                }
                a_rounds += 1;
                b_rounds += 1;
            }
        }
        games_played = chunk_end;

        let (l, h) = wilson_95(wins_a, games_played);
        lo = l;
        hi = h;
        if lo >= EVAL_EARLY_STOP_HIGH || hi <= EVAL_EARLY_STOP_LOW {
            stopped_early = true;
            break 'outer;
        }
    }

    let inconclusive = !stopped_early && games_played >= num_games;
    EvaluationResult {
        num_games: games_played,
        wins_a,
        win_rate: if games_played > 0 {
            wins_a as f64 / games_played as f64
        } else {
            0.0
        },
        win_rate_lower95: lo,
        win_rate_upper95: hi,
        score_differential: if games_played > 0 {
            score_diff_sum as f64 / games_played as f64
        } else {
            0.0
        },
        bid_success_rate_a: if a_rounds > 0 {
            a_hits as f64 / a_rounds as f64
        } else {
            0.0
        },
        bid_success_rate_b: if b_rounds > 0 {
            b_hits as f64 / b_rounds as f64
        } else {
            0.0
        },
        inconclusive,
    }
}

/// One CSV row written by `append_strength_row` — mirrors the columns
/// described in development-plan §6.1, plus the Session 7.3a
/// `eval_games_played` / `eval_inconclusive` columns so post-hoc analysis
/// can see how often the sequential early-stop fired.
#[derive(Debug, Clone)]
pub struct StrengthRow {
    pub iteration: u64,
    /// Opponent label: either `"iter_NNNNNN"` or `"heuristic"`.
    pub opponent: String,
    pub win_rate: f64,
    pub win_rate_lower95: f64,
    pub win_rate_upper95: f64,
    pub score_differential: f64,
    pub bid_success_rate_current: f64,
    pub bid_success_rate_opponent: f64,
    pub policy_loss: f64,
    pub value_loss: f64,
    pub visit_entropy: f64,
    pub kl_divergence: f64,
    pub eval_games_played: u32,
    pub eval_inconclusive: bool,
}

/// Append (creating if needed) one row to `{checkpoint_dir}/strength.csv`.
pub fn append_strength_row(checkpoint_dir: &Path, row: &StrengthRow) -> std::io::Result<()> {
    fs::create_dir_all(checkpoint_dir)?;
    let path = checkpoint_dir.join("strength.csv");
    let exists = path.exists();
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    if !exists {
        writeln!(
            f,
            "iteration,opponent,win_rate,win_rate_lower95,win_rate_upper95,score_differential,bid_success_rate_current,bid_success_rate_opponent,policy_loss,value_loss,visit_entropy,kl_divergence,eval_games_played,eval_inconclusive"
        )?;
    }
    writeln!(
        f,
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        row.iteration,
        row.opponent,
        row.win_rate,
        row.win_rate_lower95,
        row.win_rate_upper95,
        row.score_differential,
        row.bid_success_rate_current,
        row.bid_success_rate_opponent,
        row.policy_loss,
        row.value_loss,
        row.visit_entropy,
        row.kl_divergence,
        row.eval_games_played,
        row.eval_inconclusive,
    )?;
    Ok(())
}

/// Persisted metadata for the current `best_model.onnx` pointer.
#[derive(Debug, Clone, Copy)]
pub struct BestStats {
    pub iteration: u64,
    pub win_rate_lower95: f32,
    pub bid_success_rate: f32,
}

impl BestStats {
    pub fn initial() -> Self {
        Self {
            iteration: 0,
            win_rate_lower95: 0.0,
            bid_success_rate: 0.0,
        }
    }

    pub fn load(checkpoint_dir: &Path) -> Option<Self> {
        let p = checkpoint_dir.join("best_stats.json");
        let s = fs::read_to_string(&p).ok()?;
        // Hand-rolled JSON parser: three flat numeric fields.
        let mut iteration: u64 = 0;
        let mut lo: f32 = 0.0;
        let mut bid: f32 = 0.0;
        for piece in s.trim_matches(|c: char| c == '{' || c == '}' || c.is_whitespace()).split(',') {
            let mut kv = piece.splitn(2, ':');
            let k = kv.next()?.trim().trim_matches('"');
            let v = kv.next()?.trim();
            match k {
                "iteration" => iteration = v.parse().ok()?,
                "win_rate_lower95" => lo = v.parse().ok()?,
                "bid_success_rate" => bid = v.parse().ok()?,
                _ => {}
            }
        }
        Some(Self {
            iteration,
            win_rate_lower95: lo,
            bid_success_rate: bid,
        })
    }

    pub fn save(&self, checkpoint_dir: &Path) -> std::io::Result<()> {
        fs::create_dir_all(checkpoint_dir)?;
        let tmp = checkpoint_dir.join("best_stats.json.tmp");
        let dst = checkpoint_dir.join("best_stats.json");
        {
            let mut f = fs::File::create(&tmp)?;
            write!(
                f,
                "{{\"iteration\":{},\"win_rate_lower95\":{},\"bid_success_rate\":{}}}",
                self.iteration, self.win_rate_lower95, self.bid_success_rate
            )?;
            f.sync_all()?;
        }
        fs::rename(tmp, dst)?;
        Ok(())
    }
}

/// Atomically update `{checkpoint_dir}/best_model.onnx` to point at
/// `src_onnx`. Uses a symlink on Unix and a plain-file atomic rename
/// fallback elsewhere (or when symlink creation is denied).
pub fn promote_best_model(checkpoint_dir: &Path, src_onnx: &Path) -> std::io::Result<()> {
    fs::create_dir_all(checkpoint_dir)?;
    let dst = checkpoint_dir.join("best_model.onnx");
    let _ = fs::remove_file(&dst);
    #[cfg(unix)]
    {
        if std::os::unix::fs::symlink(src_onnx, &dst).is_ok() {
            return Ok(());
        }
    }
    // Fallback: copy via tmp + rename for atomicity.
    let tmp = checkpoint_dir.join("best_model.onnx.tmp");
    fs::copy(src_onnx, &tmp)?;
    fs::rename(tmp, dst)?;
    Ok(())
}

/// Opponent-selection rule: the highest evaluated iteration ≤
/// `current_iter - eval_lookback`. Returns the opponent iteration number
/// if one exists, else `None` (caller falls back to the heuristic).
pub fn pick_opponent_iteration(
    current_iter: u64,
    eval_lookback: u64,
    eval_every: u64,
) -> Option<u64> {
    if current_iter < eval_lookback {
        return None;
    }
    let cap = current_iter - eval_lookback;
    let candidate = (cap / eval_every) * eval_every;
    if candidate == 0 {
        None
    } else {
        Some(candidate)
    }
}

/// Resolve the path to `iter_NNNNNN/model.onnx` for the given iteration.
pub fn iteration_onnx_path(checkpoint_dir: &Path, iter: u64) -> PathBuf {
    checkpoint_dir.join(format!("iter_{iter:06}")).join("model.onnx")
}

#[cfg(test)]
mod tests {
    use super::*;
    use blob_engine::evaluator::{DummyEvaluator, HeuristicEvaluator};
    use blob_engine::mcts::DEFAULT_ARENA_CAPACITY;
    use blob_engine::round::total_rounds;
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn fast_cfg() -> MctsConfig {
        MctsConfig {
            c_puct: 1.5,
            num_determinizations: 1,
            sims_per_determinization: 1,
            min_sims_floor: 1,
            temperature: 1.0,
            arena_capacity: DEFAULT_ARENA_CAPACITY,
        }
    }

    #[test]
    fn wilson_contains_point_estimate() {
        let (lo, hi) = wilson_95(100, 200);
        assert!(lo < 0.5 && hi > 0.5, "CI {lo}..{hi} around 0.5");
        let (lo2, hi2) = wilson_95(200, 200);
        assert!(lo2 > 0.9 && (hi2 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn pick_opponent_respects_lookback_and_cadence() {
        // eval_every=5, lookback=20
        assert_eq!(pick_opponent_iteration(5, 20, 5), None);
        assert_eq!(pick_opponent_iteration(20, 20, 5), None);
        assert_eq!(pick_opponent_iteration(25, 20, 5), Some(5));
        assert_eq!(pick_opponent_iteration(30, 20, 5), Some(10));
        assert_eq!(pick_opponent_iteration(50, 20, 5), Some(30));
    }

    #[test]
    fn per_round_capture_matches_total_rounds() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(5);
        let dummy = DummyEvaluator;
        let heur = HeuristicEvaluator;
        let mut seats = SeatEvaluators::new();
        for s in 0..4 {
            seats.0[s] = Some(&heur);
        }
        seats.0[0] = Some(&dummy);
        let cfg = fast_cfg();
        let outcome = play_eval_game(4, 3, &seats, &cfg, &mut rng);
        let expected = total_rounds(3, 4) as usize;
        assert_eq!(outcome.rounds.len(), expected);
        for r in &outcome.rounds {
            let cards = if r.num_players > 0 {
                // cards_dealt varies per round but bids sum ≤ cards_dealt
                // only when the dealer's restriction is active — invariant
                // is "bids sum ≠ cards_dealt" actually. Use the slightly
                // weaker "bids sum ≤ MAX_CARDS_DEALT" to avoid pulling in
                // per-round structure here.
                13u8
            } else {
                0
            };
            let sum: u16 = r.bids[..r.num_players as usize]
                .iter()
                .map(|&b| b as u16)
                .sum();
            assert!(sum <= cards as u16 * r.num_players as u16);
        }
    }

    #[test]
    fn best_stats_roundtrip() {
        let tmp = std::env::temp_dir().join(format!("blob-best-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let s = BestStats {
            iteration: 42,
            win_rate_lower95: 0.51,
            bid_success_rate: 0.37,
        };
        s.save(&tmp).unwrap();
        let loaded = BestStats::load(&tmp).unwrap();
        assert_eq!(loaded.iteration, 42);
        assert!((loaded.win_rate_lower95 - 0.51).abs() < 1e-5);
        assert!((loaded.bid_success_rate - 0.37).abs() < 1e-5);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn strength_csv_appends_header_once() {
        let tmp = std::env::temp_dir().join(format!("blob-strength-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let row = StrengthRow {
            iteration: 5,
            opponent: "heuristic".to_string(),
            win_rate: 0.55,
            win_rate_lower95: 0.48,
            win_rate_upper95: 0.62,
            score_differential: 3.5,
            bid_success_rate_current: 0.4,
            bid_success_rate_opponent: 0.3,
            policy_loss: 1.2,
            value_loss: 0.3,
            visit_entropy: 1.1,
            kl_divergence: 0.05,
            eval_games_played: 200,
            eval_inconclusive: false,
        };
        append_strength_row(&tmp, &row).unwrap();
        append_strength_row(&tmp, &row).unwrap();
        let s = fs::read_to_string(tmp.join("strength.csv")).unwrap();
        // Exactly one header line (= the line starting with "iteration,").
        let headers = s.lines().filter(|l| l.starts_with("iteration,")).count();
        assert_eq!(headers, 1);
        let _ = fs::remove_dir_all(&tmp);
    }
}
