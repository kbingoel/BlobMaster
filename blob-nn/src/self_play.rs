//! Session 5.2 — Self-play worker and full game generation.
//!
//! Plays a complete game with MCTS at every decision point, records one
//! `TrainingExample` per decision (from the acting player's perspective),
//! then backfills a z-scored value target computed from cumulative game
//! scores once the game ends.

use blob_engine::bidding::{apply_bid, legal_bids};
use blob_engine::dealing::start_round;
use blob_engine::evaluator::Evaluator;
use blob_engine::game::{advance_round, is_game_over, new_game};
use blob_engine::hand::Hand;
use blob_engine::mcts::{adaptive_budget, mcts_search, signal_ratio, MctsConfig};
use blob_engine::playing::{apply_play, legal_plays};
use blob_engine::replay::{SparsePolicy, MAX_BID_ACTIONS};
use blob_engine::state::{BlobState, GamePhase, MAX_PLAYERS};
use rand::Rng;
use smallvec::SmallVec;

/// One replay-buffer-ready record. `value` is `NaN` until the surrounding
/// game completes and `backfill_values` overwrites it with the z-scored
/// cumulative-score target for that example's perspective.
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub state: BlobState,
    pub policy: SparsePolicy,
    pub value: f32,
    pub phase: GamePhase,
    /// Seat whose perspective the example is from (= `state.current_player`
    /// at decision time). Stored so the backfill pass can index
    /// `cumulative_scores` correctly after the game ends.
    pub perspective: u8,
}

/// Sample `(num_players, start_cards)` from the distribution described in
/// development-plan §5.2:
/// - n=4 (10%), n=5 (60%), n=6 (25%), n=7 (5%)
/// - C=7 (40%), C=8 (60%), subject to `n * C <= 52` (so n=7 forces C=7).
pub fn sample_game_params<R: Rng + ?Sized>(rng: &mut R) -> (u8, u8) {
    let u: f32 = rng.gen();
    let n: u8 = if u < 0.10 {
        4
    } else if u < 0.70 {
        5
    } else if u < 0.95 {
        6
    } else {
        7
    };
    let c: u8 = if n == 7 {
        7
    } else if rng.gen::<f32>() < 0.40 {
        7
    } else {
        8
    };
    (n, c)
}

fn sample_from_policy<R: Rng + ?Sized>(policy: &[f32], rng: &mut R) -> usize {
    let total: f32 = policy.iter().sum();
    if total <= 0.0 {
        return policy
            .iter()
            .enumerate()
            .find(|(_, &p)| p > 0.0)
            .map(|(i, _)| i)
            .unwrap_or(0);
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

fn dense_to_sparse(policy: &[f32]) -> SparsePolicy {
    let mut out: SparsePolicy = SmallVec::new();
    for (i, &p) in policy.iter().enumerate() {
        if p > 0.0 {
            out.push((i as u8, p));
        }
    }
    out
}

/// Per-decision signal-quality record (development-plan §7.1). Emitted
/// alongside `TrainingExample`s so the training loop can write a
/// `decision_stats.jsonl` sidecar and roll signal-ratio percentiles into
/// `IterationMetrics`.
#[derive(Debug, Clone, Copy)]
pub struct DecisionStat {
    pub phase: GamePhase,
    pub num_legal: u32,
    pub sims_used: u32,
    pub signal_ratio: f32,
}

/// Play one complete game and return the decision-point examples.
///
/// MCTS is driven by `eval` + `cfg`; actions are sampled from the MCTS
/// policy (temperature is already applied inside `mcts_search`). The
/// `decision_index` argument advances once per `mcts_search` call so
/// `cfg.temperature_schedule` (Session 7.4d) can sharpen τ as the game
/// progresses. Value targets are computed once the game ends — see
/// `backfill_values`.
pub fn play_one_game<E, R>(
    num_players: u8,
    start_cards: u8,
    eval: &E,
    cfg: &MctsConfig,
    rng: &mut R,
) -> Vec<TrainingExample>
where
    E: Evaluator + ?Sized,
    R: Rng + ?Sized,
{
    let (ex, _) = play_one_game_with_stats(num_players, start_cards, eval, cfg, rng);
    ex
}

/// Like `play_one_game`, additionally returning a `DecisionStat` per
/// decision point. Introduced in Session 7.1 to feed `decision_stats.jsonl`
/// and the adaptive-budget tuning planned in Session 7.3.
pub fn play_one_game_with_stats<E, R>(
    num_players: u8,
    start_cards: u8,
    eval: &E,
    cfg: &MctsConfig,
    rng: &mut R,
) -> (Vec<TrainingExample>, Vec<DecisionStat>)
where
    E: Evaluator + ?Sized,
    R: Rng + ?Sized,
{
    blob_engine::profiling::time(&blob_engine::profiling::GAME_TOTAL, || {
        let mut state = new_game(num_players, start_cards).expect("valid game params");
        start_round(&mut state, rng);

        let mut examples: Vec<TrainingExample> = Vec::new();
        let mut stats: Vec<DecisionStat> = Vec::new();
        // Session 7.4d: global decision counter, one increment per
        // `mcts_search` call. Drives `cfg.temperature_schedule`. Counts
        // bids and plays of every seat (forced moves do not call
        // `mcts_search` here, so they are not counted — but `mcts_search`
        // itself short-circuits forced moves regardless of τ).
        let mut decision_index: usize = 0;

        while !is_game_over(&state) {
            match state.phase() {
                GamePhase::Bidding => {
                    let num_legal = legal_bids(&state).count_ones() as usize;
                    let (dets, sims) = adaptive_budget(num_legal, cfg);
                    let result = mcts_search(&state, eval, cfg, rng, decision_index);
                    decision_index += 1;
                    debug_assert_eq!(result.policy.len(), MAX_BID_ACTIONS);
                    let perspective = state.current_player;
                    let sparse = dense_to_sparse(&result.policy);
                    let snapshot = state;
                    let action = sample_from_policy(&result.policy, rng) as u8;
                    stats.push(DecisionStat {
                        phase: GamePhase::Bidding,
                        num_legal: num_legal as u32,
                        sims_used: dets.saturating_mul(sims),
                        signal_ratio: signal_ratio(&result, num_legal),
                    });
                    examples.push(TrainingExample {
                        state: snapshot,
                        policy: sparse,
                        value: f32::NAN,
                        phase: GamePhase::Bidding,
                        perspective,
                    });
                    apply_bid(&mut state, action);
                }
                GamePhase::Playing => {
                    let perspective = state.current_player;
                    let hand_cards: Vec<u8> = Hand::new(state.hands[perspective as usize])
                        .iter()
                        .map(|c| c.index())
                        .collect();
                    let num_legal = legal_plays(&state).count_ones() as usize;
                    let (dets, sims) = adaptive_budget(num_legal, cfg);
                    let result = mcts_search(&state, eval, cfg, rng, decision_index);
                    decision_index += 1;
                    debug_assert_eq!(result.policy.len(), hand_cards.len());
                    let sparse = dense_to_sparse(&result.policy);
                    let snapshot = state;
                    let pos = sample_from_policy(&result.policy, rng);
                    let card_idx = hand_cards[pos];
                    stats.push(DecisionStat {
                        phase: GamePhase::Playing,
                        num_legal: num_legal as u32,
                        sims_used: dets.saturating_mul(sims),
                        signal_ratio: signal_ratio(&result, num_legal),
                    });
                    examples.push(TrainingExample {
                        state: snapshot,
                        policy: sparse,
                        value: f32::NAN,
                        phase: GamePhase::Playing,
                        perspective,
                    });
                    apply_play(&mut state, card_idx);
                }
                GamePhase::Scoring => {
                    advance_round(&mut state, rng);
                }
                GamePhase::Complete => break,
            }
        }

        backfill_values(&mut examples, &state);
        (examples, stats)
    })
}

/// Fill each example's `value` with the z-scored cumulative score of its
/// perspective player. `clip((s - mean) / max(std, ε), -1, 1)`. If all
/// players finished with the same score, every target is 0.0.
///
/// fix-mcts-plan.md C2c: the z-score statistic itself is shared with
/// `blob_engine::scoring::terminal_z_scores` (called by MCTS terminal
/// backprop), so in-tree Q at terminal leaves and the training value
/// target cannot drift in scale.
pub fn backfill_values(examples: &mut [TrainingExample], final_state: &BlobState) {
    let n = final_state.num_players as usize;
    debug_assert!(n >= 2 && n <= MAX_PLAYERS);
    let mut scores = [0.0f32; MAX_PLAYERS];
    for i in 0..n {
        scores[i] = final_state.cumulative_scores[i] as f32;
    }
    let z = blob_engine::scoring::z_score_clip(&scores, n);
    for ex in examples.iter_mut() {
        ex.value = z[ex.perspective as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blob_engine::evaluator::DummyEvaluator;
    use blob_engine::mcts::DEFAULT_ARENA_CAPACITY;
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn fast_cfg() -> MctsConfig {
        // Keep sims tiny — correctness of the self-play loop is the target,
        // not search quality. `adaptive_budget` will raise these to meet
        // the per-branching-factor floors.
        MctsConfig {
            c_puct: 1.5,
            num_determinizations: 1,
            sims_per_determinization: 1,
            min_sims_floor: 1,
            temperature: 1.0,
            temperature_schedule: None,
            arena_capacity: DEFAULT_ARENA_CAPACITY,
            target_batch: blob_engine::mcts::DEFAULT_TARGET_BATCH,
            root_dirichlet_alpha: 0.0,
            root_dirichlet_epsilon: 0.0,
        }
    }

    #[test]
    fn sample_params_in_valid_distribution() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        for _ in 0..1000 {
            let (n, c) = sample_game_params(&mut rng);
            assert!((4..=7).contains(&n));
            assert!(c == 7 || c == 8);
            assert!((n as u16) * (c as u16) <= 52);
            if n == 7 {
                assert_eq!(c, 7);
            }
        }
    }

    #[test]
    fn five_games_produce_valid_examples() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1234);
        let eval = DummyEvaluator;
        let cfg = fast_cfg();

        for game_i in 0..5 {
            let examples = play_one_game(5, 7, &eval, &cfg, &mut rng);
            assert!(!examples.is_empty(), "game {game_i} produced no examples");

            // Each example has a valid sparse policy summing to ~1.0.
            for ex in &examples {
                let s: f32 = ex.policy.iter().map(|&(_, p)| p).sum();
                assert!(
                    (s - 1.0).abs() < 1e-4,
                    "policy sum {s} in game {game_i}, phase={:?}",
                    ex.phase,
                );
                assert!(!ex.policy.is_empty());
                for &(_, p) in &ex.policy {
                    assert!(p > 0.0);
                }
                // Value is finite and in [-1, 1].
                assert!(ex.value.is_finite(), "value not finite");
                assert!(ex.value >= -1.0 && ex.value <= 1.0);
            }

            // All examples for the same perspective in this game share a
            // single value target (backfilled from cumulative score).
            for seat in 0..5u8 {
                let vals: Vec<f32> = examples
                    .iter()
                    .filter(|ex| ex.perspective == seat)
                    .map(|ex| ex.value)
                    .collect();
                if let Some(&first) = vals.first() {
                    for v in &vals {
                        assert!((v - first).abs() < 1e-6, "seat {seat} values diverge");
                    }
                }
            }
        }
    }

    #[test]
    fn policy_nonzero_only_at_legal_actions() {
        use blob_engine::bidding::legal_bids;
        use blob_engine::playing::legal_plays;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(77);
        let eval = DummyEvaluator;
        let cfg = fast_cfg();
        let examples = play_one_game(4, 7, &eval, &cfg, &mut rng);

        for ex in &examples {
            match ex.phase {
                GamePhase::Bidding => {
                    let mask = legal_bids(&ex.state);
                    for &(bid, _) in &ex.policy {
                        assert!((mask >> bid) & 1 == 1, "illegal bid {bid} in policy");
                    }
                }
                GamePhase::Playing => {
                    let hand = Hand::new(ex.state.hands[ex.perspective as usize]);
                    let hand_cards: Vec<u8> = hand.iter().map(|c| c.index()).collect();
                    let legal = legal_plays(&ex.state);
                    for &(pos, _) in &ex.policy {
                        let card_idx = hand_cards[pos as usize];
                        assert!(
                            (legal >> card_idx) & 1 == 1,
                            "illegal play pos={pos} card={card_idx}",
                        );
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn all_tied_scores_yield_zero_values() {
        let mut examples = vec![TrainingExample {
            state: BlobState::empty(),
            policy: SmallVec::new(),
            value: f32::NAN,
            phase: GamePhase::Bidding,
            perspective: 0,
        }];
        let mut final_state = BlobState::empty();
        final_state.num_players = 4;
        final_state.cumulative_scores = [5, 5, 5, 5, 0, 0, 0, 0];
        backfill_values(&mut examples, &final_state);
        assert_eq!(examples[0].value, 0.0);
    }
}
