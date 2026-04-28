//! Session 5.3 — Rayon-parallel self-play engine.
//!
//! Distributes independent full games across a `rayon` thread pool. Each
//! worker thread builds its own `OnnxEvaluator` (one `ort::Session` per
//! thread with `intra_op_num_threads=1`, see `blob_engine::onnx`) and its
//! own `Xoshiro256PlusPlus` RNG seeded from `(iteration, thread_index,
//! game_index)` so runs are reproducible.
//!
//! The ONNX model is re-exported once per training iteration (Session 5.4),
//! and `self_play_iteration` loads it fresh on each call. Games are
//! embarrassingly parallel — no synchronization between workers.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use blob_engine::mcts::MctsConfig;
use blob_engine::onnx::OnnxEvaluator;
use indicatif::{ProgressBar, ProgressStyle};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::self_play::{
    play_one_game_with_stats, sample_game_params, DecisionStat, TrainingExample,
};

/// Default self-play worker count. Section 5.3 targets 32-thread scaling.
pub const DEFAULT_NUM_THREADS: usize = 32;

/// Configuration for one self-play iteration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SelfPlayConfig {
    pub num_games: usize,
    pub num_threads: usize,
    /// Mixed into the per-game RNG seed so successive iterations don't
    /// replay identical games.
    pub iteration: u64,
    /// If false, suppress the `indicatif` progress bar (useful for tests).
    pub show_progress: bool,
    /// Session 7.1: if `Some((n, c))`, every worker uses this `(num_players,
    /// start_cards)` instead of drawing from `sample_game_params`. Used for
    /// the 7.1 / 7.2 fixed-5P7C runs; production training leaves it `None`.
    #[serde(default)]
    pub fixed_player_count: Option<(u8, u8)>,
    /// Session 7.4b: when true, self-play swaps `model.onnx` for the
    /// `model.int8.onnx` sibling produced by `scripts/export_onnx.py
    /// --int8-out`. Eval (`blob_nn::eval`) is unaffected — it loads whatever
    /// path the caller passes, so eval continues running on FP32.
    #[serde(default)]
    pub use_int8: bool,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            num_games: 1,
            num_threads: DEFAULT_NUM_THREADS,
            iteration: 0,
            show_progress: true,
            fixed_player_count: None,
            use_int8: false,
        }
    }
}

/// Run one self-play iteration.
///
/// Loads the ONNX model once per worker thread, distributes `num_games`
/// games across the pool, and returns every decision-point example from
/// every game. Caller is expected to `extend` a `ReplayBuffer` with the
/// result.
pub fn self_play_iteration(
    model_path: &Path,
    cfg: &SelfPlayConfig,
    mcts_cfg: &MctsConfig,
) -> (Vec<TrainingExample>, Vec<DecisionStat>) {
    let pool = ThreadPoolBuilder::new()
        .num_threads(cfg.num_threads)
        .build()
        .expect("build rayon pool");

    // Session 7.4b: if INT8 self-play is requested, swap the FP32 path for
    // its `.int8.onnx` sibling. The caller still passes the FP32 path so
    // eval (downstream) and the path-tracking logic in `blob-train` keep
    // pointing at the canonical artifact.
    let resolved_model: PathBuf = if cfg.use_int8 {
        let int8 = int8_model_path(model_path);
        if int8.exists() {
            int8
        } else {
            tracing::warn!(
                ?model_path,
                "use_int8 set but model.int8.onnx not found; falling back to FP32"
            );
            model_path.to_path_buf()
        }
    } else {
        model_path.to_path_buf()
    };
    let resolved_model = std::sync::Arc::new(resolved_model);

    let progress = if cfg.show_progress {
        let pb = ProgressBar::new(cfg.num_games as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner} {pos}/{len} games ({per_sec}) examples={msg} eta={eta}",
            )
            .unwrap(),
        );
        Some(pb)
    } else {
        None
    };

    let example_count = AtomicUsize::new(0);

    // `map_init` constructs one `OnnxEvaluator` per rayon work chunk and
    // reuses it across every game in that chunk, rather than per-game. The
    // session's internal mutex still prevents sharing one evaluator across
    // threads, so each worker keeps its own.
    let results: Vec<(Vec<TrainingExample>, Vec<DecisionStat>)> = pool.install(|| {
        (0..cfg.num_games)
            .into_par_iter()
            .map_init(
                {
                    let resolved_model = resolved_model.clone();
                    move || {
                        OnnxEvaluator::from_file(resolved_model.as_ref())
                            .expect("load ONNX model for self-play worker")
                    }
                },
                |eval, game_idx| {
                    let thread_idx = rayon::current_thread_index().unwrap_or(0) as u64;
                    let seed = mix_seed(cfg.iteration, thread_idx, game_idx as u64);
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
                    let (n, c) = match cfg.fixed_player_count {
                        Some(nc) => nc,
                        None => sample_game_params(&mut rng),
                    };
                    let (examples, stats) =
                        play_one_game_with_stats(n, c, eval, mcts_cfg, &mut rng);

                    let new_total = example_count
                        .fetch_add(examples.len(), Ordering::Relaxed)
                        + examples.len();
                    if let Some(pb) = &progress {
                        pb.inc(1);
                        pb.set_message(new_total.to_string());
                    }
                    (examples, stats)
                },
            )
            .collect()
    });

    if let Some(pb) = progress {
        pb.finish_and_clear();
    }

    let total: usize = results.iter().map(|(v, _)| v.len()).sum();
    let mut out = Vec::with_capacity(total);
    let mut stats_out: Vec<DecisionStat> = Vec::with_capacity(total);
    for (v, s) in results {
        out.extend(v);
        stats_out.extend(s);
    }
    (out, stats_out)
}

/// Map `…/model.onnx` → `…/model.int8.onnx`. If `path` doesn't end in
/// `.onnx`, append `.int8.onnx` to its stem.
pub fn int8_model_path(path: &Path) -> PathBuf {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string());
    dir.join(format!("{stem}.int8.onnx"))
}

/// SplitMix64-style scramble so `(iteration, thread, game)` triples produce
/// well-distributed seeds even when the inputs are small integers.
fn mix_seed(iteration: u64, thread: u64, game: u64) -> u64 {
    let mut x = iteration
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(thread.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(game.wrapping_mul(0x94D0_49BB_1331_11EB));
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix_seed_is_deterministic_and_scrambles() {
        assert_eq!(mix_seed(1, 2, 3), mix_seed(1, 2, 3));
        // Adjacent triples should not collide on the low bits.
        assert_ne!(mix_seed(0, 0, 0), mix_seed(0, 0, 1));
        assert_ne!(mix_seed(0, 0, 1), mix_seed(0, 1, 0));
        assert_ne!(mix_seed(0, 1, 0), mix_seed(1, 0, 0));
    }

    #[test]
    fn self_play_config_defaults() {
        let c = SelfPlayConfig::default();
        assert_eq!(c.num_threads, DEFAULT_NUM_THREADS);
        assert_eq!(c.num_games, 1);
        assert!(c.show_progress);
    }

    // An end-to-end rayon self-play test needs an exported ONNX model.
    // `scripts/export_onnx.py` produces one; point `BLOB_ONNX_MODEL` at
    // it to run this test, otherwise it skips (same pattern as
    // `blob_engine::onnx::tests`).
    #[test]
    fn runs_iteration_if_model_available() {
        let Some(path) = std::env::var("BLOB_ONNX_MODEL").ok() else {
            eprintln!("BLOB_ONNX_MODEL unset; skipping");
            return;
        };
        let path = std::path::PathBuf::from(path);
        if !path.exists() {
            eprintln!("BLOB_ONNX_MODEL does not exist; skipping");
            return;
        }

        let mcts_cfg = MctsConfig {
            c_puct: 1.5,
            num_determinizations: 1,
            sims_per_determinization: 1,
            min_sims_floor: 1,
            temperature: 1.0,
            temperature_schedule: None,
            arena_capacity: blob_engine::mcts::DEFAULT_ARENA_CAPACITY,
            target_batch: blob_engine::mcts::DEFAULT_TARGET_BATCH,
        };
        let cfg = SelfPlayConfig {
            num_games: 2,
            num_threads: 2,
            iteration: 0,
            show_progress: false,
            fixed_player_count: None,
            use_int8: false,
        };
        let (examples, _stats) = self_play_iteration(&path, &cfg, &mcts_cfg);
        assert!(!examples.is_empty());
    }
}
