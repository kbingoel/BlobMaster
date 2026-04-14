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

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use blob_engine::mcts::MctsConfig;
use blob_engine::onnx::OnnxEvaluator;
use indicatif::{ProgressBar, ProgressStyle};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::self_play::{play_one_game, sample_game_params, TrainingExample};

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
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            num_games: 1,
            num_threads: DEFAULT_NUM_THREADS,
            iteration: 0,
            show_progress: true,
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
) -> Vec<TrainingExample> {
    let pool = ThreadPoolBuilder::new()
        .num_threads(cfg.num_threads)
        .build()
        .expect("build rayon pool");

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

    let results: Vec<Vec<TrainingExample>> = pool.install(|| {
        (0..cfg.num_games)
            .into_par_iter()
            .map(|game_idx| {
                // Per-thread ONNX session — `ort::Session` inside
                // `OnnxEvaluator` is guarded by a mutex, so sharing across
                // threads would serialize inference. Keep it local.
                let eval = OnnxEvaluator::from_file(model_path)
                    .expect("load ONNX model for self-play worker");
                let thread_idx = rayon::current_thread_index().unwrap_or(0) as u64;
                let seed = mix_seed(cfg.iteration, thread_idx, game_idx as u64);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
                let (n, c) = sample_game_params(&mut rng);
                let examples = play_one_game(n, c, &eval, mcts_cfg, &mut rng);

                let new_total =
                    example_count.fetch_add(examples.len(), Ordering::Relaxed) + examples.len();
                if let Some(pb) = &progress {
                    pb.inc(1);
                    pb.set_message(new_total.to_string());
                }
                examples
            })
            .collect()
    });

    if let Some(pb) = progress {
        pb.finish_and_clear();
    }

    let total: usize = results.iter().map(|v| v.len()).sum();
    let mut out = Vec::with_capacity(total);
    for v in results {
        out.extend(v);
    }
    out
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
            arena_capacity: blob_engine::mcts::DEFAULT_ARENA_CAPACITY,
        };
        let cfg = SelfPlayConfig {
            num_games: 2,
            num_threads: 2,
            iteration: 0,
            show_progress: false,
        };
        let examples = self_play_iteration(&path, &cfg, &mcts_cfg);
        assert!(!examples.is_empty());
    }
}
