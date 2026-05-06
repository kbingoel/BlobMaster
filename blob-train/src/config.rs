//! Session 6.2 — composed training configuration, loadable from TOML.
//!
//! The three existing config structs (`MctsConfig`, `SelfPlayConfig`,
//! `TrainingLoopConfig`) each gained `#[derive(Serialize, Deserialize)]`
//! in Session 6.2; this module composes them plus an `EvalConfig` into a
//! single `TrainingConfig` that round-trips through TOML.

use std::fs;
use std::path::Path;

use blob_engine::mcts::MctsConfig;
use blob_nn::engine::SelfPlayConfig;
use blob_nn::training_loop::TrainingLoopConfig;
use serde::{Deserialize, Serialize};

fn default_eval_num_threads() -> usize {
    32
}

fn default_anchor_promotion_min_gap() -> u64 {
    25
}

fn default_anchor_promotion_lower95() -> f64 {
    // Mirrors `blob_nn::eval::EVAL_EARLY_STOP_HIGH` — the same Wilson-lower
    // band the early-stop logic already uses to call a chunk decisive.
    0.55
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalConfig {
    pub eval_games: usize,
    pub eval_interval: u64,
    pub eval_lookback: u64,
    pub bid_success_promotion_delta: f32,
    #[serde(default = "default_eval_num_threads")]
    pub eval_num_threads: usize,
    /// Minimum number of iterations between the current anchor and the
    /// candidate before auto-promotion can fire. Prevents flapping when an
    /// early eval happens to clear the Wilson lower bound on a small
    /// margin. The 2026-04-28 anchor run kept anchor=iter_31 across 195
    /// iters with no advancement; the new default is to advance whenever
    /// the candidate is `min_gap` iters newer AND the eval clears the
    /// lower-95 band. Set to a very large number (e.g. `u64::MAX`) to
    /// disable auto-promotion.
    #[serde(default = "default_anchor_promotion_min_gap")]
    pub anchor_promotion_min_gap: u64,
    /// Wilson lower-95 win-rate band a candidate must clear vs the current
    /// anchor before it is promoted to the new anchor. Defaults to 0.55,
    /// matching the early-stop band so the same chunk that triggered the
    /// stop is also strong enough to promote.
    #[serde(default = "default_anchor_promotion_lower95")]
    pub anchor_promotion_lower95: f64,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            // 192 = 6 × 32-thread batches — the last decision boundary
            // under the 32-way parallel CI-check design. 200 would force
            // a 7th batch of which only 8 games count toward the CI.
            eval_games: 192,
            eval_interval: 5,
            eval_lookback: 20,
            bid_success_promotion_delta: 0.02,
            eval_num_threads: default_eval_num_threads(),
            anchor_promotion_min_gap: default_anchor_promotion_min_gap(),
            anchor_promotion_lower95: default_anchor_promotion_lower95(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingConfig {
    #[serde(default = "default_training")]
    pub training: TrainingLoopConfig,
    #[serde(default)]
    pub self_play: SelfPlayConfig,
    #[serde(default)]
    pub mcts: MctsConfig,
    #[serde(default)]
    pub eval: EvalConfig,
}

fn default_training() -> TrainingLoopConfig {
    TrainingLoopConfig::default()
}

impl TrainingConfig {
    pub fn from_toml_str(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    #[allow(dead_code)]
    pub fn to_toml_string(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        let text = fs::read_to_string(path)?;
        Self::from_toml_str(&text)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_default_config() {
        let cfg = TrainingConfig::default();
        let s = cfg.to_toml_string().expect("serialize default");
        let back = TrainingConfig::from_toml_str(&s).expect("parse default");

        assert_eq!(cfg.training.buffer_capacity, back.training.buffer_capacity);
        assert_eq!(cfg.training.batch_size, back.training.batch_size);
        assert_eq!(
            cfg.training.total_iterations,
            back.training.total_iterations
        );
        assert_eq!(cfg.training.checkpoint_dir, back.training.checkpoint_dir);
        assert_eq!(cfg.self_play.num_games, back.self_play.num_games);
        assert_eq!(cfg.self_play.num_threads, back.self_play.num_threads);
        assert_eq!(cfg.mcts.num_determinizations, back.mcts.num_determinizations);
        assert_eq!(
            cfg.mcts.sims_per_determinization,
            back.mcts.sims_per_determinization
        );
        assert_eq!(cfg.eval.eval_games, back.eval.eval_games);
        assert_eq!(cfg.eval.eval_interval, back.eval.eval_interval);
    }

    #[test]
    fn device_tag_serializes_as_string() {
        let cfg = TrainingConfig::default();
        let s = cfg.to_toml_string().unwrap();
        assert!(s.contains("device = \"cpu\""), "got:\n{s}");
    }

    /// Session 7.4d: the optional `temperature_schedule` block must
    /// parse out of the snake-case `kind = "hard_step"` form documented
    /// in `config.sample.toml`, and `MctsConfig::temperature_at` must
    /// reflect the parsed values. Also verifies that omitting the block
    /// leaves the schedule as `None` (constant `temperature` regime).
    #[test]
    fn temperature_schedule_round_trips_through_toml() {
        use blob_engine::mcts::TemperatureSchedule;

        let toml = r#"
            [training]
            checkpoint_dir = "checkpoints"
            buffer_capacity = 500000
            batch_size = 512
            epochs_per_iteration = 10
            epoch_early_stop_rel = 0.005
            total_iterations = 15
            device = "cpu"

            [self_play]
            num_games = 4
            num_threads = 1
            iteration = 0
            show_progress = false

            [mcts]
            c_puct = 1.5
            num_determinizations = 5
            sims_per_determinization = 100
            min_sims_floor = 60
            temperature = 1.0
            arena_capacity = 4096
            target_batch = 5

            [mcts.temperature_schedule]
            kind = "hard_step"
            early = 1.0
            late = 0.1
            switch_at = 15

            [eval]
            eval_games = 192
            eval_interval = 5
            eval_lookback = 20
            bid_success_promotion_delta = 0.02
        "#;

        let cfg = TrainingConfig::from_toml_str(toml).expect("parse with schedule");
        match cfg.mcts.temperature_schedule {
            Some(TemperatureSchedule::HardStep {
                early,
                late,
                switch_at,
            }) => {
                assert!((early - 1.0).abs() < 1e-6);
                assert!((late - 0.1).abs() < 1e-6);
                assert_eq!(switch_at, 15);
            }
            None => panic!("expected HardStep schedule, got None"),
        }
        assert!((cfg.mcts.temperature_at(0) - 1.0).abs() < 1e-6);
        assert!((cfg.mcts.temperature_at(15) - 0.1).abs() < 1e-6);

        // Round-trip preserves the schedule.
        let serialized = cfg.to_toml_string().unwrap();
        let back = TrainingConfig::from_toml_str(&serialized).expect("re-parse");
        assert!(matches!(
            back.mcts.temperature_schedule,
            Some(TemperatureSchedule::HardStep { switch_at: 15, .. })
        ));

        // Default config has no schedule and falls back to constant.
        let default_cfg = TrainingConfig::default();
        assert!(default_cfg.mcts.temperature_schedule.is_none());
        assert!((default_cfg.mcts.temperature_at(50) - default_cfg.mcts.temperature).abs() < 1e-6);
    }
}
