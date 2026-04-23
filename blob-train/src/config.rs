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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalConfig {
    pub eval_games: usize,
    pub eval_interval: u64,
    pub eval_lookback: u64,
    pub bid_success_promotion_delta: f32,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            eval_games: 200,
            eval_interval: 5,
            eval_lookback: 20,
            bid_success_promotion_delta: 0.02,
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
}
