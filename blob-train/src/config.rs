//! `pretrain`'s config file (gen-2.md §6 Phase 4). Every section and key
//! is optional and falls back to its default; unknown keys are an error,
//! so a stale config fails loudly. `pretrain.sample.toml` lists them all.

use blob_engine::TeacherConfig;
use blob_nn::learner::LearnerConfig;
use serde::{Deserialize, Serialize};

/// The teacher data and how it is split and fed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DataConfig {
    /// Seeds the teacher rounds, the network init and the batch sampling.
    pub seed: u64,
    /// Teacher rounds to play before training.
    pub rounds: u64,
    /// Threads that play them; 0 = every core.
    pub threads: usize,
    /// Share of rounds held out for validation (split by round).
    pub validation_fraction: f64,
    /// Threads that build batches while the GPU trains.
    pub loader_threads: usize,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self { seed: 1, rounds: 1_000_000, threads: 0, validation_fraction: 0.03, loader_threads: 4 }
    }
}

/// Metrics, held-out measurements and checkpoints, in learner steps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct LogConfig {
    /// A training row in `metrics.jsonl` every this many steps.
    pub every: u64,
    /// A held-out row every this many steps; 0 = only at the end.
    pub eval_every: u64,
    /// Examples in each periodic held-out set. The final measurement uses
    /// every validation example.
    pub eval_examples: usize,
    /// Save the checkpoint every this many steps; 0 = only at the end.
    pub checkpoint_every: u64,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self { every: 100, eval_every: 2_000, eval_examples: 20_000, checkpoint_every: 5_000 }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PretrainConfig {
    pub data: DataConfig,
    pub teacher: TeacherConfig,
    pub learner: LearnerConfig,
    pub log: LogConfig,
}

impl PretrainConfig {
    pub fn parse(text: &str) -> Result<Self, String> {
        let cfg: Self = toml::from_str(text).map_err(|e| e.to_string())?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.teacher.validate()?;
        self.learner.validate()?;
        let d = &self.data;
        if d.rounds == 0 || d.loader_threads == 0 {
            return Err("data.rounds and data.loader_threads must be > 0".into());
        }
        if !(d.validation_fraction > 0.0 && d.validation_fraction < 1.0) {
            return Err(format!("data.validation_fraction must be in (0, 1), got {}", d.validation_fraction));
        }
        if self.log.every == 0 || self.log.eval_examples == 0 {
            return Err("log.every and log.eval_examples must be > 0".into());
        }
        Ok(())
    }

    /// The config with every default filled in, for the run directory.
    pub fn to_toml(&self) -> String {
        toml::to_string(self).expect("config serializes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_config_is_the_default() {
        let sample = PretrainConfig::parse(include_str!("../pretrain.sample.toml")).unwrap();
        assert_eq!(sample, PretrainConfig::default());
    }

    #[test]
    fn empty_config_is_the_default_and_round_trips() {
        let cfg = PretrainConfig::parse("").unwrap();
        assert_eq!(cfg, PretrainConfig::default());
        assert_eq!(PretrainConfig::parse(&cfg.to_toml()).unwrap(), cfg);
    }

    #[test]
    fn unknown_keys_and_bad_values_fail() {
        assert!(PretrainConfig::parse("[data]\nround = 5\n").is_err());
        assert!(PretrainConfig::parse("[learner]\nepochs = 5\n").is_err());
        assert!(PretrainConfig::parse("[teacher.mix]\nplayers = [5]\nstart_cards = 7\nexponent = 1\n").is_err());
        assert!(PretrainConfig::parse("[gen1]\n").is_err());
        assert!(PretrainConfig::parse("[data]\nvalidation_fraction = 0.0\n").is_err());
        assert!(PretrainConfig::parse("[teacher]\nexplore = 2.0\n").is_err());
        let small = PretrainConfig::parse("[data]\nrounds = 10\n[learner]\ndevice = \"cpu\"\n").unwrap();
        assert_eq!((small.data.rounds, small.learner.device.as_str()), (10, "cpu"));
    }
}
