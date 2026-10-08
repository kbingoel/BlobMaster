//! The config files of `pretrain` (gen-2.md §6 Phase 4) and `train`
//! (Phase 5). Every section and key is optional and falls back to its
//! default, except that a mix section (`[teacher.mix]`, `[selfplay.mix]`)
//! needs `players` and `start_cards`. Unknown keys are an error, so a stale
//! config fails loudly. `pretrain.sample.toml` and `train.sample.toml` list
//! them all.

use blob_engine::rollout::RolloutConfig;
use blob_engine::{RoundMix, SelfPlayConfig, TeacherConfig};
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

/// `train`: the run as a whole.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RunConfig {
    /// Seeds the actors' rounds and the batch sampling.
    pub seed: u64,
    /// Hours of running (pauses excluded) before the final evaluation.
    pub hours: f64,
    /// Learner checkpoint the networks start from.
    pub init_checkpoint: String,
    /// The same networks exported: the actors' first model.
    pub init_model: String,
    /// Self-play threads.
    pub actors: usize,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            seed: 1,
            hours: 7.0,
            init_checkpoint: "checkpoints/pretrain-2026-10-06/checkpoint".into(),
            init_model: "checkpoints/pretrain-2026-10-06/model".into(),
            actors: 28,
        }
    }
}

/// `train`: the actors' rounds and search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields, default)]
pub struct SelfPlaySection {
    pub mix: RoundMix,
    pub search: SelfPlayConfig,
}

/// `train`: the replay buffer and the replay-ratio governor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ReplayConfig {
    /// Training examples kept (FIFO). The validation buffer keeps the
    /// same window: `capacity · f / (1 − f)`.
    pub capacity: usize,
    /// Share of rounds held out for validation (split by round).
    pub validation_fraction: f64,
    /// Training examples before the learner starts.
    pub min_examples: usize,
    /// Most V samples (learner steps × batch size) per training example
    /// produced; the learner waits when it gets ahead.
    pub replay_ratio: f64,
    /// Rounds per file of `replay/`: the actor process writes them, the
    /// learner tails them, a resume reloads them.
    pub chunk_rounds: usize,
    /// A partial file after this many seconds without one.
    pub chunk_secs: f64,
    /// Threads that build batches while the GPU trains.
    pub loader_threads: usize,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            capacity: 600_000,
            validation_fraction: 0.05,
            min_examples: 25_000,
            replay_ratio: 6.0,
            chunk_rounds: 50,
            chunk_secs: 60.0,
            loader_threads: 2,
        }
    }
}

/// `train`: rounds of P alone for V (`blob_engine::selfplay::policy_round`,
/// AlphaGo's value-net recipe). The actor process plays them on `actors`
/// extra threads; the learner keeps them in their own buffer and trains V
/// on them in V-only updates while P waits on the replay-ratio governor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ValueStreamConfig {
    /// Threads playing these rounds; 0 = off.
    pub actors: usize,
    /// States kept (FIFO); the validation buffer keeps the same window.
    pub capacity: usize,
    /// States before V-only updates start.
    pub min_examples: usize,
    /// Most V-only samples (updates × batch size) per training state
    /// produced.
    pub ratio: f64,
    /// A file of `replay-v/` every this many seconds (the learner reads and
    /// deletes it; a resume starts the buffer afresh).
    pub chunk_secs: f64,
}

impl Default for ValueStreamConfig {
    fn default() -> Self {
        Self { actors: 0, capacity: 3_000_000, min_examples: 100_000, ratio: 2.0, chunk_secs: 30.0 }
    }
}

/// `train`: policy iteration by rollouts (`blob_engine::rollout`, gen-2.md
/// §6 Phase 5, day 3). The actor process plays rounds of P on `actors`
/// threads and values `samples_per_round` decisions per round by playing
/// out every legal move on the real deal; P then trains on them
/// (`blob_nn::train::pi_loss`) instead of on search targets, one learner
/// step per batch under its own replay ratio. The state after each move,
/// with its outcome, trains V (`value_children`), in the steps and in
/// V-only updates under `value_stream.ratio`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RolloutStreamConfig {
    /// Threads playing rollout rounds; 0 = off (P trains on search targets).
    pub actors: usize,
    /// Decisions with a choice valued per round, drawn uniformly.
    pub samples_per_round: usize,
    /// τ of the bids and cards P plays in the rounds (1 = P's policy, 0 =
    /// its top move: the hidden cards then fall as behind P's own play).
    pub bid_temperature: f32,
    pub play_temperature: f32,
    /// Deals each valued bid is played out on: the real one and
    /// `bid_deals − 1` drawn from the bidder's view (weighted by the bids
    /// made); P's target averages them. Plays keep the real deal.
    pub bid_deals: usize,
    /// Candidate deals per drawn deal, weighted by the bids made.
    pub bid_candidates: u32,
    /// Chance that each seat of a round is played by rule bot 2 (in the
    /// round and its play-outs): P then learns against a mixed table, not
    /// only itself. Rule bot 2 is also the yardstick, so watch the rule bot
    /// and the start P too.
    pub rule_bot_2_share: f32,
    /// T of the loss, in utility units: P moves toward
    /// `π_ref · exp(E[u] / T)`.
    pub temperature: f64,
    /// Share of uniform in `π_ref` over the legal moves.
    pub epsilon: f64,
    /// Samples kept (FIFO); the validation buffer keeps the same window.
    pub capacity: usize,
    /// Training samples before the learner starts.
    pub min_examples: usize,
    /// Most P samples (learner steps × batch size × micro_batches) per
    /// training sample produced.
    pub ratio: f64,
    /// Batches whose gradients add up into one P update (gradient
    /// accumulation): one deal per sample is noisy, and a larger batch
    /// averages more of them per step.
    pub micro_batches: usize,
    /// V also learns the state after each valued move.
    pub value_children: bool,
    /// A file of `replay-pi/` every this many seconds (the learner reads and
    /// deletes it; a resume starts the buffers afresh).
    pub chunk_secs: f64,
}

impl Default for RolloutStreamConfig {
    fn default() -> Self {
        Self {
            actors: 0,
            samples_per_round: 2,
            bid_temperature: 0.0,
            play_temperature: 0.0,
            bid_deals: 1,
            bid_candidates: 2,
            rule_bot_2_share: 0.0,
            temperature: 0.05,
            epsilon: 0.03,
            capacity: 1_000_000,
            min_examples: 50_000,
            ratio: 4.0,
            micro_batches: 1,
            value_children: true,
            chunk_secs: 30.0,
        }
    }
}

impl RolloutStreamConfig {
    /// The actors' round settings.
    pub fn round(&self) -> RolloutConfig {
        RolloutConfig {
            samples_per_round: self.samples_per_round,
            bid_temperature: self.bid_temperature,
            play_temperature: self.play_temperature,
            bid_deals: self.bid_deals,
            bid_weighting: blob_engine::belief::BidWeighting { candidates: self.bid_candidates, noise: 0.1 },
            rule_bot_2_share: self.rule_bot_2_share,
        }
    }
}

/// `train`: the learner. The LR is constant after the warm-up: the run has
/// no fixed end (gen-2.md §5.6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TrainLearnerConfig {
    pub device: String,
    pub batch_size: usize,
    pub lr: f64,
    pub warmup_steps: u64,
    /// After a resume (AdamW restarts), the LR ramps up again over this
    /// many steps.
    pub resume_warmup_steps: u64,
    pub weight_decay: f64,
    pub augment: bool,
}

impl Default for TrainLearnerConfig {
    fn default() -> Self {
        Self {
            device: "cuda".into(),
            batch_size: 512,
            lr: 1e-4,
            warmup_steps: 300,
            resume_warmup_steps: 200,
            weight_decay: 1e-4,
            augment: true,
        }
    }
}

impl TrainLearnerConfig {
    /// The `Learner`'s settings: a constant LR after the warm-up.
    pub fn learner(&self) -> LearnerConfig {
        LearnerConfig {
            device: self.device.clone(),
            batch_size: self.batch_size,
            steps: 1 << 40,
            warmup_steps: self.warmup_steps,
            peak_lr: self.lr,
            min_lr: self.lr,
            weight_decay: self.weight_decay,
            augment: self.augment,
            policy_from: String::new(),
        }
    }
}

/// `train`: metrics, held-out measurements, publishing and checkpoints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TrainLogConfig {
    /// A training row every this many learner steps.
    pub every: u64,
    /// A held-out row every this many steps.
    pub eval_every: u64,
    /// Validation examples per held-out row (with as many training ones).
    pub eval_examples: usize,
    /// Export a model every this many steps: the actors switch to it and
    /// the evaluator benches it.
    pub publish_every: u64,
    /// `status.md` and a self-play row every this many seconds.
    pub status_secs: u64,
    /// Save the resume checkpoint every this many minutes (and at every
    /// publish).
    pub checkpoint_minutes: f64,
}

impl Default for TrainLogConfig {
    fn default() -> Self {
        Self {
            every: 50,
            eval_every: 200,
            eval_examples: 10_000,
            publish_every: 400,
            status_secs: 60,
            checkpoint_minutes: 10.0,
        }
    }
}

/// `train`: benches (bots never search; duplicate deals, gen-2.md §5.7).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct EvalConfig {
    /// Threads of the network-only benches at every publish (they run
    /// beside the actors).
    pub threads: usize,
    /// Deals of the network-only benches against rule bot 2 and the rule
    /// bot (default seed); 0 = skip.
    pub net_deals_rule_bot_2: usize,
    pub net_deals_rule_bot: usize,
    /// A search bench against rule bot 2 every this many hours (0 = only
    /// at the end). The actors park while it runs on every core.
    pub search_every_hours: f64,
    pub search_deals: usize,
    pub search_seed: u64,
    /// A per-deal file of the same search bench (`bench --per-deal-out`)
    /// to compare with, paired; empty = none.
    pub search_baseline: String,
    /// With every search bench, search against four copies of the same
    /// model's P (network only; `--seed search_seed`): the improvement margin
    /// where the targets are made, since P alone scores 0 there. Against
    /// rule bot 2, search models the opponents as P, which stops fitting as
    /// P leaves rule bot 2 behind. 0 = skip.
    pub search_vs_p_deals: usize,
    /// Network-only bench of every publish against four copies of the
    /// run's starting P (default seed): progress in self-play's own
    /// setting, where the start scores 0. 0 = skip.
    pub net_deals_vs_start: usize,
    /// The final search benches (against rule bot 2, and against P with
    /// `search_vs_p_deals`); false skips them.
    pub final_search: bool,
    /// At the end, also a search bench against the rule bot (default seed);
    /// 0 = skip.
    pub final_rule_bot_deals: usize,
    pub final_rule_bot_baseline: String,
    /// Teacher rounds for the probe set: the same states at every publish,
    /// for P's change (KL) and drift from rule bot 2, and V's error.
    pub probe_rounds: u64,
    /// A fixed V check at every held-out row: the last `value_rounds`
    /// validation rounds of this earlier run's `replay/` (unseen by every
    /// model trained in it); "" = none.
    pub value_rounds_from: String,
    pub value_rounds: usize,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            threads: 4,
            net_deals_rule_bot_2: 256,
            net_deals_rule_bot: 128,
            search_every_hours: 3.5,
            search_deals: 128,
            search_seed: 7,
            search_baseline: "checkpoints/pretrain-2026-10-06/bench4b/search-rb2-128-seed7.csv".into(),
            search_vs_p_deals: 64,
            net_deals_vs_start: 512,
            final_search: true,
            final_rule_bot_deals: 64,
            final_rule_bot_baseline: "checkpoints/pretrain-2026-10-06/bench4b/search-rb-64.csv".into(),
            probe_rounds: 600,
            value_rounds_from: "checkpoints/rl-2026-10-06".into(),
            value_rounds: 600,
        }
    }
}

/// `train`'s config file (`train.sample.toml`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TrainConfig {
    pub run: RunConfig,
    pub selfplay: SelfPlaySection,
    pub replay: ReplayConfig,
    pub learner: TrainLearnerConfig,
    pub log: TrainLogConfig,
    pub eval: EvalConfig,
    pub value_stream: ValueStreamConfig,
    pub rollout: RolloutStreamConfig,
}

impl TrainConfig {
    pub fn parse(text: &str) -> Result<Self, String> {
        let cfg: Self = toml::from_str(text).map_err(|e| e.to_string())?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.selfplay.mix.validate().map_err(|e| format!("selfplay.mix: {e:?}"))?;
        self.selfplay.search.validate()?;
        self.learner.learner().validate()?;
        let (r, run, log) = (&self.replay, &self.run, &self.log);
        if !(run.hours > 0.0) || run.init_checkpoint.is_empty() || run.init_model.is_empty() {
            return Err("run: hours must be > 0, init_checkpoint and init_model set".into());
        }
        let ro = &self.rollout;
        if run.actors == 0 && ro.actors == 0 {
            return Err("run.actors must be > 0 unless rollout.actors is".into());
        }
        if ro.actors > 0 {
            ro.round().validate()?;
            let ok = ro.capacity > 0 && ro.min_examples <= ro.capacity && ro.ratio > 0.0 && ro.chunk_secs > 0.0 && ro.micro_batches > 0;
            if !ok || !(ro.temperature > 0.0) || !(0.0..1.0).contains(&ro.epsilon) {
                return Err("rollout: capacity > 0, min_examples <= capacity, ratio, chunk_secs and temperature > 0, epsilon in [0, 1)".into());
            }
        }
        if r.capacity == 0 || r.min_examples == 0 || r.chunk_rounds == 0 || r.loader_threads == 0 {
            return Err("replay: capacity, min_examples, chunk_rounds and loader_threads must be > 0".into());
        }
        if !(r.chunk_secs > 0.0) {
            return Err("replay.chunk_secs must be > 0".into());
        }
        if r.min_examples > r.capacity {
            return Err("replay.min_examples must be <= replay.capacity".into());
        }
        if !(r.validation_fraction > 0.0 && r.validation_fraction < 0.5) {
            return Err(format!("replay.validation_fraction must be in (0, 0.5), got {}", r.validation_fraction));
        }
        if !(r.replay_ratio > 0.0) {
            return Err("replay.replay_ratio must be > 0".into());
        }
        if log.every == 0 || log.eval_every == 0 || log.publish_every == 0 || log.eval_examples == 0 || log.status_secs == 0 {
            return Err("log: every, eval_every, publish_every, eval_examples and status_secs must be > 0".into());
        }
        if !(log.checkpoint_minutes > 0.0) {
            return Err("log.checkpoint_minutes must be > 0".into());
        }
        let v = &self.value_stream;
        if (v.actors > 0 || (ro.actors > 0 && ro.value_children)) && (v.capacity == 0 || v.min_examples > v.capacity || !(v.ratio > 0.0) || !(v.chunk_secs > 0.0)) {
            return Err("value_stream: capacity > 0, min_examples <= capacity, ratio and chunk_secs > 0".into());
        }
        if self.eval.threads == 0 || self.eval.search_deals < 2 || !(self.eval.search_every_hours >= 0.0) {
            return Err("eval: threads > 0, search_deals >= 2, search_every_hours >= 0".into());
        }
        Ok(())
    }

    pub fn to_toml(&self) -> String {
        toml::to_string(self).expect("config serializes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_sample_config_is_the_default() {
        let sample = TrainConfig::parse(include_str!("../train.sample.toml")).unwrap();
        assert_eq!(sample, TrainConfig::default());
    }

    #[test]
    fn train_config_round_trips_and_rejects_unknown_keys() {
        let cfg = TrainConfig::parse("").unwrap();
        assert_eq!(cfg, TrainConfig::default());
        assert_eq!(TrainConfig::parse(&cfg.to_toml()).unwrap(), cfg);
        assert!(TrainConfig::parse("[replay]
ratio = 4
").is_err());
        assert!(TrainConfig::parse("[selfplay.search]
cpuct = 1
").is_err());
        assert!(TrainConfig::parse("[replay]
min_examples = 700000
").is_err());
        let c = TrainConfig::parse("[learner]
lr = 2e-4
[selfplay.search.play_budget]
determinizations = 2
sims_per_determinization = 10
").unwrap();
        assert_eq!((c.learner.lr, c.learner.batch_size), (2e-4, 512));
        assert_eq!(c.selfplay.search.play_budget.determinizations, 2);
        let l = c.learner.learner();
        assert_eq!((l.peak_lr, l.min_lr), (2e-4, 2e-4));
    }

    #[test]
    fn rollouts_may_replace_the_search_actors() {
        let c = TrainConfig::parse("[run]\nactors = 0\n[rollout]\nactors = 30\n").unwrap();
        assert_eq!((c.run.actors, c.rollout.actors, c.rollout.round().samples_per_round), (0, 30, 2));
        assert!(TrainConfig::parse("[run]\nactors = 0\n").is_err(), "no actors at all");
        assert!(TrainConfig::parse("[rollout]\nactors = 4\nepsilon = 1.0\n").is_err());
        assert!(TrainConfig::parse("[rollout]\nactors = 4\nsamples_per_round = 0\n").is_err());
        assert!(TrainConfig::parse("[rollout]\nactors = 4\ntemp = 0.1\n").is_err());
    }

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
        assert!(PretrainConfig::parse("[teacher.mix]\nstart_cards = 8\n").is_err(), "a mix names its players");
        let small = PretrainConfig::parse("[data]\nrounds = 10\n[learner]\ndevice = \"cpu\"\n").unwrap();
        assert_eq!((small.data.rounds, small.learner.device.as_str()), (10, "cpu"));
    }
}
