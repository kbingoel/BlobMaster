//! Session 5.4 — integrated training loop with diagnostics.
//!
//! Wires together the self-play engine (Session 5.3), the replay buffer
//! (Session 5.1), the tch-based training step (Session 3.4), and the ONNX
//! export hook (Session 3.5) into a single `run_iteration` entry point.
//!
//! Per-iteration diagnostics are logged as one JSON line per iteration to a
//! `metrics.jsonl` file next to the checkpoint directory and also echoed
//! via `tracing::info!`. Checkpoint retention follows the Session 5.4 rule:
//! every-5th iteration is evaluated and kept permanently; otherwise only
//! the most recent "rolling" checkpoint is retained (the previous rolling
//! checkpoint is deleted once the next iteration completes).

use std::fs;
use std::path::{Path, PathBuf};

use blob_engine::bidding::legal_bids;
use blob_engine::encoder::{encode, TOKEN_TYPE_HAND};
use blob_engine::mcts::MctsConfig;
use blob_engine::playing::legal_plays;
use blob_engine::replay::{BidBatch, PlayBatch, ReplayBuffer};
use blob_engine::state::GamePhase;
use rand::Rng;
use tch::{nn, Device, Kind, Tensor};

use crate::engine::{
    self_play_iteration, self_play_iteration_with_evaluator, SelfPlayConfig,
};
use crate::gpu_eval::{BatchCfg, BatchedGpuEvaluator};
use crate::heads::NUM_BIDS;
use crate::input::pad_batch;
use crate::model::BlobNet;
use crate::self_play::{DecisionStat, TrainingExample};
use crate::train::{
    build_optimizer, policy_cross_entropy, save_checkpoint as save_model_checkpoint, value_mse,
    LrSchedule, Phase, TrainBatch, VALUE_LOSS_COEF, GRAD_CLIP_MAX_NORM,
};

/// Target batch size sampled from the replay buffer each training step.
pub const DEFAULT_BATCH_SIZE: usize = 512;

/// Base per-iteration epoch count. May stop earlier if the loss plateaus
/// (see [`TrainingLoopConfig::epoch_early_stop_rel`]).
pub const DEFAULT_EPOCHS_PER_ITERATION: usize = 10;

/// Iterations that are comparison-evaluated (Section 6) and whose
/// checkpoints are kept permanently. Matches Session 5.4's retention plan.
pub const EVAL_CHECKPOINT_EVERY: u64 = 5;

fn default_total_iterations() -> u64 {
    1
}

/// Configuration for one training run.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingLoopConfig {
    pub checkpoint_dir: PathBuf,
    pub buffer_capacity: usize,
    pub batch_size: usize,
    pub epochs_per_iteration: usize,
    /// Stop epochs early when `(prev_loss - loss) / prev_loss` drops below
    /// this threshold. 0.0 disables early stop.
    pub epoch_early_stop_rel: f64,
    pub total_training_steps: i64,
    /// Session 7.1: number of `run_iteration` calls the `blobmaster-train
    /// train` driver should perform for a single invocation. The in-process
    /// training loop doesn't enforce this itself — it's read by the CLI.
    #[serde(default = "default_total_iterations")]
    pub total_iterations: u64,
    /// `tch::Device` is not serde-friendly, so round-trip as a string
    /// tag (`"cpu"` / `"cuda"` / `"cuda:N"` / `"mps"`).
    #[serde(with = "device_serde")]
    pub device: Device,
}

mod device_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use tch::Device;

    pub fn serialize<S: Serializer>(dev: &Device, s: S) -> Result<S::Ok, S::Error> {
        let tag = match dev {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(i) => format!("cuda:{i}"),
            Device::Mps => "mps".to_string(),
            Device::Vulkan => "vulkan".to_string(),
        };
        s.serialize_str(&tag)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Device, D::Error> {
        let tag = String::deserialize(d)?;
        let t = tag.to_ascii_lowercase();
        if t == "cpu" {
            Ok(Device::Cpu)
        } else if t == "mps" {
            Ok(Device::Mps)
        } else if t == "vulkan" {
            Ok(Device::Vulkan)
        } else if t == "cuda" {
            Ok(Device::Cuda(0))
        } else if let Some(rest) = t.strip_prefix("cuda:") {
            let i: usize = rest
                .parse()
                .map_err(|e| serde::de::Error::custom(format!("invalid cuda index: {e}")))?;
            Ok(Device::Cuda(i))
        } else {
            Err(serde::de::Error::custom(format!(
                "unknown device tag: {tag}"
            )))
        }
    }
}

impl Default for TrainingLoopConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: PathBuf::from("checkpoints"),
            buffer_capacity: 500_000,
            batch_size: DEFAULT_BATCH_SIZE,
            epochs_per_iteration: DEFAULT_EPOCHS_PER_ITERATION,
            epoch_early_stop_rel: 0.005,
            total_training_steps: 200_000,
            total_iterations: 1,
            device: Device::Cpu,
        }
    }
}

/// Per-iteration diagnostic metrics (development-plan §5.4).
#[derive(Debug, Clone, Default)]
pub struct IterationMetrics {
    pub iteration: u64,
    pub learning_rate: f64,
    pub bid_policy_loss: f64,
    pub play_policy_loss: f64,
    pub value_loss: f64,
    pub combined_loss: f64,
    pub bid_top1_accuracy: f64,
    pub play_top1_accuracy: f64,
    pub policy_kl_divergence: f64,
    pub value_mean: f64,
    pub value_variance: f64,
    pub visit_entropy_mean: f64,
    pub top1_visit_share_mean: f64,
    /// Grouped by module prefix (input / transformer.layers.N / *_head).
    /// Logged as `"grad_norm.<group>"` so a missing-gradient layer is
    /// visible in metrics.jsonl.
    pub grad_norms: Vec<(String, f64)>,
    pub num_epochs_run: usize,
    pub num_nn_evaluations: u64,
    pub examples_generated: usize,
    pub buffer_len: usize,
    /// Session 7.1 — P10/P50/P90 of `DecisionStat::signal_ratio`, bucketed
    /// by branching factor (`num_legal`): `bucket_low` = `num_legal ≤ 3`,
    /// `bucket_mid` = `4..=7`, `bucket_high` = `> 7`. Missing buckets emit
    /// NaN so downstream tooling can distinguish "no data" from "zero".
    pub signal_p10_low: f64,
    pub signal_p50_low: f64,
    pub signal_p90_low: f64,
    pub signal_p10_mid: f64,
    pub signal_p50_mid: f64,
    pub signal_p90_mid: f64,
    pub signal_p10_high: f64,
    pub signal_p50_high: f64,
    pub signal_p90_high: f64,
    pub num_decisions: usize,
}

impl IterationMetrics {
    /// Serialize as a single JSON object followed by a newline. Hand-rolled
    /// to avoid a `serde_json` dependency for what is a flat struct.
    pub fn to_json_line(&self) -> String {
        let mut s = String::new();
        s.push('{');
        macro_rules! kv {
            ($k:expr, $v:expr) => {{
                if s.len() > 1 {
                    s.push(',');
                }
                s.push('"');
                s.push_str($k);
                s.push_str("\":");
                s.push_str(&format!("{}", $v));
            }};
        }
        kv!("iteration", self.iteration);
        kv!("learning_rate", json_f64(self.learning_rate));
        kv!("bid_policy_loss", json_f64(self.bid_policy_loss));
        kv!("play_policy_loss", json_f64(self.play_policy_loss));
        kv!("value_loss", json_f64(self.value_loss));
        kv!("combined_loss", json_f64(self.combined_loss));
        kv!("bid_top1_accuracy", json_f64(self.bid_top1_accuracy));
        kv!("play_top1_accuracy", json_f64(self.play_top1_accuracy));
        kv!("policy_kl_divergence", json_f64(self.policy_kl_divergence));
        kv!("value_mean", json_f64(self.value_mean));
        kv!("value_variance", json_f64(self.value_variance));
        kv!("visit_entropy_mean", json_f64(self.visit_entropy_mean));
        kv!("top1_visit_share_mean", json_f64(self.top1_visit_share_mean));
        kv!("num_epochs_run", self.num_epochs_run);
        kv!("num_nn_evaluations", self.num_nn_evaluations);
        kv!("examples_generated", self.examples_generated);
        kv!("buffer_len", self.buffer_len);
        kv!("num_decisions", self.num_decisions);
        kv!("signal_p10_low", json_f64(self.signal_p10_low));
        kv!("signal_p50_low", json_f64(self.signal_p50_low));
        kv!("signal_p90_low", json_f64(self.signal_p90_low));
        kv!("signal_p10_mid", json_f64(self.signal_p10_mid));
        kv!("signal_p50_mid", json_f64(self.signal_p50_mid));
        kv!("signal_p90_mid", json_f64(self.signal_p90_mid));
        kv!("signal_p10_high", json_f64(self.signal_p10_high));
        kv!("signal_p50_high", json_f64(self.signal_p50_high));
        kv!("signal_p90_high", json_f64(self.signal_p90_high));
        s.push(',');
        s.push_str("\"grad_norms\":{");
        for (i, (k, v)) in self.grad_norms.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push('"');
            s.push_str(k);
            s.push_str("\":");
            s.push_str(&json_f64(*v));
        }
        s.push('}');
        s.push('}');
        s.push('\n');
        s
    }
}

fn percentile_sorted(sorted: &[f32], pct: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = (pct / 100.0) * (sorted.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f64;
    let a = sorted[lo] as f64;
    let b = sorted[hi] as f64;
    a + (b - a) * frac
}

fn fold_decision_stats(metrics: &mut IterationMetrics, stats: &[DecisionStat]) {
    metrics.num_decisions = stats.len();
    let mut low: Vec<f32> = Vec::new();
    let mut mid: Vec<f32> = Vec::new();
    let mut high: Vec<f32> = Vec::new();
    for s in stats {
        let r = s.signal_ratio;
        if s.num_legal <= 3 {
            low.push(r);
        } else if s.num_legal <= 7 {
            mid.push(r);
        } else {
            high.push(r);
        }
    }
    for v in [&mut low, &mut mid, &mut high] {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }
    metrics.signal_p10_low = percentile_sorted(&low, 10.0);
    metrics.signal_p50_low = percentile_sorted(&low, 50.0);
    metrics.signal_p90_low = percentile_sorted(&low, 90.0);
    metrics.signal_p10_mid = percentile_sorted(&mid, 10.0);
    metrics.signal_p50_mid = percentile_sorted(&mid, 50.0);
    metrics.signal_p90_mid = percentile_sorted(&mid, 90.0);
    metrics.signal_p10_high = percentile_sorted(&high, 10.0);
    metrics.signal_p50_high = percentile_sorted(&high, 50.0);
    metrics.signal_p90_high = percentile_sorted(&high, 90.0);
}

fn json_f64(x: f64) -> String {
    if x.is_finite() {
        format!("{x}")
    } else {
        "null".to_string()
    }
}

/// Convert a `BidBatch` from the replay buffer into a `TrainBatch` ready
/// for `train_step`.
pub fn bid_train_batch(batch: &BidBatch, device: Device) -> Option<TrainBatch> {
    let n = batch.states.len();
    if n == 0 {
        return None;
    }
    let encs: Vec<_> = batch.states.iter().map(|s| encode(s, s.current_player)).collect();
    let input = pad_batch(&encs, device);

    let mut mask = vec![false; n * NUM_BIDS as usize];
    for (row, state) in batch.states.iter().enumerate() {
        let legal = legal_bids(state);
        for b in 0..NUM_BIDS as usize {
            if (legal >> b) & 1 == 1 {
                mask[row * NUM_BIDS as usize + b] = true;
            }
        }
    }
    let legal_mask =
        Tensor::from_slice(&mask).view([n as i64, NUM_BIDS]).to_device(device);
    let policy_target = Tensor::from_slice(&batch.policies)
        .view([n as i64, NUM_BIDS])
        .to_device(device);
    let value_target = Tensor::from_slice(&batch.values)
        .view([n as i64])
        .to_device(device);

    Some(TrainBatch {
        input,
        phase: Phase::Bidding,
        legal_mask,
        policy_target,
        value_target,
    })
}

/// Convert a `PlayBatch` into a `TrainBatch`.
///
/// Play-head policies in the replay buffer are indexed by **hand position**
/// (0..hand_size). The play head outputs one score per **sequence
/// position**. This helper scatters each row's hand-indexed policy onto
/// the sequence positions whose `token_types[i] == TOKEN_TYPE_HAND`, in
/// encoder emission order (matches `EncodedState::hand_card_indices`).
pub fn play_train_batch(batch: &PlayBatch, device: Device) -> Option<TrainBatch> {
    let n = batch.states.len();
    if n == 0 {
        return None;
    }
    let encs: Vec<_> = batch.states.iter().map(|s| encode(s, s.current_player)).collect();
    let input = pad_batch(&encs, device);
    let seq_len = input.attention_mask.size()[1] as usize;

    let mut mask = vec![false; n * seq_len];
    let mut target = vec![0.0f32; n * seq_len];
    for (row, (state, enc)) in batch.states.iter().zip(encs.iter()).enumerate() {
        let legal = legal_plays(state);
        let mut hand_slot = 0usize;
        for (seq_i, &tt) in enc.token_types.iter().enumerate() {
            if tt != TOKEN_TYPE_HAND {
                continue;
            }
            let card_idx = enc.hand_card_indices[hand_slot];
            let pol_base = row * batch.max_hand_size;
            // `max_hand_size` in a PlayBatch is the largest *nonzero* hand
            // position observed — may be smaller than the encoder's actual
            // hand-token count. Treat out-of-range positions as zero prob.
            if hand_slot < batch.max_hand_size {
                target[row * seq_len + seq_i] = batch.policies[pol_base + hand_slot];
            }
            if (legal >> card_idx) & 1 == 1 {
                mask[row * seq_len + seq_i] = true;
            }
            hand_slot += 1;
        }
    }

    let legal_mask =
        Tensor::from_slice(&mask).view([n as i64, seq_len as i64]).to_device(device);
    let policy_target = Tensor::from_slice(&target)
        .view([n as i64, seq_len as i64])
        .to_device(device);
    let value_target = Tensor::from_slice(&batch.values)
        .view([n as i64])
        .to_device(device);

    Some(TrainBatch {
        input,
        phase: Phase::Playing,
        legal_mask,
        policy_target,
        value_target,
    })
}

/// Classify a `VarStore` variable name into a diagnostic group.
///
/// Groups match the parameter layout in `BlobNet::new`:
/// - `input/...`, `value_head/...`, `play_head/...`, `bid_head/...`
/// - `transformer/layers/N/...` → one group per layer (catches a dead
///   transformer layer instantly).
fn grad_norm_group(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("transformer.layers.") {
        if let Some(layer) = rest.split('.').next() {
            return format!("transformer.layer_{layer}");
        }
    }
    for top in [
        "input",
        "transformer",
        "play_head",
        "bid_head",
        "value_head",
    ] {
        if name.starts_with(top) {
            return top.to_string();
        }
    }
    "other".to_string()
}

fn aggregate_grad_norms(vs: &nn::VarStore) -> Vec<(String, f64)> {
    use std::collections::BTreeMap;
    let mut sqsum: BTreeMap<String, f64> = BTreeMap::new();
    for (name, var) in vs.variables() {
        if !var.requires_grad() {
            continue;
        }
        let g = var.grad();
        if !g.defined() {
            continue;
        }
        let n2: f64 = g.square().sum(Kind::Float).double_value(&[]);
        let key = grad_norm_group(&name);
        *sqsum.entry(key).or_insert(0.0) += n2;
    }
    sqsum
        .into_iter()
        .map(|(k, v)| (k, v.sqrt()))
        .collect()
}

/// One forward/backward/step. Returns losses **and** per-group gradient
/// norms captured between `backward` and `step`.
fn train_step_with_grad_norms(
    model: &BlobNet,
    vs: &nn::VarStore,
    optimizer: &mut nn::Optimizer,
    batch: &TrainBatch,
) -> (f64, f64, f64, Vec<(String, f64)>) {
    let (policy_probs, value_pred) = match batch.phase {
        Phase::Bidding => model.forward_bid(&batch.input, &batch.legal_mask, true),
        Phase::Playing => model.forward_play(&batch.input, &batch.legal_mask, true),
    };
    let policy_loss = policy_cross_entropy(&policy_probs, &batch.policy_target);
    let value_loss = value_mse(&value_pred, &batch.value_target);
    let total = &policy_loss + VALUE_LOSS_COEF * &value_loss;

    optimizer.zero_grad();
    total.backward();
    let grad_norms = aggregate_grad_norms(vs);
    optimizer.clip_grad_norm(GRAD_CLIP_MAX_NORM);
    optimizer.step();

    (
        policy_loss.double_value(&[]),
        value_loss.double_value(&[]),
        total.double_value(&[]),
        grad_norms,
    )
}

fn top1_accuracy(pred_probs: &Tensor, target: &Tensor) -> f64 {
    let pred_idx = pred_probs.argmax(-1, false);
    let tgt_idx = target.argmax(-1, false);
    let eq = pred_idx.eq_tensor(&tgt_idx).to_kind(Kind::Float);
    eq.mean(Kind::Float).double_value(&[])
}

fn kl_divergence(target: &Tensor, pred: &Tensor) -> f64 {
    // KL(target || pred) = Σ t · (log t - log p). Illegal actions have t=0
    // so 0·anything contributes zero.
    let eps = 1e-8;
    let log_t = (target + eps).log();
    let log_p = (pred + eps).log();
    let per_sample = (target * (log_t - log_p)).sum_dim_intlist(&[-1i64][..], false, Kind::Float);
    per_sample.mean(Kind::Float).double_value(&[])
}

/// Training-loop state. Owns the `VarStore`, model, optimizer, replay
/// buffer, and iteration/step counters.
pub struct TrainingLoop {
    pub cfg: TrainingLoopConfig,
    pub vs: nn::VarStore,
    pub model: BlobNet,
    pub optimizer: nn::Optimizer,
    pub buffer: ReplayBuffer,
    pub lr_schedule: LrSchedule,
    pub iteration: u64,
    pub global_step: i64,
}

impl TrainingLoop {
    pub fn new(cfg: TrainingLoopConfig) -> Self {
        let vs = nn::VarStore::new(cfg.device);
        let model = BlobNet::new(&vs.root());
        let optimizer = build_optimizer(&vs).expect("build optimizer");
        let buffer = ReplayBuffer::new(cfg.buffer_capacity);
        let lr_schedule = LrSchedule::new(cfg.total_training_steps);
        Self {
            cfg,
            vs,
            model,
            optimizer,
            buffer,
            lr_schedule,
            iteration: 0,
            global_step: 0,
        }
    }

    fn iteration_dir(&self, iter: u64) -> PathBuf {
        self.cfg.checkpoint_dir.join(format!("iter_{iter:06}"))
    }

    fn metrics_path(&self) -> PathBuf {
        self.cfg.checkpoint_dir.join("metrics.jsonl")
    }

    /// Sample a batch from the buffer, convert to tensors, and run a
    /// single step. Returns `None` if the buffer is empty.
    fn train_one_step<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        accumulators: &mut LossAccumulators,
    ) -> Option<()> {
        if self.buffer.is_empty() {
            return None;
        }
        let lr = self.lr_schedule.lr(self.global_step);
        self.optimizer.set_lr(lr);
        accumulators.last_lr = lr;

        let (bid, play) = self.buffer.sample_batch(self.cfg.batch_size, rng);

        if let Some(tb) = bid_train_batch(&bid, self.cfg.device) {
            let (pp, vl, _tot, gnorms) =
                train_step_with_grad_norms(&self.model, &self.vs, &mut self.optimizer, &tb);
            accumulators.add_bid(&self.model, &tb, pp, vl);
            accumulators.add_grad_norms(gnorms);
            self.global_step += 1;
        }
        if let Some(tb) = play_train_batch(&play, self.cfg.device) {
            let (pp, vl, _tot, gnorms) =
                train_step_with_grad_norms(&self.model, &self.vs, &mut self.optimizer, &tb);
            accumulators.add_play(&self.model, &tb, pp, vl);
            accumulators.add_grad_norms(gnorms);
            self.global_step += 1;
        }
        Some(())
    }

    /// Train for up to `epochs_per_iteration` passes, each consisting of
    /// `ceil(buffer_len / batch_size)` sampled batches. Early-stops when
    /// relative improvement drops below the configured threshold.
    pub fn train_phase<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        examples_from_self_play: &[TrainingExample],
    ) -> IterationMetrics {
        let mut acc = LossAccumulators::default();
        acc.add_selfplay_signals(examples_from_self_play);

        if self.buffer.is_empty() {
            return acc.finalize(self.iteration, 0, 0, self.buffer.len());
        }

        let steps_per_epoch =
            ((self.buffer.len() + self.cfg.batch_size - 1) / self.cfg.batch_size).max(1);

        let mut prev_epoch_loss = f64::INFINITY;
        let mut epochs_run = 0usize;
        for _epoch in 0..self.cfg.epochs_per_iteration {
            let start_combined = acc.combined_count;
            let start_combined_sum = acc.combined_sum;
            for _ in 0..steps_per_epoch {
                self.train_one_step(rng, &mut acc);
            }
            epochs_run += 1;
            let epoch_combined = acc.combined_sum - start_combined_sum;
            let epoch_count = (acc.combined_count - start_combined).max(1) as f64;
            let epoch_loss = epoch_combined / epoch_count;
            let improved = prev_epoch_loss - epoch_loss;
            let rel = if prev_epoch_loss.is_finite() && prev_epoch_loss > 1e-8 {
                improved / prev_epoch_loss
            } else {
                f64::INFINITY
            };
            if rel < self.cfg.epoch_early_stop_rel && prev_epoch_loss.is_finite() {
                break;
            }
            prev_epoch_loss = epoch_loss;
        }

        acc.finalize(
            self.iteration,
            epochs_run,
            examples_from_self_play.len(),
            self.buffer.len(),
        )
    }

    /// Run one full training iteration driven by a `BatchedGpuEvaluator`
    /// built from the current in-memory weights — bypasses ONNX export on
    /// the hot path (the export still happens post-iteration so deployment
    /// / anchor-comparison paths can consume an `.onnx`).
    pub fn run_iteration_gpu<R, F>(
        &mut self,
        rng: &mut R,
        self_play_cfg: &SelfPlayConfig,
        mcts_cfg: &MctsConfig,
        inference_device: Device,
        batch_cfg: BatchCfg,
        on_export: F,
    ) -> std::io::Result<IterationMetrics>
    where
        R: Rng + ?Sized,
        F: FnOnce(&Path, &Path) -> std::io::Result<()>,
    {
        let eval = BatchedGpuEvaluator::new(&self.vs, inference_device, batch_cfg)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        let started = std::time::Instant::now();
        let (examples, decision_stats) =
            self_play_iteration_with_evaluator(&eval, self_play_cfg, mcts_cfg);
        let sp_elapsed = started.elapsed();
        tracing::info!(
            iteration = self.iteration,
            examples = examples.len(),
            self_play_secs = sp_elapsed.as_secs_f64(),
            stats = %eval.stats().summary(),
            "gpu self-play complete"
        );
        drop(eval);

        for ex in &examples {
            if matches!(ex.phase, GamePhase::Bidding | GamePhase::Playing) {
                self.buffer.push(ex.state, ex.policy.clone(), ex.value, ex.phase);
            }
        }

        let mut metrics = self.train_phase(rng, &examples);
        fold_decision_stats(&mut metrics, &decision_stats);

        let dir = self.iteration_dir(self.iteration);
        save_model_checkpoint(&self.vs, self.iteration, &dir)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        self.append_metrics_line(&metrics)?;
        self.append_decision_stats(&decision_stats)?;

        let new_onnx_path = dir.join("model.onnx");
        on_export(&dir.join("model.ot"), &new_onnx_path)?;

        self.prune_checkpoints()?;
        self.iteration += 1;
        Ok(metrics)
    }

    /// Run one full training iteration: self-play → buffer → training →
    /// diagnostics → checkpoint. The ONNX export step is delegated to
    /// `on_export`, which receives the freshly-saved checkpoint directory
    /// and is expected to produce `model.onnx` inside it (this is the
    /// artifact used by the next iteration's self-play).
    pub fn run_iteration<R, F>(
        &mut self,
        rng: &mut R,
        self_play_cfg: &SelfPlayConfig,
        mcts_cfg: &MctsConfig,
        onnx_model_path: &Path,
        on_export: F,
    ) -> std::io::Result<IterationMetrics>
    where
        R: Rng + ?Sized,
        F: FnOnce(&Path, &Path) -> std::io::Result<()>,
    {
        let (examples, decision_stats) =
            self_play_iteration(onnx_model_path, self_play_cfg, mcts_cfg);
        for ex in &examples {
            if matches!(ex.phase, GamePhase::Bidding | GamePhase::Playing) {
                self.buffer.push(ex.state, ex.policy.clone(), ex.value, ex.phase);
            }
        }

        let mut metrics = self.train_phase(rng, &examples);
        fold_decision_stats(&mut metrics, &decision_stats);

        let dir = self.iteration_dir(self.iteration);
        save_model_checkpoint(&self.vs, self.iteration, &dir)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        self.append_metrics_line(&metrics)?;
        self.append_decision_stats(&decision_stats)?;

        let new_onnx_path = dir.join("model.onnx");
        on_export(&dir.join("model.ot"), &new_onnx_path)?;

        self.prune_checkpoints()?;
        self.iteration += 1;
        Ok(metrics)
    }

    fn decision_stats_path(&self) -> PathBuf {
        self.cfg.checkpoint_dir.join("decision_stats.jsonl")
    }

    fn append_decision_stats(&self, stats: &[DecisionStat]) -> std::io::Result<()> {
        if stats.is_empty() {
            return Ok(());
        }
        fs::create_dir_all(&self.cfg.checkpoint_dir)?;
        use std::io::Write;
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.decision_stats_path())?;
        for s in stats {
            let phase = match s.phase {
                GamePhase::Bidding => "bidding",
                GamePhase::Playing => "playing",
                GamePhase::Scoring => "scoring",
                GamePhase::Complete => "complete",
            };
            writeln!(
                f,
                "{{\"iteration\":{},\"phase\":\"{}\",\"num_legal\":{},\"sims_used\":{},\"signal_ratio\":{}}}",
                self.iteration,
                phase,
                s.num_legal,
                s.sims_used,
                json_f64(s.signal_ratio as f64)
            )?;
        }
        Ok(())
    }

    fn append_metrics_line(&self, m: &IterationMetrics) -> std::io::Result<()> {
        fs::create_dir_all(&self.cfg.checkpoint_dir)?;
        let line = m.to_json_line();
        use std::io::Write;
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.metrics_path())?;
        f.write_all(line.as_bytes())?;
        Ok(())
    }

    /// Keep every-5th checkpoint permanently and the most recent rolling
    /// one; delete previous rolling checkpoints. "Every-5th" here means
    /// `iteration % EVAL_CHECKPOINT_EVERY == 0`.
    fn prune_checkpoints(&self) -> std::io::Result<()> {
        let dir = &self.cfg.checkpoint_dir;
        if !dir.exists() {
            return Ok(());
        }
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let name = match path.file_name().and_then(|s| s.to_str()) {
                Some(n) => n,
                None => continue,
            };
            let iter_num = match name.strip_prefix("iter_").and_then(|s| s.parse::<u64>().ok()) {
                Some(n) => n,
                None => continue,
            };
            if iter_num == self.iteration {
                continue; // just-saved rolling checkpoint
            }
            if iter_num % EVAL_CHECKPOINT_EVERY == 0 {
                continue; // evaluated — keep permanently
            }
            let _ = fs::remove_dir_all(&path);
        }
        Ok(())
    }

    /// Scan `checkpoint_dir` for `iter_*` directories and load weights
    /// from the highest-numbered one. Returns the iteration resumed from,
    /// or `None` if no checkpoint exists.
    /// Session 7.1 alias — development-plan refers to `try_resume`. Behavior
    /// is identical to `resume_from_latest`.
    pub fn try_resume(&mut self) -> std::io::Result<Option<u64>> {
        self.resume_from_latest()
    }

    pub fn resume_from_latest(&mut self) -> std::io::Result<Option<u64>> {
        let dir = &self.cfg.checkpoint_dir;
        if !dir.exists() {
            return Ok(None);
        }
        let mut best: Option<(u64, PathBuf)> = None;
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let p = entry.path();
            let name = match p.file_name().and_then(|s| s.to_str()) {
                Some(n) => n,
                None => continue,
            };
            if let Some(n) = name.strip_prefix("iter_").and_then(|s| s.parse::<u64>().ok()) {
                if best.as_ref().map(|(m, _)| n > *m).unwrap_or(true) {
                    best = Some((n, p));
                }
            }
        }
        let Some((iter, path)) = best else {
            return Ok(None);
        };
        self.vs
            .load(path.join("model.ot"))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        self.iteration = iter + 1;
        Ok(Some(iter))
    }
}

/// Running sums for a single iteration. Broken out so `train_phase` can
/// compose multiple passes without losing totals.
#[derive(Default)]
struct LossAccumulators {
    bid_policy_sum: f64,
    bid_count: usize,
    play_policy_sum: f64,
    play_count: usize,
    value_sum: f64,
    value_count: usize,
    combined_sum: f64,
    combined_count: usize,
    last_lr: f64,

    bid_top1_sum: f64,
    play_top1_sum: f64,
    kl_sum: f64,
    kl_count: usize,
    value_preds: Vec<f64>,

    grad_norms: std::collections::BTreeMap<String, (f64, usize)>,

    visit_entropy_sum: f64,
    top1_share_sum: f64,
    signal_count: usize,
    num_nn_evals: u64,
}

impl LossAccumulators {
    fn add_bid(&mut self, model: &BlobNet, tb: &TrainBatch, policy_loss: f64, value_loss: f64) {
        self.bid_policy_sum += policy_loss;
        self.bid_count += 1;
        // Count one forward pass per example in this batch (Session 6.1
        // carry-over fix — the counter was declared but never incremented).
        self.num_nn_evals += tb.input.features.size()[0] as u64;
        self.value_sum += value_loss;
        self.value_count += 1;
        let combined = policy_loss + VALUE_LOSS_COEF * value_loss;
        self.combined_sum += combined;
        self.combined_count += 1;

        // Diagnostic forward pass (eval mode, no backward).
        let (probs, value_pred) = tch::no_grad(|| {
            model.forward_bid(&tb.input, &tb.legal_mask, false)
        });
        self.bid_top1_sum += top1_accuracy(&probs, &tb.policy_target);
        self.kl_sum += kl_divergence(&tb.policy_target, &probs);
        self.kl_count += 1;
        self.record_value_preds(&value_pred);
    }

    fn add_play(&mut self, model: &BlobNet, tb: &TrainBatch, policy_loss: f64, value_loss: f64) {
        self.play_policy_sum += policy_loss;
        self.play_count += 1;
        self.num_nn_evals += tb.input.features.size()[0] as u64;
        self.value_sum += value_loss;
        self.value_count += 1;
        let combined = policy_loss + VALUE_LOSS_COEF * value_loss;
        self.combined_sum += combined;
        self.combined_count += 1;

        let (probs, value_pred) = tch::no_grad(|| {
            model.forward_play(&tb.input, &tb.legal_mask, false)
        });
        self.play_top1_sum += top1_accuracy(&probs, &tb.policy_target);
        self.kl_sum += kl_divergence(&tb.policy_target, &probs);
        self.kl_count += 1;
        self.record_value_preds(&value_pred);
    }

    fn record_value_preds(&mut self, value_pred: &Tensor) {
        let v: Vec<f32> = value_pred.to_kind(Kind::Float).try_into().unwrap_or_default();
        self.value_preds.extend(v.iter().map(|x| *x as f64));
    }

    fn add_grad_norms(&mut self, gn: Vec<(String, f64)>) {
        for (k, v) in gn {
            let e = self.grad_norms.entry(k).or_insert((0.0, 0));
            e.0 += v;
            e.1 += 1;
        }
    }

    fn add_selfplay_signals(&mut self, examples: &[TrainingExample]) {
        // MCTS visit entropy & top-1 share per example, reconstructed from
        // the sparse policy (already a probability distribution).
        for ex in examples {
            let mut h = 0.0f64;
            let mut top = 0.0f64;
            for &(_, p) in &ex.policy {
                let p = p as f64;
                if p > 0.0 {
                    h -= p * p.ln();
                }
                if p > top {
                    top = p;
                }
            }
            self.visit_entropy_sum += h;
            self.top1_share_sum += top;
            self.signal_count += 1;
        }
    }

    fn finalize(
        self,
        iteration: u64,
        num_epochs_run: usize,
        examples_generated: usize,
        buffer_len: usize,
    ) -> IterationMetrics {
        let bidc = self.bid_count.max(1) as f64;
        let playc = self.play_count.max(1) as f64;
        let valc = self.value_count.max(1) as f64;
        let combc = self.combined_count.max(1) as f64;
        let klc = self.kl_count.max(1) as f64;
        let sigc = self.signal_count.max(1) as f64;

        let v_mean = if self.value_preds.is_empty() {
            0.0
        } else {
            self.value_preds.iter().sum::<f64>() / self.value_preds.len() as f64
        };
        let v_var = if self.value_preds.len() < 2 {
            0.0
        } else {
            let m = v_mean;
            self.value_preds.iter().map(|x| (x - m).powi(2)).sum::<f64>()
                / self.value_preds.len() as f64
        };

        let grad_norms = self
            .grad_norms
            .into_iter()
            .map(|(k, (s, n))| (k, s / n.max(1) as f64))
            .collect();

        IterationMetrics {
            iteration,
            learning_rate: self.last_lr,
            bid_policy_loss: self.bid_policy_sum / bidc,
            play_policy_loss: self.play_policy_sum / playc,
            value_loss: self.value_sum / valc,
            combined_loss: self.combined_sum / combc,
            bid_top1_accuracy: self.bid_top1_sum / bidc,
            play_top1_accuracy: self.play_top1_sum / playc,
            policy_kl_divergence: self.kl_sum / klc,
            value_mean: v_mean,
            value_variance: v_var,
            visit_entropy_mean: self.visit_entropy_sum / sigc,
            top1_visit_share_mean: self.top1_share_sum / sigc,
            grad_norms,
            num_epochs_run,
            num_nn_evaluations: self.num_nn_evals,
            examples_generated,
            buffer_len,
            num_decisions: 0,
            signal_p10_low: f64::NAN,
            signal_p50_low: f64::NAN,
            signal_p90_low: f64::NAN,
            signal_p10_mid: f64::NAN,
            signal_p50_mid: f64::NAN,
            signal_p90_mid: f64::NAN,
            signal_p10_high: f64::NAN,
            signal_p50_high: f64::NAN,
            signal_p90_high: f64::NAN,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blob_engine::dealing::deal;
    use blob_engine::game::new_game;
    use blob_engine::state::GamePhase;
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use smallvec::smallvec;

    fn dummy_play_example(seed: u64) -> TrainingExample {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut s = new_game(4, 5).unwrap();
        deal(&mut s, &mut rng);
        s.game_phase = GamePhase::Playing as u8;
        TrainingExample {
            state: s,
            policy: smallvec![(0u8, 0.5f32), (1, 0.5)],
            value: 0.0,
            phase: GamePhase::Playing,
            perspective: s.current_player,
        }
    }

    fn dummy_bid_example(seed: u64) -> TrainingExample {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut s = new_game(4, 5).unwrap();
        deal(&mut s, &mut rng);
        s.game_phase = GamePhase::Bidding as u8;
        TrainingExample {
            state: s,
            policy: smallvec![(0u8, 0.5f32), (1u8, 0.5)],
            value: 0.1,
            phase: GamePhase::Bidding,
            perspective: s.current_player,
        }
    }

    #[test]
    fn grad_norm_groups_split_transformer_layers() {
        assert_eq!(grad_norm_group("transformer.layers.0.attn.qkv.weight"), "transformer.layer_0");
        assert_eq!(grad_norm_group("transformer.layers.7.ffn.fc2.bias"), "transformer.layer_7");
        assert_eq!(grad_norm_group("input.hand_proj.weight"), "input");
        assert_eq!(grad_norm_group("value_head.fc1.bias"), "value_head");
    }

    #[test]
    fn metrics_json_line_is_valid() {
        let mut m = IterationMetrics::default();
        m.iteration = 3;
        m.learning_rate = 3e-4;
        m.combined_loss = 1.23;
        m.grad_norms = vec![("input".to_string(), 0.1), ("transformer.layer_0".to_string(), 0.5)];
        let line = m.to_json_line();
        assert!(line.ends_with('\n'));
        assert!(line.contains("\"iteration\":3"));
        assert!(line.contains("\"grad_norms\":"));
        assert!(line.contains("\"transformer.layer_0\":"));
    }

    #[test]
    fn train_phase_reduces_loss_on_small_buffer() {
        let tmp = std::env::temp_dir().join(format!("blob-tl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let cfg = TrainingLoopConfig {
            checkpoint_dir: tmp.clone(),
            buffer_capacity: 64,
            batch_size: 4,
            epochs_per_iteration: 4,
            epoch_early_stop_rel: -1.0, // never stop early
            total_training_steps: 10_000,
            total_iterations: 1,
            device: Device::Cpu,
        };
        let mut tl = TrainingLoop::new(cfg);
        tl.optimizer.set_lr(3e-3);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        // Populate the replay buffer with a mix of bid & play examples.
        let mut examples = Vec::new();
        for i in 0..8u64 {
            let ex = dummy_play_example(i);
            tl.buffer.push(ex.state, ex.policy.clone(), ex.value, ex.phase);
            examples.push(ex);
        }
        for i in 0..8u64 {
            let ex = dummy_bid_example(100 + i);
            tl.buffer.push(ex.state, ex.policy.clone(), ex.value, ex.phase);
            examples.push(ex);
        }

        let metrics = tl.train_phase(&mut rng, &examples);
        assert!(metrics.num_epochs_run >= 1);
        assert!(metrics.combined_loss.is_finite());
        // At least one grad-norm group should have registered a value.
        assert!(!metrics.grad_norms.is_empty());
        // Signal stats backfilled from `examples`.
        assert!(metrics.visit_entropy_mean > 0.0);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn checkpoint_retention_keeps_eval_and_rolling() {
        let tmp = std::env::temp_dir().join(format!("blob-tl-ret-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        for i in 0..=12u64 {
            std::fs::create_dir_all(tmp.join(format!("iter_{i:06}"))).unwrap();
        }
        let cfg = TrainingLoopConfig {
            checkpoint_dir: tmp.clone(),
            ..Default::default()
        };
        let mut tl = TrainingLoop::new(cfg);
        tl.iteration = 12;
        tl.prune_checkpoints().unwrap();

        let mut kept: Vec<u64> = std::fs::read_dir(&tmp)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                e.path()
                    .file_name()
                    .and_then(|s| s.to_str())
                    .and_then(|n| n.strip_prefix("iter_").and_then(|s| s.parse::<u64>().ok()))
            })
            .collect();
        kept.sort();
        // Every-5th (0, 5, 10) kept, plus rolling (12). 11 pruned.
        assert_eq!(kept, vec![0, 5, 10, 12]);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn resume_picks_highest_iteration() {
        let tmp = std::env::temp_dir().join(format!("blob-tl-res-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let cfg = TrainingLoopConfig {
            checkpoint_dir: tmp.clone(),
            ..Default::default()
        };
        // No checkpoints yet → None.
        let mut tl = TrainingLoop::new(cfg.clone());
        assert_eq!(tl.resume_from_latest().unwrap(), None);

        // Save at iter 3 and 7; resume should pick 7.
        save_model_checkpoint(&tl.vs, 3, tmp.join("iter_000003")).unwrap();
        save_model_checkpoint(&tl.vs, 7, tmp.join("iter_000007")).unwrap();
        let mut tl2 = TrainingLoop::new(cfg);
        let resumed = tl2.resume_from_latest().unwrap();
        assert_eq!(resumed, Some(7));
        assert_eq!(tl2.iteration, 8);

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
