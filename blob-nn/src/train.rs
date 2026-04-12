//! Session 3.4 — losses, optimizer, LR schedule, training step, checkpoints.
//!
//! - Policy loss: cross-entropy against an MCTS-visit-count distribution,
//!   `-Σ t · log(p + ε)`. Illegal actions have `t = 0`, contributing nothing.
//! - Value loss: MSE against a z-scored cumulative-score target clipped to
//!   `[-1, 1]`.
//! - Combined: `policy_loss + 2.0 · value_loss`.
//! - Optimizer: AdamW (β₁=0.9, β₂=0.999, wd=1e-4).
//! - LR schedule: linear warmup to 3e-4 over `warmup_steps`, then cosine
//!   annealing to 1e-5 over the remaining steps.
//! - Grad clip: global norm 1.0.
//! - Checkpoints: `VarStore::save` + sidecar JSON with iteration.

use std::path::{Path, PathBuf};

use tch::{
    nn::{self, OptimizerConfig, VarStore},
    Kind, Tensor,
};

use crate::input::InputBatch;
use crate::model::BlobNet;

pub const LOG_EPS: f64 = 1e-8;
pub const VALUE_LOSS_COEF: f64 = 2.0;
pub const PEAK_LR: f64 = 3e-4;
pub const MIN_LR: f64 = 1e-5;
pub const DEFAULT_WARMUP_STEPS: i64 = 1000;
pub const GRAD_CLIP_MAX_NORM: f64 = 1.0;
pub const ADAM_BETA1: f64 = 0.9;
pub const ADAM_BETA2: f64 = 0.999;
pub const WEIGHT_DECAY: f64 = 1e-4;

/// Which head the batch targets. The value head always contributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Bidding,
    Playing,
}

/// One training batch. Policy target and legal mask shapes depend on phase:
///
/// - `Phase::Bidding`: `policy_target` and `legal_mask` are `[B, 14]`.
/// - `Phase::Playing`: `policy_target` and `legal_mask` are `[B, S]` where
///   `S` matches `input.attention_mask.size()[1]`. Non-hand and illegal
///   positions must be 0 in the target and `false` in the mask.
///
/// `value_target: [B]` f32 in `[-1, 1]` (z-scored, already clipped).
pub struct TrainBatch {
    pub input: InputBatch,
    pub phase: Phase,
    pub legal_mask: Tensor,
    pub policy_target: Tensor,
    pub value_target: Tensor,
}

/// Cross-entropy against a probability target. Target entries on illegal
/// actions must be zero (they contribute zero term regardless of `p`).
///
/// `pred_probs` should already be masked+softmaxed by the head, so
/// illegal actions have `p = 0`; we add `LOG_EPS` before `log` to keep
/// the gradient finite there, and the target's zero zeros out the term.
pub fn policy_cross_entropy(pred_probs: &Tensor, target: &Tensor) -> Tensor {
    let log_p = (pred_probs + LOG_EPS).log();
    let per_sample = -(target * log_p).sum_dim_intlist(
        &[-1i64][..],
        false,
        Kind::Float,
    );
    per_sample.mean(Kind::Float)
}

/// MSE `(pred - target)²`.
pub fn value_mse(pred: &Tensor, target: &Tensor) -> Tensor {
    (pred - target).square().mean(Kind::Float)
}

/// Z-score a slice of cumulative scores into `[-1, 1]`.
///
/// `clip((x - mean) / max(std, eps), -1, 1)`. Exposed for test fixtures
/// and for the self-play pipeline in Section 5.
pub fn z_score_clip(scores: &[f32], eps: f32) -> Vec<f32> {
    let n = scores.len() as f32;
    if n == 0.0 {
        return vec![];
    }
    let mean = scores.iter().sum::<f32>() / n;
    let var = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt().max(eps);
    scores
        .iter()
        .map(|s| ((s - mean) / std).clamp(-1.0, 1.0))
        .collect()
}

/// Learning-rate schedule: linear warmup → cosine annealing.
#[derive(Debug, Clone, Copy)]
pub struct LrSchedule {
    pub warmup_steps: i64,
    pub total_steps: i64,
    pub peak_lr: f64,
    pub min_lr: f64,
}

impl LrSchedule {
    pub fn new(total_steps: i64) -> Self {
        Self {
            warmup_steps: DEFAULT_WARMUP_STEPS,
            total_steps,
            peak_lr: PEAK_LR,
            min_lr: MIN_LR,
        }
    }

    /// LR at 0-indexed training `step`. Steps past `total_steps` stay at `min_lr`.
    pub fn lr(&self, step: i64) -> f64 {
        if step < self.warmup_steps {
            let frac = (step + 1) as f64 / self.warmup_steps.max(1) as f64;
            return self.peak_lr * frac;
        }
        let decay_steps = (self.total_steps - self.warmup_steps).max(1);
        let t = ((step - self.warmup_steps) as f64 / decay_steps as f64).min(1.0);
        let cos = 0.5 * (1.0 + (std::f64::consts::PI * t).cos());
        self.min_lr + (self.peak_lr - self.min_lr) * cos
    }
}

/// Individual loss terms produced by [`train_step`].
#[derive(Debug, Clone, Copy)]
pub struct StepLosses {
    pub policy: f64,
    pub value: f64,
    pub total: f64,
}

/// Build the AdamW optimizer with the project defaults, bound to `vs`.
pub fn build_optimizer(vs: &VarStore) -> Result<nn::Optimizer, tch::TchError> {
    nn::AdamW {
        beta1: ADAM_BETA1,
        beta2: ADAM_BETA2,
        wd: WEIGHT_DECAY,
        ..Default::default()
    }
    .build(vs, PEAK_LR)
}

/// One training step: forward, loss, backward, clip, step. Returns the
/// three scalar loss values (after the step).
pub fn train_step(
    model: &BlobNet,
    optimizer: &mut nn::Optimizer,
    batch: &TrainBatch,
) -> StepLosses {
    let (policy_probs, value_pred) = match batch.phase {
        Phase::Bidding => model.forward_bid(&batch.input, &batch.legal_mask, true),
        Phase::Playing => model.forward_play(&batch.input, &batch.legal_mask, true),
    };

    let policy_loss = policy_cross_entropy(&policy_probs, &batch.policy_target);
    let value_loss = value_mse(&value_pred, &batch.value_target);
    let total = &policy_loss + VALUE_LOSS_COEF * &value_loss;

    optimizer.zero_grad();
    total.backward();
    optimizer.clip_grad_norm(GRAD_CLIP_MAX_NORM);
    optimizer.step();

    StepLosses {
        policy: policy_loss.double_value(&[]),
        value: value_loss.double_value(&[]),
        total: total.double_value(&[]),
    }
}

/// On-disk checkpoint layout:
/// - `{dir}/model.ot`: weights via `VarStore::save`
/// - `{dir}/meta.json`: `{"iteration": <u64>}`
///
/// Note on optimizer state: `tch-rs` 0.20 does not expose the Adam moment
/// buffers for serialization. On resume we rebuild the optimizer and
/// re-enter the LR schedule at `iteration`. The first few post-resume
/// steps will see warmup-like transient moments; this is an acceptable
/// trade-off given this project's short training horizons (Section 5).
pub fn save_checkpoint(
    vs: &VarStore,
    iteration: u64,
    dir: impl AsRef<Path>,
) -> Result<(), tch::TchError> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir).map_err(|e| tch::TchError::Io(e))?;
    vs.save(dir.join("model.ot"))?;
    let meta = format!("{{\"iteration\":{iteration}}}\n");
    std::fs::write(dir.join("meta.json"), meta).map_err(|e| tch::TchError::Io(e))?;
    Ok(())
}

/// Load weights into `vs` and return the iteration counter.
pub fn load_checkpoint(
    vs: &mut VarStore,
    dir: impl AsRef<Path>,
) -> Result<u64, tch::TchError> {
    let dir: PathBuf = dir.as_ref().to_path_buf();
    vs.load(dir.join("model.ot"))?;
    let raw = std::fs::read_to_string(dir.join("meta.json"))
        .map_err(|e| tch::TchError::Io(e))?;
    // Minimal parse: extract the integer after `"iteration":`.
    let iteration = raw
        .split("\"iteration\":")
        .nth(1)
        .and_then(|s| s.trim().trim_end_matches('}').trim().trim_end_matches('\n').parse::<u64>().ok())
        .unwrap_or(0);
    Ok(iteration)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heads::NUM_BIDS;
    use crate::input::{pad_batch, FEAT_DIM};
    use blob_engine::encoder::{encode, TOKEN_TYPE_HAND};
    use blob_engine::{dealing::deal, game::new_game};
    use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256PlusPlus};
    use tch::{Device, Kind};

    fn make_play_batch(seed: u64, batch_size: usize) -> TrainBatch {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut encs = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let mut s = new_game(4, 5).unwrap();
            deal(&mut s, &mut rng);
            let e = encode(&s, s.current_player);
            encs.push(e);
        }
        let input = pad_batch(&encs, Device::Cpu);
        assert_eq!(input.features.size()[2], FEAT_DIM);
        let s = input.attention_mask.size()[1];
        let b = batch_size as i64;

        // legal_mask = hand tokens (every hand card is treated legal here).
        let hand_mask = input.token_types.eq(TOKEN_TYPE_HAND as i64);

        // Policy target: uniform over legal hand positions per row.
        let counts = hand_mask.to_kind(Kind::Float).sum_dim_intlist(
            &[-1i64][..],
            true,
            Kind::Float,
        );
        let target = hand_mask.to_kind(Kind::Float) / counts.clamp_min(1.0);

        // Value target: random in [-1, 1].
        let vt_vec: Vec<f32> = (0..b).map(|i| ((i as f32) * 0.37 - 0.5).clamp(-1.0, 1.0)).collect();
        let value_target = Tensor::from_slice(&vt_vec).view([b]);

        let _ = s;
        TrainBatch {
            input,
            phase: Phase::Playing,
            legal_mask: hand_mask,
            policy_target: target,
            value_target,
        }
    }

    #[test]
    fn policy_xent_zero_on_perfect_prediction() {
        // Target = pred (one-hot), loss ≈ 0 (up to LOG_EPS).
        let pred = Tensor::from_slice(&[0.0f32, 1.0, 0.0]).view([1, 3]);
        let target = pred.shallow_clone();
        let loss = policy_cross_entropy(&pred, &target).double_value(&[]);
        assert!(loss < 1e-6, "expected near-zero, got {loss}");
    }

    #[test]
    fn policy_xent_ignores_illegal_with_zero_target() {
        // Illegal action has p=0, t=0 → 0·log(ε) = 0, not NaN.
        let pred = Tensor::from_slice(&[0.0f32, 0.5, 0.5]).view([1, 3]);
        let target = Tensor::from_slice(&[0.0f32, 0.5, 0.5]).view([1, 3]);
        let loss = policy_cross_entropy(&pred, &target).double_value(&[]);
        assert!(loss.is_finite(), "loss not finite: {loss}");
    }

    #[test]
    fn z_score_clip_produces_unit_variance_then_clips() {
        let xs = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let z = z_score_clip(&xs, 1e-6);
        assert_eq!(z.len(), 5);
        // Symmetric around 0.
        let sum: f32 = z.iter().sum();
        assert!(sum.abs() < 1e-5, "sum not zero: {sum}");
        for v in &z {
            assert!((-1.0..=1.0).contains(v));
        }
        // Degenerate: all equal → all zeros.
        let z2 = z_score_clip(&[5.0, 5.0, 5.0], 1e-6);
        assert!(z2.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn lr_schedule_warmup_peak_cosine() {
        let s = LrSchedule {
            warmup_steps: 100,
            total_steps: 1000,
            peak_lr: 3e-4,
            min_lr: 1e-5,
        };
        assert!(s.lr(0) > 0.0 && s.lr(0) < s.peak_lr);
        assert!((s.lr(99) - s.peak_lr).abs() < 1e-9);
        // End-of-schedule is min_lr.
        assert!((s.lr(999) - s.min_lr).abs() < 1e-6);
        // Mid-decay is between min and peak.
        let mid = s.lr(550);
        assert!(mid > s.min_lr && mid < s.peak_lr);
        // Past the end stays at min_lr.
        assert!((s.lr(5_000) - s.min_lr).abs() < 1e-9);
    }

    #[test]
    fn train_step_reduces_loss_on_fixed_batch() {
        tch::manual_seed(42);
        let vs = VarStore::new(Device::Cpu);
        let model = BlobNet::new(&vs.root());
        let mut opt = build_optimizer(&vs).unwrap();
        opt.set_lr(3e-4); // project-default peak LR.

        // Build a batch whose value target matches the model's *initial*
        // value prediction (detached). This removes value-loss variance
        // from the smoke test, so loss changes reflect the policy signal.
        let mut batch = make_play_batch(123, 2);
        let (_p0, v0) = model.forward_play(&batch.input, &batch.legal_mask, false);
        batch.value_target = v0.detach().copy();

        let (p0, v0b) = model.forward_play(&batch.input, &batch.legal_mask, false);
        let initial_policy =
            policy_cross_entropy(&p0, &batch.policy_target).double_value(&[]);
        let initial_value = value_mse(&v0b, &batch.value_target).double_value(&[]);
        assert!(initial_value < 1e-6, "value loss must start at zero");

        let mut last_policy = f64::INFINITY;
        for _ in 0..200 {
            let l = train_step(&model, &mut opt, &batch);
            last_policy = l.policy;
            assert!(l.total.is_finite(), "loss diverged: {l:?}");
        }
        assert!(
            last_policy < initial_policy - 0.05,
            "policy loss did not decrease: initial={initial_policy} last={last_policy}"
        );
    }

    #[test]
    fn bidding_train_step_runs_and_decreases() {
        let vs = VarStore::new(Device::Cpu);
        let model = BlobNet::new(&vs.root());
        let mut opt = build_optimizer(&vs).unwrap();
        opt.set_lr(1e-3);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(9);
        let mut encs = Vec::new();
        for _ in 0..2 {
            let mut s = new_game(4, 5).unwrap();
            deal(&mut s, &mut rng);
            encs.push(encode(&s, s.current_player));
        }
        let input = pad_batch(&encs, Device::Cpu);
        let b = 2i64;
        let legal = Tensor::ones([b, NUM_BIDS], (Kind::Bool, Device::Cpu));
        // Target: one-hot on bid 3 for both rows.
        let mut tgt = vec![0.0f32; (b * NUM_BIDS) as usize];
        tgt[3] = 1.0;
        tgt[(NUM_BIDS + 3) as usize] = 1.0;
        let policy_target = Tensor::from_slice(&tgt).view([b, NUM_BIDS]);
        let value_target = Tensor::from_slice(&[0.2f32, -0.3]).view([b]);

        let batch = TrainBatch {
            input,
            phase: Phase::Bidding,
            legal_mask: legal,
            policy_target,
            value_target,
        };

        let first = train_step(&model, &mut opt, &batch).total;
        let mut last = first;
        for _ in 0..50 {
            last = train_step(&model, &mut opt, &batch).total;
        }
        assert!(last < first, "bidding loss did not decrease: {first} -> {last}");
    }

    #[test]
    fn checkpoint_round_trip_restores_weights() {
        let tmp = std::env::temp_dir().join(format!("blobnet-ckpt-{}", std::process::id()));

        // Train a bit to make weights non-default.
        let vs = VarStore::new(Device::Cpu);
        let model = BlobNet::new(&vs.root());
        let mut opt = build_optimizer(&vs).unwrap();
        let batch = make_play_batch(7, 2);
        for _ in 0..5 {
            train_step(&model, &mut opt, &batch);
        }

        // Snapshot one parameter tensor for comparison.
        let snap_name = vs.variables().keys().next().unwrap().clone();
        let snap_before = vs.variables()[&snap_name].shallow_clone().copy();

        save_checkpoint(&vs, 42, &tmp).unwrap();

        // Fresh model + fresh varstore, then load.
        let mut vs2 = VarStore::new(Device::Cpu);
        let _model2 = BlobNet::new(&vs2.root());
        let iter = load_checkpoint(&mut vs2, &tmp).unwrap();
        assert_eq!(iter, 42);

        let snap_after = vs2.variables()[&snap_name].shallow_clone();
        let diff: f64 = (&snap_before - &snap_after)
            .abs()
            .sum(Kind::Float)
            .double_value(&[]);
        assert!(diff < 1e-6, "weights did not round-trip (diff={diff})");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
