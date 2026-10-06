//! Losses, optimizer, LR schedule, training steps and checkpoints for the
//! two networks (gen-2.md §5.3, §5.6).
//!
//! - **P loss:** cross-entropy against the target policy, `-Σ t · log(p + ε)`.
//!   Illegal moves have `t = 0` and contribute nothing. A step's bid and
//!   play sub-batches are weighted by their example counts.
//! - **V loss:** sigmoid cross-entropy between V's logits and the actual ŝ
//!   (a soft target in \[0, 1\]) over the real seats of each table. Like MSE
//!   it is minimized by the expected ŝ, but its gradient doesn't vanish
//!   when the sigmoid saturates: under MSE a high learning rate pinned V at
//!   0 for good. Held-out V is still reported as MSE.
//! - **Optimizer:** AdamW (β₁ = 0.9, β₂ = 0.999), one per network; global
//!   grad-norm clip 1.0.
//! - **LR schedule** keyed to learner steps ([`LrSchedule`]): linear warm-up,
//!   then cosine to `min_lr` at `total_steps`. The LR is a function of the
//!   step alone, so a resume continues it exactly (gen-2.md §3.3).
//! - **Checkpoints:** one directory with `policy.ot` and `value.ot` (tch
//!   `VarStore`s) and `meta.json` (`{"learner_step": …}`).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tch::{
    nn::{self, OptimizerConfig, VarStore},
    Device, Kind, Tensor,
};

use crate::input::InputBatch;
use crate::model::{PolicyNet, ValueNet};

pub const LOG_EPS: f64 = 1e-8;
pub const GRAD_CLIP_MAX_NORM: f64 = 1.0;
pub const ADAM_BETA1: f64 = 0.9;
pub const ADAM_BETA2: f64 = 0.999;

/// Which of P's heads a batch targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Bidding,
    Playing,
}

/// One phase's examples for P.
///
/// - `Phase::Bidding`: `legal_mask` and `target` are `[B, 14]`.
/// - `Phase::Playing`: `legal_mask` and `target` are `[B, S]`, `S` the
///   padded sequence length. Only hand-card tokens of legal plays are true
///   in the mask; the target is 0 everywhere else.
pub struct PolicyBatch {
    pub input: InputBatch,
    pub phase: Phase,
    pub legal_mask: Tensor,
    pub target: Tensor,
}

impl PolicyBatch {
    pub fn rows(&self) -> i64 {
        self.target.size()[0]
    }

    pub fn to_device(&self, device: Device) -> Self {
        Self {
            input: self.input.to_device(device),
            phase: self.phase,
            legal_mask: self.legal_mask.to_device(device),
            target: self.target.to_device(device),
        }
    }
}

/// Examples for V: V-mode inputs, and each row's actual ŝ per relative seat
/// with the mask of real seats, both `[B, K]` (`K` = the largest table in
/// the batch).
pub struct ValueBatch {
    pub input: InputBatch,
    pub target: Tensor,
    pub seat_mask: Tensor,
}

impl ValueBatch {
    pub fn to_device(&self, device: Device) -> Self {
        Self {
            input: self.input.to_device(device),
            target: self.target.to_device(device),
            seat_mask: self.seat_mask.to_device(device),
        }
    }
}

/// P's policy for a batch: `[B, 14]` or `[B, S]`.
pub fn policy_probs(net: &PolicyNet, batch: &PolicyBatch, train: bool) -> Tensor {
    match batch.phase {
        Phase::Bidding => net.forward_bid(&batch.input, &batch.legal_mask, train),
        Phase::Playing => net.forward_play(&batch.input, &batch.legal_mask, train),
    }
}

/// Per-row cross-entropy `-Σ t · log(p + ε)`, `[B]`. Target entries on
/// illegal actions must be zero; `p` is already 0 there.
pub fn policy_cross_entropy_rows(pred_probs: &Tensor, target: &Tensor) -> Tensor {
    -(target * (pred_probs + LOG_EPS).log()).sum_dim_intlist(&[-1i64][..], false, Kind::Float)
}

/// Mean of [`policy_cross_entropy_rows`].
pub fn policy_cross_entropy(pred_probs: &Tensor, target: &Tensor) -> Tensor {
    policy_cross_entropy_rows(pred_probs, target).mean(Kind::Float)
}

/// Sigmoid cross-entropy `−t·log σ(z) − (1−t)·log(1−σ(z)) = softplus(z) − t·z`
/// over the entries where `mask` is true.
pub fn seat_bce_with_logits(logits: &Tensor, target: &Tensor, mask: &Tensor) -> Tensor {
    let m = mask.to_kind(Kind::Float);
    ((logits.softplus() - target * logits) * &m).sum(Kind::Float) / m.sum(Kind::Float).clamp_min(1.0)
}

/// P's loss over a step's sub-batches, weighted by their example counts.
pub fn policy_loss(net: &PolicyNet, batches: &[&PolicyBatch], train: bool) -> Tensor {
    let rows: i64 = batches.iter().map(|b| b.rows()).sum();
    assert!(rows > 0, "policy step without examples");
    batches
        .iter()
        .map(|b| policy_cross_entropy_rows(&policy_probs(net, b, train), &b.target).sum(Kind::Float))
        .reduce(|a, b| a + b)
        .expect("at least one batch")
        / rows as f64
}

/// V's training loss on a batch ([`seat_bce_with_logits`]).
pub fn value_loss(net: &ValueNet, batch: &ValueBatch, train: bool) -> Tensor {
    let seats = batch.target.size()[1];
    seat_bce_with_logits(&net.seat_logits(&batch.input, seats, train), &batch.target, &batch.seat_mask)
}

/// Learning rate by learner step: linear warm-up to `peak_lr` over
/// `warmup_steps`, then a cosine down to `min_lr` at `total_steps`, and
/// `min_lr` after that.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LrSchedule {
    pub warmup_steps: u64,
    pub total_steps: u64,
    pub peak_lr: f64,
    pub min_lr: f64,
}

impl LrSchedule {
    pub fn lr(&self, step: u64) -> f64 {
        if step < self.warmup_steps {
            return self.peak_lr * (step + 1) as f64 / self.warmup_steps as f64;
        }
        let span = self.total_steps.saturating_sub(self.warmup_steps).max(1);
        let t = ((step - self.warmup_steps) as f64 / span as f64).min(1.0);
        self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (1.0 + (std::f64::consts::PI * t).cos())
    }
}

/// AdamW over every variable of `vs`. The LR is set before each step.
pub fn build_optimizer(vs: &VarStore, weight_decay: f64) -> Result<nn::Optimizer, tch::TchError> {
    nn::AdamW { beta1: ADAM_BETA1, beta2: ADAM_BETA2, wd: weight_decay, ..Default::default() }.build(vs, 0.0)
}

/// Scale the gradients of `vars` down to a global norm of at most `max`.
/// Unlike `Optimizer::clip_grad_norm`, the norm never leaves the device, so
/// the step doesn't wait for the GPU.
pub fn clip_grad_norm(vars: &[Tensor], max: f64) {
    tch::no_grad(|| {
        let grads: Vec<Tensor> = vars.iter().map(|v| v.grad()).filter(|g| g.defined()).collect();
        if grads.is_empty() {
            return;
        }
        let norms: Vec<Tensor> = grads.iter().map(|g| g.norm()).collect();
        let total = Tensor::stack(&norms, 0).norm();
        let coef = ((total + 1e-6).reciprocal() * max).clamp_max(1.0);
        for mut g in grads {
            let _ = g.g_mul_(&coef);
        }
    })
}

/// One optimizer step on `loss`. Returns the loss, detached.
pub fn optimize(opt: &mut nn::Optimizer, vars: &[Tensor], lr: f64, loss: Tensor) -> Tensor {
    opt.set_lr(lr);
    opt.zero_grad();
    loss.backward();
    clip_grad_norm(vars, GRAD_CLIP_MAX_NORM);
    opt.step();
    loss.detach()
}

pub const POLICY_WEIGHTS: &str = "policy.ot";
pub const VALUE_WEIGHTS: &str = "value.ot";
pub const CHECKPOINT_META: &str = "meta.json";

/// A checkpoint's `meta.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointMeta {
    /// Learner steps taken: the step the next update gets.
    pub learner_step: u64,
}

fn io(e: std::io::Error) -> tch::TchError {
    tch::TchError::Io(e)
}

/// Write a checkpoint directory. It is written next to `dir` and renamed
/// into place, so an interrupted save leaves the previous one intact.
///
/// Optimizer state is not saved (tch 0.20 doesn't expose the AdamW
/// moments): a resumed run rebuilds the optimizers.
pub fn save_checkpoint(
    dir: impl AsRef<Path>,
    policy: &VarStore,
    value: &VarStore,
    meta: CheckpointMeta,
) -> Result<(), tch::TchError> {
    let dir = dir.as_ref();
    let sibling = |suffix: &str| -> PathBuf {
        let mut name = dir.file_name().unwrap_or_default().to_os_string();
        name.push(suffix);
        dir.with_file_name(name)
    };
    let (tmp, old) = (sibling(".tmp"), sibling(".old"));
    if tmp.exists() {
        std::fs::remove_dir_all(&tmp).map_err(io)?;
    }
    std::fs::create_dir_all(&tmp).map_err(io)?;
    policy.save(tmp.join(POLICY_WEIGHTS))?;
    value.save(tmp.join(VALUE_WEIGHTS))?;
    let json = serde_json::to_string(&meta).expect("meta serializes") + "\n";
    std::fs::write(tmp.join(CHECKPOINT_META), json).map_err(io)?;
    if dir.exists() {
        if old.exists() {
            std::fs::remove_dir_all(&old).map_err(io)?;
        }
        std::fs::rename(dir, &old).map_err(io)?;
    }
    std::fs::rename(&tmp, dir).map_err(io)?;
    if old.exists() {
        std::fs::remove_dir_all(&old).map_err(io)?;
    }
    Ok(())
}

/// Load a checkpoint written by [`save_checkpoint`] into `policy` and
/// `value`, which must hold a [`PolicyNet`] and a [`ValueNet`].
pub fn load_checkpoint(
    dir: impl AsRef<Path>,
    policy: &mut VarStore,
    value: &mut VarStore,
) -> Result<CheckpointMeta, tch::TchError> {
    let dir = dir.as_ref();
    policy.load(dir.join(POLICY_WEIGHTS))?;
    value.load(dir.join(VALUE_WEIGHTS))?;
    let raw = std::fs::read_to_string(dir.join(CHECKPOINT_META)).map_err(io)?;
    serde_json::from_str(&raw)
        .map_err(|e| tch::TchError::FileFormat(format!("{}: {e}", dir.join(CHECKPOINT_META).display())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::pad_batch;
    use blob_engine::encoder::{encode, encode_value, TOKEN_TYPE_HAND};
    use blob_engine::{new_round, RoundParams};
    use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256PlusPlus};

    fn states(seed: u64, n: usize) -> Vec<blob_engine::BlobState> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        (0..n)
            .map(|i| {
                let p = RoundParams { num_players: 4 + (i % 2) as u8, cards_dealt: 5, trump: 0, dealer: 0 };
                new_round(p, &mut rng).unwrap()
            })
            .collect()
    }

    /// Every hand card treated as legal, uniform target over them.
    fn play_batch(seed: u64, n: usize) -> PolicyBatch {
        let encs: Vec<_> = states(seed, n).iter().map(|s| encode(s, s.current_player)).collect();
        let input = pad_batch(&encs, Device::Cpu);
        let hand = input.token_types.eq(TOKEN_TYPE_HAND as i64);
        let counts = hand.to_kind(Kind::Float).sum_dim_intlist(&[-1i64][..], true, Kind::Float);
        let target = hand.to_kind(Kind::Float) / counts.clamp_min(1.0);
        PolicyBatch { input, phase: Phase::Playing, legal_mask: hand, target }
    }

    fn value_batch(seed: u64, n: usize) -> ValueBatch {
        let ss = states(seed, n);
        let encs: Vec<_> = ss.iter().map(|s| encode_value(s, s.current_player)).collect();
        let input = pad_batch(&encs, Device::Cpu);
        let k = 5usize;
        let mut target = vec![0.0f32; n * k];
        let mut mask = vec![false; n * k];
        for (row, s) in ss.iter().enumerate() {
            for seat in 0..s.num_players as usize {
                target[row * k + seat] = if seat % 2 == 0 { 0.8 } else { 0.0 };
                mask[row * k + seat] = true;
            }
        }
        ValueBatch {
            input,
            target: Tensor::from_slice(&target).view([n as i64, k as i64]),
            seat_mask: Tensor::from_slice(&mask).view([n as i64, k as i64]),
        }
    }

    #[test]
    fn policy_xent_zero_on_perfect_prediction() {
        let pred = Tensor::from_slice(&[0.0f32, 1.0, 0.0]).view([1, 3]);
        let loss = policy_cross_entropy(&pred, &pred).double_value(&[]);
        assert!(loss < 1e-6, "expected near-zero, got {loss}");
    }

    #[test]
    fn policy_xent_ignores_illegal_with_zero_target() {
        let pred = Tensor::from_slice(&[0.0f32, 0.5, 0.5]).view([1, 3]);
        let loss = policy_cross_entropy(&pred, &pred).double_value(&[]);
        assert!(loss.is_finite(), "loss not finite: {loss}");
    }

    /// The cross-entropy matches its definition, ignores masked seats, and is
    /// smallest where σ(z) equals the target.
    #[test]
    fn seat_bce_matches_definition_and_targets_the_mean() {
        let z = Tensor::from_slice(&[0.3f32, -2.0, 50.0]).view([1, 3]);
        let t = Tensor::from_slice(&[0.7f32, 0.0, 0.0]).view([1, 3]);
        let mask = Tensor::from_slice(&[true, true, false]).view([1, 3]);
        let p = z.sigmoid();
        let (one_t, one_p): (Tensor, Tensor) = (1.0 - &t, 1.0 - &p);
        let want = -(&t * p.log() + one_t * one_p.log()).narrow(1, 0, 2).mean(Kind::Float);
        let got = seat_bce_with_logits(&z, &t, &mask);
        assert!((got - want).abs().double_value(&[]) < 1e-6);
        let at = |x: f64| {
            let z = Tensor::from_slice(&[x as f32]).view([1, 1]);
            let t = Tensor::from_slice(&[0.6f32]).view([1, 1]);
            seat_bce_with_logits(&z, &t, &Tensor::from_slice(&[true]).view([1, 1])).double_value(&[])
        };
        let best = (0.6f64 / 0.4).ln();
        assert!(at(best) < at(best + 0.1) && at(best) < at(best - 0.1));
    }

    #[test]
    fn lr_schedule_warms_up_then_decays_to_min() {
        let s = LrSchedule { warmup_steps: 100, total_steps: 1100, peak_lr: 3e-4, min_lr: 1e-5 };
        assert!(s.lr(0) > 0.0 && s.lr(0) < s.peak_lr);
        assert!((s.lr(99) - s.peak_lr).abs() < 1e-12);
        assert!((s.lr(100) - s.peak_lr).abs() < 1e-12);
        let mid = s.lr(600);
        assert!((mid - (s.min_lr + s.peak_lr) / 2.0).abs() < 1e-9, "{mid}");
        assert!((s.lr(1100) - s.min_lr).abs() < 1e-12);
        assert!((s.lr(50_000) - s.min_lr).abs() < 1e-12);
        // Strictly decreasing after warm-up.
        assert!((100..1100).all(|t| s.lr(t + 1) < s.lr(t)));
    }

    #[test]
    fn policy_and_value_steps_reduce_their_losses() {
        tch::manual_seed(42);
        let vs = VarStore::new(Device::Cpu);
        let p = PolicyNet::new(&vs.root());
        let mut opt = build_optimizer(&vs, 1e-4).unwrap();
        let vars = vs.trainable_variables();
        let batch = play_batch(123, 4);
        let first = policy_loss(&p, &[&batch], false).double_value(&[]);
        for _ in 0..100 {
            let l = optimize(&mut opt, &vars, 3e-4, policy_loss(&p, &[&batch], true));
            assert!(l.double_value(&[]).is_finite());
        }
        let last = policy_loss(&p, &[&batch], false).double_value(&[]);
        assert!(last < first - 0.05, "P: {first} -> {last}");

        let vs = VarStore::new(Device::Cpu);
        let v = ValueNet::new(&vs.root());
        let mut opt = build_optimizer(&vs, 1e-4).unwrap();
        let vars = vs.trainable_variables();
        let batch = value_batch(7, 6);
        let mse = || {
            let pred = v.seat_values(&batch.input, 5, false);
            let m = batch.seat_mask.to_kind(Kind::Float);
            ((pred - &batch.target).square() * &m).sum(Kind::Float).double_value(&[]) / m.sum(Kind::Float).double_value(&[])
        };
        let first = mse();
        for _ in 0..100 {
            let _ = optimize(&mut opt, &vars, 1e-3, value_loss(&v, &batch, true));
        }
        let last = mse();
        assert!(last < first / 2.0, "V: {first} -> {last}");
    }

    #[test]
    fn clip_bounds_the_global_grad_norm() {
        let vs = VarStore::new(Device::Cpu);
        let x = vs.root().var("x", &[3], nn::Init::Const(1.0));
        let y = vs.root().var("y", &[1], nn::Init::Const(1.0));
        let loss = (&x * 100.0).sum(Kind::Float) + (&y * 100.0).sum(Kind::Float);
        loss.backward();
        let vars = vs.trainable_variables();
        clip_grad_norm(&vars, 1.0);
        let norm: f64 = vars.iter().map(|v| v.grad().square().sum(Kind::Float).double_value(&[])).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "{norm}");
        // A small gradient is left alone.
        let vs = VarStore::new(Device::Cpu);
        let z = vs.root().var("z", &[2], nn::Init::Const(1.0));
        (&z * 0.1).sum(Kind::Float).backward();
        clip_grad_norm(&vs.trainable_variables(), 1.0);
        let g: Vec<f32> = z.grad().try_into().unwrap();
        assert!(g.iter().all(|&x| (x - 0.1).abs() < 1e-7), "{g:?}");
    }

    #[test]
    fn checkpoint_round_trip_restores_weights_and_step() {
        let tmp = std::env::temp_dir().join(format!("blob-ckpt-{}", std::process::id()));
        let (pv, vv) = (VarStore::new(Device::Cpu), VarStore::new(Device::Cpu));
        let (_p, _v) = (PolicyNet::new(&pv.root()), ValueNet::new(&vv.root()));
        save_checkpoint(&tmp, &pv, &vv, CheckpointMeta { learner_step: 41 }).unwrap();
        // A second save replaces the first in place.
        save_checkpoint(&tmp, &pv, &vv, CheckpointMeta { learner_step: 42 }).unwrap();

        let (mut pv2, mut vv2) = (VarStore::new(Device::Cpu), VarStore::new(Device::Cpu));
        let (_p2, _v2) = (PolicyNet::new(&pv2.root()), ValueNet::new(&vv2.root()));
        let meta = load_checkpoint(&tmp, &mut pv2, &mut vv2).unwrap();
        assert_eq!(meta.learner_step, 42);
        for (a, b) in [(&pv, &pv2), (&vv, &vv2)] {
            let (va, vb) = (a.variables(), b.variables());
            assert_eq!(va.len(), vb.len());
            for (name, t) in &va {
                let diff = (t - &vb[name]).abs().sum(Kind::Float).double_value(&[]);
                assert_eq!(diff, 0.0, "{name}");
            }
        }
        // P's weights don't load into V.
        let mut wrong = VarStore::new(Device::Cpu);
        let _ = ValueNet::new(&wrong.root());
        assert!(wrong.load(tmp.join(POLICY_WEIGHTS)).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
