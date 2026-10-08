//! The learner (gen-2.md §5.6): P and V with an optimizer each, trained in
//! alternating updates on replay batches, plus the held-out measurements
//! and checkpoints.
//!
//! - **One learner step** is one P update and one V update, each on its own
//!   batch. The LR of both follows the step ([`LrSchedule`]). With
//!   `policy_from`, P is a checkpoint's and only V updates.
//! - **P** trains on the target policies of decisions with more than one
//!   legal move ([`is_forced`]); **V** on every state, against the actual
//!   ŝ of every seat ([`SeatScores`], relative to the seat to move: V's
//!   output order).
//! - **Validation by round** ([`is_validation_round`]): positions of one
//!   round share their outcome, so a position-level split would leak.
//! - **Held-out measurements** run with dropout off and no gradients.
//!   Compare a validation set with the same measurement on an equally large
//!   training sample, never with the losses logged during training.

use std::path::{Path, PathBuf};

use blob_engine::bidding::legal_bids;
use blob_engine::encoder::{encode, encode_value, TOKEN_TYPE_HAND};
use blob_engine::playing::legal_plays;
use blob_engine::replay::{BidBatch, PlayBatch, ReplayBuffer, SeatScores};
use blob_engine::state::{BlobState, GamePhase};
use serde::{Deserialize, Serialize};
use tch::nn::{self, VarStore};
use tch::{Device, Kind, Tensor};

use crate::heads::NUM_BIDS;
use crate::input::pad_batch;
use crate::model::{PolicyNet, ValueNet};
use crate::train::{
    build_optimizer, load_checkpoint, optimize, policy_cross_entropy_rows, policy_loss, policy_probs,
    save_checkpoint, value_loss, CheckpointMeta, LrSchedule, Phase, PolicyBatch, ValueBatch, POLICY_WEIGHTS,
};

/// Whether round `round_id` is held out for validation.
///
/// Split by round, not by position: positions from one round share a label,
/// so a position-level split would leave near-copies of every held-out
/// label in training (gen-2.md §5.6). Deterministic, and a larger
/// `fraction` holds out a superset.
pub fn is_validation_round(round_id: u64, fraction: f64) -> bool {
    let mut x = round_id ^ 0x5A11_DA7E_0F0F_0F0F;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    ((x >> 11) as f64 / (1u64 << 53) as f64) < fraction
}

/// Whether the seat to move has a single legal move. Such a decision
/// teaches P nothing (its loss is 0 whatever P outputs); V still learns
/// from the state.
pub fn is_forced(state: &BlobState) -> bool {
    match state.phase() {
        GamePhase::Bidding => legal_bids(state).count_ones() == 1,
        _ => legal_plays(state).count_ones() == 1,
    }
}

/// P's examples from a `BidBatch`.
pub fn bid_policy_batch(batch: &BidBatch, device: Device) -> Option<PolicyBatch> {
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
            mask[row * NUM_BIDS as usize + b] = (legal >> b) & 1 == 1;
        }
    }
    let legal_mask = Tensor::from_slice(&mask).view([n as i64, NUM_BIDS]).to_device(device);
    let target = Tensor::from_slice(&batch.policies).view([n as i64, NUM_BIDS]).to_device(device);
    Some(PolicyBatch { input, phase: Phase::Bidding, legal_mask, target })
}

/// P's examples from a `PlayBatch`.
///
/// Play policies in the replay buffer are indexed by **hand position**
/// (0..hand_size). The play head outputs one score per **sequence
/// position**, so each row's policy is scattered onto the sequence
/// positions whose `token_types[i] == TOKEN_TYPE_HAND`, in encoder
/// emission order (`EncodedState::hand_card_indices`).
pub fn play_policy_batch(batch: &PlayBatch, device: Device) -> Option<PolicyBatch> {
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
            // `max_hand_size` is the largest hand position with a policy
            // entry in the batch, plus one: positions past it have none.
            if hand_slot < batch.max_hand_size {
                target[row * seq_len + seq_i] = batch.policies[row * batch.max_hand_size + hand_slot];
            }
            mask[row * seq_len + seq_i] = (legal >> card_idx) & 1 == 1;
            hand_slot += 1;
        }
    }
    let legal_mask = Tensor::from_slice(&mask).view([n as i64, seq_len as i64]).to_device(device);
    let target = Tensor::from_slice(&target).view([n as i64, seq_len as i64]).to_device(device);
    Some(PolicyBatch { input, phase: Phase::Playing, legal_mask, target })
}

/// V's examples from both halves of a sampled batch, bids first: V-mode
/// inputs and each seat's actual ŝ.
pub fn value_batch(bid: &BidBatch, play: &PlayBatch, device: Device) -> Option<ValueBatch> {
    let states: Vec<&BlobState> = bid.states.iter().chain(&play.states).collect();
    let scores: Vec<&SeatScores> = bid.seat_scores.iter().chain(&play.seat_scores).collect();
    let n = states.len();
    if n == 0 {
        return None;
    }
    let encs: Vec<_> = states.iter().map(|s| encode_value(s, s.current_player)).collect();
    let input = pad_batch(&encs, device);
    let k = states.iter().map(|s| s.num_players as usize).max().unwrap_or(0);
    let mut target = vec![0.0f32; n * k];
    let mut mask = vec![false; n * k];
    for (row, (s, sc)) in states.iter().zip(&scores).enumerate() {
        for seat in 0..s.num_players as usize {
            target[row * k + seat] = sc[seat];
            mask[row * k + seat] = true;
        }
    }
    let shape = [n as i64, k as i64];
    Some(ValueBatch {
        input,
        target: Tensor::from_slice(&target).view(shape).to_device(device),
        seat_mask: Tensor::from_slice(&mask).view(shape).to_device(device),
    })
}

/// The tensors of one learner step: P's bid and play batches (either may
/// be missing) and V's batch. Built off the training thread on the CPU,
/// then moved with [`StepBatches::to_device`].
pub struct StepBatches {
    pub policy: Vec<PolicyBatch>,
    pub value: Option<ValueBatch>,
}

impl StepBatches {
    pub fn build(p: &(BidBatch, PlayBatch), v: &(BidBatch, PlayBatch), device: Device) -> Self {
        Self {
            policy: [bid_policy_batch(&p.0, device), play_policy_batch(&p.1, device)].into_iter().flatten().collect(),
            value: value_batch(&v.0, &v.1, device),
        }
    }

    pub fn to_device(&self, device: Device) -> Self {
        Self {
            policy: self.policy.iter().map(|b| b.to_device(device)).collect(),
            value: self.value.as_ref().map(|b| b.to_device(device)),
        }
    }
}

/// The learner's settings. Unknown keys are an error.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct LearnerConfig {
    /// `cuda`, `cuda:N` or `cpu`.
    pub device: String,
    /// Examples per P update, and per V update.
    pub batch_size: usize,
    /// Learner steps; each is one P and one V update.
    pub steps: u64,
    pub warmup_steps: u64,
    pub peak_lr: f64,
    pub min_lr: f64,
    pub weight_decay: f64,
    /// Relabel the suits of every sampled example at random.
    pub augment: bool,
    /// Copy P from this checkpoint directory and don't train it; empty:
    /// train P. Only for a layout that left P mode unchanged (the P hash
    /// in `encoder::golden_layout_hash`), e.g. a change to V's input.
    pub policy_from: String,
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            device: "cuda".into(),
            batch_size: 512,
            steps: 30_000,
            warmup_steps: 1_000,
            peak_lr: 3e-4,
            min_lr: 1e-5,
            weight_decay: 1e-4,
            augment: true,
            policy_from: String::new(),
        }
    }
}

impl LearnerConfig {
    pub fn validate(&self) -> Result<(), String> {
        parse_device_tag(&self.device)?;
        if self.batch_size == 0 || self.steps == 0 {
            return Err("learner.batch_size and learner.steps must be > 0".into());
        }
        if !(self.peak_lr > 0.0 && (0.0..=self.peak_lr).contains(&self.min_lr)) {
            return Err(format!("learner: need 0 <= min_lr <= peak_lr, peak_lr > 0 (got {} / {})", self.min_lr, self.peak_lr));
        }
        if self.weight_decay < 0.0 {
            return Err("learner.weight_decay must be >= 0".into());
        }
        Ok(())
    }

    /// Whether P trains (no `policy_from`).
    pub fn trains_policy(&self) -> bool {
        self.policy_from.is_empty()
    }

    pub fn schedule(&self) -> LrSchedule {
        LrSchedule { warmup_steps: self.warmup_steps, total_steps: self.steps, peak_lr: self.peak_lr, min_lr: self.min_lr }
    }
}

fn parse_device_tag(tag: &str) -> Result<Device, String> {
    match tag.trim().to_ascii_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::Cuda(0)),
        t => t
            .strip_prefix("cuda:")
            .and_then(|i| i.parse().ok())
            .map(Device::Cuda)
            .ok_or_else(|| format!("unknown device {tag:?}: use cuda, cuda:N or cpu")),
    }
}

/// The device named by `tag`. CUDA must actually be there: libtorch falls
/// back to the CPU silently when `libtorch_cuda.so` isn't preloaded.
pub fn parse_device(tag: &str) -> Result<Device, String> {
    let device = parse_device_tag(tag)?;
    if let Device::Cuda(i) = device {
        if !tch::Cuda::is_available() || i as i64 >= tch::Cuda::device_count() {
            return Err(format!(
                "{tag}: CUDA is not available to libtorch. Preload it: \
                 LD_PRELOAD=$LIBTORCH_DIR/libtorch_cuda.so (AGENTS.md, Runtime environment)"
            ));
        }
    }
    Ok(device)
}

/// P's held-out measurement. `NaN` where a phase has no examples.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct PolicyHeldOut {
    pub bids: usize,
    pub plays: usize,
    /// Mean cross-entropy against the target policies.
    pub bid_loss: f64,
    pub play_loss: f64,
    /// Share of examples whose most likely move is the target's top move.
    pub bid_agreement: f64,
    pub play_agreement: f64,
}

/// V's held-out measurement over every (state, seat) pair. `NaN` where a
/// set is empty.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ValueHeldOut {
    pub states: usize,
    pub seats: usize,
    /// MSE of the predicted ŝ.
    pub mse: f64,
    /// MSE of predicting the set's mean ŝ: the scale for `mse`.
    pub variance: f64,
    /// Pearson correlation of predicted and actual ŝ (G1: > 0.7).
    pub correlation: f64,
    /// Last-trick positions, at the seats the trick still decides (they
    /// need 0 or 1 more trick). Every remaining play is forced, so the deal
    /// decides each such seat's ŝ (G1: ≈ exact). Every round size counts.
    pub last_trick_seats: usize,
    pub last_trick_mse: f64,
    pub last_trick_max_error: f64,
    /// States of 1-card rounds after bidding, every seat: the G1 measure
    /// before layout 4, kept to compare runs.
    pub one_card_states: usize,
    pub one_card_mse: f64,
    pub one_card_max_error: f64,
}

/// Whether `seat` (relative to the seat to move) is one the last trick of
/// `s` still decides: `s` is on its last trick and the seat needs 0 or 1
/// more trick.
pub fn last_trick_decides(s: &BlobState, seat: usize) -> bool {
    if s.phase() != GamePhase::Playing || s.tricks_completed + 1 != s.cards_dealt {
        return false;
    }
    let p = (s.current_player as usize + seat) % s.num_players as usize;
    matches!(s.bids[p] as i32 - s.tricks_won[p] as i32, 0 | 1)
}

/// P and V with their optimizers.
pub struct Learner {
    pub policy: PolicyNet,
    pub value: ValueNet,
    policy_vs: VarStore,
    value_vs: VarStore,
    policy_opt: nn::Optimizer,
    value_opt: nn::Optimizer,
    policy_vars: Vec<Tensor>,
    value_vars: Vec<Tensor>,
    schedule: LrSchedule,
    /// Whether P is updated; `false` with `policy_from`.
    pub trains_policy: bool,
    /// Learner steps taken.
    pub step: u64,
    /// Multiplies the schedule's LR (1 by default), e.g. to warm up again
    /// after a resume restarts AdamW.
    pub lr_scale: f64,
    /// V updates taken, those of learner steps and V-only ones
    /// ([`Learner::train_value_on`]): V's LR follows the schedule at this
    /// count. Equal to `step` when V updates only in learner steps.
    pub value_updates: u64,
    /// `lr_scale` for V.
    pub value_lr_scale: f64,
    pub device: Device,
}

impl Learner {
    /// Fresh networks, or P copied from `cfg.policy_from`; seed with
    /// `tch::manual_seed` first for a reproducible init.
    pub fn new(cfg: &LearnerConfig) -> Result<Self, String> {
        cfg.validate()?;
        let device = parse_device(&cfg.device)?;
        let mut policy_vs = VarStore::new(device);
        let policy = PolicyNet::new(&policy_vs.root());
        if !cfg.trains_policy() {
            let path = PathBuf::from(&cfg.policy_from).join(POLICY_WEIGHTS);
            policy_vs.load(&path).map_err(|e| format!("learner.policy_from: {}: {e}", path.display()))?;
        }
        let value_vs = VarStore::new(device);
        let value = ValueNet::new(&value_vs.root());
        let err = |e: tch::TchError| format!("optimizer: {e}");
        let policy_opt = build_optimizer(&policy_vs, cfg.weight_decay).map_err(err)?;
        let value_opt = build_optimizer(&value_vs, cfg.weight_decay).map_err(err)?;
        Ok(Self {
            policy_vars: policy_vs.trainable_variables(),
            value_vars: value_vs.trainable_variables(),
            policy,
            value,
            policy_vs,
            value_vs,
            policy_opt,
            value_opt,
            schedule: cfg.schedule(),
            trains_policy: cfg.trains_policy(),
            step: 0,
            lr_scale: 1.0,
            value_updates: 0,
            value_lr_scale: 1.0,
            device,
        })
    }

    /// Load a checkpoint's weights and step. The optimizers start afresh.
    pub fn resume(&mut self, dir: &Path) -> Result<(), String> {
        let meta = load_checkpoint(dir, &mut self.policy_vs, &mut self.value_vs)
            .map_err(|e| format!("{}: {e}", dir.display()))?;
        self.step = meta.learner_step;
        self.value_updates = meta.learner_step;
        Ok(())
    }

    pub fn save(&self, dir: &Path) -> Result<(), String> {
        save_checkpoint(dir, &self.policy_vs, &self.value_vs, CheckpointMeta { learner_step: self.step })
            .map_err(|e| format!("{}: {e}", dir.display()))
    }

    /// The LR of the next step: the schedule's, times `lr_scale`.
    pub fn lr(&self) -> f64 {
        self.schedule.lr(self.step) * self.lr_scale
    }

    /// V's LR for its next update: the schedule's at `value_updates`, times
    /// `value_lr_scale`.
    pub fn value_lr(&self) -> f64 {
        self.schedule.lr(self.value_updates) * self.value_lr_scale
    }

    /// One learner step: a P update on `p` and a V update on `v` (each a
    /// sampled `(bids, plays)` pair; leave forced decisions out of `p`).
    /// Returns the two training losses, detached and still on the device:
    /// reading one waits for the GPU. `None` for an empty batch, and for P
    /// when it doesn't train.
    pub fn train_step(&mut self, p: &(BidBatch, PlayBatch), v: &(BidBatch, PlayBatch)) -> (Option<Tensor>, Option<Tensor>) {
        let batches = StepBatches::build(p, v, self.device);
        self.train_on(&batches)
    }

    /// [`Learner::train_step`] on batches built already, on this learner's
    /// device.
    pub fn train_on(&mut self, batches: &StepBatches) -> (Option<Tensor>, Option<Tensor>) {
        let lr = self.lr();
        let p_loss = (self.trains_policy && !batches.policy.is_empty()).then(|| {
            let loss = policy_loss(&self.policy, &batches.policy.iter().collect::<Vec<_>>(), true);
            optimize(&mut self.policy_opt, &self.policy_vars, lr, loss)
        });
        let v_lr = self.value_lr();
        let v_loss = batches.value.as_ref().map(|vb| {
            let loss = value_loss(&self.value, vb, true);
            optimize(&mut self.value_opt, &self.value_vars, v_lr, loss)
        });
        self.value_updates += v_loss.is_some() as u64;
        self.step += 1;
        (p_loss, v_loss)
    }

    /// A V update alone, at [`Learner::value_lr`]. Doesn't count as a learner step
    /// (steps pace publishing and the replay-ratio governor), e.g. for V's
    /// own stream of rounds while P waits for search data. Returns the
    /// training loss, detached, on the device.
    pub fn train_value_on(&mut self, batch: &ValueBatch) -> Tensor {
        let lr = self.value_lr();
        let loss = value_loss(&self.value, batch, true);
        self.value_updates += 1;
        optimize(&mut self.value_opt, &self.value_vars, lr, loss)
    }

    /// P on `buf[indices]`, in chunks of `chunk` examples.
    pub fn policy_held_out(&self, buf: &ReplayBuffer, indices: &[usize], chunk: usize) -> PolicyHeldOut {
        // [examples, loss sum, agreements] per phase.
        let mut acc = [[0.0f64; 3]; 2];
        for idx in indices.chunks(chunk.max(1)) {
            let (bid, play) = buf.batch_from_indices(idx);
            let parts = [bid_policy_batch(&bid, self.device), play_policy_batch(&play, self.device)];
            for (slot, pb) in parts.iter().enumerate() {
                let Some(pb) = pb else { continue };
                let (ce, agree) = tch::no_grad(|| {
                    let probs = policy_probs(&self.policy, pb, false);
                    let ce = policy_cross_entropy_rows(&probs, &pb.target).sum(Kind::Double);
                    let agree = probs.argmax(-1, false).eq_tensor(&pb.target.argmax(-1, false)).sum(Kind::Double);
                    (ce.double_value(&[]), agree.double_value(&[]))
                });
                acc[slot][0] += pb.rows() as f64;
                acc[slot][1] += ce;
                acc[slot][2] += agree;
            }
        }
        let mean = |a: &[f64; 3], i: usize| if a[0] > 0.0 { a[i] / a[0] } else { f64::NAN };
        PolicyHeldOut {
            bids: acc[0][0] as usize,
            plays: acc[1][0] as usize,
            bid_loss: mean(&acc[0], 1),
            play_loss: mean(&acc[1], 1),
            bid_agreement: mean(&acc[0], 2),
            play_agreement: mean(&acc[1], 2),
        }
    }

    /// V on `buf[indices]`, in chunks of `chunk` examples.
    pub fn value_held_out(&self, buf: &ReplayBuffer, indices: &[usize], chunk: usize) -> ValueHeldOut {
        let (mut states, mut n, mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0usize, 0usize, 0.0, 0.0, 0.0, 0.0, 0.0f64);
        let (mut one_states, mut one_n, mut one_se, mut one_max) = (0usize, 0usize, 0.0f64, 0.0f64);
        let (mut last_n, mut last_se, mut last_max) = (0usize, 0.0f64, 0.0f64);
        for idx in indices.chunks(chunk.max(1)) {
            let (bid, play) = buf.batch_from_indices(idx);
            let Some(vb) = value_batch(&bid, &play, self.device) else { continue };
            let k = vb.target.size()[1] as usize;
            let pred: Vec<f32> = tch::no_grad(|| self.value.seat_values(&vb.input, k as i64, false))
                .to_device(Device::Cpu)
                .flatten(0, -1)
                .try_into()
                .expect("f32 values");
            let rows = bid.states.iter().chain(&play.states).zip(bid.seat_scores.iter().chain(&play.seat_scores));
            for (row, (s, target)) in rows.enumerate() {
                states += 1;
                let one_card = s.cards_dealt == 1 && s.phase() == GamePhase::Playing;
                one_states += one_card as usize;
                for seat in 0..s.num_players as usize {
                    let (x, y) = (pred[row * k + seat] as f64, target[seat] as f64);
                    n += 1;
                    (sx, sy, sxx, syy, sxy) = (sx + x, sy + y, sxx + x * x, syy + y * y, sxy + x * y);
                    if one_card {
                        one_n += 1;
                        one_se += (x - y).powi(2);
                        one_max = one_max.max((x - y).abs());
                    }
                    if last_trick_decides(s, seat) {
                        last_n += 1;
                        last_se += (x - y).powi(2);
                        last_max = last_max.max((x - y).abs());
                    }
                }
            }
        }
        let nf = n as f64;
        let mean = |sum: f64, k: usize| if k > 0 { sum / k as f64 } else { f64::NAN };
        let (vx, vy) = (sxx / nf - (sx / nf).powi(2), syy / nf - (sy / nf).powi(2));
        ValueHeldOut {
            states,
            seats: n,
            mse: mean(sxx - 2.0 * sxy + syy, n),
            variance: if n > 0 { vy } else { f64::NAN },
            correlation: if n > 0 { (sxy / nf - sx / nf * sy / nf) / (vx * vy).sqrt() } else { f64::NAN },
            last_trick_seats: last_n,
            last_trick_mse: mean(last_se, last_n),
            last_trick_max_error: if last_n > 0 { last_max } else { f64::NAN },
            one_card_states: one_states,
            one_card_mse: mean(one_se, one_n),
            one_card_max_error: if one_n > 0 { one_max } else { f64::NAN },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blob_engine::bidding::apply_bid;
    use blob_engine::replay::{Decision, SparsePolicy};
    use blob_engine::scoring::round_points;
    use blob_engine::{fill_buffer, new_round, RoundParams, TeacherConfig};
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use smallvec::smallvec;

    fn bidding_state(seed: u64) -> BlobState {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        new_round(RoundParams { num_players: 4, cards_dealt: 5, trump: 0, dealer: 3 }, &mut rng).unwrap()
    }

    /// First trick, first card: every card in the leader's hand is legal.
    fn playing_state(seed: u64) -> BlobState {
        let mut s = bidding_state(seed);
        while s.phase() == GamePhase::Bidding {
            let bid = legal_bids(&s).trailing_zeros() as u8;
            apply_bid(&mut s, bid);
        }
        s
    }

    /// Push `s` as a one-decision round in which every seat made its bid
    /// (4 players, 5 cards: every seat's ŝ is (10 + bid) / 15).
    fn push(buf: &mut ReplayBuffer, s: BlobState, policy: SparsePolicy) {
        let mut end = s;
        end.game_phase = GamePhase::Scoring as u8;
        end.tricks_won = end.bids;
        buf.push_round(&[Decision { state: s, policy }], &end);
    }

    fn cpu_config() -> LearnerConfig {
        LearnerConfig { device: "cpu".into(), batch_size: 16, steps: 40, warmup_steps: 5, peak_lr: 1e-3, ..Default::default() }
    }

    #[test]
    fn validation_split_is_deterministic_and_near_fraction() {
        let held: usize = (0..100_000u64).filter(|&r| is_validation_round(r, 0.03)).count();
        assert!((2_700..3_300).contains(&held), "held out {held} of 100k");
        for r in 0..1000u64 {
            assert_eq!(is_validation_round(r, 0.03), is_validation_round(r, 0.03));
            assert!(!is_validation_round(r, 0.0));
            assert!(is_validation_round(r, 1.0));
        }
        // A larger fraction holds out a superset.
        assert!((0..10_000u64).all(|r| !is_validation_round(r, 0.03) || is_validation_round(r, 0.1)));
    }

    #[test]
    fn config_rejects_unknown_keys_and_bad_values() {
        let cfg: LearnerConfig = toml::from_str("device = \"cpu\"\nsteps = 7\n").unwrap();
        assert_eq!((cfg.steps, cfg.batch_size), (7, LearnerConfig::default().batch_size));
        assert!(toml::from_str::<LearnerConfig>("step = 7\n").is_err());
        assert!(LearnerConfig { device: "gpu".into(), ..cpu_config() }.validate().is_err());
        assert!(LearnerConfig { min_lr: 1.0, ..cpu_config() }.validate().is_err());
        assert_eq!(parse_device_tag("cuda:1"), Ok(Device::Cuda(1)));
    }

    #[test]
    fn bid_batch_masks_exactly_the_legal_bids() {
        let mut buf = ReplayBuffer::new(8);
        let states: Vec<BlobState> = (0..3).map(bidding_state).collect();
        for s in &states {
            push(&mut buf, *s, smallvec![(0u8, 0.5f32), (1, 0.5)]);
        }
        let (bid, play) = buf.batch_from_indices(&[0, 1, 2]);
        assert!(play_policy_batch(&play, Device::Cpu).is_none());
        let pb = bid_policy_batch(&bid, Device::Cpu).unwrap();
        assert_eq!(pb.phase, Phase::Bidding);
        assert_eq!(pb.legal_mask.size(), vec![3, NUM_BIDS]);
        let mask: Vec<bool> = pb.legal_mask.flatten(0, -1).try_into().unwrap();
        for (row, s) in states.iter().enumerate() {
            let legal = legal_bids(s);
            for b in 0..NUM_BIDS as usize {
                assert_eq!(mask[row * NUM_BIDS as usize + b], (legal >> b) & 1 == 1);
            }
        }
        let target: Vec<f32> = pb.target.flatten(0, -1).try_into().unwrap();
        assert_eq!(&target[..3], &[0.5, 0.5, 0.0]);
    }

    /// Hand-position policies land on the matching hand-card tokens, in
    /// `hand_card_indices` order, and the mask covers exactly the legal ones.
    #[test]
    fn play_batch_scatters_policy_onto_hand_tokens() {
        let mut buf = ReplayBuffer::new(8);
        let states: Vec<BlobState> = (10..13).map(playing_state).collect();
        for s in &states {
            push(&mut buf, *s, smallvec![(1u8, 0.25f32), (3, 0.75)]);
        }
        let (_, play) = buf.batch_from_indices(&[0, 1, 2]);
        let pb = play_policy_batch(&play, Device::Cpu).unwrap();
        assert_eq!(pb.phase, Phase::Playing);
        let seq_len = pb.legal_mask.size()[1] as usize;
        let mask: Vec<bool> = pb.legal_mask.flatten(0, -1).try_into().unwrap();
        let target: Vec<f32> = pb.target.flatten(0, -1).try_into().unwrap();
        for (row, s) in states.iter().enumerate() {
            let enc = encode(s, s.current_player);
            let legal = legal_plays(s);
            let hand_positions: Vec<usize> =
                (0..enc.num_tokens).filter(|&i| enc.token_types[i] == TOKEN_TYPE_HAND).collect();
            assert_eq!(hand_positions.len(), enc.hand_card_indices.len());
            for i in 0..seq_len {
                let slot = hand_positions.iter().position(|&p| p == i);
                let want_target = match slot {
                    Some(1) => 0.25,
                    Some(3) => 0.75,
                    _ => 0.0,
                };
                assert_eq!(target[row * seq_len + i], want_target, "row {row} pos {i}");
                let want_mask = slot.is_some_and(|h| (legal >> enc.hand_card_indices[h]) & 1 == 1);
                assert_eq!(mask[row * seq_len + i], want_mask, "row {row} pos {i}");
            }
        }
    }

    /// V's targets are each seat's ŝ from the mover's seat on, masked to the
    /// table; the inputs show every hand.
    #[test]
    fn value_batch_targets_every_real_seat() {
        let mut buf = ReplayBuffer::new(64);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(5);
        let mut ends = Vec::new();
        for (n, cards) in [(3u8, 4u8), (5, 2)] {
            let mut s = new_round(RoundParams { num_players: n, cards_dealt: cards, trump: 1, dealer: 0 }, &mut rng).unwrap();
            let mut decisions = Vec::new();
            while matches!(s.phase(), GamePhase::Bidding | GamePhase::Playing) {
                decisions.push(Decision { state: s, policy: smallvec![(0u8, 1.0f32)] });
                let action = blob_engine::rule_bot_action(&s);
                blob_engine::mcts::apply_action(&mut s, action);
            }
            buf.push_round(&decisions, &s);
            ends.push(s);
        }
        let all: Vec<usize> = (0..buf.len()).collect();
        let (bid, play) = buf.batch_from_indices(&all);
        let vb = value_batch(&bid, &play, Device::Cpu).unwrap();
        assert_eq!(vb.target.size(), vec![buf.len() as i64, 5]);
        let target: Vec<f32> = vb.target.flatten(0, -1).try_into().unwrap();
        let mask: Vec<bool> = vb.seat_mask.flatten(0, -1).try_into().unwrap();
        for (row, s) in bid.states.iter().chain(&play.states).enumerate() {
            let end = ends.iter().find(|e| e.num_players == s.num_players).unwrap();
            let points = round_points(end);
            for seat in 0..5 {
                let real = seat < s.num_players as usize;
                assert_eq!(mask[row * 5 + seat], real);
                let abs = (s.current_player as usize + seat) % s.num_players as usize;
                let want = if real { points[abs] as f32 / (10.0 + s.cards_dealt as f32) } else { 0.0 };
                assert_eq!(target[row * 5 + seat], want);
            }
        }
        let types: Vec<i64> = vb.input.token_types.flatten(0, -1).try_into().unwrap();
        assert!(types.contains(&(blob_engine::encoder::TOKEN_TYPE_OPP_HAND as i64)));
    }

    #[test]
    fn last_trick_decides_seats_needing_zero_or_one_trick() {
        let mut s = playing_state(4);
        assert!(!(0..4).any(|seat| last_trick_decides(&s, seat)), "first of five tricks");
        s.tricks_completed = 4;
        s.current_player = 1;
        s.bids[..4].copy_from_slice(&[2, 1, 3, 0]);
        s.tricks_won[..4].copy_from_slice(&[2, 0, 1, 1]);
        // Relative seat k is absolute seat (1 + k) % 4: needs 1, 2, -1, 0.
        let decided: Vec<bool> = (0..4).map(|seat| last_trick_decides(&s, seat)).collect();
        assert_eq!(decided, [true, false, false, true]);
    }

    #[test]
    fn forced_means_one_legal_move() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(9);
        let mut s = new_round(RoundParams { num_players: 4, cards_dealt: 1, trump: 0, dealer: 0 }, &mut rng).unwrap();
        assert!(!is_forced(&s));
        while s.phase() == GamePhase::Bidding {
            apply_bid(&mut s, 0);
        }
        assert!(is_forced(&s), "a 1-card hand has one play");
    }

    /// Teacher data through the whole learner: both losses fall on a small
    /// buffer, held-out sets cover every example, and a checkpoint resumes
    /// at its step.
    #[test]
    fn learner_trains_measures_and_resumes() {
        tch::manual_seed(11);
        let mut buf = ReplayBuffer::new(4_000);
        let teacher = TeacherConfig { mix: blob_engine::RoundMix { players: vec![4], start_cards: 3, large_round_exponent: 0.0 }, ..Default::default() };
        fill_buffer(&mut buf, &teacher, 60, 1, 4);
        let all: Vec<usize> = (0..buf.len()).collect();
        let unforced: Vec<usize> = all.iter().copied().filter(|&i| !is_forced(buf.state(i))).collect();

        let cfg = cpu_config();
        let mut learner = Learner::new(&cfg).unwrap();
        let p0 = learner.policy_held_out(&buf, &unforced, 64);
        let v0 = learner.value_held_out(&buf, &all, 64);
        assert_eq!(p0.bids + p0.plays, unforced.len());
        assert_eq!(v0.states, all.len());
        assert!(v0.one_card_states > 0 && v0.last_trick_seats > 0 && v0.correlation.is_finite());

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2);
        for _ in 0..cfg.steps {
            let p = buf.sample_batch_from(&unforced, cfg.batch_size, &mut rng, cfg.augment);
            let v = buf.sample_batch_from(&all, cfg.batch_size, &mut rng, cfg.augment);
            let (pl, vl) = learner.train_step(&p, &v);
            assert!(pl.unwrap().double_value(&[]).is_finite() && vl.unwrap().double_value(&[]).is_finite());
        }
        let p1 = learner.policy_held_out(&buf, &unforced, 64);
        let v1 = learner.value_held_out(&buf, &all, 64);
        assert!(p1.bid_loss + p1.play_loss < p0.bid_loss + p0.play_loss, "{p0:?} -> {p1:?}");
        assert!(v1.mse < v0.mse, "{v0:?} -> {v1:?}");
        assert!((v1.variance - v0.variance).abs() < 1e-9, "variance is the targets'");

        // Chunking doesn't change the result.
        let whole = learner.value_held_out(&buf, &all, 10_000);
        assert!((whole.mse - v1.mse).abs() < 1e-6 && (whole.correlation - v1.correlation).abs() < 1e-6);

        let dir = std::env::temp_dir().join(format!("blob-learner-{}", std::process::id()));
        learner.save(&dir).unwrap();
        let mut again = Learner::new(&cfg).unwrap();
        again.resume(&dir).unwrap();
        assert_eq!((again.step, again.lr()), (cfg.steps, learner.lr()));
        let v2 = again.value_held_out(&buf, &all, 64);
        assert_eq!(v2.mse, v1.mse);

        // `policy_from` copies P and leaves it untrained; V still trains.
        let frozen_cfg = LearnerConfig { policy_from: dir.display().to_string(), ..cfg.clone() };
        let mut frozen = Learner::new(&frozen_cfg).unwrap();
        assert!(!frozen.trains_policy);
        let fp0 = frozen.policy_held_out(&buf, &unforced, 64);
        assert_eq!((fp0.bid_loss, fp0.play_loss), (p1.bid_loss, p1.play_loss), "P is the checkpoint's");
        let fv0 = frozen.value_held_out(&buf, &all, 64);
        for _ in 0..5 {
            let p = buf.sample_batch_from(&unforced, cfg.batch_size, &mut rng, cfg.augment);
            let v = buf.sample_batch_from(&all, cfg.batch_size, &mut rng, cfg.augment);
            let (pl, vl) = frozen.train_step(&p, &v);
            assert!(pl.is_none() && vl.is_some());
        }
        let fp1 = frozen.policy_held_out(&buf, &unforced, 64);
        assert_eq!((fp1.bid_loss, fp1.play_loss), (fp0.bid_loss, fp0.play_loss), "P unchanged");
        assert_ne!(frozen.value_held_out(&buf, &all, 64).mse, fv0.mse, "V trained");
        assert!(Learner::new(&LearnerConfig { policy_from: "/nonexistent".into(), ..cfg.clone() }).is_err());
        let _ = std::fs::remove_dir_all(&dir);

        // An empty set reports NaN, not 0.
        let empty = learner.value_held_out(&buf, &[], 64);
        assert_eq!(empty.states, 0);
        assert!(empty.mse.is_nan() && learner.policy_held_out(&buf, &[], 64).bid_loss.is_nan());
    }
}
