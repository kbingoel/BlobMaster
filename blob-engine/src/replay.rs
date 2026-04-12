//! Replay buffer for self-play training examples.
//!
//! Session 5.1: stores raw `BlobState` snapshots along with their MCTS
//! policies, backfilled value targets, and the `GamePhase` the decision was
//! taken in. The buffer is a circular FIFO — once capacity is reached, new
//! writes overwrite the oldest entries.
//!
//! Design rationale (see development-plan.md §5.1): storing raw states plus
//! sparse policies costs ~410B + small policy per example, so 500K examples
//! fits in ~250MB. Re-encoding to entity tokens happens at batch construction
//! time, which decouples encoder changes from buffer compatibility.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use rand::seq::IteratorRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::state::{BlobState, GamePhase};

pub const MAX_BID_ACTIONS: usize = 14;
pub const MAX_PLAY_ACTIONS: usize = 13;

pub type SparsePolicy = SmallVec<[(u8, f32); MAX_BID_ACTIONS]>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BidBatch {
    pub states: Vec<BlobState>,
    /// Flattened dense policy tensor: `states.len() * MAX_BID_ACTIONS`, row-major.
    pub policies: Vec<f32>,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayBatch {
    pub states: Vec<BlobState>,
    /// Flattened dense policy tensor: `states.len() * max_hand_size`, row-major.
    pub policies: Vec<f32>,
    pub values: Vec<f32>,
    /// Column count of the `policies` tensor — the largest hand size in the batch.
    pub max_hand_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBuffer {
    states: Vec<BlobState>,
    policies: Vec<SparsePolicy>,
    values: Vec<f32>,
    phases: Vec<GamePhase>,
    capacity: usize,
    write_idx: usize,
    len: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "replay buffer capacity must be > 0");
        Self {
            states: Vec::with_capacity(capacity),
            policies: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
            phases: Vec::with_capacity(capacity),
            capacity,
            write_idx: 0,
            len: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn push(
        &mut self,
        state: BlobState,
        policy: SparsePolicy,
        value: f32,
        phase: GamePhase,
    ) {
        debug_assert!(
            matches!(phase, GamePhase::Bidding | GamePhase::Playing),
            "replay buffer only accepts decision-point phases (Bidding/Playing)"
        );
        if self.len < self.capacity {
            self.states.push(state);
            self.policies.push(policy);
            self.values.push(value);
            self.phases.push(phase);
            self.len += 1;
        } else {
            let idx = self.write_idx;
            self.states[idx] = state;
            self.policies[idx] = policy;
            self.values[idx] = value;
            self.phases[idx] = phase;
        }
        self.write_idx = (self.write_idx + 1) % self.capacity;
    }

    /// Uniformly sample `n` indices (without replacement if `n <= len`, else with replacement)
    /// and split into per-phase dense batches.
    pub fn sample_batch<R: Rng + ?Sized>(&self, n: usize, rng: &mut R) -> (BidBatch, PlayBatch) {
        assert!(self.len > 0, "cannot sample from empty replay buffer");
        let indices: Vec<usize> = if n <= self.len {
            (0..self.len).choose_multiple(rng, n)
        } else {
            (0..n).map(|_| rng.gen_range(0..self.len)).collect()
        };

        let mut bid_idx = Vec::new();
        let mut play_idx = Vec::new();
        for &i in &indices {
            match self.phases[i] {
                GamePhase::Bidding => bid_idx.push(i),
                GamePhase::Playing => play_idx.push(i),
                _ => {}
            }
        }

        let bid_batch = self.build_bid_batch(&bid_idx);
        let play_batch = self.build_play_batch(&play_idx);
        (bid_batch, play_batch)
    }

    fn build_bid_batch(&self, indices: &[usize]) -> BidBatch {
        let n = indices.len();
        let mut states = Vec::with_capacity(n);
        let mut policies = vec![0.0_f32; n * MAX_BID_ACTIONS];
        let mut values = Vec::with_capacity(n);
        for (row, &i) in indices.iter().enumerate() {
            states.push(self.states[i].clone());
            values.push(self.values[i]);
            let base = row * MAX_BID_ACTIONS;
            for &(action, prob) in &self.policies[i] {
                let a = action as usize;
                assert!(a < MAX_BID_ACTIONS, "bid action index out of range");
                policies[base + a] = prob;
            }
        }
        BidBatch {
            states,
            policies,
            values,
        }
    }

    fn build_play_batch(&self, indices: &[usize]) -> PlayBatch {
        let n = indices.len();
        let max_hand_size = indices
            .iter()
            .flat_map(|&i| self.policies[i].iter().map(|&(a, _)| a as usize + 1))
            .max()
            .unwrap_or(0);
        let cols = max_hand_size.max(1);
        let mut states = Vec::with_capacity(n);
        let mut policies = vec![0.0_f32; n * cols];
        let mut values = Vec::with_capacity(n);
        for (row, &i) in indices.iter().enumerate() {
            states.push(self.states[i].clone());
            values.push(self.values[i]);
            let base = row * cols;
            for &(action, prob) in &self.policies[i] {
                policies[base + action as usize] = prob;
            }
        }
        PlayBatch {
            states,
            policies,
            values,
            max_hand_size: cols,
        }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<bincode::ErrorKind>> {
        let file = File::create(path).map_err(|e| Box::new(bincode::ErrorKind::Io(e)))?;
        let mut writer = BufWriter::new(file);
        bincode::serialize_into(&mut writer, self)
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<bincode::ErrorKind>> {
        let file = File::open(path).map_err(|e| Box::new(bincode::ErrorKind::Io(e)))?;
        let reader = BufReader::new(file);
        bincode::deserialize_from(reader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::BlobState;
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use smallvec::smallvec;

    fn dummy_state() -> BlobState {
        BlobState::empty()
    }

    fn bid_policy(bid: u8) -> SparsePolicy {
        smallvec![(bid, 1.0_f32)]
    }

    fn play_policy(pos: u8, hand_size: u8) -> SparsePolicy {
        let mut p: SparsePolicy = SmallVec::new();
        let uniform = 1.0_f32 / hand_size as f32;
        for i in 0..hand_size {
            p.push((i, uniform));
        }
        // nudge one position so the argmax is well-defined
        if (pos as usize) < p.len() {
            p[pos as usize].1 += 0.0;
        }
        p
    }

    #[test]
    fn push_and_len() {
        let mut buf = ReplayBuffer::new(4);
        assert!(buf.is_empty());
        for i in 0..3 {
            buf.push(dummy_state(), bid_policy(i), 0.1, GamePhase::Bidding);
        }
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn circular_fifo_overwrites_oldest() {
        let mut buf = ReplayBuffer::new(3);
        // fill then overwrite: values 0,1,2,3,4 into capacity-3 buffer → stored: 3,4,2 (at idx 0,1,2)
        for i in 0..5 {
            buf.push(dummy_state(), bid_policy(0), i as f32, GamePhase::Bidding);
        }
        assert_eq!(buf.len(), 3);
        // Oldest two (values 0, 1) were overwritten; remaining set is {2, 3, 4}.
        let mut vals = buf.values.clone();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(vals, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn sample_batch_splits_by_phase() {
        let mut buf = ReplayBuffer::new(100);
        for i in 0..50 {
            buf.push(dummy_state(), bid_policy((i % 14) as u8), 0.25, GamePhase::Bidding);
        }
        for i in 0..50 {
            let hs = ((i % 5) + 3) as u8; // hand sizes 3..=7
            buf.push(
                dummy_state(),
                play_policy(i as u8 % hs, hs),
                -0.5,
                GamePhase::Playing,
            );
        }
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let (bid, play) = buf.sample_batch(40, &mut rng);
        assert_eq!(bid.states.len() + play.states.len(), 40);
        assert_eq!(bid.policies.len(), bid.states.len() * MAX_BID_ACTIONS);
        assert_eq!(play.policies.len(), play.states.len() * play.max_hand_size);
        // Each bid row sums to 1.0 (single (action, 1.0) entry).
        for row in 0..bid.states.len() {
            let s: f32 = bid.policies[row * MAX_BID_ACTIONS..(row + 1) * MAX_BID_ACTIONS]
                .iter()
                .sum();
            assert!((s - 1.0).abs() < 1e-5);
        }
        // Each play row sums to ~1.0 (uniform over the hand).
        for row in 0..play.states.len() {
            let s: f32 = play.policies[row * play.max_hand_size..(row + 1) * play.max_hand_size]
                .iter()
                .sum();
            assert!((s - 1.0).abs() < 1e-5, "play row {row} sum = {s}");
        }
    }

    #[test]
    fn uniform_sampling_chi_squared() {
        // Fill a buffer of size K with distinct tag-values, draw a large sample with
        // replacement, and chi-squared test against uniform. Critical value for
        // df=19, α=0.001 is ≈43.82.
        const K: usize = 20;
        const DRAWS: usize = 100_000;
        let mut buf = ReplayBuffer::new(K);
        for i in 0..K {
            buf.push(dummy_state(), bid_policy(0), i as f32, GamePhase::Bidding);
        }
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let mut counts = vec![0u32; K];
        // Use repeated single-draw with replacement so counts reflect the index RNG path.
        for _ in 0..DRAWS {
            let idx = rng.gen_range(0..buf.len);
            counts[idx] += 1;
        }
        let expected = DRAWS as f64 / K as f64;
        let chi2: f64 = counts
            .iter()
            .map(|&c| {
                let diff = c as f64 - expected;
                diff * diff / expected
            })
            .sum();
        assert!(chi2 < 43.82, "chi2 = {chi2} (df=19, α=0.001 cutoff 43.82)");
    }

    #[test]
    fn roundtrip_serialization() {
        let mut buf = ReplayBuffer::new(8);
        for i in 0..6 {
            buf.push(dummy_state(), bid_policy((i % 14) as u8), i as f32 * 0.1, GamePhase::Bidding);
        }
        let dir = std::env::temp_dir();
        let path = dir.join("blobmaster_replay_roundtrip.bin");
        buf.save(&path).expect("save");
        let restored = ReplayBuffer::load(&path).expect("load");
        std::fs::remove_file(&path).ok();
        assert_eq!(restored.len(), buf.len());
        assert_eq!(restored.capacity(), buf.capacity());
        assert_eq!(restored.values, buf.values);
        assert_eq!(restored.phases.len(), buf.phases.len());
    }
}
