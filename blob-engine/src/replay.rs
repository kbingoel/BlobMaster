//! Replay buffer of training examples (gen-2.md §5.6).
//!
//! An example is one decision: the raw `BlobState` (every hand), its policy
//! target (search visits at τ = 1, or a teacher's policy), and the points
//! every seat scored in that round. A round's examples are pushed together
//! once it ends ([`ReplayBuffer::push_round`]), so targets are written 10–40
//! decisions after the fact rather than at the end of a whole game. Each
//! round gets an id, which the learner's validation split hashes
//! (`blob_nn::learner::is_validation_round`).
//!
//! - **Circular FIFO:** once full, new examples overwrite the oldest.
//! - **Raw states:** ~430 B per example, so 500k examples take ~215 MB.
//!   Encoding happens when batches are built, so an encoder change never
//!   invalidates a buffer.
//! - **Augmentation:** sampling can relabel suits ([`crate::augment`]), one
//!   random relabelling per sampled example, with its play policy reordered
//!   to match. Targets don't change under a relabelling.
//! - **Sharing:** [`SharedReplay`] lets actors push while the learner samples.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::{RwLock, RwLockReadGuard};

use rand::seq::IteratorRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::augment::{hand_position_map, permute_suits, random_suit_perm};
use crate::scoring::{round_points, score_scale};
use crate::state::{BlobState, GamePhase, MAX_PLAYERS};

pub const MAX_BID_ACTIONS: usize = 14;
pub const MAX_PLAY_ACTIONS: usize = 13;

/// `(action, probability)` pairs: bids by value, plays by hand position in
/// `Hand::iter` order.
pub type SparsePolicy = SmallVec<[(u8, f32); MAX_BID_ACTIONS]>;

/// ŝ (round points / (10 + cards dealt)) of every seat, relative to the
/// seat to move: index 0 is `state.current_player`, then the seats after it
/// in play order; 0 beyond `num_players`. The value net's output order.
pub type SeatScores = [f32; MAX_PLAYERS];

/// One decision of a round, recorded as it is made.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    /// The true state before the move, every hand included; phase `Bidding`
    /// or `Playing`.
    pub state: BlobState,
    pub policy: SparsePolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BidBatch {
    pub states: Vec<BlobState>,
    /// Flattened dense policy tensor: `states.len() * MAX_BID_ACTIONS`, row-major.
    pub policies: Vec<f32>,
    pub seat_scores: Vec<SeatScores>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayBatch {
    pub states: Vec<BlobState>,
    /// Flattened dense policy tensor: `states.len() * max_hand_size`, row-major.
    pub policies: Vec<f32>,
    pub seat_scores: Vec<SeatScores>,
    /// Column count of the `policies` tensor — the largest hand position
    /// with a policy entry in the batch, plus one.
    pub max_hand_size: usize,
}

/// One example as batches are built from it.
struct Example {
    state: BlobState,
    policy: SparsePolicy,
    points: [u8; MAX_PLAYERS],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBuffer {
    states: Vec<BlobState>,
    policies: Vec<SparsePolicy>,
    /// Points each seat scored in the example's round, by absolute seat.
    points: Vec<[u8; MAX_PLAYERS]>,
    round_ids: Vec<u64>,
    capacity: usize,
    write_idx: usize,
    len: usize,
    /// The next round's id. Saved with the buffer, so ids stay unique
    /// across a resume.
    next_round_id: u64,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "replay buffer capacity must be > 0");
        Self {
            states: Vec::with_capacity(capacity),
            policies: Vec::with_capacity(capacity),
            points: Vec::with_capacity(capacity),
            round_ids: Vec::with_capacity(capacity),
            capacity,
            write_idx: 0,
            len: 0,
            next_round_id: 0,
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

    /// Rounds pushed so far, including those overwritten since.
    pub fn rounds_pushed(&self) -> u64 {
        self.next_round_id
    }

    /// Round id of the example in slot `i` (`< len`).
    pub fn round_id(&self, i: usize) -> u64 {
        self.round_ids[i]
    }

    /// Store every decision of a finished round, with the points each seat
    /// scored in it, read from `end` (the round's final state, in
    /// `Scoring`). Returns the round's id.
    pub fn push_round(&mut self, decisions: &[Decision], end: &BlobState) -> u64 {
        assert_eq!(end.phase(), GamePhase::Scoring, "push_round needs the finished round");
        let points = round_points(end);
        let id = self.next_round_id;
        self.next_round_id += 1;
        for d in decisions {
            debug_assert!(
                matches!(d.state.phase(), GamePhase::Bidding | GamePhase::Playing),
                "replay buffer only accepts decision-point phases (Bidding/Playing)"
            );
            debug_assert_eq!(
                (d.state.num_players, d.state.cards_dealt, d.state.dealer),
                (end.num_players, end.cards_dealt, end.dealer),
                "decision from another round"
            );
            if self.len < self.capacity {
                self.states.push(d.state);
                self.policies.push(d.policy.clone());
                self.points.push(points);
                self.round_ids.push(id);
                self.len += 1;
            } else {
                let i = self.write_idx;
                self.states[i] = d.state;
                self.policies[i] = d.policy.clone();
                self.points[i] = points;
                self.round_ids[i] = id;
            }
            self.write_idx = (self.write_idx + 1) % self.capacity;
        }
        id
    }

    /// Uniformly sample `n` examples (without replacement if `n <= len`,
    /// else with replacement) and split them into per-phase dense batches.
    /// With `augment`, each example gets its own random suit relabelling.
    pub fn sample_batch<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        augment: bool,
    ) -> (BidBatch, PlayBatch) {
        assert!(self.len > 0, "cannot sample from empty replay buffer");
        let indices: Vec<usize> = if n <= self.len {
            (0..self.len).choose_multiple(rng, n)
        } else {
            (0..n).map(|_| rng.gen_range(0..self.len)).collect()
        };
        let examples = indices
            .iter()
            .map(|&i| {
                let ex = self.example(i);
                if augment {
                    relabel(ex, rng)
                } else {
                    ex
                }
            })
            .collect();
        build_batches(examples)
    }

    /// Per-phase dense batches for the given slots (each `< len`), in order
    /// and without augmentation. Used to sweep a whole set, e.g. the
    /// validation rounds.
    pub fn batch_from_indices(&self, indices: &[usize]) -> (BidBatch, PlayBatch) {
        build_batches(indices.iter().map(|&i| self.example(i)).collect())
    }

    fn example(&self, i: usize) -> Example {
        Example { state: self.states[i], policy: self.policies[i].clone(), points: self.points[i] }
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

/// `ex` under a random suit relabelling: the state relabelled, a play
/// policy moved to its cards' new hand positions. Bids and points don't
/// depend on suits.
fn relabel<R: Rng + ?Sized>(ex: Example, rng: &mut R) -> Example {
    let perm = random_suit_perm(rng);
    let policy = if ex.state.phase() == GamePhase::Playing {
        let to = hand_position_map(ex.state.hands[ex.state.current_player as usize], &perm);
        ex.policy.iter().map(|&(pos, p)| (to[pos as usize], p)).collect()
    } else {
        ex.policy
    };
    Example { state: permute_suits(&ex.state, &perm), policy, points: ex.points }
}

/// ŝ of every seat from `state`'s seat to move on (see [`SeatScores`]).
fn seat_scores(state: &BlobState, points: &[u8; MAX_PLAYERS]) -> SeatScores {
    let n = state.num_players as usize;
    let me = state.current_player as usize;
    let scale = score_scale(state.cards_dealt);
    let mut out = [0.0f32; MAX_PLAYERS];
    for (rel, slot) in out.iter_mut().enumerate().take(n) {
        *slot = points[(me + rel) % n] as f32 / scale;
    }
    out
}

fn build_batches(examples: Vec<Example>) -> (BidBatch, PlayBatch) {
    let (bids, plays): (Vec<Example>, Vec<Example>) =
        examples.into_iter().partition(|e| e.state.phase() == GamePhase::Bidding);

    let mut bid = BidBatch {
        states: Vec::with_capacity(bids.len()),
        policies: vec![0.0; bids.len() * MAX_BID_ACTIONS],
        seat_scores: Vec::with_capacity(bids.len()),
    };
    for (row, e) in bids.iter().enumerate() {
        bid.states.push(e.state);
        bid.seat_scores.push(seat_scores(&e.state, &e.points));
        for &(action, prob) in &e.policy {
            assert!((action as usize) < MAX_BID_ACTIONS, "bid action index out of range");
            bid.policies[row * MAX_BID_ACTIONS + action as usize] = prob;
        }
    }

    let cols = plays
        .iter()
        .flat_map(|e| e.policy.iter().map(|&(a, _)| a as usize + 1))
        .max()
        .unwrap_or(0)
        .max(1);
    let mut play = PlayBatch {
        states: Vec::with_capacity(plays.len()),
        policies: vec![0.0; plays.len() * cols],
        seat_scores: Vec::with_capacity(plays.len()),
        max_hand_size: cols,
    };
    for (row, e) in plays.iter().enumerate() {
        play.states.push(e.state);
        play.seat_scores.push(seat_scores(&e.state, &e.points));
        for &(action, prob) in &e.policy {
            play.policies[row * cols + action as usize] = prob;
        }
    }
    (bid, play)
}

/// The replay buffer shared by actors (writers) and the learner (reader).
#[derive(Debug)]
pub struct SharedReplay(RwLock<ReplayBuffer>);

impl SharedReplay {
    pub fn new(buf: ReplayBuffer) -> Self {
        Self(RwLock::new(buf))
    }

    /// [`ReplayBuffer::push_round`] under the write lock.
    pub fn push_round(&self, decisions: &[Decision], end: &BlobState) -> u64 {
        self.0.write().expect("replay lock poisoned").push_round(decisions, end)
    }

    /// [`ReplayBuffer::sample_batch`] under a read lock.
    pub fn sample_batch<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        augment: bool,
    ) -> (BidBatch, PlayBatch) {
        self.read().sample_batch(n, rng, augment)
    }

    pub fn len(&self) -> usize {
        self.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.read().is_empty()
    }

    /// Read access for longer jobs: a validation sweep, saving.
    pub fn read(&self) -> RwLockReadGuard<'_, ReplayBuffer> {
        self.0.read().expect("replay lock poisoned")
    }

    pub fn into_inner(self) -> ReplayBuffer {
        self.0.into_inner().expect("replay lock poisoned")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::augment::{all_suit_perms, permute_card};
    use crate::bidding::{apply_bid, legal_bids};
    use crate::dealing::{new_round, RoundParams};
    use crate::encoder::hand_card_indices;
    use crate::playing::{apply_play, legal_plays};
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use smallvec::smallvec;

    /// Play a 4p round with the lowest legal moves, recording each decision
    /// with a one-hot policy; returns the decisions and the final state.
    fn played_round(seed: u64, cards: u8) -> (Vec<Decision>, BlobState) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let params = RoundParams { num_players: 4, cards_dealt: cards, trump: 0, dealer: 3 };
        let mut s = new_round(params, &mut rng).unwrap();
        let mut out = Vec::new();
        loop {
            match s.phase() {
                GamePhase::Bidding => {
                    let b = legal_bids(&s).trailing_zeros() as u8;
                    out.push(Decision { state: s, policy: smallvec![(b, 1.0)] });
                    apply_bid(&mut s, b);
                }
                GamePhase::Playing => {
                    let c = legal_plays(&s).trailing_zeros() as u8;
                    let pos = hand_card_indices(&s, s.current_player).iter().position(|&x| x == c);
                    out.push(Decision { state: s, policy: smallvec![(pos.unwrap() as u8, 1.0)] });
                    apply_play(&mut s, c);
                }
                _ => return (out, s),
            }
        }
    }

    #[test]
    fn push_round_stores_every_decision_with_the_rounds_points() {
        let mut buf = ReplayBuffer::new(100);
        let (decisions, end) = played_round(1, 3);
        assert_eq!(decisions.len(), 4 * 4);
        assert_eq!(buf.push_round(&decisions, &end), 0);
        assert_eq!(buf.push_round(&decisions, &end), 1);
        assert_eq!(buf.len(), 32);
        assert_eq!(buf.rounds_pushed(), 2);
        assert_eq!((buf.round_id(0), buf.round_id(16)), (0, 1));
        let points = round_points(&end);
        assert!(buf.points.iter().all(|p| *p == points));
    }

    #[test]
    fn circular_fifo_overwrites_oldest() {
        let mut buf = ReplayBuffer::new(20);
        let (decisions, end) = played_round(2, 2); // 12 decisions a round
        for _ in 0..3 {
            buf.push_round(&decisions, &end);
        }
        assert_eq!(buf.len(), 20);
        // 36 pushed into 20 slots: the newest 20 are rounds 1 (4 left) and 2.
        let mut ids: Vec<u64> = (0..20).map(|i| buf.round_id(i)).collect();
        ids.sort();
        assert_eq!(ids, [vec![1; 8], vec![2; 12]].concat());
    }

    #[test]
    fn batches_split_by_phase_with_scores_from_the_movers_seat() {
        let mut buf = ReplayBuffer::new(100);
        let (decisions, end) = played_round(3, 4);
        buf.push_round(&decisions, &end);
        let points = round_points(&end);
        let all: Vec<usize> = (0..buf.len()).collect();
        let (bid, play) = buf.batch_from_indices(&all);
        assert_eq!((bid.states.len(), play.states.len()), (4, 16));
        assert_eq!(bid.policies.len(), 4 * MAX_BID_ACTIONS);
        assert_eq!(play.policies.len(), 16 * play.max_hand_size);
        for (s, scores) in bid.states.iter().chain(&play.states).zip(bid.seat_scores.iter().chain(&play.seat_scores)) {
            for (rel, &score) in scores.iter().enumerate().take(4) {
                let seat = (s.current_player as usize + rel) % 4;
                assert_eq!(score, points[seat] as f32 / 14.0);
            }
            assert!(scores[4..].iter().all(|&x| x == 0.0));
        }
        for row in 0..4 {
            let sum: f32 = bid.policies[row * MAX_BID_ACTIONS..(row + 1) * MAX_BID_ACTIONS].iter().sum();
            assert_eq!(sum, 1.0);
        }
    }

    /// Augmented samples are relabellings of stored examples whose play
    /// policy still points at the same cards, with unchanged targets.
    #[test]
    fn augmented_samples_keep_policy_on_the_same_cards() {
        let mut buf = ReplayBuffer::new(100);
        let (mut decisions, end) = played_round(4, 5);
        // Spread each play policy over the whole hand, unevenly.
        for d in decisions.iter_mut().filter(|d| d.state.phase() == GamePhase::Playing) {
            let k = d.state.hands[d.state.current_player as usize].count_ones() as u8;
            let total = (k as f32) * (k as f32 + 1.0) / 2.0;
            d.policy = (0..k).map(|i| (i, (i as f32 + 1.0) / total)).collect();
        }
        buf.push_round(&decisions, &end);
        let originals: Vec<&Decision> = decisions.iter().collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(5);
        let mut relabelled = 0;
        for _ in 0..20 {
            let (_, play) = buf.sample_batch(10, &mut rng, true);
            for (row, s) in play.states.iter().enumerate() {
                let (orig, perm) = originals
                    .iter()
                    .find_map(|d| all_suit_perms().into_iter().find(|p| permute_suits(&d.state, p) == *s).map(|p| (*d, p)))
                    .expect("a relabelled stored example");
                relabelled += (perm != crate::augment::IDENTITY) as usize;
                let before = hand_card_indices(&orig.state, s.current_player);
                let after = hand_card_indices(s, s.current_player);
                for &(pos, p) in &orig.policy {
                    let to = after.iter().position(|&c| c == permute_card(before[pos as usize], &perm)).unwrap();
                    assert_eq!(play.policies[row * play.max_hand_size + to], p);
                }
                assert_eq!(play.seat_scores[row], seat_scores(&orig.state, &round_points(&end)));
            }
        }
        assert!(relabelled > 100, "{relabelled}");
    }

    #[test]
    fn sampling_is_uniform_over_examples() {
        // Chi-squared over which stored example each draw returns; the
        // critical value for df = 19, α = 0.001 is ≈ 43.82.
        const K: usize = 20;
        let mut buf = ReplayBuffer::new(K);
        for seed in 0..5 {
            let (decisions, end) = played_round(10 + seed, 1); // 4 bids + 4 plays
            buf.push_round(&decisions[..4], &end);
        }
        assert_eq!(buf.len(), K);
        let key = |s: &BlobState| (s.hands, s.current_player);
        let keys: Vec<_> = (0..K).map(|i| key(&buf.states[i])).collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let mut counts = [0u32; K];
        for _ in 0..20_000 {
            let (bid, _) = buf.sample_batch(5, &mut rng, false);
            for s in &bid.states {
                counts[keys.iter().position(|k| *k == key(s)).unwrap()] += 1;
            }
        }
        let expected = 20_000.0 * 5.0 / K as f64;
        let chi2: f64 = counts.iter().map(|&c| (c as f64 - expected).powi(2) / expected).sum();
        assert!(chi2 < 43.82, "chi2 = {chi2}");
    }

    #[test]
    fn roundtrip_serialization() {
        let mut buf = ReplayBuffer::new(64);
        let (decisions, end) = played_round(6, 3);
        buf.push_round(&decisions, &end);
        buf.push_round(&decisions, &end);
        let path = std::env::temp_dir().join(format!("blobmaster_replay_{}.bin", std::process::id()));
        buf.save(&path).expect("save");
        let restored = ReplayBuffer::load(&path).expect("load");
        std::fs::remove_file(&path).ok();
        assert_eq!((restored.len(), restored.capacity()), (buf.len(), buf.capacity()));
        assert_eq!(restored.points, buf.points);
        assert_eq!(restored.round_ids, buf.round_ids);
        assert_eq!(restored.rounds_pushed(), 2, "round ids continue after a resume");
    }

    #[test]
    fn shared_replay_takes_rounds_from_many_threads() {
        let shared = SharedReplay::new(ReplayBuffer::new(10_000));
        let (decisions, end) = played_round(8, 2);
        std::thread::scope(|sc| {
            for _ in 0..4 {
                sc.spawn(|| {
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
                    for _ in 0..25 {
                        shared.push_round(&decisions, &end);
                        let (bid, play) = shared.sample_batch(8, &mut rng, true);
                        assert_eq!(bid.states.len() + play.states.len(), 8);
                    }
                });
            }
        });
        assert_eq!(shared.len(), 100 * decisions.len());
        let buf = shared.into_inner();
        let mut ids: Vec<u64> = (0..buf.len()).step_by(decisions.len()).map(|i| buf.round_id(i)).collect();
        ids.sort();
        assert_eq!(ids, (0..100).collect::<Vec<u64>>(), "every round got its own id");
    }
}
