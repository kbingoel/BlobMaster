//! ONNX Runtime inference for the two gen-2 networks (gen-2.md §5.3).
//!
//! A model is a directory written by `scripts/export_onnx.py`:
//! - `policy.onnx`: the policy net P, read by [`OnnxPolicy`];
//! - `value.onnx`: the value net V, read by [`OnnxValue`];
//! - `meta.json`: layout id, learner step and config, for people.
//!
//! [`OnnxEvaluator`] holds both, for search; network-only play loads only
//! P. Each network owns one session with one intra-op thread, so give every
//! worker thread its own.
//!
//! Graph inputs, both networks:
//! - `features: [batch, seq, FEAT_DIM]` f32 (`encoder::FEAT_DIM`)
//! - `token_types: [batch, seq]` i64
//! - `chrono_indices: [batch, seq]` i64
//! - `attention_mask: [batch, seq]` bool
//!
//! P reads [`encode`] (the seat to move's view) and outputs:
//! - `bid_policy: [batch, 14]` f32, a softmax over every bid;
//! - `play_scores: [batch, seq]` f32, raw per-token scores.
//!
//! The evaluator re-masks both to the state's legal moves rather than
//! relying on the graph, so one model serves both phases and any hand size.
//!
//! V reads [`encode_value`] (every hand, seen from the seat to move) and
//! outputs:
//! - `seat_values: [batch, seq]` f32, per token. At the player tokens it is
//!   the expected ŝ of that seat, in relative-seat order (me first); the
//!   evaluator maps it back to absolute seats.
//!
//! **Layout guard** (gen-2.md §5.5 item 10). Each file's ONNX metadata
//! carries `blob_layout_id` and `blob_network` (`policy` or `value`). A file
//! is refused unless its id is [`LAYOUT_ID`] and its network is the one
//! being loaded: a model belongs to the code that trained it.

use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array2, Array3};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;

use crate::bidding::legal_bids;
use crate::encoder::{
    encode, encode_value, EncodedState, FEAT_DIM, LAYOUT_ID, TOKEN_TYPE_HAND, TOKEN_TYPE_PLAYER,
};
use crate::evaluator::{PolicyEvaluator, ValueEvaluator, NUM_BIDS};
use crate::playing::legal_plays;
use crate::state::{BlobState, GamePhase, MAX_PLAYERS};

/// P's file in a model directory.
pub const POLICY_FILE: &str = "policy.onnx";
/// V's file in a model directory.
pub const VALUE_FILE: &str = "value.onnx";
/// ONNX metadata key holding the encoder layout the model was trained on.
pub const LAYOUT_ID_KEY: &str = "blob_layout_id";
/// ONNX metadata key naming the network: `policy` or `value`.
pub const NETWORK_KEY: &str = "blob_network";

/// Why a model file can't serve as `network` for this encoder, if it can't.
fn check_metadata(
    path: &Path,
    network: &str,
    layout: Option<&str>,
    kind: Option<&str>,
) -> Result<(), String> {
    let p = path.display();
    match layout {
        None => {
            return Err(format!(
                "{p}: no {LAYOUT_ID_KEY} in the model metadata; it predates the gen-2 layout guard"
            ))
        }
        Some(id) if id != LAYOUT_ID => {
            return Err(format!(
                "{p}: model layout {id:?} is not this encoder's {LAYOUT_ID:?}; retrain, or use the \
                 code that trained it"
            ))
        }
        Some(_) => {}
    }
    if kind != Some(network) {
        return Err(format!("{p}: {NETWORK_KEY} is {kind:?}, expected {network:?}"));
    }
    Ok(())
}

/// One session that passed the layout check.
struct Net {
    session: Mutex<Session>,
}

impl std::fmt::Debug for Net {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Net").finish_non_exhaustive()
    }
}

impl Net {
    fn load(path: &Path, network: &str) -> ort::Result<Self> {
        crate::profiling::time(&crate::profiling::SESSION_CONSTRUCTION, || {
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(1)?
                .commit_from_file(path)?;
            let (layout, kind) = {
                let meta = session.metadata()?;
                (meta.custom(LAYOUT_ID_KEY), meta.custom(NETWORK_KEY))
            };
            check_metadata(path, network, layout.as_deref(), kind.as_deref())
                .map_err(ort::Error::new)?;
            Ok(Self { session: Mutex::new(session) })
        })
    }

    /// Run one zero-padded `[B, S_max, FEAT_DIM]` batch and return each
    /// requested output flattened row-major, with `S_max`. Padding is masked
    /// by `attention_mask`, so each row matches running its state alone up
    /// to FP rounding from the batch shape.
    fn run(&self, encs: &[EncodedState], outputs: &[&str]) -> ort::Result<(usize, Vec<Vec<f32>>)> {
        let b = encs.len();
        let s_max = encs.iter().map(|e| e.num_tokens).max().unwrap_or(0);

        let inputs = crate::profiling::time(&crate::profiling::ONNX_TENSOR_BUILD, || {
            let mut features = Array3::<f32>::zeros((b, s_max, FEAT_DIM));
            let mut token_types = Array2::<i64>::zeros((b, s_max));
            let mut chrono = Array2::<i64>::zeros((b, s_max));
            let mut mask = Array2::<bool>::from_elem((b, s_max), false);
            for (bi, enc) in encs.iter().enumerate() {
                for i in 0..enc.num_tokens {
                    for (j, v) in enc.features[i].iter().enumerate() {
                        features[[bi, i, j]] = *v;
                    }
                    token_types[[bi, i]] = enc.token_types[i] as i64;
                    chrono[[bi, i]] = enc.chronological_indices[i] as i64;
                    mask[[bi, i]] = true;
                }
            }
            Ok::<_, ort::Error>(ort::inputs![
                "features" => Value::from_array(features)?,
                "token_types" => Value::from_array(token_types)?,
                "chrono_indices" => Value::from_array(chrono)?,
                "attention_mask" => Value::from_array(mask)?,
            ])
        })?;

        let mut sess = self.session.lock().expect("ONNX session mutex poisoned");
        let out = crate::profiling::time(&crate::profiling::ONNX_INFERENCE, || sess.run(inputs))?;

        let flat = crate::profiling::time(&crate::profiling::ONNX_OUTPUT_EXTRACT, || {
            outputs
                .iter()
                .map(|&name| Ok(out[name].try_extract_array::<f32>()?.iter().copied().collect()))
                .collect::<ort::Result<Vec<Vec<f32>>>>()
        })?;
        Ok((s_max, flat))
    }
}

/// The policy net P (`policy.onnx`).
#[derive(Debug)]
pub struct OnnxPolicy(Net);

impl OnnxPolicy {
    pub fn from_file(path: impl AsRef<Path>) -> ort::Result<Self> {
        Net::load(path.as_ref(), "policy").map(Self)
    }

    /// P of the model directory `dir`.
    pub fn from_dir(dir: impl AsRef<Path>) -> ort::Result<Self> {
        Self::from_file(dir.as_ref().join(POLICY_FILE))
    }
}

/// The value net V (`value.onnx`).
#[derive(Debug)]
pub struct OnnxValue(Net);

impl OnnxValue {
    pub fn from_file(path: impl AsRef<Path>) -> ort::Result<Self> {
        Net::load(path.as_ref(), "value").map(Self)
    }

    /// V of the model directory `dir`.
    pub fn from_dir(dir: impl AsRef<Path>) -> ort::Result<Self> {
        Self::from_file(dir.as_ref().join(VALUE_FILE))
    }
}

/// Both networks of a model directory, for search.
#[derive(Debug)]
pub struct OnnxEvaluator {
    pub policy: OnnxPolicy,
    pub value: OnnxValue,
}

impl OnnxEvaluator {
    pub fn from_dir(dir: impl AsRef<Path>) -> ort::Result<Self> {
        let dir = dir.as_ref();
        Ok(Self { policy: OnnxPolicy::from_dir(dir)?, value: OnnxValue::from_dir(dir)? })
    }
}

fn assert_decision_states(states: &[&BlobState]) {
    debug_assert!(
        states.iter().all(|s| matches!(s.phase(), GamePhase::Bidding | GamePhase::Playing)),
        "network called on a finished round; search scores those exactly"
    );
}

/// Legal-move mask and (re)normalization of P's raw outputs for one state.
/// `raw_bid` is its `bid_policy` row (length `NUM_BIDS`); `raw_play` its
/// `play_scores` row truncated to the state's `num_tokens`.
fn postprocess_policy(
    state: &BlobState,
    enc: &EncodedState,
    raw_bid: &[f32],
    raw_play: &[f32],
) -> Vec<f32> {
    match state.phase() {
        GamePhase::Bidding => {
            let mask = legal_bids(state);
            let mut policy = vec![0.0f32; NUM_BIDS];
            let mut sum = 0.0f32;
            for b in 0..NUM_BIDS {
                if (mask >> b) & 1 == 1 {
                    let v = *raw_bid.get(b).unwrap_or(&0.0);
                    policy[b] = v.max(0.0);
                    sum += policy[b];
                }
            }
            if sum > 0.0 {
                for v in policy.iter_mut() {
                    *v /= sum;
                }
            } else {
                let n = mask.count_ones() as f32;
                if n > 0.0 {
                    for b in 0..NUM_BIDS {
                        if (mask >> b) & 1 == 1 {
                            policy[b] = 1.0 / n;
                        }
                    }
                }
            }
            policy
        }
        GamePhase::Playing => {
            let legal = legal_plays(state);
            let n_hand = enc.hand_card_indices.len();
            let mut policy = vec![f32::NEG_INFINITY; n_hand];
            let mut any_legal = false;

            let mut hand_slot = 0usize;
            for (tok_i, tt) in enc.token_types.iter().enumerate() {
                if *tt != TOKEN_TYPE_HAND {
                    continue;
                }
                let card_idx = enc.hand_card_indices[hand_slot];
                if (legal >> card_idx) & 1 == 1 {
                    policy[hand_slot] = *raw_play.get(tok_i).unwrap_or(&0.0);
                    any_legal = true;
                }
                hand_slot += 1;
            }

            if any_legal {
                let max = policy
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite())
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in policy.iter_mut() {
                    if v.is_finite() {
                        *v = (*v - max).exp();
                        sum += *v;
                    } else {
                        *v = 0.0;
                    }
                }
                if sum > 0.0 {
                    for v in policy.iter_mut() {
                        *v /= sum;
                    }
                }
            } else {
                for v in policy.iter_mut() {
                    *v = 0.0;
                }
            }
            policy
        }
        GamePhase::Scoring | GamePhase::Complete => Vec::new(),
    }
}

/// V's per-token row for one state, read at the player tokens and mapped
/// from relative seats (me first) back to absolute seats.
fn seat_values(state: &BlobState, enc: &EncodedState, row: &[f32]) -> [f32; MAX_PLAYERS] {
    let n = state.num_players as usize;
    let me = state.current_player as usize;
    let mut out = [0.0f32; MAX_PLAYERS];
    let players = (0..enc.num_tokens).filter(|&i| enc.token_types[i] == TOKEN_TYPE_PLAYER);
    for (rel, pos) in players.enumerate() {
        out[(me + rel) % n] = row[pos];
    }
    out
}

impl PolicyEvaluator for OnnxPolicy {
    fn policy(&self, state: &BlobState) -> Vec<f32> {
        self.policy_batch(&[state]).pop().expect("one policy per state")
    }

    fn policy_batch(&self, states: &[&BlobState]) -> Vec<Vec<f32>> {
        if states.is_empty() {
            return Vec::new();
        }
        assert_decision_states(states);
        let encs: Vec<EncodedState> = states.iter().map(|s| encode(s, s.current_player)).collect();
        let (s_max, out) = self
            .0
            .run(&encs, &["bid_policy", "play_scores"])
            .unwrap_or_else(|e| panic!("ONNX policy inference failed: {e}"));
        let (bid, play) = (&out[0], &out[1]);
        debug_assert_eq!((bid.len(), play.len()), (states.len() * NUM_BIDS, states.len() * s_max));
        states
            .iter()
            .zip(&encs)
            .enumerate()
            .map(|(i, (s, e))| {
                let raw_bid = &bid[i * NUM_BIDS..(i + 1) * NUM_BIDS];
                let raw_play = &play[i * s_max..i * s_max + e.num_tokens];
                postprocess_policy(s, e, raw_bid, raw_play)
            })
            .collect()
    }
}

impl ValueEvaluator for OnnxValue {
    fn values(&self, state: &BlobState) -> [f32; MAX_PLAYERS] {
        self.values_batch(&[state])[0]
    }

    fn values_batch(&self, states: &[&BlobState]) -> Vec<[f32; MAX_PLAYERS]> {
        if states.is_empty() {
            return Vec::new();
        }
        assert_decision_states(states);
        let encs: Vec<EncodedState> =
            states.iter().map(|s| encode_value(s, s.current_player)).collect();
        let (s_max, out) = self
            .0
            .run(&encs, &["seat_values"])
            .unwrap_or_else(|e| panic!("ONNX value inference failed: {e}"));
        debug_assert_eq!(out[0].len(), states.len() * s_max);
        states
            .iter()
            .zip(&encs)
            .enumerate()
            .map(|(i, (s, e))| seat_values(s, e, &out[0][i * s_max..i * s_max + e.num_tokens]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bidding::apply_bid;
    use crate::dealing::deal;
    use crate::game::new_game;
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::path::PathBuf;

    /// A model directory from `BLOB_MODEL_DIR`; the tests that need one
    /// skip without it.
    fn model_dir() -> Option<PathBuf> {
        let pb = PathBuf::from(std::env::var("BLOB_MODEL_DIR").ok()?);
        pb.is_dir().then_some(pb)
    }

    /// A bidding and a playing state of different lengths, so batching pads.
    fn mixed_states() -> Vec<BlobState> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(31337);
        let mut bidding = new_game(4, 5).unwrap();
        deal(&mut bidding, &mut rng);
        // Seat 1 leads, so relative and absolute seats differ.
        let mut playing = new_game(5, 7).unwrap();
        deal(&mut playing, &mut rng);
        while playing.phase() == GamePhase::Bidding {
            let b = legal_bids(&playing).trailing_zeros() as u8;
            apply_bid(&mut playing, b);
        }
        vec![bidding, playing]
    }

    #[test]
    fn metadata_check_refuses_other_layouts_and_networks() {
        let p = Path::new("m/policy.onnx");
        assert!(check_metadata(p, "policy", Some(LAYOUT_ID), Some("policy")).is_ok());
        let none = check_metadata(p, "policy", None, None).unwrap_err();
        assert!(none.contains("predates"), "{none}");
        let old = check_metadata(p, "policy", Some("layout-2"), Some("policy")).unwrap_err();
        assert!(old.contains("retrain"), "{old}");
        let swapped = check_metadata(p, "value", Some(LAYOUT_ID), Some("policy")).unwrap_err();
        assert!(swapped.contains("expected \"value\""), "{swapped}");
    }

    #[test]
    fn seat_values_map_relative_seats_back() {
        let s = &mixed_states()[1];
        let enc = encode_value(s, s.current_player);
        // Row value at a player token = 10 × its relative seat + 1.
        let mut row = vec![-1.0f32; enc.num_tokens];
        let players: Vec<usize> =
            (0..enc.num_tokens).filter(|&i| enc.token_types[i] == TOKEN_TYPE_PLAYER).collect();
        for (rel, &pos) in players.iter().enumerate() {
            row[pos] = 10.0 * rel as f32 + 1.0;
        }
        let v = seat_values(s, &enc, &row);
        let n = s.num_players;
        for seat in 0..n {
            let rel = (seat + n - s.current_player) % n;
            assert_eq!(v[seat as usize], 10.0 * rel as f32 + 1.0, "seat {seat}");
        }
        assert!(v[n as usize..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn loads_model_dir_and_refuses_swapped_files() {
        let Some(dir) = model_dir() else {
            eprintln!("BLOB_MODEL_DIR unset; skipping");
            return;
        };
        OnnxEvaluator::from_dir(&dir).expect("load model dir");
        assert!(OnnxValue::from_file(dir.join(POLICY_FILE)).is_err());
        assert!(OnnxPolicy::from_file(dir.join(VALUE_FILE)).is_err());
    }

    /// Batched inference agrees with one state at a time, modulo FP rounding
    /// from the batch shape; V gives every seat a ŝ in [0, 1].
    #[test]
    fn batches_match_single_states() {
        let Some(dir) = model_dir() else {
            eprintln!("BLOB_MODEL_DIR unset; skipping");
            return;
        };
        let e = OnnxEvaluator::from_dir(&dir).expect("load model dir");
        let states = mixed_states();
        let refs: Vec<&BlobState> = states.iter().collect();

        let batched = e.policy.policy_batch(&refs);
        for (i, s) in states.iter().enumerate() {
            let single = e.policy.policy(s);
            assert_eq!(batched[i].len(), single.len());
            for (a, b) in batched[i].iter().zip(&single) {
                assert!((a - b).abs() < 1e-4, "state {i}: batched {a}, single {b}");
            }
            assert!((single.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        }

        let batched = e.value.values_batch(&refs);
        for (i, s) in states.iter().enumerate() {
            let single = e.value.values(s);
            for seat in 0..MAX_PLAYERS {
                assert!((batched[i][seat] - single[seat]).abs() < 1e-4, "state {i} seat {seat}");
                if seat < s.num_players as usize {
                    assert!((0.0..=1.0).contains(&single[seat]), "ŝ {}", single[seat]);
                } else {
                    assert_eq!(single[seat], 0.0);
                }
            }
        }
    }
}
