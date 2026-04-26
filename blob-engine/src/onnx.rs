//! Session 3.5 — ONNX Runtime-backed `Evaluator`.
//!
//! Production inference path for self-play (Section 5.3) and the deployment
//! binary. Each rayon thread should create its own `OnnxEvaluator` because
//! `ort::Session` is not itself thread-safe for concurrent `run()` calls
//! without locking — per-thread sessions with `intra_op_num_threads=1` give
//! clean scaling.
//!
//! Expected ONNX graph I/O (produced by `scripts/export_onnx.py`):
//!
//! Inputs:
//! - `features: [batch, seq, 48]` f32
//! - `token_types: [batch, seq]` i64
//! - `chrono_indices: [batch, seq]` i64
//! - `attention_mask: [batch, seq]` bool
//!
//! Outputs:
//! - `bid_policy: [batch, 14]` f32 (masked softmax, needs re-masking)
//! - `play_scores: [batch, seq]` f32 (raw; caller masks + softmaxes)
//! - `value: [batch]` f32 ∈ [-1, 1]
//!
//! The evaluator re-applies legality masking from the current `BlobState`
//! rather than relying on the graph-internal mask, so one exported model
//! works for any phase and any hand size.

use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use ndarray::{Array2, Array3};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;

use crate::bidding::legal_bids;
use crate::encoder::{
    encode, EncodedState, PLAYED_CARD_DIM, TOKEN_TYPE_HAND,
};
use crate::evaluator::{Evaluator, NUM_BIDS};
use crate::playing::legal_plays;
use crate::state::{BlobState, GamePhase};

/// Per-token feature width (matches `blob_nn::input::FEAT_DIM`).
pub const FEAT_DIM: usize = PLAYED_CARD_DIM;

// ---- Session 7.4b: calibration capture for INT8 static quantization --------
//
// `OnnxEvaluator::run_encoded` consults the global `CALIBRATION_ENABLED` flag
// before any FP32 inference and, when active, records up to
// `CALIBRATION_LIMIT` `EncodedState`s into `CALIBRATION_SINK`. The
// `dump-calibration` profile path then drains the sink to a binary file
// consumed by `scripts/export_onnx.py --calibration`.

static CALIBRATION_ENABLED: AtomicBool = AtomicBool::new(false);
static CALIBRATION_LIMIT: AtomicUsize = AtomicUsize::new(0);
static CALIBRATION_SINK: Mutex<Vec<EncodedState>> = Mutex::new(Vec::new());

/// Magic header for the calibration binary format. Python parser must match.
pub const CALIBRATION_MAGIC: u32 = 0x42_43_41_4C; // "BCAL"
/// File-format version. Bump on layout change.
pub const CALIBRATION_VERSION: u32 = 1;

/// Begin recording up to `limit` encoded states. Idempotent: re-arming with a
/// new limit clears the sink and resets the cap.
pub fn start_calibration_capture(limit: usize) {
    CALIBRATION_SINK
        .lock()
        .expect("calibration sink poisoned")
        .clear();
    CALIBRATION_LIMIT.store(limit, Ordering::Relaxed);
    CALIBRATION_ENABLED.store(true, Ordering::Release);
}

/// Stop recording and return the captured states. Resets the global sink so a
/// subsequent run starts clean.
pub fn finish_calibration_capture() -> Vec<EncodedState> {
    CALIBRATION_ENABLED.store(false, Ordering::Release);
    CALIBRATION_LIMIT.store(0, Ordering::Relaxed);
    let mut sink = CALIBRATION_SINK
        .lock()
        .expect("calibration sink poisoned");
    std::mem::take(&mut *sink)
}

#[inline]
fn maybe_capture(enc: &EncodedState) {
    if !CALIBRATION_ENABLED.load(Ordering::Acquire) {
        return;
    }
    let mut sink = CALIBRATION_SINK
        .lock()
        .expect("calibration sink poisoned");
    let limit = CALIBRATION_LIMIT.load(Ordering::Relaxed);
    if sink.len() >= limit {
        // Stop recording further calls — cheap fast-path next time.
        CALIBRATION_ENABLED.store(false, Ordering::Release);
        return;
    }
    sink.push(enc.clone());
}

/// Serialize a list of `EncodedState`s in the BCAL binary format:
///
/// ```text
/// header:    u32 magic = 0x4243414C ("BCAL"), u32 version = 1, u32 num_states
/// per state: u32 num_tokens (S),
///            f32[S * 48] features (row-major, LE),
///            i64[S]      token_types (LE),
///            i64[S]      chrono_indices (LE)
/// ```
///
/// The `attention_mask` is implicit (the first `num_tokens` positions are
/// valid; padding to a fixed sequence length happens on the Python side).
pub fn write_calibration_file(path: &Path, states: &[EncodedState]) -> std::io::Result<()> {
    let f = std::fs::File::create(path)?;
    let mut w = BufWriter::new(f);
    w.write_all(&CALIBRATION_MAGIC.to_le_bytes())?;
    w.write_all(&CALIBRATION_VERSION.to_le_bytes())?;
    w.write_all(&(states.len() as u32).to_le_bytes())?;
    for st in states {
        let s = st.num_tokens as u32;
        w.write_all(&s.to_le_bytes())?;
        // Per-row width is variable in `EncodedState` (CLS=0, context=13,
        // player=29, hand=30, played=48 — see `encoder::encode`). The ONNX
        // graph expects `[B, S, FEAT_DIM=48]` zero-padded on the right, so
        // we serialize the same shape Python will reshape into.
        let pad_zero = 0.0f32.to_le_bytes();
        for row in &st.features {
            let n = row.len().min(FEAT_DIM);
            for v in &row[..n] {
                w.write_all(&v.to_le_bytes())?;
            }
            for _ in n..FEAT_DIM {
                w.write_all(&pad_zero)?;
            }
        }
        for tt in st.token_types.iter().take(st.num_tokens) {
            w.write_all(&(*tt as i64).to_le_bytes())?;
        }
        for ci in st.chronological_indices.iter().take(st.num_tokens) {
            w.write_all(&(*ci as i64).to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}

/// ONNX-backed evaluator. Own one per thread for self-play.
pub struct OnnxEvaluator {
    session: Mutex<Session>,
}

impl std::fmt::Debug for OnnxEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxEvaluator").finish_non_exhaustive()
    }
}

impl OnnxEvaluator {
    /// Load a model from `path`. Uses `CpuExecutionProvider` with
    /// `intra_op_num_threads=1` so multiple rayon threads can each hold
    /// their own session without contention.
    pub fn from_file(path: impl AsRef<Path>) -> ort::Result<Self> {
        crate::profiling::time(&crate::profiling::SESSION_CONSTRUCTION, || {
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(1)?
                .commit_from_file(path)?;
            Ok(Self {
                session: Mutex::new(session),
            })
        })
    }

    fn run_encoded(&self, enc: &EncodedState) -> ort::Result<(Vec<f32>, Vec<f32>, f32)> {
        maybe_capture(enc);
        let s = enc.num_tokens;

        let inputs = crate::profiling::time(&crate::profiling::ONNX_TENSOR_BUILD, || {
            let mut features = Array3::<f32>::zeros((1, s, FEAT_DIM));
            let mut token_types = Array2::<i64>::zeros((1, s));
            let mut chrono = Array2::<i64>::zeros((1, s));
            let mut mask = Array2::<bool>::from_elem((1, s), false);

            for i in 0..s {
                let row = &enc.features[i];
                for (j, v) in row.iter().enumerate() {
                    features[[0, i, j]] = *v;
                }
                token_types[[0, i]] = enc.token_types[i] as i64;
                chrono[[0, i]] = enc.chronological_indices[i] as i64;
                mask[[0, i]] = true;
            }

            let inputs = ort::inputs![
                "features" => Value::from_array(features)?,
                "token_types" => Value::from_array(token_types)?,
                "chrono_indices" => Value::from_array(chrono)?,
                "attention_mask" => Value::from_array(mask)?,
            ];
            Ok::<_, ort::Error>(inputs)
        })?;

        let mut sess = self
            .session
            .lock()
            .expect("OnnxEvaluator session mutex poisoned");

        let outputs = crate::profiling::time(&crate::profiling::ONNX_INFERENCE, || {
            sess.run(inputs)
        })?;

        crate::profiling::time(&crate::profiling::ONNX_OUTPUT_EXTRACT, || {
            let bid = outputs["bid_policy"]
                .try_extract_array::<f32>()?
                .to_owned();
            let play = outputs["play_scores"]
                .try_extract_array::<f32>()?
                .to_owned();
            let value = outputs["value"].try_extract_array::<f32>()?.to_owned();

            let bid_vec: Vec<f32> = bid.iter().copied().take(NUM_BIDS).collect();
            let play_vec: Vec<f32> = play.iter().copied().take(s).collect();
            let v: f32 = *value.iter().next().unwrap_or(&0.0);
            Ok((bid_vec, play_vec, v))
        })
    }

    /// Batched inference over `encs.len()` encoded states. Builds one
    /// `[B, S_max, FEAT_DIM]` zero-padded tensor, one `sess.run`, and splits
    /// the per-state outputs back. Sequence padding is masked by the
    /// `attention_mask` input (padded positions don't participate in
    /// attention), so per-state outputs are indistinguishable from running
    /// each state through `run_encoded` independently — modulo FP rounding
    /// from the GEMM batch shape, which is below the 1e-5 element gate.
    ///
    /// Per-state `play_scores` are truncated to that state's `num_tokens`
    /// before return so callers don't need to know about padding.
    fn run_encoded_batch(
        &self,
        encs: &[EncodedState],
    ) -> ort::Result<Vec<(Vec<f32>, Vec<f32>, f32)>> {
        debug_assert!(!encs.is_empty(), "run_encoded_batch called with empty batch");
        for enc in encs {
            maybe_capture(enc);
        }

        let b = encs.len();
        let s_max = encs.iter().map(|e| e.num_tokens).max().unwrap_or(0);

        let inputs = crate::profiling::time(&crate::profiling::ONNX_TENSOR_BUILD, || {
            let mut features = Array3::<f32>::zeros((b, s_max, FEAT_DIM));
            let mut token_types = Array2::<i64>::zeros((b, s_max));
            let mut chrono = Array2::<i64>::zeros((b, s_max));
            let mut mask = Array2::<bool>::from_elem((b, s_max), false);

            for (bi, enc) in encs.iter().enumerate() {
                for i in 0..enc.num_tokens {
                    let row = &enc.features[i];
                    for (j, v) in row.iter().enumerate() {
                        features[[bi, i, j]] = *v;
                    }
                    token_types[[bi, i]] = enc.token_types[i] as i64;
                    chrono[[bi, i]] = enc.chronological_indices[i] as i64;
                    mask[[bi, i]] = true;
                }
            }

            let inputs = ort::inputs![
                "features" => Value::from_array(features)?,
                "token_types" => Value::from_array(token_types)?,
                "chrono_indices" => Value::from_array(chrono)?,
                "attention_mask" => Value::from_array(mask)?,
            ];
            Ok::<_, ort::Error>(inputs)
        })?;

        let mut sess = self
            .session
            .lock()
            .expect("OnnxEvaluator session mutex poisoned");

        let outputs = crate::profiling::time(&crate::profiling::ONNX_INFERENCE, || {
            sess.run(inputs)
        })?;

        crate::profiling::time(&crate::profiling::ONNX_OUTPUT_EXTRACT, || {
            // bid_policy [B, NUM_BIDS], play_scores [B, S_max], value [B].
            // ORT outputs are standard layout; iterate row-major.
            let bid = outputs["bid_policy"].try_extract_array::<f32>()?.to_owned();
            let play = outputs["play_scores"].try_extract_array::<f32>()?.to_owned();
            let value = outputs["value"].try_extract_array::<f32>()?.to_owned();

            let bid_iter: Vec<f32> = bid.iter().copied().collect();
            let play_iter: Vec<f32> = play.iter().copied().collect();
            let value_iter: Vec<f32> = value.iter().copied().collect();

            debug_assert_eq!(bid_iter.len(), b * NUM_BIDS);
            debug_assert_eq!(play_iter.len(), b * s_max);
            debug_assert_eq!(value_iter.len(), b);

            let mut out = Vec::with_capacity(b);
            for (bi, enc) in encs.iter().enumerate() {
                let bid_off = bi * NUM_BIDS;
                let play_off = bi * s_max;
                let bid_vec: Vec<f32> = bid_iter[bid_off..bid_off + NUM_BIDS].to_vec();
                let play_vec: Vec<f32> =
                    play_iter[play_off..play_off + enc.num_tokens].to_vec();
                let v = value_iter[bi];
                out.push((bid_vec, play_vec, v));
            }
            Ok(out)
        })
    }
}

/// Phase-aware mask + (re)normalization shared by `evaluate` and
/// `evaluate_batch`. `raw_bid` is the network's bid_policy row (length
/// `NUM_BIDS`); `raw_play` is the play_scores row truncated to the state's
/// `num_tokens`. Returns the dense legal-action policy.
fn postprocess_policy(
    state: &BlobState,
    enc: &EncodedState,
    raw_bid: &[f32],
    raw_play: &[f32],
) -> Vec<f32> {
    let phase = GamePhase::from_u8(state.game_phase).unwrap_or(GamePhase::Scoring);
    match phase {
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

impl Evaluator for OnnxEvaluator {
    fn evaluate(&self, state: &BlobState) -> (Vec<f32>, f32) {
        let phase = GamePhase::from_u8(state.game_phase).unwrap_or(GamePhase::Scoring);
        if matches!(phase, GamePhase::Scoring | GamePhase::Complete) {
            return (Vec::new(), 0.0);
        }

        let enc = encode(state, state.current_player);
        let (raw_bid, raw_play, value) = match self.run_encoded(&enc) {
            Ok(t) => t,
            Err(e) => panic!("ONNX inference failed: {e}"),
        };

        (postprocess_policy(state, &enc, &raw_bid, &raw_play), value)
    }

    fn evaluate_batch(&self, states: &[&BlobState]) -> Vec<(Vec<f32>, f32)> {
        if states.is_empty() {
            return Vec::new();
        }

        // Caller (`mcts_search` lockstep driver) filters terminal states
        // before queueing — assert and skip the trivial single-state path
        // through `evaluate` so all batch slots flow through `sess.run`.
        debug_assert!(
            states.iter().all(|s| {
                let p = GamePhase::from_u8(s.game_phase).unwrap_or(GamePhase::Scoring);
                matches!(p, GamePhase::Bidding | GamePhase::Playing)
            }),
            "evaluate_batch called with terminal state — caller must filter"
        );

        let encs: Vec<EncodedState> = states
            .iter()
            .map(|s| encode(s, s.current_player))
            .collect();

        let triples = match self.run_encoded_batch(&encs) {
            Ok(t) => t,
            Err(e) => panic!("ONNX batched inference failed: {e}"),
        };

        states
            .iter()
            .zip(encs.iter())
            .zip(triples.into_iter())
            .map(|((s, e), (raw_bid, raw_play, value))| {
                (postprocess_policy(s, e, &raw_bid, &raw_play), value)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_path() -> Option<PathBuf> {
        let p = std::env::var("BLOB_ONNX_MODEL").ok()?;
        let pb = PathBuf::from(p);
        if pb.exists() {
            Some(pb)
        } else {
            None
        }
    }

    #[test]
    fn loads_model_if_present() {
        // Skips when BLOB_ONNX_MODEL is unset; real end-to-end parity with
        // the tch model is verified by `scripts/export_onnx.py`'s own
        // sanity check after export.
        let Some(path) = model_path() else {
            eprintln!("BLOB_ONNX_MODEL unset; skipping");
            return;
        };
        let e = OnnxEvaluator::from_file(&path).expect("load model");
        let _ = e;
    }

    /// Session 7.4c stage-1: batched ONNX inference must agree with looping
    /// `evaluate` on the same states, modulo FP rounding from the GEMM
    /// batch shape (well below the 1e-5 element gate `scripts/export_onnx`
    /// already uses for tch ↔ ORT parity). Skips when `BLOB_ONNX_MODEL`
    /// isn't set, matching the rest of this file.
    #[test]
    fn evaluate_batch_matches_serial() {
        use crate::bidding::apply_bid;
        use crate::dealing::deal;
        use crate::game::new_game;
        use rand_xoshiro::rand_core::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;

        let Some(path) = model_path() else {
            eprintln!("BLOB_ONNX_MODEL unset; skipping");
            return;
        };
        let e = OnnxEvaluator::from_file(&path).expect("load model");

        // Build a small mixed batch: a freshly-dealt bidding state plus a
        // playing state, each from a different RNG seed so sequence
        // lengths diverge (forces zero-padding to S_max).
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(31337);
        let mut bidding = new_game(4, 5).unwrap();
        deal(&mut bidding, &mut rng);

        let mut playing = new_game(5, 7).unwrap();
        deal(&mut playing, &mut rng);
        while playing.phase() == crate::state::GamePhase::Bidding {
            let mask = crate::bidding::legal_bids(&playing);
            let b = mask.trailing_zeros() as u8;
            apply_bid(&mut playing, b);
        }

        let states = vec![&bidding, &playing];
        let batched = e.evaluate_batch(&states);
        let serial: Vec<(Vec<f32>, f32)> = states.iter().map(|s| e.evaluate(s)).collect();

        assert_eq!(batched.len(), serial.len());
        for (i, ((bp, bv), (sp, sv))) in batched.iter().zip(serial.iter()).enumerate() {
            assert_eq!(bp.len(), sp.len(), "state {i}: policy length differs");
            for (j, (a, b)) in bp.iter().zip(sp.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "state {i} policy[{j}]: batched={a}, serial={b}"
                );
            }
            assert!(
                (bv - sv).abs() < 1e-4,
                "state {i} value: batched={bv}, serial={sv}"
            );
        }
    }
}
