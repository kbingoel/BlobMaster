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

use std::path::Path;
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
}

impl Evaluator for OnnxEvaluator {
    fn evaluate(&self, state: &BlobState) -> (Vec<f32>, f32) {
        let phase = GamePhase::from_u8(state.game_phase).unwrap_or(GamePhase::Scoring);
        if matches!(phase, GamePhase::Scoring | GamePhase::Complete) {
            return (Vec::new(), 0.0);
        }

        let enc = encode(state, state.current_player);
        let (bid_logits_or_probs, play_scores, value) = match self.run_encoded(&enc) {
            Ok(t) => t,
            Err(e) => panic!("ONNX inference failed: {e}"),
        };

        match phase {
            GamePhase::Bidding => {
                let mask = legal_bids(state);
                let mut policy = vec![0.0f32; NUM_BIDS];
                let mut sum = 0.0f32;
                for b in 0..NUM_BIDS {
                    if (mask >> b) & 1 == 1 {
                        let v = *bid_logits_or_probs.get(b).unwrap_or(&0.0);
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
                (policy, value)
            }
            GamePhase::Playing => {
                let legal = legal_plays(state);
                let n_hand = enc.hand_card_indices.len();
                let mut policy = vec![f32::NEG_INFINITY; n_hand];
                let mut any_legal = false;

                // Play head outputs one score per sequence position; hand
                // positions live in the slice of the sequence with
                // token_type == TOKEN_TYPE_HAND, in the same order as
                // `hand_card_indices` (Session 2.1 invariant).
                let mut hand_slot = 0usize;
                for (tok_i, tt) in enc.token_types.iter().enumerate() {
                    if *tt != TOKEN_TYPE_HAND {
                        continue;
                    }
                    let card_idx = enc.hand_card_indices[hand_slot];
                    if (legal >> card_idx) & 1 == 1 {
                        policy[hand_slot] = *play_scores.get(tok_i).unwrap_or(&0.0);
                        any_legal = true;
                    }
                    hand_slot += 1;
                }

                // Softmax over legal positions, zero elsewhere.
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
                (policy, value)
            }
            GamePhase::Scoring | GamePhase::Complete => unreachable!(),
        }
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
}
