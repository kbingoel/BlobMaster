//! ONNX ↔ tch parity for both networks.
//!
//! Loads a checkpoint (`BLOB_TCH_CHECKPOINT`: a directory with `policy.ot`,
//! `value.ot` and `meta.json`) and the model directory exported from it
//! (`BLOB_MODEL_DIR`), then pushes bidding and playing states through both
//! and asserts they agree within 1e-5:
//! - P: the legal policies (tch `PolicyNet` vs `OnnxPolicy`);
//! - V: every seat's ŝ (tch `ValueNet` vs `OnnxValue`), absolute seats.
//!
//! Skipped when either env var is unset so CI stays green on machines
//! without an exported model.

use std::path::PathBuf;

use blob_engine::encoder::{encode, encode_value, TOKEN_TYPE_HAND};
use blob_engine::{
    apply_bid, apply_play, legal_bids, legal_plays, new_round, BlobState, GamePhase, OnnxPolicy,
    OnnxValue, PolicyEvaluator, RoundParams, ValueEvaluator,
};
use blob_nn::input::pad_batch;
use blob_nn::model::{PolicyNet, ValueNet};
use blob_nn::train::load_checkpoint;
use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256PlusPlus};
use tch::{nn::VarStore, Device, Tensor};

const TOLERANCE: f32 = 1e-5;

/// The path in `key`; `None` when unset (the test skips). A path that
/// doesn't exist fails, so a typo can't pass as a skip.
fn env_path(key: &str) -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var(key).ok()?);
    assert!(p.exists(), "{key}={} doesn't exist", p.display());
    Some(p)
}

/// The tch network's legal policy for `s`, in `PolicyEvaluator` layout.
fn tch_policy(model: &PolicyNet, s: &BlobState) -> Vec<f32> {
    let enc = encode(s, s.current_player);
    let input = pad_batch(std::slice::from_ref(&enc), Device::Cpu);
    tch::no_grad(|| match s.phase() {
        GamePhase::Bidding => {
            let legal = legal_bids(s);
            let mask: Vec<bool> = (0..14).map(|b| (legal >> b) & 1 == 1).collect();
            let mask = Tensor::from_slice(&mask).view([1, 14]);
            let probs = model.forward_bid(&input, &mask, false);
            Vec::<f32>::try_from(probs.flatten(0, -1)).unwrap()
        }
        _ => {
            let legal = legal_plays(s);
            let mut hand = enc.hand_card_indices.iter();
            let mask: Vec<bool> = enc
                .token_types
                .iter()
                .map(|&t| t == TOKEN_TYPE_HAND && (legal >> hand.next().unwrap()) & 1 == 1)
                .collect();
            let mask = Tensor::from_slice(&mask).view([1, enc.num_tokens as i64]);
            let probs: Vec<f32> = model.forward_play(&input, &mask, false).flatten(0, -1).try_into().unwrap();
            (0..enc.num_tokens).filter(|&i| enc.token_types[i] == TOKEN_TYPE_HAND).map(|i| probs[i]).collect()
        }
    })
}

/// The tch value net's ŝ for every seat of `s`, by absolute seat.
fn tch_values(model: &ValueNet, s: &BlobState) -> Vec<f32> {
    let enc = encode_value(s, s.current_player);
    let input = pad_batch(std::slice::from_ref(&enc), Device::Cpu);
    let n = s.num_players as usize;
    let rel: Vec<f32> =
        tch::no_grad(|| model.seat_values(&input, n as i64, false)).flatten(0, -1).try_into().unwrap();
    (0..n).map(|seat| rel[(seat + n - s.current_player as usize) % n]).collect()
}

/// Decision states of both phases at tables of 4–7.
fn states() -> Vec<BlobState> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xBEEF_F00D);
    (0..24u8)
        .map(|t| {
            let n = 4 + t % 4;
            let params = RoundParams { num_players: n, cards_dealt: 3 + t % 5, trump: t % 5, dealer: t % n };
            let mut s = new_round(params, &mut rng).expect("valid params");
            // Walk a few moves in, so both phases get compared.
            for _ in 0..(3 * t) % (2 * n) {
                if s.phase() == GamePhase::Bidding {
                    let b = legal_bids(&s).trailing_zeros() as u8;
                    apply_bid(&mut s, b);
                } else {
                    let c = legal_plays(&s).trailing_zeros() as u8;
                    apply_play(&mut s, c);
                }
            }
            s
        })
        .collect()
}

#[test]
fn onnx_tch_parity() {
    let Some(tch_dir) = env_path("BLOB_TCH_CHECKPOINT") else {
        eprintln!("BLOB_TCH_CHECKPOINT unset; skipping parity test");
        return;
    };
    let Some(model_dir) = env_path("BLOB_MODEL_DIR") else {
        eprintln!("BLOB_MODEL_DIR unset; skipping parity test");
        return;
    };

    let (mut pv, mut vv) = (VarStore::new(Device::Cpu), VarStore::new(Device::Cpu));
    let (policy, value) = (PolicyNet::new(&pv.root()), ValueNet::new(&vv.root()));
    load_checkpoint(&tch_dir, &mut pv, &mut vv).expect("load tch checkpoint");
    let onnx_policy = OnnxPolicy::from_dir(&model_dir).expect("load ONNX policy net");
    let onnx_value = OnnxValue::from_dir(&model_dir).expect("load ONNX value net");

    let (mut p_diff, mut v_diff, mut compared) = (0.0f32, 0.0f32, [0usize; 2]);
    for (t, s) in states().iter().enumerate() {
        let (want, got) = (tch_policy(&policy, s), onnx_policy.policy(s));
        assert_eq!(want.len(), got.len(), "state {t}: policy lengths");
        for (a, b) in want.iter().zip(&got) {
            p_diff = p_diff.max((a - b).abs());
        }
        let (want, got) = (tch_values(&value, s), onnx_value.values(s));
        for (seat, a) in want.iter().enumerate() {
            v_diff = v_diff.max((a - got[seat]).abs());
        }
        compared[(s.phase() == GamePhase::Playing) as usize] += 1;
    }
    eprintln!("[parity] max abs diff: policy {p_diff:.3e}, value {v_diff:.3e}");
    assert!(compared[0] > 0 && compared[1] > 0, "both phases compared: {compared:?}");
    assert!(p_diff < TOLERANCE, "policy parity exceeds tolerance: max diff = {p_diff:.3e}");
    assert!(v_diff < TOLERANCE, "value parity exceeds tolerance: max diff = {v_diff:.3e}");
}
