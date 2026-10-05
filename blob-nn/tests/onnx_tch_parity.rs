//! ONNX ↔ tch policy parity.
//!
//! Loads a saved VarStore checkpoint (`BLOB_TCH_CHECKPOINT`: a directory
//! with `model.ot` + `meta.json`) and the model directory exported from the
//! same weights (`BLOB_MODEL_DIR`), then pushes bidding and playing states
//! through the tch network and the ONNX policy net P and asserts the legal
//! policies agree within 1e-5.
//!
//! The value net has no tch counterpart until Phase 4, so only P is
//! compared; `scripts/export_onnx.py --check` covers the PyTorch→ONNX edge
//! of both networks.
//!
//! Skipped when either env var is unset so CI stays green on machines
//! without an exported model.

use std::path::PathBuf;

use blob_engine::encoder::{encode, TOKEN_TYPE_HAND};
use blob_engine::{
    apply_bid, apply_play, legal_bids, legal_plays, new_round, BlobState, GamePhase, OnnxPolicy,
    PolicyEvaluator, RoundParams,
};
use blob_nn::input::pad_batch;
use blob_nn::model::BlobNet;
use blob_nn::train::load_checkpoint;
use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256PlusPlus};
use tch::{nn::VarStore, Device, Tensor};

fn env_path(key: &str) -> Option<PathBuf> {
    let p = std::env::var(key).ok()?;
    let pb = PathBuf::from(p);
    pb.exists().then_some(pb)
}

/// The tch network's legal policy for `s`, in `PolicyEvaluator` layout.
fn tch_policy(model: &BlobNet, s: &BlobState) -> Vec<f32> {
    let enc = encode(s, s.current_player);
    let input = pad_batch(std::slice::from_ref(&enc), Device::Cpu);
    tch::no_grad(|| match s.phase() {
        GamePhase::Bidding => {
            let legal = legal_bids(s);
            let mask: Vec<bool> = (0..14).map(|b| (legal >> b) & 1 == 1).collect();
            let mask = Tensor::from_slice(&mask).view([1, 14]);
            let probs = model.forward_bid(&input, &mask, false).0;
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
            let probs: Vec<f32> = model.forward_play(&input, &mask, false).0.flatten(0, -1).try_into().unwrap();
            (0..enc.num_tokens).filter(|&i| enc.token_types[i] == TOKEN_TYPE_HAND).map(|i| probs[i]).collect()
        }
    })
}

#[test]
fn onnx_tch_policy_parity() {
    let Some(tch_dir) = env_path("BLOB_TCH_CHECKPOINT") else {
        eprintln!("BLOB_TCH_CHECKPOINT unset; skipping parity test");
        return;
    };
    let Some(model_dir) = env_path("BLOB_MODEL_DIR") else {
        eprintln!("BLOB_MODEL_DIR unset; skipping parity test");
        return;
    };

    let mut vs = VarStore::new(Device::Cpu);
    let model = BlobNet::new(&vs.root());
    load_checkpoint(&mut vs, &tch_dir).expect("load tch checkpoint");
    let onnx = OnnxPolicy::from_dir(&model_dir).expect("load ONNX policy net");

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xBEEF_F00D);
    let (mut max_diff, mut compared) = (0.0f32, [0usize; 2]);
    for t in 0..24u8 {
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
        let (want, got) = (tch_policy(&model, &s), onnx.policy(&s));
        assert_eq!(want.len(), got.len(), "trial {t}: policy lengths");
        for (a, b) in want.iter().zip(&got) {
            max_diff = max_diff.max((a - b).abs());
        }
        compared[(s.phase() == GamePhase::Playing) as usize] += 1;
    }
    assert!(compared[0] > 0 && compared[1] > 0, "both phases compared: {compared:?}");
    assert!(max_diff < 1e-5, "policy parity exceeds tolerance: max diff = {max_diff:.3e}");
}
