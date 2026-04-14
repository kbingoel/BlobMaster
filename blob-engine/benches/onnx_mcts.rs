//! Session 6.3 — benches that require an ONNX model.
//!
//! Gated on the `BLOB_ONNX_MODEL` environment variable (path to an exported
//! `model.onnx`). When unset the benches become no-ops so `cargo bench` still
//! runs green on machines without a trained model.
//!
//! Targets:
//! - ONNX inference (batch=1): < 0.2 ms
//! - MCTS 100 sims (single determinization, ONNX eval): < 20 ms
//! - Full move (5 det × 100 sims): < 100 ms

use std::hint::black_box;
use std::path::PathBuf;

use blob_engine::mcts::{mcts_search, MctsConfig};
use blob_engine::{
    apply_bid, legal_bids, new_game, start_round, Evaluator, GamePhase, OnnxEvaluator,
};
use criterion::{criterion_group, criterion_main, Criterion};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn model_path() -> Option<PathBuf> {
    let p = std::env::var("BLOB_ONNX_MODEL").ok()?;
    let pb = PathBuf::from(p);
    pb.exists().then_some(pb)
}

fn playing_state() -> blob_engine::BlobState {
    let mut state = new_game(5, 7).expect("valid params");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0FFEE);
    start_round(&mut state, &mut rng);
    while state.phase() == GamePhase::Bidding {
        let mask = legal_bids(&state);
        let bid = (0..=13u8).find(|b| (mask >> b) & 1 == 1).expect("legal bid");
        apply_bid(&mut state, bid);
    }
    state
}

fn bench_onnx_inference(c: &mut Criterion) {
    let Some(path) = model_path() else {
        eprintln!("BLOB_ONNX_MODEL unset; skipping onnx inference bench");
        return;
    };
    let eval = OnnxEvaluator::from_file(&path).expect("load ONNX model");
    let state = playing_state();
    c.bench_function("onnx_inference_batch1", |b| {
        b.iter(|| black_box(eval.evaluate(black_box(&state))))
    });
}

fn bench_mcts_100_sims(c: &mut Criterion) {
    let Some(path) = model_path() else {
        eprintln!("BLOB_ONNX_MODEL unset; skipping mcts benches");
        return;
    };
    let eval = OnnxEvaluator::from_file(&path).expect("load ONNX model");
    let state = playing_state();
    let cfg = MctsConfig {
        num_determinizations: 1,
        sims_per_determinization: 100,
        min_sims_floor: 0,
        ..MctsConfig::default()
    };
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
    c.bench_function("mcts_1det_100sims", |b| {
        b.iter(|| black_box(mcts_search(black_box(&state), &eval, &cfg, &mut rng)))
    });

    let cfg_full = MctsConfig {
        num_determinizations: 5,
        sims_per_determinization: 100,
        min_sims_floor: 0,
        ..MctsConfig::default()
    };
    c.bench_function("mcts_full_move_5x100", |b| {
        b.iter(|| black_box(mcts_search(black_box(&state), &eval, &cfg_full, &mut rng)))
    });
}

criterion_group!(benches, bench_onnx_inference, bench_mcts_100_sims);
criterion_main!(benches);
