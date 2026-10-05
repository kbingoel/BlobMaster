//! Benches that require a model directory.
//!
//! Gated on the `BLOB_MODEL_DIR` environment variable (a directory with
//! `policy.onnx` and `value.onnx`, e.g. from `blobmaster-train export`).
//! When unset the benches become no-ops so `cargo bench` still runs green
//! on machines without a model.
//!
//! Benches: one policy and one value call (batch 1), search with 1 × 100
//! simulations, and a full play (5 × 100) and bid (20 × 25) decision with
//! P + V. Gen-1 numbers on this machine (one network) are in gen-2.md §3.1.

use std::hint::black_box;
use std::path::PathBuf;

use blob_engine::mcts::{mcts_search, MctsConfig, SearchBudget};
use blob_engine::{
    apply_bid, legal_bids, new_game, start_round, BlobState, GamePhase, OnnxEvaluator,
    PolicyEvaluator, ValueEvaluator,
};
use criterion::{criterion_group, criterion_main, Criterion};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn model() -> Option<OnnxEvaluator> {
    let dir = PathBuf::from(std::env::var("BLOB_MODEL_DIR").ok()?);
    if !dir.is_dir() {
        return None;
    }
    Some(OnnxEvaluator::from_dir(&dir).expect("load model directory"))
}

fn bidding_state() -> BlobState {
    let mut state = new_game(5, 7).expect("valid params");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0FFEE);
    start_round(&mut state, &mut rng);
    state
}

fn playing_state() -> BlobState {
    let mut state = bidding_state();
    while state.phase() == GamePhase::Bidding {
        let mask = legal_bids(&state);
        let bid = (0..=13u8).find(|b| (mask >> b) & 1 == 1).expect("legal bid");
        apply_bid(&mut state, bid);
    }
    state
}

fn bench_onnx_inference(c: &mut Criterion) {
    let Some(m) = model() else {
        eprintln!("BLOB_MODEL_DIR unset; skipping onnx inference benches");
        return;
    };
    let state = playing_state();
    c.bench_function("onnx_policy_batch1", |b| {
        b.iter(|| black_box(m.policy.policy(black_box(&state))))
    });
    c.bench_function("onnx_value_batch1", |b| {
        b.iter(|| black_box(m.value.values(black_box(&state))))
    });
}

fn bench_search(c: &mut Criterion) {
    let Some(m) = model() else {
        eprintln!("BLOB_MODEL_DIR unset; skipping search benches");
        return;
    };
    let (playing, bidding) = (playing_state(), bidding_state());
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
    let one_tree = MctsConfig { play_budget: SearchBudget::new(1, 100), ..MctsConfig::default() };
    c.bench_function("search_play_1x100", |b| {
        b.iter(|| black_box(mcts_search(black_box(&playing), &m.policy, &m.value, &one_tree, &mut rng, 0)))
    });
    let full = MctsConfig::default();
    c.bench_function("search_play_5x100", |b| {
        b.iter(|| black_box(mcts_search(black_box(&playing), &m.policy, &m.value, &full, &mut rng, 0)))
    });
    c.bench_function("search_bid_20x25", |b| {
        b.iter(|| black_box(mcts_search(black_box(&bidding), &m.policy, &m.value, &full, &mut rng, 0)))
    });
}

criterion_group!(benches, bench_onnx_inference, bench_search);
criterion_main!(benches);
