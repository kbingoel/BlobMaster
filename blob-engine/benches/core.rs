//! Session 6.3 — criterion benches for the always-available engine primitives.
//!
//! Run with: `cargo bench -p blob-engine --bench core`.
//!
//! Targets from `development-plan.md` Session 6.3:
//! - `BlobState` copy: ~100 ns
//! - Legal move generation: ~5 ns
//! - Entity encoding: < 1 µs

use std::hint::black_box;

use blob_engine::encoder::encode;
use blob_engine::{
    apply_bid, apply_play, legal_bids, legal_plays, new_game, start_round, BlobState, GamePhase,
};
use criterion::{criterion_group, criterion_main, Criterion};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn mid_trick_state() -> BlobState {
    let mut state = new_game(5, 7).expect("valid params");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xBEEF);
    start_round(&mut state, &mut rng);
    while state.phase() == GamePhase::Bidding {
        let mask = legal_bids(&state);
        let bid = (0..=13u8).find(|b| (mask >> b) & 1 == 1).expect("legal bid");
        apply_bid(&mut state, bid);
    }
    let first = legal_plays(&state).trailing_zeros() as u8;
    apply_play(&mut state, first);
    state
}

fn bench_state_copy(c: &mut Criterion) {
    let mut src = BlobState::empty();
    src.num_players = 5;
    src.cards_dealt = 7;
    src.hands[0] = 0xDEAD_BEEF_CAFE_BABE & ((1u64 << 52) - 1);
    c.bench_function("blobstate_copy", |b| {
        b.iter(|| {
            let dst = black_box(black_box(src));
            black_box(dst)
        })
    });
}

fn bench_legal_plays(c: &mut Criterion) {
    let state = mid_trick_state();
    c.bench_function("legal_plays_mid_trick", |b| {
        b.iter(|| black_box(legal_plays(black_box(&state))))
    });
}

fn bench_legal_bids(c: &mut Criterion) {
    let mut state = new_game(5, 7).expect("valid params");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
    start_round(&mut state, &mut rng);
    c.bench_function("legal_bids_first_seat", |b| {
        b.iter(|| black_box(legal_bids(black_box(&state))))
    });
}

fn bench_encode(c: &mut Criterion) {
    let state = mid_trick_state();
    let p = state.current_player;
    c.bench_function("encode_mid_trick_5p7c", |b| {
        b.iter(|| black_box(encode(black_box(&state), black_box(p))))
    });
}

criterion_group!(
    benches,
    bench_state_copy,
    bench_legal_plays,
    bench_legal_bids,
    bench_encode,
);
criterion_main!(benches);
