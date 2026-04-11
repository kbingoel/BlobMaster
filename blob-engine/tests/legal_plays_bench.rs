//! Throwaway benchmark for `legal_plays`. Run with:
//!
//! ```text
//! cargo test -p blob-engine --release -- --ignored --nocapture legal_plays_bench
//! ```
//!
//! Target from `development-plan.md` Session 1.4: legal-move generation
//! must be cheap enough that MCTS expansion isn't bottlenecked. The
//! implementation in `playing.rs` is a single suit-mask AND, so we expect
//! handfuls of nanoseconds per call. As with the state-copy bench this is
//! informational — no hard assert, because CI hardware varies.

use std::hint::black_box;
use std::time::Instant;

use blob_engine::{apply_bid, legal_bids, legal_plays, new_game, start_round, GamePhase};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[test]
#[ignore]
fn legal_plays_bench() {
    // Build a realistic mid-trick state: 5 players × 7 cards, fully bid,
    // one card already led so the suit-following branch is exercised.
    let mut state = new_game(5, 7).expect("valid params");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xBEEF);
    start_round(&mut state, &mut rng);
    while state.phase() == GamePhase::Bidding {
        let mask = legal_bids(&state);
        let bid = (0..=13u8)
            .find(|b| (mask >> b) & 1 == 1)
            .expect("legal bid");
        apply_bid(&mut state, bid);
    }
    // Lead the first card so subsequent calls hit the must-follow-suit path.
    let first = legal_plays(&state).trailing_zeros() as u8;
    blob_engine::apply_play(&mut state, first);

    // Warm-up.
    let mut acc: u64 = 0;
    for _ in 0..10_000 {
        acc ^= legal_plays(black_box(&state));
    }
    black_box(acc);

    let iters: u64 = 10_000_000;
    let start = Instant::now();
    let mut acc: u64 = 0;
    for _ in 0..iters {
        acc ^= legal_plays(black_box(&state));
    }
    let elapsed = start.elapsed();
    black_box(acc);

    let ns_per_call = elapsed.as_nanos() as f64 / iters as f64;
    println!(
        "legal_plays: {:.2} ns/iter over {} iters (mid-trick 5p×7c state)",
        ns_per_call, iters
    );
}
