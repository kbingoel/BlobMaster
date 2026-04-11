//! Throwaway copy benchmark for `BlobState`. Run with:
//!
//! ```text
//! cargo test -p blob-engine --release -- --ignored --nocapture blobstate_copy_bench
//! ```
//!
//! Target from `development-plan.md` Session 1.1: ~100 ns per copy
//! (~410 B across ~6 cache lines). This is informational — no hard assert,
//! because CI hardware varies wildly. A `criterion` harness lands in a later
//! session once benchmarking infrastructure is set up.

use std::hint::black_box;
use std::time::Instant;

use blob_engine::BlobState;

#[test]
#[ignore]
fn blobstate_copy_bench() {
    let mut src = BlobState::empty();
    src.num_players = 5;
    src.cards_dealt = 7;
    src.hands[0] = 0xDEAD_BEEF_CAFE_BABE & ((1u64 << 52) - 1);
    src.cumulative_scores[3] = 123;

    // Warm-up
    let mut acc = BlobState::empty();
    for _ in 0..10_000 {
        acc = black_box(black_box(src));
    }
    black_box(&acc);

    let iters: u64 = 10_000_000;
    let start = Instant::now();
    for _ in 0..iters {
        acc = black_box(black_box(src));
    }
    let elapsed = start.elapsed();
    black_box(&acc);

    let ns_per_copy = elapsed.as_nanos() as f64 / iters as f64;
    println!(
        "BlobState copy: {:.2} ns/iter over {} iters ({} bytes)",
        ns_per_copy,
        iters,
        std::mem::size_of::<BlobState>()
    );
}
