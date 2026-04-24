//! Lightweight stopwatch instrumentation for ad-hoc self-play profiling.
//!
//! Globally toggled via [`enable`] / [`disable`]. While disabled, [`time`]
//! short-circuits to a plain call so production self-play pays nothing. While
//! enabled, each wrapped call records `(elapsed_nanos, 1)` into a named
//! bucket; multiple threads accumulate into the same relaxed atomic counters.
//!
//! Intended for the `blobmaster-train profile` subcommand — not a general
//! profiler. Buckets are nested (e.g. `ONNX_INFERENCE` is a sub-slice of
//! `MCTS_SEARCH`), so shares do not sum to 100%.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(false);

pub fn enable() {
    ENABLED.store(true, Ordering::Relaxed);
}

pub fn disable() {
    ENABLED.store(false, Ordering::Relaxed);
}

#[inline]
pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub struct Bucket {
    pub name: &'static str,
    pub nanos: AtomicU64,
    pub count: AtomicU64,
}

impl Bucket {
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            nanos: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    #[inline]
    fn record(&self, nanos: u64) {
        self.nanos.fetch_add(nanos, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn reset(&self) {
        self.nanos.store(0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> (u64, u64) {
        (
            self.nanos.load(Ordering::Relaxed),
            self.count.load(Ordering::Relaxed),
        )
    }
}

#[inline]
pub fn time<F, T>(bucket: &Bucket, f: F) -> T
where
    F: FnOnce() -> T,
{
    if is_enabled() {
        let t = Instant::now();
        let out = f();
        bucket.record(t.elapsed().as_nanos() as u64);
        out
    } else {
        f()
    }
}

pub static GAME_TOTAL: Bucket = Bucket::new("game_total");
pub static MCTS_SEARCH: Bucket = Bucket::new("mcts_search");
pub static DETERMINIZE: Bucket = Bucket::new("determinize");
pub static ENCODE: Bucket = Bucket::new("encode");
pub static ONNX_TENSOR_BUILD: Bucket = Bucket::new("onnx_tensor_build");
pub static ONNX_INFERENCE: Bucket = Bucket::new("onnx_inference");
pub static ONNX_OUTPUT_EXTRACT: Bucket = Bucket::new("onnx_output_extract");
pub static EXPAND: Bucket = Bucket::new("expand");
pub static BACKPROP: Bucket = Bucket::new("backprop");
pub static SESSION_CONSTRUCTION: Bucket = Bucket::new("session_construction");

pub const ALL_BUCKETS: &[&Bucket] = &[
    &GAME_TOTAL,
    &MCTS_SEARCH,
    &DETERMINIZE,
    &ENCODE,
    &ONNX_TENSOR_BUILD,
    &ONNX_INFERENCE,
    &ONNX_OUTPUT_EXTRACT,
    &EXPAND,
    &BACKPROP,
    &SESSION_CONSTRUCTION,
];

pub fn reset_all() {
    for b in ALL_BUCKETS {
        b.reset();
    }
}
