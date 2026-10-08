//! Self-play cost probe: single rounds from the 5p/7c game mix, every seat
//! searching with the default settings, as the Phase-5 actors will. Reports
//! rounds and decisions per hour and where the threads' time goes.
//!
//! Ignored; needs a model directory:
//! ```text
//! BLOB_MODEL_DIR=<dir> cargo test --release -p blob-engine --test selfplay_profile -- --ignored --nocapture
//! ```
//! `BLOB_PROFILE_THREADS` (default: every core) and `BLOB_PROFILE_ROUNDS`
//! (default 4 per thread) size the run.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use blob_engine::mcts::{apply_action, forced_action, is_terminal, mcts_search, MctsConfig};
use blob_engine::profiling;
use blob_engine::{
    bench::search_action, new_round, BlobState, OnnxPolicy, OnnxValue, PolicyEvaluator, RoundMix,
    ValueEvaluator, MAX_PLAYERS,
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[derive(Default)]
struct Tally {
    nanos: AtomicU64,
    calls: AtomicU64,
    states: AtomicU64,
}

impl Tally {
    fn add(&self, t: Instant, states: usize) {
        self.nanos.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.states.fetch_add(states as u64, Ordering::Relaxed);
    }
}

static P: Tally = Tally { nanos: AtomicU64::new(0), calls: AtomicU64::new(0), states: AtomicU64::new(0) };
static V: Tally = Tally { nanos: AtomicU64::new(0), calls: AtomicU64::new(0), states: AtomicU64::new(0) };

struct Timed(OnnxPolicy, OnnxValue);

impl PolicyEvaluator for Timed {
    fn policy(&self, s: &BlobState) -> Vec<f32> {
        self.policy_batch(&[s]).pop().unwrap()
    }
    fn policy_batch(&self, states: &[&BlobState]) -> Vec<Vec<f32>> {
        let t = Instant::now();
        let out = self.0.policy_batch(states);
        P.add(t, states.len());
        out
    }
}

impl ValueEvaluator for Timed {
    fn values(&self, s: &BlobState) -> [f32; MAX_PLAYERS] {
        self.values_batch(&[s])[0]
    }
    fn values_batch(&self, states: &[&BlobState]) -> Vec<[f32; MAX_PLAYERS]> {
        let t = Instant::now();
        let out = self.1.values_batch(states);
        V.add(t, states.len());
        out
    }
}

#[test]
#[ignore]
fn selfplay_profile() {
    let Ok(dir) = std::env::var("BLOB_MODEL_DIR") else { return };
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let threads = env("BLOB_PROFILE_THREADS", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8));
    let per_thread = env("BLOB_PROFILE_ROUNDS", 4);
    let cfg = MctsConfig::default();
    let mix = RoundMix::default();
    let decisions = AtomicUsize::new(0);
    let forced = AtomicUsize::new(0);
    profiling::reset_all();
    profiling::enable();
    let started = Instant::now();
    std::thread::scope(|sc| {
        for t in 0..threads {
            let (dir, cfg, mix, decisions, forced) = (&dir, &cfg, &mix, &decisions, &forced);
            sc.spawn(move || {
                let ev = Timed(OnnxPolicy::from_dir(dir).unwrap(), OnnxValue::from_dir(dir).unwrap());
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(1000 + t as u64);
                for _ in 0..per_thread {
                    let mut s = new_round(mix.sample(&mut rng), &mut rng).unwrap();
                    let mut i = 0;
                    while !is_terminal(&s) {
                        forced.fetch_add(forced_action(&s).is_some() as usize, Ordering::Relaxed);
                        let r = mcts_search(&s, &ev, &ev, cfg, &mut rng, i);
                        let a = search_action(&s, &r);
                        apply_action(&mut s, a);
                        decisions.fetch_add(1, Ordering::Relaxed);
                        i += 1;
                    }
                }
            });
        }
    });
    profiling::disable();
    let wall = started.elapsed().as_secs_f64();
    let thread_secs = wall * threads as f64;
    let rounds = threads * per_thread;
    let d = decisions.load(Ordering::Relaxed);
    println!("{threads} threads, {rounds} rounds, {d} decisions ({} forced) in {wall:.0} s", forced.load(Ordering::Relaxed));
    println!(
        "  {:.0} rounds/h, {:.0} decisions/h; {:.1} thread-s per round",
        rounds as f64 / wall * 3600.0,
        d as f64 / wall * 3600.0,
        thread_secs / rounds as f64
    );
    let share = |ns: u64| 100.0 * ns as f64 / 1e9 / thread_secs;
    for (name, t) in [("P (policy net)", &P), ("V (value net)", &V)] {
        let (ns, calls, states) = (t.nanos.load(Ordering::Relaxed), t.calls.load(Ordering::Relaxed), t.states.load(Ordering::Relaxed));
        println!(
            "  {name:16} {:5.1}% of thread time, {calls:9} calls, {states:9} states, {:.2} ms/call, {:.3} ms/state",
            share(ns),
            ns as f64 / 1e6 / calls.max(1) as f64,
            ns as f64 / 1e6 / states.max(1) as f64
        );
    }
    for b in profiling::ALL_BUCKETS {
        let (ns, count) = b.snapshot();
        if count > 0 {
            println!("  bucket {:20} {:5.1}% of thread time, {count:10} calls", b.name, share(ns));
        }
    }
}
