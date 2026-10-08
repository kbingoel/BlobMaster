//! Throughput of rollout rounds (`blob_engine::rollout`, policy iteration
//! by rollouts) on the actor process's thread count, ONNX P only.
//!
//! ```text
//! cargo run --release -p blob-engine --example rollout_profile -- \
//!     <model dir> <threads> <secs> <samples per round>,... [<bid deals> [<rule bot 2 share>]]
//! ```
//! For each samples-per-round setting (0 = rounds of P alone, nothing
//! valued) it plays rounds on every thread for `<secs>` and prints rounds
//! and valued decisions per hour, per thread, the moves valued per
//! decision, P calls, and how often P's top move is not the best on the
//! deal (hindsight).

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use blob_engine::evaluator::PolicyEvaluator;
use blob_engine::rollout::{rollout_round, RolloutConfig, RolloutStats};
use blob_engine::round::RoundMix;
use blob_engine::{BlobState, OnnxPolicy};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// P with a count of the states it evaluated.
struct Counted<'a> {
    p: &'a OnnxPolicy,
    states: &'a AtomicU64,
    calls: &'a AtomicU64,
}

impl PolicyEvaluator for Counted<'_> {
    fn policy(&self, s: &BlobState) -> Vec<f32> {
        self.policy_batch(&[s]).remove(0)
    }

    fn policy_batch(&self, states: &[&BlobState]) -> Vec<Vec<f32>> {
        self.states.fetch_add(states.len() as u64, Relaxed);
        self.calls.fetch_add(1, Relaxed);
        self.p.policy_batch(states)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if !(5..=7).contains(&args.len()) {
        eprintln!("usage: rollout_profile <model dir> <threads> <secs> <samples per round>,... [<bid deals> [<rule bot 2 share>]]");
        std::process::exit(2);
    }
    let bid_deals: usize = args.get(5).map_or(1, |x| x.parse().expect("bid deals"));
    let share: f32 = args.get(6).map_or(0.0, |x| x.parse().expect("rule bot 2 share"));
    let model = &args[1];
    let threads: usize = args[2].parse().expect("threads");
    let secs: f64 = args[3].parse().expect("secs");
    let settings: Vec<usize> = args[4].split(',').map(|x| x.parse().expect("samples per round")).collect();
    let mix = RoundMix::default();
    for &k in &settings {
        let cfg = RolloutConfig { samples_per_round: k, bid_deals, rule_bot_2_share: share, ..Default::default() };
        let (rounds, states, calls) = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
        let stats = Mutex::new(RolloutStats::default());
        let deadline = Instant::now() + Duration::from_secs_f64(secs);
        std::thread::scope(|sc| {
            for t in 0..threads {
                let (rounds, states, calls, stats, mix) = (&rounds, &states, &calls, &stats, &mix);
                sc.spawn(move || {
                    let p = OnnxPolicy::from_dir(model).expect("policy.onnx");
                    let counted = Counted { p: &p, states, calls };
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1000 + t as u64);
                    let mut local = RolloutStats::default();
                    while Instant::now() < deadline {
                        let params = mix.sample(&mut rng);
                        let (_, st) = rollout_round(params, &counted, &cfg, &mut rng);
                        local.merge(&st);
                        rounds.fetch_add(1, Relaxed);
                    }
                    stats.lock().unwrap().merge(&local);
                });
            }
        });
        let st = stats.into_inner().unwrap();
        let h = secs / 3600.0;
        let n = st.samples[0] + st.samples[1];
        let r = rounds.load(Relaxed) as f64;
        println!(
            "samples/round {k}, bid deals {bid_deals}, rule bot 2 share {share}: {threads} threads, {:.0} rounds/h ({:.0}/thread); valued decisions {:.0}/h ({:.0}/thread; bids {:.0}%); \
             moves per decision {:.2}; P states per round {:.0} in {:.1} calls",
            r / h,
            r / h / threads as f64,
            n as f64 / h,
            n as f64 / h / threads as f64,
            100.0 * st.samples[0] as f64 / n.max(1) as f64,
            (st.moves[0] + st.moves[1]) as f64 / n.max(1) as f64,
            states.load(Relaxed) as f64 / r,
            calls.load(Relaxed) as f64 / r,
        );
        for (name, ph) in [("bids", 0), ("plays", 1)] {
            if st.samples[ph] > 0 {
                println!(
                    "  {name}: P's top not best on the deal {:.1}%, hindsight gap {:.4} utility per decision",
                    100.0 * st.top_not_best[ph] as f64 / st.samples[ph] as f64,
                    st.hindsight_sum[ph] / st.samples[ph] as f64
                );
            }
        }
    }
}
