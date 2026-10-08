//! V of several models on the same held-out self-play rounds of a `train`
//! run (gen-2.md §6 Phase 5): is the trained V better than the warm start's
//! on the positions self-play reaches now? ONNX only, so it runs beside the
//! run.
//!
//! ```text
//! cargo run --release -p blob-engine --example rl_value_check -- \
//!     <run> <validation fraction> <rounds> <model dir>...
//! ```
//! Takes the last `<rounds>` validation rounds in `<run>/replay/` (split by
//! round id as the learner does) and prints each model's MSE and correlation
//! of ŝ over every (state, seat), by phase.

use std::path::PathBuf;

use blob_engine::scoring::{round_points, score_scale};
use blob_engine::selfplay::{chunk_files, read_chunk};
use blob_engine::{BlobState, GamePhase, OnnxValue, ValueEvaluator};

/// `blob_nn::learner::is_validation_round`, copied: blob-engine can't
/// depend on blob-nn.
fn is_validation_round(round_id: u64, fraction: f64) -> bool {
    let mut x = round_id ^ 0x5A11_DA7E_0F0F_0F0F;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    ((x >> 11) as f64 / (1u64 << 53) as f64) < fraction
}

#[derive(Default)]
struct Acc {
    n: f64,
    sx: f64,
    sy: f64,
    sxx: f64,
    syy: f64,
    sxy: f64,
}

impl Acc {
    fn add(&mut self, x: f64, y: f64) {
        self.n += 1.0;
        self.sx += x;
        self.sy += y;
        self.sxx += x * x;
        self.syy += y * y;
        self.sxy += x * y;
    }
    fn mse(&self) -> f64 {
        (self.sxx - 2.0 * self.sxy + self.syy) / self.n
    }
    fn corr(&self) -> f64 {
        let n = self.n;
        let cov = self.sxy / n - self.sx / n * self.sy / n;
        let vx = self.sxx / n - (self.sx / n).powi(2);
        let vy = self.syy / n - (self.sy / n).powi(2);
        cov / (vx * vy).sqrt()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: rl_value_check <run> <validation fraction> <rounds> <model dir>...");
        std::process::exit(2);
    }
    let run = PathBuf::from(&args[1]);
    let fraction: f64 = args[2].parse().expect("fraction");
    let want: usize = args[3].parse().expect("rounds");
    let models: Vec<PathBuf> = args[4..].iter().map(PathBuf::from).collect();

    // The last `want` validation rounds: (states, ŝ by absolute seat).
    let mut rounds = Vec::new();
    for (_, path) in chunk_files(&run).iter().rev() {
        let Ok(chunk) = read_chunk(path) else { continue };
        for r in chunk.rounds.into_iter().rev() {
            if is_validation_round(r.id, fraction) {
                rounds.push(r);
            }
        }
        if rounds.len() >= want {
            break;
        }
    }
    rounds.truncate(want);
    let model_steps: Vec<u64> = rounds.iter().map(|r| r.model_step).collect();
    let mut examples: Vec<(BlobState, Vec<f32>)> = Vec::new();
    for r in &rounds {
        let pts = round_points(&r.end);
        let scale = score_scale(r.end.cards_dealt);
        let target: Vec<f32> = (0..r.end.num_players as usize).map(|s| pts[s] as f32 / scale).collect();
        for d in &r.decisions {
            examples.push((d.state, target.clone()));
        }
    }
    println!(
        "{} validation rounds (played by models of steps {}..={}), {} states",
        rounds.len(),
        model_steps.iter().min().unwrap_or(&0),
        model_steps.iter().max().unwrap_or(&0),
        examples.len()
    );

    let threads = 8;
    for dir in &models {
        let per = examples.len().div_ceil(threads);
        let parts: Vec<[Acc; 2]> = std::thread::scope(|sc| {
            let hs: Vec<_> = examples
                .chunks(per.max(1))
                .map(|part| {
                    sc.spawn(move || {
                        let v = OnnxValue::from_dir(dir).expect("value net");
                        let mut acc = [Acc::default(), Acc::default()];
                        for batch in part.chunks(64) {
                            let states: Vec<&BlobState> = batch.iter().map(|(s, _)| s).collect();
                            let out = v.values_batch(&states);
                            for ((s, t), o) in batch.iter().zip(&out) {
                                let k = (s.phase() != GamePhase::Bidding) as usize;
                                for (seat, &y) in t.iter().enumerate() {
                                    acc[k].add(o[seat] as f64, y as f64);
                                }
                            }
                        }
                        acc
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut all = Acc::default();
        let mut by = [Acc::default(), Acc::default()];
        for p in &parts {
            for k in 0..2 {
                for (dst, src) in [(&mut by[k], &p[k]), (&mut all, &p[k])] {
                    dst.n += src.n;
                    dst.sx += src.sx;
                    dst.sy += src.sy;
                    dst.sxx += src.sxx;
                    dst.syy += src.syy;
                    dst.sxy += src.sxy;
                }
            }
        }
        println!(
            "{:70} MSE {:.4} corr {:.4} | bids MSE {:.4} corr {:.4} | plays MSE {:.4} corr {:.4}",
            dir.display(),
            all.mse(),
            all.corr(),
            by[0].mse(),
            by[0].corr(),
            by[1].mse(),
            by[1].corr()
        );
    }
}
