//! Batch-size sweep: raw forward-pass timing across CUDA, tch-CPU, and ONNX.
//!
//! Measures wall-clock per forward pass at batch sizes 1, 4, 16, 64, 256 and
//! converts to seconds-per-game using the measured evals/game from real
//! self-play (~98,000 evals/game at 5×100 MCTS for 5P7C).
//!
//! The ONNX path is always batch=1 (matches production self-play), so it
//! appears as a single row in the table. The "32-thread ONNX" column shows
//! throughput with 32 independent sessions in parallel.
//!
//! Usage:
//!   cargo run -p blob-nn --example batch_sweep                   # CPU only
//!   cargo run -p blob-nn --example batch_sweep --features cuda   # CPU + CUDA
//!
//! Requires BLOB_ONNX_MODEL env var pointing at a .onnx file for the ONNX
//! column (otherwise that column is skipped).

use std::time::Instant;

use blob_engine::dealing::deal;
use blob_engine::encoder::{encode, EncodedState};
use blob_engine::game::new_game;
use blob_engine::onnx::OnnxEvaluator;
use blob_engine::evaluator::Evaluator;
use blob_nn::gpu_eval::{pad_batch_with_fixed, MAX_SEQ};
use blob_nn::heads::NUM_BIDS;
use blob_nn::model::BlobNet;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use tch::{nn, Device, Kind, Tensor};

const BATCH_SIZES: &[usize] = &[1, 4, 16, 64, 256];
const WARMUP_ITERS: usize = 10;
const TIMED_ITERS: usize = 50;

/// Approximate evals per game from real self-play data:
/// 118 games produced 11,564,313 evals → ~98,003 evals/game.
const EVALS_PER_GAME: f64 = 98_000.0;

struct SweepResult {
    batch_size: usize,
    per_eval_us: f64,
    evals_per_sec: f64,
}

fn main() {
    println!("=== BlobNet batch-size sweep ===");
    println!(
        "Model: d={}, heads={}, layers={}, FFN={}, MAX_SEQ={}",
        128, 8, 8, 512, MAX_SEQ
    );
    println!("Evals/game assumed: {:.0}", EVALS_PER_GAME);
    println!(
        "Warmup: {} iters, Timed: {} iters per batch size\n",
        WARMUP_ITERS, TIMED_ITERS
    );

    // Generate encoded states.
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xBEEF);
    let states: Vec<_> = (0..256)
        .map(|i| {
            let np = 4 + (i % 4) as u8;
            let c = if np >= 7 { 7 } else { 8 };
            let mut s = new_game(np, c).unwrap();
            deal(&mut s, &mut rng);
            s
        })
        .collect();
    let encoded: Vec<_> = states
        .iter()
        .map(|s| encode(s, s.current_player))
        .collect();

    // --- ONNX (batch=1 only, single thread) ---
    let onnx_result = measure_onnx(&states);

    // --- tch CPU ---
    println!("--- tch CPU ---");
    let cpu_results = {
        let vs = nn::VarStore::new(Device::Cpu);
        let model = BlobNet::new(&vs.root());
        run_sweep(&model, Device::Cpu, &encoded)
    };

    // --- CUDA ---
    #[cfg(feature = "cuda")]
    let cuda_results = if tch::Cuda::is_available() {
        println!("\n--- CUDA ---");
        let vs = nn::VarStore::new(Device::Cuda(0));
        let model = BlobNet::new(&vs.root());
        Some(run_sweep(&model, Device::Cuda(0), &encoded))
    } else {
        println!("\nCUDA not available.");
        None
    };

    #[cfg(not(feature = "cuda"))]
    let cuda_results: Option<Vec<SweepResult>> = {
        println!("\nBuilt without --features cuda.");
        None
    };

    // === Final comparison table ===
    println!();
    println!("============================================================");
    println!("  FINAL TABLE — seconds per game (lower is better)");
    println!("  Evals/game = {:.0},  ONNX column = 32× parallel sessions", EVALS_PER_GAME);
    println!("============================================================\n");

    // Header
    let has_cuda = cuda_results.is_some();
    if has_cuda {
        println!(
            "{:>6}  {:>14}  {:>14}  {:>14}",
            "batch", "CUDA (s/game)", "tch-CPU×1", "ONNX×32"
        );
        println!("{}", "-".repeat(56));
    } else {
        println!(
            "{:>6}  {:>14}  {:>14}",
            "batch", "tch-CPU×1", "ONNX×32"
        );
        println!("{}", "-".repeat(40));
    }

    // ONNX row: batch=1 only, but 32 threads in parallel
    let onnx_sec_per_game_32t = match &onnx_result {
        Some(r) => EVALS_PER_GAME / (r.evals_per_sec * 32.0),
        None => f64::NAN,
    };

    for cpu_r in &cpu_results {
        let bs = cpu_r.batch_size;
        // CPU: single-core throughput (no threading — raw forward speed)
        let cpu_sec_per_game = EVALS_PER_GAME / cpu_r.evals_per_sec;

        let cuda_cell = if let Some(ref cuda_res) = cuda_results {
            if let Some(cr) = cuda_res.iter().find(|r| r.batch_size == bs) {
                // GPU: one inference thread, batched
                let sec = EVALS_PER_GAME / cr.evals_per_sec;
                format!("{:>12.1}s", sec)
            } else {
                "          n/a".to_string()
            }
        } else {
            String::new()
        };

        // ONNX column only on the batch=1 row (ONNX is always batch=1)
        let onnx_cell = if bs == 1 {
            if onnx_sec_per_game_32t.is_finite() {
                format!("{:>12.1}s", onnx_sec_per_game_32t)
            } else {
                "     no model".to_string()
            }
        } else {
            "            -".to_string()
        };

        if has_cuda {
            println!(
                "{:>6}  {:>14}  {:>12.1}s  {:>14}",
                bs, cuda_cell, cpu_sec_per_game, onnx_cell
            );
        } else {
            println!(
                "{:>6}  {:>12.1}s  {:>14}",
                bs, cpu_sec_per_game, onnx_cell
            );
        }
    }

    // Summary with 32-thread / batched extrapolations
    println!();
    println!("--- Extrapolated iteration time (118 games, 5×100 MCTS) ---");
    let games = 118.0;

    if let Some(ref onnx_r) = onnx_result {
        let onnx_iter = (games * EVALS_PER_GAME) / (onnx_r.evals_per_sec * 32.0);
        println!("  ONNX ×32 threads (batch=1):     {:>7.1}s  ({:.1} min)", onnx_iter, onnx_iter / 60.0);
    }

    // CPU best batch (single core × 32 threads)
    if let Some(best_cpu) = cpu_results.iter().max_by(|a, b| a.evals_per_sec.partial_cmp(&b.evals_per_sec).unwrap()) {
        let cpu_iter = (games * EVALS_PER_GAME) / (best_cpu.evals_per_sec * 32.0);
        println!(
            "  tch-CPU ×32 threads (batch={}):  {:>7.1}s  ({:.1} min)",
            best_cpu.batch_size, cpu_iter, cpu_iter / 60.0
        );
    }

    if let Some(ref cuda_res) = cuda_results {
        if let Some(best_cuda) = cuda_res.iter().max_by(|a, b| a.evals_per_sec.partial_cmp(&b.evals_per_sec).unwrap()) {
            let cuda_iter = (games * EVALS_PER_GAME) / best_cuda.evals_per_sec;
            println!(
                "  CUDA ×1 stream (batch={}):     {:>7.1}s  ({:.1} min)",
                best_cuda.batch_size, cuda_iter, cuda_iter / 60.0
            );
        }
    }
}

fn run_sweep(
    model: &BlobNet,
    device: Device,
    encoded: &[EncodedState],
) -> Vec<SweepResult> {
    println!(
        "{:>6}  {:>12}  {:>12}  {:>14}",
        "batch", "fwd (ms)", "per-eval(us)", "evals/sec"
    );
    println!("{}", "-".repeat(52));

    let mut results = Vec::new();

    for &bs in BATCH_SIZES {
        let encs: Vec<_> = encoded.iter().take(bs).cloned().collect();
        let input = pad_batch_with_fixed(&encs, device, true);
        let legal_mask = Tensor::ones([bs as i64, NUM_BIDS], (Kind::Bool, device));

        // Warmup
        for _ in 0..WARMUP_ITERS {
            let _out = tch::no_grad(|| model.forward_bid(&input, &legal_mask, false));
            if device != Device::Cpu {
                let _ = Tensor::zeros([1], (Kind::Float, device)).to_device(Device::Cpu);
            }
        }

        // Timed
        let start = Instant::now();
        for _ in 0..TIMED_ITERS {
            let _out = tch::no_grad(|| model.forward_bid(&input, &legal_mask, false));
        }
        if device != Device::Cpu {
            let _ = Tensor::zeros([1], (Kind::Float, device)).to_device(Device::Cpu);
        }
        let elapsed = start.elapsed();

        let total_fwd_ms = elapsed.as_secs_f64() * 1000.0;
        let per_fwd_ms = total_fwd_ms / TIMED_ITERS as f64;
        let per_eval_us = per_fwd_ms * 1000.0 / bs as f64;
        let evals_per_sec = (bs as f64 * TIMED_ITERS as f64) / elapsed.as_secs_f64();

        println!(
            "{:>6}  {:>10.3}ms  {:>10.1}us  {:>12.0}",
            bs, per_fwd_ms, per_eval_us, evals_per_sec
        );

        results.push(SweepResult {
            batch_size: bs,
            per_eval_us,
            evals_per_sec,
        });
    }
    results
}

fn measure_onnx(states: &[blob_engine::state::BlobState]) -> Option<SweepResult> {
    let onnx_path = match std::env::var("BLOB_ONNX_MODEL") {
        Ok(p) => p,
        Err(_) => {
            // Try default location
            let default = "checkpoints_smoke/iter_000000/model.onnx";
            if std::path::Path::new(default).exists() {
                default.to_string()
            } else {
                println!("--- ONNX ---");
                println!("  Skipped (set BLOB_ONNX_MODEL or place model at {default})");
                return None;
            }
        }
    };

    println!("--- ONNX (batch=1, ort CPU, intra_threads=1) ---");
    println!("  Model: {onnx_path}");

    let eval = match OnnxEvaluator::from_file(&onnx_path) {
        Ok(e) => e,
        Err(e) => {
            println!("  Failed to load: {e}");
            return None;
        }
    };

    // Warmup
    for s in states.iter().take(WARMUP_ITERS) {
        let _ = eval.evaluate(s);
    }

    // Timed: 200 sequential evaluations
    let n = 200;
    let start = Instant::now();
    for i in 0..n {
        let _ = eval.evaluate(&states[i % states.len()]);
    }
    let elapsed = start.elapsed();

    let per_eval_ms = elapsed.as_secs_f64() * 1000.0 / n as f64;
    let per_eval_us = per_eval_ms * 1000.0;
    let evals_per_sec = n as f64 / elapsed.as_secs_f64();

    println!(
        "  per-eval: {:.3} ms ({:.0} us)  →  {:.0} evals/sec  →  ×32 threads: {:.0} evals/sec",
        per_eval_ms, per_eval_us, evals_per_sec, evals_per_sec * 32.0
    );

    Some(SweepResult {
        batch_size: 1,
        per_eval_us,
        evals_per_sec,
    })
}
