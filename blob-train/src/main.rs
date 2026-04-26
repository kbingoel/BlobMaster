//! blob-train — training / evaluation / self-play / export CLI.
//!
//! Session 7.1 wires the `train` subcommand to the `TrainingLoop` driver:
//!   1. Load the TOML config, apply CLI overrides.
//!   2. Build (or resume) a `TrainingLoop`.
//!   3. Seed the first iteration's ONNX model from the in-memory weights
//!      (unless resuming, in which case the latest iter's `model.onnx` is
//!      reused).
//!   4. Loop `total_iterations` times: run one iteration, export the
//!      freshly-trained weights via `scripts/export_onnx.py`, and use the
//!      produced `model.onnx` as self-play seed for the next iteration.
//!   5. Every `eval.eval_interval` iterations (skipping the anchor), pit
//!      the current model against the anchor checkpoint via
//!      `blob_nn::eval::run_evaluation` and append the result to
//!      `strength.csv`.

mod config;

use std::path::{Path, PathBuf};
use std::process::Command as ProcCommand;

use blob_nn::eval::{
    append_strength_row, iteration_onnx_path, run_evaluation, StrengthRow,
};
use blob_nn::training_loop::TrainingLoop;
use clap::{Parser, Subcommand};
use rand_xoshiro::rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::config::TrainingConfig;

#[derive(Parser, Debug)]
#[command(
    name = "blobmaster-train",
    about = "Blob training / evaluation / self-play / export CLI.",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Train {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        resume: bool,
        #[arg(long)]
        checkpoint_dir: Option<PathBuf>,
        #[arg(long)]
        batch_size: Option<usize>,
        #[arg(long)]
        num_games: Option<usize>,
        #[arg(long)]
        num_threads: Option<usize>,
        #[arg(long)]
        device: Option<String>,
    },
    Evaluate {
        #[arg(long)]
        model_a: PathBuf,
        /// Path to opponent ONNX model.
        #[arg(long)]
        model_b: PathBuf,
        /// Game cap; the evaluator stops earlier as soon as the Wilson
        /// 95% CI clears the ±0.55/0.45 bands (chunks of 50).
        #[arg(long)]
        num_games: usize,
        #[arg(long)]
        num_players: u8,
        #[arg(long)]
        cards_dealt: u8,
        /// Optional config TOML — its `[mcts]` section is used so the
        /// eval matches the training-time MCTS budget. Defaults to
        /// `MctsConfig::default()` (5 × 100, c_puct=1.5).
        #[arg(long)]
        config: Option<PathBuf>,
        /// Optional RNG seed for reproducibility.
        #[arg(long, default_value_t = 0xE5A1_5EEDu64)]
        seed: u64,
    },
    SelfPlay {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        num_games: usize,
        #[arg(long)]
        output: PathBuf,
    },
    Export {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Self-play profiler — plays `games_per_thread * num_threads` games
    /// through the live rayon engine and prints a bucket breakdown of
    /// time spent in MCTS, ONNX, encoding, determinization, etc.
    Profile {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 5)]
        games_per_thread: usize,
        #[arg(long, default_value_t = 32)]
        num_threads: usize,
        #[arg(long)]
        num_players: Option<u8>,
        #[arg(long)]
        cards_dealt: Option<u8>,
        /// Optional config TOML — its `[mcts]` section is used so profiling
        /// matches the real self-play MCTS budget. Defaults to
        /// `MctsConfig::default()` (5 × 100, c_puct=1.5).
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long, default_value_t = 0xB10B_5EEDu64)]
        seed: u64,
        /// Session 7.4b: load the `model.int8.onnx` sibling of `--model`
        /// instead of the FP32 path. Useful for the post-quantization
        /// per-call profile (gate: ONNX < 0.9 ms vs ~1.30 ms FP32 at 16T).
        #[arg(long, default_value_t = false)]
        use_int8: bool,
        /// Session 7.4b: capture up to `dump_calibration_limit` encoded
        /// states to this file (BCAL binary format) for use as
        /// `quantize_static` calibration data in `scripts/export_onnx.py`.
        /// Run with the FP32 model; pair with `--use-int8 false`.
        #[arg(long)]
        dump_calibration: Option<PathBuf>,
        #[arg(long, default_value_t = 500)]
        dump_calibration_limit: usize,
    },
}

fn init_logging() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}

fn apply_overrides(
    cfg: &mut TrainingConfig,
    checkpoint_dir: Option<PathBuf>,
    batch_size: Option<usize>,
    num_games: Option<usize>,
    num_threads: Option<usize>,
    device: Option<String>,
) -> Result<(), String> {
    if let Some(dir) = checkpoint_dir {
        cfg.training.checkpoint_dir = dir;
    }
    if let Some(bs) = batch_size {
        cfg.training.batch_size = bs;
    }
    if let Some(n) = num_games {
        cfg.self_play.num_games = n;
    }
    if let Some(t) = num_threads {
        cfg.self_play.num_threads = t;
    }
    if let Some(d) = device {
        cfg.training.device = parse_device(&d)?;
    }
    Ok(())
}

fn parse_device(tag: &str) -> Result<tch::Device, String> {
    let t = tag.to_ascii_lowercase();
    if t == "cpu" {
        Ok(tch::Device::Cpu)
    } else if t == "mps" {
        Ok(tch::Device::Mps)
    } else if t == "vulkan" {
        Ok(tch::Device::Vulkan)
    } else if t == "cuda" {
        Ok(tch::Device::Cuda(0))
    } else if let Some(rest) = t.strip_prefix("cuda:") {
        let i: usize = rest
            .parse()
            .map_err(|e| format!("invalid cuda index: {e}"))?;
        Ok(tch::Device::Cuda(i))
    } else {
        Err(format!("unknown device tag: {tag}"))
    }
}

/// Optional INT8 export settings — when `Some`, `run_export_script` also
/// emits `model.int8.onnx` next to the FP32 file.
#[derive(Debug, Clone)]
struct Int8ExportArgs {
    /// `…/model.int8.onnx` to emit.
    int8_out: PathBuf,
    /// Calibration file (BCAL binary; produced by `profile --dump-calibration`).
    calibration: PathBuf,
}

/// Invoke `scripts/export_onnx.py --weights <ot> --out <onnx>`. When `int8`
/// is supplied, also passes `--int8-out` and `--calibration` so the script
/// emits the QDQ-quantized sibling in the same invocation. Returns an
/// `io::Error` if the script exits non-zero so the caller can propagate it
/// through `run_iteration`'s error channel.
fn run_export_script(
    ot_path: &Path,
    onnx_path: &Path,
    int8: Option<&Int8ExportArgs>,
) -> std::io::Result<()> {
    let mut cmd = ProcCommand::new("python3");
    cmd.env_remove("LD_PRELOAD")
        .arg("scripts/export_onnx.py")
        .arg("--weights")
        .arg(ot_path)
        .arg("--out")
        .arg(onnx_path);
    if let Some(i) = int8 {
        cmd.arg("--int8-out").arg(&i.int8_out);
        cmd.arg("--calibration").arg(&i.calibration);
    }
    let status = cmd.status()?;
    if !status.success() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("export_onnx.py failed: {status}"),
        ));
    }
    Ok(())
}

/// Build the per-iteration INT8 export args from the training config. Returns
/// `None` when `use_int8` is false; warns and returns `None` (so the run
/// degrades cleanly to FP32) if the calibration file is missing.
fn int8_args_for(cfg: &TrainingConfig, onnx_path: &Path) -> Option<Int8ExportArgs> {
    if !cfg.self_play.use_int8 {
        return None;
    }
    let calibration = cfg.training.checkpoint_dir.join("calibration.bin");
    if !calibration.exists() {
        tracing::warn!(
            ?calibration,
            "use_int8 set but calibration file missing — run \
             `blobmaster-train profile --dump-calibration <path>` once first; \
             skipping INT8 export this iter"
        );
        return None;
    }
    Some(Int8ExportArgs {
        int8_out: blob_nn::engine::int8_model_path(onnx_path),
        calibration,
    })
}

/// Export the in-memory weights to a bootstrap ONNX so the first iteration
/// has an evaluator to drive self-play.
fn bootstrap_initial_onnx(
    tl: &TrainingLoop,
    cfg: &TrainingConfig,
) -> std::io::Result<PathBuf> {
    use blob_nn::train::save_checkpoint;
    let dir = tl.cfg.checkpoint_dir.join("bootstrap");
    std::fs::create_dir_all(&dir)?;
    save_checkpoint(&tl.vs, 0, &dir)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    let ot = dir.join("model.ot");
    let onnx = dir.join("model.onnx");
    let int8 = int8_args_for(cfg, &onnx);
    run_export_script(&ot, &onnx, int8.as_ref())?;
    Ok(onnx)
}

fn run_train(mut cfg: TrainingConfig, resume: bool) -> std::io::Result<()> {
    let mut tl = TrainingLoop::new(cfg.training.clone());

    let resumed_from = if resume {
        tl.try_resume()?
    } else {
        None
    };

    // Locate the ONNX model that drives the next iteration's self-play.
    let mut onnx_path: PathBuf = if let Some(iter_resumed) = resumed_from {
        let p = iteration_onnx_path(&cfg.training.checkpoint_dir, iter_resumed);
        if !p.exists() {
            tracing::warn!(?p, "resumed checkpoint missing model.onnx; re-exporting");
            bootstrap_initial_onnx(&tl, &cfg)?
        } else {
            p
        }
    } else {
        bootstrap_initial_onnx(&tl, &cfg)?
    };

    // The anchor is the first saved iteration (= the `try_resume` baseline
    // if resuming, else iter 0 once produced). Eval against it starting at
    // `iteration == eval_interval`.
    let anchor_iter: u64 = tl.iteration;

    cfg.self_play.iteration = tl.iteration;

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB10B_5EED ^ tl.iteration);
    let total = cfg.training.total_iterations;
    tracing::info!(
        iterations = total,
        start = tl.iteration,
        ?onnx_path,
        "train — starting driver loop"
    );
    for _ in 0..total {
        let iter = tl.iteration;
        cfg.self_play.iteration = iter;
        let started = std::time::Instant::now();
        let cfg_for_export = cfg.clone();
        let metrics = tl.run_iteration(
            &mut rng,
            &cfg.self_play,
            &cfg.mcts,
            &onnx_path,
            |ot, onnx| {
                let int8 = int8_args_for(&cfg_for_export, onnx);
                run_export_script(ot, onnx, int8.as_ref())
            },
        )?;
        let elapsed = started.elapsed();
        tracing::info!(
            iteration = iter,
            wall_clock_secs = elapsed.as_secs_f64(),
            bid_policy_loss = metrics.bid_policy_loss,
            play_policy_loss = metrics.play_policy_loss,
            value_loss = metrics.value_loss,
            combined_loss = metrics.combined_loss,
            visit_entropy_mean = metrics.visit_entropy_mean,
            examples = metrics.examples_generated,
            decisions = metrics.num_decisions,
            signal_p50_mid = metrics.signal_p50_mid,
            "iteration complete"
        );

        onnx_path = iteration_onnx_path(&cfg.training.checkpoint_dir, iter);

        if cfg.eval.eval_interval > 0
            && iter > anchor_iter
            && iter % cfg.eval.eval_interval == 0
        {
            if let Err(e) = run_eval_against_anchor(&cfg, anchor_iter, iter, &metrics, &mut rng) {
                tracing::warn!(error = %e, "periodic evaluation failed");
            }
        }
    }

    Ok(())
}

fn run_eval_against_anchor(
    cfg: &TrainingConfig,
    anchor_iter: u64,
    current_iter: u64,
    metrics: &blob_nn::training_loop::IterationMetrics,
    rng: &mut Xoshiro256PlusPlus,
) -> std::io::Result<()> {
    let anchor_onnx = iteration_onnx_path(&cfg.training.checkpoint_dir, anchor_iter);
    let current_onnx = iteration_onnx_path(&cfg.training.checkpoint_dir, current_iter);
    if !anchor_onnx.exists() || !current_onnx.exists() {
        tracing::warn!(?anchor_onnx, ?current_onnx, "eval: missing ONNX; skipping");
        return Ok(());
    }
    let (n_players, cards) = cfg.self_play.fixed_player_count.unwrap_or((5, 7));
    // Pull a single u64 from the training-loop RNG so eval remains
    // reproducible across runs (downstream per-game seeds come from this).
    let base_seed: u64 = rng.next_u64();
    let result = run_evaluation(
        &current_onnx,
        &anchor_onnx,
        cfg.eval.eval_games,
        n_players,
        cards,
        &cfg.mcts,
        base_seed,
        cfg.eval.eval_num_threads,
    );
    tracing::info!(
        current_iter,
        anchor_iter,
        win_rate = result.win_rate,
        win_rate_lower95 = result.win_rate_lower95,
        bid_success_current = result.bid_success_rate_a,
        bid_success_anchor = result.bid_success_rate_b,
        "eval vs anchor"
    );
    append_strength_row(
        &cfg.training.checkpoint_dir,
        &StrengthRow {
            iteration: current_iter,
            opponent: format!("iter_{anchor_iter:06}"),
            win_rate: result.win_rate,
            win_rate_lower95: result.win_rate_lower95,
            win_rate_upper95: result.win_rate_upper95,
            score_differential: result.score_differential,
            bid_success_rate_current: result.bid_success_rate_a,
            bid_success_rate_opponent: result.bid_success_rate_b,
            policy_loss: metrics.play_policy_loss,
            value_loss: metrics.value_loss,
            visit_entropy: metrics.visit_entropy_mean,
            kl_divergence: metrics.policy_kl_divergence,
            eval_games_played: result.num_games as u32,
            eval_inconclusive: result.inconclusive,
        },
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_profile(
    model: &Path,
    games_per_thread: usize,
    num_threads: usize,
    num_players: Option<u8>,
    cards_dealt: Option<u8>,
    config: Option<&Path>,
    seed: u64,
    use_int8: bool,
    dump_calibration: Option<&Path>,
    dump_calibration_limit: usize,
) -> std::io::Result<()> {
    use blob_engine::profiling;
    use blob_nn::engine::{self_play_iteration, SelfPlayConfig};

    let mcts_cfg = if let Some(p) = config {
        TrainingConfig::load(p)?.mcts
    } else {
        blob_engine::mcts::MctsConfig::default()
    };

    let fixed = match (num_players, cards_dealt) {
        (Some(n), Some(c)) => Some((n, c)),
        (None, None) => None,
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "pass both --num-players and --cards-dealt, or neither",
            ));
        }
    };

    let num_games = games_per_thread.saturating_mul(num_threads);
    let sp_cfg = SelfPlayConfig {
        num_games,
        num_threads,
        iteration: seed,
        show_progress: false,
        fixed_player_count: fixed,
        use_int8,
    };

    if dump_calibration.is_some() {
        blob_engine::start_calibration_capture(dump_calibration_limit);
    }

    tracing::info!(
        ?model,
        num_games,
        num_threads,
        games_per_thread,
        ?fixed,
        c_puct = mcts_cfg.c_puct,
        num_determinizations = mcts_cfg.num_determinizations,
        sims_per_determinization = mcts_cfg.sims_per_determinization,
        "profile — starting self-play profiling run"
    );

    profiling::reset_all();
    profiling::enable();
    let started = std::time::Instant::now();
    let (examples, stats) = self_play_iteration(model, &sp_cfg, &mcts_cfg);
    let wall = started.elapsed();
    profiling::disable();

    if let Some(out) = dump_calibration {
        let captured = blob_engine::finish_calibration_capture();
        if captured.is_empty() {
            tracing::warn!(?out, "dump-calibration: no states captured");
        } else {
            blob_engine::write_calibration_file(out, &captured)?;
            tracing::info!(
                ?out,
                num_states = captured.len(),
                limit = dump_calibration_limit,
                "dump-calibration: wrote BCAL file"
            );
        }
    }

    let thread_seconds_ns = (wall.as_nanos() as u64).saturating_mul(num_threads as u64);

    let total_sims: u64 = stats.iter().map(|s| s.sims_used as u64).sum();
    let num_decisions = stats.len();

    println!();
    println!("=== blobmaster-train profile ===");
    println!("games               : {num_games} ({games_per_thread} × {num_threads} threads)");
    println!("fixed_player_count  : {fixed:?}");
    println!(
        "mcts                : {} det × {} sims (floor={}, c_puct={})",
        mcts_cfg.num_determinizations,
        mcts_cfg.sims_per_determinization,
        mcts_cfg.min_sims_floor,
        mcts_cfg.c_puct
    );
    println!("wall clock (s)      : {:.3}", wall.as_secs_f64());
    println!("thread-seconds      : {:.3}", thread_seconds_ns as f64 / 1e9);
    println!("decisions           : {num_decisions}");
    println!("examples            : {}", examples.len());
    println!("total sims          : {total_sims}");
    if num_decisions > 0 {
        println!(
            "avg per-game wall   : {:.3} ms",
            wall.as_secs_f64() * 1000.0 / num_games as f64
        );
        println!(
            "avg per-decision    : {:.3} ms  ({:.1} decisions/game)",
            (thread_seconds_ns as f64 / 1e6) / num_decisions as f64,
            num_decisions as f64 / num_games as f64
        );
    }
    println!();

    println!(
        "{:<22} {:>14} {:>10} {:>14} {:>8} {:>8}",
        "bucket", "total_ms", "calls", "avg_us", "%wall", "%threads"
    );
    println!("{}", "-".repeat(80));
    for b in profiling::ALL_BUCKETS {
        let (nanos, count) = b.snapshot();
        let ms = nanos as f64 / 1e6;
        let avg_us = if count > 0 {
            (nanos as f64 / 1e3) / count as f64
        } else {
            0.0
        };
        let pct_wall = 100.0 * nanos as f64 / wall.as_nanos() as f64;
        let pct_threads = if thread_seconds_ns > 0 {
            100.0 * nanos as f64 / thread_seconds_ns as f64
        } else {
            0.0
        };
        println!(
            "{:<22} {:>14.2} {:>10} {:>14.2} {:>7.1}% {:>7.1}%",
            b.name, ms, count, avg_us, pct_wall, pct_threads
        );
    }
    println!();
    println!(
        "Notes: buckets are nested — ONNX_* are a sub-slice of MCTS_SEARCH. %wall is"
    );
    println!(
        "summed-thread-time over wall clock (>100% when multi-threaded, divided by"
    );
    println!("num_threads gives per-thread share). %threads is share of wall × threads.");
    Ok(())
}

fn run_evaluate(
    model_a: &Path,
    model_b: &Path,
    num_games: usize,
    num_players: u8,
    cards_dealt: u8,
    config: Option<&Path>,
    seed: u64,
) -> std::io::Result<()> {
    let (mcts_cfg, num_threads) = if let Some(p) = config {
        let loaded = TrainingConfig::load(p)?;
        (loaded.mcts, loaded.eval.eval_num_threads)
    } else {
        (blob_engine::mcts::MctsConfig::default(), 32)
    };
    tracing::info!(
        ?model_a,
        ?model_b,
        cap_games = num_games,
        num_players,
        cards_dealt,
        num_threads,
        c_puct = mcts_cfg.c_puct,
        num_determinizations = mcts_cfg.num_determinizations,
        sims_per_determinization = mcts_cfg.sims_per_determinization,
        "evaluate — starting head-to-head (parallel, adaptive early-stop)"
    );
    let result = run_evaluation(
        model_a,
        model_b,
        num_games,
        num_players,
        cards_dealt,
        &mcts_cfg,
        seed,
        num_threads,
    );
    tracing::info!(
        games_played = result.num_games,
        wins_a = result.wins_a,
        win_rate = result.win_rate,
        win_rate_lower95 = result.win_rate_lower95,
        win_rate_upper95 = result.win_rate_upper95,
        score_differential = result.score_differential,
        bid_success_a = result.bid_success_rate_a,
        bid_success_b = result.bid_success_rate_b,
        inconclusive = result.inconclusive,
        "evaluate — result"
    );
    Ok(())
}

fn main() {
    init_logging();

    let cli = Cli::parse();
    match cli.command {
        Command::Train {
            config,
            resume,
            checkpoint_dir,
            batch_size,
            num_games,
            num_threads,
            device,
        } => {
            let mut cfg = match TrainingConfig::load(&config) {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!(path = ?config, error = %e, "failed to load config");
                    std::process::exit(1);
                }
            };
            if let Err(e) = apply_overrides(
                &mut cfg,
                checkpoint_dir,
                batch_size,
                num_games,
                num_threads,
                device,
            ) {
                tracing::error!(error = %e, "failed to apply CLI overrides");
                std::process::exit(1);
            }
            tracing::info!(
                resume,
                checkpoint_dir = ?cfg.training.checkpoint_dir,
                batch_size = cfg.training.batch_size,
                num_games = cfg.self_play.num_games,
                num_threads = cfg.self_play.num_threads,
                total_iterations = cfg.training.total_iterations,
                fixed_player_count = ?cfg.self_play.fixed_player_count,
                "train — driver starting"
            );
            if let Err(e) = run_train(cfg, resume) {
                tracing::error!(error = %e, "training run failed");
                std::process::exit(1);
            }
        }
        Command::Evaluate {
            model_a,
            model_b,
            num_games,
            num_players,
            cards_dealt,
            config,
            seed,
        } => {
            if let Err(e) = run_evaluate(
                &model_a,
                &model_b,
                num_games,
                num_players,
                cards_dealt,
                config.as_deref(),
                seed,
            ) {
                tracing::error!(error = %e, "evaluation failed");
                std::process::exit(1);
            }
        }
        Command::SelfPlay {
            model,
            num_games,
            output,
        } => {
            tracing::info!(?model, num_games, ?output, "self-play — driver wiring lands in later session");
        }
        Command::Export { checkpoint, output } => {
            tracing::info!(?checkpoint, ?output, "export — driver wiring lands in later session");
        }
        Command::Profile {
            model,
            games_per_thread,
            num_threads,
            num_players,
            cards_dealt,
            config,
            seed,
            use_int8,
            dump_calibration,
            dump_calibration_limit,
        } => {
            if let Err(e) = run_profile(
                &model,
                games_per_thread,
                num_threads,
                num_players,
                cards_dealt,
                config.as_deref(),
                seed,
                use_int8,
                dump_calibration.as_deref(),
                dump_calibration_limit,
            ) {
                tracing::error!(error = %e, "profile run failed");
                std::process::exit(1);
            }
        }
    }
}
