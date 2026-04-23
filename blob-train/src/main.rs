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

/// Invoke `scripts/export_onnx.py --weights <ot> --out <onnx>`. Returns an
/// `io::Error` if the script exits non-zero so the caller can propagate it
/// through `run_iteration`'s error channel.
fn run_export_script(ot_path: &Path, onnx_path: &Path) -> std::io::Result<()> {
    let status = ProcCommand::new("python3")
        .env_remove("LD_PRELOAD")
        .arg("scripts/export_onnx.py")
        .arg("--weights")
        .arg(ot_path)
        .arg("--out")
        .arg(onnx_path)
        .status()?;
    if !status.success() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("export_onnx.py failed: {status}"),
        ));
    }
    Ok(())
}

/// Export the in-memory weights to a bootstrap ONNX so the first iteration
/// has an evaluator to drive self-play.
fn bootstrap_initial_onnx(tl: &TrainingLoop) -> std::io::Result<PathBuf> {
    use blob_nn::train::save_checkpoint;
    let dir = tl.cfg.checkpoint_dir.join("bootstrap");
    std::fs::create_dir_all(&dir)?;
    save_checkpoint(&tl.vs, 0, &dir)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    let ot = dir.join("model.ot");
    let onnx = dir.join("model.onnx");
    run_export_script(&ot, &onnx)?;
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
            bootstrap_initial_onnx(&tl)?
        } else {
            p
        }
    } else {
        bootstrap_initial_onnx(&tl)?
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
        let metrics = tl.run_iteration(
            &mut rng,
            &cfg.self_play,
            &cfg.mcts,
            &onnx_path,
            |ot, onnx| run_export_script(ot, onnx),
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
    }
}
