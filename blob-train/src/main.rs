//! blob-train — training / evaluation / self-play / export CLI.
//!
//! Session 6.2 stands up the command surface and the TOML-driven
//! `TrainingConfig`. The subcommands themselves dispatch into existing
//! `blob_nn` entry points (training loop, evaluation) where available;
//! anything still marked TODO is wired up in later sessions.

mod config;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

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
    /// Run the training loop driven by a TOML config. CLI flags override
    /// file values where both are set.
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
    /// Head-to-head evaluation between two models (or model vs baseline).
    Evaluate {
        #[arg(long)]
        model_a: PathBuf,
        #[arg(long)]
        model_b: String,
        #[arg(long)]
        num_games: usize,
        #[arg(long)]
        num_players: u8,
        #[arg(long)]
        cards_dealt: u8,
    },
    /// Generate self-play examples without training.
    SelfPlay {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        num_games: usize,
        #[arg(long)]
        output: PathBuf,
    },
    /// Export a checkpoint to ONNX (wraps `scripts/export_onnx.py` /
    /// Section 3.5 Rust equivalent).
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
                "train — driver wiring lands in later session"
            );
        }
        Command::Evaluate {
            model_a,
            model_b,
            num_games,
            num_players,
            cards_dealt,
        } => {
            tracing::info!(
                ?model_a,
                model_b,
                num_games,
                num_players,
                cards_dealt,
                "evaluate — driver wiring lands in later session"
            );
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
