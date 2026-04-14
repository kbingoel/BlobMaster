//! blob-bin — inference / deployment CLI. Session 6.2.
//!
//! Intentionally free of training-only dependencies (`tch`, `rayon`,
//! `indicatif`). Session 5.3 invariant: this binary must load and run ONNX
//! models without pulling in libtorch, so it is safe to ship for the
//! Windows + Intel iGPU target in AGENTS.md.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "blobmaster",
    about = "Blob inference and deployment CLI (ONNX only).",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Human-vs-AI scaffolding (filled in in Section 9).
    Play {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        num_players: u8,
        #[arg(long)]
        seat: u8,
    },
    /// Single-state policy+value dump for debugging.
    Analyze {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        state: PathBuf,
    },
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Play {
            model,
            num_players,
            seat,
        } => {
            tracing::info!(?model, num_players, seat, "play — not yet implemented (Section 9)");
        }
        Command::Analyze { model, state } => {
            tracing::info!(?model, ?state, "analyze — not yet implemented (Section 9)");
        }
    }
}
