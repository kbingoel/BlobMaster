//! blob-train — training CLI.
//!
//! - `pretrain`: the supervised warm start from rule bot 2 (gen-2.md §6
//!   Phase 4), ending in a model directory.
//! - `export`: a model directory (`policy.onnx`, `value.onnx`,
//!   `meta.json`) from a learner checkpoint, or random-init.
//!
//! Links libtorch: run it with `LD_LIBRARY_PATH` (and, for CUDA,
//! `LD_PRELOAD`) set as AGENTS.md describes, e.g. through
//! `scripts/blobmaster-train.sh`. `train` (async self-play RL) comes in
//! Phase 5.

mod config;
mod export;
mod pretrain;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use crate::config::PretrainConfig;

#[derive(Parser, Debug)]
#[command(name = "blobmaster-train", about = "Blob training CLI.", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Supervised warm start: play rule-bot-2 rounds, train P and V on them,
    /// measure on held-out rounds, export `<output>/model`.
    Pretrain {
        /// Run directory: config, metrics, checkpoint, held-out report, model.
        #[arg(long)]
        output: PathBuf,
        /// Config TOML (`blob-train/pretrain.sample.toml`); defaults without one.
        #[arg(long, conflicts_with = "resume")]
        config: Option<PathBuf>,
        /// Continue the run in `<output>` from its checkpoint, with its own config.
        #[arg(long)]
        resume: bool,
    },
    /// Write a model directory with `scripts/export_onnx.py`: both networks
    /// from a learner checkpoint, or random-init without one.
    Export {
        /// A learner checkpoint directory (`policy.ot`, `value.ot`, `meta.json`).
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// Model directory to write (created if missing).
        #[arg(long)]
        output: PathBuf,
        /// Also compare the exported graphs with the PyTorch forward passes.
        #[arg(long)]
        check: bool,
    },
}

fn read_config(path: &std::path::Path) -> Result<PretrainConfig, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    PretrainConfig::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Pretrain { output, config, resume } => {
            let cfg = if resume {
                read_config(&output.join(pretrain::CONFIG_FILE))
            } else {
                config.as_deref().map_or_else(|| Ok(PretrainConfig::default()), read_config)
            };
            cfg.and_then(|cfg| pretrain::pretrain(cfg, &output, resume))
        }
        Command::Export { checkpoint, output, check } => export::export(checkpoint.as_deref(), &output, check),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
