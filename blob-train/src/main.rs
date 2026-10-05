//! blob-train — training CLI.
//!
//! Until the gen-2 learner lands (gen-2.md §6 Phase 4: `pretrain`; Phase 5:
//! `train`), the only subcommand is `export`: a model directory
//! (`policy.onnx`, `value.onnx`, `meta.json`) through
//! `scripts/export_onnx.py`.

use std::path::{Path, PathBuf};
use std::process::{Command as ProcCommand, ExitCode};

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "blobmaster-train", about = "Blob training CLI.", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Write a model directory with `scripts/export_onnx.py`: the policy net
    /// from a tch checkpoint (random-init without one) and the value net
    /// (random-init until the Phase-4 learner trains it).
    Export {
        /// The policy net's `model.ot`, or a checkpoint directory containing it.
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

fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("blob-train sits in the workspace")
}

/// The repo's pinned venv if present (AGENTS.md), else `python3` on `PATH`.
fn python() -> PathBuf {
    let venv = workspace_root().join(".venv/bin/python");
    if venv.exists() {
        venv
    } else {
        PathBuf::from("python3")
    }
}

/// Run `scripts/export_onnx.py`. `LD_PRELOAD` is removed: a preloaded tch
/// libtorch crashes the venv's `import torch` (different C++ ABI).
fn export(checkpoint: Option<&Path>, output: &Path, check: bool) -> Result<(), String> {
    let mut cmd = ProcCommand::new(python());
    cmd.env_remove("LD_PRELOAD")
        .arg(workspace_root().join("scripts/export_onnx.py"))
        .arg("--out-dir")
        .arg(output);
    if let Some(checkpoint) = checkpoint {
        let weights =
            if checkpoint.is_dir() { checkpoint.join("model.ot") } else { checkpoint.to_path_buf() };
        if !weights.is_file() {
            return Err(format!("no checkpoint at {}", weights.display()));
        }
        cmd.arg("--weights").arg(weights);
    }
    if check {
        cmd.arg("--check");
    }
    let status = cmd
        .status()
        .map_err(|e| format!("failed to run {}: {e}", cmd.get_program().to_string_lossy()))?;
    if !status.success() {
        return Err(format!("export_onnx.py failed: {status}"));
    }
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Export {
            checkpoint,
            output,
            check,
        } => export(checkpoint.as_deref(), &output, check),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
