//! blob-train — training CLI.
//!
//! Until the gen-2 learner lands (gen-2.md §6 Phase 4: `pretrain`; Phase 5:
//! `train`), the only subcommand is `export`: tch checkpoint → ONNX through
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
    /// Export a tch checkpoint to ONNX with `scripts/export_onnx.py`.
    Export {
        /// `model.ot`, or a checkpoint directory containing it.
        #[arg(long)]
        checkpoint: PathBuf,
        /// Output `.onnx` path.
        #[arg(long)]
        output: PathBuf,
        /// Also compare the exported graph with the PyTorch forward pass.
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
fn export(checkpoint: &Path, output: &Path, check: bool) -> Result<(), String> {
    let weights = if checkpoint.is_dir() {
        checkpoint.join("model.ot")
    } else {
        checkpoint.to_path_buf()
    };
    if !weights.is_file() {
        return Err(format!("no checkpoint at {}", weights.display()));
    }
    let mut cmd = ProcCommand::new(python());
    cmd.env_remove("LD_PRELOAD")
        .arg(workspace_root().join("scripts/export_onnx.py"))
        .arg("--weights")
        .arg(&weights)
        .arg("--out")
        .arg(output);
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
        } => export(&checkpoint, &output, check),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
