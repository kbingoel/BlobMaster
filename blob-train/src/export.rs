//! `export`: write a model directory (`policy.onnx`, `value.onnx`,
//! `meta.json`; gen-2.md §5.3) with `scripts/export_onnx.py`.

use std::path::{Path, PathBuf};
use std::process::Command;

use blob_nn::train::{CHECKPOINT_META, POLICY_WEIGHTS, VALUE_WEIGHTS};

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

/// Run `scripts/export_onnx.py` on a learner checkpoint directory, or with
/// random-init networks without one. `LD_PRELOAD` is removed: a preloaded
/// tch libtorch crashes the venv's `import torch` (different C++ ABI).
pub fn export(checkpoint: Option<&Path>, output: &Path, check: bool) -> Result<(), String> {
    let mut cmd = Command::new(python());
    cmd.env_remove("LD_PRELOAD")
        .arg(workspace_root().join("scripts/export_onnx.py"))
        .arg("--out-dir")
        .arg(output);
    if let Some(dir) = checkpoint {
        for file in [POLICY_WEIGHTS, VALUE_WEIGHTS, CHECKPOINT_META] {
            if !dir.join(file).is_file() {
                return Err(format!("{}: no {file}; expected a learner checkpoint directory", dir.display()));
            }
        }
        cmd.arg("--checkpoint").arg(dir);
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
