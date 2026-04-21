#!/usr/bin/env bash
# Wrapper to launch blobmaster-train against the tch-downloaded libtorch +
# venv python (for export_onnx.py). Pass the same args you'd pass to the
# binary, e.g. `scripts/run-train.sh train --config foo.toml`.
set -euo pipefail
cd "$(dirname "$0")/.."

LIBTORCH_DIR="$(find target/release/build -maxdepth 5 -type d -name lib -path '*/libtorch/libtorch/lib' | head -n1)"
if [[ -z "${LIBTORCH_DIR:-}" ]]; then
  echo "libtorch lib dir not found — build target/release first" >&2
  exit 1
fi
export LD_LIBRARY_PATH="$PWD/$LIBTORCH_DIR:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$PWD/$LIBTORCH_DIR/libtorch_cuda.so"
export PATH="$PWD/.venv/bin:$PATH"
exec ./target/release/blobmaster-train "$@"
