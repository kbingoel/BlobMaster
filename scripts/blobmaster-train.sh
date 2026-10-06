#!/usr/bin/env bash
# Run the release `blobmaster-train` with the downloaded libtorch on the
# library path and its CUDA backend preloaded (AGENTS.md, Runtime
# environment). Arguments pass through, e.g.
#
#   scripts/blobmaster-train.sh pretrain --config blob-train/pretrain.sample.toml --output checkpoints/pretrain-1
#   scripts/blobmaster-train.sh export --output /tmp/random-model
#
# `export` strips LD_PRELOAD again before it starts Python.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bin="$root/target/release/blobmaster-train"
[[ -x "$bin" ]] || { echo "no $bin; run: cargo build --release -p blob-train" >&2; exit 1; }
# The newest libtorch: the torch-sys-* hash changes whenever tch rebuilds.
lib="$(find "$root/target/release/build" -maxdepth 6 -type d -name lib -path '*/libtorch/libtorch/lib' -printf '%T@ %p\n' \
  | sort -nr | head -n1 | cut -d' ' -f2-)"
[[ -n "$lib" ]] || { echo "no libtorch under target/release/build" >&2; exit 1; }
export LD_LIBRARY_PATH="$lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$lib/libtorch_cuda.so${LD_PRELOAD:+:$LD_PRELOAD}"
exec "$bin" "$@"
