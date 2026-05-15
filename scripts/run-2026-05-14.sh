#!/usr/bin/env bash
# Launch the 2026-05-14 long run. Wraps `scripts/run-train.sh train --config ...`
# with stdout/stderr captured to logs/ so the user can tail them while the
# run is unattended. The training driver itself never auto-stops — it
# runs while `iter < total_iterations` (230 here, so iter 229 is the
# last processed), or until the user creates `<checkpoint_dir>/STOP`
# (graceful exit at the next iteration boundary; see AGENTS.md).
#
# Resume after a manual stop:
#   scripts/run-2026-05-14.sh --resume
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="blob-train/run-2026-05-14.toml"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_BASE="$LOG_DIR/run-2026-05-14"

extra_args=()
if [[ "${1:-}" == "--resume" ]]; then
  extra_args+=("--resume")
  shift
fi
extra_args+=("$@")

echo "[$(date -Iseconds)] launching blobmaster-train train --config $CONFIG ${extra_args[*]:-}"
echo "  stdout → ${LOG_BASE}.stdout"
echo "  stderr → ${LOG_BASE}.stderr"
echo "  STOP    → checkpoints/run-2026-05-14/STOP (touch to ask for graceful exit)"

exec ./scripts/run-train.sh train \
  --config "$CONFIG" \
  "${extra_args[@]}" \
  > >(tee -a "${LOG_BASE}.stdout") \
  2> >(tee -a "${LOG_BASE}.stderr" >&2)
