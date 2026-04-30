#!/usr/bin/env bash
# Trajectory evaluation: four 192-game 5P7C evaluate calls to recover the
# in-loop eval rows that the prune-checkpoints / resume interaction
# silently dropped (iter_000016 was pruned before iter 20's eval could
# read it; same for iter 25). Then iter_29 vs iter_0 (real trajectory)
# and iter_29 vs iter_15 (resume-window gain).
#
# Output: logs/sweep-2026-04-28/trajectory.{log,md}

set -u
set -o pipefail

REPO_ROOT="/home/kbuntu/Documents/Github/BlobMaster"
cd "$REPO_ROOT"

CKPT_DIR="checkpoints/sweep-2026-04-28-anchor"
LOG_DIR="logs/sweep-2026-04-28"
mkdir -p "$LOG_DIR"
OUT_LOG="$LOG_DIR/trajectory.log"
OUT_MD="$LOG_DIR/trajectory.md"
CFG="blob-train/sweep-2026-04-28/anchor.toml"

# ─── runtime env ───
LIBTORCH_DIR="$(find target/release/build -maxdepth 6 -type d -name lib -path '*/libtorch/libtorch/lib' | head -n1)"
[[ -n "$LIBTORCH_DIR" ]] || { echo "FATAL: libtorch dir not found" >&2; exit 1; }
export LD_LIBRARY_PATH="$LIBTORCH_DIR:${LD_LIBRARY_PATH:-}"
[[ -f "$LIBTORCH_DIR/libtorch_cuda.so" ]] && \
    export LD_PRELOAD="$LIBTORCH_DIR/libtorch_cuda.so${LD_PRELOAD:+:$LD_PRELOAD}"
[[ -x "$REPO_ROOT/.venv/bin/python3" ]] && export PATH="$REPO_ROOT/.venv/bin:$PATH"
export RUST_LOG="${RUST_LOG:-info}"   # `evaluate` emits its result line via
                                      # tracing::info!, so we MUST allow info
                                      # logs through or stdout has nothing to
                                      # parse. (First version of this script
                                      # set RUST_LOG=warn and the trajectory
                                      # table came out as all-? rows.)
BIN="./target/release/blobmaster-train"

: > "$OUT_LOG"
: > "$OUT_MD"

# ─── header ───
{
    echo "# Trajectory eval — sweep-2026-04-28 anchor"
    echo
    echo "Started: $(date -Iseconds)"
    echo
    echo "All evals: 192 games, 5 players, 7 cards (matches the 7.3c historical anchor)."
    echo "Early-stop active — terminates as soon as Wilson lo95 ≥ 0.55 or hi95 ≤ 0.45."
    echo
    echo "| pair | wins | games | win_rate | lo95 | hi95 | score_diff | bid_succ_a | bid_succ_b | wall (s) |"
    echo "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
} >> "$OUT_MD"

# Run one evaluate, parse, append a markdown row.
run_one () {
    local label="$1" model_a="$2" model_b="$3"

    echo "================================================================" >> "$OUT_LOG"
    echo "=== $label ===" >> "$OUT_LOG"
    echo "================================================================" >> "$OUT_LOG"
    echo "[$(date +%H:%M:%S)] starting: $label" >&2

    local started=$EPOCHSECONDS
    "$BIN" evaluate \
        --model-a "$model_a" \
        --model-b "$model_b" \
        --num-games 192 \
        --num-players 5 \
        --cards-dealt 7 \
        --config "$CFG" >> "$OUT_LOG" 2>&1
    local rc=$?
    local elapsed=$((EPOCHSECONDS - started))

    if [[ $rc -ne 0 ]]; then
        echo "| $label | — | — | **rc=$rc (FAILED)** | — | — | — | — | — | $elapsed |" >> "$OUT_MD"
        echo "[$(date +%H:%M:%S)] FAILED: $label (rc=$rc, ${elapsed}s)" >&2
        return
    fi

    # Final eval-result line example:
    #   eval result wins=131 games=192 win_rate=0.682 win_rate_lower95=0.612 ...
    # Pull from `eval result` if present, else from final `eval: CI update`.
    local last_ci
    last_ci=$(grep -E "eval (result|vs anchor)|eval: CI update" "$OUT_LOG" | tail -n1)
    local wins games wr lo hi sd bsa bsb
    wins=$(grep -oE "wins(_a)?=[0-9]+" <<<"$last_ci" | tail -n1 | grep -oE "[0-9]+")
    games=$(grep -oE "games(_played)?=[0-9]+" <<<"$last_ci" | tail -n1 | grep -oE "[0-9]+")
    wr=$(grep -oE "win_rate=[0-9.]+" <<<"$last_ci" | head -n1 | grep -oE "[0-9.]+")
    lo=$(grep -oE "lower95=[0-9.]+|win_rate_lower95=[0-9.]+" <<<"$last_ci" | head -n1 | grep -oE "[0-9.]+")
    hi=$(grep -oE "upper95=[0-9.]+|win_rate_upper95=[0-9.]+" <<<"$last_ci" | head -n1 | grep -oE "[0-9.]+")

    # Score diff + bid success — need a separate grep on the full log for this run.
    sd=$(grep -oE "score_differential=[-0-9.]+" "$OUT_LOG" | tail -n1 | grep -oE "[-0-9.]+")
    bsa=$(grep -oE "bid_success_(rate_)?(current|a)=[0-9.]+" "$OUT_LOG" | tail -n1 | grep -oE "[0-9.]+")
    bsb=$(grep -oE "bid_success_(rate_)?(opponent|b)=[0-9.]+" "$OUT_LOG" | tail -n1 | grep -oE "[0-9.]+")

    echo "| $label | ${wins:-?} | ${games:-?} | ${wr:-?} | ${lo:-?} | ${hi:-?} | ${sd:-?} | ${bsa:-?} | ${bsb:-?} | $elapsed |" >> "$OUT_MD"
    echo "[$(date +%H:%M:%S)] done: $label  → wr=${wr:-?} lo95=${lo:-?} (${elapsed}s)" >&2
}

# ─── the four evals ───
run_one "iter_29 vs iter_0  (full trajectory)" \
    "$CKPT_DIR/iter_000029/model.onnx" \
    "$CKPT_DIR/iter_000000/model.onnx"

run_one "iter_25 vs iter_0  (recover skipped in-loop)" \
    "$CKPT_DIR/iter_000025/model.onnx" \
    "$CKPT_DIR/iter_000000/model.onnx"

run_one "iter_20 vs iter_0  (recover skipped in-loop)" \
    "$CKPT_DIR/iter_000020/model.onnx" \
    "$CKPT_DIR/iter_000000/model.onnx"

run_one "iter_29 vs iter_15 (resume-window gain)" \
    "$CKPT_DIR/iter_000029/model.onnx" \
    "$CKPT_DIR/iter_000015/model.onnx"

# ─── footer ───
{
    echo
    echo "Finished: $(date -Iseconds)"
    echo
    echo "Full eval stdout: $OUT_LOG"
} >> "$OUT_MD"

echo
echo "=== trajectory eval complete ==="
cat "$OUT_MD"
