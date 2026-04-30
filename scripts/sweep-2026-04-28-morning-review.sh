#!/usr/bin/env bash
# Morning review — what to look at after the overnight anchor resume.
#
# Three things in order:
#   1. Latest saved iter on disk (sanity check the resume actually finished).
#   2. In-loop strength.csv rows from the resume — these are vs iter_000016
#      (the resume baseline), NOT vs iter_000000. Useful for spotting
#      regressions inside the resume window but NOT for trajectory tracking.
#   3. Direct `evaluate` runs at 5P7C (the historical anchor format) of:
#      - iter_000029 vs iter_000000  (full-trajectory health check)
#      - iter_000029 vs iter_000015  (does the second 14 iters keep paying off,
#                                     or is the gain front-loaded?)
#      Each runs 192 games with eval early-stop disabled by passing
#      `--num-games 192` (the harness still bails early if it clears the
#      Wilson bands, which is fine — it just terminates with a clean CI).
#
# Outputs land in:
#   logs/sweep-2026-04-28/morning-review.log   (full stdout)
#   logs/sweep-2026-04-28/morning-review.md    (curated summary)

set -u
set -o pipefail

REPO_ROOT="/home/kbuntu/Documents/Github/BlobMaster"
cd "$REPO_ROOT"

CKPT_DIR="checkpoints/sweep-2026-04-28-anchor"
LOG_DIR="logs/sweep-2026-04-28"
mkdir -p "$LOG_DIR"
OUT_MD="$LOG_DIR/morning-review.md"
OUT_LOG="$LOG_DIR/morning-review.log"
CFG="blob-train/sweep-2026-04-28/anchor.toml"

# ─── runtime env (AGENTS.md canonical launch template) ───
LIBTORCH_DIR="$(find target/release/build -maxdepth 6 -type d -name lib -path '*/libtorch/libtorch/lib' | head -n1)"
[[ -n "$LIBTORCH_DIR" ]] || { echo "FATAL: libtorch dir not found" >&2; exit 1; }
export LD_LIBRARY_PATH="$LIBTORCH_DIR:${LD_LIBRARY_PATH:-}"
[[ -f "$LIBTORCH_DIR/libtorch_cuda.so" ]] && \
    export LD_PRELOAD="$LIBTORCH_DIR/libtorch_cuda.so${LD_PRELOAD:+:$LD_PRELOAD}"
[[ -x "$REPO_ROOT/.venv/bin/python3" ]] && export PATH="$REPO_ROOT/.venv/bin:$PATH"
export RUST_LOG="${RUST_LOG:-info}"
BIN="./target/release/blobmaster-train"

# ─── 1. Sanity: confirm the resume actually finished ───
{
    echo "# Morning review — sweep-2026-04-28 anchor resume (iter 16..29)"
    echo
    echo "Generated: $(date -Iseconds)"
    echo
    echo "## Saved checkpoints"
    echo
    ls -1d "$CKPT_DIR"/iter_* 2>/dev/null | awk -F/ '{print "- " $NF}'
    echo
    LATEST=$(ls -1d "$CKPT_DIR"/iter_* 2>/dev/null | sort | tail -n1 | awk -F/ '{print $NF}')
    echo "Latest on disk: **$LATEST**"
    if [[ "$LATEST" != "iter_000029" ]]; then
        echo
        echo "WARNING: expected iter_000029. Resume may not have finished — check"
        echo "\`$LOG_DIR/resume-anchor.log\` for crash / abort. The eval calls"
        echo "below will still run against whichever checkpoint is latest."
    fi
} | tee "$OUT_MD" > /dev/null

# Pick whichever final iter we actually have (handles partial finish).
FINAL_ITER_DIR=$(ls -1d "$CKPT_DIR"/iter_* 2>/dev/null | sort | tail -n1)
FINAL_ITER_NAME=$(basename "$FINAL_ITER_DIR")
FINAL_MODEL="$FINAL_ITER_DIR/model.onnx"

# ─── 2. In-loop strength.csv ───
{
    echo
    echo "## In-loop strength.csv (vs whichever anchor was active per row)"
    echo
    if [[ -f "$CKPT_DIR/strength.csv" ]]; then
        echo '```'
        # Just iteration / opponent / win_rate / lo95 / hi95 / score_diff /
        # bid_success_current — cols 1-7. Keep header + rows where iteration
        # is one of 5/10/15/20/25 (anchor + resume in-loop).
        awk -F, 'NR==1 || $1 ~ /^(5|10|15|20|25|30)$/ { print $1","$2","$3","$4","$5","$6","$7 }' \
            "$CKPT_DIR/strength.csv" | column -t -s,
        echo '```'
        echo
        echo "Note: rows with opponent=\`iter_000000\` are from the original"
        echo "fresh run (anchor 0..15). Rows with opponent=\`iter_000016\` are"
        echo "from the overnight resume — that's the resume baseline, not"
        echo "the from-scratch anchor."
    else
        echo "(no strength.csv at $CKPT_DIR — something went very wrong)"
    fi
} | tee -a "$OUT_MD" > /dev/null

# ─── 3. Direct evaluate calls (the real trajectory metric) ───
{
    echo
    echo "## Direct evaluate: $FINAL_ITER_NAME vs from-scratch anchor"
    echo
    echo "5P7C is the historical anchor format (matches 7.3c baseline)."
    echo "Both calls are 192 games with the standard early-stop bands."
    echo
} | tee -a "$OUT_MD" > /dev/null

run_eval () {
    local label="$1" model_a="$2" model_b="$3"
    {
        echo
        echo "### $label"
        echo
        echo "\`\`\`"
    } >> "$OUT_MD"

    "$BIN" evaluate \
        --model-a "$model_a" \
        --model-b "$model_b" \
        --num-games 192 \
        --num-players 5 \
        --cards-dealt 7 \
        --config "$CFG" 2>&1 | tee -a "$OUT_LOG" | \
        grep -E "win_rate|score_differential|bid_success|games" | \
        tee -a "$OUT_MD"

    echo "\`\`\`" >> "$OUT_MD"
}

run_eval "$FINAL_ITER_NAME vs iter_000000 (full trajectory)" \
    "$FINAL_MODEL" \
    "$CKPT_DIR/iter_000000/model.onnx"

run_eval "$FINAL_ITER_NAME vs iter_000015 (resume window only)" \
    "$FINAL_MODEL" \
    "$CKPT_DIR/iter_000015/model.onnx"

# ─── 4. Tail of the loss trajectory from metrics.jsonl ───
{
    echo
    echo "## Loss trajectory (last 5 iters)"
    echo
    if [[ -f "$CKPT_DIR/metrics.jsonl" ]]; then
        echo '```'
        tail -5 "$CKPT_DIR/metrics.jsonl" | python3 -c '
import json, sys
for line in sys.stdin:
    m = json.loads(line)
    fields = ["iteration", "bid_policy_loss", "play_policy_loss",
              "value_loss", "combined_loss", "visit_entropy_mean",
              "policy_kl_divergence"]
    print(" ".join(f"{k}={m.get(k)!s:.10}" for k in fields if k in m))
'
        echo '```'
        echo
        echo "Watch for: combined_loss flat or rising (saturation/regression),"
        echo "policy_kl_divergence > 0.15 (LR contingency trigger per dev plan §7.4d),"
        echo "policy_kl_divergence < 0.05 sustained (cut num_games trigger)."
    fi
    echo
    echo "Full review log: $OUT_LOG"
} | tee -a "$OUT_MD" > /dev/null

echo "=== morning review complete ==="
echo
cat "$OUT_MD"
