#!/usr/bin/env bash
# Diagnostic 2026-05-11 — re-run of the two 2× sims evals only.
#
# The original run produced bit-identical 1× and 2× results because
# `adaptive_budget` in blob-engine/src/mcts.rs:802 hardcodes (5, 100)
# and ignores `cfg.sims_per_determinization`. The 2× config has been
# rewired via `min_sims_floor = 1000`, which IS honored by
# adaptive_budget (raises sims to ceil(floor / dets) = 200). This
# script reruns just the two 2× pairs and appends fresh rows to
# RESULTS.md; the 1× rows from the original run remain valid.

set -u
set -o pipefail

REPO_ROOT="/home/kbuntu/Documents/Github/BlobMaster"
cd "$REPO_ROOT"

CKPT_DIR="checkpoints/run-2026-05-06"
LOG_DIR="logs/diagnostic-2026-05-11"
mkdir -p "$LOG_DIR"
RESULTS_MD="$LOG_DIR/RESULTS.md"
RUN_TS="$(date +%Y%m%dT%H%M%S)"
RUN_LOG="$LOG_DIR/eval-2x-rerun-${RUN_TS}.log"

CFG_2X="blob-train/diagnostic-2026-05-11/mcts-2x.toml"

LIBTORCH_DIR="$(find target/release/build -maxdepth 6 -type d -name lib -path '*/libtorch/libtorch/lib' | head -n1)"
[[ -n "$LIBTORCH_DIR" ]] || { echo "FATAL: libtorch dir not found" >&2; exit 1; }
export LD_LIBRARY_PATH="$LIBTORCH_DIR:${LD_LIBRARY_PATH:-}"
[[ -f "$LIBTORCH_DIR/libtorch_cuda.so" ]] && \
    export LD_PRELOAD="$LIBTORCH_DIR/libtorch_cuda.so${LD_PRELOAD:+:$LD_PRELOAD}"
[[ -x "$REPO_ROOT/.venv/bin/python3" ]] && export PATH="$REPO_ROOT/.venv/bin:$PATH"
export RUST_LOG="${RUST_LOG:-info}"
BIN="./target/release/blobmaster-train"
[[ -x "$BIN" ]] || { echo "FATAL: $BIN not built" >&2; exit 1; }

# Same seeds as the original run, so the 1× and 2× arms remain paired
# game-for-game up to the point MCTS diverges.
SEED_145=305110145
SEED_80=305110080

: > "$RUN_LOG"
echo "=== diagnostic-2026-05-11 2× rerun — start $RUN_TS ==="

run_one_eval () {
    local cur_iter="$1" opp_iter="$2" sims_label="$3" cfg="$4" seed="$5"
    local cur_label="iter_$(printf %d "$cur_iter")"
    local opp_label="iter_$(printf %d "$opp_iter")"
    local cur_onnx="$CKPT_DIR/iter_$(printf %06d "$cur_iter")/model.onnx"
    local opp_onnx="$CKPT_DIR/iter_$(printf %06d "$opp_iter")/model.onnx"

    echo "[$(date +%H:%M:%S)] $cur_label vs $opp_label @ $sims_label …"
    local started=$EPOCHSECONDS
    {
        echo
        echo "================================================================"
        echo "=== $cur_label vs $opp_label @ $sims_label  (seed=$seed) ==="
        echo "================================================================"
        "$BIN" evaluate \
            --model-a "$cur_onnx" \
            --model-b "$opp_onnx" \
            --num-games 192 \
            --num-players 5 \
            --cards-dealt 7 \
            --config "$cfg" \
            --seed "$seed" 2>&1
    } >> "$RUN_LOG"
    local elapsed=$((EPOCHSECONDS - started))

    python3 - "$cur_label" "$opp_label" "$sims_label" "$RUN_TS" \
             "$RESULTS_MD" "$RUN_LOG" "$elapsed" <<'PY'
import re, sys
cur, opp, sims, ts, md_path, log_path, elapsed = sys.argv[1:8]
with open(log_path) as f:
    text = f.read()
text = re.sub(r'\x1b\[[0-9;]*m', '', text)
matches = re.findall(r'evaluate — result\s+([^\n]+)', text)
if not matches:
    print(f"  ! no result line for {cur} vs {opp} @ {sims}", file=sys.stderr)
    sys.exit(0)
fields = dict(re.findall(r'(\w+)=([\w\.\-+e]+)', matches[-1]))
def f(k, fmt=".4f"):
    v = fields.get(k, "")
    try: return format(float(v), fmt)
    except: return "—"
row = (f"| {ts} | {cur} | {opp} | {sims} | "
       f"{fields.get('wins_a','?')}/{fields.get('games_played','?')} | "
       f"{f('win_rate', '.3f')} | {f('win_rate_lower95', '.3f')} | "
       f"{f('win_rate_upper95', '.3f')} | {f('score_differential', '.2f')} | "
       f"{f('bid_success_a', '.3f')} | {f('bid_success_b', '.3f')} | "
       f"{fields.get('inconclusive','?')} | {elapsed} |\n")
with open(md_path, "a") as fh:
    fh.write(row)
print(f"  → wr={f('win_rate', '.3f')} lo95={f('win_rate_lower95', '.3f')} "
      f"hi95={f('win_rate_upper95', '.3f')} inconc={fields.get('inconclusive','?')} "
      f"({elapsed}s)", file=sys.stderr)
PY
}

run_one_eval 145 0 "5x200-floor1000" "$CFG_2X" "$SEED_145"
run_one_eval  80 0 "5x200-floor1000" "$CFG_2X" "$SEED_80"

echo
echo "=== $RESULTS_MD ==="
cat "$RESULTS_MD"
echo
echo "Per-eval stdout: $RUN_LOG"
