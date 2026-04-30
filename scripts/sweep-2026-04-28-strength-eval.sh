#!/usr/bin/env bash
# Strength-eval: run a list of head-to-head 5P7C evals (192 games each,
# early-stop active) and append rows to the persistent strength tracker
# at logs/sweep-2026-04-28/STRENGTH.md.
#
# Usage:
#   bash scripts/sweep-2026-04-28-strength-eval.sh           # default 4-eval set
#   bash scripts/sweep-2026-04-28-strength-eval.sh CUR=29 OPP=15 [CUR=N OPP=M ...]
#
# The default set covers the new (post-Bug-#2-fix) checkpoints:
#   iter_29 vs iter_0,  iter_25 vs iter_0,  iter_20 vs iter_0,  iter_29 vs iter_15
#
# Output:
#   - logs/sweep-2026-04-28/STRENGTH.md  (persistent, append-only — one row per call)
#   - logs/sweep-2026-04-28/strength-eval-<timestamp>.log  (per-invocation eval stdout)

set -u
set -o pipefail

REPO_ROOT="/home/kbuntu/Documents/Github/BlobMaster"
cd "$REPO_ROOT"

CKPT_DIR="checkpoints/sweep-2026-04-28-anchor"
LOG_DIR="logs/sweep-2026-04-28"
mkdir -p "$LOG_DIR"
STRENGTH_MD="$LOG_DIR/STRENGTH.md"
RUN_TS="$(date +%Y%m%dT%H%M%S)"
RUN_LOG="$LOG_DIR/strength-eval-${RUN_TS}.log"
CFG="blob-train/sweep-2026-04-28/anchor.toml"

# ─── runtime env ───
LIBTORCH_DIR="$(find target/release/build -maxdepth 6 -type d -name lib -path '*/libtorch/libtorch/lib' | head -n1)"
[[ -n "$LIBTORCH_DIR" ]] || { echo "FATAL: libtorch dir not found" >&2; exit 1; }
export LD_LIBRARY_PATH="$LIBTORCH_DIR:${LD_LIBRARY_PATH:-}"
[[ -f "$LIBTORCH_DIR/libtorch_cuda.so" ]] && \
    export LD_PRELOAD="$LIBTORCH_DIR/libtorch_cuda.so${LD_PRELOAD:+:$LD_PRELOAD}"
[[ -x "$REPO_ROOT/.venv/bin/python3" ]] && export PATH="$REPO_ROOT/.venv/bin:$PATH"
export RUST_LOG="${RUST_LOG:-info}"
BIN="./target/release/blobmaster-train"
[[ -x "$BIN" ]] || { echo "FATAL: $BIN not built" >&2; exit 1; }

# ─── pairs to evaluate ───
# Default: 4 evals matching yesterday's trajectory check, on the new checkpoints.
# Override by passing pairs as args (CUR=N OPP=M repeated).
PAIRS=()
if [[ $# -gt 0 ]]; then
    cur=""; opp=""
    for kv in "$@"; do
        case "$kv" in
            CUR=*) cur="${kv#CUR=}";;
            OPP=*) opp="${kv#OPP=}";;
            *) echo "ignoring unparseable arg: $kv" >&2;;
        esac
        if [[ -n "$cur" && -n "$opp" ]]; then
            PAIRS+=("$cur:$opp")
            cur=""; opp=""
        fi
    done
else
    PAIRS=("29:0" "25:0" "20:0" "29:15")
fi

# ─── ensure STRENGTH.md exists with header ───
if [[ ! -f "$STRENGTH_MD" ]]; then
    cat > "$STRENGTH_MD" <<'HDR'
# Sweep 2026-04-28 anchor — strength tracker

Persistent append-only log of head-to-head evals run via
`scripts/sweep-2026-04-28-strength-eval.sh`. All rows are 5P7C, 192 games
cap, Wilson-95 early-stop active (so `n` may be 32/64/96/.../192 depending
on how decisive the matchup was). `inconc=true` means the CI didn't clear
the [0.45, 0.55] decision band — treat the win rate as "approximately tied".

| ts | current | opponent | wins/n | wr | lo95 | hi95 | score_diff | bid_a | bid_b | inconc |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
HDR
fi

# ─── run each eval, parse, append ───
echo "=== strength-eval $RUN_TS — ${#PAIRS[@]} pair(s) ==="
echo
: > "$RUN_LOG"

run_one_eval () {
    local cur_iter="$1" opp_iter="$2"
    local cur_label="iter_$(printf %d "$cur_iter")"
    local opp_label="iter_$(printf %d "$opp_iter")"
    local cur_onnx="$CKPT_DIR/iter_$(printf %06d "$cur_iter")/model.onnx"
    local opp_onnx="$CKPT_DIR/iter_$(printf %06d "$opp_iter")/model.onnx"

    if [[ ! -f "$cur_onnx" ]]; then
        echo "  SKIP $cur_label vs $opp_label — $cur_onnx missing" >&2
        return
    fi
    if [[ ! -f "$opp_onnx" ]]; then
        echo "  SKIP $cur_label vs $opp_label — $opp_onnx missing" >&2
        return
    fi

    echo "[$(date +%H:%M:%S)] $cur_label vs $opp_label …"
    local started=$EPOCHSECONDS
    {
        echo
        echo "================================================================"
        echo "=== $cur_label vs $opp_label ==="
        echo "================================================================"
        "$BIN" evaluate \
            --model-a "$cur_onnx" \
            --model-b "$opp_onnx" \
            --num-games 192 \
            --num-players 5 \
            --cards-dealt 7 \
            --config "$CFG" 2>&1
    } >> "$RUN_LOG"
    local elapsed=$((EPOCHSECONDS - started))

    # Parse the latest "evaluate — result" line for this pair from the log
    # and append a row to STRENGTH.md.
    python3 - "$cur_label" "$opp_label" "$RUN_TS" "$STRENGTH_MD" "$RUN_LOG" <<'PY'
import re, sys
cur, opp, ts, md_path, log_path = sys.argv[1:6]
with open(log_path) as f:
    text = f.read()
text = re.sub(r'\x1b\[[0-9;]*m', '', text)
# Find the LAST "evaluate — result" block in the log (this run's pair).
matches = re.findall(r'evaluate — result\s+([^\n]+)', text)
if not matches:
    print(f"  ! no result line for {cur} vs {opp} — eval may have failed", file=sys.stderr)
    sys.exit(0)
last = matches[-1]
fields = dict(re.findall(r'(\w+)=([\w\.\-+e]+)', last))
def f(k, fmt=".4f"):
    v = fields.get(k, "")
    try: return format(float(v), fmt)
    except: return "—"
row = (f"| {ts} | {cur} | {opp} | "
       f"{fields.get('wins_a','?')}/{fields.get('games_played','?')} | "
       f"{f('win_rate', '.3f')} | {f('win_rate_lower95', '.3f')} | "
       f"{f('win_rate_upper95', '.3f')} | {f('score_differential', '.2f')} | "
       f"{f('bid_success_a', '.3f')} | {f('bid_success_b', '.3f')} | "
       f"{fields.get('inconclusive','?')} |\n")
with open(md_path, "a") as f:
    f.write(row)
print(f"  → wr={f('win_rate', '.3f')} lo95={f('win_rate_lower95', '.3f')} "
      f"hi95={f('win_rate_upper95', '.3f')} inconc={fields.get('inconclusive','?')}",
      file=sys.stderr)
PY
    echo "  ($((elapsed))s)"
}

for pair in "${PAIRS[@]}"; do
    cur="${pair%%:*}"
    opp="${pair##*:}"
    run_one_eval "$cur" "$opp"
done

echo
echo "=== STRENGTH.md (full table) ==="
cat "$STRENGTH_MD"
echo
echo "Per-eval stdout:  $RUN_LOG"
echo "Persistent table: $STRENGTH_MD"
