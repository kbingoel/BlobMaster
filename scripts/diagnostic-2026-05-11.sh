#!/usr/bin/env bash
# Diagnostic 2026-05-11 — does iter_145 of run-2026-05-06 beat iter_0
# more decisively with 2× MCTS sims than with the training-time sim
# budget? If yes, the plateau is MCTS-budget-limited and a bigger MCTS
# at training time should restart improvement. If no, the net is at its
# representational ceiling and the next move is capacity (D_MODEL,
# layers, FFN_DIM) — see `scripts/diagnostic-2026-05-11.md` for that.
#
# Four head-to-head evals, 192-game cap each, Wilson-95 early-stop on:
#
#   1. iter_145 vs iter_0  @ 5×100 sims  (where we actually are vs init)
#   2. iter_145 vs iter_0  @ 5×200 sims  (does 2× MCTS rescue iter_145)
#   3. iter_80  vs iter_0  @ 5×100 sims  (sanity-replay the promote eval —
#                                          ran on 96 games early-stopped;
#                                          this is the full-192 version)
#   4. iter_80  vs iter_0  @ 5×200 sims  (does 2× MCTS lift the OLD peak)
#
# Reading the matrix:
#   - if (1)≈(3) and both ≈ 0.55–0.65, training added ~nothing past iter 80
#   - if (2) > (1) by >5pp AND (4) > (3) by >5pp, MCTS budget is the lever
#   - if (2) ≈ (1) AND (4) ≈ (3), bigger MCTS isn't the answer — go capacity
#   - if (2) > (1) but (4) ≈ (3), iter_145 has latent skill that low-MCTS
#     self-play wasn't exposing — also a training-pipeline finding
#
# Walltime estimate: ~1–2h total. Evals can early-stop after 32–96 games
# if the CI clears the [0.45, 0.55] band fast.

set -u
set -o pipefail

REPO_ROOT="/home/kbuntu/Documents/Github/BlobMaster"
cd "$REPO_ROOT"

CKPT_DIR="checkpoints/run-2026-05-06"
LOG_DIR="logs/diagnostic-2026-05-11"
mkdir -p "$LOG_DIR"
RESULTS_MD="$LOG_DIR/RESULTS.md"
RUN_TS="$(date +%Y%m%dT%H%M%S)"
RUN_LOG="$LOG_DIR/eval-${RUN_TS}.log"

CFG_1X="blob-train/diagnostic-2026-05-11/mcts-1x.toml"
CFG_2X="blob-train/diagnostic-2026-05-11/mcts-2x.toml"

# ── runtime env (libtorch LD_LIBRARY_PATH / LD_PRELOAD; see memory) ──
LIBTORCH_DIR="$(find target/release/build -maxdepth 6 -type d -name lib -path '*/libtorch/libtorch/lib' | head -n1)"
[[ -n "$LIBTORCH_DIR" ]] || { echo "FATAL: libtorch dir not found — build target/release first" >&2; exit 1; }
export LD_LIBRARY_PATH="$LIBTORCH_DIR:${LD_LIBRARY_PATH:-}"
[[ -f "$LIBTORCH_DIR/libtorch_cuda.so" ]] && \
    export LD_PRELOAD="$LIBTORCH_DIR/libtorch_cuda.so${LD_PRELOAD:+:$LD_PRELOAD}"
[[ -x "$REPO_ROOT/.venv/bin/python3" ]] && export PATH="$REPO_ROOT/.venv/bin:$PATH"
export RUST_LOG="${RUST_LOG:-info}"
BIN="./target/release/blobmaster-train"
[[ -x "$BIN" ]] || { echo "FATAL: $BIN not built — run cargo build --release -p blob-train" >&2; exit 1; }

# ── pre-flight: confirm all checkpoints exist ──
for iter in 0 80 145; do
    onnx="$CKPT_DIR/iter_$(printf %06d "$iter")/model.onnx"
    [[ -f "$onnx" ]] || { echo "FATAL: missing $onnx" >&2; exit 1; }
done

# ── ensure RESULTS.md exists with header ──
if [[ ! -f "$RESULTS_MD" ]]; then
    cat > "$RESULTS_MD" <<'HDR'
# Diagnostic 2026-05-11 — MCTS-budget vs capacity

Each row: one 192-game head-to-head, Wilson-95 early-stop active (so
`n` may be 32/64/96/.../192). Same `base_seed` across the 1× and 2×
arms for a given current↔opponent pair, so identical games up to the
point where the trees diverge.

| ts | current | opponent | sims | wins/n | wr | lo95 | hi95 | score_diff | bid_a | bid_b | inconc | elapsed_s |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
HDR
fi

: > "$RUN_LOG"
echo "=== diagnostic-2026-05-11 — start $RUN_TS ==="
echo "  log:     $RUN_LOG"
echo "  results: $RESULTS_MD"
echo

# Identical seed across arms so games are paired (same deals up to MCTS
# divergence). Different seed per current-iter pair so the two
# matchups don't share trajectories.
SEED_145=305110145
SEED_80=305110080

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
    print(f"  ! no result line for {cur} vs {opp} @ {sims} — eval may have failed", file=sys.stderr)
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

run_one_eval 145 0 "5x100" "$CFG_1X" "$SEED_145"
run_one_eval 145 0 "5x200" "$CFG_2X" "$SEED_145"
run_one_eval  80 0 "5x100" "$CFG_1X" "$SEED_80"
run_one_eval  80 0 "5x200" "$CFG_2X" "$SEED_80"

echo
echo "=== $RESULTS_MD ==="
cat "$RESULTS_MD"
echo
echo "Per-eval stdout: $RUN_LOG"
