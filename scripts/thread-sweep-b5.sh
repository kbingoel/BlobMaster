#!/usr/bin/env bash
# Thread-count sweep for self-play after Session 7.4c stage-1 (cross-
# determinization batching). Runs the same shape as `thread-sweep.sh`
# (5 games per thread, 5P7C, iter_000014 model) but at B=5 lockstep
# batched ONNX inference instead of B=1.
#
# Walks T over the candidate set and writes results.csv + a markdown
# fragment that gets appended to self-play-profile.md.

set -euo pipefail

REPO_ROOT="/home/kbuntu/Documents/Github/BlobMaster"
cd "$REPO_ROOT"

MODEL="checkpoints/7.3c-run/iter_000014/model.onnx"
GAMES_PER_THREAD=5
NUM_PLAYERS=5
CARDS_DEALT=7

# Plan expects optimum ~8T at B=5 (vs 16T at B=1). Sweep brackets that.
THREAD_LIST=(4 6 8 10 12 14 16)

LIBTORCH_DIR="$(find target/release/build -maxdepth 6 -type d -name lib -path '*/libtorch/libtorch/lib' | head -n1)"
if [[ -z "$LIBTORCH_DIR" ]]; then
    echo "ERROR: libtorch dir not found under target/release/build" >&2
    exit 1
fi
export LD_LIBRARY_PATH="$LIBTORCH_DIR:${LD_LIBRARY_PATH:-}"

LOG_DIR="logs/thread-sweep-b5-2026-04-26"
mkdir -p "$LOG_DIR"
RESULTS_CSV="$LOG_DIR/results.csv"
STATUS_FILE="$LOG_DIR/status.txt"

if [[ ! -s "$RESULTS_CSV" ]]; then
    echo "threads,wall_s,per_game_s,per_decision_ms,onnx_avg_us,total_games,timestamp" > "$RESULTS_CSV"
fi

echo "started: $(date -Iseconds)" > "$STATUS_FILE"

run_one() {
    local T=$1
    local total_games=$((GAMES_PER_THREAD * T))
    local logfile="$LOG_DIR/T$(printf %02d "$T").log"
    local ts
    ts=$(date -Iseconds)
    echo "============================================="
    echo "Running T=$T  (5 × $T = $total_games games)  $ts"
    echo "============================================="
    echo "current=T$T started=$ts" >> "$STATUS_FILE"

    set +e
    RUST_LOG=info ./target/release/blobmaster-train profile \
        --model "$MODEL" \
        --games-per-thread "$GAMES_PER_THREAD" \
        --num-threads "$T" \
        --num-players "$NUM_PLAYERS" \
        --cards-dealt "$CARDS_DEALT" \
        > "$logfile" 2>&1
    local rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
        echo "ERROR: T=$T run failed (rc=$rc); see $logfile" >&2
        echo "T=$T FAILED rc=$rc at $(date -Iseconds)" >> "$STATUS_FILE"
        return 1
    fi

    local wall per_game_ms per_game per_decision onnx_us
    wall=$(awk -F: '/^wall clock \(s\)/ {gsub(/ /,"",$2); print $2; exit}' "$logfile")
    per_game_ms=$(awk -F: '/^avg per-game wall/ {gsub(/[ ms]/,"",$2); print $2; exit}' "$logfile")
    per_game=$(awk -v ms="$per_game_ms" 'BEGIN{printf "%.6f", ms/1000.0}')
    per_decision=$(awk -F: '/^avg per-decision/ {split($2,a," "); print a[1]; exit}' "$logfile")
    onnx_us=$(awk '/^onnx_inference/ {print $4; exit}' "$logfile")

    if [[ -z "$wall" || -z "$per_game" ]]; then
        echo "ERROR: failed to parse output for T=$T; see $logfile" >&2
        echo "T=$T PARSE_FAIL at $(date -Iseconds)" >> "$STATUS_FILE"
        return 1
    fi

    echo "$T,$wall,$per_game,$per_decision,$onnx_us,$total_games,$ts" >> "$RESULTS_CSV"
    echo "  -> wall=${wall}s  per_game=${per_game}s  per_decision=${per_decision}ms  onnx_avg=${onnx_us}us"
}

for T in "${THREAD_LIST[@]}"; do
    run_one "$T" || echo "  (continuing despite failure)"
done

# ---- results table ----------------------------------------------------------

# Pull the prior B=1 baseline at T=16 (7.3c) for the speedup column.
B1_BASELINE_PER_GAME=$(awk -F, '$1==16 && NR>1 {print $3; exit}' \
    "logs/thread-sweep-2026-04-24/results.csv" 2>/dev/null || true)

TABLE_FILE="$LOG_DIR/results-table.md"
{
    echo ""
    echo "## Follow-ups (2026-04-26) — thread-count sweep at B=5 (Session 7.4c stage-1)"
    echo ""
    echo "After landing the cross-determinization batching driver in"
    echo "[blob-engine/src/mcts.rs](blob-engine/src/mcts.rs), re-swept thread counts at B=5 lockstep batched ONNX"
    echo "inference. Same workload as [logs/thread-sweep-2026-04-24/](logs/thread-sweep-2026-04-24/): 5 games per thread,"
    echo "fixed 5P7C, \`iter_000014/model.onnx\`, MCTS at flat 5×100. Speedup column compares"
    echo "per-game wall against the B=1 T=16 baseline (${B1_BASELINE_PER_GAME:-?} s/game)."
    echo ""
    echo "Script: [scripts/thread-sweep-b5.sh](scripts/thread-sweep-b5.sh), raw logs: [$LOG_DIR/]($LOG_DIR/)."
    echo ""
    echo "| threads | total games | wall (s) | per-game wall (s) | per-decision (ms) | onnx_inference avg (µs) | speedup vs B=1 16T |"
    echo "|--------:|------------:|---------:|------------------:|------------------:|------------------------:|-------------------:|"
    awk -F, -v base="$B1_BASELINE_PER_GAME" 'NR>1' "$RESULTS_CSV" \
      | sort -t, -k1,1n \
      | awk -F, -v base="$B1_BASELINE_PER_GAME" '{
            t=$1; wall=$2; pg=$3; pd=$4; onnx=$5; total=$6;
            sp = (base+0 > 0 && pg+0 > 0) ? (base+0)/(pg+0) : 0;
            printf "| %d | %d | %.1f | %.3f | %.1f | %.1f | %.3f× |\n", t, total, wall, pg, pd, onnx, sp
        }'
    echo ""
} > "$TABLE_FILE"

cat "$TABLE_FILE" >> "$REPO_ROOT/self-play-profile.md"

echo ""
echo "==========================================="
echo "B=5 sweep complete: $(date -Iseconds)"
echo "results CSV   : $RESULTS_CSV"
echo "table fragment: $TABLE_FILE"
echo "appended to   : $REPO_ROOT/self-play-profile.md"
echo "==========================================="
echo "completed: $(date -Iseconds)" >> "$STATUS_FILE"
