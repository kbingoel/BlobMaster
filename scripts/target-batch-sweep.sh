#!/usr/bin/env bash
# Session 7.4c stage-2 target_batch sweep: pin T=32 (B=5 stage-1 optimum),
# vary target_batch ∈ {5, 8, 12, 16}, plus a T=16 / target_batch=8 row to
# re-confirm SMT direction at stage 2. Same workload shape as the prior
# B=5 sweep: 5 games per thread, fixed 5P7C, iter_000014 model.
#
# Each config gets its own `cfg-tb<N>.toml` (already generated alongside
# this script under logs/target-batch-sweep-2026-04-27/) so MctsConfig
# carries `target_batch` through the regular `--config` plumbing.

set -euo pipefail

REPO_ROOT="/home/kbuntu/Documents/Github/BlobMaster"
cd "$REPO_ROOT"

MODEL="checkpoints/7.3c-run/iter_000014/model.onnx"
GAMES_PER_THREAD=5
NUM_PLAYERS=5
CARDS_DEALT=7

LIBTORCH_DIR="$(find target/release/build -maxdepth 6 -type d -name lib -path '*/libtorch/libtorch/lib' | head -n1)"
if [[ -z "$LIBTORCH_DIR" ]]; then
    echo "ERROR: libtorch dir not found under target/release/build" >&2
    exit 1
fi
export LD_LIBRARY_PATH="$LIBTORCH_DIR:${LD_LIBRARY_PATH:-}"

LOG_DIR="logs/target-batch-sweep-2026-04-27"
mkdir -p "$LOG_DIR"
RESULTS_CSV="$LOG_DIR/results.csv"
STATUS_FILE="$LOG_DIR/status.txt"

if [[ ! -s "$RESULTS_CSV" ]]; then
    echo "threads,target_batch,wall_s,per_game_s,per_decision_ms,onnx_avg_us,total_games,timestamp" > "$RESULTS_CSV"
fi
echo "started: $(date -Iseconds)" > "$STATUS_FILE"

run_one() {
    local T=$1
    local TB=$2
    local cfg="$LOG_DIR/cfg-tb${TB}.toml"
    local total_games=$((GAMES_PER_THREAD * T))
    local logfile="$LOG_DIR/T$(printf %02d "$T")-tb$(printf %02d "$TB").log"
    local ts
    ts=$(date -Iseconds)
    echo "============================================="
    echo "Running T=$T tb=$TB ($total_games games)  $ts"
    echo "============================================="
    echo "current=T${T}-tb${TB} started=$ts" >> "$STATUS_FILE"

    set +e
    RUST_LOG=info ./target/release/blobmaster-train profile \
        --model "$MODEL" \
        --config "$cfg" \
        --games-per-thread "$GAMES_PER_THREAD" \
        --num-threads "$T" \
        --num-players "$NUM_PLAYERS" \
        --cards-dealt "$CARDS_DEALT" \
        > "$logfile" 2>&1
    local rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
        echo "ERROR: T=$T tb=$TB run failed (rc=$rc); see $logfile" >&2
        echo "T=${T} tb=${TB} FAILED rc=$rc at $(date -Iseconds)" >> "$STATUS_FILE"
        return 1
    fi

    local wall per_game_ms per_game per_decision onnx_us
    wall=$(awk -F: '/^wall clock \(s\)/ {gsub(/ /,"",$2); print $2; exit}' "$logfile")
    per_game_ms=$(awk -F: '/^avg per-game wall/ {gsub(/[ ms]/,"",$2); print $2; exit}' "$logfile")
    per_game=$(awk -v ms="$per_game_ms" 'BEGIN{printf "%.6f", ms/1000.0}')
    per_decision=$(awk -F: '/^avg per-decision/ {split($2,a," "); print a[1]; exit}' "$logfile")
    onnx_us=$(awk '/^onnx_inference/ {print $4; exit}' "$logfile")

    if [[ -z "$wall" || -z "$per_game" ]]; then
        echo "ERROR: failed to parse output for T=$T tb=$TB; see $logfile" >&2
        echo "T=${T} tb=${TB} PARSE_FAIL at $(date -Iseconds)" >> "$STATUS_FILE"
        return 1
    fi

    echo "$T,$TB,$wall,$per_game,$per_decision,$onnx_us,$total_games,$ts" >> "$RESULTS_CSV"
    echo "  -> wall=${wall}s per_game=${per_game}s per_decision=${per_decision}ms onnx_avg=${onnx_us}us"
}

# T=32 sweep across target_batch.
for TB in 5 8 12 16; do
    run_one 32 "$TB" || echo "  (continuing despite failure)"
done

# T=16 SMT cross-check at the default target_batch=8.
run_one 16 8 || echo "  (continuing despite failure)"

echo ""
echo "==========================================="
echo "stage-2 sweep complete: $(date -Iseconds)"
echo "results CSV   : $RESULTS_CSV"
echo "==========================================="
echo "completed: $(date -Iseconds)" >> "$STATUS_FILE"
