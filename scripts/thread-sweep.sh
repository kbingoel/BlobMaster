#!/usr/bin/env bash
# Thread-count sweep for self-play after the OnnxEvaluator-per-thread reuse fix.
#
# Algorithm: starting from a baseline (T=16), sweep DOWN then UP one step
# at a time. Whenever a step regresses against the best-so-far in that
# direction, take ONE additional validation step. Stop the direction if
# the validation step also regresses; otherwise continue.
#
# Each run uses 5 games per thread. Runs are sequential (only one config
# active at a time); the machine should be otherwise idle for clean numbers.
# Results land in $LOG_DIR/results.csv and are appended to self-play-profile.md
# in a new follow-up section when the sweep completes.

set -euo pipefail

REPO_ROOT="/home/kbuntu/Documents/Github/BlobMaster"
cd "$REPO_ROOT"

MODEL="checkpoints/7.3c-run/iter_000014/model.onnx"
GAMES_PER_THREAD=5
NUM_PLAYERS=5
CARDS_DEALT=7
MIN_T=1
MAX_T=32
BASELINE_T=16

LIBTORCH_DIR="$(find target/release/build -maxdepth 6 -type d -name lib -path '*/libtorch/libtorch/lib' | head -n1)"
if [[ -z "$LIBTORCH_DIR" ]]; then
    echo "ERROR: libtorch dir not found under target/release/build" >&2
    exit 1
fi
export LD_LIBRARY_PATH="$LIBTORCH_DIR:${LD_LIBRARY_PATH:-}"

LOG_DIR="logs/thread-sweep-2026-04-24"
mkdir -p "$LOG_DIR"
RESULTS_CSV="$LOG_DIR/results.csv"
STATUS_FILE="$LOG_DIR/status.txt"
ORDER_FILE="$LOG_DIR/run-order.txt"

if [[ ! -s "$RESULTS_CSV" ]]; then
    echo "threads,wall_s,per_game_s,per_decision_ms,onnx_avg_us,total_games,timestamp" > "$RESULTS_CSV"
fi
: > "$ORDER_FILE"

echo "started: $(date -Iseconds)" > "$STATUS_FILE"

# ---- helpers ----------------------------------------------------------------

declare -A SEEN

prior_per_game() {
    awk -F, -v t="$1" 'NR>1 && $1==t {print $3; exit}' "$RESULTS_CSV"
}

run_one() {
    local T=$1
    local prior
    prior=$(prior_per_game "$T")
    if [[ -n "$prior" ]]; then
        SEEN[$T]=1
        echo "$T (cached prior result: ${prior}s/game)" >> "$ORDER_FILE"
        echo "[skip] T=$T already in results.csv (per_game=${prior}s)"
        return 0
    fi

    SEEN[$T]=1
    local total_games=$((GAMES_PER_THREAD * T))
    local logfile="$LOG_DIR/T$(printf %02d "$T").log"
    local ts
    ts=$(date -Iseconds)
    echo "============================================="
    echo "Running T=$T  (5 × $T = $total_games games)  $ts"
    echo "============================================="
    echo "current=T$T started=$ts" > "$STATUS_FILE.tmp" && mv "$STATUS_FILE.tmp" "$STATUS_FILE"

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
    echo "$T (per_game=${per_game}s)" >> "$ORDER_FILE"
    echo "  -> wall=${wall}s  per_game=${per_game}s  per_decision=${per_decision}ms  onnx_avg=${onnx_us}us"
}

faster() {  # faster $a $b -> success if a < b
    awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0 < b+0)}'
}

sweep_direction() {
    # $1 = step (-1 for down, +1 for up)
    local step=$1
    local label
    if [[ $step -lt 0 ]]; then label="DOWN"; else label="UP"; fi
    echo ""
    echo "### Sweeping $label from T=$BASELINE_T ###"

    local best_T=$BASELINE_T
    local best_t
    best_t=$(prior_per_game "$BASELINE_T")
    local T=$((BASELINE_T + step))

    while [[ $T -ge $MIN_T && $T -le $MAX_T ]]; do
        run_one "$T" || { echo "stopping $label direction due to run failure"; break; }
        local t
        t=$(prior_per_game "$T")
        if faster "$t" "$best_t"; then
            best_T=$T
            best_t=$t
            T=$((T + step))
            continue
        fi

        local Tv=$((T + step))
        if [[ $Tv -lt $MIN_T || $Tv -gt $MAX_T ]]; then
            echo "  $label: T=$T regressed; no validation step possible (out of range). Stopping."
            break
        fi
        echo "  $label: T=$T regressed (${t}s vs best ${best_t}s); validating with T=$Tv"
        run_one "$Tv" || { echo "stopping $label direction due to run failure"; break; }
        local tv
        tv=$(prior_per_game "$Tv")
        if faster "$tv" "$best_t"; then
            best_T=$Tv
            best_t=$tv
            T=$((Tv + step))
            continue
        fi
        echo "  $label: validation T=$Tv also regressed (${tv}s vs best ${best_t}s). Stopping $label."
        break
    done

    echo "$label best: T=$best_T at ${best_t}s/game"
    echo "${label}_BEST_T=$best_T"  >> "$STATUS_FILE"
    echo "${label}_BEST_T=$best_T ${label}_BEST_S=$best_t"
}

# ---- run --------------------------------------------------------------------

run_one "$BASELINE_T"
sweep_direction -1
sweep_direction 1

# ---- results table ----------------------------------------------------------

# Build the markdown table sorted by thread count.
TABLE_FILE="$LOG_DIR/results-table.md"
{
    echo ""
    echo "## Follow-ups (2026-04-24) — thread-count sweep after evaluator-reuse"
    echo ""
    echo "After landing the per-thread \`OnnxEvaluator\` reuse change ([blob-nn/src/engine.rs](blob-nn/src/engine.rs)),"
    echo "re-baselined T=16 and swept neighbouring thread counts to find the per-game-wall optimum."
    echo "Each row is 5 games per thread (so total games = 5 × T). Same model, MCTS, and game shape as the original profile."
    echo ""
    echo "Algorithm: from T=16, walk one step in each direction; on a regression vs best-so-far, take one validation"
    echo "step further; stop that direction if the validation step also regresses. Script: [scripts/thread-sweep.sh](scripts/thread-sweep.sh),"
    echo "raw logs: [logs/thread-sweep-2026-04-24/](logs/thread-sweep-2026-04-24/)."
    echo ""
    # Re-read baseline 16 per-game time for vs-baseline column
    BASE=$(prior_per_game 16)
    echo "| threads | total games | wall (s) | per-game wall (s) | per-decision (ms) | onnx_inference avg (µs) | vs 16T |"
    echo "|--------:|------------:|---------:|------------------:|------------------:|------------------------:|-------:|"
    awk -F, -v base="$BASE" 'NR>1 {print}' "$RESULTS_CSV" \
      | sort -t, -k1,1n \
      | awk -F, -v base="$BASE" '{
            t=$1; wall=$2; pg=$3; pd=$4; onnx=$5; total=$6;
            ratio = (base+0 > 0) ? (pg+0)/(base+0) : 0;
            printf "| %d | %d | %.1f | %.3f | %.1f | %.1f | %.3f× |\n", t, total, wall, pg, pd, onnx, ratio
        }'
    echo ""
    echo "Run order (sequential, one config at a time):"
    echo ""
    echo '```'
    cat "$ORDER_FILE"
    echo '```'
    echo ""
    if [[ -f "$STATUS_FILE" ]]; then
        DBEST=$(awk -F= '/^DOWN_BEST_T/ {print $2; exit}' "$STATUS_FILE")
        UBEST=$(awk -F= '/^UP_BEST_T/ {print $2; exit}' "$STATUS_FILE")
        DBEST_T=$(prior_per_game "${DBEST:-16}")
        UBEST_T=$(prior_per_game "${UBEST:-16}")
        BASET=$(prior_per_game 16)
        echo "**Direction bests:** DOWN best at T=${DBEST} (${DBEST_T}s/game), UP best at T=${UBEST} (${UBEST_T}s/game), baseline T=16 (${BASET}s/game)."
    fi
} > "$TABLE_FILE"

# Append to the main profile markdown.
cat "$TABLE_FILE" >> "$REPO_ROOT/self-play-profile.md"

echo ""
echo "==========================================="
echo "sweep complete: $(date -Iseconds)"
echo "results CSV : $RESULTS_CSV"
echo "table fragment: $TABLE_FILE"
echo "appended to : $REPO_ROOT/self-play-profile.md"
echo "==========================================="
echo "completed: $(date -Iseconds)" >> "$STATUS_FILE"
