#!/usr/bin/env bash
# Overnight validation run for sweep-2026-04-28-anchor.
#
# Resumes from iter_15 (the last good Run-1 checkpoint), runs forward
# under the 2026-04-29 fixes:
#   - replay buffer persisted to disk every iter (buffer.bin)
#   - cold-buffer epoch cap (2 epochs while buffer < buffer_capacity post-resume)
#   - absolute-target cosine LR schedule (span = 101)
# Auto-stops after the configured iter completes a clean checkpoint save,
# so no manual Ctrl-C is needed in the morning. The cosine span is fixed
# at 101 so tonight's iter K sits at the *same* LR it will sit at in the
# planned full week-run, and tomorrow can `--resume` directly without
# reshifting the schedule.
#
# Usage: bash scripts/sweep-2026-04-29-overnight.sh [STOP_AT_ITER]
#   STOP_AT_ITER  Last iter to keep (default 30; iter 31 may also land if
#                 the watcher's 60s poll misses the boundary — fine).

set -u
set -o pipefail

REPO_ROOT="/home/kbuntu/Documents/Github/BlobMaster"
cd "$REPO_ROOT"

CKPT_DIR="checkpoints/sweep-2026-04-28-anchor"
STOP_AT_ITER="${1:-30}"
STOP_FILE="$CKPT_DIR/STOP"
LOG_DIR="logs/sweep-2026-04-28"
mkdir -p "$LOG_DIR"
TONIGHT_LOG="$LOG_DIR/overnight-2026-04-29.log"
WATCHER_LOG="$LOG_DIR/overnight-2026-04-29-watcher.log"
PIDFILE="$LOG_DIR/overnight-2026-04-29.pids"

# ─── pre-flight checks ───
LATEST_ITER=$(ls -1d "$CKPT_DIR"/iter_* 2>/dev/null \
    | awk -F/iter_ '{print $2}' | sort -n | tail -n1 | sed 's/^0*//')
LATEST_ITER="${LATEST_ITER:-MISSING}"
if [[ "$LATEST_ITER" != "15" ]]; then
    echo "FATAL: expected latest on-disk iter to be 15, got '$LATEST_ITER'." >&2
    echo "       Move any later iters into a sibling archive dir before launching." >&2
    exit 1
fi
if (( STOP_AT_ITER <= LATEST_ITER )); then
    echo "FATAL: STOP_AT_ITER ($STOP_AT_ITER) must be > latest iter ($LATEST_ITER)" >&2
    exit 2
fi
if [[ ! -x ./target/release/blobmaster-train ]]; then
    echo "FATAL: ./target/release/blobmaster-train not built" >&2
    exit 3
fi
if pgrep -f "target/release/blobmaster-train train" > /dev/null; then
    echo "FATAL: another training process is already running:" >&2
    pgrep -af "target/release/blobmaster-train train" >&2
    exit 4
fi

# Ensure no stale STOP file from a prior aborted run.
rm -f "$STOP_FILE"

# ADD_ITERS picks the cosine span: target = LATEST + 1 + ADD = 101.
# Loop will run until STOP file fires or it hits iter 100 — STOP fires far first.
ADD_ITERS=$((101 - LATEST_ITER - 1))

echo "=== overnight validation ==="
echo "  resume from     : iter_$(printf %06d "$LATEST_ITER")"
echo "  cosine span     : total_iterations = 101 (full-run-aligned)"
echo "  auto-stop after : iter $STOP_AT_ITER completes a full checkpoint"
echo "  training log    : $TONIGHT_LOG"
echo "  watcher log     : $WATCHER_LOG"
echo "  STOP file       : $STOP_FILE  (touch manually to abort early)"
echo "  started         : $(date -Iseconds)"
echo

# ─── launch trainer ───
nohup bash scripts/sweep-2026-04-28-resume.sh anchor "$ADD_ITERS" \
    > "$TONIGHT_LOG" 2>&1 &
TRAIN_PID=$!
disown $TRAIN_PID

# ─── launch watcher ───
# Polls every 60s for STOP_AT_ITER's `model.ot` (written before metrics +
# ONNX export). When seen, touches STOP. The trainer finishes the in-flight
# iter cleanly (export ONNX, append metrics if not yet, save buffer.bin),
# then exits at the *next* iter-loop boundary because the STOP file is
# checked at the top of each iter (main.rs:343). Worst case one extra iter
# sneaks in (race between watcher poll and iter K+1 starting) — that's
# fine; iter 31 is just as useful as iter 30.
(
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        target_dir="$CKPT_DIR/iter_$(printf %06d "$STOP_AT_ITER")"
        if [[ -f "$target_dir/model.ot" ]]; then
            echo "$(date -Iseconds) iter_$STOP_AT_ITER model.ot present — touching $STOP_FILE"
            touch "$STOP_FILE"
            break
        fi
        sleep 60
    done
    # Wait for trainer to exit cleanly so partial writes don't strand state.
    while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 30; done
    echo "$(date -Iseconds) trainer exited"
) > "$WATCHER_LOG" 2>&1 &
WATCHER_PID=$!
disown $WATCHER_PID

echo "$TRAIN_PID train" > "$PIDFILE"
echo "$WATCHER_PID watcher" >> "$PIDFILE"

echo "  train PID       : $TRAIN_PID"
echo "  watcher PID     : $WATCHER_PID"
echo
echo "Run will auto-stop after iter $STOP_AT_ITER. PID file: $PIDFILE"
echo "Tail with: tail -f $TONIGHT_LOG | grep -E 'iteration complete|wall_clock|eval'"
