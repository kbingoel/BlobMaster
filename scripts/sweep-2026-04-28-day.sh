#!/usr/bin/env bash
# Session 7.4d temperature-schedule sweep — daytime 12h batch.
#
# Sequentially trains four 16-iter (iters 0..15, in-loop evals at 5/10/15)
# mixed-player runs that differ only in their `[mcts.temperature_schedule]`
# block:
#
#   anchor — no schedule (constant τ=1.0 for all decisions; control)
#   A      — hard step, switch_at=15, late=0.1 (canonical AlphaZero-style)
#   B      — hard step, switch_at=50, late=0.1 (later switch, longer
#            exploratory phase covering bidding + early plays)
#   C      — hard step, switch_at=15, late=0.0 (full greedy late; sharper
#            policy-target one-hot signal)
#
# Per-arm wall-clock budget at 9 min/iter × 16 iters + 3 evals ≈ 3 h.
# Four arms ≈ 12 h. Each arm is self-contained: writes its own
# checkpoint dir + strength.csv, so a crash in one arm doesn't block
# the rest. After all arms complete (or fail), this script appends a
# side-by-side iter-5/10/15 win-rate table to logs/sweep-2026-04-28/SUMMARY.md
# so the overnight resume can pick the winner.
#
# AGENTS.md off-by-one trap (read before changing total_iterations):
#   `total_iterations` is a count, not an absolute target. To get an
#   in-loop eval row at iter K, set `total_iterations = K + 1`. With
#   K=15 → total_iterations=16, saved-on-disk after the run is
#   iter_000015 (the rolling latest), and that checkpoint is what the
#   overnight resume script feeds into `--resume`.

set -u
set -o pipefail

REPO_ROOT="/home/kbuntu/Documents/Github/BlobMaster"
cd "$REPO_ROOT"

# ─── runtime env (matches AGENTS.md "Canonical launch template") ───
LIBTORCH_DIR="$(find target/release/build -maxdepth 6 -type d -name lib -path '*/libtorch/libtorch/lib' | head -n1)"
if [[ -z "$LIBTORCH_DIR" ]]; then
    echo "FATAL: libtorch dir not found; run 'cargo build --release -p blob-train' first" >&2
    exit 1
fi
export LD_LIBRARY_PATH="$LIBTORCH_DIR:${LD_LIBRARY_PATH:-}"
if [[ -f "$LIBTORCH_DIR/libtorch_cuda.so" ]]; then
    export LD_PRELOAD="$LIBTORCH_DIR/libtorch_cuda.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi
if [[ -x "$REPO_ROOT/.venv/bin/python3" ]]; then
    export PATH="$REPO_ROOT/.venv/bin:$PATH"
fi
export RUST_LOG="${RUST_LOG:-info}"

BIN="./target/release/blobmaster-train"
if [[ ! -x "$BIN" ]]; then
    echo "FATAL: $BIN not built; run 'cargo build --release -p blob-train' first" >&2
    exit 1
fi

# ─── output layout ───
TOML_DIR="blob-train/sweep-2026-04-28"
LOG_DIR="logs/sweep-2026-04-28"
CKPT_PARENT="checkpoints"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/SUMMARY.md"
STATUS="$LOG_DIR/status.txt"

# ─── arms (label : toml : ckpt-dir-tail) ───
ARMS=(
    "anchor:anchor.toml:sweep-2026-04-28-anchor"
    "A:A-switch15-late01.toml:sweep-2026-04-28-A"
    "B:B-switch50-late01.toml:sweep-2026-04-28-B"
    "C:C-switch15-late00.toml:sweep-2026-04-28-C"
)

run_one() {
    local label="$1" toml="$2" ckpt="$3"
    local toml_path="$TOML_DIR/$toml"
    local log_file="$LOG_DIR/${label}.log"

    echo "[$(date -Iseconds)] arm=$label START toml=$toml_path ckpt=$ckpt" >> "$STATUS"
    echo "=== arm $label — config $toml_path → $CKPT_PARENT/$ckpt ===" | tee -a "$SUMMARY"

    local started=$EPOCHSECONDS
    if "$BIN" train --config "$toml_path" >"$log_file" 2>&1; then
        local elapsed=$((EPOCHSECONDS - started))
        echo "[$(date -Iseconds)] arm=$label OK ${elapsed}s" >> "$STATUS"
        echo "    ok   ($((elapsed / 60)) min) — log: $log_file" >> "$SUMMARY"
    else
        local rc=$?
        local elapsed=$((EPOCHSECONDS - started))
        echo "[$(date -Iseconds)] arm=$label FAIL rc=$rc ${elapsed}s" >> "$STATUS"
        echo "    FAIL rc=$rc ($((elapsed / 60)) min) — log: $log_file" >> "$SUMMARY"
    fi
}

# ─── kick off ───
{
    echo "# Sweep 2026-04-28 — temperature-schedule daytime batch"
    echo
    echo "Started: $(date -Iseconds)"
    echo
} > "$SUMMARY"

for arm in "${ARMS[@]}"; do
    IFS=':' read -r label toml ckpt <<< "$arm"
    run_one "$label" "$toml" "$ckpt"
done

# ─── side-by-side eval summary ───
{
    echo
    echo "## Eval rows (iter / win_rate / lower95 / upper95 / bid_success_current)"
    echo
    for arm in "${ARMS[@]}"; do
        IFS=':' read -r label _toml ckpt <<< "$arm"
        local_strength="$CKPT_PARENT/$ckpt/strength.csv"
        echo "### arm $label — $local_strength"
        if [[ -f "$local_strength" ]]; then
            # Header + iter 5/10/15 rows. strength.csv columns:
            # iteration,opponent,win_rate,win_rate_lower95,win_rate_upper95,
            # score_differential,bid_success_rate_current,...
            awk -F, '
                NR==1 { print "    " $1 "  " $2 "  " $3 "  " $4 "  " $5 "  " $7 }
                NR>1 && ($1=="5" || $1=="10" || $1=="15") {
                    print "    " $1 "    " $2 "  " $3 "  " $4 "  " $5 "  " $7
                }
            ' "$local_strength"
        else
            echo "    (no strength.csv — arm probably failed)"
        fi
        echo
    done
    echo
    echo "## Picking the winner"
    echo
    echo "Pick the arm whose iter-15 \`win_rate_lower95\` is highest. Tiebreak"
    echo "on bid_success_rate_current. If two arms are within ±0.03 of each"
    echo "other on iter-15 lower95 (Wilson noise floor at 192 games), prefer"
    echo "the simpler schedule (anchor > A > C > B)."
    echo
    echo "Then resume the winner overnight:"
    echo
    echo "    bash scripts/sweep-2026-04-28-resume.sh <arm> [additional_iters=40]"
    echo
    echo "e.g. \`bash scripts/sweep-2026-04-28-resume.sh A\` resumes from"
    echo "\`checkpoints/sweep-2026-04-28-A/iter_000015\` and runs 40 more iters"
    echo "(reaching iter_000055), with in-loop evals at iter 20/25/.../55."
    echo
    echo "Finished: $(date -Iseconds)"
} >> "$SUMMARY"

echo
echo "=== sweep complete — see $SUMMARY ==="
cat "$SUMMARY"
