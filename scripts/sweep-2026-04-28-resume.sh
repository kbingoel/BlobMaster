#!/usr/bin/env bash
# Session 7.4d temperature-schedule sweep — overnight resume.
#
# Usage: bash scripts/sweep-2026-04-28-resume.sh <arm> [additional_iters]
#   <arm>             : anchor | A | B | C   (the winner of the daytime batch)
#   additional_iters  : default 40 (≈ 7.3 h at 9 min/iter + 8 evals × ~10 min)
#
# Resumes the chosen arm from its iter_000015 checkpoint and runs
# `additional_iters` more iterations under the same schedule. The arm's
# original TOML is copied to a derived `resume-<arm>.toml` with
# `total_iterations` overridden to the additional count — `total_iterations`
# is a count, not an absolute target (AGENTS.md), so for "40 more iters
# from iter 16 onwards" we pass `total_iterations = 40` not 56.
#
# Eval semantics under --resume (AGENTS.md):
#   `try_resume` sets tl.iteration = K + 1 = 16 and anchor_iter = 16.
#   In-loop evals fire at iter > 16 && iter % eval_interval == 0, so:
#       eval_interval = 5 → iter 20, 25, 30, ..., 55  (8 rows)
#   These evals are vs `iter_000016` (the resume-baseline anchor), NOT
#   vs `iter_000000`. To compare the post-resume model directly to the
#   from-scratch anchor (the eval that matters for trajectory work), run
#   a one-shot evaluate after the resume completes — this script prints
#   the exact command at the end.

set -u
set -o pipefail

REPO_ROOT="/home/kbuntu/Documents/Github/BlobMaster"
cd "$REPO_ROOT"

ARM="${1:-}"
ADD_ITERS="${2:-40}"

case "$ARM" in
    anchor) BASE_TOML="blob-train/sweep-2026-04-28/anchor.toml" ;;
    A)      BASE_TOML="blob-train/sweep-2026-04-28/A-switch15-late01.toml" ;;
    B)      BASE_TOML="blob-train/sweep-2026-04-28/B-switch50-late01.toml" ;;
    C)      BASE_TOML="blob-train/sweep-2026-04-28/C-switch15-late00.toml" ;;
    *)
        echo "Usage: $0 <anchor|A|B|C> [additional_iters]" >&2
        exit 2
        ;;
esac

if [[ ! -f "$BASE_TOML" ]]; then
    echo "FATAL: base TOML not found: $BASE_TOML" >&2
    exit 1
fi

# ─── derive resume TOML with total_iterations overridden ───
RESUME_TOML="blob-train/sweep-2026-04-28/resume-${ARM}.toml"
# In-place sed on a copy: the only field we change is total_iterations on
# the [training] line. The pattern is anchored to the start of line so we
# don't accidentally rewrite a comment that mentions the field.
sed -E "s/^total_iterations[[:space:]]*=[[:space:]]*[0-9]+/total_iterations = ${ADD_ITERS}/" \
    "$BASE_TOML" > "$RESUME_TOML"

# Sanity: confirm the substitution stuck.
if ! grep -qE "^total_iterations = ${ADD_ITERS}$" "$RESUME_TOML"; then
    echo "FATAL: failed to set total_iterations = ${ADD_ITERS} in $RESUME_TOML" >&2
    exit 1
fi

# ─── verify the checkpoint we will resume from exists ───
CKPT_DIR_TAIL="sweep-2026-04-28-${ARM}"
case "$ARM" in
    anchor) CKPT_DIR_TAIL="sweep-2026-04-28-anchor" ;;
esac
CKPT_DIR="checkpoints/${CKPT_DIR_TAIL}"
if [[ ! -d "$CKPT_DIR/iter_000015" ]]; then
    echo "FATAL: ${CKPT_DIR}/iter_000015 not found — did the daytime arm finish?" >&2
    exit 1
fi
if [[ ! -f "$CKPT_DIR/iter_000015/model.onnx" ]]; then
    echo "FATAL: ${CKPT_DIR}/iter_000015/model.onnx missing" >&2
    exit 1
fi

# ─── runtime env (AGENTS.md canonical launch template) ───
LIBTORCH_DIR="$(find target/release/build -maxdepth 6 -type d -name lib -path '*/libtorch/libtorch/lib' | head -n1)"
if [[ -z "$LIBTORCH_DIR" ]]; then
    echo "FATAL: libtorch dir not found" >&2
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
[[ -x "$BIN" ]] || { echo "FATAL: $BIN not built" >&2; exit 1; }

# ─── launch ───
LOG_DIR="logs/sweep-2026-04-28"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/resume-${ARM}.log"

FINAL_ITER=$((15 + ADD_ITERS))   # iter_000015 + ADD_ITERS more = iter_000(15+ADD_ITERS)

echo "=== resuming arm=$ARM from $CKPT_DIR/iter_000015 ==="
echo "    additional iters : $ADD_ITERS"
echo "    final saved iter : iter_$(printf %06d "$FINAL_ITER")"
echo "    in-loop evals at : iter 20, 25, ..., (multiples of 5 up to $FINAL_ITER)"
echo "    log              : $LOG_FILE"
echo "    started          : $(date -Iseconds)"
echo

started=$EPOCHSECONDS
if "$BIN" train --config "$RESUME_TOML" --resume 2>&1 | tee "$LOG_FILE"; then
    elapsed=$((EPOCHSECONDS - started))
    echo
    echo "=== resume complete in $((elapsed / 60)) min ==="
else
    rc=$?
    echo
    echo "=== resume FAILED rc=$rc ==="
    exit "$rc"
fi

# ─── post-resume helper: how to compare the resumed model to the from-scratch anchor ───
cat <<EOF

────────────────────────────────────────────────────────────────────────
Post-resume note (AGENTS.md): in-loop eval rows in
  $CKPT_DIR/strength.csv
after the resume are vs iter_000016 (the resume-baseline anchor), NOT vs
iter_000000. To compare the final model to the from-scratch starting
point — the trajectory metric that matters for 7.5 — run:

  bash scripts/run-train.sh evaluate \\
      --model-a $CKPT_DIR/iter_$(printf %06d "$FINAL_ITER")/model.onnx \\
      --model-b $CKPT_DIR/iter_000000/model.onnx \\
      --num-games 192 --num-players 5 --cards-dealt 7 \\
      --config $RESUME_TOML

(192 games at 5P7C reproduces the in-loop eval cap. Switch num-players /
cards-dealt to whatever you want to spot-check; the mixed distribution
isn't directly evaluatable from a single (num_players, start_cards) pair.)
────────────────────────────────────────────────────────────────────────
EOF
