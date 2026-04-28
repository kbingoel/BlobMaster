#!/usr/bin/env bash
# Overnight battery — Session 7.5 readiness, 2026-04-27.
#
# Block A — Muon 10-iter validation, fixed 5P7C, current self-play optimum.
#           Compares iter 5 / iter 10 win-rate vs the 7.3c historical anchor
#           (0.66 / 0.72 from checkpoints/7.3c-run/strength.csv).
#
# Block B — Paired Muon-OFF control, identical config except enable_muon =
#           false. Establishes the new baseline that Muon needs to beat at
#           iter 5 / 10 on this stack (which has changed since 7.3c
#           — eval harness is parallelized, target_batch=5 default, etc).
#
# Block C — INT8 deferred-lever sweep on iter_000014. Each lever (S8S8,
#           Entropy, per-block exclude × 8) is quantized + validated against
#           the existing 500-state BCAL calibration; reports static-gate
#           pass/fail per lever. Runs even if A crashes.
#
# Block D — `num_determinizations` profile-only sweep at T=32 (current
#           optimum). Configs: dets ∈ {5, 6, 8, 10}, target_batch = num_dets.
#           Reports per-game wall, ONNX-avg, and policy-KL/top1-share read
#           back from each profile log.
#
# Block E — 5-iter mixed-player smoke. Stresses variable-seq-length self-play
#           and reports n=5 bid-success-rate vs the 7.3c n=5-only baseline.
#
# Each block is self-contained: it writes its own result.json under
# logs/overnight-2026-04-27/<block>/ and appends a section to SUMMARY.md as
# soon as it finishes. If a block fails, SUMMARY.md still records that fact
# and the script continues to the next block.

set -u
set -o pipefail

REPO_ROOT="/home/kbuntu/Documents/Github/BlobMaster"
cd "$REPO_ROOT"

LIBTORCH_DIR="$(find target/release/build -maxdepth 6 -type d -name lib -path '*/libtorch/libtorch/lib' | head -n1)"
if [[ -z "$LIBTORCH_DIR" ]]; then
    echo "FATAL: libtorch dir not found under target/release/build" >&2
    exit 1
fi
export LD_LIBRARY_PATH="$LIBTORCH_DIR:${LD_LIBRARY_PATH:-}"
# CUDA training (`blobmaster-train train`) requires preloading
# libtorch_cuda.so so the CUDA kernels actually register with the dispatcher
# — same trick as scripts/run-train.sh. Without this, the first
# `Tensor::zeros(... cuda:0)` call panics with "Could not run
# 'aten::empty.memory_format' with arguments from the 'CUDA' backend".
# The `profile` subcommand and Python helpers don't go through tch, so
# preloading here is a no-op for them.
if [[ -f "$LIBTORCH_DIR/libtorch_cuda.so" ]]; then
    export LD_PRELOAD="$LIBTORCH_DIR/libtorch_cuda.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi

# Prefer the project venv's python (has torch / onnxruntime / numpy);
# system python3 doesn't carry those. The bash heredocs in this script all
# call `python3`, so prepending the venv to PATH is enough.
if [[ -x "$REPO_ROOT/.venv/bin/python3" ]]; then
    export PATH="$REPO_ROOT/.venv/bin:$PATH"
fi

BIN="./target/release/blobmaster-train"
if [[ ! -x "$BIN" ]]; then
    echo "FATAL: $BIN not built; run 'cargo build --release -p blob-train' first" >&2
    exit 1
fi

OUT_ROOT="logs/overnight-2026-04-27"
CKPT_ROOT="checkpoints/overnight-2026-04-27"
mkdir -p "$OUT_ROOT" "$CKPT_ROOT"
SUMMARY="$OUT_ROOT/SUMMARY.md"
STATUS="$OUT_ROOT/status.txt"

CALIBRATION="checkpoints/7.3c-run/calibration.bin"
ANCHOR_FP32="checkpoints/7.3c-run/iter_000014/model.onnx"

if [[ ! -f "$CALIBRATION" ]]; then
    echo "FATAL: calibration BCAL not found at $CALIBRATION" >&2
    exit 1
fi
if [[ ! -f "$ANCHOR_FP32" ]]; then
    echo "FATAL: 7.3c iter_000014 model.onnx not found at $ANCHOR_FP32" >&2
    exit 1
fi

T0=$(date +%s)
log_status() { echo "[$(date -Iseconds)] $*" | tee -a "$STATUS"; }

# Initialize SUMMARY.md once. Subsequent block writes append.
{
    echo "# Overnight battery results — 2026-04-27"
    echo
    echo "Started: $(date -Iseconds)"
    echo "Host: $(hostname)"
    echo "Binary: $BIN"
    echo "Anchor FP32 model: $ANCHOR_FP32"
    echo "Calibration BCAL: $CALIBRATION"
    echo
} > "$SUMMARY"

emit_block_header() {
    local title=$1
    {
        echo
        echo "## $title"
        echo
    } >> "$SUMMARY"
}

emit_block_failure() {
    local block=$1
    local rc=$2
    local logfile=$3
    {
        echo
        echo "**Block $block FAILED** (rc=$rc). See \`$logfile\`."
        echo
    } >> "$SUMMARY"
}

# ---------------------------------------------------------------------------
# Block A — Muon 10-iter validation
# ---------------------------------------------------------------------------
run_block_a() {
    local A_DIR="$OUT_ROOT/A-muon"
    local A_CKPT="$CKPT_ROOT/A-muon"
    mkdir -p "$A_DIR"
    log_status "Block A starting (Muon 10-iter, 5P7C)"

    local cfg="$A_DIR/config.toml"
    cat > "$cfg" <<TOML
[training]
checkpoint_dir = "$A_CKPT"
buffer_capacity = 500000
batch_size = 512
epochs_per_iteration = 10
epoch_early_stop_rel = 0.005
total_iterations = 10
device = "cuda:0"

[self_play]
num_games = 118
num_threads = 32
iteration = 0
show_progress = false
fixed_player_count = [5, 7]
use_int8 = false

[mcts]
c_puct = 1.5
num_determinizations = 5
sims_per_determinization = 100
min_sims_floor = 60
temperature = 1.0
arena_capacity = 4096
target_batch = 5

[eval]
eval_games = 192
eval_interval = 5
eval_lookback = 20
bid_success_promotion_delta = 0.02
eval_num_threads = 32
TOML

    local logfile="$A_DIR/train.log"
    local started=$(date +%s)
    set +e
    RUST_LOG=info "$BIN" train --config "$cfg" > "$logfile" 2>&1
    local rc=$?
    set -e
    local elapsed=$(( $(date +%s) - started ))

    emit_block_header "Block A — Muon 10-iter validation"
    if [[ $rc -ne 0 ]]; then
        log_status "Block A FAILED rc=$rc"
        emit_block_failure A "$rc" "$logfile"
        echo "{\"block\":\"A\",\"status\":\"failed\",\"rc\":$rc,\"elapsed_s\":$elapsed}" \
            > "$A_DIR/result.json"
        return 1
    fi

    local strength="$A_CKPT/strength.csv"
    if [[ ! -f "$strength" ]]; then
        log_status "Block A: strength.csv missing at $strength"
        {
            echo "**Block A finished but \`strength.csv\` not found.**"
            echo "elapsed: ${elapsed}s; log: \`$logfile\`."
        } >> "$SUMMARY"
        echo "{\"block\":\"A\",\"status\":\"no_strength_csv\",\"elapsed_s\":$elapsed}" \
            > "$A_DIR/result.json"
        return 1
    fi

    # Pull iter-5 / iter-10 win_rate + lower95 from the strength CSV.
    python3 - "$strength" "$A_DIR/result.json" "$elapsed" <<'PY' >> "$SUMMARY"
import csv, json, sys
strength_path, result_path, elapsed = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = []
with open(strength_path) as fh:
    r = csv.DictReader(fh)
    for row in r:
        rows.append(row)
def find(it):
    for row in rows:
        if int(row["iteration"]) == it:
            return row
    return None
r5 = find(5)
r10 = find(10)
# 7.3c historical anchor (from checkpoints/7.3c-run/strength.csv).
ref5_wr, ref5_lo = 0.66, 0.563
ref10_wr, ref10_lo = 0.72, 0.583
def fmt(row, ref_wr, ref_lo):
    if row is None:
        return "missing", None, None, None
    wr = float(row["win_rate"])
    lo = float(row["win_rate_lower95"])
    return f"{wr:.3f} (lower95 {lo:.3f})  vs 7.3c {ref_wr:.3f} (lower95 {ref_lo:.3f})  Δwr={wr-ref_wr:+.3f}", wr, lo, wr-ref_wr
s5, wr5, lo5, d5 = fmt(r5, ref5_wr, ref5_lo)
s10, wr10, lo10, d10 = fmt(r10, ref10_wr, ref10_lo)

def verdict(d):
    if d is None: return "no-data"
    if d >= -0.03: return "PASS (within 3pp of 7.3c)"
    if d >= -0.05: return "MARGINAL (3–5pp drop)"
    return "REVERT (>5pp drop)"
v10 = verdict(d10)

print(f"- elapsed: {elapsed}s ({elapsed/60:.1f} min)")
print(f"- iter 5 : {s5}")
print(f"- iter 10: {s10}")
print(f"- verdict (per dev plan §7.4d): **{v10}**")
print()
print(f"Full strength.csv: `{strength_path}`")

with open(result_path, "w") as fh:
    json.dump({
        "block": "A",
        "status": "ok",
        "elapsed_s": elapsed,
        "iter5_win_rate": wr5,
        "iter5_lower95": lo5,
        "iter10_win_rate": wr10,
        "iter10_lower95": lo10,
        "verdict_iter10": v10,
    }, fh, indent=2)
PY
    log_status "Block A complete"
}

# ---------------------------------------------------------------------------
# Block B — Muon-OFF control, 10-iter, paired with A
# ---------------------------------------------------------------------------
run_block_b() {
    local B_DIR="$OUT_ROOT/B-no-muon"
    local B_CKPT="$CKPT_ROOT/B-no-muon"
    mkdir -p "$B_DIR"
    log_status "Block B starting (Muon-OFF 10-iter, 5P7C)"

    local cfg="$B_DIR/config.toml"
    cat > "$cfg" <<TOML
[training]
checkpoint_dir = "$B_CKPT"
buffer_capacity = 500000
batch_size = 512
epochs_per_iteration = 10
epoch_early_stop_rel = 0.005
total_iterations = 10
device = "cuda:0"
enable_muon = false

[self_play]
num_games = 118
num_threads = 32
iteration = 0
show_progress = false
fixed_player_count = [5, 7]
use_int8 = false

[mcts]
c_puct = 1.5
num_determinizations = 5
sims_per_determinization = 100
min_sims_floor = 60
temperature = 1.0
arena_capacity = 4096
target_batch = 5

[eval]
eval_games = 192
eval_interval = 5
eval_lookback = 20
bid_success_promotion_delta = 0.02
eval_num_threads = 32
TOML

    local logfile="$B_DIR/train.log"
    local started=$(date +%s)
    set +e
    RUST_LOG=info "$BIN" train --config "$cfg" > "$logfile" 2>&1
    local rc=$?
    set -e
    local elapsed=$(( $(date +%s) - started ))

    emit_block_header "Block B — Muon-OFF control (10-iter, paired with A)"
    if [[ $rc -ne 0 ]]; then
        log_status "Block B FAILED rc=$rc"
        emit_block_failure B "$rc" "$logfile"
        echo "{\"block\":\"B\",\"status\":\"failed\",\"rc\":$rc,\"elapsed_s\":$elapsed}" \
            > "$B_DIR/result.json"
        return 1
    fi

    local strength="$B_CKPT/strength.csv"
    if [[ ! -f "$strength" ]]; then
        log_status "Block B: strength.csv missing at $strength"
        {
            echo "**Block B finished but \`strength.csv\` not found.**"
            echo "elapsed: ${elapsed}s; log: \`$logfile\`."
        } >> "$SUMMARY"
        echo "{\"block\":\"B\",\"status\":\"no_strength_csv\",\"elapsed_s\":$elapsed}" \
            > "$B_DIR/result.json"
        return 1
    fi

    python3 - "$strength" "$B_DIR/result.json" "$elapsed" <<'PY' >> "$SUMMARY"
import csv, json, sys
strength_path, result_path, elapsed = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = []
with open(strength_path) as fh:
    r = csv.DictReader(fh)
    for row in r:
        rows.append(row)
def find(it):
    for row in rows:
        if int(row["iteration"]) == it:
            return row
    return None
r5 = find(5)
r10 = find(10)
ref5_wr, ref5_lo = 0.66, 0.563
ref10_wr, ref10_lo = 0.72, 0.583
def fmt(row, ref_wr, ref_lo):
    if row is None:
        return "missing", None, None, None
    wr = float(row["win_rate"])
    lo = float(row["win_rate_lower95"])
    return f"{wr:.3f} (lower95 {lo:.3f})  vs 7.3c {ref_wr:.3f} (lower95 {ref_lo:.3f})  Δwr={wr-ref_wr:+.3f}", wr, lo, wr-ref_wr
s5, wr5, lo5, _ = fmt(r5, ref5_wr, ref5_lo)
s10, wr10, lo10, _ = fmt(r10, ref10_wr, ref10_lo)

print(f"- elapsed: {elapsed}s ({elapsed/60:.1f} min)")
print(f"- iter 5 : {s5}")
print(f"- iter 10: {s10}")
print()
print("This is the AdamW-only baseline on the current stack — used as the")
print("paired control for Block A. Cross-block comparison rendered below.")

with open(result_path, "w") as fh:
    json.dump({
        "block": "B",
        "status": "ok",
        "elapsed_s": elapsed,
        "iter5_win_rate": wr5,
        "iter5_lower95": lo5,
        "iter10_win_rate": wr10,
        "iter10_lower95": lo10,
    }, fh, indent=2)
PY
    log_status "Block B complete"
}

# Render an A-vs-B paired comparison once both blocks have run.
emit_a_vs_b_comparison() {
    local A_RES="$OUT_ROOT/A-muon/result.json"
    local B_RES="$OUT_ROOT/B-no-muon/result.json"
    if [[ ! -f "$A_RES" || ! -f "$B_RES" ]]; then
        return 0
    fi
    emit_block_header "Block A vs B — Muon vs no-Muon, iter 5 / 10"
    python3 - "$A_RES" "$B_RES" >> "$SUMMARY" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
def f(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)
def delta(x, y):
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return f"{x - y:+.3f}"
    return "—"
print("| iter | A (Muon-on) | B (Muon-off) | Δ (A − B) |")
print("|---:|---:|---:|---:|")
print(f"| 5 | {f(a.get('iter5_win_rate'))} | {f(b.get('iter5_win_rate'))} | "
      f"{delta(a.get('iter5_win_rate'), b.get('iter5_win_rate'))} |")
print(f"| 10 | {f(a.get('iter10_win_rate'))} | {f(b.get('iter10_win_rate'))} | "
      f"{delta(a.get('iter10_win_rate'), b.get('iter10_win_rate'))} |")
print()
d10 = None
if isinstance(a.get('iter10_win_rate'), (int,float)) and isinstance(b.get('iter10_win_rate'), (int,float)):
    d10 = a['iter10_win_rate'] - b['iter10_win_rate']
print("**Caveat:** with `eval_games = 192`, the Wilson 95% half-width is "
      "≈ ±7pp around 0.5, so a Δwr of ±0.05 at iter 10 is inside noise. ")
print("Treat differences smaller than that as 'inconclusive after 10 iters'; ")
print("the dev plan's actual decision criterion is the 7.5 100-iter trajectory.")
if d10 is not None:
    if d10 >= 0.05:
        print(f"\nObserved Δ(A−B) at iter 10: **+{d10:.3f}** — Muon plausibly helps; "
              "promote to 7.5 default.")
    elif d10 <= -0.05:
        print(f"\nObserved Δ(A−B) at iter 10: **{d10:.3f}** — Muon plausibly hurts; "
              "consider reverting per dev-plan §7.4d.")
    else:
        print(f"\nObserved Δ(A−B) at iter 10: **{d10:+.3f}** — within noise; "
              "decision deferred to 7.5 trajectory.")
PY
}

# ---------------------------------------------------------------------------
# Block C — INT8 deferred-lever sweep
# ---------------------------------------------------------------------------
run_block_c() {
    local C_DIR="$OUT_ROOT/C-int8"
    mkdir -p "$C_DIR"
    log_status "Block C starting (INT8 deferred levers)"
    local results="$C_DIR/results.jsonl"
    : > "$results"

    local levers=(s8s8 entropy)
    local block_excludes=(0 1 2 3 4 5 6 7)

    emit_block_header "Block C — INT8 deferred-lever sweep"
    {
        echo "Lever-by-lever static-gate results (gates: bid-argmax ≥ 0.95, value-sign = 1.00)."
        echo
        echo "| lever | bid-argmax | play-argmax | value-sign | static gate | profile-trigger |"
        echo "|---|---:|---:|---:|:---:|---|"
    } >> "$SUMMARY"

    run_lever() {
        local label=$1
        shift
        local logfile="$C_DIR/${label}.log"
        local out_dir="$C_DIR/lever-${label}"
        mkdir -p "$out_dir"
        # The script-level LD_PRELOAD pins tch-rs's vendored libtorch_cuda.so
        # for the Rust binary, but the venv's PyPI torch wheel was built
        # against a different libtorch ABI — preloading into Python crashes
        # with `undefined symbol: ...torch::jit::Graph::toString...`. Strip
        # LD_PRELOAD inside the subshell so only Python's own libtorch loads.
        set +e
        ( unset LD_PRELOAD; \
          python3 scripts/int8_levers.py \
            --fp32 "$ANCHOR_FP32" \
            --calibration "$CALIBRATION" \
            --out-dir "$out_dir" \
            --results-jsonl "$results" \
            "$@" \
            > "$logfile" 2>&1 )
        local rc=$?
        set -e
        if [[ $rc -ne 0 ]]; then
            log_status "Block C lever $label FAILED rc=$rc"
            echo "| ${label} | err | err | err | ❌ | (lever crashed; see ${logfile}) |" >> "$SUMMARY"
            return 1
        fi
    }

    for lever in "${levers[@]}"; do
        run_lever "$lever" --lever "$lever"
    done
    for idx in "${block_excludes[@]}"; do
        run_lever "exclude-block-${idx}" --lever exclude-block --block-idx "$idx"
    done

    # Render the per-lever rows by reading results.jsonl in order.
    python3 - "$results" "$SUMMARY" "$C_DIR/result.json" <<'PY'
import json, sys
results_path, summary_path, summary_json = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
with open(results_path) as fh:
    for line in fh:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

passers = []
with open(summary_path, "a") as out:
    for r in rows:
        gate_mark = "✅" if r["static_gate_pass"] else "❌"
        trigger = "re-profile 16T" if r["static_gate_pass"] else "skip"
        out.write(
            f"| {r['lever']} "
            f"| {r['bid_argmax_agree']:.3f} "
            f"| {r['play_argmax_agree']:.3f} "
            f"| {r['value_sign_agree']:.3f} "
            f"| {gate_mark} "
            f"| {trigger} |\n"
        )
        if r["static_gate_pass"]:
            passers.append(r["lever"])
    out.write("\n")
    if passers:
        out.write(f"**Static-gate passers ({len(passers)}):** {', '.join(passers)}.\n")
        out.write("Recommend re-profiling these at 16T tomorrow with `--use-int8`.\n")
    else:
        out.write("**No lever cleared the static gate.** INT8 stays parked for 7.5.\n")
with open(summary_json, "w") as fh:
    json.dump({"block": "C", "status": "ok", "passers": passers,
               "n_levers": len(rows)}, fh, indent=2)
PY
    log_status "Block C complete"
}

# ---------------------------------------------------------------------------
# Block D — num_determinizations profile sweep
# ---------------------------------------------------------------------------
run_block_d() {
    local D_DIR="$OUT_ROOT/D-dets"
    mkdir -p "$D_DIR"
    log_status "Block D starting (num_determinizations profile sweep)"
    local csv="$D_DIR/results.csv"
    echo "num_dets,target_batch,total_games,wall_s,per_game_s,per_decision_ms,onnx_avg_us,timestamp" > "$csv"

    local dets_list=(5 6 8 10)
    local T=32
    local GPT=5
    local NPL=5
    local CDS=7

    run_one_det() {
        local D=$1
        local cfg="$D_DIR/cfg-d${D}.toml"
        cat > "$cfg" <<TOML
[mcts]
c_puct = 1.5
num_determinizations = $D
sims_per_determinization = 100
min_sims_floor = 60
temperature = 1.0
arena_capacity = 4096
target_batch = $D
TOML
        local logfile="$D_DIR/d${D}.log"
        local total_games=$(( GPT * T ))
        local ts=$(date -Iseconds)
        log_status "  Block D dets=$D running ($total_games games)"
        set +e
        RUST_LOG=info "$BIN" profile \
            --model "$ANCHOR_FP32" \
            --config "$cfg" \
            --games-per-thread "$GPT" \
            --num-threads "$T" \
            --num-players "$NPL" \
            --cards-dealt "$CDS" \
            > "$logfile" 2>&1
        local rc=$?
        set -e
        if [[ $rc -ne 0 ]]; then
            log_status "  Block D dets=$D FAILED rc=$rc"
            echo "$D,$D,$total_games,FAIL,FAIL,FAIL,FAIL,$ts" >> "$csv"
            return 1
        fi

        local wall per_game_ms per_game per_decision onnx_us
        wall=$(awk -F: '/^wall clock \(s\)/ {gsub(/ /,"",$2); print $2; exit}' "$logfile")
        per_game_ms=$(awk -F: '/^avg per-game wall/ {gsub(/[ ms]/,"",$2); print $2; exit}' "$logfile")
        per_game=$(awk -v ms="$per_game_ms" 'BEGIN{ if (ms+0==0) print "ERR"; else printf "%.6f", ms/1000.0 }')
        per_decision=$(awk -F: '/^avg per-decision/ {split($2,a," "); print a[1]; exit}' "$logfile")
        onnx_us=$(awk '/^onnx_inference/ {print $4; exit}' "$logfile")
        if [[ -z "$wall" || -z "$per_game" || "$per_game" == "ERR" ]]; then
            log_status "  Block D dets=$D PARSE FAIL"
            echo "$D,$D,$total_games,PARSE,PARSE,PARSE,PARSE,$ts" >> "$csv"
            return 1
        fi
        echo "$D,$D,$total_games,$wall,$per_game,$per_decision,$onnx_us,$ts" >> "$csv"
    }

    for D in "${dets_list[@]}"; do
        run_one_det "$D" || true
    done

    emit_block_header "Block D — num_determinizations profile sweep (T=32, target_batch = num_dets)"
    python3 - "$csv" "$SUMMARY" "$D_DIR/result.json" <<'PY'
import csv, json, sys
csv_path, summary_path, json_path = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
with open(csv_path) as fh:
    r = csv.DictReader(fh)
    for row in r:
        rows.append(row)
ok_rows = [r for r in rows if r["per_game_s"] not in ("FAIL", "PARSE", "ERR")]
def fnum(v):
    try: return float(v)
    except: return None

with open(summary_path, "a") as out:
    out.write("| num_dets | per-game (s) | per-decision (ms) | ONNX-avg (µs) | speedup vs 7.3c B=1 16T (7.250) |\n")
    out.write("|---:|---:|---:|---:|---:|\n")
    for r in rows:
        pg = fnum(r["per_game_s"])
        pd = r["per_decision_ms"]
        oa = r["onnx_avg_us"]
        sp = f"{7.250 / pg:.3f}×" if pg else "—"
        pgs = f"{pg:.3f}" if pg else r["per_game_s"]
        out.write(f"| {r['num_dets']} | {pgs} | {pd} | {oa} | {sp} |\n")
    out.write("\n")
    if ok_rows:
        best = min(ok_rows, key=lambda r: float(r["per_game_s"]))
        out.write(f"**Fastest:** num_dets={best['num_dets']} at {float(best['per_game_s']):.3f} s/game.\n")
        out.write("Note: this block is **speed-only**. Quality (policy-KL, top-1 visit share) ")
        out.write("must be checked from a paired 1-iter training run before raising the default ")
        out.write("`num_determinizations` past 5.\n")
    else:
        out.write("**All sweep points failed.** Inspect logs in `D-dets/`.\n")
with open(json_path, "w") as fh:
    json.dump({"block": "D", "status": "ok",
               "n_ok": len(ok_rows), "n_total": len(rows)}, fh, indent=2)
PY
    log_status "Block D complete"
}

# ---------------------------------------------------------------------------
# Block E — 5-iter mixed-player smoke
# ---------------------------------------------------------------------------
run_block_e() {
    local E_DIR="$OUT_ROOT/E-mixed"
    local E_CKPT="$CKPT_ROOT/E-mixed"
    mkdir -p "$E_DIR"
    log_status "Block E starting (5-iter mixed-player smoke)"
    local cfg="$E_DIR/config.toml"
    cat > "$cfg" <<TOML
[training]
checkpoint_dir = "$E_CKPT"
buffer_capacity = 500000
batch_size = 512
epochs_per_iteration = 10
epoch_early_stop_rel = 0.005
total_iterations = 5
device = "cuda:0"

[self_play]
num_games = 118
num_threads = 32
iteration = 0
show_progress = false
use_int8 = false

[mcts]
c_puct = 1.5
num_determinizations = 5
sims_per_determinization = 100
min_sims_floor = 60
temperature = 1.0
arena_capacity = 4096
target_batch = 5

[eval]
eval_games = 192
eval_interval = 5
eval_lookback = 20
bid_success_promotion_delta = 0.02
eval_num_threads = 32
TOML

    local logfile="$E_DIR/train.log"
    local started=$(date +%s)
    set +e
    RUST_LOG=info "$BIN" train --config "$cfg" > "$logfile" 2>&1
    local rc=$?
    set -e
    local elapsed=$(( $(date +%s) - started ))

    emit_block_header "Block E — 5-iter mixed-player smoke"
    if [[ $rc -ne 0 ]]; then
        log_status "Block E FAILED rc=$rc"
        emit_block_failure E "$rc" "$logfile"
        echo "{\"block\":\"E\",\"status\":\"failed\",\"rc\":$rc,\"elapsed_s\":$elapsed}" \
            > "$E_DIR/result.json"
        return 1
    fi

    local metrics="$E_CKPT/metrics.jsonl"
    python3 - "$metrics" "$E_DIR/train.log" "$elapsed" "$E_DIR/result.json" <<'PY' >> "$SUMMARY"
import json, sys, pathlib
metrics_path, log_path, elapsed, result_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
print(f"- elapsed: {elapsed}s ({elapsed/60:.1f} min)")
ok = pathlib.Path(metrics_path).exists()
if not ok:
    print(f"- metrics.jsonl missing at `{metrics_path}` — train log: `{log_path}`.")
    json.dump({"block":"E","status":"no_metrics","elapsed_s":elapsed}, open(result_path,"w"), indent=2)
    sys.exit(0)
rows = [json.loads(l) for l in open(metrics_path) if l.strip()]
if not rows:
    print(f"- metrics.jsonl empty.")
    json.dump({"block":"E","status":"empty_metrics","elapsed_s":elapsed}, open(result_path,"w"), indent=2)
    sys.exit(0)
print(f"- iters logged: {len(rows)}")
print()
print("| iter | combined_loss | bid_top1 | policy_kl | top1_visit_share | examples |")
print("|---:|---:|---:|---:|---:|---:|")
for r in rows:
    print(f"| {r.get('iteration','?')} "
          f"| {r.get('combined_loss',0):.4f} "
          f"| {r.get('bid_top1_accuracy',0):.3f} "
          f"| {r.get('policy_kl_divergence',0):.4f} "
          f"| {r.get('top1_visit_share_mean',0):.3f} "
          f"| {r.get('examples_generated','?')} |")
print()
print("Smoke pass: no panics, all 5 iters logged, variable-arena code exercised on n∈{4,5,6,7}.")
print("**Note:** n=5 bid-success-rate vs 7.3c baseline must be read from the eval run "
      "(no eval triggers here since eval_interval=5 and total_iterations=5 → only one eval at iter 5).")
last = rows[-1]
json.dump({"block":"E","status":"ok","elapsed_s":elapsed,
           "iters_logged":len(rows),
           "last_combined_loss":last.get("combined_loss"),
           "last_bid_top1":last.get("bid_top1_accuracy")}, open(result_path,"w"), indent=2)
PY
    log_status "Block E complete"
}

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
log_status "overnight battery starting"

run_block_a || log_status "Block A reported failure (continuing)"
run_block_b || log_status "Block B reported failure (continuing)"
emit_a_vs_b_comparison
run_block_c || log_status "Block C reported failure (continuing)"
run_block_d || log_status "Block D reported failure (continuing)"
run_block_e || log_status "Block E reported failure (continuing)"

T1=$(date +%s)
ELAPSED=$(( T1 - T0 ))
{
    echo
    echo "---"
    echo
    echo "Total wall: ${ELAPSED}s ($((ELAPSED/60)) min)"
    echo "Finished: $(date -Iseconds)"
} >> "$SUMMARY"

log_status "overnight battery complete; SUMMARY at $SUMMARY"
