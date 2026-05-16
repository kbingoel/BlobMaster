"""Visualize win-rate trajectory + training-process diagnostics for a run.

Usage (strength-only, original):
    python scripts/visualize_strength.py \
        --csv checkpoints/run-2026-05-14/strength.csv \
        --out-dir logs/run-2026-05-14

Full training-process dashboard (adds convergence + timing plots):
    python scripts/visualize_strength.py \
        --csv checkpoints/run-2026-05-14/strength.csv \
        --metrics checkpoints/run-2026-05-14/metrics.jsonl \
        --stderr logs/run-2026-05-14.stderr \
        --out-dir logs/run-2026-05-14

`--metrics` enables the convergence dashboard (loss / accuracy /
num_epochs_run from blob-nn's per-iter metrics.jsonl) AND the SP-vs-
training split on the timing plot (when `self_play_secs` /
`training_secs` are present in metrics.jsonl; added 2026-05-15).
`--stderr` enables the iteration-timing plot — wall_clock_secs is
parsed from the "iteration complete" log line, and per-iter eval wall
is derived as the gap between consecutive "iteration complete"
timestamps minus the next iter's wall_clock_secs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def group_by_opponent(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["opponent"]].append(r)
    for v in groups.values():
        v.sort(key=lambda r: int(r["iteration"]))
    return groups


def plot_winrate(rows: list[dict], out_path: Path) -> None:
    groups = group_by_opponent(rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    for opp, items in sorted(groups.items()):
        x = [int(r["iteration"]) for r in items]
        y = [float(r["win_rate"]) for r in items]
        lo = [float(r["win_rate_lower95"]) for r in items]
        hi = [float(r["win_rate_upper95"]) for r in items]
        line, = ax.plot(x, y, marker="o", label=f"vs {opp}")
        ax.fill_between(x, lo, hi, alpha=0.15, color=line.get_color())
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", label="parity")
    ax.axhline(0.55, color="tab:green", linewidth=0.6, linestyle=":", label="promote band (lower95 ≥ 0.55)")
    ax.axhline(0.45, color="tab:red", linewidth=0.6, linestyle=":", label="regress band (upper95 ≤ 0.45)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("win rate (95% CI)")
    ax.set_title("Win rate vs anchor opponents")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_score_differential(rows: list[dict], out_path: Path) -> None:
    groups = group_by_opponent(rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    for opp, items in sorted(groups.items()):
        x = [int(r["iteration"]) for r in items]
        y = [float(r["score_differential"]) for r in items]
        ax.plot(x, y, marker="o", label=f"vs {opp}")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("iteration")
    ax.set_ylabel("score differential")
    ax.set_title("Mean score differential vs anchor opponents")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_losses(rows: list[dict], out_path: Path) -> None:
    rows = sorted(rows, key=lambda r: int(r["iteration"]))
    x = [int(r["iteration"]) for r in rows]
    pol = [float(r["policy_loss"]) for r in rows]
    val = [float(r["value_loss"]) for r in rows]
    ent = [float(r["visit_entropy"]) for r in rows]
    kl = [float(r["kl_divergence"]) for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes[0, 0].plot(x, pol, marker="o"); axes[0, 0].set_title("policy loss")
    axes[0, 1].plot(x, val, marker="o", color="tab:orange"); axes[0, 1].set_title("value loss")
    axes[1, 0].plot(x, ent, marker="o", color="tab:green"); axes[1, 0].set_title("visit entropy")
    axes[1, 1].plot(x, kl, marker="o", color="tab:red"); axes[1, 1].set_title("kl divergence")
    for a in axes.flat:
        a.grid(True, alpha=0.3)
        a.set_xlabel("iteration")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- metrics.jsonl dashboard ----------------------------------------------

def load_metrics(metrics_path: Path) -> list[dict]:
    with metrics_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def plot_convergence(metrics: list[dict], out_path: Path) -> None:
    """Per-iter training-step diagnostics. Reads metrics.jsonl, not strength.csv.

    `num_epochs_run` is the early-stop signal — `epoch_early_stop_rel = 0.005`
    fires when a sweep over the buffer improves combined loss by less than
    0.5%. When this drops to 2 the trainer is "converged" on the current
    buffer; when it stays high (8-10) the buffer still contains learnable
    signal.
    """
    metrics = sorted(metrics, key=lambda r: int(r["iteration"]))
    x = [int(r["iteration"]) for r in metrics]
    combined = [float(r["combined_loss"]) for r in metrics]
    value = [float(r["value_loss"]) for r in metrics]
    bid_top1 = [float(r["bid_top1_accuracy"]) for r in metrics]
    play_top1 = [float(r["play_top1_accuracy"]) for r in metrics]
    epochs = [int(r["num_epochs_run"]) for r in metrics]
    lr = [float(r["learning_rate"]) for r in metrics]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes[0, 0].plot(x, combined, marker="o", color="tab:blue")
    axes[0, 0].set_title("combined loss (per-iter)")
    axes[0, 1].plot(x, value, marker="o", color="tab:orange")
    axes[0, 1].set_title("value loss")
    axes[0, 2].plot(x, bid_top1, marker="o", color="tab:purple", label="bid")
    axes[0, 2].plot(x, play_top1, marker="o", color="tab:green", label="play")
    axes[0, 2].set_title("top-1 accuracy")
    axes[0, 2].set_ylim(0.5, 1.0)
    axes[0, 2].legend(loc="lower right", fontsize=8)

    axes[1, 0].bar(x, epochs, color="tab:cyan", alpha=0.7)
    axes[1, 0].set_title("num_epochs_run (early_stop_rel=0.005)")
    axes[1, 0].axhline(10, color="grey", linewidth=0.6, linestyle="--")
    axes[1, 0].set_ylim(0, 11)
    axes[1, 1].plot(x, lr, marker="o", color="tab:red")
    axes[1, 1].set_title("learning rate (cosine schedule)")
    axes[1, 1].set_yscale("log")
    # 4th panel: KL divergence + visit entropy on twin axes — both are
    # MCTS-distribution stability signals; rising KL means policy is
    # shifting fast, falling visit entropy means search is converging.
    ax_kl = axes[1, 2]
    kl = [float(r["policy_kl_divergence"]) for r in metrics]
    ent = [float(r["visit_entropy_mean"]) for r in metrics]
    l1, = ax_kl.plot(x, kl, marker="o", color="tab:red", label="policy KL")
    ax_kl.set_ylabel("policy KL", color="tab:red")
    ax_ent = ax_kl.twinx()
    l2, = ax_ent.plot(x, ent, marker="s", color="tab:green", label="visit entropy")
    ax_ent.set_ylabel("visit entropy", color="tab:green")
    ax_kl.set_title("policy KL vs MCTS visit entropy")
    ax_kl.legend(handles=[l1, l2], loc="upper right", fontsize=8)

    for a in axes.flat:
        a.grid(True, alpha=0.3)
        a.set_xlabel("iteration")
    fig.suptitle("Per-iter training diagnostics (metrics.jsonl)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- iter-timing from stderr ----------------------------------------------

# tracing emits ANSI color codes by default; strip them before regex matching.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# "iteration complete iteration=N wall_clock_secs=W ..." after ANSI strip.
# The leading timestamp is captured via _TS_RE applied to the same line.
_ITER_RE = re.compile(
    r"iteration complete iteration=(\d+) wall_clock_secs=([\d.]+)"
)
_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)")


def parse_iter_timings(stderr_path: Path) -> list[tuple[int, float, datetime]]:
    """Return [(iter, wall_clock_secs, completion_ts)] in order.

    `wall_clock_secs` covers run_iteration() — SP + training step +
    checkpoint save + ONNX export — but **excludes** the post-iter eval,
    which runs in main.rs *after* the iteration_complete log line.
    """
    out: list[tuple[int, float, datetime]] = []
    with stderr_path.open() as f:
        for line in f:
            stripped = _ANSI_RE.sub("", line)
            if "iteration complete" not in stripped:
                continue
            m = _ITER_RE.search(stripped)
            ts = _TS_RE.search(stripped)
            if not m or not ts:
                continue
            iter_ = int(m.group(1))
            wall = float(m.group(2))
            t = datetime.strptime(ts.group(1), "%Y-%m-%dT%H:%M:%S.%fZ")
            out.append((iter_, wall, t))
    out.sort(key=lambda r: r[0])
    return out


def derive_eval_walls(
    iter_timings: list[tuple[int, float, datetime]],
) -> dict[int, float]:
    """Eval after iter K = (T_{K+1} - T_K) - W_{K+1} - tiny_gap.

    Iters with no following completion logged yet (last in-progress iter)
    are omitted. Iters where the derived eval is <= 5s are treated as
    "no eval ran" (just checkpoint i/o jitter) and also omitted.

    Resume gaps masquerade as huge "evals" (training was paused between
    consecutive `iteration complete` lines). Detect these via the ratio
    to the next iter's wall — a real eval at the cap is roughly the same
    order of magnitude as one iter's compute, never 5× larger. Above the
    threshold the value almost certainly contains a pause, so we drop it
    rather than letting it crush the y-axis.
    """
    RESUME_GAP_RATIO = 5.0
    out: dict[int, float] = {}
    for i in range(len(iter_timings) - 1):
        k, _w_k, t_k = iter_timings[i]
        _k1, w_k1, t_k1 = iter_timings[i + 1]
        between = (t_k1 - t_k).total_seconds()
        eval_wall = between - w_k1
        if eval_wall <= 5.0:
            continue
        if w_k1 > 0 and eval_wall > RESUME_GAP_RATIO * w_k1:
            # Looks like a pause/resume gap, not an eval.
            continue
        out[k] = eval_wall
    return out


def plot_iter_timing(
    iter_timings: list[tuple[int, float, datetime]],
    eval_walls: dict[int, float],
    out_path: Path,
    metrics: list[dict] | None = None,
) -> None:
    """Stacked bar of per-iter wall time.

    When `metrics` is provided AND `self_play_secs` / `training_secs` are
    present (added 2026-05-15), the iter wall is broken into three stacks:
    SP (bottom), training (middle), and "other" iter overhead (ckpt save +
    ONNX export + buffer.save, computed as `wall_clock_secs - SP - train`).
    Post-iter eval wall sits on top.

    When metrics are absent or the fields are missing (older runs), falls
    back to the prior behaviour: a single iter-wall bar plus the eval bar.
    """
    iters = [t[0] for t in iter_timings]
    walls = [t[1] for t in iter_timings]
    evals = [eval_walls.get(k, 0.0) for k in iters]

    sp_by_iter: dict[int, float] = {}
    tr_by_iter: dict[int, float] = {}
    epoch_secs_by_iter: dict[int, list[float]] = {}
    if metrics:
        for m in metrics:
            it = int(m["iteration"])
            sp = m.get("self_play_secs")
            tr = m.get("training_secs")
            es = m.get("epoch_secs")
            if sp is not None and tr is not None:
                sp_by_iter[it] = float(sp)
                tr_by_iter[it] = float(tr)
            if isinstance(es, list) and es:
                epoch_secs_by_iter[it] = [float(v) for v in es]
    has_split = bool(sp_by_iter) and bool(tr_by_iter)
    has_epochs = bool(epoch_secs_by_iter)

    fig, ax = plt.subplots(figsize=(13, 6))
    if has_split:
        # Only stack the split where we have data for the iter; fall back
        # to a single iter-wall bar for iters that predate the metric.
        sp_vals = [sp_by_iter.get(k, 0.0) for k in iters]
        tr_vals = [tr_by_iter.get(k, 0.0) for k in iters]
        # "other" = wall - sp - tr (checkpoint + buffer + ONNX export +
        # any scheduling slack). Clamp to 0 in case of clock skew.
        other_vals = [max(0.0, w - sp - tr) for w, sp, tr in zip(walls, sp_vals, tr_vals)]
        # Where SP/TR are missing (iter predates the metric), fall back to
        # putting the full wall in "other" so the bar still shows up.
        for i, k in enumerate(iters):
            if k not in sp_by_iter or k not in tr_by_iter:
                other_vals[i] = walls[i]
                sp_vals[i] = 0.0
                tr_vals[i] = 0.0

        ax.bar(iters, sp_vals, color="tab:blue", label="self-play")

        if has_epochs:
            # Per-epoch stack: one sub-bar per epoch, stacked on top of SP.
            # Global grey gradient — epoch 1 darkest, epoch N lightest —
            # keyed to `epochs_per_iteration` (defaulted to 10 if unknown)
            # so colours are comparable across iters. Iters that ran fewer
            # epochs just produce shorter stacks, making `num_epochs_run`
            # readable directly from the count of grey bands.
            max_epochs = max(
                (len(v) for v in epoch_secs_by_iter.values()), default=10
            )
            max_epochs = max(max_epochs, 10)
            # Dark grey = 0.25, light grey = 0.85 in matplotlib's 0..1 scale.
            def grey(epoch_zero_indexed: int) -> tuple[float, float, float]:
                if max_epochs <= 1:
                    g = 0.5
                else:
                    g = 0.25 + 0.60 * (epoch_zero_indexed / (max_epochs - 1))
                return (g, g, g)

            # Stack epoch sub-bars on top of SP, in order. We draw one bar
            # per (iter, epoch_index) slot; only iters with that epoch
            # contribute a non-zero height. Single legend entry per epoch
            # index for readability.
            legended: set[int] = set()
            for e_idx in range(max_epochs):
                heights = []
                bottoms = []
                for k, sp in zip(iters, sp_vals):
                    es = epoch_secs_by_iter.get(k, [])
                    if e_idx < len(es):
                        # Stack onto SP + sum of prior epochs.
                        prior = sum(es[:e_idx])
                        heights.append(es[e_idx])
                        bottoms.append(sp + prior)
                    else:
                        heights.append(0.0)
                        bottoms.append(sp)
                # Only label epochs 1, 5, max in the legend so it stays readable.
                show_in_legend = e_idx in (0, 4, max_epochs - 1)
                label = f"epoch {e_idx + 1}" if show_in_legend else None
                if label and e_idx not in legended:
                    legended.add(e_idx)
                ax.bar(iters, heights, bottom=bottoms, color=grey(e_idx),
                       edgecolor="white", linewidth=0.3, label=label)

            # For iters without per-epoch data but with `training_secs`,
            # draw the single training block in solid green so the bar's
            # total height still matches `wall`.
            fallback_heights = []
            fallback_bottoms = []
            for k, sp, tr in zip(iters, sp_vals, tr_vals):
                if k in epoch_secs_by_iter:
                    fallback_heights.append(0.0)
                    fallback_bottoms.append(sp)
                else:
                    fallback_heights.append(tr)
                    fallback_bottoms.append(sp)
            if any(h > 0 for h in fallback_heights):
                ax.bar(iters, fallback_heights, bottom=fallback_bottoms,
                       color="tab:green", alpha=0.6,
                       label="training (no per-epoch data)")
        else:
            ax.bar(iters, tr_vals, bottom=sp_vals,
                   color="tab:green", label="training")

        bottom_other = [s + t for s, t in zip(sp_vals, tr_vals)]
        ax.bar(iters, other_vals, bottom=bottom_other, color="tab:olive",
               alpha=0.7, label="other (ckpt + export + buffer)")
        ax.bar(iters, evals, bottom=walls, color="tab:orange",
               label="post-iter eval wall (vs anchor)")
    else:
        ax.bar(iters, walls, color="tab:blue",
               label="iter wall (SP + train + ckpt + export)")
        ax.bar(iters, evals, bottom=walls, color="tab:orange",
               label="post-iter eval wall (vs anchor)")

    if walls:
        mean_wall = sum(walls) / len(walls)
        ax.axhline(mean_wall, color="black", linewidth=0.7, linestyle=":",
                   label=f"mean iter wall: {mean_wall:.0f}s")
    if has_split:
        sp_nonzero = [v for v in sp_vals if v > 0]
        if sp_nonzero:
            ax.axhline(sum(sp_nonzero) / len(sp_nonzero),
                       color="tab:blue", linewidth=0.6, linestyle="--",
                       label=f"mean SP: {sum(sp_nonzero) / len(sp_nonzero):.0f}s")

    ax.set_xlabel("iteration")
    ax.set_ylabel("seconds")
    title = "Wall-clock per iteration"
    if has_epochs:
        title += " (SP / per-epoch greys / other / eval)"
    elif has_split:
        title += " (SP / training / other / eval stacks)"
    else:
        title += " (eval bars on top)"
    ax.set_title(title)
    # Y-axis ticks every 300s (5 min) so iter-wall totals line up to a
    # readable grid.
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(300))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path,
                    help="strength.csv — drives the 3 original plots")
    ap.add_argument("--metrics", type=Path,
                    help="metrics.jsonl — enables convergence dashboard")
    ap.add_argument("--stderr", type=Path,
                    help="run stderr log — enables iter-timing plot")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.csv)
    plot_winrate(rows, args.out_dir / "01_winrate.png")
    plot_score_differential(rows, args.out_dir / "02_score_differential.png")
    plot_losses(rows, args.out_dir / "03_train_losses.png")

    metrics: list[dict] | None = None
    if args.metrics and args.metrics.exists():
        metrics = load_metrics(args.metrics)
        plot_convergence(metrics, args.out_dir / "04_convergence.png")

    if args.stderr and args.stderr.exists():
        iter_timings = parse_iter_timings(args.stderr)
        if iter_timings:
            eval_walls = derive_eval_walls(iter_timings)
            plot_iter_timing(
                iter_timings,
                eval_walls,
                args.out_dir / "05_iter_timing.png",
                metrics=metrics,
            )

    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()
