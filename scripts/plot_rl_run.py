"""Plot a `blobmaster-train train` run (gen-2.md §6 Phase 5).

    ( unset LD_PRELOAD; .venv/bin/python scripts/plot_rl_run.py checkpoints/<run> [--no-weights] )

Reads `<run>/metrics.jsonl`, the bench reports in `<run>/bench/` and the
published checkpoints in `<run>/models/`, and writes `<run>/plots/`:

- `00_overview.png`        the run on one page: strength, held-out losses, learning signal
- `01_strength.png`        bench points/game with 95% CI bands, absolute and paired
- `02_bids.png`            bids made by cards dealt, P against its opponents (gen 1's bid-success chart)
- `03_learning.png`        training losses, P's agreement with search, change per publish, pace
- `04_generalization.png`  held out: validation vs an equal training sample; drift from the teacher
- `05_selfplay.png`        search health in self-play: disagreement with P, KL, entropy; bids made; throughput
- `06_weights.png`         weight evolution: change per publish by layer, distance from the warm start
- `08_value_margin.png`    search's margin over P alone (vs rule bot 2 paired on the same deals and step, and
                           vs four copies of P, where P alone scores 0); V on the fixed
                           rounds of an earlier run; V on its own stream of P-alone rounds (runs from 2026-10-07)
- `09_rollouts.png`        policy iteration by rollouts (runs from 2026-10-08): P vs four copies of the start P and vs
                           rule bot 2; per-decision gain of P's (and V's) top move over the playing P on held-out deals;
                           the loss, validation vs training sample; rollout throughput
- `10_panel.png`           P alone against the panel of fixed opponents (`eval.panel`, runs from 2026-10-09) and the
                           rule bots: each one's change since step 0, paired on the same deals

Every x-axis is running hours (pauses excluded); a bench sits at the hour
its model was published. Weights need the venv's torch (`--no-weights`
skips them); the rest needs only matplotlib.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Reference palette, light mode. Categorical slots 1-3 (validated all-pairs);
# aqua is below 3:1 on the surface, so it always carries a legend or label.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Sequential blue, light -> dark (magnitude: the weight heatmaps).
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
REPO = Path(__file__).resolve().parent.parent
BUCKETS = ["1 card", "2–4 cards", "5–8 cards"]
WARM_START_SEARCH_RB2 = (6.4, 1.1)  # Phase 4b, seed 7, 128 deals (gen-2.md §6)


def num(x) -> float:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else math.nan


class Run:
    def __init__(self, path: Path):
        self.path = path
        self.rows = []
        for line in (path / "metrics.jsonl").read_text().splitlines():
            try:
                self.rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        self.step_hours: dict[int, float] = {0: 0.0}
        for r in self.rows:
            if r.get("kind") in ("train", "held_out") and "active_hours" in r:
                self.step_hours[r["step"]] = r["active_hours"]
        self.known = sorted(self.step_hours)
        hours = [r.get("active_hours") for r in self.rows if isinstance(r.get("active_hours"), (int, float))]
        self.max_hours = max(hours) if hours else 1.0

    def of(self, kind: str, **match) -> list[dict]:
        return [r for r in self.rows if r.get("kind") == kind and all(r.get(k) == v for k, v in match.items())]

    def hours(self, step: int) -> float:
        """Running hours at which the learner reached `step`."""
        if step in self.step_hours:
            return self.step_hours[step]
        below = [s for s in self.known if s <= step]
        return self.step_hours[below[-1]] if below else math.nan

    def bench_table(self, name: str, step: int) -> dict | None:
        """Bid rows of a bench report: made / 0-bids, focal and opponents, by bucket."""
        f = self.path / "bench" / f"{name}-step-{step:06d}.txt"
        if not f.is_file():
            return None
        out = {}
        for line in f.read_text().splitlines():
            t = line.split()
            key = {"1": 0, "2-4": 1, "5-8": 2}.get(t[0] if t else "")
            if key is None or len(t) < 7 or t[1] not in ("card", "cards"):
                continue
            out[key] = tuple(float(x) for x in t[3:7])  # made, 0-bids, opp made, opp 0-bids
        return out


# ---- drawing helpers ------------------------------------------------------------


def style(ax, title: str, ylabel: str, xlabel: str = "running hours", run: "Run | None" = None) -> None:
    """Recessive grid and axes; with `run`, the x-axis spans the whole run so far."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=TEXT, pad=8)
    ax.set_ylabel(ylabel, fontsize=8, color=TEXT_2)
    ax.set_xlabel(xlabel, fontsize=8, color=TEXT_2)
    ax.tick_params(colors=TEXT_2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.margins(x=0.06)
    if run is not None:
        ax.set_xlim(0, run.max_hours * 1.1)
        if not ax.lines and not ax.collections:
            ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes, ha="center", va="center", fontsize=9, color=TEXT_2)


def clean(xs, *cols):
    keep = [i for i, x in enumerate(xs) if math.isfinite(x) and all(math.isfinite(c[i]) for c in cols[:1])]
    return [xs[i] for i in keep], *[[c[i] for i in keep] for c in cols]


def series(ax, xs, ys, color, label=None, *, ci=None, dashed=False, marker=True, end=None, alpha=1.0, lw=1.6):
    """A line; `ci` draws a 95% band; `end` formats a direct label at the last point."""
    cols = [ys] + ([ci] if ci is not None else [])
    got = clean(list(xs), *cols)
    x, y = got[0], got[1]
    if not x:
        return
    ax.plot(x, y, color=color, linewidth=lw, linestyle="--" if dashed else "-", alpha=alpha,
            marker="o" if marker else None, markersize=4.5, label=label)
    if ci is not None:
        c = [v if math.isfinite(v) else 0.0 for v in got[2]]
        ax.fill_between(x, [a - b for a, b in zip(y, c)], [a + b for a, b in zip(y, c)], color=color, alpha=0.15, linewidth=0)
    if end:
        note = ax.annotate(end.format(y[-1]), (x[-1], y[-1]), xytext=(6, 0), textcoords="offset pixels",
                           va="center", fontsize=8, color=TEXT_2)
        note._end_label = True


def rolling(xs, ys, window):
    out = []
    for i in range(len(ys)):
        w = [v for v in ys[max(0, i - window + 1): i + 1] if math.isfinite(v)]
        out.append(sum(w) / len(w) if w else math.nan)
    return xs, out


def smooth_series(ax, xs, ys, color, label, window=10):
    """Per-minute rows: faint raw line, rolling mean on top."""
    series(ax, xs, ys, color, None, marker=False, alpha=0.25, lw=1.0)
    x, y = rolling(xs, ys, window)
    series(ax, x, y, color, label, marker=False)


def legend(ax, loc=None):
    """For two or more series: one row above the plot, clear of the data."""
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= 2:
        ax.legend(handles, labels, fontsize=8, frameon=False, labelcolor=TEXT_2, loc="lower left",
                  bbox_to_anchor=(0, 1.0), ncol=min(len(handles), 4), borderaxespad=0.2, handlelength=1.8, columnspacing=1.2)
        ax.set_title(ax.get_title(loc="left"), loc="left", fontsize=10, color=TEXT, pad=24)
    declutter(ax)


def declutter(ax, min_px=11):
    """Push direct end labels apart vertically so none overlap."""
    notes = [t for t in ax.texts if getattr(t, "_end_label", False)]
    if len(notes) < 2:
        return
    fig = ax.figure
    fig.canvas.draw()
    pos = sorted(((ax.transData.transform(t.xy)[1], t) for t in notes), key=lambda p: p[0])
    placed = []
    for y, t in pos:
        if placed and y - placed[-1] < min_px:
            y = placed[-1] + min_px
        placed.append(y)
        dy = y - ax.transData.transform(t.xy)[1]
        t.set_position((6, dy))


def zero_line(ax, y=0.0):
    ax.axhline(y, color=TEXT_2, linewidth=0.9)


def save(fig, path: Path, title: str):
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def figure(rows, cols, w=5.0, h=3.7):
    fig, axes = plt.subplots(rows, cols, figsize=(w * cols, h * rows), facecolor=SURFACE, squeeze=False)
    return fig, axes


# ---- panels ---------------------------------------------------------------------------


def p_bench(ax, run: Run, name: str, title: str, warm=None):
    b = run.of("bench", name=name)
    xs = [run.hours(r["step"]) for r in b]
    series(ax, xs, [num(r["diff"]) for r in b], BLUE, "points/game", ci=[num(r["ci"]) for r in b], end="{:+.1f}")
    paired = [(x, q) for x, r in zip(xs, b) for q in r.get("paired", []) if q.get("vs") in ("start", "warm start", "baseline")]
    if paired:
        label = "change vs " + ("step 0 (paired)" if paired[0][1]["vs"] == "start" else "warm start (paired)")
        series(ax, [x for x, _ in paired], [num(q["diff"]) for _, q in paired], ORANGE, label,
               ci=[num(q["ci"]) for _, q in paired], end="{:+.1f}")
    if warm is not None:
        ax.axhline(warm[0], color=AQUA, linewidth=1.4, linestyle="--", label=f"warm start {warm[0]:+.1f} ± {warm[1]:.1f}")
        ax.axhspan(warm[0] - warm[1], warm[0] + warm[1], color=AQUA, alpha=0.10, linewidth=0)
    zero_line(ax)
    style(ax, title, "points/game vs opponents' mean", run=run)
    if not b:
        ax.text(0.5, 0.5, "no bench yet", transform=ax.transAxes, ha="center", va="center", fontsize=9, color=TEXT_2)
    legend(ax)


def p_bids(ax, run: Run, bucket: int):
    for name, color, opp in (("net-rb2", BLUE, "rule bot 2"), ("net-rb", ORANGE, "rule bot")):
        b = run.of("bench", name=name)
        pts = [(run.hours(r["step"]), run.bench_table(name, r["step"])) for r in b]
        pts = [(x, t[bucket]) for x, t in pts if t and bucket in t]
        if not pts:
            continue
        xs = [x for x, _ in pts]
        series(ax, xs, [t[0] for _, t in pts], color, f"P vs {opp}", end="{:.3f}")
        series(ax, xs, [t[2] for _, t in pts], color, f"{opp} (opponents)", dashed=True, marker=False)
    style(ax, f"Bids made, {BUCKETS[bucket]}", "share of bids made", run=run)
    declutter(ax)


def p_zero_bids(ax, run: Run):
    for name, color, opp in (("net-rb2", BLUE, "rule bot 2"), ("net-rb", ORANGE, "rule bot")):
        b = run.of("bench", name=name)
        pts = [(run.hours(r["step"]), run.bench_table(name, r["step"])) for r in b]
        pts = [(x, t[2]) for x, t in pts if t and 2 in t]
        if not pts:
            continue
        xs = [x for x, _ in pts]
        series(ax, xs, [t[1] for _, t in pts], color, f"P vs {opp}", end="{:.3f}")
        series(ax, xs, [t[3] for _, t in pts], color, f"{opp} (opponents)", dashed=True, marker=False)
    style(ax, "0-bids, 5–8 cards", "share of bids that are 0", run=run)
    declutter(ax)


def p_heldout(ax, run: Run, net: str, key: str, title: str, ylabel: str, ref_key: str | None = None):
    h = run.of("held_out")
    xs = [num(r.get("active_hours")) for r in h]
    series(ax, xs, [num(r["validation"][net].get(key)) for r in h], BLUE, "validation", end="{:.3f}")
    series(ax, xs, [num(r["train_sample"][net].get(key)) for r in h], ORANGE, "training sample")
    if ref_key:
        series(ax, xs, [num(r["validation"][net].get(ref_key)) for r in h], AQUA, "targets' variance", dashed=True, marker=False)
    style(ax, title, ylabel, run=run)
    legend(ax)


def p_probe_agreement(ax, run: Run):
    h = run.of("held_out")
    xs = [num(r.get("active_hours")) for r in h]
    p = lambda r, k: 100 * num(r["teacher_probe"]["policy"].get(k))  # noqa: E731
    series(ax, xs, [p(r, "bid_agreement") for r in h], BLUE, "bids", end="{:.1f}%")
    series(ax, xs, [p(r, "play_agreement") for r in h], ORANGE, "plays", end="{:.1f}%")
    style(ax, "P's top move = rule bot 2's (fixed probe states)", "%", run=run)
    legend(ax)


def p_probe_last_trick(ax, run: Run):
    h = run.of("held_out")
    xs = [num(r.get("active_hours")) for r in h]
    series(ax, xs, [math.sqrt(num(r["teacher_probe"]["value"].get("last_trick_mse"))) for r in h], BLUE, end="{:.3f}")
    ax.axhline(0.05, color=AQUA, linewidth=1.4, linestyle="--", label="G1 bar 0.05")
    style(ax, "V's last-trick RMSE (probe states)", "RMSE of ŝ", run=run)
    ax.set_ylim(bottom=0)
    legend(ax)


def p_agreement_search(ax, run: Run):
    h = run.of("held_out")
    xs = [num(r.get("active_hours")) for r in h]
    series(ax, xs, [100 * num(r["validation"]["policy"].get("bid_agreement")) for r in h], BLUE, "bids", end="{:.1f}%")
    series(ax, xs, [100 * num(r["validation"]["policy"].get("play_agreement")) for r in h], ORANGE, "plays", end="{:.1f}%")
    style(ax, "P's top move = search's (held-out self-play)", "%", run=run)
    legend(ax)


def p_train_loss(ax, run: Run, key: str, title: str):
    t = run.of("train")
    smooth_series(ax, [num(r.get("active_hours")) for r in t], [num(r.get(key)) for r in t], BLUE, None, window=8)
    style(ax, title, "training loss (dropout on)", run=run)


def p_change(ax, run: Run, keys, labels, title, ylabel):
    pr = run.of("probe")
    xs = [run.hours(r["step"]) for r in pr]
    for key, label, color in zip(keys, labels, (BLUE, ORANGE, AQUA)):
        series(ax, xs, [num(r.get(key)) for r in pr], color, label)
    style(ax, title, ylabel, run=run)
    ax.set_ylim(bottom=0)
    legend(ax)


def p_pace(ax, run: Run):
    t = run.of("train")
    series(ax, [num(r.get("active_hours")) for r in t], [num(r.get("steps_per_hour")) for r in t], BLUE, marker=False, lw=1.2, end="{:.0f}")
    style(ax, "Learner pace (catch-up burst, then held to the replay ratio)", "learner steps per hour (log)", run=run)
    ax.set_yscale("log")


def p_selfplay(ax, run: Run, getters, labels, title, ylabel, pct=False, bottom0=True):
    sp = run.of("selfplay")
    xs = [num(r.get("active_hours")) for r in sp]
    for get, label, color in zip(getters, labels, (BLUE, ORANGE, AQUA)):
        ys = [num(get(r)) * (100 if pct else 1) for r in sp]
        smooth_series(ax, xs, ys, color, label)
    style(ax, title, ylabel, run=run)
    if bottom0:
        ax.set_ylim(bottom=0)
    legend(ax)


def p_margin(ax, run: Run):
    """Search and P alone on the search bench's deals, and search's margin (paired)."""
    nets = [r for r in run.rows if r.get("kind") == "bench" and str(r.get("name", "")).startswith("net-rb2-s")]
    series(ax, [run.hours(r["step"]) for r in nets], [num(r["diff"]) for r in nets], ORANGE, "P alone", end="{:+.1f}", marker=False)
    b = run.of("bench", name="search-rb2")
    series(ax, [run.hours(r["step"]) for r in b], [num(r["diff"]) for r in b], BLUE, "search", ci=[num(r["ci"]) for r in b], end="{:+.1f}")
    m = [(run.hours(r["step"]), q) for r in b for q in r.get("paired", []) if q.get("vs") == "P alone"]
    series(ax, [x for x, _ in m], [num(q["diff"]) for _, q in m], AQUA, "search − P alone (paired)",
           ci=[num(q["ci"]) for _, q in m], end="{:+.1f}")
    vp = run.of("bench", name="search-vsP")
    series(ax, [run.hours(r["step"]) for r in vp], [num(r["diff"]) for r in vp], TEXT_2, "search vs 4 × P (P alone: 0)",
           ci=[num(r["ci"]) for r in vp], end="{:+.1f}", dashed=True)
    zero_line(ax)
    style(ax, "Search's margin over P alone", "points/game", run=run)
    legend(ax)


def p_fixed_value(ax, run: Run):
    h = [r for r in run.of("held_out") if isinstance(r.get("fixed_value"), dict)]
    xs = [num(r.get("active_hours")) for r in h]
    for key, label, color in (("all", "all states", BLUE), ("bids", "bid states", ORANGE), ("plays", "play states", AQUA)):
        series(ax, xs, [num(r["fixed_value"][key].get("mse")) for r in h], color, label, end="{:.4f}", marker=False)
    src = h[0]["fixed_value"].get("from", "") if h else ""
    style(ax, f"V on fixed rounds ({Path(src).name or 'none'})", "MSE of ŝ", run=run)
    legend(ax)


def p_value_stream(ax, run: Run):
    h = [r for r in run.of("held_out") if isinstance(r.get("value_stream"), dict)]
    xs = [num(r.get("active_hours")) for r in h]
    series(ax, xs, [num(r["value_stream"]["validation"].get("mse")) for r in h], BLUE, "validation", end="{:.4f}")
    series(ax, xs, [num(r["value_stream"]["train_sample"].get("mse")) for r in h], ORANGE, "training sample")
    series(ax, xs, [num(r["value_stream"]["validation"].get("variance")) for r in h], AQUA, "targets' variance", dashed=True, marker=False)
    style(ax, "V on its stream of P-alone rounds (held out)", "MSE of ŝ", run=run)
    legend(ax)


def p_value_pace(ax, run: Run):
    t = run.of("train")
    xs = [num(r.get("active_hours")) for r in t]
    smooth_series(ax, xs, [num(r.get("value_only_steps_per_hour")) for r in t], BLUE, "V-only updates", window=8)
    smooth_series(ax, xs, [num(r.get("steps_per_hour")) for r in t], ORANGE, "P + V steps", window=8)
    style(ax, "Learner pace", "updates per hour", run=run)
    ax.set_ylim(bottom=0)
    legend(ax)


def p_rollout_gain(ax, run: Run, key: str, ci_key: str, title: str, parts=("validation", "recent")):
    """Held-out gain per decision (utility units) by phase: the first part with 95% bands, the second dashed."""
    h = [r for r in run.of("held_out") if isinstance(r.get("rollout"), dict)]
    xs = [num(r.get("active_hours")) for r in h]
    names = {"validation": "validation", "recent": "newest"}
    for ph, color in (("bids", BLUE), ("plays", ORANGE)):
        main = parts[0]
        series(ax, xs, [num(r["rollout"][main][ph].get(key)) for r in h], color, f"{ph} ({names[main]})",
               ci=[num(r["rollout"][main][ph].get(ci_key)) for r in h], end="{:+.4f}")
        for extra in parts[1:]:
            series(ax, xs, [num(r["rollout"][extra][ph].get(key)) for r in h], color, f"{ph} ({names[extra]})", dashed=True, marker=False)
    zero_line(ax)
    style(ax, title, "utility per decision (1 pt of a 7-card round ≈ 0.06)", run=run)
    legend(ax)


def p_rollout_loss(ax, run: Run):
    h = [r for r in run.of("held_out") if isinstance(r.get("rollout"), dict)]
    xs = [num(r.get("active_hours")) for r in h]

    def total(part):
        n = sum(num(part[ph].get("samples")) for ph in ("bids", "plays"))
        return sum(num(part[ph].get("loss")) * num(part[ph].get("samples")) for ph in ("bids", "plays")) / n if n else math.nan

    series(ax, xs, [total(r["rollout"]["validation"]) for r in h], BLUE, "validation", end="{:.4f}")
    series(ax, xs, [total(r["rollout"]["train_sample"]) for r in h], ORANGE, "training sample")
    style(ax, "Policy-iteration loss, held out", "T·KL − Σπu (a growing gap = memorizing)", run=run)
    legend(ax)


def p_rollout_pace(ax, run: Run):
    sp = [r for r in run.of("selfplay") if isinstance(r.get("rollout"), dict)]
    xs = [num(r.get("active_hours")) for r in sp]
    smooth_series(ax, xs, [num(r["rollout"].get("samples_per_hour")) / 1000 for r in sp], BLUE, None)
    style(ax, "Rollouts: valued decisions per hour", "thousands per hour", run=run)
    ax.set_ylim(bottom=0)


def fig_rollouts(run: Run, out: Path):
    if not any(isinstance(r.get("rollout"), dict) for r in run.of("held_out")):
        return
    fig, a = figure(2, 3)
    p_bench(a[0][0], run, "net-vs0", "Network only vs four copies of the start P")
    p_bench(a[0][1], run, "net-rb2", "Network only vs rule bot 2")
    p_rollout_pace(a[0][2], run)
    p_rollout_gain(a[1][0], run, "gain_vs_start", "gain_vs_start_ci", "P's top move vs the start P's (newest held-out deals)",
                   parts=("recent",))
    p_rollout_gain(a[1][1], run, "v_gain", "v_gain_ci", "V's pick (next state, true deal) vs the playing P's")
    p_rollout_loss(a[1][2], run)
    save(fig, out / "09_rollouts.png", f"{run.path.name}: policy iteration by rollouts")


def p_paired_start(ax, run: Run, name: str, title: str):
    """A bench's change since step 0, paired on the same deals, with its 95% band."""
    b = run.of("bench", name=name)
    m = [(run.hours(r["step"]), q) for r in b for q in r.get("paired", []) if q.get("vs") == "start"]
    series(ax, [x for x, _ in m], [num(q["diff"]) for _, q in m], BLUE, None, ci=[num(q["ci"]) for _, q in m], end="{:+.2f}")
    zero_line(ax)
    style(ax, title, "change since step 0, paired (points/game)", run=run)


def fig_panel(run: Run, out: Path):
    """The panel benches (`eval.panel`, runs from 2026-10-09): one plot per fixed opponent, then the rule bots."""
    names = sorted({r["name"] for r in run.of("bench") if str(r.get("name", "")).startswith("net-vs-")})
    if not names:
        return
    plots = [(n, f"vs four copies of {n[len('net-vs-'):]}") for n in names]
    plots += [("net-rb2", "vs rule bot 2"), ("net-rb", "vs the rule bot")]
    cols = 3
    fig, a = figure(math.ceil(len(plots) / cols), cols)
    for k, (name, title) in enumerate(plots):
        p_paired_start(a[k // cols][k % cols], run, name, title)
    for k in range(len(plots), a.size):
        a[k // cols][k % cols].set_visible(False)
    save(fig, out / "10_panel.png", f"{run.path.name}: P alone against the panel of fixed opponents")


# ---- weights ------------------------------------------------------------------------------


def group_of(name: str) -> str:
    if name.startswith("input."):
        return "input"
    m = re.match(r"transformer\.layers\.(\d+)\.", name)
    if m:
        return f"layer {m.group(1)}"
    if name.startswith("transformer."):
        return "final norm"
    return "heads"


def load_checkpoints(run: Run):
    """[(step, {"policy": {name: array}, "value": {...}})], the warm start first."""
    sys.path.insert(0, str(REPO / "scripts"))
    import export_onnx as ex  # noqa: E402  (torch)

    cfg = (run.path / "config.toml").read_text()
    m = re.search(r'init_checkpoint\s*=\s*"([^"]+)"', cfg)
    dirs = []
    if m:
        init = Path(m.group(1))
        dirs.append((0, init if init.is_absolute() or init.exists() else REPO / init))
    for d in sorted((run.path / "models").glob("step-*/checkpoint")):
        dirs.append((int(d.parent.name.split("-")[1]), d))
    out = []
    for step, d in dirs:
        if not (d / "policy.ot").is_file():
            continue
        nets = {}
        for key, cls, file in (("policy", ex.PolicyNet, "policy.ot"), ("value", ex.ValueNet, "value.ot")):
            net = cls()
            ex.load_varstore_into(net, d / file)
            nets[key] = {n: t.detach().double().numpy() for n, t in net.state_dict().items()}
        out.append((step, nets))
    return out


def group_norms(weights: dict, other: dict | None = None) -> dict[str, float]:
    """Per group: sqrt(sum of squares) of the weights, or of their difference from `other`."""
    acc: dict[str, float] = {}
    for n, w in weights.items():
        d = w - other[n] if other is not None else w
        acc[group_of(n)] = acc.get(group_of(n), 0.0) + float((d * d).sum())
    return {g: math.sqrt(v) for g, v in acc.items()}


def order_groups(groups) -> list[str]:
    def key(g):
        if g == "input":
            return (0, 0)
        if g.startswith("layer "):
            return (1, int(g.split()[1]))
        return (2, 0 if g == "final norm" else 1)
    return sorted(groups, key=key)


def fig_weights(run: Run, out: Path):
    ckpts = load_checkpoints(run)
    if len(ckpts) < 2:
        print("weights: fewer than two checkpoints; skipped")
        return
    steps = [s for s, _ in ckpts]
    xs = [run.hours(s) for s in steps]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    fig, axes = figure(1, 3, w=5.6, h=4.4)
    dist = {}
    for col, net in enumerate(("policy", "value")):
        ax = axes[0][col]
        groups = order_groups(group_norms(ckpts[0][1][net]).keys())
        grid = []
        for g in groups:
            row = []
            for k in range(1, len(ckpts)):
                cur, prev = ckpts[k][1][net], ckpts[k - 1][1][net]
                num_ = group_norms(cur, prev).get(g, 0.0)
                den = group_norms(cur).get(g, 1.0) or 1.0
                row.append(100 * num_ / den)
            grid.append(row)
        im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, interpolation="nearest")
        ax.set_yticks(range(len(groups)), groups, fontsize=8, color=TEXT_2)
        ticks = list(range(len(steps) - 1))
        every = max(1, len(ticks) // 8)
        ax.set_xticks(ticks[::every], [str(steps[k + 1]) for k in ticks[::every]], fontsize=8, color=TEXT_2)
        style(ax, f"{'P' if net == 'policy' else 'V'}: weight change per publish", "", xlabel="learner step of the publish")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("‖ΔW‖ / ‖W‖ vs previous publish, %", fontsize=8, color=TEXT_2)
        cb.ax.tick_params(labelsize=8, colors=TEXT_2, length=0)
        cb.outline.set_visible(False)
        w0 = ckpts[0][1][net]
        base = math.sqrt(sum(float((w * w).sum()) for w in w0.values()))
        dist[net] = [100 * math.sqrt(sum(float(((ck[1][net][n] - w0[n]) ** 2).sum()) for n in w0)) / base for ck in ckpts]
    ax = axes[0][2]
    series(ax, xs, dist["policy"], BLUE, "P", end="{:.1f}%")
    series(ax, xs, dist["value"], ORANGE, "V", end="{:.1f}%")
    style(ax, "Distance from the warm start", "‖W − W₀‖ / ‖W₀‖, %", run=run)
    ax.set_ylim(bottom=0)
    legend(ax)
    save(fig, out / "06_weights.png", f"{run.path.name}: weight evolution")


# ---- figures --------------------------------------------------------------------------------


def fig_overview(run: Run, out: Path):
    fig, a = figure(3, 3)
    p_bench(a[0][0], run, "net-rb2", "Network only vs rule bot 2")
    p_bench(a[0][1], run, "net-rb", "Network only vs rule bot")
    p_bench(a[0][2], run, "search-rb2", "Search vs rule bot 2 (seed 7)", warm=WARM_START_SEARCH_RB2)
    p_heldout(a[1][0], run, "policy", "play_loss", "P play cross-entropy (held out)", "nats")
    p_heldout(a[1][1], run, "policy", "bid_loss", "P bid cross-entropy (held out)", "nats")
    p_heldout(a[1][2], run, "value", "mse", "V MSE (held out)", "MSE of ŝ")
    p_change(a[2][0], run, ("kl_bid_vs_prev", "kl_play_vs_prev"), ("bids", "plays"), "P's change per publish (probe KL)", "nats")
    p_selfplay(a[2][1], run, (lambda r: r["bid"]["top_differs"], lambda r: r["play"]["top_differs"]), ("bids", "plays"),
               "Self-play: search's top move ≠ P's", "% of decisions with a choice", pct=True)
    p_probe_agreement(a[2][2], run)
    save(fig, out / "00_overview.png", f"{run.path.name}: overview")


def fig_strength(run: Run, out: Path):
    fig, a = figure(1, 3, w=5.4, h=4.2)
    p_bench(a[0][0], run, "net-rb2", "Network only vs rule bot 2 (256 deals)")
    p_bench(a[0][1], run, "net-rb", "Network only vs rule bot (128 deals)")
    p_bench(a[0][2], run, "search-rb2", "Search vs rule bot 2 (seed 7, 128 deals)", warm=WARM_START_SEARCH_RB2)
    save(fig, out / "01_strength.png", f"{run.path.name}: strength (95% CI bands over deals)")


def fig_bids(run: Run, out: Path):
    fig, a = figure(2, 2, w=5.6, h=3.9)
    p_bids(a[0][0], run, 0)
    p_bids(a[0][1], run, 1)
    p_bids(a[1][0], run, 2)
    p_zero_bids(a[1][1], run)
    handles, labels = a[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=4, frameon=False, fontsize=9, labelcolor=TEXT_2, bbox_to_anchor=(0.99, 0.955))
    save(fig, out / "02_bids.png", f"{run.path.name}: bids in the network-only benches (solid: P; dashed: its opponents)")


def fig_learning(run: Run, out: Path):
    fig, a = figure(2, 3)
    p_train_loss(a[0][0], run, "policy_loss", "P training loss")
    p_train_loss(a[0][1], run, "value_loss", "V training loss")
    p_agreement_search(a[0][2], run)
    p_change(a[1][0], run, ("kl_bid_vs_prev", "kl_play_vs_prev"), ("bids", "plays"), "P's change per publish (probe KL)", "nats")
    p_change(a[1][1], run, ("v_change_vs_prev",), ("V",), "V's change per publish (probe states)", "mean |Δŝ|")
    p_pace(a[1][2], run)
    save(fig, out / "03_learning.png", f"{run.path.name}: learning")


def fig_generalization(run: Run, out: Path):
    fig, a = figure(2, 3)
    p_heldout(a[0][0], run, "policy", "bid_loss", "P bid cross-entropy", "nats")
    p_heldout(a[0][1], run, "policy", "play_loss", "P play cross-entropy", "nats")
    p_heldout(a[0][2], run, "value", "mse", "V MSE", "MSE of ŝ", ref_key="variance")
    p_heldout(a[1][0], run, "value", "correlation", "V correlation with the outcome", "Pearson r")
    p_probe_agreement(a[1][1], run)
    p_probe_last_trick(a[1][2], run)
    save(fig, out / "04_generalization.png", f"{run.path.name}: held out (top: self-play validation vs training sample; bottom: fixed teacher probe)")


def fig_selfplay(run: Run, out: Path):
    fig, a = figure(2, 3)
    p_selfplay(a[0][0], run, (lambda r: r["bid"]["top_differs"], lambda r: r["play"]["top_differs"]), ("bids", "plays"),
               "Search's top move ≠ P's", "% of decisions with a choice", pct=True)
    p_selfplay(a[0][1], run, (lambda r: r["bid"]["kl_target_prior"], lambda r: r["play"]["kl_target_prior"]), ("bids", "plays"),
               "KL(search target ‖ P)", "nats")
    p_selfplay(a[0][2], run, (lambda r: r["bid"]["off_top_played"],), ("bids",),
               "Bids played off search's top (τ = 1)", "% of bids with a choice", pct=True)
    p_selfplay(a[1][0], run, (lambda r: r["bid"]["target_entropy"], lambda r: r["bid"]["prior_entropy"]), ("search target", "P"),
               "Bid entropy", "nats")
    p_selfplay(a[1][1], run, tuple((lambda i: lambda r: r["made"][i])(i) for i in range(3)), ("1 card", "2–4 cards", "5+ cards"),
               "Self-play bids made", "share", bottom0=False)
    p_selfplay(a[1][2], run, (lambda r: r["examples_per_hour"] / 1000,), ("examples",), "Self-play throughput", "thousand examples per hour")
    save(fig, out / "05_selfplay.png", f"{run.path.name}: self-play (faint: per minute; bold: 10-minute mean)")


def fig_value_margin(run: Run, out: Path):
    has = any(isinstance(r.get("fixed_value"), dict) for r in run.of("held_out")) or any(
        str(r.get("name", "")).startswith("net-rb2-s") for r in run.of("bench"))
    if not has:
        return
    fig, a = figure(2, 2, w=5.8, h=3.9)
    p_margin(a[0][0], run)
    p_fixed_value(a[0][1], run)
    p_value_stream(a[1][0], run)
    p_value_pace(a[1][1], run)
    save(fig, out / "08_value_margin.png", f"{run.path.name}: search's margin and V")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, help="output directory (default <run>/plots)")
    p.add_argument("--no-weights", action="store_true", help="skip 06_weights.png (needs torch)")
    args = p.parse_args()
    run = Run(args.run)
    out = args.out or args.run / "plots"
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_overview, fig_strength, fig_bids, fig_learning, fig_generalization, fig_selfplay, fig_value_margin,
              fig_rollouts, fig_panel):
        f(run, out)
    if not args.no_weights:
        fig_weights(run, out)


if __name__ == "__main__":
    main()
