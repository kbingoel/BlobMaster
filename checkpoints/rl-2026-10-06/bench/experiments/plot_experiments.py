"""One-off: the step-7600 search experiments of rl-2026-10-06 (seed 7, 128 deals, vs rule bot 2)."""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RUN = Path("checkpoints/rl-2026-10-06")
E = RUN / "bench" / "experiments"
WARM = Path("checkpoints/pretrain-2026-10-06/bench4b/search-rb2-128-seed7.csv")
BASE = RUN / "bench" / "search-rb2-step-007600.csv"
NET = E / "net-s7-7600.csv"


def rd(p):
    lines = p.read_text().splitlines()
    return {int(a): float(b) for a, b in (l.split(",") for l in lines[2:])}


def mean_ci(xs):
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, 1.96 * math.sqrt(v / n)


def paired(a, b):
    xa, xb = rd(a), rd(b)
    return mean_ci([xa[k] - xb[k] for k in xa if k in xb])


rows = [
    ("warm start, search c 0.2", WARM),
    ("step 7600, network only", NET),
    ("step 7600, search c 0.2", BASE),
    ("step 7600, search c 0.5", E / "search-c0.5-step7600.csv"),
    ("step 7600, search c 1.0", E / "search-c1.0-step7600.csv"),
    ("step 7600, search c 2.0", E / "search-c2.0-step7600.csv"),
    ("step 7600, search c 1.0, plays 5×200", E / "search-c1.0-plays5x200-step7600.csv"),
    ("step 7600 P + warm-start V, c 0.2", E / "search-c0.2-hybrid-p7600-v0.csv"),
]
rows = [(l, p) for l, p in rows if p.is_file()]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 0.55 * len(rows) + 1.6), facecolor=SURFACE, sharey=True)
ys = list(range(len(rows)))[::-1]
for ax in (a1, a2):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
for y, (label, p) in zip(ys, rows):
    m, c = mean_ci(list(rd(p).values()))
    color = TEXT_2 if "warm" in label and "P +" not in label else (ORANGE if "network" in label else BLUE)
    a1.errorbar(m, y, xerr=c, fmt="o", color=color, markersize=6, capsize=3, linewidth=1.6)
    a1.annotate(f"{m:+.1f}", (m + c, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=TEXT_2)
    if p != NET:
        pm, pc = paired(p, NET)
        a2.errorbar(pm, y, xerr=pc, fmt="o", color=color, markersize=6, capsize=3, linewidth=1.6)
        a2.annotate(f"{pm:+.1f}", (pm + pc, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=TEXT_2)
a1.set_yticks(ys, [l for l, _ in rows], fontsize=9, color=TEXT)
a1.axvline(0, color=TEXT_2, linewidth=0.9)
a2.axvline(0, color=TEXT_2, linewidth=0.9)
a1.set_title("Points/game vs four rule bot 2s (95% CI over deals)", loc="left", fontsize=10, color=TEXT)
a2.set_title("Minus step-7600 network only, paired on the same deals", loc="left", fontsize=10, color=TEXT)
a1.set_xlabel("points/game vs opponents' mean", fontsize=8, color=TEXT_2)
a2.set_xlabel("difference, points/game (> 0: search beats P alone)", fontsize=8, color=TEXT_2)
fig.suptitle("rl-2026-10-06: search settings on the step-7600 networks (seed 7, 128 deals)", x=0.01, ha="left", fontsize=12, color=TEXT)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = RUN / "plots" / "07_search_experiments.png"
fig.savefig(out, dpi=110, facecolor=SURFACE)
print("wrote", out)
