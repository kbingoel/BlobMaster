"""Visualize win-rate trajectory from a sweep's strength.csv.

Usage:
    python scripts/visualize_strength.py \
        --csv checkpoints/sweep-2026-04-28-anchor/strength.csv \
        --out-dir logs/strength-sweep-2026-04-28-anchor
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
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
    ax.set_xlabel("iteration")
    ax.set_ylabel("win rate (95% CI)")
    ax.set_title("Win rate vs anchor opponents")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.csv)
    plot_winrate(rows, args.out_dir / "01_winrate.png")
    plot_score_differential(rows, args.out_dir / "02_score_differential.png")
    plot_losses(rows, args.out_dir / "03_train_losses.png")
    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()
