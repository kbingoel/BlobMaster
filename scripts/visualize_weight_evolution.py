"""Visualize weight evolution across training checkpoints.

Usage:
    python scripts/visualize_weight_evolution.py \
        --checkpoint-dir checkpoints/sweep-2026-04-28-anchor \
        --out-dir logs/weight-evolution-sweep-2026-04-28-anchor
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnx
from onnx import numpy_helper

ITER_DIR_RE = re.compile(r"^iter_(\d{6})$")


def find_iter_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in sorted(ckpt_dir.iterdir()):
        m = ITER_DIR_RE.match(p.name)
        if not m:
            continue
        onnx_path = p / "model.onnx"
        if onnx_path.exists():
            out.append((int(m.group(1)), onnx_path))
    out.sort(key=lambda x: x[0])
    return out


def load_weights(onnx_path: Path) -> dict[str, np.ndarray]:
    m = onnx.load(str(onnx_path))
    return {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}


def is_weight_matrix(name: str, arr: np.ndarray) -> bool:
    if arr.ndim < 2:
        return False
    if name.endswith(".bias"):
        return False
    return True


def group_label(name: str) -> str:
    """Drop trailing .weight / .bias for grouping; pass through onnx::MatMul names."""
    if name.endswith(".weight") or name.endswith(".bias"):
        return name.rsplit(".", 1)[0]
    return name


def cos_distance(a: np.ndarray, b: np.ndarray) -> float:
    af = a.ravel().astype(np.float64)
    bf = b.ravel().astype(np.float64)
    na = np.linalg.norm(af)
    nb = np.linalg.norm(bf)
    if na == 0 or nb == 0:
        return 0.0
    return 1.0 - float(np.dot(af, bf) / (na * nb))


def plot_velocity(iters, weights_per_iter, weight_names, out_path):
    """||W_t - W_{t-1}|| / ||W_t||  (across consecutive sampled checkpoints)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    xs = iters[1:]
    for name in weight_names:
        ys = []
        for k in range(1, len(iters)):
            wt = weights_per_iter[k][name].astype(np.float64)
            wp = weights_per_iter[k - 1][name].astype(np.float64)
            denom = np.linalg.norm(wt.ravel())
            if denom == 0:
                ys.append(0.0)
            else:
                ys.append(float(np.linalg.norm((wt - wp).ravel()) / denom))
        ax.plot(xs, ys, alpha=0.4, linewidth=0.9)
    ax.set_xlabel("iter")
    ax.set_ylabel(r"$\|W_t - W_{t-\Delta}\|_2 \;/\; \|W_t\|_2$")
    ax.set_title("Per-layer relative weight velocity (one line per weight tensor)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_distance_from_init(iters, weights_per_iter, weight_names, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axL, axC = axes
    w0 = weights_per_iter[0]
    for name in weight_names:
        l2s, coss = [], []
        denom0 = np.linalg.norm(w0[name].ravel().astype(np.float64))
        denom0 = denom0 if denom0 > 0 else 1.0
        for k in range(len(iters)):
            wt = weights_per_iter[k][name].astype(np.float64)
            l2s.append(float(np.linalg.norm((wt - w0[name]).ravel()) / denom0))
            coss.append(cos_distance(wt, w0[name]))
        axL.plot(iters, l2s, alpha=0.4, linewidth=0.9)
        axC.plot(iters, coss, alpha=0.4, linewidth=0.9)
    axL.set_xlabel("iter"); axL.set_ylabel(r"$\|W_t - W_0\|_2 / \|W_0\|_2$")
    axL.set_title("Relative L2 distance from init")
    axL.grid(True, alpha=0.3)
    axC.set_xlabel("iter"); axC.set_ylabel(r"$1 - \cos(W_t, W_0)$")
    axC.set_title("Cosine distance from init")
    axC.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def pick_representative_layers(weight_names, max_n=12):
    """Pick a spread: input layers, a few transformer layers, heads."""
    picked = []
    seen_groups = set()
    priority_substrings = [
        "input.",
        "transformer.layers.0.",
        "transformer.layers.1.",
        "transformer.layers.2.",
        "transformer.layers.3.",
        "head", "policy", "value",
    ]
    for sub in priority_substrings:
        for name in weight_names:
            if sub in name and group_label(name) not in seen_groups:
                picked.append(name)
                seen_groups.add(group_label(name))
                if len(picked) >= max_n:
                    return picked
                break
    for name in weight_names:
        if group_label(name) in seen_groups:
            continue
        picked.append(name)
        seen_groups.add(group_label(name))
        if len(picked) >= max_n:
            break
    return picked


def plot_weight_histograms(iters, weights_per_iter, picked, out_path):
    n = len(picked)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap("viridis")
    for i, name in enumerate(picked):
        ax = axes[i]
        all_vals = np.concatenate(
            [weights_per_iter[k][name].ravel().astype(np.float64) for k in range(len(iters))]
        )
        lo, hi = np.percentile(all_vals, [0.5, 99.5])
        bins = np.linspace(lo, hi, 81)
        for k, it in enumerate(iters):
            vals = weights_per_iter[k][name].ravel().astype(np.float64)
            color = cmap(k / max(len(iters) - 1, 1))
            ax.hist(vals, bins=bins, histtype="step", linewidth=1.1,
                    color=color, label=f"iter {it}", density=True)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Weight value distributions across training (darker → earlier, brighter → later)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_singular_values(iters, weights_per_iter, picked, out_path, top_k=8):
    n = len(picked)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, name in enumerate(picked):
        ax = axes[i]
        spectra = []
        for k in range(len(iters)):
            W = weights_per_iter[k][name].astype(np.float64)
            if W.ndim > 2:
                W = W.reshape(W.shape[0], -1)
            try:
                s = np.linalg.svd(W, compute_uv=False)
            except np.linalg.LinAlgError:
                s = np.zeros(min(W.shape))
            spectra.append(s)
        K = min(top_k, min(len(s) for s in spectra))
        for j in range(K):
            ys = [spectra[k][j] for k in range(len(iters))]
            ax.plot(iters, ys, linewidth=1.0, alpha=0.85, label=f"σ{j+1}" if i == 0 else None)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7, ncol=2)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Top-{top_k} singular values per weight matrix vs iter", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_first_vs_last_heatmap(iters, weights_per_iter, picked, out_path):
    n = len(picked)
    fig, axes = plt.subplots(n, 2, figsize=(8, 2.6 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    for i, name in enumerate(picked):
        W0 = weights_per_iter[0][name]
        Wf = weights_per_iter[-1][name]
        if W0.ndim > 2:
            W0 = W0.reshape(W0.shape[0], -1)
            Wf = Wf.reshape(Wf.shape[0], -1)
        vmax = max(np.abs(W0).max(), np.abs(Wf).max())
        for ax, W, label in [(axes[i, 0], W0, f"iter {iters[0]}"),
                             (axes[i, 1], Wf, f"iter {iters[-1]}")]:
            im = ax.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_title(f"{name} — {label}", fontsize=8)
            ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=axes[i, 1], fraction=0.04)
    fig.suptitle("Weight matrix snapshot: init vs final", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-layers-histogram", type=int, default=12)
    args = ap.parse_args()

    ckpts = find_iter_checkpoints(args.checkpoint_dir)
    if not ckpts:
        raise SystemExit(f"no iter_NNNNNN/model.onnx found in {args.checkpoint_dir}")
    iters = [it for it, _ in ckpts]
    print(f"checkpoints: {iters}")

    weights_per_iter = []
    for it, p in ckpts:
        ws = load_weights(p)
        weights_per_iter.append(ws)
        print(f"  iter {it}: {len(ws)} initializers loaded from {p.name}")

    base_names = list(weights_per_iter[0].keys())
    weight_names = [n for n in base_names
                    if is_weight_matrix(n, weights_per_iter[0][n])]
    print(f"weight tensors (≥2D, non-bias): {len(weight_names)}")

    picked = pick_representative_layers(weight_names, max_n=args.max_layers_histogram)
    print("representative layers for histogram/SVD/heatmap:")
    for n in picked:
        print("  -", n, weights_per_iter[0][n].shape)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("plotting velocity ...")
    plot_velocity(iters, weights_per_iter, weight_names,
                  args.out_dir / "01_velocity.png")
    print("plotting distance/cosine from init ...")
    plot_distance_from_init(iters, weights_per_iter, weight_names,
                            args.out_dir / "02_distance_from_init.png")
    print("plotting weight histograms ...")
    plot_weight_histograms(iters, weights_per_iter, picked,
                           args.out_dir / "03_weight_histograms.png")
    print("plotting singular value spectra ...")
    plot_singular_values(iters, weights_per_iter, picked,
                         args.out_dir / "04_singular_values.png")
    print("plotting init-vs-final heatmaps ...")
    plot_first_vs_last_heatmap(iters, weights_per_iter, picked[:6],
                               args.out_dir / "05_init_vs_final_heatmap.png")
    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()
