#!/usr/bin/env python3
"""
Generate visualization and markdown report for Bayesian Optimization autotune results.

Usage:
    python benchmarks/performance/auto_tune_bo_report.py results/auto_tune_bo_20251114_140907
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import cm
    import numpy as np
except ImportError:
    print("ERROR: matplotlib/numpy not installed. Please run: pip install matplotlib numpy")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Please run: pip install pandas")
    sys.exit(1)


# =============================================================================
# Data Loading
# =============================================================================

def load_results(results_dir: Path) -> Dict:
    """
    Load results from experiments.json.

    Args:
        results_dir: Results directory

    Returns:
        Dictionary with parsed results per stage
    """
    experiments_file = results_dir / "experiments.json"
    if not experiments_file.exists():
        print(f"ERROR: {experiments_file} not found")
        sys.exit(1)

    with open(experiments_file, 'r') as f:
        experiments = json.load(f)

    # Group by stage
    stages = defaultdict(lambda: {"trials": [], "validation": []})

    for exp in experiments:
        stage_num = exp["stage_num"]
        if exp["phase"] == "bo_trial":
            stages[stage_num]["trials"].append(exp)
        elif exp["phase"] == "validation":
            stages[stage_num]["validation"].append(exp)

    # Sort trials by trial_num
    for stage_num in stages:
        stages[stage_num]["trials"].sort(key=lambda x: x["trial_num"])

    return dict(stages)


# =============================================================================
# Visualizations
# =============================================================================

def plot_search_space(stage_data: Dict, stage_num: int, output_path: Path):
    """
    Generate 2D scatter/heatmap of search space exploration.

    Args:
        stage_data: Stage trial data
        stage_num: Stage number
        output_path: Output file path
    """
    trials = stage_data["trials"]

    # Extract data
    batch_sizes = [t["parallel_batch_size"] for t in trials]
    timeouts = [t["batch_timeout_ms"] for t in trials]
    perfs = [t["rounds_per_min"] for t in trials]
    trial_types = [t["trial_type"] for t in trials]

    # Find best trial
    best_idx = perfs.index(max(perfs))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot with color-coded performance
    scatter = ax.scatter(
        batch_sizes,
        timeouts,
        c=perfs,
        s=100,
        cmap='viridis',
        edgecolors='black',
        linewidth=0.5,
        alpha=0.7,
        zorder=2,
    )

    # Highlight different trial types
    for i, trial_type in enumerate(trial_types):
        if trial_type == "baseline":
            ax.scatter(batch_sizes[i], timeouts[i], s=300, marker='*',
                      edgecolors='red', facecolors='none', linewidth=2, zorder=3)
        elif trial_type == "random":
            ax.scatter(batch_sizes[i], timeouts[i], s=200, marker='o',
                      edgecolors='blue', facecolors='none', linewidth=1.5, zorder=3)
        elif trial_type == "tpe":
            ax.scatter(batch_sizes[i], timeouts[i], s=200, marker='^',
                      edgecolors='green', facecolors='none', linewidth=1.5, zorder=3)

    # Highlight best trial
    ax.scatter(batch_sizes[best_idx], timeouts[best_idx], s=500, marker='*',
              edgecolors='gold', facecolors='none', linewidth=3, zorder=4)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Rounds/min', rotation=270, labelpad=20, fontsize=12)

    # Labels and title
    ax.set_xlabel('parallel_batch_size', fontsize=12)
    ax.set_ylabel('batch_timeout_ms', fontsize=12)
    ax.set_title(f'Stage {stage_num} Search Space Exploration', fontsize=14, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Legend
    baseline_patch = mpatches.Patch(color='red', label='Baseline')
    random_patch = mpatches.Patch(color='blue', label='Random')
    tpe_patch = mpatches.Patch(color='green', label='TPE')
    best_patch = mpatches.Patch(color='gold', label='Best')
    ax.legend(handles=[baseline_patch, random_patch, tpe_patch, best_patch],
             loc='upper right', fontsize=10)

    # Annotate best
    ax.annotate(
        f'Best: ({batch_sizes[best_idx]}, {timeouts[best_idx]})\n{perfs[best_idx]:.1f} r/min',
        xy=(batch_sizes[best_idx], timeouts[best_idx]),
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black')
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Created: {output_path.name}")


def plot_contour(stage_data: Dict, stage_num: int, output_path: Path):
    """
    Generate contour plot showing performance landscape.

    Args:
        stage_data: Stage trial data
        stage_num: Stage number
        output_path: Output file path
    """
    trials = stage_data["trials"]

    # Extract data
    batch_sizes = np.array([t["parallel_batch_size"] for t in trials])
    timeouts = np.array([t["batch_timeout_ms"] for t in trials])
    perfs = np.array([t["rounds_per_min"] for t in trials])

    # Find best
    best_idx = perfs.argmax()

    # Create grid for interpolation
    batch_grid = np.linspace(batch_sizes.min(), batch_sizes.max(), 50)
    timeout_grid = np.linspace(timeouts.min(), timeouts.max(), 50)
    B, T = np.meshgrid(batch_grid, timeout_grid)

    # Interpolate performance on grid (using scipy if available, otherwise simple nearest)
    try:
        from scipy.interpolate import griddata
        perf_grid = griddata(
            (batch_sizes, timeouts), perfs, (B, T), method='cubic', fill_value=perfs.min()
        )
    except ImportError:
        # Simple fallback: just show scatter
        perf_grid = None

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Contour plot if interpolation succeeded
    if perf_grid is not None:
        contour = ax.contourf(B, T, perf_grid, levels=15, cmap='viridis', alpha=0.6)
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Rounds/min (interpolated)', rotation=270, labelpad=20, fontsize=12)

    # Scatter actual trials on top
    scatter = ax.scatter(
        batch_sizes,
        timeouts,
        c=perfs,
        s=150,
        cmap='viridis',
        edgecolors='black',
        linewidth=1,
        zorder=3,
    )

    # Highlight best
    ax.scatter(
        batch_sizes[best_idx],
        timeouts[best_idx],
        s=500,
        marker='*',
        edgecolors='gold',
        facecolors='yellow',
        linewidth=3,
        zorder=4,
    )

    # Labels and title
    ax.set_xlabel('parallel_batch_size', fontsize=12)
    ax.set_ylabel('batch_timeout_ms', fontsize=12)
    ax.set_title(f'Stage {stage_num} Performance Landscape', fontsize=14, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', zorder=1)

    # Annotate best
    ax.annotate(
        f'Optimum\n({batch_sizes[best_idx]}, {timeouts[best_idx]})',
        xy=(batch_sizes[best_idx], timeouts[best_idx]),
        xytext=(15, 15),
        textcoords='offset points',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='black', lw=2)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Created: {output_path.name}")


def plot_convergence(stage_data: Dict, stage_num: int, n_startup: int, output_path: Path):
    """
    Generate convergence plot showing best-so-far progression.

    Args:
        stage_data: Stage trial data
        stage_num: Stage number
        n_startup: Number of startup trials (for marking TPE start)
        output_path: Output file path
    """
    trials = stage_data["trials"]

    # Extract data
    trial_nums = [t["trial_num"] for t in trials]
    perfs = [t["rounds_per_min"] for t in trials]

    # Calculate best-so-far
    best_so_far = []
    current_best = 0
    for perf in perfs:
        current_best = max(current_best, perf)
        best_so_far.append(current_best)

    # Find when best was found
    final_best = max(perfs)
    best_found_at = perfs.index(final_best)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot best-so-far line
    ax.plot(trial_nums, best_so_far, marker='o', linewidth=2, markersize=8,
           color='darkblue', label='Best so far', zorder=3)

    # Plot all trials as scatter
    ax.scatter(trial_nums, perfs, s=100, alpha=0.5, color='lightblue',
              edgecolors='black', linewidth=0.5, label='Trial result', zorder=2)

    # Highlight baseline
    ax.scatter(0, perfs[0], s=300, marker='*', color='red',
              edgecolors='darkred', linewidth=2, label='Baseline', zorder=4)

    # Highlight best trial
    ax.scatter(best_found_at, final_best, s=400, marker='*', color='gold',
              edgecolors='orange', linewidth=2, label='Best trial', zorder=5)

    # Mark TPE start
    ax.axvline(x=n_startup, color='green', linestyle='--', linewidth=2,
              alpha=0.7, label=f'TPE starts (trial {n_startup})', zorder=1)

    # Shade regions
    ax.axvspan(0, n_startup, alpha=0.1, color='blue', label='Random sampling')
    ax.axvspan(n_startup, len(trials)-1, alpha=0.1, color='green', label='TPE optimization')

    # Labels and title
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Rounds/min', fontsize=12)
    ax.set_title(f'Stage {stage_num} Convergence', fontsize=14, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)

    # Legend
    ax.legend(loc='lower right', fontsize=10)

    # Annotate improvement
    baseline_perf = perfs[0]
    improvement = ((final_best - baseline_perf) / baseline_perf * 100) if baseline_perf > 0 else 0
    ax.text(
        0.02, 0.98,
        f'Baseline: {baseline_perf:.1f} r/min\n'
        f'Best: {final_best:.1f} r/min\n'
        f'Improvement: {improvement:+.1f}%\n'
        f'Found at trial: {best_found_at}',
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Created: {output_path.name}")


def plot_multi_stage_summary(all_stages: Dict, output_path: Path):
    """
    Generate multi-panel comparison across all stages.

    Args:
        all_stages: All stage data
        output_path: Output file path
    """
    stage_nums = sorted(all_stages.keys())

    # Extract best configs per stage
    best_perfs = []
    best_batch_sizes = []
    best_timeouts = []
    baseline_perfs = []
    stage_labels = []

    for stage_num in stage_nums:
        trials = all_stages[stage_num]["trials"]
        perfs = [t["rounds_per_min"] for t in trials]

        best_idx = perfs.index(max(perfs))
        best_perfs.append(perfs[best_idx])
        best_batch_sizes.append(trials[best_idx]["parallel_batch_size"])
        best_timeouts.append(trials[best_idx]["batch_timeout_ms"])
        baseline_perfs.append(perfs[0])  # First trial is baseline
        stage_labels.append(trials[0]["stage_mcts"])

    # Create figure with 3 panels
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Panel 1: Best performance per stage
    ax = axes[0]
    x = np.arange(len(stage_nums))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_perfs, width, label='Baseline', color='lightcoral', edgecolor='black')
    bars2 = ax.bar(x + width/2, best_perfs, width, label='Best (Optimized)', color='lightgreen', edgecolor='black')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}',
                   ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Rounds/min', fontsize=12)
    ax.set_title('Performance Across Stages', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Panel 2: Optimal batch_size vs curriculum
    ax = axes[1]
    ax.plot(stage_nums, best_batch_sizes, marker='o', linewidth=2, markersize=10,
           color='blue', label='Optimal parallel_batch_size')
    ax.axhline(y=30, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Baseline (30)')

    # Annotate values
    for i, (stage, val) in enumerate(zip(stage_nums, best_batch_sizes)):
        ax.text(stage, val, f'{val}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Stage', fontsize=12)
    ax.set_ylabel('parallel_batch_size', fontsize=12)
    ax.set_title('Optimal Batch Size vs Curriculum Difficulty', fontsize=14, fontweight='bold')
    ax.set_xticks(stage_nums)
    ax.set_xticklabels(stage_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Panel 3: Optimal timeout vs curriculum
    ax = axes[2]
    ax.plot(stage_nums, best_timeouts, marker='s', linewidth=2, markersize=10,
           color='purple', label='Optimal batch_timeout_ms')
    ax.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Baseline (10ms)')

    # Annotate values
    for i, (stage, val) in enumerate(zip(stage_nums, best_timeouts)):
        ax.text(stage, val, f'{val}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Stage', fontsize=12)
    ax.set_ylabel('batch_timeout_ms', fontsize=12)
    ax.set_title('Optimal Timeout vs Curriculum Difficulty', fontsize=14, fontweight='bold')
    ax.set_xticks(stage_nums)
    ax.set_xticklabels(stage_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Created: {output_path.name}")


def plot_parameter_importance(all_stages: Dict, output_path: Path):
    """
    Analyze parameter importance across stages.

    Args:
        all_stages: All stage data
        output_path: Output file path
    """
    # For each stage, calculate correlation of each parameter with performance
    stage_nums = sorted(all_stages.keys())

    batch_size_importance = []
    timeout_importance = []

    for stage_num in stage_nums:
        trials = all_stages[stage_num]["trials"]

        batch_sizes = [t["parallel_batch_size"] for t in trials]
        timeouts = [t["batch_timeout_ms"] for t in trials]
        perfs = [t["rounds_per_min"] for t in trials]

        # Calculate correlation (simple)
        batch_corr = np.corrcoef(batch_sizes, perfs)[0, 1]
        timeout_corr = np.corrcoef(timeouts, perfs)[0, 1]

        batch_size_importance.append(abs(batch_corr))
        timeout_importance.append(abs(timeout_corr))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(stage_nums))
    width = 0.35

    bars1 = ax.bar(x - width/2, batch_size_importance, width,
                   label='parallel_batch_size', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, timeout_importance, width,
                   label='batch_timeout_ms', color='coral', edgecolor='black')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Stage', fontsize=12)
    ax.set_ylabel('Correlation with Performance (absolute)', fontsize=12)
    ax.set_title('Parameter Importance Across Stages', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([all_stages[s]["trials"][0]["stage_mcts"] for s in stage_nums])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Created: {output_path.name}")


# =============================================================================
# Report Generation
# =============================================================================

def generate_markdown_report(
    all_stages: Dict,
    viz_dir: Path,
    output_path: Path,
    n_startup: int,
    total_runtime_min: float
):
    """
    Generate comprehensive markdown report with embedded images.

    Args:
        all_stages: All stage data
        viz_dir: Directory containing visualization PNGs
        output_path: Output markdown file path
        n_startup: Number of startup trials
        total_runtime_min: Total runtime in minutes
    """
    stage_nums = sorted(all_stages.keys())

    # Calculate overall stats
    total_trials = sum(len(all_stages[s]["trials"]) for s in stage_nums)
    total_stages = len(stage_nums)

    # Find best overall config
    best_overall_perf = 0
    best_overall_stage = None
    for stage_num in stage_nums:
        trials = all_stages[stage_num]["trials"]
        perfs = [t["rounds_per_min"] for t in trials]
        stage_best = max(perfs)
        if stage_best > best_overall_perf:
            best_overall_perf = stage_best
            best_overall_stage = stage_num

    best_overall_trial = all_stages[best_overall_stage]["trials"][
        [t["rounds_per_min"] for t in all_stages[best_overall_stage]["trials"]].index(best_overall_perf)
    ]

    # Start writing report
    lines = []
    lines.append("# Bayesian Optimization Autotune Report (Optuna TPE)\n")
    lines.append("## Executive Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Stages | {total_stages} |")
    lines.append(f"| Trials per Stage | {total_trials // total_stages} (1 baseline + {n_startup-1} random + {total_trials // total_stages - n_startup} TPE) |")
    lines.append(f"| Total Evaluations | {total_trials} |")
    lines.append(f"| Runtime | {total_runtime_min:.1f} minutes |")
    lines.append(f"| Optimization Method | Optuna TPE Sampler |")
    lines.append(f"| Best Overall Config | Stage {best_overall_stage}: {best_overall_trial['parallel_batch_size']} batch / {best_overall_trial['batch_timeout_ms']}ms timeout |")
    lines.append("")

    # Per-stage sections
    for stage_num in stage_nums:
        stage_data = all_stages[stage_num]
        trials = stage_data["trials"]
        validation = stage_data["validation"]

        # Get stage info
        stage_mcts = trials[0]["stage_mcts"]
        perfs = [t["rounds_per_min"] for t in trials]
        best_idx = perfs.index(max(perfs))
        best_trial = trials[best_idx]
        baseline_trial = trials[0]

        best_perf = best_trial["rounds_per_min"]
        baseline_perf = baseline_trial["rounds_per_min"]
        improvement = ((best_perf - baseline_perf) / baseline_perf * 100) if baseline_perf > 0 else 0

        lines.append("---\n")
        lines.append(f"## Stage {stage_num}: {stage_mcts} MCTS\n")

        # Best config
        lines.append("### Best Configuration")
        lines.append("```python")
        lines.append("{")
        lines.append(f"    'workers': {best_trial['workers']},")
        lines.append(f"    'parallel_batch_size': {best_trial['parallel_batch_size']},  # ← {((best_trial['parallel_batch_size'] - 30) / 30 * 100):+.0f}% vs baseline")
        lines.append(f"    'batch_timeout_ms': {best_trial['batch_timeout_ms']},      # ← {((best_trial['batch_timeout_ms'] - 10) / 10 * 100):+.0f}% vs baseline")
        lines.append(f"    'num_determinizations': {best_trial['num_determinizations']},")
        lines.append(f"    'simulations_per_det': {best_trial['simulations_per_det']}")
        lines.append("}")
        lines.append("```\n")

        # Performance
        lines.append("**Performance:**")
        lines.append(f"- **Best**: {best_perf:.1f} r/min")
        lines.append(f"- **Baseline**: {baseline_perf:.1f} r/min")
        lines.append(f"- **Improvement**: {improvement:+.1f}%\n")

        # Validation
        if validation:
            val_perfs = [v["rounds_per_min"] for v in validation]
            val_mean = np.mean(val_perfs)
            val_std = np.std(val_perfs)
            val_std_pct = (val_std / val_mean * 100) if val_mean > 0 else 0

            lines.append(f"**Validation ({len(validation)} runs):**")
            lines.append(f"- Mean: {val_mean:.1f} r/min")
            lines.append(f"- Std Dev: {val_std:.1f} r/min (±{val_std_pct:.1f}%)")
            lines.append(f"- Range: [{min(val_perfs):.1f}, {max(val_perfs):.1f}]\n")

        # Visualizations
        lines.append("---\n")
        lines.append("### Optimization Progress\n")
        lines.append(f"![Search Space Exploration](stage_{stage_num}_search_space.png)")
        lines.append(f"*{len(trials)} trials exploring the 2D parameter space. Baseline (red star) vs best found (gold star).*\n")
        lines.append(f"![Contour Plot](stage_{stage_num}_contour.png)")
        lines.append(f"*Performance landscape showing optimum region.*\n")
        lines.append(f"![Convergence](stage_{stage_num}_convergence.png)")
        lines.append(f"*Best-so-far progression: TPE found optimum within {best_trial['trial_num']} trials.*\n")

        # Trial summary table
        lines.append("---\n")
        lines.append("### Trial Summary\n")
        lines.append("| Trial | Type | Batch Size | Timeout (ms) | Rounds/min | Improvement |")
        lines.append("|-------|------|------------|--------------|------------|-------------|")

        # Show first 5, last 5, and best
        trials_to_show = []
        if len(trials) <= 12:
            trials_to_show = trials
        else:
            trials_to_show = trials[:5] + [None] + trials[-5:]
            if best_idx >= 5 and best_idx < len(trials) - 5:
                trials_to_show.insert(6, trials[best_idx])

        for t in trials_to_show:
            if t is None:
                lines.append("| ... | ... | ... | ... | ... | ... |")
                continue

            trial_num = t["trial_num"]
            trial_type = t["trial_type"].capitalize()
            batch_size = t["parallel_batch_size"]
            timeout = t["batch_timeout_ms"]
            perf = t["rounds_per_min"]
            imp = t["improvement_vs_baseline"]

            marker = " ⭐" if trial_num == best_idx else ""
            lines.append(f"| {trial_num} | {trial_type} | {batch_size} | {timeout} | **{perf:.1f}**{marker} | {imp:+.1f}% |")

        lines.append("")

        # Key findings
        lines.append("**Key Findings:**")

        # When was best found
        if best_trial["trial_type"] == "baseline":
            lines.append(f"- Baseline was optimal (no improvement found)")
        elif best_trial["trial_type"] == "tpe":
            tpe_iteration = best_trial["trial_num"] - n_startup
            lines.append(f"- TPE converged to optimum by trial {best_trial['trial_num']} ({tpe_iteration} TPE iterations)")
        else:
            lines.append(f"- Best found during random sampling (trial {best_trial['trial_num']})")

        # Parameter insights
        batch_change = best_trial["parallel_batch_size"] - 30
        if abs(batch_change) >= 5:
            direction = "higher" if batch_change > 0 else "lower"
            lines.append(f"- Optimal batch_size significantly {direction} than baseline ({best_trial['parallel_batch_size']} vs 30)")
        else:
            lines.append(f"- Optimal batch_size close to baseline ({best_trial['parallel_batch_size']} vs 30)")

        timeout_change = best_trial["batch_timeout_ms"] - 10
        if abs(timeout_change) >= 3:
            direction = "higher" if timeout_change > 0 else "lower"
            lines.append(f"- timeout_ms {direction} than baseline ({best_trial['batch_timeout_ms']}ms vs 10ms)")
        else:
            lines.append(f"- timeout_ms close to baseline ({best_trial['batch_timeout_ms']}ms vs 10ms)")

        lines.append("")

    # Cross-stage analysis
    lines.append("---\n")
    lines.append("## Cross-Stage Analysis\n")
    lines.append("![Multi-Stage Summary](all_stages_summary.png)")
    lines.append("*Performance and optimal parameters across all curriculum stages.*\n")
    lines.append("![Parameter Importance](parameter_importance.png)")
    lines.append("*Correlation of parameters with performance across stages.*\n")

    # Trends
    lines.append("### Trends Observed\n")

    # Extract trend data
    best_batch_sizes = []
    best_timeouts = []
    for stage_num in stage_nums:
        trials = all_stages[stage_num]["trials"]
        perfs = [t["rounds_per_min"] for t in trials]
        best_idx = perfs.index(max(perfs))
        best_batch_sizes.append(trials[best_idx]["parallel_batch_size"])
        best_timeouts.append(trials[best_idx]["batch_timeout_ms"])

    lines.append("1. **Batch Size vs Difficulty**:")
    if best_batch_sizes[0] > best_batch_sizes[-1]:
        lines.append(f"   - Early stages ({all_stages[1]['trials'][0]['stage_mcts']}): Batch size = {best_batch_sizes[0]}")
        lines.append(f"   - Late stages ({all_stages[stage_nums[-1]]['trials'][0]['stage_mcts']}): Batch size = {best_batch_sizes[-1]}")
        lines.append("   - Trend: Lower MCTS load allows bigger batches\n")
    else:
        lines.append("   - Batch size relatively stable across stages\n")

    lines.append("2. **Timeout Sensitivity**:")
    timeout_range = max(best_timeouts) - min(best_timeouts)
    if timeout_range <= 5:
        lines.append(f"   - Relatively stable across stages ({min(best_timeouts)}-{max(best_timeouts)}ms)")
        lines.append("   - Less impact on performance than batch size\n")
    else:
        lines.append(f"   - Varies significantly across stages ({min(best_timeouts)}-{max(best_timeouts)}ms)")
        lines.append("   - Stage-specific tuning important\n")

    lines.append("3. **Optimization Efficiency**:")
    avg_trials_to_best = np.mean([
        [t["rounds_per_min"] for t in all_stages[s]["trials"]].index(
            max([t["rounds_per_min"] for t in all_stages[s]["trials"]])
        )
        for s in stage_nums
    ])
    success_rate = sum(
        1 for s in stage_nums
        if all_stages[s]["trials"][[t["rounds_per_min"] for t in all_stages[s]["trials"]].index(
            max([t["rounds_per_min"] for t in all_stages[s]["trials"]])
        )]["improvement_vs_baseline"] > 2.0
    ) / len(stage_nums) * 100

    lines.append(f"   - Avg trials to optimum: {avg_trials_to_best:.1f} trials (out of {total_trials // total_stages})")
    lines.append(f"   - TPE effective: {success_rate:.0f}% success rate finding >2% improvement\n")

    # Recommended configs
    lines.append("---\n")
    lines.append("## Recommended Configurations\n")
    lines.append("### For TrainingConfig (Python)")
    lines.append("```python")
    lines.append("# Auto-tuned optimal configs per curriculum stage")
    lines.append("CURRICULUM_OPTIMAL_CONFIGS = {")
    for stage_num in stage_nums:
        trials = all_stages[stage_num]["trials"]
        perfs = [t["rounds_per_min"] for t in trials]
        best_idx = perfs.index(max(perfs))
        best_trial = trials[best_idx]

        det = best_trial["num_determinizations"]
        sims = best_trial["simulations_per_det"]
        batch = best_trial["parallel_batch_size"]
        timeout = best_trial["batch_timeout_ms"]

        lines.append(f"    ({det}, {sims}): {{'parallel_batch_size': {batch}, 'batch_timeout_ms': {timeout}}},")

    lines.append("}\n")
    lines.append("# To use in training:")
    lines.append("def get_optimal_batch_params(det, sims):")
    lines.append("    return CURRICULUM_OPTIMAL_CONFIGS.get((det, sims),")
    lines.append("           {'parallel_batch_size': 30, 'batch_timeout_ms': 10})")
    lines.append("```\n")

    # Methodology
    lines.append("---\n")
    lines.append("## Methodology\n")
    lines.append("**Bayesian Optimization Setup:**")
    lines.append("- Framework: Optuna 3.x")
    lines.append("- Sampler: TPE (Tree-structured Parzen Estimator)")
    lines.append("- Search Space:")
    lines.append("  - `parallel_batch_size`: [5, 100] (integer)")
    lines.append("  - `batch_timeout_ms`: [1, 50] (integer)")
    lines.append(f"- Trials per Stage: {total_trials // total_stages} (1 baseline + {n_startup-1} random startup + {total_trials // total_stages - n_startup} TPE)")
    lines.append("- Evaluation: 200 rounds per trial")
    lines.append("- Validation: 3 runs × 200 rounds for best config\n")
    lines.append("**TPE Configuration:**")
    lines.append(f"- `n_startup_trials={n_startup}`: Random exploration before TPE")
    lines.append("- `seed=42`: Reproducible results")
    lines.append("- Direction: Maximize rounds/min\n")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n  Created: {output_path.name}")


def generate_csv_summary(all_stages: Dict, output_path: Path):
    """
    Generate flat CSV summary of all trials.

    Args:
        all_stages: All stage data
        output_path: Output CSV path
    """
    rows = []
    for stage_num in sorted(all_stages.keys()):
        trials = all_stages[stage_num]["trials"]
        for trial in trials:
            rows.append({
                "stage_num": trial["stage_num"],
                "stage_mcts": trial["stage_mcts"],
                "trial_num": trial["trial_num"],
                "trial_type": trial["trial_type"],
                "parallel_batch_size": trial["parallel_batch_size"],
                "batch_timeout_ms": trial["batch_timeout_ms"],
                "rounds_per_min": trial["rounds_per_min"],
                "is_best_so_far": trial["is_best_so_far"],
                "improvement_vs_baseline": trial["improvement_vs_baseline"],
                "optuna_state": trial["optuna_state"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Created: {output_path.name}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate BO autotune report")
    parser.add_argument("results_dir", type=str, help="Results directory")
    parser.add_argument("--n-startup", type=int, default=5, help="Number of startup trials (for marking)")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    print(f"Loading results from: {results_dir}")

    # Load data
    all_stages = load_results(results_dir)
    print(f"Loaded {len(all_stages)} stages")

    # Calculate total runtime (approximation from timestamps)
    experiments_file = results_dir / "experiments.json"
    with open(experiments_file, 'r') as f:
        experiments = json.load(f)

    if experiments:
        first_time = experiments[0]["timestamp"]
        last_time = experiments[-1]["timestamp"]
        from datetime import datetime
        start = datetime.fromisoformat(first_time)
        end = datetime.fromisoformat(last_time)
        total_runtime_min = (end - start).total_seconds() / 60
    else:
        total_runtime_min = 0

    # Generate visualizations
    print("\nGenerating visualizations...")

    for stage_num in sorted(all_stages.keys()):
        stage_data = all_stages[stage_num]

        print(f"\nStage {stage_num}:")
        plot_search_space(stage_data, stage_num, results_dir / f"stage_{stage_num}_search_space.png")
        plot_contour(stage_data, stage_num, results_dir / f"stage_{stage_num}_contour.png")
        plot_convergence(stage_data, stage_num, args.n_startup, results_dir / f"stage_{stage_num}_convergence.png")

    print(f"\nOverall:")
    plot_multi_stage_summary(all_stages, results_dir / "all_stages_summary.png")
    plot_parameter_importance(all_stages, results_dir / "parameter_importance.png")

    # Generate reports
    print("\nGenerating reports...")
    generate_markdown_report(
        all_stages,
        results_dir,
        results_dir / "bo_report.md",
        args.n_startup,
        total_runtime_min
    )
    generate_csv_summary(all_stages, results_dir / "bo_summary.csv")

    print(f"\n{'='*80}")
    print(f"REPORT GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Files created in: {results_dir}")
    print(f"  - bo_report.md (markdown with embedded images)")
    print(f"  - bo_summary.csv (flat export)")
    print(f"  - 17 PNG visualizations")


if __name__ == "__main__":
    main()
