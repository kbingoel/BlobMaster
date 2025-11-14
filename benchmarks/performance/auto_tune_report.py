#!/usr/bin/env python3
"""
Generate comprehensive report from auto-tune sweep results

This script reads the SQLite database from auto_tune.py and generates:
- Markdown summary report with tables and recommendations
- Plots for visualization (worker scaling, parameter sweeps, heatmaps)
- Exportable config for ml/config.py

Usage:
    python auto_tune_report.py results/auto_tune_20251114_153000/auto_tune_results.db
    python auto_tune_report.py results.db --output custom_report.md
"""

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass
class SweepConfig:
    """Configuration for a single sweep test"""
    workers: int
    parallel_batch_size: int
    num_determinizations: int
    simulations_per_det: int
    batch_timeout_ms: int = 10

    def total_sims_per_move(self) -> int:
        return self.num_determinizations * self.simulations_per_det

    def __str__(self) -> str:
        return f"{self.workers}w × {self.parallel_batch_size}batch × {self.num_determinizations}×{self.simulations_per_det}"


def load_results_from_db(db_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load successful and failed results from database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Successful results
    cursor.execute("""
        SELECT workers, parallel_batch_size, num_determinizations,
               simulations_per_det, batch_timeout_ms, rounds_per_min,
               variance, num_rounds, phase, elapsed_sec,
               examples_per_round, cpu_percent, gpu_memory_mb
        FROM experiments
        WHERE success = 1 AND rounds_per_min IS NOT NULL
        ORDER BY rounds_per_min DESC
    """)

    successful = []
    for row in cursor.fetchall():
        config = SweepConfig(
            workers=row[0],
            parallel_batch_size=row[1],
            num_determinizations=row[2],
            simulations_per_det=row[3],
            batch_timeout_ms=row[4]
        )
        successful.append({
            'config': config,
            'rounds_per_min': row[5],
            'variance': row[6] if row[6] is not None else 0.0,
            'num_rounds': row[7],
            'phase': row[8],
            'elapsed_sec': row[9],
            'examples_per_round': row[10],
            'cpu_percent': row[11],
            'gpu_memory_mb': row[12]
        })

    # Failed results
    cursor.execute("""
        SELECT workers, parallel_batch_size, num_determinizations,
               simulations_per_det, batch_timeout_ms, error_msg, phase
        FROM experiments
        WHERE success = 0
    """)

    failed = []
    for row in cursor.fetchall():
        config = SweepConfig(
            workers=row[0],
            parallel_batch_size=row[1],
            num_determinizations=row[2],
            simulations_per_det=row[3],
            batch_timeout_ms=row[4]
        )
        failed.append({
            'config': config,
            'error': row[5],
            'phase': row[6]
        })

    conn.close()
    return successful, failed


def generate_markdown_report(
    successful: List[Dict],
    failed: List[Dict],
    output_path: str,
    baseline_perf: float = 741.0
):
    """Generate comprehensive Markdown report"""

    report = []
    report.append("# BlobMaster Auto-Tune Parameter Sweep Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\nBaseline: {baseline_perf:.1f} rounds/min (32w × 30batch × 3×30 MCTS)")

    # Summary statistics
    report.append("\n## Summary")
    report.append(f"- Configurations tested: {len(successful) + len(failed)}")
    report.append(f"- Successful: {len(successful)}")
    report.append(f"- Failed: {len(failed)}")

    if successful:
        best = successful[0]
        improvement = ((best['rounds_per_min'] / baseline_perf) - 1) * 100
        report.append(f"- Best configuration: {best['config']}")
        report.append(f"- Best performance: {best['rounds_per_min']:.1f} rounds/min")
        report.append(f"- Improvement: {improvement:+.1f}% vs baseline")

    # Top configurations table
    if successful:
        report.append("\n## Top 10 Configurations")
        report.append("\n| Rank | Workers | Batch Size | Det × Sims | Total Sims | Rounds/Min | vs Baseline | Variance |")
        report.append("|------|---------|------------|------------|------------|------------|-------------|----------|")

        for idx, result in enumerate(successful[:10]):
            config = result['config']
            perf = result['rounds_per_min']
            variance = result.get('variance', 0.0)
            vs_baseline = ((perf / baseline_perf) - 1) * 100

            report.append(
                f"| {idx+1} | {config.workers} | {config.parallel_batch_size} | "
                f"{config.num_determinizations}×{config.simulations_per_det} | "
                f"{config.total_sims_per_move()} | {perf:.1f} | "
                f"{vs_baseline:+.1f}% | {variance:.1f}% |"
            )

    # Parameter analysis
    report.append("\n## Parameter Analysis")

    # Analyze each parameter
    params = ['workers', 'parallel_batch_size', 'num_determinizations', 'simulations_per_det']

    for param_name in params:
        report.append(f"\n### {param_name}")

        # Group by parameter value
        param_groups = {}
        for result in successful:
            config = result['config']
            value = getattr(config, param_name)

            if value not in param_groups:
                param_groups[value] = []
            param_groups[value].append(result['rounds_per_min'])

        # Calculate statistics for each value
        param_stats = []
        for value, perfs in param_groups.items():
            avg = sum(perfs) / len(perfs)
            max_perf = max(perfs)
            count = len(perfs)
            param_stats.append((value, avg, max_perf, count))

        param_stats.sort(key=lambda x: x[1], reverse=True)

        report.append(f"\n| Value | Avg Perf | Max Perf | Count |")
        report.append("|-------|----------|----------|-------|")
        for value, avg, max_perf, count in param_stats:
            report.append(f"| {value} | {avg:.1f} | {max_perf:.1f} | {count} |")

        # Best value
        best_value = param_stats[0][0]
        best_avg = param_stats[0][1]
        report.append(f"\n**Best value: {best_value}** (avg: {best_avg:.1f} rounds/min)")

    # Failures
    if failed:
        report.append("\n## Failed Configurations")
        report.append(f"\nTotal failures: {len(failed)}")

        # Group by error type
        error_types = {}
        for failure in failed:
            error = failure['error']
            error_type = error.split(':')[0] if ':' in error else error

            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(failure)

        report.append("\n| Error Type | Count | Examples |")
        report.append("|------------|-------|----------|")
        for error_type, failures_list in error_types.items():
            count = len(failures_list)
            examples = ', '.join([str(f['config']) for f in failures_list[:2]])
            report.append(f"| {error_type} | {count} | {examples} |")

    # Key findings
    report.append("\n## Key Findings")

    if successful:
        # Find best for each parameter
        best_workers = max(successful, key=lambda x: x['rounds_per_min'] if x['config'].workers else 0)
        best_batch = max(successful, key=lambda x: x['rounds_per_min'] if x['config'].parallel_batch_size else 0)

        report.append(f"- **Workers**: Optimal at {best_workers['config'].workers} ({best_workers['rounds_per_min']:.1f} r/min)")
        report.append(f"- **Parallel batch size**: Optimal at {best_batch['config'].parallel_batch_size} ({best_batch['rounds_per_min']:.1f} r/min)")

        # Variance analysis
        high_variance = [r for r in successful if r.get('variance', 0) > 5.0]
        if high_variance:
            report.append(f"- **Stability concern**: {len(high_variance)} configs have variance >5%")

    # Recommendations
    report.append("\n## Recommendations")

    if successful:
        best = successful[0]
        improvement = ((best['rounds_per_min'] / baseline_perf) - 1) * 100

        if improvement > 10:
            report.append(f"1. **Immediate adoption**: {best['config']} provides {improvement:.1f}% speedup")
        elif improvement > 5:
            report.append(f"1. **Consider adoption**: {best['config']} provides {improvement:.1f}% speedup")
        else:
            report.append(f"1. **Current baseline is near-optimal**: Only {improvement:.1f}% improvement found")

        # Check variance
        if best.get('variance', 0) > 5:
            report.append(f"2. **Stability testing needed**: Best config has {best['variance']:.1f}% variance")

        # Future work
        report.append("3. **Future investigation**:")
        report.append("   - Fine-tune around optimal values")
        report.append("   - Test with different game configurations (players, cards)")
        report.append("   - Validate on longer training runs")

    # Export config
    if successful:
        best = successful[0]
        config = best['config']

        report.append("\n## Export Configuration")
        report.append("\nAdd to `ml/config.py`:")
        report.append("\n```python")
        report.append(f"# Auto-tuned configuration (generated {datetime.now().strftime('%Y-%m-%d')})")
        report.append(f"# Performance: {best['rounds_per_min']:.1f} rounds/min ({improvement:+.1f}% vs baseline)")
        report.append("")
        report.append("def get_autotuned_config() -> TrainingConfig:")
        report.append("    config = get_production_config()")
        report.append(f"    config.num_workers = {config.workers}")
        report.append(f"    config.parallel_batch_size = {config.parallel_batch_size}")
        report.append(f"    config.num_determinizations = {config.num_determinizations}")
        report.append(f"    config.simulations_per_determinization = {config.simulations_per_det}")
        report.append("    return config")
        report.append("```")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Report written to: {output_path}")


def generate_plots(successful: List[Dict], output_dir: Path, baseline_perf: float = 741.0):
    """Generate visualization plots"""

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Convert to DataFrame for easier plotting
    df_data = []
    for result in successful:
        config = result['config']
        df_data.append({
            'workers': config.workers,
            'parallel_batch_size': config.parallel_batch_size,
            'num_determinizations': config.num_determinizations,
            'simulations_per_det': config.simulations_per_det,
            'total_sims': config.total_sims_per_move(),
            'rounds_per_min': result['rounds_per_min'],
            'variance': result.get('variance', 0.0),
            'phase': result['phase']
        })

    df = pd.DataFrame(df_data)

    # 1. Worker scaling plot
    fig, ax = plt.subplots(figsize=(10, 6))
    worker_groups = df.groupby('workers')['rounds_per_min'].agg(['mean', 'max', 'min'])

    ax.plot(worker_groups.index, worker_groups['mean'], 'o-', label='Average', linewidth=2)
    ax.fill_between(worker_groups.index, worker_groups['min'], worker_groups['max'], alpha=0.3)
    ax.axhline(y=baseline_perf, color='r', linestyle='--', label='Baseline (741 r/min)')

    ax.set_xlabel('Number of Workers', fontsize=12)
    ax.set_ylabel('Rounds per Minute', fontsize=12)
    ax.set_title('Worker Scaling Performance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'worker_scaling.png', dpi=300)
    plt.close()

    # 2. Parallel batch size sweep
    fig, ax = plt.subplots(figsize=(10, 6))
    batch_groups = df.groupby('parallel_batch_size')['rounds_per_min'].agg(['mean', 'max', 'min'])

    ax.plot(batch_groups.index, batch_groups['mean'], 'o-', label='Average', linewidth=2)
    ax.fill_between(batch_groups.index, batch_groups['min'], batch_groups['max'], alpha=0.3)
    ax.axhline(y=baseline_perf, color='r', linestyle='--', label='Baseline')

    ax.set_xlabel('Parallel Batch Size', fontsize=12)
    ax.set_ylabel('Rounds per Minute', fontsize=12)
    ax.set_title('Parallel Batch Size Impact', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'batch_size_sweep.png', dpi=300)
    plt.close()

    # 3. MCTS configuration heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Pivot for heatmap
    heatmap_data = df.pivot_table(
        values='rounds_per_min',
        index='num_determinizations',
        columns='simulations_per_det',
        aggfunc='mean'
    )

    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Rounds/Min'})
    ax.set_xlabel('Simulations per Determinization', fontsize=12)
    ax.set_ylabel('Number of Determinizations', fontsize=12)
    ax.set_title('MCTS Configuration Heatmap', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(plots_dir / 'mcts_heatmap.png', dpi=300)
    plt.close()

    # 4. Top 10 comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    top_10 = df.nlargest(10, 'rounds_per_min')
    labels = [f"{row['workers']}w×{row['parallel_batch_size']}b×{row['num_determinizations']}×{row['simulations_per_det']}"
              for _, row in top_10.iterrows()]

    bars = ax.barh(range(len(top_10)), top_10['rounds_per_min'])
    ax.axvline(x=baseline_perf, color='r', linestyle='--', label='Baseline', linewidth=2)

    # Color bars based on performance
    for i, (bar, perf) in enumerate(zip(bars, top_10['rounds_per_min'])):
        if perf > baseline_perf * 1.1:
            bar.set_color('green')
        elif perf > baseline_perf:
            bar.set_color('lightgreen')
        else:
            bar.set_color('gray')

    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Rounds per Minute', fontsize=12)
    ax.set_title('Top 10 Configurations', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(plots_dir / 'top10_comparison.png', dpi=300)
    plt.close()

    # 5. Variance vs Performance
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter to configs with variance data
    variance_data = df[df['variance'] > 0]

    if not variance_data.empty:
        scatter = ax.scatter(
            variance_data['rounds_per_min'],
            variance_data['variance'],
            c=variance_data['workers'],
            s=100,
            alpha=0.6,
            cmap='viridis'
        )

        ax.axvline(x=baseline_perf, color='r', linestyle='--', label='Baseline', alpha=0.5)
        ax.axhline(y=5.0, color='orange', linestyle='--', label='5% variance threshold', alpha=0.5)

        ax.set_xlabel('Rounds per Minute', fontsize=12)
        ax.set_ylabel('Variance (%)', fontsize=12)
        ax.set_title('Performance vs Stability', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Workers', fontsize=10)

        plt.tight_layout()
        plt.savefig(plots_dir / 'variance_analysis.png', dpi=300)
        plt.close()

    print(f"Plots saved to: {plots_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate report from auto-tune sweep results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('db_path', type=str, help='Path to auto_tune_results.db')
    parser.add_argument('--output', type=str, default=None,
                       help='Output markdown file (default: auto_tune_report.md in same dir as db)')
    parser.add_argument('--baseline', type=float, default=741.0,
                       help='Baseline performance for comparison (default: 741.0)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')

    args = parser.parse_args()

    # Check database exists
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        return 1

    # Setup output paths
    output_dir = db_path.parent
    if args.output:
        report_path = args.output
    else:
        report_path = str(output_dir / 'auto_tune_report.md')

    print(f"Loading results from: {db_path}")
    successful, failed = load_results_from_db(str(db_path))

    print(f"Found {len(successful)} successful and {len(failed)} failed configurations")

    # Generate report
    print("Generating markdown report...")
    generate_markdown_report(successful, failed, report_path, args.baseline)

    # Generate plots
    if not args.no_plots:
        try:
            print("Generating plots...")
            generate_plots(successful, output_dir, args.baseline)
        except Exception as e:
            print(f"WARNING: Plot generation failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n✓ Report generation complete!")
    print(f"  Report: {report_path}")
    if not args.no_plots:
        print(f"  Plots: {output_dir / 'plots'}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
