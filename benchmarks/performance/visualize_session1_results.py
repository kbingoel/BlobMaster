"""
Visualization script for Session 1+2 validation benchmark results.

Generates plots and summary statistics from the benchmark CSV output.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_results(csv_path: str) -> List[Dict]:
    """Load benchmark results from CSV."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in ['workers', 'num_determinizations', 'simulations_per_determinization',
                       'total_sims_per_move', 'games', 'parallel_batch_size', 'batch_timeout_ms',
                       'elapsed_seconds', 'games_per_minute', 'seconds_per_game',
                       'total_examples', 'examples_per_minute']:
                if key in row and row[key]:
                    row[key] = float(row[key])
            # Convert boolean field
            if 'use_parallel_expansion' in row:
                row['use_parallel_expansion'] = row['use_parallel_expansion'].lower() == 'true'
            results.append(row)
    return results


def plot_batch_size_sweep(results: List[Dict], output_path: str):
    """Plot parallel_batch_size vs performance."""
    # Filter for batch size sweep (fixed workers)
    worker_counts = set(r['workers'] for r in results)
    if len(worker_counts) == 1:
        # Single worker count
        batch_sizes = sorted(set(r['parallel_batch_size'] for r in results))
        games_per_min = [next(r['games_per_minute'] for r in results
                              if r['parallel_batch_size'] == bs)
                        for bs in batch_sizes]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(batch_sizes, games_per_min, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Parallel Batch Size', fontsize=12)
        ax.set_ylabel('Games per Minute', fontsize=12)
        ax.set_title(f'Parallel Batch Size Impact ({int(list(worker_counts)[0])} workers)',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Mark optimal
        best_idx = games_per_min.index(max(games_per_min))
        ax.axvline(batch_sizes[best_idx], color='red', linestyle='--', alpha=0.5,
                  label=f'Optimal: {batch_sizes[best_idx]}')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved batch size sweep plot: {output_path}")


def plot_worker_scaling(results: List[Dict], output_path: str):
    """Plot worker scaling efficiency."""
    # Filter for worker scaling (fixed batch size)
    batch_sizes = set(r['parallel_batch_size'] for r in results)
    if len(batch_sizes) == 1:
        workers = sorted(set(r['workers'] for r in results))
        games_per_min = [next(r['games_per_minute'] for r in results
                             if r['workers'] == w)
                        for w in workers]

        # Calculate efficiency
        baseline_throughput = games_per_min[0]
        baseline_workers = workers[0]
        speedups = [g / baseline_throughput for g in games_per_min]
        ideal_speedups = [w / baseline_workers for w in workers]
        efficiencies = [(s / i) * 100 for s, i in zip(speedups, ideal_speedups)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Throughput
        ax1.plot(workers, games_per_min, 'o-', linewidth=2, markersize=8, label='Actual')
        ax1.plot(workers, [baseline_throughput * w / baseline_workers for w in workers],
                '--', alpha=0.5, label='Ideal Linear')
        ax1.set_xlabel('Number of Workers', fontsize=12)
        ax1.set_ylabel('Games per Minute', fontsize=12)
        ax1.set_title('Worker Scaling: Throughput', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Right: Efficiency
        ax2.plot(workers, efficiencies, 'o-', linewidth=2, markersize=8, color='orange')
        ax2.axhline(100, color='gray', linestyle='--', alpha=0.5, label='100% Efficient')
        ax2.set_xlabel('Number of Workers', fontsize=12)
        ax2.set_ylabel('Scaling Efficiency (%)', fontsize=12)
        ax2.set_title('Worker Scaling: Efficiency', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved worker scaling plot: {output_path}")


def plot_2d_heatmap(results: List[Dict], output_path: str):
    """Plot 2D heatmap of workers × parallel_batch_size."""
    # Extract unique values
    workers_set = sorted(set(r['workers'] for r in results))
    batch_sizes_set = sorted(set(r['parallel_batch_size'] for r in results))

    # Create matrix
    matrix = np.zeros((len(workers_set), len(batch_sizes_set)))
    for i, w in enumerate(workers_set):
        for j, bs in enumerate(batch_sizes_set):
            matching = [r for r in results
                       if r['workers'] == w and r['parallel_batch_size'] == bs]
            if matching:
                matrix[i, j] = matching[0]['games_per_minute']

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest')

    # Set ticks
    ax.set_xticks(range(len(batch_sizes_set)))
    ax.set_xticklabels([f'{int(bs)}' for bs in batch_sizes_set])
    ax.set_yticks(range(len(workers_set)))
    ax.set_yticklabels([f'{int(w)}' for w in workers_set])

    ax.set_xlabel('Parallel Batch Size', fontsize=12)
    ax.set_ylabel('Number of Workers', fontsize=12)
    ax.set_title('Performance Heatmap: Workers × Batch Size',
                 fontsize=14, fontweight='bold')

    # Add values to cells
    for i in range(len(workers_set)):
        for j in range(len(batch_sizes_set)):
            if matrix[i, j] > 0:
                text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Games per Minute', fontsize=11)

    # Mark best configuration
    best_i, best_j = np.unravel_index(matrix.argmax(), matrix.shape)
    ax.add_patch(plt.Rectangle((best_j-0.5, best_i-0.5), 1, 1,
                               fill=False, edgecolor='red', linewidth=3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved 2D heatmap: {output_path}")


def plot_mcts_comparison(results: List[Dict], output_path: str):
    """Plot MCTS configuration comparison."""
    # Filter for different MCTS configs
    mcts_configs = []
    for r in results:
        config_name = r.get('mcts_config_name', 'Unknown')
        if config_name != 'Unknown':
            mcts_configs.append((
                config_name,
                r['total_sims_per_move'],
                r['games_per_minute']
            ))

    if not mcts_configs:
        return

    # Sort by sims per move
    mcts_configs.sort(key=lambda x: x[1])
    names, sims, games_per_min = zip(*mcts_configs)

    # Calculate training time
    GAMES_NEEDED = 500 * 10_000
    training_days = [GAMES_NEEDED / (g * 60 * 24) for g in games_per_min]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Throughput
    colors = ['green', 'orange', 'red']
    ax1.bar(names, games_per_min, color=colors[:len(names)], alpha=0.7)
    ax1.set_ylabel('Games per Minute', fontsize=12)
    ax1.set_title('MCTS Configuration: Throughput', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for i, (n, g) in enumerate(zip(names, games_per_min)):
        ax1.text(i, g, f'{g:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Right: Training time
    ax2.bar(names, training_days, color=colors[:len(names)], alpha=0.7)
    ax2.set_ylabel('Training Time (days)', fontsize=12)
    ax2.set_title('MCTS Configuration: Training Duration', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for i, (n, t) in enumerate(zip(names, training_days)):
        ax2.text(i, t, f'{t:.0f}d', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved MCTS comparison plot: {output_path}")


def generate_summary_report(results: List[Dict], output_path: str):
    """Generate text summary report."""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SESSION 1+2 VALIDATION BENCHMARK - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")

        # Overall best
        best = max(results, key=lambda r: r['games_per_minute'])
        f.write("BEST CONFIGURATION FOUND:\n")
        f.write("-"*80 + "\n")
        f.write(f"Workers: {int(best['workers'])}\n")
        f.write(f"Parallel batch size: {int(best['parallel_batch_size'])}\n")
        f.write(f"MCTS config: {best['num_determinizations']} det × {best['simulations_per_determinization']} sims\n")
        f.write(f"Performance: {best['games_per_minute']:.1f} games/min\n")
        f.write(f"Speedup vs baseline (36.7 g/min): {best['games_per_minute'] / 36.7:.2f}x\n")

        GAMES_NEEDED = 500 * 10_000
        training_days = GAMES_NEEDED / (best['games_per_minute'] * 60 * 24)
        f.write(f"Estimated training time: {training_days:.1f} days\n")
        f.write("\n")

        # Batch size analysis
        batch_size_results = {}
        for r in results:
            bs = r['parallel_batch_size']
            if bs not in batch_size_results:
                batch_size_results[bs] = []
            batch_size_results[bs].append(r['games_per_minute'])

        if len(batch_size_results) > 1:
            f.write("PARALLEL BATCH SIZE ANALYSIS:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Batch Size':<15} {'Avg Games/Min':<20} {'Max Games/Min':<20}\n")
            for bs in sorted(batch_size_results.keys()):
                avg = np.mean(batch_size_results[bs])
                max_val = max(batch_size_results[bs])
                f.write(f"{int(bs):<15} {avg:<20.1f} {max_val:<20.1f}\n")
            f.write("\n")

        # Worker scaling analysis
        worker_results = {}
        for r in results:
            w = r['workers']
            if w not in worker_results:
                worker_results[w] = []
            worker_results[w].append(r['games_per_minute'])

        if len(worker_results) > 1:
            f.write("WORKER SCALING ANALYSIS:\n")
            f.write("-"*80 + "\n")
            sorted_workers = sorted(worker_results.keys())
            baseline_throughput = np.mean(worker_results[sorted_workers[0]])
            baseline_workers = sorted_workers[0]

            f.write(f"{'Workers':<10} {'Avg G/Min':<12} {'Speedup':<10} {'Efficiency':<12}\n")
            for w in sorted_workers:
                avg = np.mean(worker_results[w])
                speedup = avg / baseline_throughput
                ideal_speedup = w / baseline_workers
                efficiency = (speedup / ideal_speedup) * 100
                f.write(f"{int(w):<10} {avg:<12.1f} {speedup:<10.2f}x {efficiency:<12.1f}%\n")
            f.write("\n")

        # MCTS config comparison
        mcts_results = [r for r in results if 'mcts_config_name' in r]
        if mcts_results:
            f.write("MCTS CONFIGURATION COMPARISON:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Config':<12} {'Sims/Move':<12} {'Games/Min':<12} {'Training Days':<15}\n")

            mcts_by_config = {}
            for r in mcts_results:
                name = r['mcts_config_name']
                if name not in mcts_by_config:
                    mcts_by_config[name] = r

            for name in sorted(mcts_by_config.keys()):
                r = mcts_by_config[name]
                training_days = GAMES_NEEDED / (r['games_per_minute'] * 60 * 24)
                f.write(f"{name:<12} {int(r['total_sims_per_move']):<12} "
                       f"{r['games_per_minute']:<12.1f} {training_days:<15.1f}\n")
            f.write("\n")

        # Recommendations
        f.write("RECOMMENDATIONS:\n")
        f.write("-"*80 + "\n")
        f.write(f"1. Use parallel_batch_size = {int(best['parallel_batch_size'])}\n")
        f.write(f"2. Use {int(best['workers'])} workers for optimal throughput\n")

        speedup = best['games_per_minute'] / 36.7
        if speedup >= 3.0:
            f.write(f"3. ✅ Target achieved! {speedup:.2f}x speedup (≥3x goal)\n")
            f.write("4. Sessions 3-5 optional but will provide additional gains\n")
        elif speedup >= 2.0:
            f.write(f"3. ⚠️ Good progress: {speedup:.2f}x speedup (target: 3x)\n")
            f.write("4. Proceed with Sessions 3-5 to reach 3x target\n")
        else:
            f.write(f"3. ⚠️ Below expectations: {speedup:.2f}x speedup (target: 3x)\n")
            f.write("4. Debug batch sizes and GPU utilization before Sessions 3-5\n")

        f.write("\n")
        f.write("="*80 + "\n")

    print(f"Saved summary report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Session 1+2 validation benchmark results"
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to benchmark results CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same as input CSV)",
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.input_csv}")
    results = load_results(args.input_csv)
    print(f"Loaded {len(results)} benchmark results")

    # Determine output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.input_csv).parent)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    base_name = Path(args.input_csv).stem

    print("\nGenerating visualizations...")

    # Try each plot type (some may not apply depending on sweeps run)
    try:
        plot_batch_size_sweep(results, str(output_dir / f"{base_name}_batch_size.png"))
    except Exception as e:
        print(f"Skipping batch size plot: {e}")

    try:
        plot_worker_scaling(results, str(output_dir / f"{base_name}_worker_scaling.png"))
    except Exception as e:
        print(f"Skipping worker scaling plot: {e}")

    try:
        plot_2d_heatmap(results, str(output_dir / f"{base_name}_heatmap.png"))
    except Exception as e:
        print(f"Skipping 2D heatmap: {e}")

    try:
        plot_mcts_comparison(results, str(output_dir / f"{base_name}_mcts_comparison.png"))
    except Exception as e:
        print(f"Skipping MCTS comparison: {e}")

    # Generate summary report
    generate_summary_report(results, str(output_dir / f"{base_name}_summary.txt"))

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
