"""
Visualization script for diagnostic benchmark results.

Creates plots to analyze:
- Games/min vs worker count
- GPU utilization vs worker count
- Batch size vs worker count
- Comparison of batching strategies
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


def load_results(csv_path="benchmark_diagnostic_results.csv"):
    """Load benchmark results from CSV."""
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found. Run benchmark_diagnostic.py first.")
        return None

    df = pd.read_csv(csv_path)
    return df


def plot_worker_scaling(df):
    """Plot performance metrics vs worker count."""

    # Filter for worker scaling tests (batched, threaded, 3 cards)
    worker_data = df[
        (df['use_batched_evaluator'] == True) &
        (df['use_thread_pool'] == True) &
        (df['cards_to_deal'] == 3)
    ].copy()

    if worker_data.empty:
        print("No worker scaling data found")
        return

    worker_data = worker_data.sort_values('num_workers')

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Worker Scaling Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Games/min vs Workers
    ax = axes[0, 0]
    ax.plot(worker_data['num_workers'], worker_data['games_per_min'],
            marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax.axhline(y=500, color='green', linestyle='--', label='Target (500 games/min)')
    ax.set_xlabel('Number of Workers', fontsize=12)
    ax.set_ylabel('Games per Minute', fontsize=12)
    ax.set_title('Throughput vs Worker Count', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: GPU Utilization vs Workers
    ax = axes[0, 1]
    ax.plot(worker_data['num_workers'], worker_data['avg_gpu_utilization'],
            marker='s', linewidth=2, markersize=8, color='#A23B72', label='Average')
    ax.plot(worker_data['num_workers'], worker_data['max_gpu_utilization'],
            marker='^', linewidth=2, markersize=8, color='#F18F01', label='Maximum',
            linestyle='--', alpha=0.7)
    ax.axhline(y=70, color='green', linestyle='--', label='Target (70%)')
    ax.set_xlabel('Number of Workers', fontsize=12)
    ax.set_ylabel('GPU Utilization (%)', fontsize=12)
    ax.set_title('GPU Utilization vs Worker Count', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 100)

    # Plot 3: Batch Size vs Workers
    ax = axes[1, 0]
    ax.plot(worker_data['num_workers'], worker_data['avg_batch_size'],
            marker='D', linewidth=2, markersize=8, color='#6A994E')
    ax.axhline(y=128, color='green', linestyle='--', label='Target (128)')
    ax.set_xlabel('Number of Workers', fontsize=12)
    ax.set_ylabel('Average Batch Size', fontsize=12)
    ax.set_title('Batch Size vs Worker Count', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 4: Efficiency (games/min per worker)
    ax = axes[1, 1]
    efficiency = worker_data['games_per_min'] / worker_data['num_workers']
    ax.plot(worker_data['num_workers'], efficiency,
            marker='*', linewidth=2, markersize=10, color='#C1121F')
    ax.set_xlabel('Number of Workers', fontsize=12)
    ax.set_ylabel('Games/min per Worker', fontsize=12)
    ax.set_title('Worker Efficiency', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('benchmark_worker_scaling.png', dpi=300, bbox_inches='tight')
    print("Saved: benchmark_worker_scaling.png")
    plt.close()


def plot_batching_comparison(df):
    """Compare batched vs direct inference."""

    # Get batching comparison data
    comparison_data = df[
        (df['use_thread_pool'] == True) &
        (df['cards_to_deal'] == 3) &
        (df['num_workers'].isin([4, 16, 32]))
    ].copy()

    if comparison_data.empty:
        print("No batching comparison data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Batched vs Direct Inference Comparison', fontsize=16, fontweight='bold')

    # Prepare data for plotting
    workers = sorted(comparison_data['num_workers'].unique())

    batched_games = []
    direct_games = []
    batched_gpu = []
    direct_gpu = []

    for w in workers:
        batched = comparison_data[
            (comparison_data['num_workers'] == w) &
            (comparison_data['use_batched_evaluator'] == True)
        ]
        direct = comparison_data[
            (comparison_data['num_workers'] == w) &
            (comparison_data['use_batched_evaluator'] == False)
        ]

        if not batched.empty:
            batched_games.append(batched['games_per_min'].values[0])
            batched_gpu.append(batched['avg_gpu_utilization'].values[0])
        else:
            batched_games.append(0)
            batched_gpu.append(0)

        if not direct.empty:
            direct_games.append(direct['games_per_min'].values[0])
            direct_gpu.append(direct['avg_gpu_utilization'].values[0])
        else:
            direct_games.append(0)
            direct_gpu.append(0)

    # Plot 1: Games/min comparison
    ax = axes[0]
    x = range(len(workers))
    width = 0.35
    ax.bar([i - width/2 for i in x], batched_games, width, label='Batched', color='#2E86AB')
    ax.bar([i + width/2 for i in x], direct_games, width, label='Direct', color='#F18F01')
    ax.set_xlabel('Number of Workers', fontsize=12)
    ax.set_ylabel('Games per Minute', fontsize=12)
    ax.set_title('Throughput: Batched vs Direct', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(workers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: GPU utilization comparison
    ax = axes[1]
    ax.bar([i - width/2 for i in x], batched_gpu, width, label='Batched', color='#2E86AB')
    ax.bar([i + width/2 for i in x], direct_gpu, width, label='Direct', color='#F18F01')
    ax.axhline(y=70, color='green', linestyle='--', label='Target (70%)')
    ax.set_xlabel('Number of Workers', fontsize=12)
    ax.set_ylabel('GPU Utilization (%)', fontsize=12)
    ax.set_title('GPU Utilization: Batched vs Direct', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(workers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig('benchmark_batching_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: benchmark_batching_comparison.png")
    plt.close()


def plot_game_complexity(df):
    """Plot impact of game complexity (cards dealt)."""

    complexity_data = df[
        (df['use_batched_evaluator'] == True) &
        (df['use_thread_pool'] == True) &
        (df['num_workers'] == 16)
    ].copy()

    if complexity_data.empty:
        print("No game complexity data found")
        return

    complexity_data = complexity_data.sort_values('cards_to_deal')

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Impact of Game Complexity (Cards Dealt)', fontsize=16, fontweight='bold')

    # Plot 1: Games/min vs cards
    ax = axes[0]
    ax.plot(complexity_data['cards_to_deal'], complexity_data['games_per_min'],
            marker='o', linewidth=2, markersize=8, color='#6A994E')
    ax.set_xlabel('Cards Dealt per Player', fontsize=12)
    ax.set_ylabel('Games per Minute', fontsize=12)
    ax.set_title('Throughput vs Game Size', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 2: Examples per game
    ax = axes[1]
    ax.plot(complexity_data['cards_to_deal'], complexity_data['examples_per_game'],
            marker='s', linewidth=2, markersize=8, color='#A23B72')
    ax.set_xlabel('Cards Dealt per Player', fontsize=12)
    ax.set_ylabel('Training Examples per Game', fontsize=12)
    ax.set_title('Data Generation Efficiency', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('benchmark_game_complexity.png', dpi=300, bbox_inches='tight')
    print("Saved: benchmark_game_complexity.png")
    plt.close()


def create_summary_table(df):
    """Create a summary table of key findings."""

    print("\n" + "="*80)
    print("BENCHMARK ANALYSIS SUMMARY")
    print("="*80)

    # Find best overall configuration
    best = df.loc[df['games_per_min'].idxmax()]

    print("\nBest Configuration:")
    print(f"  Workers: {best['num_workers']}")
    print(f"  Batched: {best['use_batched_evaluator']}")
    print(f"  Threads: {best['use_thread_pool']}")
    print(f"  Cards: {best['cards_to_deal']}")
    print(f"\nPerformance:")
    print(f"  Games/min: {best['games_per_min']:.1f}")
    print(f"  GPU utilization: {best['avg_gpu_utilization']:.1f}%")
    print(f"  Batch size: {best['avg_batch_size']:.1f}")
    print(f"  Examples/game: {best['examples_per_game']:.1f}")

    # Training time estimates
    print("\nTraining Time Estimates (500 iterations, 10k games/iter):")

    configs = [
        ("Current Best", best['games_per_min']),
        ("Target (500 games/min)", 500),
        ("Stretch Goal (1000 games/min)", 1000),
    ]

    for name, games_per_min in configs:
        if games_per_min > 0:
            minutes_per_iter = 10_000 / games_per_min
            hours_per_iter = minutes_per_iter / 60
            total_days = (hours_per_iter * 500) / 24

            print(f"  {name}:")
            print(f"    Per iteration: {minutes_per_iter:.1f} min")
            print(f"    Total training: {total_days:.1f} days")

    # Worker scaling analysis
    print("\nWorker Scaling Analysis:")
    worker_data = df[
        (df['use_batched_evaluator'] == True) &
        (df['use_thread_pool'] == True) &
        (df['cards_to_deal'] == 3)
    ].sort_values('num_workers')

    if not worker_data.empty:
        print(f"\n  {'Workers':<10} {'Games/min':<15} {'GPU%':<10} {'Batch':<10} {'Speedup'}")
        print("  " + "-"*60)

        baseline = worker_data.iloc[0]['games_per_min']

        for _, row in worker_data.iterrows():
            speedup = row['games_per_min'] / baseline if baseline > 0 else 0
            print(f"  {row['num_workers']:<10} {row['games_per_min']:<15.1f} "
                  f"{row['avg_gpu_utilization']:<10.1f} {row['avg_batch_size']:<10.1f} "
                  f"{speedup:.2f}x")

    # Recommendations
    print("\nRecommendations:")

    if best['avg_gpu_utilization'] < 70:
        print(f"  • GPU utilization is low ({best['avg_gpu_utilization']:.1f}%)")
        print(f"    → Try {int(best['num_workers'] * 1.5)}-{int(best['num_workers'] * 2)} workers")

    if best['avg_batch_size'] < 128:
        print(f"  • Batch sizes are small ({best['avg_batch_size']:.1f})")
        print(f"    → Increase concurrent work (more workers or more determinizations)")

    if best['games_per_min'] < 500:
        print(f"  • Throughput is below target ({best['games_per_min']:.1f} vs 500 games/min)")
        estimated_workers_needed = int((500 / best['games_per_min']) * best['num_workers'])
        print(f"    → Estimated {estimated_workers_needed} workers needed for 500 games/min")

    # Check if batching helps
    if len(df) > 2:
        batching_comparison = df[
            (df['num_workers'] == 16) &
            (df['use_thread_pool'] == True) &
            (df['cards_to_deal'] == 3)
        ]

        if len(batching_comparison) >= 2:
            batched = batching_comparison[batching_comparison['use_batched_evaluator'] == True]
            direct = batching_comparison[batching_comparison['use_batched_evaluator'] == False]

            if not batched.empty and not direct.empty:
                batched_perf = batched.iloc[0]['games_per_min']
                direct_perf = direct.iloc[0]['games_per_min']

                if batched_perf > direct_perf * 1.2:
                    print(f"  • Batching provides {batched_perf/direct_perf:.2f}x speedup - keep using it")
                elif batched_perf < direct_perf * 0.9:
                    print(f"  • Batching is slower ({batched_perf/direct_perf:.2f}x) - consider removing it")
                else:
                    print(f"  • Batching provides minimal benefit - neutral")

    print("\n" + "="*80)


def main():
    """Main visualization function."""
    print("Loading benchmark results...")

    df = load_results()

    if df is None or df.empty:
        print("No results to visualize!")
        return

    print(f"Loaded {len(df)} benchmark results\n")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    # Create visualizations
    print("Creating visualizations...")

    plot_worker_scaling(df)
    plot_batching_comparison(df)
    plot_game_complexity(df)

    # Print summary
    create_summary_table(df)

    print("\nVisualization complete!")
    print("Generated files:")
    print("  - benchmark_worker_scaling.png")
    print("  - benchmark_batching_comparison.png")
    print("  - benchmark_game_complexity.png")


if __name__ == "__main__":
    main()
