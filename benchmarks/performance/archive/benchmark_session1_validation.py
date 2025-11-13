"""
Session 1+2 Optimization Validation Benchmark

Comprehensive parameter sweep to explore the performance landscape with:
- Batch submission API (Session 1)
- Parallel MCTS expansion (Session 2)

This benchmark systematically tests:
1. parallel_batch_size sweep (find optimal batching parameter)
2. Worker scaling with new optimizations (better scaling expected)
3. 2D interaction: workers × parallel_batch_size
4. MCTS configuration impact with optimal settings
5. Batch timeout sensitivity (optional)

Expected outcomes:
- Optimal parallel_batch_size (likely 15-25 for 32 workers)
- Improved worker scaling curve (better efficiency at high worker counts)
- 2-3x speedup over baseline (110+ games/min target)
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.training.selfplay import SelfPlayEngine


def run_benchmark(
    workers: int,
    num_determinizations: int,
    simulations_per_determinization: int,
    games: int,
    device: str,
    use_parallel_expansion: bool,
    parallel_batch_size: int,
    batch_timeout_ms: float = 10.0,
    mode: str = "mp",  # "mp" | "threads" | "gpu_server"
    cards_to_deal: int = 5,
) -> Dict:
    """Run a single benchmark configuration."""
    print(f"\n{'='*70}")
    print(
        f"Testing: {workers} workers, parallel_expansion={use_parallel_expansion}, batch_size={parallel_batch_size}"
    )
    print(f"  - Determinizations: {num_determinizations}")
    print(f"  - Simulations/det: {simulations_per_determinization}")
    print(f"  - Total MCTS sims/move: {num_determinizations * simulations_per_determinization}")
    print(f"  - Games to generate: {games}")
    print(f"  - Batch timeout: {batch_timeout_ms}ms")
    print(f"  - Mode: {mode}")
    print(f"  - Cards/game: {cards_to_deal}")
    print(f"{'='*70}")

    # Create network
    network = BlobNet(
        state_dim=256,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        feedforward_dim=1024,
        dropout=0.1,
    )
    network.to(device)
    network.eval()

    # Create encoder and masker
    encoder = StateEncoder()
    masker = ActionMasker()

    # Create self-play engine
    # Select concurrency mode
    use_thread_pool = (mode == "threads")
    use_gpu_server = (mode == "gpu_server")

    engine = SelfPlayEngine(
        network=network,
        encoder=encoder,
        masker=masker,
        num_workers=workers,
        num_determinizations=num_determinizations,
        simulations_per_determinization=simulations_per_determinization,
        device=device,
        # Explicitly select concurrency to avoid unintended auto-threading on CUDA
        use_thread_pool=use_thread_pool,
        use_gpu_server=use_gpu_server,
        gpu_server_max_batch=512,
        gpu_server_timeout_ms=batch_timeout_ms,
        use_batched_evaluator=True,
        batch_size=512,
        batch_timeout_ms=batch_timeout_ms,
        use_parallel_expansion=use_parallel_expansion,
        parallel_batch_size=parallel_batch_size,
    )

    # Generate games
    print("Generating games...")
    start_time = time.time()

    examples = engine.generate_games(
        num_games=games,
        num_players=4,
        cards_to_deal=cards_to_deal,
    )

    end_time = time.time()
    elapsed = end_time - start_time

    # Calculate metrics
    games_per_minute = (games / elapsed) * 60
    seconds_per_game = elapsed / games
    examples_per_minute = (len(examples) / elapsed) * 60

    print(f"\nResults:")
    print(f"  - Games/minute: {games_per_minute:.1f}")
    print(f"  - Seconds/game: {seconds_per_game:.2f}")
    print(f"  - Training examples: {len(examples):,}")
    print(f"  - Examples/minute: {examples_per_minute:.1f}")

    # Cleanup
    engine.shutdown()
    del engine, network
    torch.cuda.empty_cache()

    return {
        "workers": workers,
        "num_determinizations": num_determinizations,
        "simulations_per_determinization": simulations_per_determinization,
        "total_sims_per_move": num_determinizations * simulations_per_determinization,
        "games": games,
        "use_parallel_expansion": use_parallel_expansion,
        "parallel_batch_size": parallel_batch_size,
        "batch_timeout_ms": batch_timeout_ms,
        "mode": mode,
        "cards_per_game": cards_to_deal,
        "elapsed_seconds": elapsed,
        "games_per_minute": games_per_minute,
        "seconds_per_game": seconds_per_game,
        "total_examples": len(examples),
        "examples_per_minute": examples_per_minute,
    }


def sweep_parallel_batch_size(
    workers: int,
    num_det: int,
    sims_per_det: int,
    games_per_config: int,
    device: str,
    batch_sizes: List[int],
    batch_timeout_ms: float = 10.0,
    mode: str = "mp",
    cards_to_deal: int = 5,
) -> List[Dict]:
    """Sweep 1: Find optimal parallel_batch_size at fixed worker count."""
    print("\n" + "="*80)
    print("SWEEP 1: PARALLEL_BATCH_SIZE OPTIMIZATION")
    print("="*80)
    print(f"Testing parallel_batch_size values: {batch_sizes}")
    print(f"Fixed: workers={workers}, det={num_det}, sims={sims_per_det}")
    print(f"Games per config: {games_per_config}")
    print("="*80)

    results = []
    for batch_size in batch_sizes:
        result = run_benchmark(
            workers=workers,
            num_determinizations=num_det,
            simulations_per_determinization=sims_per_det,
            games=games_per_config,
            device=device,
            use_parallel_expansion=True,
            parallel_batch_size=batch_size,
            batch_timeout_ms=batch_timeout_ms,
            mode=mode,
            cards_to_deal=cards_to_deal,
        )
        results.append(result)

        # Brief cooldown
        time.sleep(2)

    # Find optimal
    best = max(results, key=lambda r: r["games_per_minute"])
    print(f"\n{'='*80}")
    print(f"SWEEP 1 WINNER: parallel_batch_size={best['parallel_batch_size']}")
    print(f"  Performance: {best['games_per_minute']:.1f} games/min")
    print(f"{'='*80}")

    return results


def sweep_worker_scaling(
    worker_counts: List[int],
    num_det: int,
    sims_per_det: int,
    games_per_config: int,
    device: str,
    parallel_batch_size: int,
    batch_timeout_ms: float = 10.0,
    mode: str = "mp",
    cards_to_deal: int = 5,
) -> List[Dict]:
    """Sweep 2: Worker scaling with optimal parallel_batch_size."""
    print("\n" + "="*80)
    print("SWEEP 2: WORKER SCALING WITH NEW OPTIMIZATIONS")
    print("="*80)
    print(f"Testing worker counts: {worker_counts}")
    print(f"Fixed: parallel_batch_size={parallel_batch_size}, det={num_det}, sims={sims_per_det}")
    print(f"Games per config: {games_per_config}")
    print("="*80)

    results = []
    for workers in worker_counts:
        result = run_benchmark(
            workers=workers,
            num_determinizations=num_det,
            simulations_per_determinization=sims_per_det,
            games=games_per_config,
            device=device,
            use_parallel_expansion=True,
            parallel_batch_size=parallel_batch_size,
            batch_timeout_ms=batch_timeout_ms,
            mode=mode,
            cards_to_deal=cards_to_deal,
        )
        results.append(result)

        # Brief cooldown
        time.sleep(2)

    # Analyze scaling efficiency
    print(f"\n{'='*80}")
    print("WORKER SCALING ANALYSIS:")
    print(f"{'='*80}")
    if len(results) > 0:
        baseline = results[0]
        baseline_throughput = baseline["games_per_minute"]
        baseline_workers = baseline["workers"]

        print(f"{'Workers':<10} {'Games/Min':<12} {'Speedup':<10} {'Efficiency':<12}")
        print("-" * 80)
        for r in results:
            speedup = r["games_per_minute"] / baseline_throughput
            ideal_speedup = r["workers"] / baseline_workers
            efficiency = (speedup / ideal_speedup) * 100
            print(f"{r['workers']:<10} {r['games_per_minute']:<12.1f} {speedup:<10.2f}x {efficiency:<12.1f}%")

    return results


def sweep_2d_interaction(
    worker_counts: List[int],
    batch_sizes: List[int],
    num_det: int,
    sims_per_det: int,
    games_per_config: int,
    device: str,
    batch_timeout_ms: float = 10.0,
    mode: str = "mp",
    cards_to_deal: int = 5,
) -> List[Dict]:
    """Sweep 3: 2D heatmap of workers × parallel_batch_size."""
    print("\n" + "="*80)
    print("SWEEP 3: 2D INTERACTION MATRIX")
    print("="*80)
    print(f"Workers: {worker_counts}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Total configurations: {len(worker_counts) * len(batch_sizes)}")
    print(f"Games per config: {games_per_config}")
    print("="*80)

    results = []
    for workers in worker_counts:
        for batch_size in batch_sizes:
            result = run_benchmark(
                workers=workers,
                num_determinizations=num_det,
                simulations_per_determinization=sims_per_det,
                games=games_per_config,
                device=device,
                use_parallel_expansion=True,
                parallel_batch_size=batch_size,
                batch_timeout_ms=batch_timeout_ms,
                mode=mode,
                cards_to_deal=cards_to_deal,
            )
            results.append(result)

            time.sleep(2)

    # Create heatmap data
    print(f"\n{'='*80}")
    print("2D PERFORMANCE HEATMAP (games/min):")
    print(f"{'='*80}")

    # Print header
    print(f"{'Workers':<10}", end="")
    for batch_size in batch_sizes:
        print(f"bs={batch_size:<8}", end="")
    print()
    print("-" * 80)

    # Print data
    for workers in worker_counts:
        print(f"{workers:<10}", end="")
        for batch_size in batch_sizes:
            # Find result
            matching = [r for r in results if r["workers"] == workers and r["parallel_batch_size"] == batch_size]
            if matching:
                print(f"{matching[0]['games_per_minute']:<10.1f}", end="")
            else:
                print(f"{'N/A':<10}", end="")
        print()

    return results


def sweep_mcts_configs(
    workers: int,
    parallel_batch_size: int,
    mcts_configs: List[Tuple[str, int, int]],
    games_per_config: int,
    device: str,
    batch_timeout_ms: float = 10.0,
    mode: str = "mp",
    cards_to_deal: int = 5,
) -> List[Dict]:
    """Sweep 4: MCTS configuration impact with optimal settings."""
    print("\n" + "="*80)
    print("SWEEP 4: MCTS CONFIGURATION COMPARISON")
    print("="*80)
    print(f"Fixed: workers={workers}, parallel_batch_size={parallel_batch_size}")
    print(f"Testing MCTS configs: {[c[0] for c in mcts_configs]}")
    print(f"Games per config: {games_per_config}")
    print("="*80)

    results = []
    for name, num_det, sims_per_det in mcts_configs:
        print(f"\n--- Testing {name} MCTS ---")
        result = run_benchmark(
            workers=workers,
            num_determinizations=num_det,
            simulations_per_determinization=sims_per_det,
            games=games_per_config,
            device=device,
            use_parallel_expansion=True,
            parallel_batch_size=parallel_batch_size,
            batch_timeout_ms=batch_timeout_ms,
            mode=mode,
            cards_to_deal=cards_to_deal,
        )
        result["mcts_config_name"] = name
        results.append(result)

        time.sleep(2)

    # Summary
    print(f"\n{'='*80}")
    print("MCTS CONFIGURATION SUMMARY:")
    print(f"{'='*80}")
    print(f"{'Config':<12} {'Sims/Move':<12} {'Games/Min':<12} {'Training Days':<15}")
    print("-" * 80)

    GAMES_NEEDED = 500 * 10_000  # 500 iterations × 10k games
    for r in results:
        training_days = GAMES_NEEDED / (r["games_per_minute"] * 60 * 24)
        print(f"{r['mcts_config_name']:<12} {r['total_sims_per_move']:<12} "
              f"{r['games_per_minute']:<12.1f} {training_days:<15.1f}")

    return results


def sweep_batch_timeout(
    workers: int,
    parallel_batch_size: int,
    num_det: int,
    sims_per_det: int,
    games_per_config: int,
    device: str,
    timeouts: List[float],
    mode: str = "mp",
    cards_to_deal: int = 5,
) -> List[Dict]:
    """Sweep 5: Batch timeout sensitivity."""
    print("\n" + "="*80)
    print("SWEEP 5: BATCH TIMEOUT SENSITIVITY")
    print("="*80)
    print(f"Testing timeouts (ms): {timeouts}")
    print(f"Fixed: workers={workers}, parallel_batch_size={parallel_batch_size}")
    print(f"Games per config: {games_per_config}")
    print("="*80)

    results = []
    for timeout in timeouts:
        result = run_benchmark(
            workers=workers,
            num_determinizations=num_det,
            simulations_per_determinization=sims_per_det,
            games=games_per_config,
            device=device,
            use_parallel_expansion=True,
            parallel_batch_size=parallel_batch_size,
            batch_timeout_ms=timeout,
            mode=mode,
            cards_to_deal=cards_to_deal,
        )
        results.append(result)

        time.sleep(2)

    # Find optimal
    best = max(results, key=lambda r: r["games_per_minute"])
    print(f"\n{'='*80}")
    print(f"OPTIMAL TIMEOUT: {best['batch_timeout_ms']}ms")
    print(f"  Performance: {best['games_per_minute']:.1f} games/min")
    print(f"{'='*80}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive validation benchmark for Session 1+2 optimizations"
    )
    parser.add_argument(
        "--sweep",
        choices=["all", "batch_size", "workers", "2d", "mcts", "timeout", "quick"],
        default="all",
        help="Which sweep to run (default: all)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="Games per configuration (default: 50)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--mode",
        choices=["mp", "threads", "gpu_server"],
        default="mp",
        help="Concurrency mode: mp=multiprocessing, threads=ThreadPool, gpu_server=central GPU server (default: mp)",
    )
    parser.add_argument(
        "--cards",
        type=int,
        default=5,
        help="Cards to deal per game (default: 5 to match baseline)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Comma-separated batch sizes for Sweep 1 (e.g., '5,10,20,40')",
    )

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        args.output = f"benchmarks/results/session1_validation_{timestamp}.csv"

    print("="*80)
    print("SESSION 1+2 OPTIMIZATION VALIDATION BENCHMARK")
    print("="*80)
    print(f"Sweep type: {args.sweep}")
    print(f"Games per config: {args.games}")
    print(f"Device: {args.device}")
    print(f"Mode: {args.mode}")
    print(f"Cards/game: {args.cards}")
    print(f"Output: {args.output}")
    print("="*80)

    # Collect all results
    all_results = []

    if args.sweep in ["all", "batch_size", "quick"]:
        # Sweep 1: parallel_batch_size (most important!)
        # Use exponential spacing to quickly cover broad spectrum
        if args.batch_sizes:
            batch_sizes_to_test = [int(x.strip()) for x in args.batch_sizes.split(',')]
        elif args.sweep == "quick":
            batch_sizes_to_test = [10, 20, 40]
        else:
            batch_sizes_to_test = [5, 10, 20, 40]

        batch_size_results = sweep_parallel_batch_size(
            workers=32,
            num_det=3,
            sims_per_det=30,
            games_per_config=args.games if args.sweep != "quick" else 20,
            device=args.device,
            batch_sizes=batch_sizes_to_test,
            batch_timeout_ms=10.0,
            mode=args.mode,
            cards_to_deal=args.cards,
        )
        all_results.extend(batch_size_results)

        # Find optimal for subsequent sweeps
        optimal_batch_size = max(batch_size_results, key=lambda r: r["games_per_minute"])["parallel_batch_size"]
    else:
        optimal_batch_size = 20  # Use reasonable default

    if args.sweep in ["all", "workers"]:
        # Sweep 2: Worker scaling
        worker_results = sweep_worker_scaling(
            worker_counts=[1, 4, 8, 16, 24, 32, 40, 48],
            num_det=3,
            sims_per_det=30,
            games_per_config=args.games,
            device=args.device,
            parallel_batch_size=optimal_batch_size,
            batch_timeout_ms=10.0,
            mode=args.mode,
            cards_to_deal=args.cards,
        )
        all_results.extend(worker_results)

    if args.sweep in ["all", "2d", "quick"]:
        # Sweep 3: 2D interaction matrix
        interaction_results = sweep_2d_interaction(
            worker_counts=[8, 16, 32] if args.sweep == "quick" else [8, 16, 24, 32, 40],
            batch_sizes=[10, 20, 30] if args.sweep == "quick" else [10, 15, 20, 25, 30],
            num_det=3,
            sims_per_det=30,
            games_per_config=20 if args.sweep == "quick" else 30,
            device=args.device,
            batch_timeout_ms=10.0,
            mode=args.mode,
            cards_to_deal=args.cards,
        )
        all_results.extend(interaction_results)

    if args.sweep in ["all", "mcts"]:
        # Sweep 4: MCTS configs
        mcts_results = sweep_mcts_configs(
            workers=32,
            parallel_batch_size=optimal_batch_size,
            mcts_configs=[
                ("Light", 2, 20),    # 40 sims/move
                ("Medium", 3, 30),   # 90 sims/move
                ("Heavy", 5, 50),    # 250 sims/move
            ],
            games_per_config=args.games,
            device=args.device,
            batch_timeout_ms=10.0,
            mode=args.mode,
            cards_to_deal=args.cards,
        )
        all_results.extend(mcts_results)

    if args.sweep in ["all", "timeout"]:
        # Sweep 5: Batch timeout
        timeout_results = sweep_batch_timeout(
            workers=32,
            parallel_batch_size=optimal_batch_size,
            num_det=3,
            sims_per_det=30,
            games_per_config=args.games,
            device=args.device,
            timeouts=[3.0, 5.0, 8.0, 10.0, 15.0],
            mode=args.mode,
            cards_to_deal=args.cards,
        )
        all_results.extend(timeout_results)

    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", newline="") as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    print(f"Results saved to: {args.output}")

    # Final summary
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Total configurations tested: {len(all_results)}")

    if all_results:
        best = max(all_results, key=lambda r: r["games_per_minute"])
        print(f"\nBEST CONFIGURATION FOUND:")
        print(f"  Workers: {best['workers']}")
        print(f"  Parallel batch size: {best['parallel_batch_size']}")
        print(f"  Performance: {best['games_per_minute']:.1f} games/min")
        print(f"  Speedup vs baseline (36.7 g/min): {best['games_per_minute'] / 36.7:.2f}x")

        GAMES_NEEDED = 500 * 10_000
        training_days = GAMES_NEEDED / (best["games_per_minute"] * 60 * 24)
        print(f"  Estimated training time: {training_days:.1f} days")


if __name__ == "__main__":
    main()
