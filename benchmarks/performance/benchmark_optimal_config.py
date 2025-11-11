#!/usr/bin/env python3
"""
Benchmark Optimal Configuration for Training

Tests the exact configuration validated in profiling (2025-11-11) to measure
accurate training throughput for estimating full training run time.

This benchmark uses the OPTIMAL settings from profiling analysis:
- 32 workers (multiprocessing, not threads)
- Medium MCTS: 3 determinizations × 30 simulations = 90 total
- Parallel MCTS expansion enabled (critical optimization!)
- Parallel batch size: 30 (optimal from Session 1+2)
- Batched evaluator: 512 max batch, 10ms timeout
- 5 cards per game (matches training config)

Expected performance: ~350-400 games/min (based on profiling)

Usage:
    python benchmarks/performance/benchmark_optimal_config.py --games 1000
    python benchmarks/performance/benchmark_optimal_config.py --games 500 --test-mcts-variants
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.training.selfplay import SelfPlayEngine


def create_optimal_network(device: str = "cuda"):
    """Create network with optimal settings."""
    network = BlobNet(
        state_dim=256,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        feedforward_dim=1024,
        dropout=0.0,  # No dropout for inference
    )
    network.to(device)
    network.eval()
    return network


def benchmark_optimal_config(
    num_games: int = 1000,
    device: str = "cuda",
    warmup_games: int = 5,
) -> dict:
    """
    Benchmark the optimal configuration.

    Args:
        num_games: Number of games to generate (default: 1000)
        device: Device to use (default: cuda)
        warmup_games: Number of warmup games (default: 5)

    Returns:
        Dictionary with benchmark results
    """
    print("\n" + "="*80)
    print("OPTIMAL CONFIGURATION BENCHMARK")
    print("="*80)
    print(f"Configuration validated from profiling (2025-11-11):")
    print(f"  Workers: 32 (multiprocessing)")
    print(f"  MCTS: 3 det × 30 sims = 90 total (Medium)")
    print(f"  Parallel expansion: ENABLED")
    print(f"  Parallel batch size: 30")
    print(f"  Batched evaluator: 512 max batch, 10ms timeout")
    print(f"  Cards per game: 5")
    print(f"\nTest parameters:")
    print(f"  Games to generate: {num_games}")
    print(f"  Warmup games: {warmup_games}")
    print(f"  Device: {device}")
    print("="*80)

    # Create network
    print("\nInitializing network...")
    network = create_optimal_network(device)
    encoder = StateEncoder()
    masker = ActionMasker()

    # Create engine with optimal settings
    print("Creating self-play engine...")
    engine = SelfPlayEngine(
        network=network,
        encoder=encoder,
        masker=masker,
        num_workers=32,
        num_determinizations=3,
        simulations_per_determinization=30,
        device=device,
        use_batched_evaluator=True,
        batch_size=512,
        batch_timeout_ms=10.0,
        use_thread_pool=False,  # Use multiprocessing, not threads!
        use_parallel_expansion=True,  # Critical optimization!
        parallel_batch_size=30,  # Optimal value from Session 1+2
    )

    # Warmup
    print(f"\nRunning warmup ({warmup_games} games)...")
    warmup_start = time.time()
    engine.generate_games(num_games=warmup_games, num_players=4, cards_to_deal=5)
    warmup_elapsed = time.time() - warmup_start
    print(f"  Warmup completed in {warmup_elapsed:.2f}s")

    # Main benchmark
    print(f"\nRunning benchmark ({num_games} games)...")
    start_time = time.time()

    examples = engine.generate_games(
        num_games=num_games,
        num_players=4,
        cards_to_deal=5,
    )

    elapsed = time.time() - start_time

    # Calculate metrics
    games_per_minute = (num_games / elapsed) * 60
    seconds_per_game = elapsed / num_games
    examples_per_minute = (len(examples) / elapsed) * 60
    examples_per_game = len(examples) / num_games

    # Training time estimates
    total_games = 500 * 10_000  # 500 iterations × 10,000 games
    training_days = (total_games / games_per_minute) / (60 * 24)

    # Get batch evaluator stats if available
    batch_stats = {}
    if engine.batch_evaluator:
        batch_stats = engine.batch_evaluator.get_stats()

    # Cleanup
    engine.shutdown()
    del engine, network
    torch.cuda.empty_cache()

    # Results
    results = {
        "timestamp": datetime.now().isoformat(),
        "num_games": num_games,
        "elapsed_seconds": elapsed,
        "games_per_minute": games_per_minute,
        "seconds_per_game": seconds_per_game,
        "total_examples": len(examples),
        "examples_per_game": examples_per_game,
        "examples_per_minute": examples_per_minute,
        "warmup_games": warmup_games,
        "warmup_seconds": warmup_elapsed,
        "training_estimate_days": training_days,
        "config": {
            "workers": 32,
            "num_determinizations": 3,
            "simulations_per_determinization": 30,
            "total_sims_per_move": 90,
            "use_parallel_expansion": True,
            "parallel_batch_size": 30,
            "batch_size": 512,
            "batch_timeout_ms": 10.0,
            "cards_per_game": 5,
        },
        "batch_evaluator_stats": batch_stats,
    }

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Games generated: {num_games}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Games/minute: {games_per_minute:.2f}")
    print(f"Seconds/game: {seconds_per_game:.3f}")
    print(f"Training examples: {len(examples):,}")
    print(f"Examples/game: {examples_per_game:.1f}")
    print(f"Examples/minute: {examples_per_minute:.1f}")

    if batch_stats:
        print(f"\nBatch Evaluator Stats:")
        print(f"  Total evaluations: {batch_stats.get('total_requests', 0):,}")
        print(f"  Total batches: {batch_stats.get('total_batches', 0):,}")
        print(f"  Avg batch size: {batch_stats.get('avg_batch_size', 0):.1f}")
        if 'min_batch_size' in batch_stats:
            print(f"  Min/Max batch: {batch_stats['min_batch_size']}/{batch_stats['max_batch_size']}")

    print(f"\n{'='*80}")
    print("TRAINING TIME ESTIMATE")
    print("="*80)
    print(f"Full training (500 iterations × 10,000 games):")
    print(f"  Total games: {total_games:,}")
    print(f"  Estimated time: {training_days:.1f} days")
    print(f"  At rate: {games_per_minute:.1f} games/min")
    print("="*80)

    return results


def benchmark_mcts_variants(num_games: int = 200, device: str = "cuda"):
    """
    Compare Light, Medium, and Heavy MCTS configurations.

    Args:
        num_games: Number of games per config (default: 200)
        device: Device to use

    Returns:
        List of result dictionaries
    """
    configs = [
        ("Light", 2, 20, 40),
        ("Medium", 3, 30, 90),
        ("Heavy", 5, 50, 250),
    ]

    results = []

    for name, num_det, sims_per_det, total_sims in configs:
        print("\n" + "="*80)
        print(f"Testing {name} MCTS ({num_det} det × {sims_per_det} sims = {total_sims} total)")
        print("="*80)

        network = create_optimal_network(device)
        encoder = StateEncoder()
        masker = ActionMasker()

        engine = SelfPlayEngine(
            network=network,
            encoder=encoder,
            masker=masker,
            num_workers=32,
            num_determinizations=num_det,
            simulations_per_determinization=sims_per_det,
            device=device,
            use_batched_evaluator=True,
            batch_size=512,
            batch_timeout_ms=10.0,
            use_thread_pool=False,
            use_parallel_expansion=True,
            parallel_batch_size=30,
        )

        # Warmup
        engine.generate_games(num_games=2, num_players=4, cards_to_deal=5)

        # Benchmark
        start = time.time()
        examples = engine.generate_games(num_games=num_games, num_players=4, cards_to_deal=5)
        elapsed = time.time() - start

        games_per_min = (num_games / elapsed) * 60
        training_days = (500 * 10_000 / games_per_min) / (60 * 24)

        result = {
            "mcts_config": name,
            "num_determinizations": num_det,
            "simulations_per_det": sims_per_det,
            "total_sims": total_sims,
            "games": num_games,
            "elapsed": elapsed,
            "games_per_minute": games_per_min,
            "training_days": training_days,
            "examples": len(examples),
        }
        results.append(result)

        print(f"\nResults:")
        print(f"  Games/minute: {games_per_min:.1f}")
        print(f"  Training estimate: {training_days:.1f} days")

        engine.shutdown()
        del engine, network
        torch.cuda.empty_cache()

        time.sleep(2)  # Cooldown

    # Print comparison
    print("\n" + "="*80)
    print("MCTS CONFIGURATION COMPARISON")
    print("="*80)
    print(f"{'Config':<12} {'Sims/Move':<12} {'Games/Min':<15} {'Training (days)':<18} {'Speedup'}")
    print("-"*80)

    baseline = results[0]['games_per_minute']
    for r in results:
        speedup = baseline / r['games_per_minute']
        print(f"{r['mcts_config']:<12} {r['total_sims']:<12} "
              f"{r['games_per_minute']:<15.1f} {r['training_days']:<18.1f} {speedup:.2f}x")

    print("-"*80)

    return results


def main():
    """Run benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark optimal configuration for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1000,
        help="Number of games to generate (default: 1000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--test-mcts-variants",
        action="store_true",
        help="Also test Light and Heavy MCTS configs for comparison",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_optimal_results.json",
        help="Output JSON path (default: benchmark_optimal_results.json)",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        print("ERROR: CUDA not available. Use --device cpu")
        sys.exit(1)

    # Run main benchmark
    results = benchmark_optimal_config(
        num_games=args.games,
        device=args.device,
    )

    # Optionally test MCTS variants
    if args.test_mcts_variants:
        print("\n\n" + "#"*80)
        print("TESTING MCTS CONFIGURATION VARIANTS")
        print("#"*80)

        mcts_results = benchmark_mcts_variants(
            num_games=min(args.games, 200),  # Limit for variant testing
            device=args.device,
        )
        results['mcts_variants'] = mcts_results

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    # Also save to CSV for easy viewing
    csv_path = output_path.with_suffix('.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'timestamp', 'num_games', 'games_per_minute', 'seconds_per_game',
            'total_examples', 'examples_per_game', 'training_estimate_days'
        ])
        writer.writeheader()
        writer.writerow({
            'timestamp': results['timestamp'],
            'num_games': results['num_games'],
            'games_per_minute': results['games_per_minute'],
            'seconds_per_game': results['seconds_per_game'],
            'total_examples': results['total_examples'],
            'examples_per_game': results['examples_per_game'],
            'training_estimate_days': results['training_estimate_days'],
        })

    print(f"CSV summary saved to: {csv_path}")

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
