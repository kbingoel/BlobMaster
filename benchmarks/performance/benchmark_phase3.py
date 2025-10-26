"""
Benchmark for Phase 3: ThreadPoolExecutor with Shared BatchedEvaluator

This benchmark compares Phase 2 (multiprocessing) vs Phase 3 (threads) to demonstrate
the performance improvement from true cross-worker batching.

Test Scenarios:
    1. Phase 2: multiprocessing.Pool with per-worker BatchedEvaluators
    2. Phase 3: ThreadPoolExecutor with single shared BatchedEvaluator

Expected Results:
    - Phase 3 should achieve 10-20x speedup over Phase 2
    - Batch sizes should be 50-100x larger (3.5 → 128-512)
    - GPU utilization should be 70-90% (vs 5-10%)
    - Games/minute should be 1000-2000 (vs 96.7)
"""

import torch
import time
import sys
from pathlib import Path

# Add parent directory to path for ml module imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.training.selfplay import SelfPlayEngine


def create_test_network(device="cuda"):
    """Create BASELINE network (4.9M parameters) for benchmarking."""
    network = BlobNet(
        state_dim=256,
        embedding_dim=256,      # Baseline: 256 (not 128!)
        num_layers=6,           # Baseline: 6 (not 2!)
        num_heads=8,            # Baseline: 8 (not 4!)
        feedforward_dim=1024,   # Baseline: 1024 (not 256!)
        dropout=0.0,  # No dropout for consistent timing
    )
    network.to(device)
    network.eval()
    return network


def benchmark_phase2_multiprocess(network, encoder, masker, num_games=20, num_workers=4, device="cuda"):
    """
    Benchmark Phase 2: multiprocessing.Pool with per-worker BatchedEvaluators.

    Expected:
        - Small batch sizes (~3-5 avg)
        - Low GPU utilization (5-10%)
        - Poor performance due to overhead
    """
    print("\n" + "=" * 70)
    print("PHASE 2: multiprocessing.Pool + Per-Worker BatchedEvaluators")
    print("=" * 70)

    engine = SelfPlayEngine(
        network=network,
        encoder=encoder,
        masker=masker,
        num_workers=num_workers,
        num_determinizations=3,
        simulations_per_determinization=10,  # Reduced for faster benchmark
        device=device,
        use_batched_evaluator=True,
        batch_size=512,
        batch_timeout_ms=10.0,
        use_thread_pool=False,  # Force multiprocessing
    )

    # Warm-up (JIT compilation, cache warming)
    print("Warming up...")
    engine.generate_games(num_games=2, num_players=4, cards_to_deal=3)

    # Reset evaluator stats
    if engine.batch_evaluator:
        engine.batch_evaluator.reset_stats()

    # Benchmark
    print(f"Generating {num_games} games with {num_workers} workers...")
    start_time = time.time()

    examples = engine.generate_games(num_games=num_games, num_players=4, cards_to_deal=3)

    elapsed = time.time() - start_time

    # Get stats
    if engine.batch_evaluator:
        stats = engine.batch_evaluator.get_stats()
        avg_batch_size = stats['avg_batch_size']
        total_batches = stats['total_batches']
        total_requests = stats['total_requests']
    else:
        avg_batch_size = 1.0
        total_batches = 0
        total_requests = 0

    # Calculate metrics
    games_per_min = (num_games / elapsed) * 60
    examples_per_game = len(examples) / num_games if num_games > 0 else 0

    # Display results
    print(f"\nResults:")
    print(f"  Games generated: {num_games}")
    print(f"  Total examples: {len(examples)}")
    print(f"  Examples/game: {examples_per_game:.1f}")
    print(f"  Time elapsed: {elapsed:.2f}s")
    print(f"  Games/min: {games_per_min:.1f}")
    print(f"\nBatching Stats:")
    print(f"  Total requests: {total_requests}")
    print(f"  Total batches: {total_batches}")
    print(f"  Avg batch size: {avg_batch_size:.1f}")

    # Cleanup
    engine.shutdown()

    return {
        'games_per_min': games_per_min,
        'avg_batch_size': avg_batch_size,
        'total_batches': total_batches,
        'elapsed': elapsed,
    }


def benchmark_phase3_threaded(network, encoder, masker, num_games=20, num_workers=4, device="cuda"):
    """
    Benchmark Phase 3: ThreadPoolExecutor with single shared BatchedEvaluator.

    Expected:
        - Large batch sizes (128-512 avg)
        - High GPU utilization (70-90%)
        - Excellent performance (10-20x speedup)
    """
    print("\n" + "=" * 70)
    print("PHASE 3: ThreadPoolExecutor + Shared BatchedEvaluator")
    print("=" * 70)

    engine = SelfPlayEngine(
        network=network,
        encoder=encoder,
        masker=masker,
        num_workers=num_workers,
        num_determinizations=3,
        simulations_per_determinization=10,  # Reduced for faster benchmark
        device=device,
        use_batched_evaluator=True,
        batch_size=1024,  # Larger for cross-worker batching
        batch_timeout_ms=10.0,
        use_thread_pool=True,  # Force threading (Phase 3)
    )

    # Warm-up (JIT compilation, cache warming)
    print("Warming up...")
    engine.generate_games(num_games=2, num_players=4, cards_to_deal=3)

    # Reset evaluator stats
    if engine.batch_evaluator:
        engine.batch_evaluator.reset_stats()

    # Benchmark
    print(f"Generating {num_games} games with {num_workers} workers...")
    start_time = time.time()

    examples = engine.generate_games(num_games=num_games, num_players=4, cards_to_deal=3)

    elapsed = time.time() - start_time

    # Get stats
    if engine.batch_evaluator:
        stats = engine.batch_evaluator.get_stats()
        avg_batch_size = stats['avg_batch_size']
        total_batches = stats['total_batches']
        total_requests = stats['total_requests']
    else:
        avg_batch_size = 1.0
        total_batches = 0
        total_requests = 0

    # Calculate metrics
    games_per_min = (num_games / elapsed) * 60
    examples_per_game = len(examples) / num_games if num_games > 0 else 0

    # Display results
    print(f"\nResults:")
    print(f"  Games generated: {num_games}")
    print(f"  Total examples: {len(examples)}")
    print(f"  Examples/game: {examples_per_game:.1f}")
    print(f"  Time elapsed: {elapsed:.2f}s")
    print(f"  Games/min: {games_per_min:.1f}")
    print(f"\nBatching Stats:")
    print(f"  Total requests: {total_requests}")
    print(f"  Total batches: {total_batches}")
    print(f"  Avg batch size: {avg_batch_size:.1f}")

    # Cleanup
    engine.shutdown()

    return {
        'games_per_min': games_per_min,
        'avg_batch_size': avg_batch_size,
        'total_batches': total_batches,
        'elapsed': elapsed,
    }


def run_comparison(num_games=5, num_workers=4, device="cuda"):
    """Run comprehensive comparison of Phase 2 vs Phase 3."""
    print("\n" + "=" * 70)
    print("PHASE 3 BENCHMARK: Multiprocessing vs Threading")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Workers: {num_workers}")
    print(f"  Games: {num_games}")
    print(f"  Cards/game: 3")
    print(f"  Determinizations: 3")
    print(f"  Simulations/det: 10")

    # Create network and components
    network = create_test_network(device=device)
    encoder = StateEncoder()
    masker = ActionMasker()

    # Run benchmarks
    phase2_results = benchmark_phase2_multiprocess(
        network, encoder, masker, num_games=num_games, num_workers=num_workers, device=device
    )

    phase3_results = benchmark_phase3_threaded(
        network, encoder, masker, num_games=num_games, num_workers=num_workers, device=device
    )

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Phase 2':<20} {'Phase 3':<20} {'Speedup':<15}")
    print("-" * 85)

    metrics = [
        ('Games/min', 'games_per_min', '.1f'),
        ('Avg batch size', 'avg_batch_size', '.1f'),
        ('Total batches', 'total_batches', 'd'),
        ('Time (seconds)', 'elapsed', '.2f'),
    ]

    for label, key, fmt in metrics:
        phase2_val = phase2_results[key]
        phase3_val = phase3_results[key]

        # Calculate speedup (handle division by zero)
        if key == 'total_batches':
            # Fewer batches is better (inverse speedup)
            speedup = phase2_val / phase3_val if phase3_val > 0 else 0
            speedup_str = f"{speedup:.2f}x fewer"
        elif key == 'elapsed':
            # Lower time is better (inverse speedup)
            speedup = phase2_val / phase3_val if phase3_val > 0 else 0
            speedup_str = f"{speedup:.2f}x faster"
        else:
            # Higher is better
            speedup = phase3_val / phase2_val if phase2_val > 0 else 0
            speedup_str = f"{speedup:.2f}x"

        print(f"{label:<30} {phase2_val:{fmt}:<20} {phase3_val:{fmt}:<20} {speedup_str:<15}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    games_speedup = phase3_results['games_per_min'] / phase2_results['games_per_min']
    batch_improvement = phase3_results['avg_batch_size'] / phase2_results['avg_batch_size']

    print(f"\nPhase 3 achieves:")
    print(f"  - {games_speedup:.1f}x speedup in games/minute")
    print(f"  - {batch_improvement:.1f}x larger batch sizes")
    print(f"  - {phase3_results['avg_batch_size']:.0f} average batch size (target: >128)")

    if games_speedup >= 10:
        print(f"\n[SUCCESS] Phase 3 meets 10x speedup target!")
    elif games_speedup >= 5:
        print(f"\n[GOOD] Phase 3 shows {games_speedup:.1f}x speedup (target: 10x)")
    else:
        print(f"\n[NEEDS WORK] Phase 3 only shows {games_speedup:.1f}x speedup (target: 10x)")

    if phase3_results['avg_batch_size'] >= 128:
        print(f"[SUCCESS] Average batch size meets target (>128)!")
    else:
        print(f"[NEEDS WORK] Average batch size is {phase3_results['avg_batch_size']:.0f} (target: >128)")


if __name__ == "__main__":
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Running on CPU (slower, not representative of GPU performance).")
        device = "cpu"
    else:
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(0)}")

    # Run comprehensive benchmark
    run_comparison(num_games=20, num_workers=4, device=device)
