"""
Benchmark Script for GPU-Batched MCTS

This script benchmarks the GPU-batched parallel expansion implementation
against baseline and Phase 1 implementations to measure performance gains.

Configurations Tested:
    1. Baseline: Sequential MCTS (no batching)
    2. Phase 1: Intra-game batched MCTS (batch_size=90)
    3. GPU-Batched: Cross-worker batched MCTS (parallel_batch_size=10)

Performance Metrics:
    - Games per minute
    - Average batch size (GPU utilization indicator)
    - GPU utilization percentage
    - Training time estimate for 500 iterations

Expected Results:
    - Baseline: ~32 games/min
    - Phase 1: ~56 games/min (1.76x speedup)
    - GPU-Batched: 160-320 games/min (5-10x speedup)

Usage:
    # Quick test (5 games)
    python benchmarks/performance/benchmark_gpu_batch_mcts.py --games 5

    # Full validation (50 games)
    python benchmarks/performance/benchmark_gpu_batch_mcts.py --games 50

    # Sweep multiple configs
    python benchmarks/performance/benchmark_gpu_batch_mcts.py --sweep
"""

import argparse
import time
import csv
import sys
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Any
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.training.selfplay import SelfPlayWorker
from ml.mcts.batch_evaluator import BatchedEvaluator
from ml.mcts.gpu_server import GPUInferenceServer


def create_network(device='cuda'):
    """Create test network."""
    network = BlobNet(
        state_dim=256,
        embedding_dim=128,
        num_layers=4,
        num_heads=4,
        feedforward_dim=512,
        dropout=0.0,
    )
    network.to(device)
    network.eval()
    return network


def benchmark_baseline(num_games: int = 5) -> Dict[str, Any]:
    """
    Benchmark baseline (sequential MCTS, no batching).

    Args:
        num_games: Number of games to generate

    Returns:
        Dict with performance metrics
    """
    print(f"\n=== Benchmarking Baseline (Sequential) ===")
    print(f"  Games: {num_games}")
    print(f"  Config: num_det=3, sims_per_det=30")

    network = create_network(device='cuda')
    encoder = StateEncoder()
    masker = ActionMasker()

    worker = SelfPlayWorker(
        network=network,
        encoder=encoder,
        masker=masker,
        num_determinizations=3,
        simulations_per_determinization=30,
        use_imperfect_info=True,
        use_parallel_expansion=False,
        batch_size=None,  # No batching
    )

    start_time = time.time()
    for i in range(num_games):
        worker.generate_game(num_players=4, cards_to_deal=5)
        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            games_per_min = (i + 1) / elapsed * 60
            print(f"  Progress: {i+1}/{num_games} games ({games_per_min:.1f} games/min)")

    elapsed = time.time() - start_time
    games_per_min = num_games / elapsed * 60

    return {
        'config': 'Baseline (Sequential)',
        'num_games': num_games,
        'elapsed_sec': elapsed,
        'games_per_min': games_per_min,
        'avg_batch_size': 1.0,
        'speedup': 1.0,
    }


def benchmark_phase1(num_games: int = 5) -> Dict[str, Any]:
    """
    Benchmark Phase 1 (intra-game batching).

    Args:
        num_games: Number of games to generate

    Returns:
        Dict with performance metrics
    """
    print(f"\n=== Benchmarking Phase 1 (Intra-Game Batching) ===")
    print(f"  Games: {num_games}")
    print(f"  Config: num_det=3, sims_per_det=30, batch_size=90")

    network = create_network(device='cuda')
    encoder = StateEncoder()
    masker = ActionMasker()

    worker = SelfPlayWorker(
        network=network,
        encoder=encoder,
        masker=masker,
        num_determinizations=3,
        simulations_per_determinization=30,
        use_imperfect_info=True,
        use_parallel_expansion=False,
        batch_size=90,  # Intra-game batching
    )

    start_time = time.time()
    for i in range(num_games):
        worker.generate_game(num_players=4, cards_to_deal=5)
        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            games_per_min = (i + 1) / elapsed * 60
            print(f"  Progress: {i+1}/{num_games} games ({games_per_min:.1f} games/min)")

    elapsed = time.time() - start_time
    games_per_min = num_games / elapsed * 60

    return {
        'config': 'Phase 1 (Intra-Game Batching)',
        'num_games': num_games,
        'elapsed_sec': elapsed,
        'games_per_min': games_per_min,
        'avg_batch_size': 90.0,  # Approximate
        'speedup': games_per_min / 32.0,  # vs baseline of 32 games/min
    }


def worker_process(
    worker_id: int,
    num_games: int,
    gpu_client,
    results_queue: mp.Queue,
):
    """
    Worker process for GPU-batched benchmark.

    Args:
        worker_id: Worker identifier
        num_games: Number of games to generate
        gpu_client: GPUServerClient instance (created in main process)
        results_queue: Queue to put results
    """
    try:
        encoder = StateEncoder()
        masker = ActionMasker()

        # Create worker with GPU-batched parallel expansion
        worker = SelfPlayWorker(
            network=None,  # Network is on GPU server
            encoder=encoder,
            masker=masker,
            num_determinizations=3,
            simulations_per_determinization=30,
            use_imperfect_info=True,
            use_parallel_expansion=True,
            parallel_batch_size=10,  # Cross-worker batching
            gpu_server_client=gpu_client,
        )

        # Generate games
        for _ in range(num_games):
            worker.generate_game(num_players=4, cards_to_deal=5)

        results_queue.put({'worker_id': worker_id, 'status': 'success'})

    except Exception as e:
        results_queue.put({'worker_id': worker_id, 'status': 'error', 'error': str(e)})


def benchmark_gpu_batched(
    num_games: int = 5,
    num_workers: int = 4,
    parallel_batch_size: int = 10,
    max_batch_size: int = 512,
    timeout_ms: float = 5.0,
) -> Dict[str, Any]:
    """
    Benchmark GPU-batched MCTS (cross-worker batching).

    Args:
        num_games: Number of games per worker
        num_workers: Number of parallel workers
        parallel_batch_size: Leaves to expand per iteration
        max_batch_size: Maximum GPU batch size
        timeout_ms: Batch collection timeout

    Returns:
        Dict with performance metrics
    """
    print(f"\n=== Benchmarking GPU-Batched MCTS ===")
    print(f"  Games: {num_games} per worker × {num_workers} workers = {num_games * num_workers} total")
    print(f"  Config: parallel_batch_size={parallel_batch_size}, max_batch={max_batch_size}, timeout={timeout_ms}ms")

    # Create GPU server
    network = create_network(device='cuda')
    gpu_server = GPUInferenceServer(
        network=network,
        device='cuda',
        max_batch_size=max_batch_size,
        timeout_ms=timeout_ms,
    )
    gpu_server.start()

    try:
        # Create results queue
        results_queue = mp.Queue()

        # Create GPU clients for each worker (must be done before spawning processes)
        clients = [gpu_server.create_client(client_id=f"worker_{i}") for i in range(num_workers)]

        # Start workers
        processes = []
        start_time = time.time()

        for worker_id in range(num_workers):
            p = mp.Process(
                target=worker_process,
                args=(worker_id, num_games, clients[worker_id], results_queue),
            )
            p.start()
            processes.append(p)

        # Wait for workers to complete
        for p in processes:
            p.join()

        elapsed = time.time() - start_time

        # Collect results
        successful_workers = 0
        while not results_queue.empty():
            result = results_queue.get()
            if result['status'] == 'success':
                successful_workers += 1
            else:
                print(f"  Worker {result['worker_id']} failed: {result['error']}")

        # Get GPU server stats
        stats = gpu_server.get_stats()
        avg_batch_size = stats.get('avg_batch_size', 0.0)
        max_batch = stats.get('max_batch_size', 0)

        total_games = num_games * successful_workers
        games_per_min = total_games / elapsed * 60

        print(f"\n  Results:")
        print(f"    Total games: {total_games}")
        print(f"    Elapsed: {elapsed:.1f}s")
        print(f"    Games/min: {games_per_min:.1f}")
        print(f"    Avg batch size: {avg_batch_size:.1f}")
        print(f"    Max batch size: {max_batch}")

        return {
            'config': f'GPU-Batched (workers={num_workers}, pb={parallel_batch_size})',
            'num_games': total_games,
            'num_workers': num_workers,
            'parallel_batch_size': parallel_batch_size,
            'max_batch_size': max_batch_size,
            'timeout_ms': timeout_ms,
            'elapsed_sec': elapsed,
            'games_per_min': games_per_min,
            'avg_batch_size': avg_batch_size,
            'max_batch': max_batch,
            'speedup': games_per_min / 32.0,  # vs baseline of 32 games/min
        }

    finally:
        gpu_server.shutdown()


def run_sweep():
    """
    Run parameter sweep to find optimal configuration.

    Tests different combinations of:
    - parallel_batch_size: [5, 10, 20]
    - timeout_ms: [2, 5, 10]
    """
    print("\n=== Running Parameter Sweep ===")

    results = []

    # Test different configurations
    configs = [
        {'parallel_batch_size': 5, 'timeout_ms': 2.0},
        {'parallel_batch_size': 10, 'timeout_ms': 5.0},
        {'parallel_batch_size': 20, 'timeout_ms': 10.0},
    ]

    for config in configs:
        result = benchmark_gpu_batched(
            num_games=3,  # Quick test
            num_workers=4,
            **config,
        )
        results.append(result)

    # Find best configuration
    best = max(results, key=lambda r: r['games_per_min'])
    print(f"\n=== Best Configuration ===")
    print(f"  Parallel batch size: {best['parallel_batch_size']}")
    print(f"  Timeout: {best['timeout_ms']}ms")
    print(f"  Games/min: {best['games_per_min']:.1f}")
    print(f"  Avg batch size: {best['avg_batch_size']:.1f}")

    return results


def save_results(results: List[Dict[str, Any]], filename: str):
    """Save benchmark results to CSV."""
    output_path = Path('results') / filename
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark GPU-Batched MCTS')
    parser.add_argument('--games', type=int, default=5, help='Games per worker')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--parallel-batch-size', type=int, default=10, help='Parallel batch size')
    parser.add_argument('--max-batch-size', type=int, default=512, help='Max GPU batch size')
    parser.add_argument('--timeout-ms', type=float, default=5.0, help='Batch timeout (ms)')
    parser.add_argument('--sweep', action='store_true', help='Run parameter sweep')
    parser.add_argument('--compare', action='store_true', help='Compare all configurations')
    parser.add_argument('--output', type=str, default='gpu_batch_mcts_benchmark.csv', help='Output CSV filename')

    args = parser.parse_args()

    if args.sweep:
        results = run_sweep()
    elif args.compare:
        # Run all three configurations for comparison
        print("\n=== Running Full Comparison ===")
        results = []

        # Baseline
        baseline = benchmark_baseline(num_games=args.games)
        results.append(baseline)

        # Phase 1
        phase1 = benchmark_phase1(num_games=args.games)
        results.append(phase1)

        # GPU-Batched
        gpu_batched = benchmark_gpu_batched(
            num_games=args.games,
            num_workers=args.workers,
            parallel_batch_size=args.parallel_batch_size,
            max_batch_size=args.max_batch_size,
            timeout_ms=args.timeout_ms,
        )
        results.append(gpu_batched)

        # Print comparison
        print("\n=== Performance Comparison ===")
        print(f"{'Config':<40} {'Games/Min':<15} {'Speedup':<10} {'Batch Size':<15}")
        print("-" * 80)
        for r in results:
            print(f"{r['config']:<40} {r['games_per_min']:<15.1f} {r['speedup']:<10.2f}x {r['avg_batch_size']:<15.1f}")

    else:
        # Single GPU-batched run
        results = [benchmark_gpu_batched(
            num_games=args.games,
            num_workers=args.workers,
            parallel_batch_size=args.parallel_batch_size,
            max_batch_size=args.max_batch_size,
            timeout_ms=args.timeout_ms,
        )]

    # Save results
    save_results(results, args.output)

    print("\n=== Benchmark Complete ===")


if __name__ == '__main__':
    # Set multiprocessing start method for Windows
    mp.set_start_method('spawn', force=True)
    main()
