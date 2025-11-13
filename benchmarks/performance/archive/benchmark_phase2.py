"""
Benchmark for Phase 2: Multi-Game Batched MCTS

This benchmark measures the real-world performance improvement from batched
evaluation when running multiple games in parallel (simulating self-play).

Test Scenarios:
    1. Sequential games with direct inference (baseline)
    2. Parallel games with direct inference
    3. Parallel games with BatchedEvaluator

Expected Results:
    - BatchedEvaluator should significantly outperform direct inference
    - GPU utilization should be higher with batching
    - Games/minute should increase 2-10x with batching
"""

import torch
import time
import threading
from typing import List

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.game.blob import BlobGame
from ml.mcts.batch_evaluator import BatchedEvaluator
from ml.mcts.search import MCTS


def generate_single_game(mcts, encoder):
    """Generate a single game using MCTS."""
    game = BlobGame(num_players=4)
    game.setup_round(cards_to_deal=5)

    # Run MCTS for each player's bid (4 decisions)
    for i in range(4):
        player = game.players[i]
        action_probs = mcts.search(game, player)

        # Select best bid
        best_bid = max(action_probs, key=action_probs.get)
        player.bid = best_bid

    return game


def benchmark_sequential_direct(network, encoder, masker, num_games=10):
    """Benchmark: Sequential games with direct inference."""
    print(f"\n=== Sequential Games (Direct Inference) ===")

    mcts = MCTS(
        network=network,
        encoder=encoder,
        masker=masker,
        num_simulations=25,  # Reduced for faster benchmarking
        batch_evaluator=None,
    )

    start = time.time()

    for i in range(num_games):
        generate_single_game(mcts, encoder)

    elapsed = time.time() - start
    games_per_min = (num_games / elapsed) * 60

    print(f"Games: {num_games}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Games/min: {games_per_min:.1f}")

    return games_per_min


def benchmark_parallel_direct(network, encoder, masker, num_games=10, num_threads=4):
    """Benchmark: Parallel games with direct inference."""
    print(f"\n=== Parallel Games (Direct Inference, {num_threads} threads) ===")

    # Each thread gets its own MCTS instance
    completed_games = []
    lock = threading.Lock()

    def worker(num_worker_games):
        mcts = MCTS(
            network=network,
            encoder=encoder,
            masker=masker,
            num_simulations=25,
            batch_evaluator=None,
        )

        for _ in range(num_worker_games):
            game = generate_single_game(mcts, encoder)
            with lock:
                completed_games.append(game)

    start = time.time()

    # Distribute games across threads
    games_per_thread = num_games // num_threads
    threads = []

    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(games_per_thread,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - start
    games_per_min = (len(completed_games) / elapsed) * 60

    print(f"Games: {len(completed_games)}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Games/min: {games_per_min:.1f}")

    return games_per_min


def benchmark_parallel_batched(network, encoder, masker, num_games=10, num_threads=4):
    """Benchmark: Parallel games with BatchedEvaluator."""
    print(f"\n=== Parallel Games (BatchedEvaluator, {num_threads} threads) ===")

    # Create shared BatchedEvaluator
    evaluator = BatchedEvaluator(
        network=network,
        max_batch_size=256,  # Batch up to 256 evaluations
        timeout_ms=5.0,      # 5ms timeout
    )
    evaluator.start()

    # Each thread shares the same evaluator
    completed_games = []
    lock = threading.Lock()

    def worker(num_worker_games):
        # Each MCTS instance uses the shared evaluator
        mcts = MCTS(
            network=network,
            encoder=encoder,
            masker=masker,
            num_simulations=25,
            batch_evaluator=evaluator,
        )

        for _ in range(num_worker_games):
            game = generate_single_game(mcts, encoder)
            with lock:
                completed_games.append(game)

    start = time.time()

    # Distribute games across threads
    games_per_thread = num_games // num_threads
    threads = []

    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(games_per_thread,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - start
    games_per_min = (len(completed_games) / elapsed) * 60

    # Get evaluator statistics
    stats = evaluator.get_stats()

    print(f"Games: {len(completed_games)}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Games/min: {games_per_min:.1f}")
    print(f"\nBatching Statistics:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Total batches: {stats['total_batches']}")
    print(f"  Avg batch size: {stats['avg_batch_size']:.1f}")
    print(f"  Batch efficiency: {stats['total_requests'] / max(stats['total_batches'], 1):.1f}x")

    evaluator.shutdown()

    return games_per_min


def run_benchmark():
    """Run complete benchmark suite."""
    print("=" * 70)
    print("Phase 2: Multi-Game Batched MCTS Benchmark")
    print("=" * 70)

    # Create network (BASELINE: 4.9M parameters)
    print("\nCreating neural network...")
    network = BlobNet(
        state_dim=256,
        embedding_dim=256,      # Baseline: 256 (not 128!)
        num_layers=6,           # Baseline: 6 (not 2!)
        num_heads=8,            # Baseline: 8 (not 4!)
        feedforward_dim=1024,   # Baseline: 1024 (not 256!)
        dropout=0.0,
    )
    network.eval()

    encoder = StateEncoder()
    masker = ActionMasker()

    # Benchmark parameters
    num_games = 5  # Total games to generate (quick screening)
    num_threads = 4  # Parallel threads

    print(f"\nBenchmark Configuration:")
    print(f"  Total games: {num_games}")
    print(f"  Parallel threads: {num_threads}")
    print(f"  MCTS simulations: 25 per decision")
    print(f"  Network: 2-layer Transformer (~1M params)")

    # Run benchmarks
    results = {}

    # 1. Sequential baseline
    results['sequential'] = benchmark_sequential_direct(
        network, encoder, masker, num_games=num_games
    )

    # 2. Parallel without batching
    results['parallel_direct'] = benchmark_parallel_direct(
        network, encoder, masker, num_games=num_games, num_threads=num_threads
    )

    # 3. Parallel with batching
    results['parallel_batched'] = benchmark_parallel_batched(
        network, encoder, masker, num_games=num_games, num_threads=num_threads
    )

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\n{'Method':<30} {'Games/min':<15} {'Speedup':<10}")
    print("-" * 70)

    baseline = results['sequential']

    for method, games_per_min in results.items():
        speedup = games_per_min / baseline
        method_name = {
            'sequential': 'Sequential (Direct)',
            'parallel_direct': f'Parallel x{num_threads} (Direct)',
            'parallel_batched': f'Parallel x{num_threads} (Batched)',
        }[method]

        print(f"{method_name:<30} {games_per_min:>10.1f}     {speedup:>6.2f}x")

    print("\n" + "=" * 70)

    # Calculate improvement from batching
    parallel_improvement = results['parallel_batched'] / results['parallel_direct']
    print(f"\nBatching improvement over parallel direct: {parallel_improvement:.2f}x")

    if parallel_improvement > 1.5:
        print("\n[SUCCESS] Phase 2 batching provides significant speedup!")
    elif parallel_improvement > 1.1:
        print("\n[OK] Phase 2 batching provides moderate speedup")
    else:
        print("\n[WARNING] Phase 2 batching provides minimal speedup")
        print("Note: Batching works best with GPU and many concurrent requests")

    print("=" * 70)


def test_run_benchmark():
    """Wrapper for pytest."""
    run_benchmark()


if __name__ == "__main__":
    run_benchmark()
