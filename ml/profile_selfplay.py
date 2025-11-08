"""
Profiling script for self-play pipeline.

This script profiles the self-play game generation to identify bottlenecks
and understand where time is actually being spent.

Profiles:
1. Overall function-level profiling (cProfile)
2. Time spent in different components (MCTS, network, game logic)
3. Queue/synchronization overhead
"""

import cProfile
import pstats
import io
import time
import torch
from pathlib import Path
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.training.selfplay import SelfPlayEngine


def profile_selfplay_with_cprofile(
    num_workers=32,
    num_games=10,
    cards_to_deal=3,
    use_batched=True,
    use_threads=False,
    use_parallel_expansion=True,
    parallel_batch_size=30,
    device="cuda"
):
    """
    Profile self-play using cProfile.

    Args:
        num_workers: Number of workers
        num_games: Number of games to generate
        cards_to_deal: Cards per player
        use_batched: Use BatchedEvaluator
        use_threads: Use ThreadPoolExecutor
        use_parallel_expansion: Enable parallel MCTS expansion
        parallel_batch_size: Batch size for parallel expansion
        device: Device to use
    """
    print("="*80)
    print(f"Profiling Self-Play with cProfile")
    print(f"  Workers: {num_workers}")
    print(f"  Games: {num_games}")
    print(f"  Cards: {cards_to_deal}")
    print(f"  Batched: {use_batched}")
    print(f"  Threads: {use_threads}")
    print(f"  Parallel Expansion: {use_parallel_expansion}")
    print(f"  Parallel Batch Size: {parallel_batch_size}")
    print("="*80)

    # Create network
    network = BlobNet(
        state_dim=256,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        feedforward_dim=1024,
        dropout=0.0,
    )
    network.to(device)
    network.eval()

    encoder = StateEncoder()
    masker = ActionMasker()

    # Create engine
    engine = SelfPlayEngine(
        network=network,
        encoder=encoder,
        masker=masker,
        num_workers=num_workers,
        num_determinizations=3,
        simulations_per_determinization=30,
        device=device,
        use_batched_evaluator=use_batched,
        batch_size=512,
        batch_timeout_ms=10.0,
        use_thread_pool=use_threads,
        use_parallel_expansion=use_parallel_expansion,
        parallel_batch_size=parallel_batch_size,
    )

    # Profile the generation
    profiler = cProfile.Profile()

    print("\nStarting profiling...")
    profiler.enable()

    examples = engine.generate_games(
        num_games=num_games,
        num_players=4,
        cards_to_deal=cards_to_deal
    )

    profiler.disable()
    print(f"Generated {len(examples)} examples")

    # Print statistics
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)

    print("\n" + "="*80)
    print("TOP 30 FUNCTIONS BY CUMULATIVE TIME")
    print("="*80)
    stats.sort_stats('cumulative')
    stats.print_stats(30)
    print(s.getvalue())

    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)

    print("\n" + "="*80)
    print("TOP 30 FUNCTIONS BY TOTAL TIME")
    print("="*80)
    stats.sort_stats('tottime')
    stats.print_stats(30)
    print(s.getvalue())

    # Save profile to file
    profile_file = f"profile_w{num_workers}_g{num_games}_c{cards_to_deal}.prof"
    profiler.dump_stats(profile_file)
    print(f"\nProfile saved to: {profile_file}")
    print(f"View with: python -m pstats {profile_file}")

    # Cleanup
    engine.shutdown()

    return stats


def manual_timing_profile(
    num_workers=32,
    num_games=10,
    cards_to_deal=3,
    device="cuda"
):
    """
    Manual timing profile to understand component breakdown.

    This profiles specific parts of the pipeline to understand where
    time is being spent.
    """
    print("\n" + "="*80)
    print(f"Manual Timing Profile")
    print("="*80)

    # Create network
    network = BlobNet(
        state_dim=256,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        feedforward_dim=1024,
        dropout=0.0,
    )
    network.to(device)
    network.eval()

    encoder = StateEncoder()
    masker = ActionMasker()

    # Time different configurations
    # Format: (name, use_batched, use_threads, use_parallel_expansion, parallel_batch_size)
    configs = [
        ("Direct (no batching, threads)", False, True, False, 10),
        ("Batched (threads)", True, True, False, 10),
        ("Batched (processes)", True, False, False, 10),
        ("Sessions 1+2 Optimized (batched + parallel expansion)", True, False, True, 30),
    ]

    results = []

    for name, use_batched, use_threads, use_parallel_expansion, parallel_batch_size in configs:
        print(f"\nTesting: {name}")

        engine = SelfPlayEngine(
            network=network,
            encoder=encoder,
            masker=masker,
            num_workers=num_workers,
            num_determinizations=3,
            simulations_per_determinization=30,
            device=device,
            use_batched_evaluator=use_batched,
            batch_size=512,
            batch_timeout_ms=10.0,
            use_thread_pool=use_threads,
            use_parallel_expansion=use_parallel_expansion,
            parallel_batch_size=parallel_batch_size,
        )

        # Warm-up
        engine.generate_games(num_games=2, num_players=4, cards_to_deal=cards_to_deal)

        # Timed run
        start = time.time()
        examples = engine.generate_games(
            num_games=num_games,
            num_players=4,
            cards_to_deal=cards_to_deal
        )
        elapsed = time.time() - start

        games_per_min = (num_games / elapsed) * 60

        result = {
            'name': name,
            'elapsed': elapsed,
            'games_per_min': games_per_min,
            'examples': len(examples),
        }

        if use_batched and engine.batch_evaluator:
            stats = engine.batch_evaluator.get_stats()
            result['avg_batch_size'] = stats['avg_batch_size']
            result['total_batches'] = stats['total_batches']

        results.append(result)

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Games/min: {games_per_min:.1f}")
        print(f"  Examples: {len(examples)}")

        if 'avg_batch_size' in result:
            print(f"  Avg batch size: {result['avg_batch_size']:.1f}")

        engine.shutdown()

    # Summary
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)

    print(f"\n{'Configuration':<30} {'Games/min':<15} {'Speedup':<10}")
    print("-"*60)

    baseline = results[0]['games_per_min']

    for r in results:
        speedup = r['games_per_min'] / baseline if baseline > 0 else 0
        print(f"{r['name']:<30} {r['games_per_min']:<15.1f} {speedup:.2f}x")

    return results


def profile_sessions_1_2_config(
    num_workers=32,
    num_games=50,
    cards_to_deal=5,  # Fixed: was 3, should be 5 to match 75.85 games/min benchmark
    device="cuda"
):
    """
    Profile the exact Sessions 1+2 optimized configuration (75.85 games/min).

    This is the validated best-performing setup from session1_validation_20251106_1951.csv:
    - 32 workers (multiprocessing)
    - Medium MCTS: 3 determinizations × 30 simulations
    - 5 cards per game (NOT 3!)
    - Cross-worker batching via BatchedEvaluator
    - Parallel MCTS expansion with virtual loss
    - parallel_batch_size=30

    Args:
        num_workers: Number of parallel workers (default: 32)
        num_games: Number of games to profile (default: 50)
        cards_to_deal: Cards per player (default: 3)
        device: Device to use (default: "cuda")

    Returns:
        Dict with timing statistics and throughput metrics
    """
    print("\n" + "="*80)
    print("PROFILING SESSIONS 1+2 OPTIMIZED CONFIGURATION")
    print("="*80)
    print(f"  Workers: {num_workers}")
    print(f"  Games: {num_games}")
    print(f"  Cards: {cards_to_deal}")
    print(f"  MCTS: 3 det × 30 sims (Medium)")
    print(f"  Batching: Enabled (512 batch size, 10ms timeout)")
    print(f"  Parallel Expansion: Enabled (batch_size=30)")
    print(f"  Threading: Disabled (multiprocessing)")
    print("="*80)

    # Create network
    network = BlobNet(
        state_dim=256,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        feedforward_dim=1024,
        dropout=0.0,
    )
    network.to(device)
    network.eval()

    encoder = StateEncoder()
    masker = ActionMasker()

    # Create engine with exact Sessions 1+2 configuration
    engine = SelfPlayEngine(
        network=network,
        encoder=encoder,
        masker=masker,
        num_workers=num_workers,
        num_determinizations=3,
        simulations_per_determinization=30,
        device=device,
        use_batched_evaluator=True,
        batch_size=512,
        batch_timeout_ms=10.0,
        use_thread_pool=False,  # multiprocessing
        use_parallel_expansion=True,
        parallel_batch_size=30,
    )

    print("\nWarm-up (2 games)...")
    engine.generate_games(num_games=2, num_players=4, cards_to_deal=cards_to_deal)

    print(f"Running timed test ({num_games} games)...")
    start = time.time()
    examples = engine.generate_games(
        num_games=num_games,
        num_players=4,
        cards_to_deal=cards_to_deal
    )
    elapsed = time.time() - start

    games_per_min = (num_games / elapsed) * 60

    # Get batch evaluator stats
    stats = {}
    if engine.batch_evaluator:
        batch_stats = engine.batch_evaluator.get_stats()
        stats['avg_batch_size'] = batch_stats['avg_batch_size']
        stats['total_batches'] = batch_stats['total_batches']
        stats['total_evals'] = batch_stats['total_requests']  # Fixed: use total_requests instead of total_evals

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Games/min: {games_per_min:.2f}")
    print(f"  Examples generated: {len(examples)}")

    if stats:
        print(f"\nBatch Evaluator Stats:")
        print(f"  Total evaluations: {stats.get('total_evals', 0)}")
        print(f"  Total batches: {stats.get('total_batches', 0)}")
        print(f"  Avg batch size: {stats.get('avg_batch_size', 0):.1f}")

    # Compare to expected performance
    expected_games_per_min = 75.85
    performance_ratio = games_per_min / expected_games_per_min

    print(f"\nPerformance vs Expected (75.85 games/min):")
    print(f"  Ratio: {performance_ratio:.2f}x")

    if performance_ratio < 0.9:
        print(f"  ⚠️  WARNING: Performance is {(1-performance_ratio)*100:.1f}% below expected!")
        print(f"      This profiling may not reflect the actual best configuration.")
    elif performance_ratio > 0.95:
        print(f"  ✅ Performance is within expected range")

    print("="*80)

    engine.shutdown()

    return {
        'elapsed': elapsed,
        'games_per_min': games_per_min,
        'examples': len(examples),
        'expected_games_per_min': expected_games_per_min,
        'performance_ratio': performance_ratio,
        **stats
    }


def analyze_batch_evaluator_overhead(device="cuda"):
    """
    Specifically analyze BatchedEvaluator overhead.

    Tests if queue operations and synchronization are causing slowdowns.
    """
    print("\n" + "="*80)
    print("Batch Evaluator Overhead Analysis")
    print("="*80)

    from ml.mcts.batch_evaluator import BatchedEvaluator
    import queue
    import threading

    # Create network
    network = BlobNet(
        state_dim=256,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        feedforward_dim=1024,
        dropout=0.0,
    )
    network.to(device)
    network.eval()

    encoder = StateEncoder()
    masker = ActionMasker()

    # Test 1: Direct network calls (baseline)
    print("\nTest 1: Direct network calls (baseline)")

    from ml.game.blob import BlobGame

    game = BlobGame(num_players=4)
    game.setup_round(cards_to_deal=3)
    player = game.players[0]

    num_calls = 1000

    start = time.time()
    for _ in range(num_calls):
        state = encoder.encode(game, player)
        mask = masker.create_bidding_mask(3, False, None)
        state = state.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            policy, value = network(state, mask)

    direct_time = time.time() - start
    direct_per_call = (direct_time / num_calls) * 1000  # ms

    print(f"  {num_calls} calls: {direct_time:.3f}s")
    print(f"  Per call: {direct_per_call:.3f}ms")
    print(f"  Calls/sec: {num_calls / direct_time:.0f}")

    # Test 2: Through BatchedEvaluator
    print("\nTest 2: Through BatchedEvaluator (single thread)")

    evaluator = BatchedEvaluator(
        network=network,
        max_batch_size=512,
        timeout_ms=10.0,
        device=device
    )
    evaluator.start()

    start = time.time()
    for _ in range(num_calls):
        state = encoder.encode(game, player)
        mask = masker.create_bidding_mask(3, False, None)

        policy, value = evaluator.evaluate(state, mask)

    batched_time = time.time() - start
    batched_per_call = (batched_time / num_calls) * 1000  # ms

    stats = evaluator.get_stats()

    print(f"  {num_calls} calls: {batched_time:.3f}s")
    print(f"  Per call: {batched_per_call:.3f}ms")
    print(f"  Calls/sec: {num_calls / batched_time:.0f}")
    print(f"  Avg batch size: {stats['avg_batch_size']:.1f}")
    print(f"  Total batches: {stats['total_batches']}")

    overhead = ((batched_time - direct_time) / direct_time) * 100
    print(f"\n  Overhead: {overhead:.1f}%")

    evaluator.shutdown()

    # Test 3: Multiple threads hitting evaluator
    print("\nTest 3: Multiple threads (4) through BatchedEvaluator")

    evaluator = BatchedEvaluator(
        network=network,
        max_batch_size=512,
        timeout_ms=10.0,
        device=device
    )
    evaluator.start()

    calls_per_thread = num_calls // 4
    barrier = threading.Barrier(4)

    def worker():
        barrier.wait()  # Synchronize start

        for _ in range(calls_per_thread):
            state = encoder.encode(game, player)
            mask = masker.create_bidding_mask(3, False, None)
            policy, value = evaluator.evaluate(state, mask)

    threads = [threading.Thread(target=worker) for _ in range(4)]

    start = time.time()

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    multi_time = time.time() - start
    total_calls = calls_per_thread * 4

    stats = evaluator.get_stats()

    print(f"  {total_calls} calls (4 threads): {multi_time:.3f}s")
    print(f"  Per call: {(multi_time / total_calls) * 1000:.3f}ms")
    print(f"  Calls/sec: {total_calls / multi_time:.0f}")
    print(f"  Avg batch size: {stats['avg_batch_size']:.1f}")
    print(f"  Total batches: {stats['total_batches']}")

    speedup = direct_time / multi_time
    print(f"\n  Speedup vs direct: {speedup:.2f}x")

    evaluator.shutdown()

    return {
        'direct_per_call_ms': direct_per_call,
        'batched_per_call_ms': batched_per_call,
        'overhead_percent': overhead,
        'multi_speedup': speedup,
    }


def main():
    """Run all profiling tests."""

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This profiling requires a GPU.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # 0. Validate Sessions 1+2 configuration performance
    print("\n" + "="*80)
    print("PHASE 0: Sessions 1+2 Configuration Validation")
    print("="*80)

    sessions_result = profile_sessions_1_2_config(
        num_workers=32,
        num_games=50,
        cards_to_deal=5,  # Fixed: Match benchmark config (5 cards, not 3)
        device="cuda"
    )

    # Warn if performance is significantly below expected
    if sessions_result['performance_ratio'] < 0.9:
        print("\n⚠️  WARNING: Performance is below expected!")
        print("The profiling results may not reflect the actual best configuration.")
        response = input("\nContinue with remaining profiling phases? (y/n): ")
        if response.lower() != 'y':
            print("Profiling aborted.")
            return

    # 1. cProfile analysis (using Sessions 1+2 config)
    print("\n" + "="*80)
    print("PHASE 1: cProfile Analysis (Sessions 1+2 Config)")
    print("="*80)

    profile_selfplay_with_cprofile(
        num_workers=32,
        num_games=10,
        cards_to_deal=5,  # Fixed: Match benchmark config (5 cards, not 3)
        use_batched=True,
        use_threads=False,
        use_parallel_expansion=True,
        parallel_batch_size=30,
        device="cuda"
    )

    # 2. Manual timing
    print("\n" + "="*80)
    print("PHASE 2: Manual Timing Profile (Compare Configs)")
    print("="*80)

    manual_timing_profile(
        num_workers=32,
        num_games=10,
        cards_to_deal=5,  # Fixed: Match benchmark config (5 cards, not 3)
        device="cuda"
    )

    # 3. Batch evaluator overhead
    print("\n" + "="*80)
    print("PHASE 3: Batch Evaluator Overhead")
    print("="*80)

    analyze_batch_evaluator_overhead(device="cuda")

    print("\n" + "="*80)
    print("Profiling complete!")
    print("="*80)


if __name__ == "__main__":
    main()
