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
import argparse
from pathlib import Path
import sys
import uuid
import json
import glob


# ---------------------------
# Aggregation helper routines
# ---------------------------

def _aggregate_worker_metrics(run_id: str) -> dict:
    pattern = f"profile_{run_id}_*_metrics.json"
    files = glob.glob(pattern)
    agg = {
        'workers': 0,
        'determinization': {
            'sample_determinization_calls': 0,
            'samples_succeeded': 0,
            'attempts_total': 0,
            'sample_determinization_total_sec': 0.0,
            'validate_calls': 0,
            'validate_total_sec': 0.0,
        },
        'node': {
            'simulate_action_calls': 0,
            'simulate_action_total_sec': 0.0,
        },
        'batch_evaluator': {
            'total_requests': 0,
            'total_batches': 0,
            'total_batch_size': 0,
            'min_batch_size': None,
            'max_batch_size': None,
            'total_infer_time_ms': 0.0,
        }
    }

    for fp in files:
        try:
            with open(fp, 'r') as f:
                data = json.load(f)
        except Exception:
            continue
        agg['workers'] += 1
        d = data.get('determinization', {})
        n = data.get('node', {})
        be = data.get('batch_evaluator') or {}

        for k in agg['determinization']:
            agg['determinization'][k] += d.get(k, 0)
        for k in agg['node']:
            agg['node'][k] += n.get(k, 0)

        # Merge batch evaluator stats (sum totals, min/max across workers)
        agg['batch_evaluator']['total_requests'] += int(be.get('total_requests', 0))
        agg['batch_evaluator']['total_batches'] += int(be.get('total_batches', 0))
        # total_batch_size may not be directly in stats; derive from avg * count if needed
        if 'total_batch_size' in be:
            agg['batch_evaluator']['total_batch_size'] += int(be.get('total_batch_size', 0))
        else:
            agg['batch_evaluator']['total_batch_size'] += int(be.get('avg_batch_size', 0) * be.get('total_batches', 0))
        if be.get('min_batch_size') is not None:
            mb = be.get('min_batch_size')
            cur = agg['batch_evaluator']['min_batch_size']
            agg['batch_evaluator']['min_batch_size'] = mb if cur is None else min(cur, mb)
        if be.get('max_batch_size') is not None:
            xb = be.get('max_batch_size')
            curx = agg['batch_evaluator']['max_batch_size']
            agg['batch_evaluator']['max_batch_size'] = xb if curx is None else max(curx, xb)
        agg['batch_evaluator']['total_infer_time_ms'] += float(be.get('total_infer_time_ms', 0.0))

    # Derive averages
    det = agg['determinization']
    calls = det['sample_determinization_calls'] or 0
    successes = det['samples_succeeded'] or 0
    validate_calls = det['validate_calls'] or 0
    agg['determinization']['avg_attempts_per_call'] = (det['attempts_total'] / calls) if calls else 0.0
    agg['determinization']['avg_attempts_per_success'] = (det['attempts_total'] / successes) if successes else 0.0
    agg['determinization']['avg_sample_ms'] = ((det['sample_determinization_total_sec'] / (calls or 1)) * 1000.0)
    agg['determinization']['avg_validate_ms'] = ((det['validate_total_sec'] / (validate_calls or 1)) * 1000.0) if validate_calls else 0.0

    node = agg['node']
    node_calls = node['simulate_action_calls'] or 0
    agg['node']['avg_simulate_action_ms'] = ((node['simulate_action_total_sec'] / (node_calls or 1)) * 1000.0)

    be = agg['batch_evaluator']
    if be['total_batches'] > 0:
        be['avg_batch_size'] = be['total_batch_size'] / be['total_batches']
    else:
        be['avg_batch_size'] = 0.0
    if be['total_requests'] > 0:
        be['avg_infer_per_item_us'] = (be['total_infer_time_ms'] / 1000.0) / be['total_requests'] * 1e6
    else:
        be['avg_infer_per_item_us'] = 0.0

    return agg


def _print_and_save_aggregate(run_id: str) -> None:
    agg = _aggregate_worker_metrics(run_id)
    if agg.get('workers', 0) == 0:
        print(f"No instrumentation metrics found for run_id={run_id}")
        return
    print("\n" + "=" * 80)
    print(f"INSTRUMENTATION SUMMARY (run {run_id})")
    print("=" * 80)
    det = agg['determinization']
    print("Determinization:")
    print(f"  calls={det['sample_determinization_calls']} successes={det['samples_succeeded']} attempts={det['attempts_total']}")
    print(f"  avg_attempts_per_call={det['avg_attempts_per_call']:.2f} avg_attempts_per_success={det['avg_attempts_per_success']:.2f}")
    print(f"  avg_sample_ms={det['avg_sample_ms']:.2f} avg_validate_ms={det['avg_validate_ms']:.2f}")
    node = agg['node']
    print("Node simulate_action:")
    print(f"  calls={node['simulate_action_calls']} avg_ms={node['avg_simulate_action_ms']:.3f}")
    be = agg['batch_evaluator']
    print("Batch evaluator (per-worker totals combined):")
    print(f"  total_requests={be['total_requests']} total_batches={be['total_batches']} avg_batch_size={be['avg_batch_size']:.1f}")
    if be['min_batch_size'] is not None:
        print(f"  min_batch={be['min_batch_size']} max_batch={be['max_batch_size']}")
    print(f"  total_infer_time_ms={be['total_infer_time_ms']:.1f} avg_infer_per_item_us={be['avg_infer_per_item_us']:.1f}")

    out_path = f"profile_{run_id}_aggregate.json"
    with open(out_path, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"Aggregate metrics saved to {out_path}")

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
    device="cuda",
    enable_metrics=False,
    run_id=None,
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
        enable_worker_metrics=enable_metrics,
        run_id=run_id,
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
    device="cuda",
    enable_metrics=False,
    run_id=None,
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
            enable_worker_metrics=enable_metrics,
            run_id=run_id,
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
    device="cuda",
    enable_metrics=False,
    run_id=None,
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
        enable_worker_metrics=enable_metrics,
        run_id=run_id,
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

    # Synchronize GPU to measure actual execution time, not just kernel launch
    if device == "cuda":
        torch.cuda.synchronize()

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

    # Synchronize GPU to measure actual execution time (includes async .cpu() transfers)
    if device == "cuda":
        torch.cuda.synchronize()

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

        # Synchronize GPU at end of worker to ensure all operations complete
        if device == "cuda":
            torch.cuda.synchronize()

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


def profile_with_and_without_worker_profiling(
    num_workers=32,
    num_games=50,
    cards_to_deal=5,
    device="cuda",
    enable_metrics=False,
    run_id=None,
):
    """
    Compare performance WITH and WITHOUT worker profiling enabled.

    This function runs the same configuration twice:
    1. With worker profiling disabled (baseline performance)
    2. With worker profiling enabled (measures overhead)

    The difference reveals the actual overhead cost of cProfile on worker 0.

    Args:
        num_workers: Number of parallel workers (default: 32)
        num_games: Number of games to profile (default: 50)
        cards_to_deal: Cards per player (default: 5)
        device: Device to use (default: "cuda")

    Returns:
        Dict with comparison statistics
    """
    print("\n" + "="*80)
    print("WORKER PROFILING OVERHEAD COMPARISON")
    print("="*80)
    print(f"  Workers: {num_workers}")
    print(f"  Games: {num_games}")
    print(f"  Cards: {cards_to_deal}")
    print(f"  MCTS: 3 det × 30 sims (Medium)")
    print(f"  Device: {device}")
    print("="*80)

    # Create network once (shared for both runs)
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

    results = {}

    # ========================================================================
    # Run 1: WITHOUT worker profiling (baseline)
    # ========================================================================
    print("\n" + "-"*80)
    print("RUN 1: Worker Profiling DISABLED (baseline)")
    print("-"*80)

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
        use_thread_pool=False,
        use_parallel_expansion=True,
        parallel_batch_size=30,
        enable_worker_profiling=False,  # DISABLED
        enable_worker_metrics=enable_metrics,
        run_id=run_id,
    )

    print("Warm-up (2 games)...")
    engine.generate_games(num_games=2, num_players=4, cards_to_deal=cards_to_deal)

    print(f"Running timed test ({num_games} games)...")
    start = time.time()
    examples = engine.generate_games(
        num_games=num_games,
        num_players=4,
        cards_to_deal=cards_to_deal
    )
    elapsed_without = time.time() - start

    games_per_min_without = (num_games / elapsed_without) * 60

    print(f"  Time: {elapsed_without:.2f}s")
    print(f"  Games/min: {games_per_min_without:.2f}")
    print(f"  Examples: {len(examples)}")

    engine.shutdown()

    # ========================================================================
    # Run 2: WITH worker profiling (overhead measurement)
    # ========================================================================
    print("\n" + "-"*80)
    print("RUN 2: Worker Profiling ENABLED (measuring overhead)")
    print("-"*80)

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
        use_thread_pool=False,
        use_parallel_expansion=True,
        parallel_batch_size=30,
        enable_worker_profiling=True,  # ENABLED
        enable_worker_metrics=enable_metrics,
        run_id=run_id,
    )

    print("Warm-up (2 games)...")
    engine.generate_games(num_games=2, num_players=4, cards_to_deal=cards_to_deal)

    print(f"Running timed test ({num_games} games)...")
    start = time.time()
    examples = engine.generate_games(
        num_games=num_games,
        num_players=4,
        cards_to_deal=cards_to_deal
    )
    elapsed_with = time.time() - start

    games_per_min_with = (num_games / elapsed_with) * 60

    print(f"  Time: {elapsed_with:.2f}s")
    print(f"  Games/min: {games_per_min_with:.2f}")
    print(f"  Examples: {len(examples)}")

    engine.shutdown()

    # ========================================================================
    # Calculate overhead
    # ========================================================================
    overhead_seconds = elapsed_with - elapsed_without
    overhead_percent = (overhead_seconds / elapsed_without) * 100
    throughput_ratio = games_per_min_without / games_per_min_with if games_per_min_with > 0 else 0
    throughput_loss_percent = ((games_per_min_without - games_per_min_with) / games_per_min_without) * 100

    print("\n" + "="*80)
    print("OVERHEAD ANALYSIS")
    print("="*80)
    print(f"\nWithout profiling:")
    print(f"  Time: {elapsed_without:.2f}s")
    print(f"  Games/min: {games_per_min_without:.2f}")

    print(f"\nWith profiling:")
    print(f"  Time: {elapsed_with:.2f}s")
    print(f"  Games/min: {games_per_min_with:.2f}")

    print(f"\nOverhead:")
    print(f"  Additional time: {overhead_seconds:.2f}s ({overhead_percent:.1f}%)")
    print(f"  Throughput loss: {throughput_loss_percent:.1f}%")
    print(f"  Throughput ratio: {throughput_ratio:.3f}x")

    if overhead_percent < 5:
        print(f"\n  ✅ Profiling overhead is minimal (<5%)")
    elif overhead_percent < 10:
        print(f"\n  ⚠️  Profiling overhead is moderate (5-10%)")
    else:
        print(f"\n  🔴 Profiling overhead is significant (>10%)")
        print(f"      Consider disabling worker profiling for production benchmarks")

    print("="*80)

    return {
        'elapsed_without_profiling': elapsed_without,
        'elapsed_with_profiling': elapsed_with,
        'games_per_min_without': games_per_min_without,
        'games_per_min_with': games_per_min_with,
        'overhead_seconds': overhead_seconds,
        'overhead_percent': overhead_percent,
        'throughput_loss_percent': throughput_loss_percent,
        'throughput_ratio': throughput_ratio,
    }


def main():
    """Run all profiling tests."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Profile BlobMaster self-play pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run default profiling phases (no worker profiling)
  python ml/profile_selfplay.py

  # Compare overhead with and without worker profiling
  python ml/profile_selfplay.py --compare-overhead

  # Run custom overhead comparison
  python ml/profile_selfplay.py --compare-overhead --workers 16 --games 100
"""
    )

    parser.add_argument(
        '--compare-overhead',
        action='store_true',
        help='Run overhead comparison (with/without worker profiling) instead of default phases'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=32,
        help='Number of workers for overhead comparison (default: 32)'
    )

    parser.add_argument(
        '--games',
        type=int,
        default=50,
        help='Number of games for overhead comparison (default: 50)'
    )

    parser.add_argument(
        '--cards',
        type=int,
        default=5,
        help='Cards per player for overhead comparison (default: 5)'
    )

    parser.add_argument(
        '--instrument',
        action='store_true',
        help='Enable lightweight instrumentation and per-worker JSON metrics'
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This profiling requires a GPU.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # If --compare-overhead flag is set, run overhead comparison and exit
    if args.compare_overhead:
        run_id = uuid.uuid4().hex[:8]
        profile_with_and_without_worker_profiling(
            num_workers=args.workers,
            num_games=args.games,
            cards_to_deal=args.cards,
            device="cuda",
            enable_metrics=args.instrument,
            run_id=run_id,
        )
        if args.instrument:
            _print_and_save_aggregate(run_id)
        return

    # 0. Validate Sessions 1+2 configuration performance
    print("\n" + "="*80)
    print("PHASE 0: Sessions 1+2 Configuration Validation")
    print("="*80)

    run_id_0 = uuid.uuid4().hex[:8]
    sessions_result = profile_sessions_1_2_config(
        num_workers=32,
        num_games=50,
        cards_to_deal=5,  # Fixed: Match benchmark config (5 cards, not 3)
        device="cuda",
        enable_metrics=args.instrument,
        run_id=run_id_0,
    )
    if args.instrument:
        _print_and_save_aggregate(run_id_0)

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

    run_id_1 = uuid.uuid4().hex[:8]
    profile_selfplay_with_cprofile(
        num_workers=32,
        num_games=10,
        cards_to_deal=5,  # Fixed: Match benchmark config (5 cards, not 3)
        use_batched=True,
        use_threads=False,
        use_parallel_expansion=True,
        parallel_batch_size=30,
        device="cuda",
        enable_metrics=args.instrument,
        run_id=run_id_1,
    )
    if args.instrument:
        _print_and_save_aggregate(run_id_1)

    # 2. Manual timing
    print("\n" + "="*80)
    print("PHASE 2: Manual Timing Profile (Compare Configs)")
    print("="*80)

    run_id_2 = uuid.uuid4().hex[:8]
    manual_timing_profile(
        num_workers=32,
        num_games=10,
        cards_to_deal=5,  # Fixed: Match benchmark config (5 cards, not 3)
        device="cuda",
        enable_metrics=args.instrument,
        run_id=run_id_2,
    )
    if args.instrument:
        _print_and_save_aggregate(run_id_2)

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


# (duplicate aggregation helpers removed; see definitions near top of file)

def _aggregate_worker_metrics(run_id: str) -> dict:
    pattern = f"profile_{run_id}_*_metrics.json"
    files = glob.glob(pattern)
    agg = {
        'workers': 0,
        'determinization': {
            'sample_determinization_calls': 0,
            'samples_succeeded': 0,
            'attempts_total': 0,
            'sample_determinization_total_sec': 0.0,
            'validate_calls': 0,
            'validate_total_sec': 0.0,
        },
        'node': {
            'simulate_action_calls': 0,
            'simulate_action_total_sec': 0.0,
        },
        'batch_evaluator': {
            'total_requests': 0,
            'total_batches': 0,
            'total_batch_size': 0,
            'min_batch_size': None,
            'max_batch_size': None,
            'total_infer_time_ms': 0.0,
        }
    }

    for fp in files:
        try:
            with open(fp, 'r') as f:
                data = json.load(f)
        except Exception:
            continue
        agg['workers'] += 1
        d = data.get('determinization', {})
        n = data.get('node', {})
        be = data.get('batch_evaluator') or {}

        for k in agg['determinization']:
            agg['determinization'][k] += d.get(k, 0)
        for k in agg['node']:
            agg['node'][k] += n.get(k, 0)

        # Merge batch evaluator stats (sum totals, min/max across workers)
        agg['batch_evaluator']['total_requests'] += int(be.get('total_requests', 0))
        agg['batch_evaluator']['total_batches'] += int(be.get('total_batches', 0))
        # total_batch_size may not be directly in stats; derive from avg * count if needed
        if 'total_batch_size' in be:
            agg['batch_evaluator']['total_batch_size'] += int(be.get('total_batch_size', 0))
        else:
            agg['batch_evaluator']['total_batch_size'] += int(be.get('avg_batch_size', 0) * be.get('total_batches', 0))
        if be.get('min_batch_size') is not None:
            mb = be.get('min_batch_size')
            cur = agg['batch_evaluator']['min_batch_size']
            agg['batch_evaluator']['min_batch_size'] = mb if cur is None else min(cur, mb)
        if be.get('max_batch_size') is not None:
            xb = be.get('max_batch_size')
            curx = agg['batch_evaluator']['max_batch_size']
            agg['batch_evaluator']['max_batch_size'] = xb if curx is None else max(curx, xb)
        agg['batch_evaluator']['total_infer_time_ms'] += float(be.get('total_infer_time_ms', 0.0))

    # Derive averages
    det = agg['determinization']
    calls = det['sample_determinization_calls'] or 0
    successes = det['samples_succeeded'] or 0
    validate_calls = det['validate_calls'] or 0
    agg['determinization']['avg_attempts_per_call'] = (det['attempts_total'] / calls) if calls else 0.0
    agg['determinization']['avg_attempts_per_success'] = (det['attempts_total'] / successes) if successes else 0.0
    agg['determinization']['avg_sample_ms'] = ((det['sample_determinization_total_sec'] / (calls or 1)) * 1000.0)
    agg['determinization']['avg_validate_ms'] = ((det['validate_total_sec'] / (validate_calls or 1)) * 1000.0) if validate_calls else 0.0

    node = agg['node']
    node_calls = node['simulate_action_calls'] or 0
    agg['node']['avg_simulate_action_ms'] = ((node['simulate_action_total_sec'] / (node_calls or 1)) * 1000.0)

    be = agg['batch_evaluator']
    if be['total_batches'] > 0:
        be['avg_batch_size'] = be['total_batch_size'] / be['total_batches']
    else:
        be['avg_batch_size'] = 0.0
    if be['total_requests'] > 0:
        be['avg_infer_per_item_us'] = (be['total_infer_time_ms'] / 1000.0) / be['total_requests'] * 1e6
    else:
        be['avg_infer_per_item_us'] = 0.0

    return agg


def _print_and_save_aggregate(run_id: str) -> None:
    agg = _aggregate_worker_metrics(run_id)
    if agg.get('workers', 0) == 0:
        print(f"No instrumentation metrics found for run_id={run_id}")
        return
    print("\n" + "=" * 80)
    print(f"INSTRUMENTATION SUMMARY (run {run_id})")
    print("=" * 80)
    det = agg['determinization']
    print("Determinization:")
    print(f"  calls={det['sample_determinization_calls']} successes={det['samples_succeeded']} attempts={det['attempts_total']}")
    print(f"  avg_attempts_per_call={det['avg_attempts_per_call']:.2f} avg_attempts_per_success={det['avg_attempts_per_success']:.2f}")
    print(f"  avg_sample_ms={det['avg_sample_ms']:.2f} avg_validate_ms={det['avg_validate_ms']:.2f}")
    node = agg['node']
    print("Node simulate_action:")
    print(f"  calls={node['simulate_action_calls']} avg_ms={node['avg_simulate_action_ms']:.3f}")
    be = agg['batch_evaluator']
    print("Batch evaluator (per-worker totals combined):")
    print(f"  total_requests={be['total_requests']} total_batches={be['total_batches']} avg_batch_size={be['avg_batch_size']:.1f}")
    if be['min_batch_size'] is not None:
        print(f"  min_batch={be['min_batch_size']} max_batch={be['max_batch_size']}")
    print(f"  total_infer_time_ms={be['total_infer_time_ms']:.1f} avg_infer_per_item_us={be['avg_infer_per_item_us']:.1f}")

    out_path = f"profile_{run_id}_aggregate.json"
    with open(out_path, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"Aggregate metrics saved to {out_path}")
