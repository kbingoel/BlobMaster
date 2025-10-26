"""
Windows-Optimized Performance Tests

Windows has specific limitations that affect the original action plan:
1. Multiprocessing handle limit: Max 63 processes on Windows
2. Memory overhead: Each process loads full network copy (20MB × workers)
3. VRAM duplication: Each process may allocate GPU memory

Adapted Test Strategy:
- Test 1: 32 workers, multiprocessing, no batching (within Windows limits)
- Test 2: 60 workers, multiprocessing, no batching (near Windows limit)
- Test 3: 128 workers, threading, batching (no process limit)
- Test 4: 256 workers, threading, batching (stress test)
- Test 5: 128 workers, threading, no batching (overhead comparison)

The threading tests are now more important on Windows since they avoid
the multiprocessing limitations.
"""

import torch
import time
import subprocess
import threading
import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.training.selfplay import SelfPlayEngine


@dataclass
class TestResult:
    """Results from a single test configuration."""

    # Test identification
    test_name: str
    num_workers: int
    use_multiprocessing: bool  # True = multiprocessing, False = threading
    use_batched_evaluator: bool

    # Performance metrics
    num_games: int
    elapsed_seconds: float
    games_per_min: float

    # GPU metrics
    avg_gpu_utilization: float
    max_gpu_utilization: float
    avg_gpu_memory_mb: float

    # Batching metrics (if applicable)
    total_batch_requests: int
    total_batches: int
    avg_batch_size: float

    # Status
    success: bool
    error_message: Optional[str] = None


class GPUMonitor:
    """Monitor GPU utilization using nvidia-smi."""

    def __init__(self, interval_seconds: float = 0.5):
        self.interval = interval_seconds
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.samples: List[Dict[str, float]] = []
        self.lock = threading.Lock()

    def start(self):
        """Start monitoring GPU in background thread."""
        if self.running:
            return

        self.running = True
        self.samples = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return statistics."""
        if not self.running:
            return {
                'avg_gpu_utilization': 0.0,
                'max_gpu_utilization': 0.0,
                'avg_gpu_memory_mb': 0.0,
            }

        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)

        with self.lock:
            if not self.samples:
                return {
                    'avg_gpu_utilization': 0.0,
                    'max_gpu_utilization': 0.0,
                    'avg_gpu_memory_mb': 0.0,
                }

            utils = [s['utilization'] for s in self.samples]
            mems = [s['memory_mb'] for s in self.samples]

            return {
                'avg_gpu_utilization': sum(utils) / len(utils),
                'max_gpu_utilization': max(utils),
                'avg_gpu_memory_mb': sum(mems) / len(mems),
            }

    def _monitor_loop(self):
        """Background loop that samples GPU stats."""
        while self.running:
            try:
                result = subprocess.run(
                    [
                        'nvidia-smi',
                        '--query-gpu=utilization.gpu,memory.used',
                        '--format=csv,noheader,nounits'
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                )

                if result.returncode == 0:
                    line = result.stdout.strip()
                    util_str, mem_str = line.split(',')

                    sample = {
                        'utilization': float(util_str.strip()),
                        'memory_mb': float(mem_str.strip()),
                        'timestamp': time.time(),
                    }

                    with self.lock:
                        self.samples.append(sample)

            except Exception as e:
                print(f"Warning: GPU monitoring failed: {e}")

            time.sleep(self.interval)


def run_test(
    test_name: str,
    num_workers: int,
    use_multiprocessing: bool,
    use_batched_evaluator: bool,
    num_games: int = 50,
    cards_to_deal: int = 3,
    device: str = "cuda",
) -> TestResult:
    """
    Run a single test configuration.

    Args:
        test_name: Descriptive name for the test
        num_workers: Number of parallel workers
        use_multiprocessing: True for multiprocessing, False for threading
        use_batched_evaluator: Whether to use BatchedEvaluator
        num_games: Number of games to generate
        cards_to_deal: Cards dealt per player
        device: Device to run on

    Returns:
        TestResult with performance metrics
    """
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")
    print(f"Workers: {num_workers}")
    print(f"Parallelism: {'multiprocessing' if use_multiprocessing else 'threading'}")
    print(f"Batching: {'yes' if use_batched_evaluator else 'no'}")
    print(f"Games: {num_games}")
    print(f"Cards: {cards_to_deal}")
    print(f"{'='*80}\n")

    try:
        # Create network
        print("Creating network...")
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

        # Create self-play engine
        print("Creating self-play engine...")
        engine = SelfPlayEngine(
            network=network,
            encoder=encoder,
            masker=masker,
            num_workers=num_workers,
            num_determinizations=3,
            simulations_per_determinization=10,
            device=device,
            use_batched_evaluator=use_batched_evaluator,
            batch_size=1024,  # Large batch size for batched evaluator
            batch_timeout_ms=5.0,
            use_thread_pool=not use_multiprocessing,  # False = multiprocessing, True = threading
        )

        # Warm-up run
        print("Warming up...")
        engine.generate_games(num_games=2, num_players=4, cards_to_deal=cards_to_deal)

        # Reset evaluator stats if available
        if engine.batch_evaluator:
            engine.batch_evaluator.reset_stats()

        # Start GPU monitoring
        gpu_monitor = GPUMonitor(interval_seconds=0.5)
        gpu_monitor.start()

        # Run benchmark
        print(f"\nGenerating {num_games} games...")
        start_time = time.time()

        examples = engine.generate_games(
            num_games=num_games,
            num_players=4,
            cards_to_deal=cards_to_deal
        )

        elapsed = time.time() - start_time

        # Stop GPU monitoring
        gpu_stats = gpu_monitor.stop()

        # Get batch evaluator stats
        if engine.batch_evaluator:
            batch_stats = engine.batch_evaluator.get_stats()
            total_batch_requests = batch_stats['total_requests']
            total_batches = batch_stats['total_batches']
            avg_batch_size = batch_stats['avg_batch_size']
        else:
            total_batch_requests = 0
            total_batches = 0
            avg_batch_size = 0.0

        # Calculate metrics
        games_per_min = (num_games / elapsed) * 60

        # Create result
        result = TestResult(
            test_name=test_name,
            num_workers=num_workers,
            use_multiprocessing=use_multiprocessing,
            use_batched_evaluator=use_batched_evaluator,
            num_games=num_games,
            elapsed_seconds=elapsed,
            games_per_min=games_per_min,
            avg_gpu_utilization=gpu_stats['avg_gpu_utilization'],
            max_gpu_utilization=gpu_stats['max_gpu_utilization'],
            avg_gpu_memory_mb=gpu_stats['avg_gpu_memory_mb'],
            total_batch_requests=total_batch_requests,
            total_batches=total_batches,
            avg_batch_size=avg_batch_size,
            success=True,
        )

        # Print results
        print(f"\nRESULTS:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Games/min: {games_per_min:.1f}")
        print(f"  Examples: {len(examples)}")
        print(f"\nGPU Metrics:")
        print(f"  Avg utilization: {gpu_stats['avg_gpu_utilization']:.1f}%")
        print(f"  Max utilization: {gpu_stats['max_gpu_utilization']:.1f}%")
        print(f"  Avg memory: {gpu_stats['avg_gpu_memory_mb']:.0f} MB")

        if use_batched_evaluator:
            print(f"\nBatching Metrics:")
            print(f"  Total requests: {total_batch_requests}")
            print(f"  Total batches: {total_batches}")
            print(f"  Avg batch size: {avg_batch_size:.1f}")

        # Cleanup
        engine.shutdown()

        # Clean up network
        del network
        del engine
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"\nERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()

        return TestResult(
            test_name=test_name,
            num_workers=num_workers,
            use_multiprocessing=use_multiprocessing,
            use_batched_evaluator=use_batched_evaluator,
            num_games=num_games,
            elapsed_seconds=0.0,
            games_per_min=0.0,
            avg_gpu_utilization=0.0,
            max_gpu_utilization=0.0,
            avg_gpu_memory_mb=0.0,
            total_batch_requests=0,
            total_batches=0,
            avg_batch_size=0.0,
            success=False,
            error_message=str(e),
        )


def main():
    """Run Windows-optimized performance tests."""

    print("\n" + "="*80)
    print("WINDOWS-OPTIMIZED PERFORMANCE TESTS")
    print("Adapted for Windows multiprocessing limitations")
    print("="*80)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("\nERROR: CUDA not available! This test requires a GPU.")
        sys.exit(1)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    results: List[TestResult] = []

    # Test 1: 32 workers, multiprocessing, no batching (safe Windows limit)
    print("\n" + "="*80)
    print("TEST 1: Moderate Workers + Multiprocessing + No Batching")
    print("32 workers - safe for Windows handle limits")
    print("="*80)

    test1_result = run_test(
        test_name="Test 1: 32 workers, multiprocessing, no batching",
        num_workers=32,
        use_multiprocessing=True,
        use_batched_evaluator=False,
        num_games=50,
    )
    results.append(test1_result)

    # Test 2: 128 workers, threading, batching (PRIMARY TEST)
    print("\n" + "="*80)
    print("TEST 2: High Workers + Threading + Batching")
    print("128 workers - no process limit, shared memory")
    print("="*80)

    test2_result = run_test(
        test_name="Test 2: 128 workers, threading, batching",
        num_workers=128,
        use_multiprocessing=False,  # Threading
        use_batched_evaluator=True,
        num_games=50,
    )
    results.append(test2_result)

    # Test 3: 256 workers, threading, batching (STRESS TEST)
    print("\n" + "="*80)
    print("TEST 3: Very High Workers + Threading + Batching")
    print("256 workers - maximum parallelism with threading")
    print("="*80)

    test3_result = run_test(
        test_name="Test 3: 256 workers, threading, batching",
        num_workers=256,
        use_multiprocessing=False,
        use_batched_evaluator=True,
        num_games=50,
    )
    results.append(test3_result)

    # Test 4: 128 workers, threading, no batching (compare overhead)
    print("\n" + "="*80)
    print("TEST 4: High Workers + Threading + No Batching")
    print("128 workers - test if batching overhead is worth it")
    print("="*80)

    test4_result = run_test(
        test_name="Test 4: 128 workers, threading, no batching",
        num_workers=128,
        use_multiprocessing=False,
        use_batched_evaluator=False,
        num_games=50,
    )
    results.append(test4_result)

    # Test 5: 60 workers, multiprocessing, no batching (max Windows multiprocessing)
    # Only run if Test 1 showed promise
    if test1_result.success and test1_result.games_per_min > 50:
        print("\n" + "="*80)
        print("TEST 5: Max Windows Multiprocessing (60 workers)")
        print("Testing near Windows 63-handle limit")
        print("="*80)

        test5_result = run_test(
            test_name="Test 5: 60 workers, multiprocessing, no batching",
            num_workers=60,
            use_multiprocessing=True,
            use_batched_evaluator=False,
            num_games=50,
        )
        results.append(test5_result)

    # Print summary
    print_summary(results)

    # Save results
    save_results(results)


def print_summary(results: List[TestResult]):
    """Print comprehensive summary and recommendations."""

    print("\n" + "="*80)
    print("WINDOWS-OPTIMIZED TEST SUMMARY")
    print("="*80)

    # Filter successful results
    successful = [r for r in results if r.success]

    if not successful:
        print("\nNo successful tests!")
        return

    # Find best configuration
    best = max(successful, key=lambda r: r.games_per_min)

    print(f"\nBest Configuration:")
    print(f"  Test: {best.test_name}")
    print(f"  Games/min: {best.games_per_min:.1f}")
    print(f"  GPU utilization: {best.avg_gpu_utilization:.1f}% (avg), {best.max_gpu_utilization:.1f}% (max)")
    if best.use_batched_evaluator:
        print(f"  Avg batch size: {best.avg_batch_size:.1f}")

    # Performance comparison table
    print(f"\nDetailed Results:")
    print(f"{'Test':<55} {'Games/min':<12} {'GPU Avg%':<10} {'GPU Max%':<10} {'Batch Size':<12}")
    print("-" * 110)

    for result in results:
        if result.success:
            batch_size_str = f"{result.avg_batch_size:.1f}" if result.use_batched_evaluator else "N/A"
            print(f"{result.test_name:<55} {result.games_per_min:<12.1f} {result.avg_gpu_utilization:<10.1f} {result.max_gpu_utilization:<10.1f} {batch_size_str:<12}")
        else:
            print(f"{result.test_name:<55} {'FAILED':<12} {'-':<10} {'-':<10} {'-':<12}")

    # Training time estimates
    print(f"\nTraining Time Estimate (500 iterations, 10,000 games each):")
    games_per_iteration = 10_000
    iterations = 500

    minutes_per_iteration = games_per_iteration / best.games_per_min
    hours_per_iteration = minutes_per_iteration / 60
    days_total = (hours_per_iteration * iterations) / 24

    print(f"  Per iteration: {minutes_per_iteration:.1f} min ({hours_per_iteration:.2f} hours)")
    print(f"  500 iterations: {days_total:.1f} days")

    # Decision and recommendations
    print(f"\nDECISION:")
    if best.games_per_min >= 500:
        print(f"  SUCCESS: Achieved {best.games_per_min:.1f} games/min (target: >500)")
        print(f"  Training time: {days_total:.1f} days (excellent)")
        print(f"  ACTION: Use this configuration for production training")
    elif best.games_per_min >= 100:
        print(f"  ACCEPTABLE: Achieved {best.games_per_min:.1f} games/min")
        print(f"  Training time: {days_total:.1f} days")
        print(f"  ACTION: Usable, but consider GPU-batched MCTS for 3-5x speedup")
    else:
        print(f"  INSUFFICIENT: Only {best.games_per_min:.1f} games/min (<100)")
        print(f"  Training time: {days_total:.1f} days (too slow)")
        print(f"  ACTION: MUST implement GPU-batched MCTS")

    # Key insights
    print(f"\nKey Insights:")

    # Compare threading vs multiprocessing
    threading_results = [r for r in successful if not r.use_multiprocessing]
    multiproc_results = [r for r in successful if r.use_multiprocessing]

    if threading_results and multiproc_results:
        best_threading = max(threading_results, key=lambda r: r.games_per_min)
        best_multiproc = max(multiproc_results, key=lambda r: r.games_per_min)

        speedup = best_threading.games_per_min / best_multiproc.games_per_min
        print(f"  Threading vs Multiprocessing: {speedup:.2f}x speedup with threading")
        print(f"    - Threading: {best_threading.games_per_min:.1f} games/min ({best_threading.num_workers} workers)")
        print(f"    - Multiproc: {best_multiproc.games_per_min:.1f} games/min ({best_multiproc.num_workers} workers)")

    # Compare batching overhead
    batched = [r for r in successful if r.use_batched_evaluator and not r.use_multiprocessing]
    unbatched = [r for r in successful if not r.use_batched_evaluator and not r.use_multiprocessing]

    if batched and unbatched:
        # Compare same worker count if available
        for b in batched:
            matching_unbatched = [u for u in unbatched if u.num_workers == b.num_workers]
            if matching_unbatched:
                u = matching_unbatched[0]
                overhead = ((b.games_per_min / u.games_per_min) - 1) * 100
                print(f"  Batching impact at {b.num_workers} workers: {overhead:+.1f}%")
                print(f"    - With batching: {b.games_per_min:.1f} games/min (batch size {b.avg_batch_size:.1f})")
                print(f"    - Without batching: {u.games_per_min:.1f} games/min")

    # GPU utilization analysis
    print(f"\nGPU Utilization:")
    print(f"  Best config: {best.avg_gpu_utilization:.1f}% avg, {best.max_gpu_utilization:.1f}% max")
    if best.avg_gpu_utilization < 50:
        print(f"  WARNING: GPU is significantly underutilized")
        print(f"  This suggests CPU/Python overhead is the bottleneck, not GPU compute")
        print(f"  GPU-batched MCTS could achieve 5-10x improvement")

    # Recommended configuration
    print(f"\nRECOMMENDED PRODUCTION CONFIGURATION:")
    print(f"  num_workers: {best.num_workers}")
    print(f"  use_thread_pool: {not best.use_multiprocessing}")
    print(f"  use_batched_evaluator: {best.use_batched_evaluator}")
    if best.use_batched_evaluator:
        print(f"  batch_size: 1024")
        print(f"  batch_timeout_ms: 5.0")
    print(f"  Expected performance: {best.games_per_min:.1f} games/min")


def save_results(results: List[TestResult]):
    """Save test results to files."""

    # Save to CSV
    csv_path = Path("action_plan_windows_results.csv")
    with open(csv_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))

    print(f"\nResults saved to: {csv_path}")

    # Save to JSON
    json_path = Path("action_plan_windows_results.json")
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()
