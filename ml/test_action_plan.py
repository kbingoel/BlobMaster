"""
Targeted Performance Tests - TRAINING-PERFORMANCE-MASTER.md Action Plan

This script implements the 3 priority tests identified in the master performance document:

Test 1: 128 workers, multiprocessing, no batching
  - Expected: 150-300 games/min, 60-80% GPU
  - Reasoning: Maximize parallelism, avoid batching overhead

Test 2: 256 workers, multiprocessing, no batching
  - Expected: 300-600 games/min, 70-90% GPU
  - Reasoning: Push parallelism further to saturate GPU

Test 3: 128 workers, threading, batching
  - Expected: 30-60 games/min, 40-70% GPU
  - Reasoning: Validate if batching helps at high worker counts

Decision Criteria:
- >500 games/min: Solution found, proceed to Phase 4
- 100-500 games/min: Acceptable, use best config
- <100 games/min: Need GPU-batched MCTS implementation
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
    """Run the action plan tests."""

    print("\n" + "="*80)
    print("TRAINING-PERFORMANCE-MASTER.md ACTION PLAN")
    print("Testing 3 priority configurations to find optimal training performance")
    print("="*80)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("\nERROR: CUDA not available! This test requires a GPU.")
        sys.exit(1)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    results: List[TestResult] = []

    # Test 1: 128 workers, multiprocessing, no batching
    print("\n" + "="*80)
    print("TEST 1: High Workers + Multiprocessing + No Batching")
    print("Expected: 150-300 games/min, 60-80% GPU")
    print("="*80)

    test1_result = run_test(
        test_name="Test 1: 128 workers, multiprocessing, no batching",
        num_workers=128,
        use_multiprocessing=True,
        use_batched_evaluator=False,
        num_games=50,
    )
    results.append(test1_result)

    # Decide whether to run Test 2 based on Test 1 results
    run_test2 = False
    if test1_result.success and test1_result.games_per_min >= 100:
        print(f"\n{'='*80}")
        print(f"Test 1 achieved {test1_result.games_per_min:.1f} games/min (>100)")
        print("Proceeding to Test 2 to push performance further...")
        print(f"{'='*80}")
        run_test2 = True
    elif test1_result.success:
        print(f"\n{'='*80}")
        print(f"Test 1 achieved {test1_result.games_per_min:.1f} games/min (<100)")
        print("Skipping Test 2, will need GPU-batched MCTS for >500 games/min")
        print(f"{'='*80}")

    # Test 2: 256 workers, multiprocessing, no batching (conditional)
    if run_test2:
        print("\n" + "="*80)
        print("TEST 2: Very High Workers + Multiprocessing + No Batching")
        print("Expected: 300-600 games/min, 70-90% GPU")
        print("="*80)

        test2_result = run_test(
            test_name="Test 2: 256 workers, multiprocessing, no batching",
            num_workers=256,
            use_multiprocessing=True,
            use_batched_evaluator=False,
            num_games=50,
        )
        results.append(test2_result)

    # Test 3: 128 workers, threading, batching (validation)
    print("\n" + "="*80)
    print("TEST 3: High Workers + Threading + Batching (Validation)")
    print("Expected: 30-60 games/min, 40-70% GPU")
    print("="*80)

    test3_result = run_test(
        test_name="Test 3: 128 workers, threading, batching",
        num_workers=128,
        use_multiprocessing=False,  # Threading
        use_batched_evaluator=True,
        num_games=50,
    )
    results.append(test3_result)

    # Print summary
    print_summary(results)

    # Save results
    save_results(results)


def print_summary(results: List[TestResult]):
    """Print comprehensive summary and recommendations."""

    print("\n" + "="*80)
    print("ACTION PLAN TEST SUMMARY")
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

    # Performance comparison table
    print(f"\nDetailed Results:")
    print(f"{'Test':<50} {'Games/min':<12} {'GPU Avg%':<10} {'GPU Max%':<10} {'Status'}")
    print("-" * 100)

    for result in results:
        if result.success:
            status_icon = "EXCELLENT" if result.games_per_min >= 500 else "GOOD" if result.games_per_min >= 100 else "POOR"
            status = f"{status_icon} ({result.games_per_min:.1f})"
        else:
            status = "FAILED"

        print(f"{result.test_name:<50} {result.games_per_min:<12.1f} {result.avg_gpu_utilization:<10.1f} {result.max_gpu_utilization:<10.1f} {status}")

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
        print(f"  Training time: {days_total:.1f} days (acceptable)")
        print(f"  ACTION: Use this configuration for production training")
    elif best.games_per_min >= 100:
        print(f"  ACCEPTABLE: Achieved {best.games_per_min:.1f} games/min")
        print(f"  Training time: {days_total:.1f} days")
        print(f"  ACTION: Can proceed with this config, or optimize further for faster training")
    else:
        print(f"  INSUFFICIENT: Only achieved {best.games_per_min:.1f} games/min (<100)")
        print(f"  Training time: {days_total:.1f} days (too slow)")
        print(f"  ACTION: Implement GPU-batched MCTS (Phase 1 from PLAN-GPUBatchedMCTS.md)")

    # GPU utilization analysis
    print(f"\nGPU Utilization Analysis:")
    if best.avg_gpu_utilization >= 70:
        print(f"  EXCELLENT: GPU well saturated ({best.avg_gpu_utilization:.1f}% avg)")
    elif best.avg_gpu_utilization >= 50:
        print(f"  GOOD: Decent GPU utilization ({best.avg_gpu_utilization:.1f}% avg)")
    elif best.avg_gpu_utilization >= 30:
        print(f"  MODERATE: Some room for improvement ({best.avg_gpu_utilization:.1f}% avg)")
    else:
        print(f"  POOR: GPU severely underutilized ({best.avg_gpu_utilization:.1f}% avg)")
        print(f"  This indicates bottleneck is not in GPU, but in task generation or synchronization")

    # Recommended configuration
    print(f"\nRECOMMENDED PRODUCTION CONFIGURATION:")
    print(f"  num_workers: {best.num_workers}")
    print(f"  use_thread_pool: {not best.use_multiprocessing}")
    print(f"  use_batched_evaluator: {best.use_batched_evaluator}")
    print(f"  Expected performance: {best.games_per_min:.1f} games/min")


def save_results(results: List[TestResult]):
    """Save test results to files."""

    # Save to CSV
    csv_path = Path("action_plan_results.csv")
    with open(csv_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))

    print(f"\nResults saved to: {csv_path}")

    # Save to JSON
    json_path = Path("action_plan_results.json")
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()
