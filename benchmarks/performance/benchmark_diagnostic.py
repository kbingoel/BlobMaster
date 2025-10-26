"""
Comprehensive Diagnostic Benchmark for BlobNet Training Performance

This benchmark systematically tests different configurations to identify
the root cause of poor GPU/CPU utilization and slow training speeds.

Goals:
1. Measure performance across different worker counts (4, 8, 16, 32, 64, 128)
2. Track GPU utilization, batch sizes, and CPU usage
3. Profile code to find bottlenecks
4. Test with/without BatchedEvaluator
5. Vary game complexity (cards dealt)

Expected Findings:
- Identify optimal worker count for GPU saturation (target: >70% GPU util)
- Measure actual games/min at different scales
- Find synchronization bottlenecks
- Determine if batching helps or hurts at different worker counts
"""

import torch
import time
import subprocess
import threading
import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import cProfile
import pstats
import io

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.training.selfplay import SelfPlayEngine


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    # Configuration
    num_workers: int
    cards_to_deal: int
    use_batched_evaluator: bool
    use_thread_pool: bool
    batch_size: int
    batch_timeout_ms: float
    num_determinizations: int
    simulations_per_det: int

    # Performance metrics
    num_games: int
    total_examples: int
    elapsed_seconds: float
    games_per_min: float
    examples_per_game: float

    # GPU metrics
    avg_gpu_utilization: float
    max_gpu_utilization: float
    avg_gpu_memory_mb: float
    max_gpu_memory_mb: float

    # Batching metrics
    total_batch_requests: int
    total_batches: int
    avg_batch_size: float

    # CPU metrics
    avg_cpu_percent: float

    # Status
    success: bool
    error_message: Optional[str] = None


class GPUMonitor:
    """Monitor GPU utilization using nvidia-smi."""

    def __init__(self, interval_seconds: float = 0.5):
        """
        Initialize GPU monitor.

        Args:
            interval_seconds: How often to sample GPU stats
        """
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
        """
        Stop monitoring and return statistics.

        Returns:
            Dict with avg/max GPU utilization and memory usage
        """
        if not self.running:
            return {
                'avg_gpu_utilization': 0.0,
                'max_gpu_utilization': 0.0,
                'avg_gpu_memory_mb': 0.0,
                'max_gpu_memory_mb': 0.0,
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
                    'max_gpu_memory_mb': 0.0,
                }

            utils = [s['utilization'] for s in self.samples]
            mems = [s['memory_mb'] for s in self.samples]

            return {
                'avg_gpu_utilization': sum(utils) / len(utils),
                'max_gpu_utilization': max(utils),
                'avg_gpu_memory_mb': sum(mems) / len(mems),
                'max_gpu_memory_mb': max(mems),
            }

    def _monitor_loop(self):
        """Background loop that samples GPU stats."""
        while self.running:
            try:
                # Query nvidia-smi for GPU utilization and memory
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


def create_test_network(device="cuda") -> BlobNet:
    """Create neural network for benchmarking."""
    network = BlobNet(
        state_dim=256,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        feedforward_dim=1024,
        dropout=0.0,  # No dropout for consistent timing
    )
    network.to(device)
    network.eval()

    # Count parameters
    num_params = sum(p.numel() for p in network.parameters())
    print(f"Network: {num_params:,} parameters (~{num_params/1e6:.1f}M)")

    return network


def run_benchmark_configuration(
    network: BlobNet,
    encoder: StateEncoder,
    masker: ActionMasker,
    num_workers: int,
    cards_to_deal: int,
    num_games: int,
    use_batched_evaluator: bool,
    use_thread_pool: bool,
    batch_size: int = 512,
    batch_timeout_ms: float = 10.0,
    num_determinizations: int = 3,
    simulations_per_det: int = 10,
    device: str = "cuda",
) -> BenchmarkResult:
    """
    Run a single benchmark configuration.

    Args:
        network: Neural network to use
        encoder: State encoder
        masker: Action masker
        num_workers: Number of parallel workers
        cards_to_deal: Cards to deal per player
        num_games: Number of games to generate
        use_batched_evaluator: Whether to use BatchedEvaluator
        use_thread_pool: Whether to use ThreadPoolExecutor (vs multiprocessing)
        batch_size: Max batch size for evaluator
        batch_timeout_ms: Batch collection timeout
        num_determinizations: Determinizations per MCTS search
        simulations_per_det: Simulations per determinization
        device: Device to run on

    Returns:
        BenchmarkResult with performance metrics
    """
    print(f"\n{'='*80}")
    print(f"Configuration:")
    print(f"  Workers: {num_workers}")
    print(f"  Cards: {cards_to_deal}")
    print(f"  Games: {num_games}")
    print(f"  Batched evaluator: {use_batched_evaluator}")
    print(f"  Thread pool: {use_thread_pool}")
    print(f"  Batch size: {batch_size}")
    print(f"  Timeout: {batch_timeout_ms}ms")
    print(f"  Determinizations: {num_determinizations}")
    print(f"  Simulations/det: {simulations_per_det}")
    print(f"{'='*80}")

    try:
        # Create self-play engine
        engine = SelfPlayEngine(
            network=network,
            encoder=encoder,
            masker=masker,
            num_workers=num_workers,
            num_determinizations=num_determinizations,
            simulations_per_determinization=simulations_per_det,
            device=device,
            use_batched_evaluator=use_batched_evaluator,
            batch_size=batch_size,
            batch_timeout_ms=batch_timeout_ms,
            use_thread_pool=use_thread_pool,
        )

        # Warm-up run
        print("Warming up (JIT compilation, cache warming)...")
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
        examples_per_game = len(examples) / num_games if num_games > 0 else 0.0

        # Create result
        result = BenchmarkResult(
            num_workers=num_workers,
            cards_to_deal=cards_to_deal,
            use_batched_evaluator=use_batched_evaluator,
            use_thread_pool=use_thread_pool,
            batch_size=batch_size,
            batch_timeout_ms=batch_timeout_ms,
            num_determinizations=num_determinizations,
            simulations_per_det=simulations_per_det,
            num_games=num_games,
            total_examples=len(examples),
            elapsed_seconds=elapsed,
            games_per_min=games_per_min,
            examples_per_game=examples_per_game,
            avg_gpu_utilization=gpu_stats['avg_gpu_utilization'],
            max_gpu_utilization=gpu_stats['max_gpu_utilization'],
            avg_gpu_memory_mb=gpu_stats['avg_gpu_memory_mb'],
            max_gpu_memory_mb=gpu_stats['max_gpu_memory_mb'],
            total_batch_requests=total_batch_requests,
            total_batches=total_batches,
            avg_batch_size=avg_batch_size,
            avg_cpu_percent=0.0,  # TODO: Add CPU monitoring
            success=True,
        )

        # Print results
        print(f"\nResults:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Games/min: {games_per_min:.1f}")
        print(f"  Examples: {len(examples)} ({examples_per_game:.1f} per game)")
        print(f"\nGPU Metrics:")
        print(f"  Avg utilization: {gpu_stats['avg_gpu_utilization']:.1f}%")
        print(f"  Max utilization: {gpu_stats['max_gpu_utilization']:.1f}%")
        print(f"  Avg memory: {gpu_stats['avg_gpu_memory_mb']:.0f} MB")
        print(f"  Max memory: {gpu_stats['max_gpu_memory_mb']:.0f} MB")
        print(f"\nBatching Metrics:")
        print(f"  Total requests: {total_batch_requests}")
        print(f"  Total batches: {total_batches}")
        print(f"  Avg batch size: {avg_batch_size:.1f}")

        # Cleanup
        engine.shutdown()

        return result

    except Exception as e:
        print(f"\nERROR: Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

        return BenchmarkResult(
            num_workers=num_workers,
            cards_to_deal=cards_to_deal,
            use_batched_evaluator=use_batched_evaluator,
            use_thread_pool=use_thread_pool,
            batch_size=batch_size,
            batch_timeout_ms=batch_timeout_ms,
            num_determinizations=num_determinizations,
            simulations_per_det=simulations_per_det,
            num_games=num_games,
            total_examples=0,
            elapsed_seconds=0.0,
            games_per_min=0.0,
            examples_per_game=0.0,
            avg_gpu_utilization=0.0,
            max_gpu_utilization=0.0,
            avg_gpu_memory_mb=0.0,
            max_gpu_memory_mb=0.0,
            total_batch_requests=0,
            total_batches=0,
            avg_batch_size=0.0,
            avg_cpu_percent=0.0,
            success=False,
            error_message=str(e),
        )


def run_comprehensive_benchmark_suite(device="cuda"):
    """
    Run comprehensive benchmark suite across different configurations.

    Tests:
    1. Worker scaling: 4, 8, 16, 32, 64 workers
    2. Batching: with/without BatchedEvaluator
    3. Thread vs Process: ThreadPoolExecutor vs multiprocessing
    4. Game complexity: 3, 5, 8 cards
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE DIAGNOSTIC BENCHMARK SUITE")
    print("="*80)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("\nWARNING: CUDA not available! Running on CPU (not representative).")
        device = "cpu"
    else:
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    # Create network
    print("\nCreating network...")
    network = create_test_network(device=device)
    encoder = StateEncoder()
    masker = ActionMasker()

    # Test configurations
    results: List[BenchmarkResult] = []

    # Fixed parameters for all tests
    num_games = 20
    base_config = {
        'network': network,
        'encoder': encoder,
        'masker': masker,
        'num_games': num_games,
        'device': device,
        'num_determinizations': 3,
        'simulations_per_det': 10,
        'batch_size': 512,
        'batch_timeout_ms': 10.0,
    }

    # Test Suite 1: Worker Scaling (most important)
    print("\n" + "="*80)
    print("TEST SUITE 1: Worker Scaling")
    print("Testing worker counts: 4, 8, 16, 32, 64")
    print("="*80)

    for num_workers in [4, 8, 16, 32, 64]:
        result = run_benchmark_configuration(
            **base_config,
            num_workers=num_workers,
            cards_to_deal=3,
            use_batched_evaluator=True,
            use_thread_pool=True,
        )
        results.append(result)

        # Stop early if we're seeing errors
        if not result.success:
            print(f"\nStopping worker scaling tests due to error at {num_workers} workers")
            break

    # Test Suite 2: Batching Comparison
    print("\n" + "="*80)
    print("TEST SUITE 2: Batching Comparison")
    print("Testing with/without BatchedEvaluator at different worker counts")
    print("="*80)

    for num_workers in [4, 16, 32]:
        # With batching
        result_batched = run_benchmark_configuration(
            **base_config,
            num_workers=num_workers,
            cards_to_deal=3,
            use_batched_evaluator=True,
            use_thread_pool=True,
        )
        results.append(result_batched)

        # Without batching
        result_direct = run_benchmark_configuration(
            **base_config,
            num_workers=num_workers,
            cards_to_deal=3,
            use_batched_evaluator=False,
            use_thread_pool=True,
        )
        results.append(result_direct)

    # Test Suite 3: Thread vs Process
    print("\n" + "="*80)
    print("TEST SUITE 3: Thread vs Process Comparison")
    print("="*80)

    for use_threads in [True, False]:
        result = run_benchmark_configuration(
            **base_config,
            num_workers=16,
            cards_to_deal=3,
            use_batched_evaluator=True,
            use_thread_pool=use_threads,
        )
        results.append(result)

    # Test Suite 4: Game Complexity
    print("\n" + "="*80)
    print("TEST SUITE 4: Game Complexity")
    print("Testing different cards dealt: 3, 5, 8")
    print("="*80)

    for cards in [3, 5, 8]:
        result = run_benchmark_configuration(
            **base_config,
            num_workers=16,
            cards_to_deal=cards,
            use_batched_evaluator=True,
            use_thread_pool=True,
        )
        results.append(result)

    # Save results
    save_results(results)

    # Print summary
    print_summary(results)

    return results


def save_results(results: List[BenchmarkResult]):
    """Save benchmark results to CSV and JSON files."""

    # Save to CSV
    csv_path = Path("benchmark_diagnostic_results.csv")
    with open(csv_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))

    print(f"\nResults saved to: {csv_path}")

    # Save to JSON (more detailed)
    json_path = Path("benchmark_diagnostic_results.json")
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"Results saved to: {json_path}")


def print_summary(results: List[BenchmarkResult]):
    """Print summary analysis of benchmark results."""

    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    # Filter successful results
    successful = [r for r in results if r.success]

    if not successful:
        print("\nNo successful benchmarks!")
        return

    # Find best configuration
    best = max(successful, key=lambda r: r.games_per_min)

    print(f"\nBest Configuration:")
    print(f"  Workers: {best.num_workers}")
    print(f"  Batched: {best.use_batched_evaluator}")
    print(f"  Threads: {best.use_thread_pool}")
    print(f"  Games/min: {best.games_per_min:.1f}")
    print(f"  GPU utilization: {best.avg_gpu_utilization:.1f}%")
    print(f"  Batch size: {best.avg_batch_size:.1f}")

    # Analyze worker scaling
    print(f"\n{'Workers':<10} {'Games/min':<15} {'GPU%':<10} {'Batch Size':<12} {'Status'}")
    print("-" * 80)

    worker_results = sorted(
        [r for r in successful if r.use_batched_evaluator and r.use_thread_pool and r.cards_to_deal == 3],
        key=lambda r: r.num_workers
    )

    for result in worker_results:
        status = "✓" if result.avg_gpu_utilization > 70 else "⚠" if result.avg_gpu_utilization > 40 else "✗"
        print(f"{result.num_workers:<10} {result.games_per_min:<15.1f} {result.avg_gpu_utilization:<10.1f} {result.avg_batch_size:<12.1f} {status}")

    # Key findings
    print(f"\nKey Findings:")

    if best.avg_gpu_utilization < 70:
        print(f"  ⚠ GPU utilization is low ({best.avg_gpu_utilization:.1f}%) - need more workers or larger batches")
    else:
        print(f"  ✓ GPU is well utilized ({best.avg_gpu_utilization:.1f}%)")

    if best.avg_batch_size < 128:
        print(f"  ⚠ Batch sizes are small ({best.avg_batch_size:.1f}) - need more concurrent requests")
    else:
        print(f"  ✓ Batch sizes are good ({best.avg_batch_size:.1f})")

    if best.games_per_min < 500:
        print(f"  ⚠ Games/min is low ({best.games_per_min:.1f}) - target is >500")
    else:
        print(f"  ✓ Games/min is good ({best.games_per_min:.1f})")

    # Recommendations
    print(f"\nRecommendations:")

    if best.num_workers < 64 and best.avg_gpu_utilization < 70:
        print(f"  → Try more workers (current best: {best.num_workers}, try 64-128)")

    if best.avg_batch_size < 128:
        print(f"  → Increase concurrent work (more workers, more determinizations, or more simulations)")

    if not best.use_batched_evaluator and best.num_workers >= 16:
        print(f"  → BatchedEvaluator might help with {best.num_workers} workers")

    # Training time estimate
    games_per_iteration = 10_000
    iterations = 500
    total_games = games_per_iteration * iterations
    minutes_per_iteration = games_per_iteration / best.games_per_min
    hours_per_iteration = minutes_per_iteration / 60
    days_total = (hours_per_iteration * iterations) / 24

    print(f"\nTraining Time Estimate (with best config):")
    print(f"  Per iteration: {minutes_per_iteration:.1f} min ({hours_per_iteration:.2f} hours)")
    print(f"  500 iterations: {days_total:.1f} days")

    if days_total > 30:
        print(f"  ⚠ Training will take >{days_total:.0f} days - optimization needed!")
    elif days_total > 15:
        print(f"  ⚠ Training will take ~{days_total:.0f} days - could be improved")
    else:
        print(f"  ✓ Training time is acceptable ({days_total:.0f} days)")


if __name__ == "__main__":
    # Check for CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        print("If you have a GPU, ensure PyTorch is installed with CUDA support.")
        sys.exit(1)

    # Run comprehensive benchmark
    results = run_comprehensive_benchmark_suite(device="cuda")

    print("\n" + "="*80)
    print("Benchmark complete! Check benchmark_diagnostic_results.csv for full data.")
    print("="*80)
