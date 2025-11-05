"""
Benchmark self-play game generation performance.

Tests different worker counts and MCTS configurations to find optimal settings
for the AlphaZero training pipeline.

Usage:
    python ml/benchmark_selfplay.py
    python ml/benchmark_selfplay.py --workers 16 --games 100
    python ml/benchmark_selfplay.py --quick  # Fast test with fewer configs

Output:
    - Console: Progress updates and results table
    - CSV: benchmark_selfplay_results.csv
"""

import argparse
import time
import csv
import psutil
import torch
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.training.selfplay import SelfPlayEngine


# Test configurations
DEFAULT_WORKER_COUNTS = [1, 4, 8, 16, 32]
QUICK_WORKER_COUNTS = [1, 8, 16]

DEFAULT_MCTS_CONFIGS = [
    ("light", 2, 20),    # 2 determinizations × 20 sims = 40 tree searches/move
    ("medium", 3, 30),   # 3 determinizations × 30 sims = 90 tree searches/move
    ("heavy", 5, 50),    # 5 determinizations × 50 sims = 250 tree searches/move
]
QUICK_MCTS_CONFIGS = [
    ("medium", 3, 30),   # Only test medium config for quick run
]


class SelfPlayBenchmark:
    """Benchmark self-play game generation performance."""

    def __init__(self, device: str = "cpu"):
        """
        Initialize benchmark.

        Args:
            device: Device for neural network inference
        """
        self.device = device

        print("Initializing benchmark components...")

        # Create network (BASELINE: 4.9M parameters)
        self.network = BlobNet(
            state_dim=256,
            embedding_dim=256,      # Baseline: 256 (not 128!)
            num_layers=6,           # Baseline: 6 (not 2!)
            num_heads=8,            # Baseline: 8 (not 4!)
            feedforward_dim=1024,   # Baseline: 1024 (not 256!)
            dropout=0.0,  # No dropout for inference
        )
        self.network.to(device)
        self.network.eval()

        # Create encoder and masker
        self.encoder = StateEncoder()
        self.masker = ActionMasker()

        print(f"  - Network: {self._count_parameters():,} parameters")
        print(f"  - Device: {device}")

    def _count_parameters(self) -> int:
        """Count network parameters."""
        return sum(p.numel() for p in self.network.parameters())

    def benchmark_config(
        self,
        num_workers: int,
        num_determinizations: int,
        simulations_per_det: int,
        num_games: int,
        config_name: str,
        use_gpu_server: bool = False,
        gpu_server_max_batch: int = 512,
        gpu_server_timeout_ms: float = 10.0,
        use_batched_evaluator: bool = True,
        use_thread_pool: bool = False,
    ) -> Dict[str, Any]:
        """
        Benchmark a specific configuration.

        Args:
            num_workers: Number of parallel workers
            num_determinizations: Determinizations per MCTS search
            simulations_per_det: Simulations per determinization
            num_games: Number of games to generate
            config_name: Human-readable config name
            use_gpu_server: Enable GPU inference server
            gpu_server_max_batch: Max batch size for GPU server
            gpu_server_timeout_ms: Batch accumulation timeout (ms)
            use_batched_evaluator: Use batched evaluator (ignored if GPU server enabled)
            use_thread_pool: Use thread pool (ignored if GPU server enabled)

        Returns:
            Dictionary with performance metrics
        """
        print(f"\n{'='*70}")
        print(f"Testing: {num_workers} workers, {config_name} MCTS")
        print(f"  - Determinizations: {num_determinizations}")
        print(f"  - Simulations/det: {simulations_per_det}")
        print(f"  - Total MCTS sims/move: {num_determinizations * simulations_per_det}")
        print(f"  - Games to generate: {num_games}")
        if use_gpu_server:
            print(f"  - GPU Server: ENABLED")
            print(f"    - Max batch: {gpu_server_max_batch}")
            print(f"    - Timeout: {gpu_server_timeout_ms}ms")
        else:
            print(f"  - GPU Server: DISABLED")
            print(f"    - Batched evaluator: {use_batched_evaluator}")
            print(f"    - Thread pool: {use_thread_pool}")
        print(f"{'='*70}")

        # Create self-play engine
        engine = SelfPlayEngine(
            network=self.network,
            encoder=self.encoder,
            masker=self.masker,
            num_workers=num_workers,
            num_determinizations=num_determinizations,
            simulations_per_determinization=simulations_per_det,
            device=self.device,
            use_gpu_server=use_gpu_server,
            gpu_server_max_batch=gpu_server_max_batch,
            gpu_server_timeout_ms=gpu_server_timeout_ms,
            use_batched_evaluator=use_batched_evaluator,
            use_thread_pool=use_thread_pool,
        )

        # Record initial CPU usage
        cpu_percent_before = psutil.cpu_percent(interval=1.0)

        # Progress tracking
        games_completed = [0]
        last_update_time = [time.time()]

        def progress_callback(count: int):
            games_completed[0] = count
            current_time = time.time()
            if current_time - last_update_time[0] >= 5.0 or count == num_games:
                elapsed = current_time - start_time
                rate = count / elapsed if elapsed > 0 else 0
                cpu_percent = psutil.cpu_percent(interval=None)
                print(f"  Progress: {count}/{num_games} games ({rate:.1f} games/sec, CPU: {cpu_percent:.1f}%)")
                last_update_time[0] = current_time

        # Generate games and measure time
        print("Generating games...")
        start_time = time.time()

        try:
            examples = engine.generate_games(
                num_games=num_games,
                num_players=4,
                cards_to_deal=5,
                progress_callback=progress_callback,
            )
        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            print("Full traceback:")
            traceback.print_exc()
            engine.shutdown()
            return {
                "num_workers": num_workers,
                "mcts_config": config_name,
                "num_determinizations": num_determinizations,
                "simulations_per_det": simulations_per_det,
                "error": str(e),
            }

        elapsed_time = time.time() - start_time

        # Record final CPU usage
        cpu_percent_after = psutil.cpu_percent(interval=1.0)
        avg_cpu_percent = (cpu_percent_before + cpu_percent_after) / 2.0

        # Calculate metrics
        games_per_minute = (num_games / elapsed_time) * 60.0
        seconds_per_game = elapsed_time / num_games
        examples_per_minute = (len(examples) / elapsed_time) * 60.0

        # Shutdown engine
        engine.shutdown()

        # Results
        metrics = {
            "num_workers": num_workers,
            "mcts_config": config_name,
            "num_determinizations": num_determinizations,
            "simulations_per_det": simulations_per_det,
            "total_sims_per_move": num_determinizations * simulations_per_det,
            "num_games": num_games,
            "elapsed_time_sec": elapsed_time,
            "games_per_minute": games_per_minute,
            "seconds_per_game": seconds_per_game,
            "num_examples": len(examples),
            "examples_per_minute": examples_per_minute,
            "cpu_percent": avg_cpu_percent,
        }

        print(f"\nResults:")
        print(f"  - Games/minute: {games_per_minute:.1f}")
        print(f"  - Seconds/game: {seconds_per_game:.2f}")
        print(f"  - Training examples: {len(examples):,}")
        print(f"  - Examples/minute: {examples_per_minute:.1f}")
        print(f"  - CPU utilization: {avg_cpu_percent:.1f}%")

        return metrics

    def run_benchmarks(
        self,
        worker_counts: List[int],
        mcts_configs: List[Tuple[str, int, int]],
        games_per_config: int,
        use_gpu_server: bool = False,
        gpu_server_max_batch: int = 512,
        gpu_server_timeout_ms: float = 10.0,
        use_batched_evaluator: bool = True,
        use_thread_pool: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run full benchmark suite.

        Args:
            worker_counts: List of worker counts to test
            mcts_configs: List of (name, num_det, sims_per_det) tuples
            games_per_config: Games to generate per configuration
            use_gpu_server: Enable GPU inference server
            gpu_server_max_batch: Max batch size for GPU server
            gpu_server_timeout_ms: Batch accumulation timeout (ms)
            use_batched_evaluator: Use batched evaluator
            use_thread_pool: Use thread pool

        Returns:
            List of benchmark results
        """
        results = []

        total_configs = len(worker_counts) * len(mcts_configs)
        config_num = 0

        for num_workers in worker_counts:
            for config_name, num_det, sims_per_det in mcts_configs:
                config_num += 1

                print(f"\n\n{'#'*70}")
                print(f"Configuration {config_num}/{total_configs}")
                print(f"{'#'*70}")

                metrics = self.benchmark_config(
                    num_workers=num_workers,
                    num_determinizations=num_det,
                    simulations_per_det=sims_per_det,
                    num_games=games_per_config,
                    config_name=config_name,
                    use_gpu_server=use_gpu_server,
                    gpu_server_max_batch=gpu_server_max_batch,
                    gpu_server_timeout_ms=gpu_server_timeout_ms,
                    use_batched_evaluator=use_batched_evaluator,
                    use_thread_pool=use_thread_pool,
                )

                results.append(metrics)

                # Brief pause between configs
                time.sleep(2.0)

        return results


def save_results_csv(results: List[Dict[str, Any]], filepath: str):
    """
    Save benchmark results to CSV.

    Args:
        results: List of benchmark result dictionaries
        filepath: Output CSV path
    """
    if not results:
        print("No results to save")
        return

    # Filter out results with errors
    valid_results = [r for r in results if "error" not in r]

    if not valid_results:
        print("No valid results to save")
        return

    fieldnames = list(valid_results[0].keys())

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(valid_results)

    print(f"\nResults saved to: {filepath}")


def print_results_table(results: List[Dict[str, Any]]):
    """
    Print results in a formatted table.

    Args:
        results: List of benchmark result dictionaries
    """
    print(f"\n\n{'='*80}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}\n")

    # Filter out results with errors
    valid_results = [r for r in results if "error" not in r]

    if not valid_results:
        print("No valid results to display")
        return

    # Print table header
    print(f"{'Workers':<10} {'MCTS':<10} {'Sims/Move':<12} {'Games/Min':<12} {'Sec/Game':<12} {'CPU%':<8}")
    print("-" * 80)

    # Print rows
    for r in valid_results:
        print(f"{r['num_workers']:<10} "
              f"{r['mcts_config']:<10} "
              f"{r['total_sims_per_move']:<12} "
              f"{r['games_per_minute']:<12.1f} "
              f"{r['seconds_per_game']:<12.2f} "
              f"{r['cpu_percent']:<8.1f}")

    print("-" * 80)

    # Print recommendations
    print("\nRECOMMENDATIONS:")

    # Find best worker count for medium MCTS
    medium_results = [r for r in valid_results if r['mcts_config'] == 'medium']
    if medium_results:
        best_workers = max(medium_results, key=lambda x: x['games_per_minute'])
        print(f"  - Best worker count: {best_workers['num_workers']} "
              f"({best_workers['games_per_minute']:.1f} games/min)")

    # Compare MCTS configs at best worker count
    if medium_results:
        workers = best_workers['num_workers']
        worker_results = [r for r in valid_results if r['num_workers'] == workers]

        print(f"\n  MCTS Configuration Impact (at {workers} workers):")
        for r in worker_results:
            speedup = 1.0
            if r['mcts_config'] != 'light':
                light = next((x for x in worker_results if x['mcts_config'] == 'light'), None)
                if light:
                    speedup = light['games_per_minute'] / r['games_per_minute']

            print(f"    - {r['mcts_config']:<10}: {r['games_per_minute']:>6.1f} games/min "
                  f"({r['total_sims_per_move']:>3} sims/move, {speedup:.2f}x slower)")

    # Error summary
    error_results = [r for r in results if "error" in r]
    if error_results:
        print(f"\n  ERRORS: {len(error_results)} configurations failed")
        for r in error_results:
            print(f"    - {r['num_workers']} workers, {r['mcts_config']}: {r['error']}")


def main():
    """Run self-play benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark self-play performance")
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        help="Worker counts to test (default: 1 4 8 16 32)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=5,
        help="Games per configuration (default: 5 for quick screening)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with fewer configurations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for neural network (default: cpu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_selfplay_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--use_gpu_server",
        action="store_true",
        help="Enable GPU inference server (multiprocessing-based batching)",
    )
    parser.add_argument(
        "--gpu_server_max_batch",
        type=int,
        default=512,
        help="Max batch size for GPU server (default: 512)",
    )
    parser.add_argument(
        "--gpu_server_timeout_ms",
        type=float,
        default=10.0,
        help="GPU server batch timeout in ms (default: 10.0)",
    )
    parser.add_argument(
        "--use_batched_evaluator",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Use batched evaluator (default: true, ignored if GPU server enabled)",
    )
    parser.add_argument(
        "--use_thread_pool",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use thread pool (default: false, ignored if GPU server enabled)",
    )

    args = parser.parse_args()

    # Determine configurations
    if args.quick:
        worker_counts = QUICK_WORKER_COUNTS
        mcts_configs = QUICK_MCTS_CONFIGS
        games_per_config = min(args.games, 5)  # Limit games for quick test
        print("\nQUICK TEST MODE: Limited configurations")
    else:
        worker_counts = args.workers if args.workers else [32]  # Default: single worker count
        # If multiple worker counts specified, test all MCTS configs; otherwise just medium
        if args.workers and len(args.workers) > 1:
            mcts_configs = DEFAULT_MCTS_CONFIGS  # Test all MCTS configs
        else:
            mcts_configs = QUICK_MCTS_CONFIGS  # Default: medium MCTS only
        games_per_config = args.games

    # Print benchmark plan
    print("\n" + "="*80)
    print("SELF-PLAY PERFORMANCE BENCHMARK")
    print("="*80)
    print(f"Worker counts: {worker_counts}")
    print(f"MCTS configs: {[c[0] for c in mcts_configs]}")
    print(f"Games per config: {games_per_config}")
    print(f"Total configurations: {len(worker_counts) * len(mcts_configs)}")
    print(f"Device: {args.device}")
    print("="*80)

    # Run benchmark
    benchmark = SelfPlayBenchmark(device=args.device)
    results = benchmark.run_benchmarks(
        worker_counts,
        mcts_configs,
        games_per_config,
        use_gpu_server=args.use_gpu_server,
        gpu_server_max_batch=args.gpu_server_max_batch,
        gpu_server_timeout_ms=args.gpu_server_timeout_ms,
        use_batched_evaluator=args.use_batched_evaluator,
        use_thread_pool=args.use_thread_pool,
    )

    # Save and display results
    save_results_csv(results, args.output)
    print_results_table(results)

    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
