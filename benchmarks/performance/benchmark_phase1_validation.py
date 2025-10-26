"""
Phase 1 (GPU-Batched MCTS) Validation Benchmark

Tests whether Phase 1 intra-game batching with virtual loss mechanism
provides the predicted 2-10x speedup over baseline sequential MCTS.

This benchmark implements:
- Streaming CSV output (writes after each game completion)
- Real-time console progress with timestamp
- Batch size sweep to find optimal configuration
- Fast iteration mode (start with 5 games for quick feedback)
- Crash-resilient (preserves partial results)

Usage:
    # Quick validation (5 games, fast feedback)
    python benchmarks/performance/benchmark_phase1_validation.py \\
        --games 5 \\
        --batch-start 1 \\
        --batch-end 30 \\
        --batch-step 10

    # Fine-grained sweep
    python benchmarks/performance/benchmark_phase1_validation.py \\
        --games 20 \\
        --batch-start 10 \\
        --batch-end 40 \\
        --batch-step 2

    # Production validation at discovered optimum
    python benchmarks/performance/benchmark_phase1_validation.py \\
        --games 100 \\
        --batch-size 30
"""

import argparse
import time
import csv
import torch
import psutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.training.selfplay import SelfPlayEngine


class StreamingCSVWriter:
    """CSV writer that flushes after each row for crash-resilience."""

    def __init__(self, filepath: Path):
        """Initialize streaming CSV writer."""
        self.filepath = filepath
        self.file = None
        self.writer = None
        self.fieldnames = [
            "timestamp",
            "config_id",
            "batch_size",
            "game_num",
            "elapsed_sec",
            "games_per_min",
            "gpu_util",
            "status"
        ]

        # Create parent directory if needed
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Open file in append mode
        self.file = open(self.filepath, 'a', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)

        # Write header if file is empty
        if self.filepath.stat().st_size == 0:
            self.writer.writeheader()
            self.file.flush()

    def write_row(self, row: Dict[str, Any]):
        """Write row and flush immediately."""
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        """Close the file."""
        if self.file:
            self.file.close()


def get_gpu_utilization() -> float:
    """Get GPU utilization percentage."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(info.gpu)
    except:
        return 0.0


class Phase1Benchmark:
    """Benchmark Phase 1 intra-game batching vs baseline."""

    def __init__(self, device: str = "cuda"):
        """Initialize benchmark."""
        self.device = device

        print(f"\n{'='*80}")
        print("Phase 1 (GPU-Batched MCTS) Validation Benchmark")
        print(f"{'='*80}")
        print(f"Device: {device}")
        print("Initializing components...")

        # Create network (small version for faster inference)
        self.network = BlobNet(
            state_dim=256,
            embedding_dim=128,
            num_layers=2,
            num_heads=4,
            feedforward_dim=256,
            dropout=0.0,
        )
        self.network.to(device)
        self.network.eval()

        self.encoder = StateEncoder()
        self.masker = ActionMasker()

        param_count = sum(p.numel() for p in self.network.parameters())
        print(f"Network parameters: {param_count:,}")
        print(f"{'='*80}\n")

    def benchmark_games_streaming(
        self,
        engine: SelfPlayEngine,
        config_id: str,
        batch_size: Optional[int],
        num_games: int,
        csv_writer: StreamingCSVWriter,
    ) -> List[Dict[str, Any]]:
        """
        Benchmark multiple games through a single engine with streaming output.

        Generates games one at a time to provide streaming progress updates,
        but reuses the same engine instance to avoid overhead.

        Returns:
            List of game results
        """
        results = []

        for game_num in range(1, num_games + 1):
            # GPU monitoring
            gpu_util_start = get_gpu_utilization()

            try:
                # Time single game generation
                start_time = time.time()

                examples = engine.generate_games(
                    num_games=1,
                    num_players=4,
                    cards_to_deal=5,
                )

                elapsed = time.time() - start_time
                gpu_util_end = get_gpu_utilization()

                # Calculate metrics
                games_per_min = 60.0 / elapsed if elapsed > 0 else 0
                avg_gpu_util = (gpu_util_start + gpu_util_end) / 2

                result = {
                    "timestamp": datetime.now().isoformat(),
                    "config_id": config_id,
                    "batch_size": batch_size if batch_size is not None else "None",
                    "game_num": game_num,
                    "elapsed_sec": f"{elapsed:.3f}",
                    "games_per_min": f"{games_per_min:.1f}",
                    "gpu_util": f"{avg_gpu_util:.1f}",
                    "status": "complete"
                }

                # Stream to CSV immediately
                csv_writer.write_row(result)

                # Console output
                batch_str = f"batch={batch_size}" if batch_size is not None else "batch=None"
                print(f"[{datetime.now():%H:%M:%S}] {config_id:20} ({batch_str:12}) | "
                      f"Game {game_num}/{num_games} | "
                      f"{elapsed:5.2f}s | {games_per_min:6.1f} games/min | "
                      f"GPU {avg_gpu_util:5.1f}% "
                      f"{'[OK]' if sys.platform.startswith('win') else '✓'}")

                results.append(result)

            except Exception as e:
                # Stream error immediately
                error_result = {
                    "timestamp": datetime.now().isoformat(),
                    "config_id": config_id,
                    "batch_size": batch_size if batch_size is not None else "None",
                    "game_num": game_num,
                    "elapsed_sec": "ERROR",
                    "games_per_min": "ERROR",
                    "gpu_util": "ERROR",
                    "status": f"error: {str(e)[:50]}"
                }
                csv_writer.write_row(error_result)

                print(f"[{datetime.now():%H:%M:%S}] {config_id:20} | "
                      f"Game {game_num}/{num_games} | ERROR: {e}")

                results.append(error_result)

        return results

    def benchmark_configuration(
        self,
        config_id: str,
        batch_size: Optional[int],
        num_games: int,
        num_workers: int,
        num_determinizations: int,
        simulations_per_det: int,
        csv_writer: StreamingCSVWriter,
    ) -> List[Dict[str, Any]]:
        """
        Benchmark a configuration across multiple games.

        Creates a single engine instance and reuses it for all games
        to avoid initialization overhead.

        Returns:
            List of game results
        """
        batch_str = f"batch={batch_size}" if batch_size is not None else "batch=None (baseline)"
        print(f"\n{'='*80}")
        print(f"Testing {config_id}: {batch_str}")
        print(f"  Workers: {num_workers}")
        print(f"  MCTS: {num_determinizations} det × {simulations_per_det} sims = "
              f"{num_determinizations * simulations_per_det} total sims/move")
        print(f"  Games: {num_games}")
        print(f"{'='*80}\n")

        # Create engine once for all games in this configuration
        engine = SelfPlayEngine(
            network=self.network,
            encoder=self.encoder,
            masker=self.masker,
            num_workers=num_workers,
            num_determinizations=num_determinizations,
            simulations_per_determinization=simulations_per_det,
            device=self.device,
            use_batched_evaluator=False,  # No Phase 2/3 batching
            use_thread_pool=False,  # Use multiprocessing
            mcts_batch_size=batch_size,  # Phase 1 batching parameter
        )

        try:
            # Generate all games through this engine
            results = self.benchmark_games_streaming(
                engine=engine,
                config_id=config_id,
                batch_size=batch_size,
                num_games=num_games,
                csv_writer=csv_writer,
            )
        finally:
            # Cleanup engine
            engine.shutdown()

        return results

    def run_batch_size_sweep(
        self,
        batch_start: int,
        batch_end: int,
        batch_step: int,
        num_games: int,
        num_workers: int,
        num_determinizations: int,
        simulations_per_det: int,
        csv_writer: StreamingCSVWriter,
        include_baseline: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run benchmark sweep across batch sizes.

        Returns:
            Dict mapping config_id to list of game results
        """
        all_results = {}

        # Baseline (no batching)
        if include_baseline:
            config_id = "baseline"
            results = self.benchmark_configuration(
                config_id=config_id,
                batch_size=None,
                num_games=num_games,
                num_workers=num_workers,
                num_determinizations=num_determinizations,
                simulations_per_det=simulations_per_det,
                csv_writer=csv_writer,
            )
            all_results[config_id] = results

        # Phase 1 with varying batch sizes
        for batch_size in range(batch_start, batch_end + 1, batch_step):
            config_id = f"phase1_bs{batch_size}"
            results = self.benchmark_configuration(
                config_id=config_id,
                batch_size=batch_size,
                num_games=num_games,
                num_workers=num_workers,
                num_determinizations=num_determinizations,
                simulations_per_det=simulations_per_det,
                csv_writer=csv_writer,
            )
            all_results[config_id] = results

        return all_results


def print_summary(results: Dict[str, List[Dict[str, Any]]]):
    """Print summary of benchmark results."""
    print(f"\n\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Config':<20} {'Batch Size':<12} {'Games':<8} {'Avg Games/Min':<15} {'Speedup':<10}")
    print('-' * 80)

    baseline_avg = None

    for config_id, games in results.items():
        # Filter out errors
        valid_games = [g for g in games if g['status'] == 'complete']

        if not valid_games:
            print(f"{config_id:<20} {'N/A':<12} {len(games):<8} {'ERROR':<15} {'N/A':<10}")
            continue

        # Calculate average
        games_per_min_vals = [float(g['games_per_min']) for g in valid_games]
        avg_games_per_min = sum(games_per_min_vals) / len(games_per_min_vals)

        # Get batch size
        batch_size = valid_games[0]['batch_size']

        # Calculate speedup vs baseline
        if config_id == "baseline":
            baseline_avg = avg_games_per_min
            speedup_str = "1.0x"
        elif baseline_avg is not None:
            speedup = avg_games_per_min / baseline_avg
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"

        print(f"{config_id:<20} {str(batch_size):<12} {len(valid_games):<8} "
              f"{avg_games_per_min:>14.1f} {speedup_str:<10}")

    print('-' * 80)

    # Recommendation
    print("\nRECOMMENDATION:")
    if baseline_avg is not None:
        # Find best Phase 1 config
        phase1_configs = [(config_id, games) for config_id, games in results.items()
                         if config_id.startswith("phase1_")]

        if phase1_configs:
            best_config_id = None
            best_avg = 0

            for config_id, games in phase1_configs:
                valid_games = [g for g in games if g['status'] == 'complete']
                if valid_games:
                    avg = sum(float(g['games_per_min']) for g in valid_games) / len(valid_games)
                    if avg > best_avg:
                        best_avg = avg
                        best_config_id = config_id

            if best_config_id:
                speedup = best_avg / baseline_avg
                batch_size = best_config_id.replace("phase1_bs", "")
                print(f"  Best configuration: {best_config_id} (batch_size={batch_size})")
                print(f"  Speedup: {speedup:.2f}x over baseline")
                print(f"  Performance: {best_avg:.1f} games/min vs {baseline_avg:.1f} games/min baseline")

                if speedup >= 5.0:
                    print(f"  [EXCELLENT] Exceeds 5x target speedup!")
                elif speedup >= 2.0:
                    print(f"  [GOOD] Exceeds 2x minimum speedup")
                elif speedup >= 1.5:
                    print(f"  [MODERATE] Below 2x target but shows improvement")
                else:
                    print(f"  [POOR] Insufficient speedup, investigate overhead")

    print(f"\n{'='*80}\n")


def main():
    """Run Phase 1 validation benchmark."""
    parser = argparse.ArgumentParser(
        description="Phase 1 (GPU-Batched MCTS) Validation Benchmark"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=5,
        help="Games per configuration (default: 5 for quick validation)"
    )
    parser.add_argument(
        "--batch-start",
        type=int,
        default=1,
        help="Starting batch size (default: 1)"
    )
    parser.add_argument(
        "--batch-end",
        type=int,
        default=30,
        help="Ending batch size (default: 30)"
    )
    parser.add_argument(
        "--batch-step",
        type=int,
        default=10,
        help="Batch size step (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Test only this specific batch size (skips sweep)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of parallel workers (default: 32)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for neural network (default: cuda)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/phase1_validation.csv",
        help="Output CSV path (default: results/phase1_validation.csv)"
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline test (only test Phase 1 batching)"
    )

    args = parser.parse_args()

    # Print configuration
    print(f"\n{'='*80}")
    print("BENCHMARK CONFIGURATION")
    print(f"{'='*80}")
    print(f"Games per config: {args.games}")
    print(f"Workers: {args.workers}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")

    if args.batch_size is not None:
        print(f"Mode: Single batch size test (batch_size={args.batch_size})")
    else:
        print(f"Mode: Batch size sweep")
        print(f"  Start: {args.batch_start}")
        print(f"  End: {args.batch_end}")
        print(f"  Step: {args.batch_step}")

    print(f"Include baseline: {not args.no_baseline}")
    print(f"{'='*80}\n")

    # Create CSV writer
    output_path = Path(args.output)
    csv_writer = StreamingCSVWriter(output_path)

    try:
        # Initialize benchmark
        benchmark = Phase1Benchmark(device=args.device)

        # Run benchmark
        if args.batch_size is not None:
            # Single batch size test
            results = {}

            if not args.no_baseline:
                baseline_results = benchmark.benchmark_configuration(
                    config_id="baseline",
                    batch_size=None,
                    num_games=args.games,
                    num_workers=args.workers,
                    num_determinizations=3,
                    simulations_per_det=30,
                    csv_writer=csv_writer,
                )
                results["baseline"] = baseline_results

            phase1_results = benchmark.benchmark_configuration(
                config_id=f"phase1_bs{args.batch_size}",
                batch_size=args.batch_size,
                num_games=args.games,
                num_workers=args.workers,
                num_determinizations=3,
                simulations_per_det=30,
                csv_writer=csv_writer,
            )
            results[f"phase1_bs{args.batch_size}"] = phase1_results
        else:
            # Batch size sweep
            results = benchmark.run_batch_size_sweep(
                batch_start=args.batch_start,
                batch_end=args.batch_end,
                batch_step=args.batch_step,
                num_games=args.games,
                num_workers=args.workers,
                num_determinizations=3,
                simulations_per_det=30,
                csv_writer=csv_writer,
                include_baseline=not args.no_baseline,
            )

        # Print summary
        print_summary(results)

        print(f"Results saved to: {output_path}")
        print(f"\nBenchmark complete!")

    finally:
        csv_writer.close()


if __name__ == "__main__":
    main()
