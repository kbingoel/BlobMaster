"""
Benchmark full training iteration (self-play + network training).

Simulates one training iteration at reduced scale to project full-scale performance.
This helps estimate the total training time for 500 iterations with adaptive curriculum.

Note: This measures Phase 1 (independent rounds) self-play + network training epochs.
The adaptive curriculum adjusts both MCTS depth and training units over iterations.

Usage:
    python ml/benchmark_iteration.py
    python ml/benchmark_iteration.py --games 500 --epochs 5
    python ml/benchmark_iteration.py --device cuda --iteration 250

Output:
    - Console: Phase breakdown and projections
    - JSON: benchmark_iteration_results.json
"""

import argparse
import time
import json
import torch
from pathlib import Path
from typing import Dict, Any

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.training.selfplay import SelfPlayEngine
from ml.training.replay_buffer import ReplayBuffer
from ml.training.trainer import NetworkTrainer


class IterationBenchmark:
    """Benchmark full training iteration at reduced scale."""

    def __init__(
        self,
        num_workers: int = 16,
        num_determinizations: int = 3,
        simulations_per_det: int = 30,
        device: str = "cuda",
    ):
        """
        Initialize iteration benchmark.

        Args:
            num_workers: Parallel self-play workers
            num_determinizations: MCTS determinizations
            simulations_per_det: MCTS simulations per determinization
            device: Device for training
        """
        print("Initializing iteration benchmark components...")

        self.num_workers = num_workers
        self.num_determinizations = num_determinizations
        self.simulations_per_det = simulations_per_det
        self.device = device

        # Create network (BASELINE: 4.9M parameters)
        self.network = BlobNet(
            state_dim=256,
            embedding_dim=256,      # Baseline: 256
            num_layers=6,           # Baseline: 6 (not 4!)
            num_heads=8,            # Baseline: 8
            feedforward_dim=1024,   # Baseline: 1024 (not 512!)
            dropout=0.1,
        )
        self.network.to(device)

        # Create encoder and masker
        self.encoder = StateEncoder()
        self.masker = ActionMasker()

        # Create self-play engine
        self.selfplay_engine = SelfPlayEngine(
            network=self.network,
            encoder=self.encoder,
            masker=self.masker,
            num_workers=num_workers,
            num_determinizations=num_determinizations,
            simulations_per_determinization=simulations_per_det,
            device=device,
            use_thread_pool=False,  # Use multiprocessing, not threads (GIL bottleneck)
        )

        # Create replay buffer
        self.replay_buffer = ReplayBuffer(capacity=500_000)

        # Create trainer
        self.trainer = NetworkTrainer(
            network=self.network,
            learning_rate=0.001,
            weight_decay=1e-4,
            use_mixed_precision=True,  # Use FP16 for faster training
            device=device,
        )

        print(f"  - Network: {self._count_parameters():,} parameters")
        print(f"  - Self-play workers: {num_workers}")
        print(f"  - MCTS config: {num_determinizations} det × {simulations_per_det} sims")
        print(f"  - Device: {device}")

    def _count_parameters(self) -> int:
        """Count network parameters."""
        return sum(p.numel() for p in self.network.parameters())

    def run_selfplay_phase(
        self,
        num_games: int,
        num_players: int = 4,
        cards_to_deal: int = 5,
    ) -> Dict[str, Any]:
        """
        Run self-play phase (Phase 1: Independent Rounds).

        Args:
            num_games: Number of rounds to generate
            num_players: Players per round
            cards_to_deal: Cards to deal per player

        Returns:
            Dictionary with self-play metrics
        """
        print(f"\n{'='*70}")
        print("PHASE 1: SELF-PLAY (Independent Rounds)")
        print(f"{'='*70}")
        print(f"Generating {num_games:,} rounds ({num_players} players, {cards_to_deal} cards)")

        # Progress tracking
        games_generated = [0]
        last_update_time = [time.time()]

        def progress_callback(count: int):
            games_generated[0] = count
            current_time = time.time()
            if current_time - last_update_time[0] >= 10.0 or count == num_games:
                elapsed = current_time - start_time
                rate = count / elapsed if elapsed > 0 else 0
                print(f"  Progress: {count:,}/{num_games:,} rounds ({rate:.1f} rounds/sec)")
                last_update_time[0] = current_time

        # Generate games
        start_time = time.time()

        examples = self.selfplay_engine.generate_games(
            num_games=num_games,
            num_players=num_players,
            cards_to_deal=cards_to_deal,
            progress_callback=progress_callback,
        )

        elapsed_time = time.time() - start_time

        # Add to replay buffer
        self.replay_buffer.add_examples(examples)

        # Calculate metrics
        rounds_per_minute = (num_games / elapsed_time) * 60.0
        seconds_per_round = elapsed_time / num_games

        metrics = {
            "num_rounds": num_games,
            "num_examples": len(examples),
            "elapsed_time_sec": elapsed_time,
            "elapsed_time_min": elapsed_time / 60.0,
            "rounds_per_minute": rounds_per_minute,
            "seconds_per_round": seconds_per_round,
            "replay_buffer_size": len(self.replay_buffer),
        }

        print(f"\nSelf-Play Complete:")
        print(f"  - Rounds generated: {num_games:,}")
        print(f"  - Training examples: {len(examples):,}")
        print(f"  - Time: {elapsed_time / 60:.1f} minutes")
        print(f"  - Rate: {rounds_per_minute:.1f} rounds/min")
        print(f"  - Replay buffer size: {len(self.replay_buffer):,}")

        return metrics

    def run_training_phase(
        self,
        num_epochs: int,
        batch_size: int = 512,
    ) -> Dict[str, Any]:
        """
        Run training phase.

        Args:
            num_epochs: Number of training epochs
            batch_size: Training batch size

        Returns:
            Dictionary with training metrics
        """
        print(f"\n{'='*70}")
        print("PHASE 2: TRAINING")
        print(f"{'='*70}")
        print(f"Training for {num_epochs} epochs (batch_size={batch_size})")

        # Check buffer size
        if len(self.replay_buffer) < 1000:
            print("WARNING: Replay buffer too small for training")
            return {
                "num_epochs": num_epochs,
                "elapsed_time_sec": 0.0,
                "elapsed_time_min": 0.0,
                "avg_total_loss": 0.0,
                "avg_policy_accuracy": 0.0,
                "error": "Replay buffer too small",
            }

        # Train for specified epochs
        start_time = time.time()

        epoch_metrics_list = []

        for epoch in range(num_epochs):
            epoch_start = time.time()

            epoch_metrics = self.trainer.train_epoch(
                replay_buffer=self.replay_buffer,
                batch_size=batch_size,
            )

            epoch_time = time.time() - epoch_start

            epoch_metrics_list.append(epoch_metrics)

            # Print progress
            if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
                print(f"  Epoch {epoch + 1}/{num_epochs}: "
                      f"loss={epoch_metrics['total_loss']:.4f}, "
                      f"acc={epoch_metrics['policy_accuracy']:.1f}%, "
                      f"time={epoch_time:.1f}s")

        elapsed_time = time.time() - start_time

        # Calculate average metrics
        import numpy as np
        avg_total_loss = np.mean([m["total_loss"] for m in epoch_metrics_list])
        avg_policy_loss = np.mean([m["policy_loss"] for m in epoch_metrics_list])
        avg_value_loss = np.mean([m["value_loss"] for m in epoch_metrics_list])
        avg_policy_accuracy = np.mean([m["policy_accuracy"] for m in epoch_metrics_list])

        metrics = {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "elapsed_time_sec": elapsed_time,
            "elapsed_time_min": elapsed_time / 60.0,
            "avg_total_loss": avg_total_loss,
            "avg_policy_loss": avg_policy_loss,
            "avg_value_loss": avg_value_loss,
            "avg_policy_accuracy": avg_policy_accuracy,
        }

        print(f"\nTraining Complete:")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - Time: {elapsed_time / 60:.1f} minutes")
        print(f"  - Avg total loss: {avg_total_loss:.4f}")
        print(f"  - Avg policy accuracy: {avg_policy_accuracy:.1f}%")

        return metrics

    def run_iteration(
        self,
        num_games: int,
        num_epochs: int,
        batch_size: int = 512,
    ) -> Dict[str, Any]:
        """
        Run full iteration (self-play + network training).

        Args:
            num_games: Number of rounds for self-play (Phase 1: Independent Rounds)
            num_epochs: Number of network training epochs
            batch_size: Training batch size

        Returns:
            Dictionary with iteration metrics and projections
        """
        print(f"\n{'#'*70}")
        print("RUNNING FULL ITERATION BENCHMARK")
        print(f"{'#'*70}")
        print(f"Configuration:")
        print(f"  - Rounds (self-play): {num_games:,}")
        print(f"  - Training epochs: {num_epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"{'#'*70}\n")

        iteration_start_time = time.time()

        # Phase 1: Self-play
        selfplay_metrics = self.run_selfplay_phase(
            num_games=num_games,
            num_players=4,
            cards_to_deal=5,
        )

        # Phase 2: Training
        training_metrics = self.run_training_phase(
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

        # Total iteration time
        iteration_time = time.time() - iteration_start_time

        # Calculate projections for full-scale iteration
        # Assume full iteration = 10,000 rounds (adaptive curriculum may vary)
        full_scale_rounds = 10_000
        scale_factor = full_scale_rounds / num_games

        projected_selfplay_time = selfplay_metrics["elapsed_time_min"] * scale_factor
        projected_training_time = training_metrics["elapsed_time_min"]  # Same epochs
        projected_iteration_time = projected_selfplay_time + projected_training_time

        # Calculate projections for 500 iterations
        total_iterations = 500
        projected_total_time_hours = (projected_iteration_time * total_iterations) / 60.0
        projected_total_time_days = projected_total_time_hours / 24.0

        # Combine metrics
        results = {
            "configuration": {
                "num_rounds": num_games,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "num_workers": self.num_workers,
                "num_determinizations": self.num_determinizations,
                "simulations_per_det": self.simulations_per_det,
                "device": self.device,
            },
            "selfplay_phase": selfplay_metrics,
            "training_phase": training_metrics,
            "iteration_summary": {
                "total_time_min": iteration_time / 60.0,
                "selfplay_time_min": selfplay_metrics["elapsed_time_min"],
                "training_time_min": training_metrics["elapsed_time_min"],
                "selfplay_percentage": (selfplay_metrics["elapsed_time_min"] / (iteration_time / 60.0)) * 100.0,
                "training_percentage": (training_metrics["elapsed_time_min"] / (iteration_time / 60.0)) * 100.0,
            },
            "projections": {
                "full_scale_rounds": full_scale_rounds,
                "scale_factor": scale_factor,
                "projected_selfplay_time_min": projected_selfplay_time,
                "projected_training_time_min": projected_training_time,
                "projected_iteration_time_min": projected_iteration_time,
                "total_iterations": total_iterations,
                "projected_total_time_hours": projected_total_time_hours,
                "projected_total_time_days": projected_total_time_days,
            },
        }

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results: Dict[str, Any]):
        """
        Print iteration summary and projections.

        Args:
            results: Iteration results dictionary
        """
        print(f"\n\n{'='*70}")
        print("ITERATION BENCHMARK SUMMARY")
        print(f"{'='*70}\n")

        # Current iteration breakdown
        print("Current Iteration (1/10th scale):")
        print(f"  - Self-play: {results['selfplay_phase']['elapsed_time_min']:.1f} minutes "
              f"({results['iteration_summary']['selfplay_percentage']:.0f}% of iteration)")
        print(f"  - Training: {results['training_phase']['elapsed_time_min']:.1f} minutes "
              f"({results['iteration_summary']['training_percentage']:.0f}% of iteration)")
        print(f"  - Total: {results['iteration_summary']['total_time_min']:.1f} minutes")

        # Projections
        proj = results['projections']
        print(f"\nProjected Full-Scale Iteration (10,000 rounds):")
        print(f"  - Self-play: {proj['projected_selfplay_time_min']:.1f} minutes")
        print(f"  - Training: {proj['projected_training_time_min']:.1f} minutes")
        print(f"  - Total: {proj['projected_iteration_time_min']:.1f} minutes")

        print(f"\nProjected Full Training Run ({proj['total_iterations']} iterations):")
        print(f"  - Total time: {proj['projected_total_time_hours']:.1f} hours")
        print(f"  - Total time: {proj['projected_total_time_days']:.1f} days")

        # Bottleneck analysis
        selfplay_pct = results['iteration_summary']['selfplay_percentage']
        print(f"\nBottleneck Analysis:")
        if selfplay_pct > 70:
            print(f"  - Self-play is the bottleneck ({selfplay_pct:.0f}% of iteration time)")
            print(f"  - Recommendation: GPU-batched MCTS would significantly reduce training time")
        elif selfplay_pct > 50:
            print(f"  - Self-play is the primary bottleneck ({selfplay_pct:.0f}% of iteration time)")
            print(f"  - Recommendation: Consider GPU-batched MCTS optimization")
        else:
            print(f"  - Training is balanced with self-play")
            print(f"  - Recommendation: Current configuration is reasonable")

        print(f"\n{'='*70}\n")

    def shutdown(self):
        """Shutdown components."""
        self.selfplay_engine.shutdown()


def save_results_json(results: Dict[str, Any], filepath: str):
    """
    Save benchmark results to JSON.

    Args:
        results: Benchmark results dictionary
        filepath: Output JSON path
    """
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {filepath}")


def main():
    """Run iteration benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark full training iteration")
    parser.add_argument(
        "--games",
        type=int,
        default=500,
        help="Rounds for self-play (default: 500 = 1/20th scale for quick screening)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs (default: 5 for quick screening)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Training batch size (default: 512)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Self-play workers (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Training device (default: cuda)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_iteration_results.json",
        help="Output JSON path",
    )

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Print benchmark plan
    print("\n" + "="*70)
    print("ITERATION PERFORMANCE BENCHMARK (Phase 1: Independent Rounds)")
    print("="*70)
    print(f"Rounds: {args.games:,} (reduced scale for quick testing)")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {args.workers}")
    print(f"Device: {args.device}")
    print("="*70)

    # Run benchmark
    benchmark = IterationBenchmark(
        num_workers=args.workers,
        num_determinizations=3,
        simulations_per_det=30,
        device=args.device,
    )

    try:
        results = benchmark.run_iteration(
            num_games=args.games,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
        )

        # Save results
        save_results_json(results, args.output)

    finally:
        # Always shutdown
        benchmark.shutdown()

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
