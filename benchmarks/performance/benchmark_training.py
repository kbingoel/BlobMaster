"""
Benchmark GPU training performance.

Tests different batch sizes and precision modes to find optimal training settings
for the AlphaZero training pipeline.

Usage:
    python ml/benchmark_training.py
    python ml/benchmark_training.py --batch-sizes 512 1024 --device cuda
    python ml/benchmark_training.py --quick  # Fast test

Output:
    - Console: Progress updates and results table
    - JSON: benchmark_training_results.json
"""

import argparse
import time
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder
from ml.training.trainer import NetworkTrainer
from ml.training.replay_buffer import ReplayBuffer


# Test configurations
DEFAULT_BATCH_SIZES = [256, 512, 1024, 2048]
QUICK_BATCH_SIZES = [512, 1024]

DEFAULT_PRECISIONS = ["fp32", "fp16"]
QUICK_PRECISIONS = ["fp32"]


class TrainingBenchmark:
    """Benchmark neural network training performance."""

    def __init__(self):
        """Initialize benchmark."""
        print("Initializing benchmark components...")

        # Create encoder for generating synthetic data
        self.encoder = StateEncoder()

        print("  - State encoder initialized")

    def create_network(self, device: str) -> BlobNet:
        """
        Create neural network.

        Args:
            device: Device to place network on

        Returns:
            Initialized network
        """
        network = BlobNet(
            state_dim=256,
            embedding_dim=256,
            num_layers=4,
            num_heads=8,
            feedforward_dim=512,
            dropout=0.1,
        )
        network.to(device)

        return network

    def generate_synthetic_data(
        self,
        num_examples: int,
        device: str = "cpu",
    ) -> ReplayBuffer:
        """
        Generate synthetic training data.

        Args:
            num_examples: Number of training examples to generate
            device: Device for data generation

        Returns:
            ReplayBuffer filled with synthetic examples
        """
        print(f"\nGenerating {num_examples:,} synthetic training examples...")

        buffer = ReplayBuffer(capacity=num_examples)

        batch_size = 10000
        num_batches = (num_examples + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_examples - batch_idx * batch_size)

            # Generate random states (256-dim)
            states = np.random.randn(current_batch_size, 256).astype(np.float32)

            # Generate random policies (52-dim, normalized)
            policies = np.random.rand(current_batch_size, 52).astype(np.float32)
            policies = policies / policies.sum(axis=1, keepdims=True)

            # Generate random values (-1 to 1)
            values = np.random.uniform(-1, 1, size=current_batch_size).astype(np.float32)

            # Create examples
            examples = [
                {
                    "state": states[i],
                    "policy": policies[i],
                    "value": values[i],
                    "player_position": 0,
                    "game_id": f"synthetic_{batch_idx}_{i}",
                    "move_number": 0,
                }
                for i in range(current_batch_size)
            ]

            buffer.add_examples(examples)

            if (batch_idx + 1) % 5 == 0:
                print(f"  Generated {len(buffer):,}/{num_examples:,} examples")

        print(f"  Complete: {len(buffer):,} examples")

        return buffer

    def benchmark_config(
        self,
        device: str,
        batch_size: int,
        precision: str,
        num_epochs: int,
        replay_buffer: ReplayBuffer,
    ) -> Dict[str, Any]:
        """
        Benchmark a specific training configuration.

        Args:
            device: Device ('cpu' or 'cuda')
            batch_size: Training batch size
            precision: Precision mode ('fp32' or 'fp16')
            num_epochs: Number of epochs to train
            replay_buffer: Replay buffer with training data

        Returns:
            Dictionary with performance metrics
        """
        print(f"\n{'='*70}")
        print(f"Testing: {device.upper()}, batch_size={batch_size}, precision={precision}")
        print(f"{'='*70}")

        # Check device availability
        if device == "cuda" and not torch.cuda.is_available():
            print("ERROR: CUDA not available, skipping")
            return {
                "device": device,
                "batch_size": batch_size,
                "precision": precision,
                "error": "CUDA not available",
            }

        # Create network and trainer
        try:
            network = self.create_network(device)
            use_mixed_precision = (precision == "fp16" and device == "cuda")

            trainer = NetworkTrainer(
                network=network,
                learning_rate=0.001,
                weight_decay=1e-4,
                use_mixed_precision=use_mixed_precision,
                device=device,
            )

            print(f"  - Network parameters: {self._count_parameters(network):,}")
            print(f"  - Mixed precision: {use_mixed_precision}")
            print(f"  - Replay buffer size: {len(replay_buffer):,}")
            print(f"  - Epochs: {num_epochs}")

        except Exception as e:
            print(f"ERROR: Failed to initialize: {e}")
            return {
                "device": device,
                "batch_size": batch_size,
                "precision": precision,
                "error": str(e),
            }

        # Record GPU memory before training (if CUDA)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            initial_memory_mb = torch.cuda.memory_allocated(device) / 1024 / 1024

        # Train for specified epochs
        print("\nTraining...")
        start_time = time.time()

        epoch_metrics_list = []
        total_batches = 0

        try:
            for epoch in range(num_epochs):
                epoch_start = time.time()

                epoch_metrics = trainer.train_epoch(
                    replay_buffer=replay_buffer,
                    batch_size=batch_size,
                )

                epoch_time = time.time() - epoch_start
                epoch_metrics["epoch_time"] = epoch_time

                epoch_metrics_list.append(epoch_metrics)

                # Calculate number of batches per epoch
                batches_per_epoch = max(1, len(replay_buffer) // batch_size)
                total_batches += batches_per_epoch

                # Print progress
                if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
                    print(f"  Epoch {epoch + 1}/{num_epochs}: "
                          f"loss={epoch_metrics['total_loss']:.4f}, "
                          f"time={epoch_time:.2f}s")

        except Exception as e:
            print(f"ERROR during training: {e}")
            if device == "cuda":
                torch.cuda.empty_cache()
            return {
                "device": device,
                "batch_size": batch_size,
                "precision": precision,
                "error": str(e),
            }

        elapsed_time = time.time() - start_time

        # Record GPU memory after training
        if device == "cuda":
            torch.cuda.synchronize(device)
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            final_memory_mb = torch.cuda.memory_allocated(device) / 1024 / 1024
            torch.cuda.empty_cache()
        else:
            peak_memory_mb = 0
            final_memory_mb = 0

        # Calculate metrics
        batches_per_second = total_batches / elapsed_time
        examples_per_second = (total_batches * batch_size) / elapsed_time

        avg_total_loss = np.mean([m["total_loss"] for m in epoch_metrics_list])
        avg_policy_loss = np.mean([m["policy_loss"] for m in epoch_metrics_list])
        avg_value_loss = np.mean([m["value_loss"] for m in epoch_metrics_list])
        avg_policy_accuracy = np.mean([m["policy_accuracy"] for m in epoch_metrics_list])

        # Results
        metrics = {
            "device": device,
            "batch_size": batch_size,
            "precision": precision,
            "num_epochs": num_epochs,
            "elapsed_time_sec": elapsed_time,
            "total_batches": total_batches,
            "batches_per_second": batches_per_second,
            "examples_per_second": examples_per_second,
            "avg_total_loss": avg_total_loss,
            "avg_policy_loss": avg_policy_loss,
            "avg_value_loss": avg_value_loss,
            "avg_policy_accuracy": avg_policy_accuracy,
            "peak_memory_mb": peak_memory_mb,
            "final_memory_mb": final_memory_mb,
        }

        print(f"\nResults:")
        print(f"  - Batches/second: {batches_per_second:.2f}")
        print(f"  - Examples/second: {examples_per_second:.1f}")
        print(f"  - Avg total loss: {avg_total_loss:.4f}")
        print(f"  - Avg policy accuracy: {avg_policy_accuracy:.1f}%")
        if device == "cuda":
            print(f"  - Peak VRAM: {peak_memory_mb:.1f} MB")

        return metrics

    def _count_parameters(self, network: torch.nn.Module) -> int:
        """Count network parameters."""
        return sum(p.numel() for p in network.parameters())

    def run_benchmarks(
        self,
        devices: List[str],
        batch_sizes: List[int],
        precisions: List[str],
        num_epochs: int,
        num_synthetic_examples: int,
    ) -> List[Dict[str, Any]]:
        """
        Run full benchmark suite.

        Args:
            devices: List of devices to test
            batch_sizes: List of batch sizes to test
            precisions: List of precision modes to test
            num_epochs: Epochs to train per configuration
            num_synthetic_examples: Synthetic training examples to generate

        Returns:
            List of benchmark results
        """
        # Generate synthetic data once
        replay_buffer = self.generate_synthetic_data(num_synthetic_examples)

        results = []

        total_configs = len(devices) * len(batch_sizes) * len(precisions)
        config_num = 0

        for device in devices:
            for batch_size in batch_sizes:
                for precision in precisions:
                    # Skip FP16 on CPU (not supported)
                    if device == "cpu" and precision == "fp16":
                        print(f"\nSkipping CPU + FP16 (not supported)")
                        continue

                    config_num += 1

                    print(f"\n\n{'#'*70}")
                    print(f"Configuration {config_num}/{total_configs}")
                    print(f"{'#'*70}")

                    metrics = self.benchmark_config(
                        device=device,
                        batch_size=batch_size,
                        precision=precision,
                        num_epochs=num_epochs,
                        replay_buffer=replay_buffer,
                    )

                    results.append(metrics)

                    # Brief pause between configs
                    time.sleep(1.0)

        return results


def save_results_json(results: List[Dict[str, Any]], filepath: str):
    """
    Save benchmark results to JSON.

    Args:
        results: List of benchmark result dictionaries
        filepath: Output JSON path
    """
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

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
    print(f"{'Device':<8} {'Batch':<8} {'Precision':<10} {'Batches/s':<12} {'Examples/s':<12} {'VRAM (MB)':<12}")
    print("-" * 80)

    # Print rows
    for r in valid_results:
        vram_str = f"{r['peak_memory_mb']:.0f}" if r['peak_memory_mb'] > 0 else "N/A"
        print(f"{r['device']:<8} "
              f"{r['batch_size']:<8} "
              f"{r['precision']:<10} "
              f"{r['batches_per_second']:<12.2f} "
              f"{r['examples_per_second']:<12.1f} "
              f"{vram_str:<12}")

    print("-" * 80)

    # Print recommendations
    print("\nRECOMMENDATIONS:")

    # Find best CUDA configuration
    cuda_results = [r for r in valid_results if r['device'] == 'cuda']
    if cuda_results:
        best_cuda = max(cuda_results, key=lambda x: x['examples_per_second'])
        print(f"  - Best CUDA config: batch_size={best_cuda['batch_size']}, "
              f"precision={best_cuda['precision']}")
        print(f"    -> {best_cuda['examples_per_second']:.1f} examples/sec, "
              f"Peak VRAM: {best_cuda['peak_memory_mb']:.0f} MB")

        # Compare FP16 vs FP32
        fp32_results = [r for r in cuda_results if r['precision'] == 'fp32']
        fp16_results = [r for r in cuda_results if r['precision'] == 'fp16']

        if fp32_results and fp16_results:
            print(f"\n  FP16 vs FP32 Speedup:")
            for fp32 in fp32_results:
                batch = fp32['batch_size']
                fp16 = next((r for r in fp16_results if r['batch_size'] == batch), None)
                if fp16:
                    speedup = fp16['examples_per_second'] / fp32['examples_per_second']
                    print(f"    - Batch {batch}: {speedup:.2f}x faster with FP16")

    # CPU baseline
    cpu_results = [r for r in valid_results if r['device'] == 'cpu']
    if cpu_results and cuda_results:
        best_cpu = max(cpu_results, key=lambda x: x['examples_per_second'])
        best_cuda = max(cuda_results, key=lambda x: x['examples_per_second'])
        speedup = best_cuda['examples_per_second'] / best_cpu['examples_per_second']
        print(f"\n  GPU vs CPU: {speedup:.1f}x faster on CUDA")

    # Error summary
    error_results = [r for r in results if "error" in r]
    if error_results:
        print(f"\n  ERRORS: {len(error_results)} configurations failed")
        for r in error_results:
            print(f"    - {r['device']}, batch={r['batch_size']}, {r['precision']}: {r['error']}")


def main():
    """Run training benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark training performance")
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        choices=["cpu", "cuda"],
        default=["cpu", "cuda"],
        help="Devices to test (default: cpu cuda)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        help="Batch sizes to test (default: 256 512 1024 2048)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Epochs per configuration (default: 5)",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=100000,
        help="Synthetic training examples (default: 100000)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with fewer configurations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_training_results.json",
        help="Output JSON path",
    )

    args = parser.parse_args()

    # Determine configurations
    if args.quick:
        batch_sizes = QUICK_BATCH_SIZES
        precisions = QUICK_PRECISIONS
        num_epochs = min(args.epochs, 3)
        num_examples = min(args.examples, 50000)
        print("\nQUICK TEST MODE: Limited configurations")
    else:
        batch_sizes = args.batch_sizes if args.batch_sizes else DEFAULT_BATCH_SIZES
        precisions = DEFAULT_PRECISIONS
        num_epochs = args.epochs
        num_examples = args.examples

    # Print benchmark plan
    print("\n" + "="*80)
    print("TRAINING PERFORMANCE BENCHMARK")
    print("="*80)
    print(f"Devices: {args.devices}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Precisions: {precisions}")
    print(f"Epochs per config: {num_epochs}")
    print(f"Synthetic examples: {num_examples:,}")
    print(f"Total configurations: {len(args.devices) * len(batch_sizes) * len(precisions)}")
    print("="*80)

    # Run benchmark
    benchmark = TrainingBenchmark()
    results = benchmark.run_benchmarks(
        devices=args.devices,
        batch_sizes=batch_sizes,
        precisions=precisions,
        num_epochs=num_epochs,
        num_synthetic_examples=num_examples,
    )

    # Save and display results
    save_results_json(results, args.output)
    print_results_table(results)

    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
