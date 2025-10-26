"""
Quick validation test for diagnostic benchmark.

Tests with just 2 configurations to verify everything works.
"""

import torch
import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.benchmark_diagnostic import (
    run_benchmark_configuration,
    create_test_network,
    GPUMonitor
)


def main():
    """Run quick validation test."""

    print("="*80)
    print("QUICK DIAGNOSTIC VALIDATION TEST")
    print("="*80)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    # Test GPU monitor
    print("\nTesting GPU monitor...")
    monitor = GPUMonitor(interval_seconds=1.0)
    monitor.start()

    import time
    time.sleep(3)

    stats = monitor.stop()
    print(f"  GPU Utilization: {stats['avg_gpu_utilization']:.1f}%")
    print(f"  GPU Memory: {stats['avg_gpu_memory_mb']:.0f} MB")
    print("  [OK] GPU monitor works!")

    # Create network
    print("\nCreating network...")
    network = create_test_network(device="cuda")
    encoder = StateEncoder()
    masker = ActionMasker()

    # Test 1: Small config (4 workers, batched)
    print("\n" + "="*80)
    print("TEST 1: 4 workers with batching")
    print("="*80)

    result1 = run_benchmark_configuration(
        network=network,
        encoder=encoder,
        masker=masker,
        num_workers=4,
        cards_to_deal=3,
        num_games=5,  # Small number for quick test
        use_batched_evaluator=True,
        use_thread_pool=True,
        batch_size=256,
        batch_timeout_ms=10.0,
        num_determinizations=3,
        simulations_per_det=10,
        device="cuda"
    )

    if not result1.success:
        print("\n[FAIL] Test 1 FAILED!")
        return

    print(f"\n[PASS] Test 1 PASSED: {result1.games_per_min:.1f} games/min, {result1.avg_gpu_utilization:.1f}% GPU")

    # Test 2: Larger config (16 workers, batched)
    print("\n" + "="*80)
    print("TEST 2: 16 workers with batching")
    print("="*80)

    result2 = run_benchmark_configuration(
        network=network,
        encoder=encoder,
        masker=masker,
        num_workers=16,
        cards_to_deal=3,
        num_games=5,  # Small number for quick test
        use_batched_evaluator=True,
        use_thread_pool=True,
        batch_size=512,
        batch_timeout_ms=10.0,
        num_determinizations=3,
        simulations_per_det=10,
        device="cuda"
    )

    if not result2.success:
        print("\n[FAIL] Test 2 FAILED!")
        return

    print(f"\n[PASS] Test 2 PASSED: {result2.games_per_min:.1f} games/min, {result2.avg_gpu_utilization:.1f}% GPU")

    # Compare
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    print(f"\n4 workers:  {result1.games_per_min:>7.1f} games/min, {result1.avg_gpu_utilization:>5.1f}% GPU, batch size {result1.avg_batch_size:>5.1f}")
    print(f"16 workers: {result2.games_per_min:>7.1f} games/min, {result2.avg_gpu_utilization:>5.1f}% GPU, batch size {result2.avg_batch_size:>5.1f}")

    speedup = result2.games_per_min / result1.games_per_min if result1.games_per_min > 0 else 0
    print(f"\nSpeedup from 4->16 workers: {speedup:.2f}x")

    if speedup > 1.5:
        print("[OK] Scaling looks good - more workers help!")
    elif speedup > 1.0:
        print("[WARN] Scaling is weak - may need more workers for GPU saturation")
    else:
        print("[FAIL] Scaling is broken - more workers make it slower!")

    print("\n" + "="*80)
    print("VALIDATION COMPLETE - Ready to run full diagnostic suite!")
    print("="*80)
    print("\nTo run full diagnostics:")
    print("  python ml/run_full_diagnostics.py")
    print("\nOr just the benchmark:")
    print("  python ml/benchmark_diagnostic.py")


if __name__ == "__main__":
    main()
