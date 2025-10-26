"""
Test script for GPU Inference Server (Phase 3.5)

This script tests the correctness and performance of the GPU inference server
by comparing it against direct inference and measuring throughput.
"""

import torch
import time
import numpy as np
from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.game.blob import BlobGame
from ml.training.selfplay import SelfPlayEngine


def test_correctness():
    """
    Test that GPU server produces same results as direct inference.
    """
    print("=" * 80)
    print("CORRECTNESS TEST: GPU Server vs Direct Inference")
    print("=" * 80)

    # Create small network for testing
    network = BlobNet(
        state_dim=256,
        action_dim=52,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
    )
    network.eval()
    network.to("cuda")

    encoder = StateEncoder()
    masker = ActionMasker()

    # Create test game state
    game = BlobGame(num_players=4)
    game.setup_round(cards_to_deal=5)
    player = game.players[0]

    # Encode state
    state = encoder.encode(game, player)
    legal_actions, legal_mask = masker.get_legal_actions_and_mask(
        game, player, is_bidding=True
    )

    # Direct inference
    print("\n1. Testing direct inference...")
    with torch.no_grad():
        state_gpu = state.to("cuda")
        mask_gpu = legal_mask.to("cuda")
        policy_direct, value_direct = network(state_gpu, mask_gpu)
        policy_direct = policy_direct.cpu()
        value_direct = value_direct.cpu()

    print(f"   Direct inference - Policy shape: {policy_direct.shape}, Value: {value_direct.item():.4f}")

    # GPU server inference
    print("\n2. Testing GPU server inference...")
    from ml.mcts.gpu_server import GPUInferenceServer

    server = GPUInferenceServer(
        network=network,
        device="cuda",
        max_batch_size=128,
        timeout_ms=10.0,
    )
    server.start()

    # Give server time to initialize
    time.sleep(0.5)

    try:
        # Create client and test
        client = server.create_client("test_client")
        policy_server, value_server = client.evaluate(state, legal_mask, timeout=5.0)

        print(f"   GPU server - Policy shape: {policy_server.shape}, Value: {value_server.item():.4f}")

        # Compare results
        policy_diff = torch.abs(policy_direct - policy_server).max().item()
        value_diff = torch.abs(value_direct - value_server).item()

        print(f"\n3. Comparison:")
        print(f"   Max policy difference: {policy_diff:.6f}")
        print(f"   Value difference: {value_diff:.6f}")

        # Check if results match (within floating point tolerance)
        tolerance = 1e-5
        if policy_diff < tolerance and value_diff < tolerance:
            print(f"\n   SUCCESS: Results match within tolerance ({tolerance})")
            return True
        else:
            print(f"\n   FAILURE: Results differ more than tolerance ({tolerance})")
            return False

    finally:
        server.shutdown()


def test_performance():
    """
    Test GPU server performance with multiple workers.
    """
    print("\n" + "=" * 80)
    print("PERFORMANCE TEST: GPU Server with Multiple Workers")
    print("=" * 80)

    # Create network
    network = BlobNet(
        state_dim=256,
        action_dim=52,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
    )
    network.eval()
    network.to("cuda")

    encoder = StateEncoder()
    masker = ActionMasker()

    # Test configurations
    configs = [
        {"workers": 4, "games": 10, "name": "4 workers, 10 games (warmup)"},
        {"workers": 16, "games": 20, "name": "16 workers, 20 games"},
        {"workers": 32, "games": 30, "name": "32 workers, 30 games"},
    ]

    results = []

    for config in configs:
        print(f"\n{config['name']}:")
        print("-" * 80)

        # Create engine with GPU server
        engine = SelfPlayEngine(
            network=network,
            encoder=encoder,
            masker=masker,
            num_workers=config["workers"],
            num_determinizations=3,
            simulations_per_determinization=30,
            device="cuda",
            use_gpu_server=True,
            gpu_server_max_batch=512,
            gpu_server_timeout_ms=10.0,
        )

        try:
            # Generate games
            start_time = time.time()
            examples = engine.generate_games(
                num_games=config["games"],
                num_players=4,
                cards_to_deal=5,
            )
            elapsed_time = time.time() - start_time

            # Calculate metrics
            games_per_min = (config["games"] / elapsed_time) * 60
            num_examples = len(examples)

            print(f"  Completed: {config['games']} games in {elapsed_time:.1f}s")
            print(f"  Throughput: {games_per_min:.1f} games/min")
            print(f"  Examples generated: {num_examples}")

            # Get server statistics
            stats = engine.gpu_server.get_stats()
            print(f"  GPU Server Stats:")
            print(f"    Total requests: {stats.get('total_requests', 0)}")
            print(f"    Total batches: {stats.get('total_batches', 0)}")
            print(f"    Avg batch size: {stats.get('avg_batch_size', 0):.1f}")
            print(f"    Max batch size: {stats.get('max_batch_size', 0)}")

            results.append({
                "workers": config["workers"],
                "games": config["games"],
                "games_per_min": games_per_min,
                "avg_batch_size": stats.get("avg_batch_size", 0),
                "max_batch_size": stats.get("max_batch_size", 0),
            })

        finally:
            engine.shutdown()

    # Print summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Workers':<10} {'Games/min':<15} {'Avg Batch':<15} {'Max Batch':<15}")
    print("-" * 80)
    for result in results:
        print(
            f"{result['workers']:<10} {result['games_per_min']:<15.1f} "
            f"{result['avg_batch_size']:<15.1f} {result['max_batch_size']:<15}"
        )

    # Check if target met
    best_throughput = max(r["games_per_min"] for r in results)
    target_throughput = 500  # games/min

    print(f"\nBest throughput: {best_throughput:.1f} games/min")
    print(f"Target throughput: {target_throughput} games/min")

    if best_throughput >= target_throughput:
        print(f"\nSUCCESS: Achieved target throughput!")
    else:
        print(f"\nNOTE: Did not reach target (achieved {best_throughput/target_throughput*100:.1f}% of target)")

    return results


def test_baseline_comparison():
    """
    Compare GPU server against baseline (multiprocessing without GPU server).
    """
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON: GPU Server vs Multiprocessing")
    print("=" * 80)

    # Create network
    network = BlobNet(
        state_dim=256,
        action_dim=52,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
    )
    network.eval()
    network.to("cuda")

    encoder = StateEncoder()
    masker = ActionMasker()

    num_workers = 32
    num_games = 20

    # Test 1: Baseline (multiprocessing without batching)
    print(f"\n1. Baseline: {num_workers} workers, no batching")
    print("-" * 80)

    engine_baseline = SelfPlayEngine(
        network=network,
        encoder=encoder,
        masker=masker,
        num_workers=num_workers,
        num_determinizations=3,
        simulations_per_determinization=30,
        device="cuda",
        use_batched_evaluator=False,
        use_thread_pool=False,
        use_gpu_server=False,
    )

    start_time = time.time()
    examples_baseline = engine_baseline.generate_games(
        num_games=num_games,
        num_players=4,
        cards_to_deal=5,
    )
    baseline_time = time.time() - start_time
    baseline_throughput = (num_games / baseline_time) * 60

    print(f"  Completed: {num_games} games in {baseline_time:.1f}s")
    print(f"  Throughput: {baseline_throughput:.1f} games/min")

    engine_baseline.shutdown()

    # Test 2: GPU Server
    print(f"\n2. GPU Server: {num_workers} workers")
    print("-" * 80)

    engine_gpu = SelfPlayEngine(
        network=network,
        encoder=encoder,
        masker=masker,
        num_workers=num_workers,
        num_determinizations=3,
        simulations_per_determinization=30,
        device="cuda",
        use_gpu_server=True,
        gpu_server_max_batch=512,
        gpu_server_timeout_ms=10.0,
    )

    start_time = time.time()
    examples_gpu = engine_gpu.generate_games(
        num_games=num_games,
        num_players=4,
        cards_to_deal=5,
    )
    gpu_time = time.time() - start_time
    gpu_throughput = (num_games / gpu_time) * 60

    print(f"  Completed: {num_games} games in {gpu_time:.1f}s")
    print(f"  Throughput: {gpu_throughput:.1f} games/min")

    stats = engine_gpu.gpu_server.get_stats()
    print(f"  GPU Server Stats:")
    print(f"    Avg batch size: {stats.get('avg_batch_size', 0):.1f}")
    print(f"    Max batch size: {stats.get('max_batch_size', 0)}")

    engine_gpu.shutdown()

    # Summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Baseline throughput: {baseline_throughput:.1f} games/min")
    print(f"GPU server throughput: {gpu_throughput:.1f} games/min")
    speedup = gpu_throughput / baseline_throughput if baseline_throughput > 0 else 0
    print(f"Speedup: {speedup:.1f}x")

    if speedup >= 7:
        print(f"\nSUCCESS: Achieved {speedup:.1f}x speedup (target was 7-15x)")
    else:
        print(f"\nNOTE: Speedup of {speedup:.1f}x is below target range (7-15x)")

    return speedup


if __name__ == "__main__":
    print("GPU Inference Server Test Suite")
    print("=" * 80)
    print()

    # Run tests
    try:
        # Test 1: Correctness
        correctness_passed = test_correctness()

        if not correctness_passed:
            print("\nERROR: Correctness test failed. Aborting performance tests.")
            exit(1)

        # Test 2: Performance
        perf_results = test_performance()

        # Test 3: Baseline comparison
        speedup = test_baseline_comparison()

        # Final summary
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"Correctness: {'PASS' if correctness_passed else 'FAIL'}")
        print(f"Speedup vs baseline: {speedup:.1f}x")

        best_throughput = max(r["games_per_min"] for r in perf_results)
        print(f"Best throughput: {best_throughput:.1f} games/min")

        if correctness_passed and speedup >= 7 and best_throughput >= 500:
            print("\nALL TESTS PASSED")
        else:
            print("\nSOME TESTS DID NOT MEET TARGETS (but may still be acceptable)")

    except Exception as e:
        print(f"\n\nERROR: Test suite failed with exception:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
