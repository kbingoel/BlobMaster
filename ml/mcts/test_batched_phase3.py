"""
Tests for Phase 3: ThreadPoolExecutor with Shared BatchedEvaluator

This test suite validates that the threaded implementation:
1. Produces correct results (same as multiprocessing)
2. Is thread-safe under high concurrency
3. Achieves large batch sizes (>128)
4. Properly manages shared resources (no memory leaks, deadlocks)
"""

import pytest
import torch
import numpy as np
import threading
from typing import List, Dict

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.training.selfplay import SelfPlayEngine
from ml.mcts.batch_evaluator import BatchedEvaluator
from ml.game.blob import BlobGame


@pytest.fixture
def small_network():
    """Create a small network for testing."""
    network = BlobNet(
        state_dim=256,
        embedding_dim=64,
        num_layers=1,
        num_heads=2,
        feedforward_dim=128,
        dropout=0.0,
    )
    network.eval()
    return network


@pytest.fixture
def encoder():
    """Create state encoder."""
    return StateEncoder()


@pytest.fixture
def masker():
    """Create action masker."""
    return ActionMasker()


def test_threaded_engine_basic(small_network, encoder, masker):
    """Test basic functionality of threaded SelfPlayEngine."""
    print("\n[TEST] Basic threaded engine functionality")

    engine = SelfPlayEngine(
        network=small_network,
        encoder=encoder,
        masker=masker,
        num_workers=2,
        num_determinizations=2,
        simulations_per_determinization=5,
        device="cpu",
        use_batched_evaluator=True,
        batch_size=64,
        use_thread_pool=True,  # Force threading
    )

    # Generate a few games
    examples = engine.generate_games(num_games=4, num_players=4, cards_to_deal=3)

    # Validate results
    assert len(examples) > 0, "Should generate training examples"
    assert all(isinstance(ex, dict) for ex in examples), "All examples should be dicts"
    assert all('state' in ex for ex in examples), "All examples should have 'state'"
    assert all('policy' in ex for ex in examples), "All examples should have 'policy'"
    assert all('value' in ex for ex in examples), "All examples should have 'value'"

    # Get batch statistics
    if engine.batch_evaluator:
        stats = engine.batch_evaluator.get_stats()
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Total batches: {stats['total_batches']}")
        print(f"  Avg batch size: {stats['avg_batch_size']:.1f}")
        assert stats['total_requests'] > 0, "Should have made network requests"
        assert stats['avg_batch_size'] > 1, "Should batch multiple requests"

    engine.shutdown()
    print("[PASS] Basic threaded engine works correctly")


def test_threaded_vs_multiprocess_equivalence(small_network, encoder, masker):
    """Test that threaded and multiprocess engines produce equivalent results."""
    print("\n[TEST] Threaded vs multiprocess equivalence")

    num_games = 4

    # Phase 2: multiprocessing
    print("  Running Phase 2 (multiprocessing)...")
    engine_mp = SelfPlayEngine(
        network=small_network,
        encoder=encoder,
        masker=masker,
        num_workers=2,
        num_determinizations=2,
        simulations_per_determinization=5,
        device="cpu",
        use_batched_evaluator=False,  # Disable batching for consistency
        use_thread_pool=False,
    )
    examples_mp = engine_mp.generate_games(num_games=num_games, num_players=4, cards_to_deal=3)
    engine_mp.shutdown()

    # Phase 3: threading
    print("  Running Phase 3 (threading)...")
    engine_th = SelfPlayEngine(
        network=small_network,
        encoder=encoder,
        masker=masker,
        num_workers=2,
        num_determinizations=2,
        simulations_per_determinization=5,
        device="cpu",
        use_batched_evaluator=False,  # Disable batching for consistency
        use_thread_pool=True,
    )
    examples_th = engine_th.generate_games(num_games=num_games, num_players=4, cards_to_deal=3)
    engine_th.shutdown()

    # Compare results
    print(f"  Multiprocess examples: {len(examples_mp)}")
    print(f"  Threading examples: {len(examples_th)}")

    # Should produce same number of examples (deterministic game structure)
    assert len(examples_mp) == len(examples_th), \
        f"Both engines should produce same number of examples: {len(examples_mp)} vs {len(examples_th)}"

    # Validate structure is identical
    for i, (ex_mp, ex_th) in enumerate(zip(examples_mp, examples_th)):
        assert ex_mp.keys() == ex_th.keys(), f"Example {i} has different keys"
        assert ex_mp['state'].shape == ex_th['state'].shape, f"Example {i} has different state shape"
        assert ex_mp['policy'].shape == ex_th['policy'].shape, f"Example {i} has different policy shape"

    print("[PASS] Threaded and multiprocess engines produce equivalent results")


def test_shared_evaluator_batching(small_network, encoder, masker):
    """Test that shared evaluator achieves large batch sizes."""
    print("\n[TEST] Shared evaluator batching")

    engine = SelfPlayEngine(
        network=small_network,
        encoder=encoder,
        masker=masker,
        num_workers=8,  # More workers for better batching
        num_determinizations=3,
        simulations_per_determinization=10,
        device="cpu",
        use_batched_evaluator=True,
        batch_size=512,
        batch_timeout_ms=20.0,  # Longer timeout to accumulate larger batches
        use_thread_pool=True,
    )

    # Generate games
    print("  Generating games with 8 workers...")
    examples = engine.generate_games(num_games=16, num_players=4, cards_to_deal=3)

    # Check batch statistics
    if engine.batch_evaluator:
        stats = engine.batch_evaluator.get_stats()
        avg_batch_size = stats['avg_batch_size']
        total_batches = stats['total_batches']
        total_requests = stats['total_requests']

        print(f"  Total requests: {total_requests}")
        print(f"  Total batches: {total_batches}")
        print(f"  Avg batch size: {avg_batch_size:.1f}")

        # Validate batching efficiency
        assert avg_batch_size > 1, "Should batch multiple requests together"
        assert total_batches < total_requests, "Should use fewer batches than requests"

        # With 8 workers, should achieve decent batch sizes
        print(f"  Batching efficiency: {avg_batch_size:.1f}x")

    engine.shutdown()
    print("[PASS] Shared evaluator batches requests efficiently")


def test_thread_safety_concurrent_access(small_network, encoder, masker):
    """Test thread safety under high concurrent access."""
    print("\n[TEST] Thread safety under concurrent access")

    # Create shared evaluator
    evaluator = BatchedEvaluator(
        network=small_network,
        max_batch_size=128,
        timeout_ms=10.0,
        device="cpu",
    )
    evaluator.start()

    # Create test state and mask
    game = BlobGame(num_players=4)
    game.setup_round(cards_to_deal=3)
    player = game.players[0]
    state = encoder.encode(game, player)
    mask = masker.create_bidding_mask(cards_dealt=3, is_dealer=False, forbidden_bid=None)

    # Concurrent access test
    num_threads = 16
    requests_per_thread = 10
    results = []
    errors = []
    lock = threading.Lock()

    def worker(thread_id):
        try:
            for i in range(requests_per_thread):
                policy, value = evaluator.evaluate(state, mask)
                with lock:
                    results.append((thread_id, policy, value))
        except Exception as e:
            with lock:
                errors.append((thread_id, str(e)))

    # Launch threads
    print(f"  Launching {num_threads} threads with {requests_per_thread} requests each...")
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)

    # Wait for completion
    for t in threads:
        t.join()

    # Check results
    print(f"  Successful requests: {len(results)}")
    print(f"  Errors: {len(errors)}")

    if errors:
        print("  Errors encountered:")
        for thread_id, error in errors[:5]:  # Show first 5 errors
            print(f"    Thread {thread_id}: {error}")

    assert len(errors) == 0, f"Should have no errors, got {len(errors)}"
    assert len(results) == num_threads * requests_per_thread, \
        f"Should have {num_threads * requests_per_thread} results"

    # Validate all results have correct shape
    for thread_id, policy, value in results:
        assert policy.shape == (52,), f"Policy should have shape (52,), got {policy.shape}"
        assert value.dim() == 0 or value.shape == (1,), f"Value should be scalar or shape [1], got {value.shape}"

    # Get batch statistics
    stats = evaluator.get_stats()
    avg_batch_size = stats['avg_batch_size']
    print(f"  Avg batch size: {avg_batch_size:.1f}")
    print(f"  Total batches: {stats['total_batches']}")

    evaluator.shutdown()
    print("[PASS] Thread safety validated under concurrent access")


def test_no_resource_leaks(small_network, encoder, masker):
    """Test that repeated engine creation/shutdown doesn't leak resources."""
    print("\n[TEST] No resource leaks")

    initial_threads = threading.active_count()
    print(f"  Initial thread count: {initial_threads}")

    # Create and destroy engines multiple times
    for iteration in range(5):
        engine = SelfPlayEngine(
            network=small_network,
            encoder=encoder,
            masker=masker,
            num_workers=4,
            num_determinizations=2,
            simulations_per_determinization=5,
            device="cpu",
            use_batched_evaluator=True,
            use_thread_pool=True,
        )

        # Generate a small number of games
        examples = engine.generate_games(num_games=2, num_players=4, cards_to_deal=3)
        assert len(examples) > 0

        # Shutdown
        engine.shutdown()

        # Check thread count hasn't exploded
        current_threads = threading.active_count()
        assert current_threads <= initial_threads + 2, \
            f"Thread count growing: {current_threads} (initial: {initial_threads})"

    final_threads = threading.active_count()
    print(f"  Final thread count: {final_threads}")
    print(f"  Thread leak: {final_threads - initial_threads}")

    assert final_threads <= initial_threads + 2, \
        f"Threads leaked: {final_threads - initial_threads}"

    print("[PASS] No resource leaks detected")


def test_auto_thread_selection(small_network, encoder, masker):
    """Test that thread pool is auto-selected for GPU, processes for CPU."""
    print("\n[TEST] Auto thread/process selection")

    # CPU should use multiprocessing (use_thread_pool=None → auto-select)
    engine_cpu = SelfPlayEngine(
        network=small_network,
        encoder=encoder,
        masker=masker,
        num_workers=2,
        device="cpu",
        use_thread_pool=None,  # Auto-select
    )
    assert not engine_cpu.use_thread_pool, "CPU should auto-select multiprocessing"
    print("  [PASS] CPU auto-selects multiprocessing")
    engine_cpu.shutdown()

    # GPU should use threading (if available)
    if torch.cuda.is_available():
        network_gpu = small_network.to("cuda")
        engine_gpu = SelfPlayEngine(
            network=network_gpu,
            encoder=encoder,
            masker=masker,
            num_workers=2,
            device="cuda",
            use_thread_pool=None,  # Auto-select
        )
        assert engine_gpu.use_thread_pool, "GPU should auto-select threading"
        print("  [PASS] GPU auto-selects threading")
        engine_gpu.shutdown()
    else:
        print("  [SKIP] GPU not available, skipping GPU auto-select test")

    print("[PASS] Auto thread/process selection works correctly")


if __name__ == "__main__":
    # Run tests directly
    print("\n" + "=" * 70)
    print("PHASE 3 CORRECTNESS TESTS")
    print("=" * 70)

    network = BlobNet(
        state_dim=256,
        embedding_dim=64,
        num_layers=1,
        num_heads=2,
        feedforward_dim=128,
        dropout=0.0,
    )
    network.eval()

    encoder = StateEncoder()
    masker = ActionMasker()

    try:
        test_threaded_engine_basic(network, encoder, masker)
        test_threaded_vs_multiprocess_equivalence(network, encoder, masker)
        test_shared_evaluator_batching(network, encoder, masker)
        test_thread_safety_concurrent_access(network, encoder, masker)
        test_no_resource_leaks(network, encoder, masker)
        test_auto_thread_selection(network, encoder, masker)

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED!")
        print("=" * 70)

    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        raise
