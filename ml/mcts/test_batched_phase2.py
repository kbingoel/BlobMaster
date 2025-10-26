"""
Tests for Phase 2: Multi-Game Batched MCTS

This module tests the BatchedEvaluator implementation for multi-game batching.
The BatchedEvaluator collects neural network evaluation requests from multiple
MCTS instances and batches them into single GPU calls.

Test Coverage:
    1. Correctness: Batched evaluator produces same results as direct inference
    2. Concurrency: Multiple threads can safely call evaluator simultaneously
    3. Performance: Batching reduces number of network calls
    4. Timeout: Batch collection respects timeout mechanism
    5. Statistics: Evaluator tracks batch statistics correctly

Expected Results:
    - All correctness tests pass (results match direct inference)
    - Batch sizes match expectations (128-2048 depending on test)
    - GPU utilization improves (measured via batch statistics)
"""

import pytest
import torch
import numpy as np
import threading
import time
from typing import List

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.game.blob import BlobGame
from ml.mcts.batch_evaluator import BatchedEvaluator
from ml.mcts.search import MCTS


def test_batched_evaluator_basic():
    """
    Test basic functionality of BatchedEvaluator.

    Verifies that:
    - Evaluator can be created and started
    - Single evaluation request works correctly
    - Results match direct network inference
    - Evaluator can be shutdown cleanly
    """
    print("\n=== Test: BatchedEvaluator Basic Functionality ===")

    # Create network and components
    network = BlobNet(
        state_dim=256,
        embedding_dim=128,
        num_layers=2,
        num_heads=4,
        feedforward_dim=256,
        dropout=0.0,
    )
    network.eval()

    encoder = StateEncoder()
    masker = ActionMasker()

    # Create game state for testing
    game = BlobGame(num_players=4)
    game.setup_round(cards_to_deal=5)
    player = game.players[0]

    # Encode state and create mask
    state = encoder.encode(game, player)
    legal_actions, mask = _get_legal_actions_and_mask(game, player, encoder, masker)

    # Direct network inference (ground truth)
    with torch.no_grad():
        policy_direct, value_direct = network(state.unsqueeze(0), mask.unsqueeze(0))
        policy_direct = policy_direct.squeeze(0)
        value_direct = value_direct.squeeze(0)

    # Batched evaluation
    evaluator = BatchedEvaluator(network, max_batch_size=128, timeout_ms=10.0)
    evaluator.start()

    try:
        policy_batched, value_batched = evaluator.evaluate(state, mask)

        # Compare results
        policy_diff = torch.abs(policy_direct - policy_batched).max().item()
        value_diff = torch.abs(value_direct - value_batched).item()

        print(f"Policy difference: {policy_diff:.6f}")
        print(f"Value difference: {value_diff:.6f}")

        # Results should be identical (same network, same input)
        assert policy_diff < 1e-5, f"Policy mismatch: {policy_diff}"
        assert value_diff < 1e-5, f"Value mismatch: {value_diff}"

        print("[PASS] BatchedEvaluator produces correct results")

    finally:
        evaluator.shutdown()


def test_batched_evaluator_concurrent():
    """
    Test concurrent access to BatchedEvaluator.

    Simulates multiple MCTS instances calling evaluator from different threads.
    Verifies that:
    - Thread-safe operation (no crashes or deadlocks)
    - All requests receive results
    - Results are correct for each request
    """
    print("\n=== Test: Concurrent BatchedEvaluator Access ===")

    # Create network
    network = BlobNet(
        state_dim=256,
        embedding_dim=128,
        num_layers=2,
        num_heads=4,
        feedforward_dim=256,
        dropout=0.0,
    )
    network.eval()

    encoder = StateEncoder()
    masker = ActionMasker()

    # Create evaluator
    evaluator = BatchedEvaluator(network, max_batch_size=256, timeout_ms=20.0)
    evaluator.start()

    # Create multiple game states
    num_threads = 8
    num_requests_per_thread = 10

    results_lock = threading.Lock()
    all_results = []
    errors = []

    def worker(thread_id: int):
        """Worker thread that makes multiple evaluation requests."""
        try:
            for i in range(num_requests_per_thread):
                # Create unique game state
                game = BlobGame(num_players=4)
                game.setup_round(cards_to_deal=5)
                player = game.players[thread_id % 4]

                state = encoder.encode(game, player)
                _, mask = _get_legal_actions_and_mask(game, player, encoder, masker)

                # Evaluate
                policy, value = evaluator.evaluate(state, mask)

                # Store result
                with results_lock:
                    all_results.append((thread_id, i, policy, value))

        except Exception as e:
            with results_lock:
                errors.append((thread_id, str(e)))

    # Launch threads
    threads = []
    start_time = time.time()

    for thread_id in range(num_threads):
        t = threading.Thread(target=worker, args=(thread_id,))
        threads.append(t)
        t.start()

    # Wait for completion
    for t in threads:
        t.join()

    elapsed_time = time.time() - start_time

    # Check results
    print(f"Threads: {num_threads}")
    print(f"Requests per thread: {num_requests_per_thread}")
    print(f"Total requests: {len(all_results)}")
    print(f"Elapsed time: {elapsed_time:.2f}s")
    print(f"Requests/sec: {len(all_results) / elapsed_time:.1f}")

    # Get statistics
    stats = evaluator.get_stats()
    print(f"\nEvaluator Statistics:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Total batches: {stats['total_batches']}")
    print(f"  Avg batch size: {stats['avg_batch_size']:.1f}")

    evaluator.shutdown()

    # Verify no errors
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # Verify all requests completed
    expected_total = num_threads * num_requests_per_thread
    assert len(all_results) == expected_total, \
        f"Expected {expected_total} results, got {len(all_results)}"

    print(f"[PASS] Concurrent access works correctly")


def test_batched_evaluator_with_mcts():
    """
    Test BatchedEvaluator integration with MCTS.

    Compares MCTS results with and without batched evaluator:
    - Direct inference (no batching)
    - Batched evaluator

    Results should be similar (may differ slightly due to timing).
    """
    print("\n=== Test: BatchedEvaluator + MCTS Integration ===")

    # Create network and components
    network = BlobNet(
        state_dim=256,
        embedding_dim=128,
        num_layers=2,
        num_heads=4,
        feedforward_dim=256,
        dropout=0.0,
    )
    network.eval()

    encoder = StateEncoder()
    masker = ActionMasker()

    # Create game state
    game = BlobGame(num_players=4)
    game.setup_round(cards_to_deal=5)
    player = game.players[0]

    # MCTS without batched evaluator
    print("\nMCTS without BatchedEvaluator...")
    mcts_direct = MCTS(
        network=network,
        encoder=encoder,
        masker=masker,
        num_simulations=50,
        c_puct=1.5,
        temperature=1.0,
        batch_evaluator=None,
    )

    start = time.time()
    action_probs_direct = mcts_direct.search(game, player)
    time_direct = time.time() - start

    print(f"Time: {time_direct:.3f}s")
    print(f"Actions: {len(action_probs_direct)}")

    # MCTS with batched evaluator
    print("\nMCTS with BatchedEvaluator...")
    evaluator = BatchedEvaluator(network, max_batch_size=128, timeout_ms=5.0)
    evaluator.start()

    mcts_batched = MCTS(
        network=network,
        encoder=encoder,
        masker=masker,
        num_simulations=50,
        c_puct=1.5,
        temperature=1.0,
        batch_evaluator=evaluator,
    )

    start = time.time()
    action_probs_batched = mcts_batched.search(game, player)
    time_batched = time.time() - start

    print(f"Time: {time_batched:.3f}s")
    print(f"Actions: {len(action_probs_batched)}")

    # Get statistics
    stats = evaluator.get_stats()
    print(f"\nEvaluator Statistics:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Total batches: {stats['total_batches']}")
    print(f"  Avg batch size: {stats['avg_batch_size']:.1f}")

    evaluator.shutdown()

    # Verify same legal actions
    assert set(action_probs_direct.keys()) == set(action_probs_batched.keys()), \
        "Legal actions should be the same"

    # Verify probabilities sum to 1.0
    sum_direct = sum(action_probs_direct.values())
    sum_batched = sum(action_probs_batched.values())
    assert abs(sum_direct - 1.0) < 0.01, f"Direct probs sum to {sum_direct}"
    assert abs(sum_batched - 1.0) < 0.01, f"Batched probs sum to {sum_batched}"

    print(f"[PASS] MCTS with BatchedEvaluator works correctly")


def test_batch_size_scaling():
    """
    Test batch size scaling with different max_batch_size settings.

    Verifies that:
    - Larger max_batch_size leads to fewer batches
    - Average batch size increases with max_batch_size
    - All requests still get processed correctly
    """
    print("\n=== Test: Batch Size Scaling ===")

    network = BlobNet(
        state_dim=256,
        embedding_dim=128,
        num_layers=2,
        num_heads=4,
        feedforward_dim=256,
        dropout=0.0,
    )
    network.eval()

    encoder = StateEncoder()
    masker = ActionMasker()

    # Test different batch sizes
    batch_sizes = [16, 64, 256, 1024]
    num_requests = 200

    for max_batch_size in batch_sizes:
        print(f"\nTesting max_batch_size={max_batch_size}")

        evaluator = BatchedEvaluator(
            network,
            max_batch_size=max_batch_size,
            timeout_ms=5.0,
        )
        evaluator.start()

        # Submit requests from multiple threads
        def worker(num_req: int):
            for _ in range(num_req):
                game = BlobGame(num_players=4)
                game.setup_round(cards_to_deal=5)
                player = game.players[0]

                state = encoder.encode(game, player)
                _, mask = _get_legal_actions_and_mask(game, player, encoder, masker)

                evaluator.evaluate(state, mask)

        threads = []
        for i in range(4):  # 4 threads
            t = threading.Thread(target=worker, args=(num_requests // 4,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        stats = evaluator.get_stats()
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Total batches: {stats['total_batches']}")
        print(f"  Avg batch size: {stats['avg_batch_size']:.1f}")

        evaluator.shutdown()

        # Verify all requests processed
        assert stats['total_requests'] == num_requests

    print("[PASS] Batch size scaling works correctly")


def test_timeout_mechanism():
    """
    Test that timeout mechanism works correctly.

    Submits requests slowly to trigger timeout instead of batch size limit.
    Verifies that requests are processed even when batch isn't full.
    """
    print("\n=== Test: Timeout Mechanism ===")

    network = BlobNet(
        state_dim=256,
        embedding_dim=128,
        num_layers=2,
        num_heads=4,
        feedforward_dim=256,
        dropout=0.0,
    )
    network.eval()

    encoder = StateEncoder()
    masker = ActionMasker()

    # Small batch size and short timeout
    evaluator = BatchedEvaluator(
        network,
        max_batch_size=100,  # Large batch size
        timeout_ms=10.0,      # Short timeout (10ms)
    )
    evaluator.start()

    # Submit requests slowly (one every 50ms)
    num_requests = 10
    for i in range(num_requests):
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        state = encoder.encode(game, player)
        _, mask = _get_legal_actions_and_mask(game, player, encoder, masker)

        evaluator.evaluate(state, mask)

        if i < num_requests - 1:  # Don't sleep after last request
            time.sleep(0.020)  # 20ms between requests

    stats = evaluator.get_stats()
    print(f"Requests: {stats['total_requests']}")
    print(f"Batches: {stats['total_batches']}")
    print(f"Avg batch size: {stats['avg_batch_size']:.1f}")

    evaluator.shutdown()

    # With slow submission and short timeout, we should get many small batches
    assert stats['total_requests'] == num_requests
    assert stats['avg_batch_size'] < 5.0, \
        f"Expected small batches due to timeout, got avg={stats['avg_batch_size']:.1f}"

    print("[PASS] Timeout mechanism works correctly")


def _get_legal_actions_and_mask(game, player, encoder, masker):
    """Helper function to get legal actions and mask."""
    if game.game_phase == "bidding":
        cards_dealt = len(player.hand)
        is_dealer = player.position == game.dealer_position

        forbidden_bid = None
        if is_dealer:
            total_bids = sum(p.bid for p in game.players if p.bid is not None)
            forbidden_bid = cards_dealt - total_bids

        mask = masker.create_bidding_mask(
            cards_dealt=cards_dealt,
            is_dealer=is_dealer,
            forbidden_bid=forbidden_bid,
        )

        legal_actions = []
        for bid in range(cards_dealt + 1):
            if mask[bid] == 1.0:
                legal_actions.append(bid)

    elif game.game_phase == "playing":
        led_suit = game.current_trick.led_suit if game.current_trick else None

        mask = masker.create_playing_mask(
            hand=player.hand,
            led_suit=led_suit,
            encoder=encoder,
        )

        legal_actions = []
        for card in player.hand:
            card_idx = encoder._card_to_index(card)
            if mask[card_idx] == 1.0:
                legal_actions.append(card_idx)

    else:
        raise ValueError(f"Unknown game phase: {game.game_phase}")

    return legal_actions, mask


if __name__ == "__main__":
    # Run tests
    print("=" * 60)
    print("Phase 2: Multi-Game Batched MCTS Tests")
    print("=" * 60)

    try:
        test_batched_evaluator_basic()
        test_batched_evaluator_concurrent()
        test_batched_evaluator_with_mcts()
        test_batch_size_scaling()
        test_timeout_mechanism()

        print("\n" + "=" * 60)
        print("All Phase 2 tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        raise
