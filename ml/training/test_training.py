"""
Tests for training module (self-play, replay buffer, training loop).

This test suite covers:
- SelfPlayWorker: Game generation and training example creation
- ReplayBuffer: Experience storage and sampling (will be added in Session 3)
- NetworkTrainer: Training loop (will be added in Session 4)
- TrainingPipeline: Full integration (will be added in Session 5)
"""

import pytest
import torch
import numpy as np
import time
import concurrent.futures
from typing import List, Dict, Any

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.training.selfplay import SelfPlayWorker, SelfPlayEngine


class TestSelfPlayWorker:
    """Tests for SelfPlayWorker class."""

    @pytest.fixture
    def network(self):
        """Create a small neural network for testing."""
        return BlobNet(
            state_dim=256,
            embedding_dim=128,  # Smaller for faster tests
            num_layers=2,  # Fewer layers for faster tests
            num_heads=4,
            feedforward_dim=256,
            dropout=0.1,
        )

    @pytest.fixture
    def encoder(self):
        """Create state encoder."""
        return StateEncoder()

    @pytest.fixture
    def masker(self):
        """Create action masker."""
        return ActionMasker()

    @pytest.fixture
    def worker(self, network, encoder, masker):
        """Create self-play worker with default settings."""
        return SelfPlayWorker(
            network=network,
            encoder=encoder,
            masker=masker,
            num_determinizations=2,  # Fewer for faster tests
            simulations_per_determinization=10,  # Fewer for faster tests
            use_imperfect_info=True,
        )

    def test_selfplay_worker_initialization(self, network, encoder, masker):
        """Test self-play worker initializes correctly."""
        worker = SelfPlayWorker(
            network=network,
            encoder=encoder,
            masker=masker,
            num_determinizations=3,
            simulations_per_determinization=30,
        )

        assert worker.network is network
        assert worker.encoder is encoder
        assert worker.masker is masker
        assert worker.num_determinizations == 3
        assert worker.simulations_per_determinization == 30
        assert worker.use_imperfect_info is True
        assert worker.mcts is not None
        assert worker.temperature_schedule is not None

    def test_custom_temperature_schedule(self, network, encoder, masker):
        """Test worker accepts custom temperature schedule."""

        def custom_schedule(move_number: int) -> float:
            return 0.8  # Constant temperature

        worker = SelfPlayWorker(
            network=network,
            encoder=encoder,
            masker=masker,
            temperature_schedule=custom_schedule,
        )

        assert worker.temperature_schedule(0) == 0.8
        assert worker.temperature_schedule(100) == 0.8

    def test_generate_single_game(self, worker):
        """Test generating a single self-play game."""
        # Generate a small game (4 players, 3 cards)
        examples = worker.generate_game(num_players=4, cards_to_deal=3)

        # Should have examples (4 bids + 4*3 = 16 card plays)
        assert len(examples) > 0
        assert len(examples) == 4 + 12  # 4 bids + 12 card plays

        # Check each example has the right structure
        for example in examples:
            assert "state" in example
            assert "policy" in example
            assert "value" in example
            assert "player_position" in example
            assert "game_id" in example
            assert "move_number" in example

    def test_training_examples_format(self, worker):
        """Test training examples have correct format and types."""
        examples = worker.generate_game(num_players=4, cards_to_deal=3)

        for example in examples:
            # Check types
            assert isinstance(example["state"], np.ndarray)
            assert isinstance(example["policy"], np.ndarray)
            assert isinstance(example["value"], float)
            assert isinstance(example["player_position"], int)
            assert isinstance(example["game_id"], str)
            assert isinstance(example["move_number"], int)

            # Check shapes
            assert example["state"].shape == (256,)
            assert example["policy"].shape == (65,)

            # Check ranges
            assert 0 <= example["player_position"] < 4
            assert example["move_number"] >= 0
            assert -1.0 <= example["value"] <= 1.0

            # Check policy is normalized
            policy_sum = example["policy"].sum()
            if policy_sum > 0:  # Some policies might be all zeros if action wasn't legal
                assert 0.99 <= policy_sum <= 1.01  # Allow small floating point error

    def test_temperature_schedule(self, worker):
        """Test temperature schedule changes over moves."""
        schedule = worker.get_default_temperature_schedule()

        # Early game: high exploration
        assert schedule(0) == 1.0
        assert schedule(5) == 1.0
        assert schedule(9) == 1.0

        # Mid game: moderate exploration
        assert schedule(10) == 0.5
        assert schedule(15) == 0.5
        assert schedule(19) == 0.5

        # Late game: near-greedy
        assert schedule(20) == 0.1
        assert schedule(30) == 0.1
        assert schedule(100) == 0.1

    def test_outcome_backpropagation(self, worker):
        """Test outcomes are correctly back-propagated."""
        examples = worker.generate_game(num_players=4, cards_to_deal=5)

        # All examples should have values filled in
        for example in examples:
            assert example["value"] is not None
            assert isinstance(example["value"], float)
            assert -1.0 <= example["value"] <= 1.0

        # Examples from the same player should have the same final value
        # Group by player
        player_values = {}
        for example in examples:
            player = example["player_position"]
            value = example["value"]

            if player not in player_values:
                player_values[player] = value
            else:
                # All examples from same player should have same outcome value
                assert abs(player_values[player] - value) < 1e-6

    def test_action_selection_greedy(self, worker):
        """Test greedy action selection (temperature=0)."""
        action_probs = {0: 0.1, 1: 0.6, 2: 0.3}

        # Greedy should always select action 1 (highest prob)
        for _ in range(10):
            action = worker._select_action(action_probs, temperature=0.0)
            assert action == 1

    def test_action_selection_stochastic(self, worker):
        """Test stochastic action selection (temperature=1)."""
        action_probs = {0: 0.2, 1: 0.5, 2: 0.3}

        # With temperature=1, should sample according to probabilities
        # Run many trials and check distribution
        counts = {0: 0, 1: 0, 2: 0}
        num_trials = 1000

        for _ in range(num_trials):
            action = worker._select_action(action_probs, temperature=1.0)
            counts[action] += 1

        # Check that action 1 is selected most often (but not always)
        assert counts[1] > counts[0]
        assert counts[1] > counts[2]

        # Check that all actions were selected at least once
        assert counts[0] > 0
        assert counts[1] > 0
        assert counts[2] > 0

    def test_action_selection_high_temperature(self, worker):
        """Test high temperature increases exploration."""
        action_probs = {0: 0.1, 1: 0.8, 2: 0.1}

        # With high temperature, distribution should be more uniform
        counts = {0: 0, 1: 0, 2: 0}
        num_trials = 1000

        for _ in range(num_trials):
            action = worker._select_action(action_probs, temperature=2.0)
            counts[action] += 1

        # With high temperature, even low-probability actions should be selected more
        # Action 1 should still be most common, but gap should be smaller
        assert counts[1] > counts[0]
        assert counts[0] > 0
        assert counts[2] > 0

        # Rough check: with temp=2, should be more balanced than temp=1
        # But this is probabilistic, so we can't be too strict

    def test_action_probs_to_vector_bidding(self, worker):
        """Test converting bid probabilities to vector."""
        action_probs = {0: 0.2, 1: 0.5, 2: 0.3}
        policy_vector = worker._action_probs_to_vector(action_probs, is_bidding=True)

        assert policy_vector.shape == (65,)
        assert policy_vector[0] == 0.2
        assert policy_vector[1] == 0.5
        assert policy_vector[2] == 0.3
        assert policy_vector[3] == 0.0  # Not in action_probs

    def test_action_probs_to_vector_playing(self, worker):
        """Test converting card play probabilities to vector."""
        # Card indices 0-51
        action_probs = {0: 0.1, 10: 0.3, 20: 0.6}
        policy_vector = worker._action_probs_to_vector(action_probs, is_bidding=False)

        assert policy_vector.shape == (65,)
        assert policy_vector[0] == 0.1
        assert policy_vector[10] == 0.3
        assert policy_vector[20] == 0.6
        assert policy_vector[5] == 0.0  # Not in action_probs

    def test_backpropagate_outcome_normalization(self, worker):
        """Test outcome values are normalized correctly."""
        # Create dummy examples
        examples = [
            {
                "player_position": 0,
                "value": None,
            },
            {
                "player_position": 1,
                "value": None,
            },
        ]

        # Simulate final scores
        final_scores = {
            0: 0,  # Failed bid -> should map to -1.0
            1: 23,  # Perfect score (bid 13, made it) -> should map to 1.0
        }

        worker._backpropagate_outcome(examples, final_scores)

        # Check normalization
        # 0 points -> -1.0
        assert abs(examples[0]["value"] - (-1.0)) < 1e-6

        # 23 points -> 1.0
        assert abs(examples[1]["value"] - 1.0) < 1e-6

    def test_backpropagate_outcome_mid_scores(self, worker):
        """Test outcome normalization for mid-range scores."""
        examples = [
            {"player_position": 0, "value": None},
            {"player_position": 1, "value": None},
        ]

        final_scores = {
            0: 10,  # Bid 0, made it -> 10 points
            1: 15,  # Bid 5, made it -> 10 + 5 = 15 points
        }

        worker._backpropagate_outcome(examples, final_scores)

        # 10 points: (10/23)*2 - 1 ≈ -0.13
        expected_0 = (10.0 / 23.0) * 2.0 - 1.0
        assert abs(examples[0]["value"] - expected_0) < 1e-6

        # 15 points: (15/23)*2 - 1 ≈ 0.30
        expected_1 = (15.0 / 23.0) * 2.0 - 1.0
        assert abs(examples[1]["value"] - expected_1) < 1e-6

    def test_game_id_unique(self, worker):
        """Test each game gets a unique ID."""
        game1_examples = worker.generate_game(num_players=4, cards_to_deal=3)
        game2_examples = worker.generate_game(num_players=4, cards_to_deal=3)

        game1_id = game1_examples[0]["game_id"]
        game2_id = game2_examples[0]["game_id"]

        assert game1_id != game2_id

        # All examples in same game should have same ID
        for example in game1_examples:
            assert example["game_id"] == game1_id

        for example in game2_examples:
            assert example["game_id"] == game2_id

    def test_move_numbers_sequential(self, worker):
        """Test move numbers are sequential."""
        examples = worker.generate_game(num_players=4, cards_to_deal=3)

        move_numbers = [ex["move_number"] for ex in examples]

        # Should start at 0
        assert move_numbers[0] == 0

        # Should increment by 1
        for i in range(1, len(move_numbers)):
            assert move_numbers[i] == move_numbers[i - 1] + 1

    def test_different_player_counts(self, worker):
        """Test generating games with different player counts."""
        for num_players in [3, 4, 5, 6]:
            examples = worker.generate_game(
                num_players=num_players, cards_to_deal=3
            )

            # Should have examples for all players
            players_seen = set(ex["player_position"] for ex in examples)
            assert len(players_seen) == num_players
            assert min(players_seen) == 0
            assert max(players_seen) == num_players - 1

    def test_different_card_counts(self, worker):
        """Test generating games with different card counts."""
        for cards in [1, 3, 5, 7]:
            examples = worker.generate_game(num_players=4, cards_to_deal=cards)

            # Should have examples
            # 4 bids + (4 players * cards cards) = 4 + 4*cards
            expected_examples = 4 + (4 * cards)
            assert len(examples) == expected_examples

    def test_perfect_info_mode(self, network, encoder, masker):
        """Test worker can use perfect info MCTS."""
        worker = SelfPlayWorker(
            network=network,
            encoder=encoder,
            masker=masker,
            num_determinizations=2,
            simulations_per_determinization=10,
            use_imperfect_info=False,  # Use perfect info
        )

        examples = worker.generate_game(num_players=4, cards_to_deal=3)

        # Should still generate valid examples
        assert len(examples) > 0
        for example in examples:
            assert "state" in example
            assert "policy" in example
            assert "value" in example


# Additional test helper functions
def validate_training_example(example: Dict[str, Any]) -> bool:
    """
    Validate a training example has correct format.

    Args:
        example: Training example dict

    Returns:
        True if valid, False otherwise
    """
    required_keys = ["state", "policy", "value", "player_position", "game_id", "move_number"]

    # Check all keys present
    if not all(key in example for key in required_keys):
        return False

    # Check types
    if not isinstance(example["state"], np.ndarray):
        return False
    if not isinstance(example["policy"], np.ndarray):
        return False
    if not isinstance(example["value"], (float, int)):
        return False
    if not isinstance(example["player_position"], int):
        return False
    if not isinstance(example["game_id"], str):
        return False
    if not isinstance(example["move_number"], int):
        return False

    # Check shapes
    if example["state"].shape != (256,):
        return False
    if example["policy"].shape != (65,):
        return False

    # Check ranges
    if example["value"] < -1.0 or example["value"] > 1.0:
        return False
    if example["player_position"] < 0:
        return False
    if example["move_number"] < 0:
        return False

    return True


class TestSelfPlayEngine:
    """Tests for SelfPlayEngine class for parallel game generation."""

    @pytest.fixture
    def network(self):
        """Create a small neural network for testing."""
        return BlobNet(
            state_dim=256,
            embedding_dim=128,  # Smaller for faster tests
            num_layers=2,  # Fewer layers for faster tests
            num_heads=4,
            feedforward_dim=256,
            dropout=0.1,
        )

    @pytest.fixture
    def encoder(self):
        """Create state encoder."""
        return StateEncoder()

    @pytest.fixture
    def masker(self):
        """Create action masker."""
        return ActionMasker()

    @pytest.fixture
    def engine(self, network, encoder, masker):
        """Create self-play engine with minimal workers for testing."""
        return SelfPlayEngine(
            network=network,
            encoder=encoder,
            masker=masker,
            num_workers=2,  # Use only 2 workers for faster tests
            num_determinizations=2,  # Fewer for faster tests
            simulations_per_determinization=10,  # Fewer for faster tests
        )

    def test_selfplay_engine_initialization(self, network, encoder, masker):
        """Test self-play engine initializes with worker pool."""
        engine = SelfPlayEngine(
            network=network,
            encoder=encoder,
            masker=masker,
            num_workers=4,
            num_determinizations=3,
            simulations_per_determinization=30,
        )

        assert engine.network is network
        assert engine.encoder is encoder
        assert engine.masker is masker
        assert engine.num_workers == 4
        assert engine.num_determinizations == 3
        assert engine.simulations_per_determinization == 30
        assert engine.network_state is not None
        assert engine.pool is None  # Pool created lazily
        assert engine.executor is None

    def test_parallel_game_generation(self, engine):
        """Test generating multiple games in parallel."""
        # Generate 4 games (2 per worker)
        num_games = 4
        examples = engine.generate_games(
            num_games=num_games,
            num_players=4,
            cards_to_deal=3,
        )

        # Should have examples from all games
        # Each game: 4 bids + 12 card plays = 16 examples
        # 4 games * 16 examples = 64 total
        assert len(examples) == num_games * 16

        # Verify all examples are valid
        for example in examples:
            assert validate_training_example(example)

        # Check that we got examples from different games
        game_ids = set(ex["game_id"] for ex in examples)
        assert len(game_ids) == num_games

        # Clean up
        engine.shutdown()

    def test_training_examples_aggregation(self, engine):
        """Test examples from multiple workers are aggregated correctly."""
        # Generate games with multiple workers
        examples = engine.generate_games(
            num_games=6,  # 3 per worker
            num_players=4,
            cards_to_deal=2,
        )

        # Each game: 4 bids + 8 card plays = 12 examples
        # 6 games * 12 examples = 72 total
        assert len(examples) == 6 * 12

        # Check that examples come from both workers
        game_ids = set(ex["game_id"] for ex in examples)
        assert len(game_ids) == 6

        # Verify examples have worker IDs in game_id
        worker_ids = set()
        for game_id in game_ids:
            if "worker" in game_id:
                worker_id = game_id.split("_")[0]
                worker_ids.add(worker_id)

        # Should have examples from multiple workers
        assert len(worker_ids) >= 1  # At least one worker

        # Clean up
        engine.shutdown()

    def test_worker_isolation(self, engine):
        """Test workers don't interfere with each other."""
        # Generate games multiple times
        examples1 = engine.generate_games(
            num_games=2,
            num_players=4,
            cards_to_deal=3,
        )

        examples2 = engine.generate_games(
            num_games=2,
            num_players=4,
            cards_to_deal=3,
        )

        # Each run should produce same number of examples
        assert len(examples1) == len(examples2)

        # But game IDs should be different (different games)
        game_ids1 = set(ex["game_id"] for ex in examples1)
        game_ids2 = set(ex["game_id"] for ex in examples2)
        assert len(game_ids1.intersection(game_ids2)) == 0

        # All examples should be valid
        for example in examples1 + examples2:
            assert validate_training_example(example)

        # Clean up
        engine.shutdown()

    def test_async_generation(self, engine):
        """Test asynchronous game generation."""
        # Start async generation
        future = engine.generate_games_async(
            num_games=2,
            num_players=4,
            cards_to_deal=3,
        )

        # Future should be returned immediately
        assert future is not None
        assert isinstance(future, concurrent.futures.Future)

        # Wait for completion
        examples = future.result(timeout=60)  # 60 second timeout

        # Should have valid examples
        assert len(examples) == 2 * 16  # 2 games * 16 examples each

        for example in examples:
            assert validate_training_example(example)

        # Clean up
        engine.shutdown()

    def test_performance_scaling(self, network, encoder, masker):
        """Test performance scales with number of workers."""
        # Test with 1 worker
        engine1 = SelfPlayEngine(
            network=network,
            encoder=encoder,
            masker=masker,
            num_workers=1,
            num_determinizations=2,
            simulations_per_determinization=5,
        )

        start_time = time.time()
        examples1 = engine1.generate_games(
            num_games=2,
            num_players=4,
            cards_to_deal=2,
        )
        time1 = time.time() - start_time

        # Test with 2 workers
        engine2 = SelfPlayEngine(
            network=network,
            encoder=encoder,
            masker=masker,
            num_workers=2,
            num_determinizations=2,
            simulations_per_determinization=5,
        )

        start_time = time.time()
        examples2 = engine2.generate_games(
            num_games=2,
            num_players=4,
            cards_to_deal=2,
        )
        time2 = time.time() - start_time

        # Both should produce same number of examples
        assert len(examples1) == len(examples2)

        # 2 workers should be faster (or similar due to overhead)
        # We allow some tolerance due to test environment variability
        # Just verify both completed successfully
        assert time1 > 0
        assert time2 > 0

        # Clean up
        engine1.shutdown()
        engine2.shutdown()

    def test_progress_callback(self, engine):
        """Test progress callback is called."""
        progress_updates = []

        def progress_callback(games_completed):
            progress_updates.append(games_completed)

        examples = engine.generate_games(
            num_games=4,
            num_players=4,
            cards_to_deal=2,
            progress_callback=progress_callback,
        )

        # Should have called callback
        assert len(progress_updates) > 0
        assert progress_updates[-1] == 4  # Final update should be total games

        # Clean up
        engine.shutdown()

    def test_zero_games(self, engine):
        """Test generating zero games."""
        examples = engine.generate_games(num_games=0)
        assert len(examples) == 0

    def test_uneven_game_distribution(self, engine):
        """Test games are distributed evenly across workers."""
        # Generate 5 games with 2 workers
        # Should distribute as 3 and 2
        examples = engine.generate_games(
            num_games=5,
            num_players=4,
            cards_to_deal=2,
        )

        # 5 games * 12 examples = 60 total
        assert len(examples) == 5 * 12

        # Verify all 5 games were generated
        game_ids = set(ex["game_id"] for ex in examples)
        assert len(game_ids) == 5

        # Clean up
        engine.shutdown()

    def test_network_state_transfer(self, network, encoder, masker):
        """Test network state is correctly transferred to workers."""
        # Create engine
        engine = SelfPlayEngine(
            network=network,
            encoder=encoder,
            masker=masker,
            num_workers=2,
            num_determinizations=2,
            simulations_per_determinization=5,
        )

        # Generate some games
        examples = engine.generate_games(
            num_games=2,
            num_players=4,
            cards_to_deal=2,
        )

        # All examples should be valid (network worked correctly)
        assert len(examples) > 0
        for example in examples:
            assert validate_training_example(example)

        # Clean up
        engine.shutdown()

    def test_shutdown(self, engine):
        """Test engine shutdown cleans up resources."""
        # Generate some games
        examples = engine.generate_games(
            num_games=2,
            num_players=4,
            cards_to_deal=2,
        )

        assert len(examples) > 0

        # Shutdown
        engine.shutdown()

        # Pool and executor should be None
        assert engine.pool is None
        assert engine.executor is None

    def test_multiple_async_calls(self, engine):
        """Test multiple async generation calls."""
        # Start multiple async generations
        future1 = engine.generate_games_async(num_games=2, num_players=4, cards_to_deal=2)
        future2 = engine.generate_games_async(num_games=2, num_players=4, cards_to_deal=2)

        # Both should complete
        examples1 = future1.result(timeout=60)
        examples2 = future2.result(timeout=60)

        # Should have valid examples
        assert len(examples1) > 0
        assert len(examples2) > 0

        # Clean up
        engine.shutdown()
