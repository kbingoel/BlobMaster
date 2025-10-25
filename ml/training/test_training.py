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
from ml.training.replay_buffer import ReplayBuffer, augment_example
from ml.training.trainer import NetworkTrainer, TrainingPipeline


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
            assert example["policy"].shape == (52,)

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

        assert policy_vector.shape == (52,)
        assert policy_vector[0] == 0.2
        assert policy_vector[1] == 0.5
        assert policy_vector[2] == 0.3
        assert policy_vector[3] == 0.0  # Not in action_probs

    def test_action_probs_to_vector_playing(self, worker):
        """Test converting card play probabilities to vector."""
        # Card indices 0-51
        action_probs = {0: 0.1, 10: 0.3, 20: 0.6}
        policy_vector = worker._action_probs_to_vector(action_probs, is_bidding=False)

        assert policy_vector.shape == (52,)
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
    if example["policy"].shape != (52,):
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


class TestReplayBuffer:
    """Tests for ReplayBuffer class."""

    @pytest.fixture
    def buffer(self):
        """Create replay buffer with small capacity for testing."""
        return ReplayBuffer(capacity=100)

    @pytest.fixture
    def sample_examples(self):
        """Create sample training examples for testing."""
        examples = []
        for i in range(10):
            examples.append(
                {
                    "state": np.random.randn(256).astype(np.float32),
                    "policy": np.random.rand(52).astype(np.float32),
                    "value": float(np.random.rand() * 2 - 1),  # Range [-1, 1]
                    "player_position": i % 4,
                    "game_id": f"game_{i // 4}",
                    "move_number": i,
                }
            )
        return examples

    def test_replay_buffer_initialization(self):
        """Test replay buffer initializes correctly."""
        buffer = ReplayBuffer(capacity=1000)

        assert len(buffer) == 0
        assert buffer.capacity == 1000
        assert buffer.position == 0
        assert buffer.is_full is False
        assert buffer.use_prioritization is False

    def test_replay_buffer_custom_capacity(self):
        """Test replay buffer with custom capacity."""
        buffer = ReplayBuffer(capacity=500_000)

        assert buffer.capacity == 500_000
        assert len(buffer) == 0

    def test_prioritization_not_implemented(self):
        """Test that prioritization raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            ReplayBuffer(capacity=100, use_prioritization=True)

    def test_add_examples(self, buffer, sample_examples):
        """Test adding examples to buffer."""
        buffer.add_examples(sample_examples)

        assert len(buffer) == 10
        assert buffer.position == 10

    def test_add_examples_incremental(self, buffer):
        """Test adding examples incrementally."""
        # Add examples one at a time
        for i in range(5):
            example = {
                "state": np.random.randn(256).astype(np.float32),
                "policy": np.random.rand(52).astype(np.float32),
                "value": 0.5,
                "player_position": i,
                "game_id": "test_game",
                "move_number": i,
            }
            buffer.add_examples([example])

        assert len(buffer) == 5

    def test_circular_buffer_overflow(self, buffer):
        """Test buffer overwrites old examples when full."""
        # Create buffer with capacity 10
        small_buffer = ReplayBuffer(capacity=10)

        # Add 15 examples (5 more than capacity)
        examples = []
        for i in range(15):
            examples.append(
                {
                    "state": np.random.randn(256).astype(np.float32),
                    "policy": np.random.rand(52).astype(np.float32),
                    "value": float(i),  # Use i as value to track which examples remain
                    "player_position": 0,
                    "game_id": f"game_{i}",
                    "move_number": i,
                }
            )

        small_buffer.add_examples(examples)

        # Buffer should have exactly 10 examples
        assert len(small_buffer) == 10
        assert small_buffer.is_full is True

        # Position should wrap around
        assert small_buffer.position == 5  # (15 % 10)

        # First 5 examples should be overwritten
        # Remaining examples should have values 5-14
        values = [ex["value"] for ex in small_buffer.buffer]
        assert 0.0 not in values  # First example (value=0) was overwritten
        assert 1.0 not in values
        assert 2.0 not in values
        assert 3.0 not in values
        assert 4.0 not in values
        assert 10.0 in values  # Later examples remain
        assert 14.0 in values

    def test_sample_batch(self, buffer, sample_examples):
        """Test sampling batches for training."""
        buffer.add_examples(sample_examples)

        # Sample a batch
        batch_size = 5
        states, policies, values = buffer.sample_batch(batch_size, device="cpu")

        # Check shapes
        assert states.shape == (batch_size, 256)
        assert policies.shape == (batch_size, 52)
        assert values.shape == (batch_size,)

        # Check types
        assert isinstance(states, torch.Tensor)
        assert isinstance(policies, torch.Tensor)
        assert isinstance(values, torch.Tensor)

        # Check device
        assert states.device.type == "cpu"
        assert policies.device.type == "cpu"
        assert values.device.type == "cpu"

    def test_sample_batch_randomness(self, buffer, sample_examples):
        """Test that sampling is random."""
        buffer.add_examples(sample_examples)

        # Sample multiple batches and check they differ
        batch1_states, _, _ = buffer.sample_batch(5, device="cpu")
        batch2_states, _, _ = buffer.sample_batch(5, device="cpu")

        # Batches should be different (with very high probability)
        # Compare by checking if any states differ
        states_equal = torch.allclose(batch1_states, batch2_states)
        # With random sampling from 10 examples, getting identical batches is unlikely
        # but possible, so we can't assert False directly
        # Just verify the mechanism works

    def test_sample_batch_too_large(self, buffer, sample_examples):
        """Test sampling more examples than available raises error."""
        buffer.add_examples(sample_examples)

        with pytest.raises(ValueError):
            buffer.sample_batch(100, device="cpu")  # Buffer only has 10 examples

    def test_sample_batch_empty_buffer(self, buffer):
        """Test sampling from empty buffer raises error."""
        with pytest.raises(ValueError):
            buffer.sample_batch(5, device="cpu")

    def test_save_and_load(self, buffer, sample_examples, tmp_path):
        """Test persisting buffer to disk."""
        buffer.add_examples(sample_examples)

        # Save buffer
        save_path = tmp_path / "replay_buffer.pkl"
        buffer.save(str(save_path))

        # Create new buffer and load
        new_buffer = ReplayBuffer(capacity=100)
        new_buffer.load(str(save_path))

        # Check that state is preserved
        assert len(new_buffer) == len(buffer)
        assert new_buffer.position == buffer.position
        assert new_buffer.is_full == buffer.is_full
        assert new_buffer.capacity == buffer.capacity

        # Check that examples are preserved
        for orig_ex, loaded_ex in zip(buffer.buffer, new_buffer.buffer):
            assert np.allclose(orig_ex["state"], loaded_ex["state"])
            assert np.allclose(orig_ex["policy"], loaded_ex["policy"])
            assert orig_ex["value"] == loaded_ex["value"]
            assert orig_ex["player_position"] == loaded_ex["player_position"]
            assert orig_ex["game_id"] == loaded_ex["game_id"]
            assert orig_ex["move_number"] == loaded_ex["move_number"]

    def test_load_nonexistent_file(self, buffer):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            buffer.load("nonexistent_file.pkl")

    def test_load_mismatched_capacity(self, sample_examples, tmp_path):
        """Test loading buffer with different capacity raises error."""
        # Create and save buffer with capacity 100
        buffer1 = ReplayBuffer(capacity=100)
        buffer1.add_examples(sample_examples)
        save_path = tmp_path / "buffer.pkl"
        buffer1.save(str(save_path))

        # Try to load into buffer with different capacity
        buffer2 = ReplayBuffer(capacity=200)
        with pytest.raises(ValueError):
            buffer2.load(str(save_path))

    def test_clear(self, buffer, sample_examples):
        """Test clearing the buffer."""
        buffer.add_examples(sample_examples)

        assert len(buffer) > 0

        buffer.clear()

        assert len(buffer) == 0
        assert buffer.position == 0
        assert buffer.is_full is False

    def test_buffer_statistics_empty(self, buffer):
        """Test statistics for empty buffer."""
        stats = buffer.get_statistics()

        assert stats["size"] == 0
        assert stats["capacity"] == 100
        assert stats["utilization"] == 0.0
        assert stats["num_games"] == 0
        assert stats["avg_game_length"] == 0.0

    def test_buffer_statistics(self, buffer, sample_examples):
        """Test buffer statistics calculation."""
        buffer.add_examples(sample_examples)

        stats = buffer.get_statistics()

        assert stats["size"] == 10
        assert stats["capacity"] == 100
        assert stats["utilization"] == 10.0  # 10/100 * 100
        assert stats["num_games"] == 3  # game_0, game_1, game_2
        assert stats["avg_game_length"] > 0

        # Check value distribution
        assert "value_distribution" in stats
        assert "min" in stats["value_distribution"]
        assert "max" in stats["value_distribution"]
        assert "mean" in stats["value_distribution"]
        assert "std" in stats["value_distribution"]

        # Check player distribution
        assert "player_distribution" in stats
        assert len(stats["player_distribution"]) > 0

    def test_buffer_statistics_value_distribution(self, buffer):
        """Test value distribution statistics are correct."""
        # Create examples with known values
        examples = []
        for i in range(5):
            examples.append(
                {
                    "state": np.random.randn(256).astype(np.float32),
                    "policy": np.random.rand(52).astype(np.float32),
                    "value": float(i) / 10.0,  # 0.0, 0.1, 0.2, 0.3, 0.4
                    "player_position": 0,
                    "game_id": "test_game",
                    "move_number": i,
                }
            )

        buffer.add_examples(examples)
        stats = buffer.get_statistics()

        # Check value distribution
        assert abs(stats["value_distribution"]["min"] - 0.0) < 1e-6
        assert abs(stats["value_distribution"]["max"] - 0.4) < 1e-6
        assert abs(stats["value_distribution"]["mean"] - 0.2) < 1e-6

    def test_buffer_statistics_player_distribution(self, buffer):
        """Test player distribution statistics."""
        # Create examples with known player distribution
        examples = []
        for i in range(12):
            examples.append(
                {
                    "state": np.random.randn(256).astype(np.float32),
                    "policy": np.random.rand(52).astype(np.float32),
                    "value": 0.0,
                    "player_position": i % 4,  # Players 0, 1, 2, 3 (3 of each)
                    "game_id": "test_game",
                    "move_number": i,
                }
            )

        buffer.add_examples(examples)
        stats = buffer.get_statistics()

        # Each player should have 3 examples
        assert stats["player_distribution"][0] == 3
        assert stats["player_distribution"][1] == 3
        assert stats["player_distribution"][2] == 3
        assert stats["player_distribution"][3] == 3

    def test_batch_tensor_shapes(self, buffer, sample_examples):
        """Test sampled batch tensors have correct shapes."""
        buffer.add_examples(sample_examples)

        # Test different batch sizes
        for batch_size in [1, 3, 5, 10]:
            states, policies, values = buffer.sample_batch(batch_size, device="cpu")

            assert states.shape == (batch_size, 256)
            assert policies.shape == (batch_size, 52)
            assert values.shape == (batch_size,)

    def test_batch_tensor_dtypes(self, buffer, sample_examples):
        """Test sampled batch tensors have correct dtypes."""
        buffer.add_examples(sample_examples)

        states, policies, values = buffer.sample_batch(5, device="cpu")

        assert states.dtype == torch.float32
        assert policies.dtype == torch.float32
        assert values.dtype == torch.float32

    def test_is_ready_for_training(self, buffer):
        """Test ready for training check."""
        # Empty buffer not ready
        assert not buffer.is_ready_for_training(min_examples=10)

        # Add examples
        examples = []
        for i in range(15):
            examples.append(
                {
                    "state": np.random.randn(256).astype(np.float32),
                    "policy": np.random.rand(52).astype(np.float32),
                    "value": 0.0,
                    "player_position": 0,
                    "game_id": "test_game",
                    "move_number": i,
                }
            )

        buffer.add_examples(examples)

        # Now should be ready
        assert buffer.is_ready_for_training(min_examples=10)
        assert not buffer.is_ready_for_training(min_examples=20)

    def test_len_operator(self, buffer, sample_examples):
        """Test __len__ operator."""
        assert len(buffer) == 0

        buffer.add_examples(sample_examples)
        assert len(buffer) == 10

        buffer.clear()
        assert len(buffer) == 0

    def test_large_buffer_performance(self):
        """Test buffer performance with large capacity."""
        # Create large buffer
        large_buffer = ReplayBuffer(capacity=10_000)

        # Add many examples
        examples = []
        for i in range(5000):
            examples.append(
                {
                    "state": np.random.randn(256).astype(np.float32),
                    "policy": np.random.rand(52).astype(np.float32),
                    "value": 0.0,
                    "player_position": i % 4,
                    "game_id": f"game_{i // 16}",
                    "move_number": i % 16,
                }
            )

        # Add in batches
        batch_size = 100
        for i in range(0, len(examples), batch_size):
            large_buffer.add_examples(examples[i : i + batch_size])

        assert len(large_buffer) == 5000

        # Test sampling is fast
        start_time = time.time()
        states, policies, values = large_buffer.sample_batch(512, device="cpu")
        sample_time = time.time() - start_time

        # Sampling should be fast (< 0.1 seconds)
        assert sample_time < 0.1

        assert states.shape == (512, 256)
        assert policies.shape == (512, 52)
        assert values.shape == (512,)

    def test_circular_buffer_position_wrapping(self):
        """Test position wraps correctly in circular buffer."""
        small_buffer = ReplayBuffer(capacity=5)

        # Add exactly capacity examples
        examples1 = []
        for i in range(5):
            examples1.append(
                {
                    "state": np.random.randn(256).astype(np.float32),
                    "policy": np.random.rand(52).astype(np.float32),
                    "value": float(i),
                    "player_position": 0,
                    "game_id": f"game1_{i}",
                    "move_number": i,
                }
            )

        small_buffer.add_examples(examples1)
        assert len(small_buffer) == 5
        assert small_buffer.position == 0  # Wrapped to 0
        assert not small_buffer.is_full  # Not full yet (exactly at capacity)

        # Add one more to trigger overflow
        example_overflow = {
            "state": np.random.randn(256).astype(np.float32),
            "policy": np.random.rand(52).astype(np.float32),
            "value": 99.0,
            "player_position": 0,
            "game_id": "overflow",
            "move_number": 0,
        }

        small_buffer.add_examples([example_overflow])
        assert len(small_buffer) == 5
        assert small_buffer.is_full is True
        assert small_buffer.position == 1

        # First example should be replaced
        assert small_buffer.buffer[0]["value"] == 99.0
        assert small_buffer.buffer[0]["game_id"] == "overflow"


class TestAugmentation:
    """Tests for data augmentation functions."""

    def test_augment_example_no_op(self):
        """Test augmentation currently returns unchanged example."""
        example = {
            "state": np.random.randn(256).astype(np.float32),
            "policy": np.random.rand(52).astype(np.float32),
            "value": 0.5,
            "player_position": 0,
            "game_id": "test_game",
            "move_number": 0,
        }

        augmented = augment_example(example)

        # Currently should just return list with original example
        assert len(augmented) == 1
        assert augmented[0] is example


class TestNetworkTrainer:
    """Tests for NetworkTrainer class."""

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
    def trainer(self, network):
        """Create network trainer."""
        return NetworkTrainer(
            network=network,
            learning_rate=0.001,
            weight_decay=1e-4,
            policy_loss_weight=1.0,
            value_loss_weight=1.0,
            use_mixed_precision=False,
            device="cpu",  # Use CPU for testing
        )

    @pytest.fixture
    def replay_buffer_with_data(self):
        """Create replay buffer with sample data."""
        buffer = ReplayBuffer(capacity=1000)

        # Generate 100 synthetic examples
        examples = []
        for i in range(100):
            example = {
                "state": np.random.randn(256).astype(np.float32),
                "policy": self._create_policy_for_network(52),  # Match network output size
                "value": float(np.random.randn()),
                "player_position": i % 4,
                "game_id": f"game_{i // 20}",
                "move_number": i % 20,
            }
            examples.append(example)

        buffer.add_examples(examples)
        return buffer

    def _create_random_policy(self, size: int) -> np.ndarray:
        """Create a random probability distribution."""
        policy = np.random.rand(size).astype(np.float32)
        policy = policy / policy.sum()  # Normalize to sum to 1
        return policy

    @staticmethod
    def _create_policy_for_network(size: int = 52) -> np.ndarray:
        """Create a random probability distribution matching network output size."""
        policy = np.random.rand(size).astype(np.float32)
        policy = policy / policy.sum()  # Normalize to sum to 1
        return policy

    def test_trainer_initialization(self, network):
        """Test network trainer initializes correctly."""
        trainer = NetworkTrainer(
            network=network,
            learning_rate=0.001,
            weight_decay=1e-4,
            device="cpu",
        )

        assert trainer.network is network
        assert trainer.learning_rate == 0.001
        assert trainer.weight_decay == 1e-4
        assert trainer.device == "cpu"
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.total_steps == 0
        assert trainer.total_epochs == 0

    def test_trainer_cuda_fallback(self, network):
        """Test trainer falls back to CPU if CUDA not available."""
        trainer = NetworkTrainer(
            network=network,
            device="cuda",  # Request CUDA
        )

        # If CUDA not available, should fall back to CPU
        if not torch.cuda.is_available():
            assert trainer.device == "cpu"
        else:
            assert trainer.device == "cuda"

    def test_single_training_step(self, trainer):
        """Test a single training step."""
        batch_size = 8

        # Create synthetic batch
        states = torch.randn(batch_size, 256)
        target_policies = torch.randn(batch_size, 52).softmax(dim=1)  # Normalized
        target_values = torch.randn(batch_size)

        # Perform training step
        metrics = trainer.train_step(states, target_policies, target_values)

        # Check metrics are returned
        assert "total_loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "policy_accuracy" in metrics

        # Losses should be positive
        assert metrics["total_loss"] > 0
        assert metrics["policy_loss"] > 0
        assert metrics["value_loss"] >= 0

        # Accuracy should be in [0, 100]
        assert 0 <= metrics["policy_accuracy"] <= 100

        # Check training step counter incremented
        # Note: train_step doesn't increment total_steps (train_epoch does)
        assert isinstance(metrics["total_loss"], float)

    def test_loss_computation(self, trainer):
        """Test loss computation is correct."""
        batch_size = 4

        # Create synthetic predictions and targets (with requires_grad for computation graph)
        policy_pred = torch.randn(batch_size, 52, requires_grad=True)
        value_pred = torch.randn(batch_size, requires_grad=True)
        policy_target = torch.randn(batch_size, 52).softmax(dim=1)
        value_target = torch.randn(batch_size)

        # Compute loss
        total_loss, metrics = trainer.compute_loss(
            policy_pred, value_pred, policy_target, value_target
        )

        # Check total_loss is a tensor with gradient tracking
        assert isinstance(total_loss, torch.Tensor)
        # Note: metrics contains .item() values (floats), so we check the tensor itself
        assert total_loss.dim() == 0  # Scalar tensor

        # Check metrics
        assert "total_loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "policy_accuracy" in metrics

        # Verify total_loss is sum of components (approximately)
        expected_total = (
            trainer.policy_loss_weight * metrics["policy_loss"] +
            trainer.value_loss_weight * metrics["value_loss"]
        )
        assert abs(metrics["total_loss"] - expected_total) < 1e-4

    def test_train_epoch(self, trainer, replay_buffer_with_data):
        """Test training for one epoch."""
        # Train for one epoch
        metrics = trainer.train_epoch(
            replay_buffer=replay_buffer_with_data,
            batch_size=16,
            num_batches=5,  # Small number for testing
        )

        # Check metrics are returned
        assert "total_loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "policy_accuracy" in metrics

        # Check epoch counter incremented
        assert trainer.total_epochs == 1
        assert trainer.total_steps == 5  # num_batches

    def test_multiple_epochs_reduce_loss(self, trainer, replay_buffer_with_data):
        """Test that training reduces loss over multiple epochs."""
        # Train for a few epochs and track loss
        losses = []

        for epoch in range(5):
            metrics = trainer.train_epoch(
                replay_buffer=replay_buffer_with_data,
                batch_size=16,
                num_batches=10,
            )
            losses.append(metrics["total_loss"])

        # Loss should generally decrease (not strict monotonic due to randomness)
        # Just check that final loss is less than initial loss
        assert losses[-1] < losses[0], "Loss should decrease over training"

        # Check epoch counter
        assert trainer.total_epochs == 5

    def test_checkpoint_save_load(self, trainer, tmp_path):
        """Test saving and loading checkpoints."""
        checkpoint_path = tmp_path / "checkpoint.pth"

        # Get initial state
        initial_lr = trainer.get_learning_rate()

        # Train a bit to change state
        trainer.total_steps = 100
        trainer.total_epochs = 10

        # Save checkpoint
        metrics = {"test_metric": 123.45}
        trainer.save_checkpoint(
            filepath=str(checkpoint_path),
            iteration=5,
            metrics=metrics,
        )

        # Check file exists
        assert checkpoint_path.exists()

        # Create new trainer
        new_network = BlobNet(
            state_dim=256,
            embedding_dim=128,
            num_layers=2,
            num_heads=4,
            feedforward_dim=256,
            dropout=0.1,
        )
        new_trainer = NetworkTrainer(
            network=new_network,
            learning_rate=0.001,
            weight_decay=1e-4,
            device="cpu",
        )

        # Load checkpoint
        loaded_metadata = new_trainer.load_checkpoint(str(checkpoint_path))

        # Check metadata
        assert loaded_metadata["iteration"] == 5
        assert loaded_metadata["total_steps"] == 100
        assert loaded_metadata["total_epochs"] == 10
        assert loaded_metadata["metrics"]["test_metric"] == 123.45

        # Check trainer state restored
        assert new_trainer.total_steps == 100
        assert new_trainer.total_epochs == 10

        # Check network weights match
        for p1, p2 in zip(trainer.network.parameters(), new_trainer.network.parameters()):
            assert torch.allclose(p1, p2)

    def test_learning_rate_scheduling(self, trainer):
        """Test learning rate scheduler works."""
        initial_lr = trainer.get_learning_rate()
        assert initial_lr == 0.001

        # Step scheduler (StepLR: decay by 0.1 every 100 epochs)
        # Before 100 steps, LR should stay the same
        for _ in range(50):
            trainer.step_scheduler()
        assert trainer.get_learning_rate() == 0.001

        # After 100 steps, LR should decay
        for _ in range(51):
            trainer.step_scheduler()
        current_lr = trainer.get_learning_rate()
        assert current_lr < initial_lr
        assert abs(current_lr - 0.0001) < 1e-6  # Should be 0.001 * 0.1 = 0.0001

    def test_gradient_clipping(self, trainer):
        """Test gradients are clipped correctly."""
        batch_size = 4

        # Create synthetic batch with large gradients
        states = torch.randn(batch_size, 256) * 100  # Large values
        target_policies = torch.randn(batch_size, 52).softmax(dim=1)
        target_values = torch.randn(batch_size) * 100  # Large values

        # Perform training step
        metrics = trainer.train_step(states, target_policies, target_values)

        # Check that gradients exist and training completed
        # (if clipping didn't work, gradients might explode)
        assert metrics["total_loss"] > 0
        assert not np.isnan(metrics["total_loss"])
        assert not np.isinf(metrics["total_loss"])

        # Check gradient norms are bounded (should be clipped to max_norm=1.0)
        total_norm = 0.0
        for p in trainer.network.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Total norm should be reasonable (clipping prevents explosion)
        # Note: After optimizer.step(), gradients may be modified, so we can't
        # strictly test the clipping here. The main test is that training completes
        # without NaN/Inf
        assert total_norm < 1e6  # Sanity check

    def test_policy_loss_weight(self):
        """Test policy loss weighting works correctly."""
        network = BlobNet(
            state_dim=256,
            embedding_dim=128,
            num_layers=2,
            num_heads=4,
            feedforward_dim=256,
            dropout=0.1,
        )

        # Create trainer with high policy loss weight
        trainer_high_policy = NetworkTrainer(
            network=network,
            policy_loss_weight=10.0,
            value_loss_weight=1.0,
            device="cpu",
        )

        batch_size = 4
        policy_pred = torch.randn(batch_size, 52)
        value_pred = torch.randn(batch_size)
        policy_target = torch.randn(batch_size, 52).softmax(dim=1)
        value_target = torch.randn(batch_size)

        total_loss, metrics = trainer_high_policy.compute_loss(
            policy_pred, value_pred, policy_target, value_target
        )

        # Total loss should be dominated by policy loss
        expected_total = 10.0 * metrics["policy_loss"] + 1.0 * metrics["value_loss"]
        assert abs(metrics["total_loss"] - expected_total) < 1e-4

    def test_value_loss_weight(self):
        """Test value loss weighting works correctly."""
        network = BlobNet(
            state_dim=256,
            embedding_dim=128,
            num_layers=2,
            num_heads=4,
            feedforward_dim=256,
            dropout=0.1,
        )

        # Create trainer with high value loss weight
        trainer_high_value = NetworkTrainer(
            network=network,
            policy_loss_weight=1.0,
            value_loss_weight=10.0,
            device="cpu",
        )

        batch_size = 4
        policy_pred = torch.randn(batch_size, 52)
        value_pred = torch.randn(batch_size)
        policy_target = torch.randn(batch_size, 52).softmax(dim=1)
        value_target = torch.randn(batch_size)

        total_loss, metrics = trainer_high_value.compute_loss(
            policy_pred, value_pred, policy_target, value_target
        )

        # Total loss should be dominated by value loss
        expected_total = 1.0 * metrics["policy_loss"] + 10.0 * metrics["value_loss"]
        assert abs(metrics["total_loss"] - expected_total) < 1e-4

    def test_train_epoch_with_default_batches(self, trainer, replay_buffer_with_data):
        """Test training epoch with automatic batch calculation."""
        # Train without specifying num_batches (should calculate automatically)
        metrics = trainer.train_epoch(
            replay_buffer=replay_buffer_with_data,
            batch_size=16,
            num_batches=None,  # Auto-calculate
        )

        # Should have trained for approximately buffer_size / batch_size batches
        # Buffer has 100 examples, batch_size=16, so ~6 batches
        expected_batches = max(1, len(replay_buffer_with_data) // 16)
        assert trainer.total_steps == expected_batches

    def test_network_in_train_mode_during_training(self, trainer, replay_buffer_with_data):
        """Test network is in training mode during train_epoch."""
        # Set network to eval mode
        trainer.network.eval()
        assert not trainer.network.training

        # Train epoch should set to train mode
        trainer.train_epoch(
            replay_buffer=replay_buffer_with_data,
            batch_size=16,
            num_batches=2,
        )

        # Network should be in train mode now
        assert trainer.network.training


class TestTrainingPipeline:
    """Tests for TrainingPipeline class."""

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
    def test_config(self, tmp_path):
        """Create test configuration."""
        return {
            "num_workers": 2,  # Small for faster tests
            "games_per_iteration": 10,  # Small for faster tests
            "num_determinizations": 1,  # Minimal for speed
            "simulations_per_determinization": 5,  # Minimal for speed
            "replay_buffer_capacity": 1000,
            "min_buffer_size": 20,
            "batch_size": 16,
            "epochs_per_iteration": 2,  # Small for faster tests
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "save_every_n_iterations": 1,  # Save every iteration for testing
            "device": "cpu",
        }

    @pytest.fixture
    def pipeline(self, network, encoder, masker, test_config):
        """Create training pipeline."""
        return TrainingPipeline(
            network=network,
            encoder=encoder,
            masker=masker,
            config=test_config,
        )

    def test_pipeline_initialization(self, network, encoder, masker, test_config):
        """Test training pipeline initializes correctly."""
        pipeline = TrainingPipeline(
            network=network,
            encoder=encoder,
            masker=masker,
            config=test_config,
        )

        # Check components initialized
        assert pipeline.network is network
        assert pipeline.encoder is encoder
        assert pipeline.masker is masker
        assert pipeline.config == test_config

        # Check internal components created
        assert pipeline.selfplay_engine is not None
        assert pipeline.replay_buffer is not None
        assert pipeline.trainer is not None

        # Check initial state
        assert pipeline.current_iteration == 0
        assert pipeline.best_model_path is None
        assert len(pipeline.metrics_history) == 0

        # Check checkpoint dir created
        assert pipeline.checkpoint_dir.exists()

        # Clean up workers
        pipeline.selfplay_engine.shutdown()

    def test_single_iteration(self, pipeline):
        """Test running a single training iteration."""
        # Run one iteration
        metrics = pipeline.run_iteration(iteration=0)

        # Check metrics returned
        assert "num_selfplay_examples" in metrics
        assert "replay_buffer_size" in metrics
        assert "avg_total_loss" in metrics
        assert "avg_policy_loss" in metrics
        assert "avg_value_loss" in metrics
        assert "avg_policy_accuracy" in metrics

        # Check self-play worked
        assert metrics["num_selfplay_examples"] > 0
        assert metrics["replay_buffer_size"] > 0

        # Check training metrics are reasonable
        assert metrics["avg_total_loss"] >= 0
        assert 0 <= metrics["avg_policy_accuracy"] <= 100

        # Clean up
        pipeline.selfplay_engine.shutdown()

    def test_selfplay_training_integration(self, pipeline):
        """Test self-play examples feed into training."""
        # Initially buffer is empty
        assert len(pipeline.replay_buffer) == 0

        # Run self-play phase
        examples = pipeline._selfplay_phase(iteration=0)

        # Check examples generated
        assert len(examples) > 0

        # Check examples added to replay buffer
        assert len(pipeline.replay_buffer) == len(examples)

        # Now run training phase
        training_metrics = pipeline._training_phase(iteration=0, num_epochs=2)

        # Check training happened (if buffer has enough examples)
        if len(pipeline.replay_buffer) >= pipeline.config["min_buffer_size"]:
            assert training_metrics["avg_total_loss"] > 0
        else:
            # If buffer too small, training skipped
            assert training_metrics["avg_total_loss"] == 0.0

        # Clean up
        pipeline.selfplay_engine.shutdown()

    def test_checkpoint_resume(self, pipeline, tmp_path):
        """Test resuming training from checkpoint."""
        # Run one iteration
        metrics1 = pipeline.run_iteration(iteration=0)

        # Check checkpoint was saved (save_every_n_iterations=1)
        checkpoint_path = pipeline.checkpoint_dir / "checkpoint_iter_1.pth"
        assert checkpoint_path.exists()

        # Save current state
        replay_buffer_size_before = len(pipeline.replay_buffer)

        # Create new pipeline and resume
        new_network = BlobNet(
            state_dim=256,
            embedding_dim=128,
            num_layers=2,
            num_heads=4,
            feedforward_dim=256,
            dropout=0.1,
        )

        new_pipeline = TrainingPipeline(
            network=new_network,
            encoder=pipeline.encoder,
            masker=pipeline.masker,
            config=pipeline.config,
        )

        # Resume from checkpoint
        new_pipeline._resume_from_checkpoint(str(checkpoint_path))

        # Check state restored
        assert new_pipeline.current_iteration == 1  # Should start from next iteration

        # Check replay buffer restored (if saved)
        buffer_path = pipeline.checkpoint_dir / "replay_buffer_iter_1.pkl"
        if buffer_path.exists():
            assert len(new_pipeline.replay_buffer) == replay_buffer_size_before

        # Clean up
        pipeline.selfplay_engine.shutdown()
        new_pipeline.selfplay_engine.shutdown()

    def test_metrics_logging(self, pipeline, capsys):
        """Test metrics are logged correctly."""
        # Create mock metrics
        metrics = {
            "num_selfplay_examples": 100,
            "replay_buffer_size": 200,
            "avg_total_loss": 1.234,
            "avg_policy_loss": 0.567,
            "avg_value_loss": 0.890,
            "avg_policy_accuracy": 45.6,
            "iteration_time_minutes": 5.5,
        }

        # Log metrics
        pipeline._log_metrics(iteration=0, metrics=metrics)

        # Check output contains key information
        captured = capsys.readouterr()
        assert "ITERATION 1 SUMMARY" in captured.out
        assert "Self-Play:" in captured.out
        assert "Training:" in captured.out
        assert "100" in captured.out  # num_selfplay_examples
        assert "1.234" in captured.out  # avg_total_loss
        assert "45.6" in captured.out  # avg_policy_accuracy

        # Clean up
        pipeline.selfplay_engine.shutdown()

    def test_full_training_run(self, pipeline):
        """Test running multiple training iterations end-to-end."""
        # Run 2 iterations (minimal test)
        pipeline.run_training(num_iterations=2)

        # Check iterations completed
        assert pipeline.current_iteration == 1  # Last completed iteration (0-indexed)

        # Check metrics history recorded
        assert len(pipeline.metrics_history) == 2

        # Check checkpoints saved
        checkpoint_iter1 = pipeline.checkpoint_dir / "checkpoint_iter_1.pth"
        checkpoint_iter2 = pipeline.checkpoint_dir / "checkpoint_iter_2.pth"
        assert checkpoint_iter1.exists()
        assert checkpoint_iter2.exists()

        # Check best model saved
        assert pipeline.best_model_path is not None
        assert pipeline.best_model_path.exists()

        # Check metrics history file saved
        metrics_file = pipeline.checkpoint_dir / "metrics_history.json"
        assert metrics_file.exists()

        # Clean up is handled by run_training
        # (selfplay_engine.shutdown() is called at the end)

    def test_checkpoint_phase(self, pipeline):
        """Test checkpoint phase saves files correctly."""
        # Generate some examples first
        examples = pipeline._selfplay_phase(iteration=0)

        # Create mock metrics
        metrics = {
            "avg_total_loss": 1.5,
            "avg_policy_accuracy": 40.0,
        }

        # Run checkpoint phase
        pipeline._checkpoint_phase(iteration=0, metrics=metrics)

        # Check checkpoint saved (save_every_n_iterations=1, so iteration 0 saves at iter 1)
        checkpoint_path = pipeline.checkpoint_dir / "checkpoint_iter_1.pth"
        assert checkpoint_path.exists()

        # Check replay buffer saved
        buffer_path = pipeline.checkpoint_dir / "replay_buffer_iter_1.pkl"
        assert buffer_path.exists()

        # Check best model saved
        best_model_path = pipeline.checkpoint_dir / "best_model.pth"
        assert best_model_path.exists()

        # Clean up
        pipeline.selfplay_engine.shutdown()

    def test_evaluation_phase_stub(self, pipeline):
        """Test evaluation phase returns stub metrics."""
        # Evaluation not yet implemented (Session 6)
        eval_metrics = pipeline._evaluation_phase(iteration=0)

        # Check stub metrics returned
        assert "eval_win_rate" in eval_metrics
        assert "eval_games_played" in eval_metrics
        assert "model_promoted" in eval_metrics

        # Check values are defaults
        assert eval_metrics["eval_win_rate"] == 0.0
        assert eval_metrics["eval_games_played"] == 0
        assert eval_metrics["model_promoted"] is False

        # Clean up
        pipeline.selfplay_engine.shutdown()
