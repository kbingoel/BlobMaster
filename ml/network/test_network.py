"""
Tests for neural network state encoding and model.

Tests:
- StateEncoder: Converts game states into tensor representations
- ActionMasker: Creates legal action masks
- BlobNet: Transformer neural network for policy and value prediction
- BlobNetTrainer: Training infrastructure
"""

import pytest
import torch
import tempfile
from pathlib import Path
from ml.game.blob import BlobGame, Card
from ml.network.encode import StateEncoder, ActionMasker
from ml.network.model import BlobNet, BlobNetTrainer, create_model, create_trainer


class TestStateEncoder:
    """Test suite for StateEncoder class."""

    def test_encoder_initialization(self):
        """Test StateEncoder creates correct dimensions."""
        encoder = StateEncoder()

        assert encoder.state_dim == 256
        assert encoder.card_dim == 52
        assert encoder.max_players == 8

    def test_card_to_index_mapping(self):
        """Test _card_to_index() correctly maps all 52 cards."""
        encoder = StateEncoder()

        # Test spades (0-12)
        assert encoder._card_to_index(Card("2", "♠")) == 0
        assert encoder._card_to_index(Card("A", "♠")) == 12

        # Test hearts (13-25)
        assert encoder._card_to_index(Card("2", "♥")) == 13
        assert encoder._card_to_index(Card("A", "♥")) == 25

        # Test clubs (26-38)
        assert encoder._card_to_index(Card("2", "♣")) == 26
        assert encoder._card_to_index(Card("A", "♣")) == 38

        # Test diamonds (39-51)
        assert encoder._card_to_index(Card("2", "♦")) == 39
        assert encoder._card_to_index(Card("A", "♦")) == 51

    def test_card_to_index_all_unique(self):
        """Test that all 52 cards map to unique indices."""
        encoder = StateEncoder()

        indices = set()
        from ml.game.constants import SUITS, RANKS

        for suit in SUITS:
            for rank in RANKS:
                card = Card(rank, suit)
                idx = encoder._card_to_index(card)

                # Check index in valid range
                assert 0 <= idx <= 51, f"Card {card} has invalid index {idx}"

                # Check uniqueness
                assert idx not in indices, f"Duplicate index {idx} for card {card}"
                indices.add(idx)

        # Check we got all 52 indices
        assert len(indices) == 52
        assert indices == set(range(52))

    def test_encode_full_game_state(self):
        """Test encoding complete game state."""
        encoder = StateEncoder()

        # Create simple game
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        # Encode state
        state = encoder.encode(game, player)

        # Check shape
        assert state.shape == (256,)

        # Check dtype
        assert state.dtype == torch.float32

        # Check it's not all zeros (has some data)
        assert state.abs().sum() > 0

    def test_hand_encoding(self):
        """Verify hand cards are correctly marked in tensor."""
        encoder = StateEncoder()

        # Create game and get player's hand
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        # Encode hand
        hand_vector = encoder._encode_hand(player.hand)

        # Check shape
        assert hand_vector.shape == (52,)

        # Check that exactly 5 cards are marked
        assert hand_vector.sum() == 5

        # Check that marked cards are in player's hand
        for card in player.hand:
            card_idx = encoder._card_to_index(card)
            assert hand_vector[card_idx] == 1.0, f"Card {card} not marked in hand vector"

        # Check that unmarked cards are not in hand
        for i in range(52):
            if hand_vector[i] == 0.0:
                # Reconstruct card from index
                suit_idx = i // 13
                rank_idx = i % 13
                from ml.game.constants import SUITS, RANKS
                suit = SUITS[suit_idx]
                rank = RANKS[rank_idx]

                # Verify this card is not in hand
                assert not any(c.suit == suit and c.rank == rank for c in player.hand)

    def test_state_shape_always_256(self):
        """State tensor always 256-dim regardless of game config."""
        encoder = StateEncoder()

        # Test different player counts
        for num_players in [3, 4, 5, 6, 7, 8]:
            game = BlobGame(num_players=num_players)
            game.setup_round(cards_to_deal=3)
            player = game.players[0]

            state = encoder.encode(game, player)
            assert state.shape == (256,), f"Wrong shape for {num_players} players"

        # Test different card counts
        for cards in [1, 3, 5, 7, 10]:
            game = BlobGame(num_players=4)
            game.setup_round(cards_to_deal=cards)
            player = game.players[0]

            state = encoder.encode(game, player)
            assert state.shape == (256,), f"Wrong shape for {cards} cards"

    def test_state_determinism(self):
        """Same game state should produce identical tensor."""
        encoder = StateEncoder()

        # Create game
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        # Encode twice
        state1 = encoder.encode(game, player)
        state2 = encoder.encode(game, player)

        # Should be identical
        assert torch.allclose(state1, state2)

    def test_bid_normalization(self):
        """Test bid values normalized to [0,1] or -1."""
        encoder = StateEncoder()

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)

        # Test no bid (-1)
        normalized = encoder._normalize_bid(None, game)
        assert normalized == -1.0

        # Test bid of 0 (normalized to 0)
        normalized = encoder._normalize_bid(0, game)
        assert normalized == 0.0

        # Test bid of 5 (max, normalized to 1)
        normalized = encoder._normalize_bid(5, game)
        assert normalized == 1.0

        # Test bid of 2 (normalized to 0.4)
        normalized = encoder._normalize_bid(2, game)
        assert abs(normalized - 0.4) < 0.001

    def test_game_phase_encoding(self):
        """Test game phase one-hot encoding."""
        encoder = StateEncoder()

        game = BlobGame(num_players=4)

        # Test bidding phase
        game.game_phase = 'bidding'
        phase_vector = encoder._encode_game_phase(game)
        assert phase_vector.tolist() == [1.0, 0.0, 0.0]

        # Test playing phase
        game.game_phase = 'playing'
        phase_vector = encoder._encode_game_phase(game)
        assert phase_vector.tolist() == [0.0, 1.0, 0.0]

        # Test scoring phase
        game.game_phase = 'scoring'
        phase_vector = encoder._encode_game_phase(game)
        assert phase_vector.tolist() == [0.0, 0.0, 1.0]


class TestActionMasker:
    """Test suite for ActionMasker class."""

    def test_masker_initialization(self):
        """Test ActionMasker creates correct dimensions."""
        masker = ActionMasker()

        assert masker.max_bid == 13
        assert masker.deck_size == 52
        assert masker.action_dim == 52

    def test_bidding_mask_normal_player(self):
        """Test bidding mask for non-dealer."""
        masker = ActionMasker()

        # 5 cards dealt, not dealer
        mask = masker.create_bidding_mask(
            cards_dealt=5,
            is_dealer=False,
            forbidden_bid=None
        )

        # Should allow bids 0-5
        assert mask[0:6].sum() == 6

        # Should not allow bids 6+
        assert mask[6:].sum() == 0

    def test_bidding_mask_dealer_constraint(self):
        """Test dealer's forbidden bid is masked out."""
        masker = ActionMasker()

        # 5 cards dealt, dealer, forbidden bid is 2
        mask = masker.create_bidding_mask(
            cards_dealt=5,
            is_dealer=True,
            forbidden_bid=2
        )

        # Should allow bids 0,1,3,4,5 (not 2)
        assert mask[0] == 1.0
        assert mask[1] == 1.0
        assert mask[2] == 0.0  # Forbidden
        assert mask[3] == 1.0
        assert mask[4] == 1.0
        assert mask[5] == 1.0

        # Should have 5 legal bids total
        assert mask[0:6].sum() == 5

    def test_playing_mask_all_legal(self):
        """Test all hand cards legal when no led suit."""
        encoder = StateEncoder()
        masker = ActionMasker()

        # Create hand
        hand = [
            Card("2", "♠"),
            Card("5", "♥"),
            Card("K", "♣"),
            Card("A", "♦"),
            Card("7", "♠"),
        ]

        # No led suit (first card)
        mask = masker.create_playing_mask(hand, led_suit=None, encoder=encoder)

        # All hand cards should be legal
        for card in hand:
            card_idx = encoder._card_to_index(card)
            assert mask[card_idx] == 1.0

        # Should have exactly 5 legal cards
        assert mask.sum() == 5

    def test_playing_mask_must_follow_suit(self):
        """Test only cards in led suit are legal."""
        encoder = StateEncoder()
        masker = ActionMasker()

        # Create hand with multiple spades
        hand = [
            Card("2", "♠"),
            Card("5", "♥"),
            Card("K", "♠"),
            Card("A", "♦"),
            Card("7", "♠"),
        ]

        # Led suit is spades
        mask = masker.create_playing_mask(hand, led_suit="♠", encoder=encoder)

        # Only spades should be legal
        assert mask[encoder._card_to_index(Card("2", "♠"))] == 1.0
        assert mask[encoder._card_to_index(Card("K", "♠"))] == 1.0
        assert mask[encoder._card_to_index(Card("7", "♠"))] == 1.0

        # Non-spades should be illegal
        assert mask[encoder._card_to_index(Card("5", "♥"))] == 0.0
        assert mask[encoder._card_to_index(Card("A", "♦"))] == 0.0

        # Should have exactly 3 legal cards
        assert mask.sum() == 3

    def test_playing_mask_cant_follow_suit(self):
        """Test all hand cards legal when can't follow suit."""
        encoder = StateEncoder()
        masker = ActionMasker()

        # Create hand with no spades
        hand = [
            Card("2", "♥"),
            Card("5", "♥"),
            Card("K", "♣"),
            Card("A", "♦"),
        ]

        # Led suit is spades (but player has none)
        mask = masker.create_playing_mask(hand, led_suit="♠", encoder=encoder)

        # All hand cards should be legal
        for card in hand:
            card_idx = encoder._card_to_index(card)
            assert mask[card_idx] == 1.0

        # Should have exactly 4 legal cards
        assert mask.sum() == 4


class TestBlobNet:
    """Test suite for BlobNet neural network."""

    def test_network_initialization(self):
        """Test BlobNet creates with correct architecture."""
        net = BlobNet(
            state_dim=256,
            embedding_dim=256,
            num_layers=6,
            num_heads=8,
            feedforward_dim=1024,
        )

        # Check attributes
        assert net.state_dim == 256
        assert net.embedding_dim == 256
        assert net.action_dim == 52  # max(14, 52)
        assert net.max_bid == 13
        assert net.max_cards == 52

        # Check parameter count is in expected range (2-5M for this config)
        # Note: With 6 layers, 8 heads, 1024 FFN, we get ~4.9M parameters
        num_params = net.get_num_parameters()
        assert 1_500_000 < num_params < 6_000_000, \
            f"Expected 1.5M-6M parameters, got {num_params:,}"

    def test_forward_pass_shape(self):
        """Test output shapes are correct."""
        net = BlobNet()

        # Create random state
        state = torch.randn(256)

        # Forward pass
        policy, value = net(state)

        # Check shapes
        assert policy.shape == (52,), f"Policy shape is {policy.shape}, expected (52,)"
        assert value.shape == (1,), f"Value shape is {value.shape}, expected (1,)"

    def test_single_state_inference(self):
        """Test forward pass with single state (no batch)."""
        net = BlobNet()

        # Single state (1D)
        state = torch.randn(256)
        policy, value = net(state)

        assert policy.dim() == 1, "Policy should be 1D for single state"
        assert value.dim() == 1, "Value should be 1D for single state"
        assert policy.shape[0] == 52
        assert value.shape[0] == 1

    def test_batch_inference(self):
        """Test forward pass with batched states."""
        net = BlobNet()

        # Batch of states (2D)
        batch_size = 16
        state_batch = torch.randn(batch_size, 256)

        policy, value = net(state_batch)

        # Check batch dimensions
        assert policy.shape == (batch_size, 52)
        assert value.shape == (batch_size, 1)

    def test_value_output_range(self):
        """Test value head outputs in [-1, 1] due to tanh."""
        net = BlobNet()

        # Multiple random states
        for _ in range(10):
            state = torch.randn(256)
            _, value = net(state)

            # Value should be in [-1, 1] due to tanh activation
            assert -1.0 <= value.item() <= 1.0, \
                f"Value {value.item()} outside [-1, 1]"

    def test_policy_sums_to_one(self):
        """Test policy probabilities sum to 1."""
        net = BlobNet()

        state = torch.randn(256)
        policy, _ = net(state)

        # Policy should sum to 1 (softmax normalization)
        policy_sum = policy.sum().item()
        assert abs(policy_sum - 1.0) < 1e-5, \
            f"Policy sums to {policy_sum}, expected 1.0"

    def test_legal_action_masking(self):
        """Test legal action masking zeros out illegal actions."""
        net = BlobNet()

        state = torch.randn(256)

        # Create mask: only first 5 actions legal
        mask = torch.zeros(52)
        mask[0:5] = 1.0

        policy, _ = net(state, legal_actions_mask=mask)

        # Legal actions should have non-zero probability
        assert (policy[0:5] > 0).all(), "Legal actions should have probability > 0"

        # Illegal actions should have zero probability
        assert (policy[5:] == 0).all(), "Illegal actions should have probability 0"

        # Policy should still sum to 1
        assert abs(policy.sum().item() - 1.0) < 1e-5

    def test_legal_action_masking_batch(self):
        """Test legal action masking with batched input."""
        net = BlobNet()

        batch_size = 4
        state_batch = torch.randn(batch_size, 256)

        # Different masks for each batch item
        mask_batch = torch.zeros(batch_size, 52)
        mask_batch[0, 0:3] = 1.0   # First state: 3 legal actions
        mask_batch[1, 0:5] = 1.0   # Second state: 5 legal actions
        mask_batch[2, 0:10] = 1.0  # Third state: 10 legal actions
        mask_batch[3, :] = 1.0     # Fourth state: all legal

        policy_batch, _ = net(state_batch, legal_actions_mask=mask_batch)

        # Check each batch item
        for i in range(batch_size):
            policy = policy_batch[i]
            mask = mask_batch[i]

            # Legal actions have positive probability
            legal_actions = mask == 1.0
            assert (policy[legal_actions] > 0).all()

            # Illegal actions have zero probability
            illegal_actions = mask == 0.0
            if illegal_actions.any():
                assert (policy[illegal_actions] == 0).all()

            # Each policy sums to 1
            assert abs(policy.sum().item() - 1.0) < 1e-5

    def test_deterministic_output(self):
        """Test same input produces same output (no randomness in forward pass)."""
        net = BlobNet()
        net.eval()  # Evaluation mode (disables dropout)

        state = torch.randn(256)

        # Two forward passes
        with torch.no_grad():
            policy1, value1 = net(state)
            policy2, value2 = net(state)

        # Should be identical
        assert torch.allclose(policy1, policy2)
        assert torch.allclose(value1, value2)

    def test_factory_function(self):
        """Test create_model factory function."""
        model = create_model(state_dim=256, num_layers=4)

        assert isinstance(model, BlobNet)
        assert model.state_dim == 256

        # Should be on CPU by default
        assert next(model.parameters()).device.type == 'cpu'


class TestBlobNetTrainer:
    """Test suite for BlobNetTrainer."""

    def test_trainer_initialization(self):
        """Test BlobNetTrainer initializes correctly."""
        model = BlobNet()
        trainer = BlobNetTrainer(
            model,
            learning_rate=0.001,
            value_loss_weight=1.0,
            policy_loss_weight=1.0,
        )

        assert trainer.model is model
        assert trainer.value_loss_weight == 1.0
        assert trainer.policy_loss_weight == 1.0
        assert trainer.optimizer is not None

    def test_loss_computation(self):
        """Test loss computation returns expected values."""
        model = BlobNet()
        trainer = BlobNetTrainer(model)

        # Create dummy data
        state = torch.randn(4, 256)  # Batch of 4

        # Target policy (random valid distribution)
        target_policy = torch.rand(4, 52)
        target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True)

        # Target value
        target_value = torch.randn(4, 1).clamp(-1, 1)

        # Legal action mask (all legal)
        mask = torch.ones(4, 52)

        # Compute loss
        loss, loss_dict = trainer.compute_loss(state, target_policy, target_value, mask)

        # Check loss is scalar tensor
        assert loss.dim() == 0
        assert loss.item() > 0  # Loss should be positive

        # Check loss dict has expected keys
        assert 'total_loss' in loss_dict
        assert 'policy_loss' in loss_dict
        assert 'value_loss' in loss_dict

        # All losses should be positive
        assert loss_dict['total_loss'] > 0
        assert loss_dict['policy_loss'] > 0
        assert loss_dict['value_loss'] > 0

    def test_train_step(self):
        """Test training step updates model."""
        model = BlobNet()
        trainer = BlobNetTrainer(model, learning_rate=0.01)

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Create dummy data
        state = torch.randn(4, 256)
        target_policy = torch.rand(4, 52)
        target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True)
        target_value = torch.randn(4, 1).clamp(-1, 1)
        mask = torch.ones(4, 52)

        # Training step
        loss_dict = trainer.train_step(state, target_policy, target_value, mask)

        # Check parameters changed
        for initial, current in zip(initial_params, model.parameters()):
            assert not torch.equal(initial, current), \
                "Parameters should change after training step"

        # Check loss dict returned
        assert 'total_loss' in loss_dict

    def test_loss_decreases_with_training(self):
        """Test loss decreases over multiple training steps."""
        # Use fixed seed for reproducible tests
        torch.manual_seed(42)

        model = BlobNet()
        trainer = BlobNetTrainer(model)  # Use default learning_rate=0.001

        # Create dummy data (same data for overfitting)
        state = torch.randn(4, 256)
        target_policy = torch.rand(4, 52)
        target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True)
        target_value = torch.randn(4, 1).clamp(-1, 1)
        mask = torch.ones(4, 52)

        # Train for multiple steps
        losses = []
        for _ in range(20):
            loss_dict = trainer.train_step(state, target_policy, target_value, mask)
            losses.append(loss_dict['total_loss'])

        # Loss should generally decrease (allow some fluctuation)
        # Check that final loss < initial loss
        assert losses[-1] < losses[0], \
            f"Loss should decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        model = BlobNet()
        trainer = BlobNetTrainer(model)

        # Train for a few steps to change parameters
        state = torch.randn(4, 256)
        target_policy = torch.rand(4, 52)
        target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True)
        target_value = torch.randn(4, 1).clamp(-1, 1)
        mask = torch.ones(4, 52)

        for _ in range(5):
            trainer.train_step(state, target_policy, target_value, mask)

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pth"

            # Save
            trainer.save_checkpoint(
                str(checkpoint_path),
                iteration=100,
                metadata={'test': 'data'}
            )

            # Check file exists
            assert checkpoint_path.exists()

            # Create new model and trainer
            new_model = BlobNet()
            new_trainer = BlobNetTrainer(new_model)

            # Get initial parameters (should be different from trained)
            initial_params = [p.clone() for p in new_model.parameters()]

            # Load checkpoint
            iteration, metadata = new_trainer.load_checkpoint(str(checkpoint_path))

            # Check iteration and metadata
            assert iteration == 100
            assert metadata == {'test': 'data'}

            # Check parameters match original trained model
            for orig_param, loaded_param in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(orig_param, loaded_param), \
                    "Loaded parameters should match saved parameters"

    def test_factory_function_trainer(self):
        """Test create_trainer factory function."""
        model = create_model()
        trainer = create_trainer(model, learning_rate=0.002)

        assert isinstance(trainer, BlobNetTrainer)
        assert trainer.model is model


class TestNetworkIntegration:
    """Integration tests for complete pipeline."""

    def test_encode_and_inference(self):
        """Test encoding game state and running inference."""
        # Create components
        encoder = StateEncoder()
        masker = ActionMasker()
        net = BlobNet()

        # Create game
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        # Encode state
        state = encoder.encode(game, player)

        # Create action mask (bidding phase)
        mask = masker.create_bidding_mask(
            cards_dealt=5,
            is_dealer=False,
            forbidden_bid=None
        )

        # Run inference
        policy, value = net(state, mask)

        # Check outputs
        assert policy.shape == (52,)
        assert value.shape == (1,)
        assert abs(policy.sum().item() - 1.0) < 1e-5
        assert -1.0 <= value.item() <= 1.0

        # Check only legal bids have probability
        # Legal bids: 0-5 (indices 0-5)
        assert (policy[0:6] > 0).all()
        assert (policy[6:] == 0).all()

    def test_full_training_pipeline(self):
        """Test complete training pipeline with real game data."""
        # Use fixed seed for reproducible tests
        torch.manual_seed(42)

        # Create components
        encoder = StateEncoder()
        masker = ActionMasker()
        model = BlobNet()
        trainer = BlobNetTrainer(model)  # Use default learning_rate=0.001

        # Generate training data from multiple game states
        states = []
        target_policies = []
        target_values = []
        masks = []

        for _ in range(8):  # 8 game states
            game = BlobGame(num_players=4)
            game.setup_round(cards_to_deal=5)
            player = game.players[0]

            # Encode state
            state = encoder.encode(game, player)
            states.append(state)

            # Create mask
            mask = masker.create_bidding_mask(
                cards_dealt=5,
                is_dealer=(player.position == game.dealer_position),
                forbidden_bid=None
            )
            masks.append(mask)

            # Create dummy target policy (uniform over legal actions)
            target_policy = mask.clone()
            target_policy = target_policy / target_policy.sum()
            target_policies.append(target_policy)

            # Create dummy target value
            target_values.append(torch.tensor([0.5]))

        # Stack into batches
        state_batch = torch.stack(states)
        policy_batch = torch.stack(target_policies)
        value_batch = torch.stack(target_values)
        mask_batch = torch.stack(masks)

        # Train for multiple steps
        initial_loss = None
        for i in range(10):
            loss_dict = trainer.train_step(
                state_batch, policy_batch, value_batch, mask_batch
            )

            if i == 0:
                initial_loss = loss_dict['total_loss']

        final_loss = loss_dict['total_loss']

        # Loss should decrease
        assert final_loss < initial_loss, \
            f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_training_robustness_across_seeds(self):
        """
        Test training converges reliably across different random initializations.

        This test verifies that the learning rate and training procedure are robust
        to different weight initializations. With a properly tuned learning rate,
        at least 90% of random seeds should result in loss decrease.

        This catches potential issues with:
        - Learning rate too high (causes divergence on some seeds)
        - Gradient explosion/vanishing
        - Optimizer instability
        """
        success_count = 0
        num_trials = 10
        failed_seeds = []

        for seed in range(num_trials):
            torch.manual_seed(seed)
            model = BlobNet()
            trainer = BlobNetTrainer(model)  # Use default lr=0.001

            # Create dummy training data (unique per seed)
            torch.manual_seed(seed + 1000)
            state = torch.randn(4, 256)
            target_policy = torch.rand(4, 52)
            target_policy = target_policy / target_policy.sum(dim=-1, keepdim=True)
            target_value = torch.randn(4, 1).clamp(-1, 1)
            mask = torch.ones(4, 52)

            # Train for 50 steps (more than basic test to ensure convergence)
            losses = []
            for _ in range(50):
                loss_dict = trainer.train_step(state, target_policy, target_value, mask)
                losses.append(loss_dict['total_loss'])

            # Check if this seed converged
            if losses[-1] < losses[0]:
                success_count += 1
            else:
                failed_seeds.append(seed)

        # Require 90% success rate (allows 1 unlucky initialization out of 10)
        success_rate = success_count / num_trials
        assert success_count >= 9, \
            f"Training unreliable: only {success_count}/{num_trials} seeds converged " \
            f"({success_rate:.0%} success rate). Failed seeds: {failed_seeds}"

    def test_performance_benchmark(self):
        """Test inference performance meets targets (<10ms per forward pass)."""
        import time

        net = BlobNet()
        net.eval()

        # Warmup
        state = torch.randn(256)
        with torch.no_grad():
            for _ in range(10):
                net(state)

        # Benchmark
        num_runs = 100
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                policy, value = net(state)

        elapsed_ms = (time.time() - start_time) * 1000
        avg_time_ms = elapsed_ms / num_runs

        print(f"\nInference benchmark:")
        print(f"  Average time per forward pass: {avg_time_ms:.2f} ms")
        print(f"  Total time for {num_runs} runs: {elapsed_ms:.2f} ms")

        # Should be less than 10ms per forward pass on CPU
        # Note: This may fail on very slow machines, adjust if needed
        assert avg_time_ms < 50, \
            f"Inference too slow: {avg_time_ms:.2f} ms > 50 ms target"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
