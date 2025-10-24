"""
Tests for neural network state encoding.

Tests the StateEncoder class that converts game states into tensor representations.
"""

import pytest
import torch
from ml.game.blob import BlobGame, Card
from ml.network.encode import StateEncoder, ActionMasker


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
