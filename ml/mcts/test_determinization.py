"""
Tests for belief tracking and determinization in imperfect information games.

This module tests the belief state tracking system and determinization sampling
that enable MCTS to work with hidden opponent hands.

Test Coverage:
    - BeliefState initialization and setup
    - PlayerConstraints validation
    - Suit elimination from card plays
    - Hand consistency checking
    - Belief state updates
"""

import pytest
from ml.game.blob import BlobGame, Card
from ml.mcts.belief_tracker import BeliefState, PlayerConstraints


class TestBeliefState:
    """Tests for BeliefState class."""

    def test_belief_state_initialization(self):
        """Test belief state initializes correctly."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        belief = BeliefState(game, player)

        # Should know own hand
        assert belief.known_cards == set(player.hand)

        # Should have constraints for other players
        assert len(belief.player_constraints) == 3

        # Unseen cards should be rest of deck
        assert len(belief.unseen_cards) == 52 - 5  # 52 cards - our 5

    def test_player_constraints_initialization(self):
        """Test player constraints are set up correctly."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        belief = BeliefState(game, player)

        for pos, constraints in belief.player_constraints.items():
            assert constraints.cards_in_hand == 5
            assert len(constraints.cards_played) == 0
            assert len(constraints.cannot_have_suits) == 0

    def test_update_on_card_played_reveals_suit_info(self):
        """Test belief updates when player doesn't follow suit."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]
        opponent = game.players[1]

        belief = BeliefState(game, observer)

        # Simulate opponent playing off-suit
        led_suit = "♠"
        card_played = Card("K", "♥")  # Hearts when Spades was led

        belief.update_on_card_played(opponent, card_played, led_suit)

        # Should now know opponent doesn't have Spades
        constraints = belief.player_constraints[opponent.position]
        assert "♠" in constraints.cannot_have_suits

    def test_get_possible_cards_respects_constraints(self):
        """Test possible cards honors suit elimination."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]
        opponent = game.players[1]

        belief = BeliefState(game, observer)

        # Manually add constraint: opponent doesn't have Spades
        constraints = belief.player_constraints[opponent.position]
        constraints.cannot_have_suits.add("♠")

        possible = belief.get_possible_cards(opponent.position)

        # No Spades should be possible
        spades_in_possible = [c for c in possible if c.suit == "♠"]
        assert len(spades_in_possible) == 0

    def test_is_consistent_hand_validates_correctly(self):
        """Test hand consistency validation."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]
        opponent_pos = 1

        belief = BeliefState(game, observer)

        # Manually set constraints
        constraints = belief.player_constraints[opponent_pos]
        constraints.cannot_have_suits.add("♠")
        constraints.cards_in_hand = 5

        # Valid hand (no Spades, 5 cards)
        valid_hand = [
            Card("2", "♥"),
            Card("3", "♥"),
            Card("4", "♣"),
            Card("5", "♦"),
            Card("6", "♦"),
        ]
        assert belief.is_consistent_hand(opponent_pos, valid_hand)

        # Invalid hand (has Spades)
        invalid_hand = [
            Card("A", "♠"),
            Card("3", "♥"),
            Card("4", "♣"),
            Card("5", "♦"),
            Card("6", "♦"),
        ]
        assert not belief.is_consistent_hand(opponent_pos, invalid_hand)

    def test_is_consistent_hand_checks_size(self):
        """Test hand consistency checks hand size."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]
        opponent_pos = 1

        belief = BeliefState(game, observer)

        # Wrong size hand (4 cards instead of 5)
        wrong_size_hand = [
            Card("2", "♥"),
            Card("3", "♥"),
            Card("4", "♣"),
            Card("5", "♦"),
        ]
        assert not belief.is_consistent_hand(opponent_pos, wrong_size_hand)

    def test_is_consistent_hand_checks_must_have_suits(self):
        """Test hand consistency checks must-have suits."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]
        opponent_pos = 1

        belief = BeliefState(game, observer)

        # Add must-have constraint
        constraints = belief.player_constraints[opponent_pos]
        constraints.must_have_suits.add("♥")

        # Valid hand (has Hearts)
        valid_hand = [
            Card("2", "♥"),
            Card("3", "♣"),
            Card("4", "♣"),
            Card("5", "♦"),
            Card("6", "♦"),
        ]
        assert belief.is_consistent_hand(opponent_pos, valid_hand)

        # Invalid hand (no Hearts)
        invalid_hand = [
            Card("2", "♣"),
            Card("3", "♣"),
            Card("4", "♣"),
            Card("5", "♦"),
            Card("6", "♦"),
        ]
        assert not belief.is_consistent_hand(opponent_pos, invalid_hand)

    def test_update_on_card_played_updates_unseen_cards(self):
        """Test that played cards are removed from unseen set."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]
        opponent = game.players[1]

        belief = BeliefState(game, observer)

        initial_unseen_count = len(belief.unseen_cards)

        # Play a card
        card_played = Card("K", "♥")
        belief.unseen_cards.add(card_played)  # Add it first to test removal

        belief.update_on_card_played(opponent, card_played, None)

        # Card should be in played cards
        assert card_played in belief.played_cards

        # Card should not be in unseen cards
        assert card_played not in belief.unseen_cards

    def test_copy_creates_independent_instance(self):
        """Test that copy creates an independent belief state."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief1 = BeliefState(game, observer)

        # Add some constraints
        belief1.player_constraints[1].cannot_have_suits.add("♠")

        # Create copy
        belief2 = belief1.copy()

        # Modify copy
        belief2.player_constraints[1].cannot_have_suits.add("♥")

        # Original should not be affected
        assert "♥" not in belief1.player_constraints[1].cannot_have_suits
        assert "♥" in belief2.player_constraints[1].cannot_have_suits

    def test_get_possible_cards_for_observer(self):
        """Test getting possible cards for observer returns known cards."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)

        possible = belief.get_possible_cards(observer.position)

        # Should return exactly the known cards (observer's hand)
        assert possible == belief.known_cards

    def test_observer_hand_validation(self):
        """Test that observer's hand is validated correctly."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)

        # Observer's actual hand should be valid
        assert belief.is_consistent_hand(observer.position, list(observer.hand))

        # Different hand should not be valid
        different_hand = [
            Card("2", "♥"),
            Card("3", "♥"),
            Card("4", "♣"),
            Card("5", "♦"),
            Card("6", "♦"),
        ]
        if set(different_hand) != set(observer.hand):
            assert not belief.is_consistent_hand(observer.position, different_hand)


class TestPlayerConstraints:
    """Tests for PlayerConstraints class."""

    def test_can_have_card_respects_played_cards(self):
        """Test can_have_card returns False for already played cards."""
        constraints = PlayerConstraints(player_position=1, cards_in_hand=5)

        card = Card("A", "♠")
        constraints.cards_played.add(card)

        assert not constraints.can_have_card(card)

    def test_can_have_card_respects_suit_elimination(self):
        """Test can_have_card respects suit elimination."""
        constraints = PlayerConstraints(player_position=1, cards_in_hand=5)

        constraints.cannot_have_suits.add("♠")

        # Should not be able to have any Spade
        assert not constraints.can_have_card(Card("A", "♠"))
        assert not constraints.can_have_card(Card("2", "♠"))

        # Should be able to have other suits
        assert constraints.can_have_card(Card("A", "♥"))

    def test_can_have_card_allows_valid_cards(self):
        """Test can_have_card allows cards that satisfy constraints."""
        constraints = PlayerConstraints(player_position=1, cards_in_hand=5)

        card = Card("K", "♥")

        # Should allow card with no constraints
        assert constraints.can_have_card(card)


class TestBeliefStateIntegration:
    """Integration tests for belief state with game history."""

    def test_belief_state_extracts_constraints_from_history(self):
        """Test that belief state correctly extracts constraints from game history."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)

        # Manually set up a trick where player 1 didn't follow suit
        observer = game.players[0]

        # Create a trick manually (need to pass trump suit)
        from ml.game.blob import Trick

        trick = Trick(trump_suit=None)
        trick.led_suit = "♠"

        # Player 1 plays off-suit (revealing they don't have Spades)
        player1 = game.players[1]
        card_played = Card("K", "♥")
        trick.cards_played.append((player1, card_played))

        # Add to history
        game.tricks_history.append(trick)

        # Create belief state
        belief = BeliefState(game, observer)

        # Should have extracted the constraint
        constraints = belief.player_constraints[1]
        assert "♠" in constraints.cannot_have_suits

    def test_belief_state_tracks_followed_suit(self):
        """Test that belief state tracks when players follow suit."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)

        observer = game.players[0]

        # Create a trick where player 1 follows suit
        from ml.game.blob import Trick

        trick = Trick(trump_suit=None)
        trick.led_suit = "♠"

        # Player 1 plays Spades (following suit)
        player1 = game.players[1]
        card_played = Card("K", "♠")
        trick.cards_played.append((player1, card_played))

        # Add to history
        game.tricks_history.append(trick)

        # Create belief state
        belief = BeliefState(game, observer)

        # Should have tracked that they have Spades
        constraints = belief.player_constraints[1]
        assert "♠" in constraints.must_have_suits
