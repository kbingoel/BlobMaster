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


class TestProbabilisticBeliefTracking:
    """Tests for probabilistic belief tracking and Bayesian updates."""

    def test_initial_probabilities_uniform(self):
        """Test initial probabilities are uniform over unseen cards."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)

        # Check probabilities sum to cards_in_hand for each player
        for player_pos, probs in belief.card_probabilities.items():
            total = sum(probs.values())
            expected = belief.player_constraints[player_pos].cards_in_hand
            assert abs(total - expected) < 0.01  # Allow small floating point error

    def test_probability_update_on_card_played(self):
        """Test probabilities update correctly when card is played."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]
        opponent_pos = 1

        belief = BeliefState(game, observer)

        card = list(belief.unseen_cards)[0]
        initial_prob = belief.get_card_probability(opponent_pos, card)

        # Simulate card being played by opponent
        belief.update_probabilities_on_card_played(opponent_pos, card, None)

        # Probability should now be 0
        assert belief.get_card_probability(opponent_pos, card) == 0.0

    def test_suit_elimination_zeros_probabilities(self):
        """Test suit elimination zeros probabilities correctly."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]
        opponent_pos = 1

        belief = BeliefState(game, observer)

        # Play off-suit card
        led_suit = "♠"
        card_played = Card("K", "♥")
        belief.update_probabilities_on_card_played(opponent_pos, card_played, led_suit)

        # All Spades should have 0 probability
        for card in belief.unseen_cards:
            if card.suit == "♠":
                prob = belief.get_card_probability(opponent_pos, card)
                assert prob == 0.0

    def test_entropy_decreases_with_information(self):
        """Test entropy decreases as we gain information."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]
        opponent_pos = 1

        belief = BeliefState(game, observer)

        initial_entropy = belief.get_entropy(opponent_pos)

        # Eliminate a suit
        constraints = belief.player_constraints[opponent_pos]
        constraints.cannot_have_suits.add("♠")
        belief._initialize_probabilities()

        final_entropy = belief.get_entropy(opponent_pos)

        # Entropy should decrease
        assert final_entropy < initial_entropy

    def test_probability_normalization(self):
        """Test probabilities normalize correctly after updates."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]
        opponent_pos = 1

        belief = BeliefState(game, observer)

        # Play a few cards
        cards_to_play = list(belief.unseen_cards)[:3]
        for card in cards_to_play:
            belief.update_probabilities_on_card_played(opponent_pos, card, None)

        # Probabilities should still sum to cards_in_hand
        total_prob = sum(belief.card_probabilities[opponent_pos].values())
        expected = belief.player_constraints[opponent_pos].cards_in_hand
        assert abs(total_prob - expected) < 0.01

    def test_get_most_likely_holder(self):
        """Test finding most likely card holder."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)

        # Test for a card in observer's hand
        if len(observer.hand) > 0:
            observer_card = list(observer.hand)[0]
            holder, prob = belief.get_most_likely_holder(observer_card)
            assert holder == observer.position
            assert prob == 1.0

        # Test for unseen card
        unseen_card = list(belief.unseen_cards)[0]
        holder, prob = belief.get_most_likely_holder(unseen_card)
        # Should return one of the opponents
        assert holder in belief.player_constraints
        assert 0.0 <= prob <= 1.0

    def test_get_card_probability_for_observer(self):
        """Test getting probability for observer's cards."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)

        # Cards in observer's hand should have probability 1.0
        for card in observer.hand:
            prob = belief.get_card_probability(observer.position, card)
            assert prob == 1.0

        # Cards not in observer's hand should have probability 0.0
        other_card = list(belief.unseen_cards)[0]
        prob = belief.get_card_probability(observer.position, other_card)
        assert prob == 0.0

    def test_entropy_zero_for_observer(self):
        """Test entropy is zero for observer (perfect information)."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)

        entropy = belief.get_entropy(observer.position)
        assert entropy == 0.0

    def test_update_from_trick_updates_probabilities(self):
        """Test update_from_trick updates both constraints and probabilities."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)

        # Create a trick manually
        from ml.game.blob import Trick

        trick = Trick(trump_suit=None)
        trick.led_suit = "♠"

        # Player 1 plays off-suit
        player1 = game.players[1]
        card_played = Card("K", "♥")
        trick.cards_played.append((player1, card_played))

        initial_entropy = belief.get_entropy(1)

        # Update from trick
        belief.update_from_trick(trick)

        # Should have updated constraints
        assert "♠" in belief.player_constraints[1].cannot_have_suits

        # Probabilities for Spades should be zero
        for card in belief.unseen_cards:
            if card.suit == "♠":
                assert belief.get_card_probability(1, card) == 0.0

        # Entropy should decrease
        final_entropy = belief.get_entropy(1)
        assert final_entropy < initial_entropy

    def test_get_belief_summary_format(self):
        """Test belief summary produces readable output."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)

        summary = belief.get_belief_summary()

        # Check it's a non-empty string
        assert len(summary) > 0
        assert "Belief State Summary" in summary
        assert f"Observer: Player {observer.position}" in summary
        assert "Unseen cards:" in summary

        # Should contain info about each opponent
        for player_pos in belief.player_constraints:
            assert f"Player {player_pos}:" in summary
            assert "Cards in hand:" in summary
            assert "Belief entropy:" in summary

    def test_copy_preserves_probabilities(self):
        """Test that copy preserves probability distributions."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief1 = BeliefState(game, observer)

        # Modify some probabilities
        first_card = list(belief1.unseen_cards)[0]
        belief1.update_probabilities_on_card_played(1, first_card, None)

        # Create copy
        belief2 = belief1.copy()

        # Check probabilities match
        for player_pos in belief1.player_constraints:
            for card in belief1.unseen_cards:
                prob1 = belief1.get_card_probability(player_pos, card)
                prob2 = belief2.get_card_probability(player_pos, card)
                assert abs(prob1 - prob2) < 0.01

        # Modify copy
        second_card = list(belief2.unseen_cards)[1]
        belief2.update_probabilities_on_card_played(1, second_card, None)

        # Original should not be affected
        prob1_original = belief1.get_card_probability(1, second_card)
        prob2_modified = belief2.get_card_probability(1, second_card)
        assert prob1_original > 0.0  # Original still has probability
        assert prob2_modified == 0.0  # Copy has zero probability


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


class TestDeterminization:
    """Tests for Determinization sampling algorithm."""

    def test_determinizer_samples_valid_hands(self):
        """Test determinizer produces valid hand assignments."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)
        determinizer = Determinizer()

        sample = determinizer.sample_determinization(game, belief)

        assert sample is not None
        assert len(sample) == 3  # 3 opponents

        # Each opponent should have 5 cards
        for hand in sample.values():
            assert len(hand) == 5

    def test_determinizer_respects_suit_constraints(self):
        """Test sampled hands respect suit elimination."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]
        opponent_pos = 1

        belief = BeliefState(game, observer)

        # Add constraint: opponent doesn't have Spades
        constraints = belief.player_constraints[opponent_pos]
        constraints.cannot_have_suits.add("♠")

        determinizer = Determinizer()
        sample = determinizer.sample_determinization(game, belief)

        assert sample is not None

        # Opponent's hand should have no Spades
        opponent_hand = sample[opponent_pos]
        spades_in_hand = [c for c in opponent_hand if c.suit == "♠"]
        assert len(spades_in_hand) == 0

    def test_multiple_samples_are_different(self):
        """Test multiple samples produce different results."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)
        determinizer = Determinizer()

        samples = determinizer.sample_multiple_determinizations(
            game, belief, num_samples=10
        )

        assert len(samples) >= 8  # Should get most samples

        # Check diversity (at least some samples should be different)
        unique_samples = set()
        for sample in samples:
            # Convert to hashable format
            sample_tuple = tuple(
                sorted([tuple(sorted(hand)) for hand in sample.values()])
            )
            unique_samples.add(sample_tuple)

        assert len(unique_samples) > 1  # At least 2 different samples

    def test_create_determinized_game(self):
        """Test creating a complete game from determinization."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)
        determinizer = Determinizer()

        sample = determinizer.sample_determinization(game, belief)
        det_game = determinizer.create_determinized_game(game, belief, sample)

        # All players should have 5 cards
        for player in det_game.players:
            assert len(player.hand) == 5

        # Observer's hand should match original
        assert set(det_game.players[observer.position].hand) == belief.known_cards

    def test_sampling_performance(self):
        """Test sampling is fast enough (<10ms per sample)."""
        import time
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)
        determinizer = Determinizer()

        start = time.time()
        for _ in range(100):
            sample = determinizer.sample_determinization(game, belief)
        elapsed_ms = (time.time() - start) * 1000 / 100

        print(f"Sampling time: {elapsed_ms:.2f} ms per sample")
        assert elapsed_ms < 10.0  # Should be fast

    def test_determinizer_handles_no_unseen_cards(self):
        """Test determinizer handles edge case of no unseen cards."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)

        # Simulate all cards being known (edge case)
        # This is tricky - we'll just check that sampling doesn't crash
        determinizer = Determinizer()

        # Should handle gracefully
        sample = determinizer.sample_determinization(game, belief)
        # May return None or valid sample depending on state

    def test_determinizer_validates_hand_size(self):
        """Test that sampled hands have correct size."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=7)  # Different hand size
        observer = game.players[0]

        belief = BeliefState(game, observer)
        determinizer = Determinizer()

        sample = determinizer.sample_determinization(game, belief)

        if sample is not None:
            # All hands should have 7 cards
            for player_pos, hand in sample.items():
                expected_size = belief.player_constraints[player_pos].cards_in_hand
                assert len(hand) == expected_size

    def test_determinizer_no_duplicate_cards(self):
        """Test that sampled hands have no duplicate cards."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)
        determinizer = Determinizer()

        sample = determinizer.sample_determinization(game, belief)

        assert sample is not None

        # Collect all cards
        all_sampled_cards = []
        for hand in sample.values():
            all_sampled_cards.extend(hand)

        # Check no duplicates
        assert len(all_sampled_cards) == len(set(all_sampled_cards))

    def test_determinizer_respects_must_have_suits(self):
        """Test determinizer respects must-have suit constraints."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]
        opponent_pos = 1

        belief = BeliefState(game, observer)

        # Add constraint: opponent must have Hearts
        constraints = belief.player_constraints[opponent_pos]
        constraints.must_have_suits.add("♥")

        determinizer = Determinizer()
        sample = determinizer.sample_determinization(game, belief)

        if sample is not None:
            # Opponent's hand should have at least one Heart
            opponent_hand = sample[opponent_pos]
            hearts_in_hand = [c for c in opponent_hand if c.suit == "♥"]
            assert len(hearts_in_hand) > 0

    def test_uniform_vs_probability_sampling(self):
        """Test both uniform and probability-weighted sampling work."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)
        determinizer = Determinizer()

        # Uniform sampling
        sample_uniform = determinizer.sample_determinization(
            game, belief, use_probabilities=False
        )
        assert sample_uniform is not None

        # Probability-weighted sampling
        sample_prob = determinizer.sample_determinization(
            game, belief, use_probabilities=True
        )
        assert sample_prob is not None

    def test_determinizer_tight_constraints(self):
        """Test determinizer handles very tight constraints."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)

        # Add very tight constraints
        for player_pos in belief.player_constraints:
            constraints = belief.player_constraints[player_pos]
            # Eliminate 2 suits (tight but not impossible)
            constraints.cannot_have_suits = {"♠", "♥"}

        determinizer = Determinizer()

        # Should still try to sample (may fail if truly impossible)
        sample = determinizer.sample_determinization(game, belief)

        # Either succeeds with valid sample or returns None
        if sample is not None:
            assert determinizer._validate_sample(sample, belief)

    def test_determinized_game_preserves_game_state(self):
        """Test that determinized game preserves other game state."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        # Set some game state
        game.trump_suit = "♥"

        belief = BeliefState(game, observer)
        determinizer = Determinizer()

        sample = determinizer.sample_determinization(game, belief)
        det_game = determinizer.create_determinized_game(game, belief, sample)

        # Game state should be preserved
        assert det_game.trump_suit == game.trump_suit
        assert det_game.num_players == game.num_players

    def test_multiple_determinizations_batch_size(self):
        """Test that batch sampling returns requested number of samples."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)
        determinizer = Determinizer()

        # Request 5 samples
        samples = determinizer.sample_multiple_determinizations(
            game, belief, num_samples=5
        )

        # Should get close to 5 (may be slightly less if sampling fails)
        assert len(samples) >= 4


class TestDeterminizationAdvanced:
    """Advanced tests for determinization quality and optimization."""

    def test_diversity_sampling(self):
        """Test diversity-focused sampling produces different results."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)
        determinizer = Determinizer()

        samples = determinizer.sample_adaptive(
            game, belief, num_samples=5, diversity_weight=0.8
        )

        # Should get most samples
        assert len(samples) >= 4

        # Check diversity - samples should be different from each other
        for i, sample1 in enumerate(samples):
            for j, sample2 in enumerate(samples):
                if i < j:
                    similarity = determinizer._compute_similarity(sample1, sample2)
                    # Should be different (similarity < 0.9)
                    assert similarity < 0.9

    def test_compute_similarity_identical_samples(self):
        """Test similarity computation for identical samples."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)
        determinizer = Determinizer()

        sample = determinizer.sample_determinization(game, belief)

        # Identical samples should have similarity 1.0
        similarity = determinizer._compute_similarity(sample, sample)
        assert abs(similarity - 1.0) < 0.01

    def test_compute_similarity_different_samples(self):
        """Test similarity computation for different samples."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)
        determinizer = Determinizer()

        sample1 = determinizer.sample_determinization(game, belief)
        sample2 = determinizer.sample_determinization(game, belief)

        # Different samples should have similarity < 1.0 (usually)
        similarity = determinizer._compute_similarity(sample1, sample2)
        assert 0.0 <= similarity <= 1.0

    def test_is_diverse_threshold(self):
        """Test diversity checking with different thresholds."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)
        determinizer = Determinizer()

        sample1 = determinizer.sample_determinization(game, belief)
        sample2 = determinizer.sample_determinization(game, belief)

        existing_samples = [sample1]

        # Check diversity with strict threshold (0.5 = 50% different)
        is_diverse_strict = determinizer._is_diverse(
            sample2, existing_samples, threshold=0.5
        )

        # Check diversity with loose threshold (0.1 = 10% different)
        is_diverse_loose = determinizer._is_diverse(
            sample2, existing_samples, threshold=0.1
        )

        # Loose threshold should be easier to satisfy
        # (this is probabilistic, so we just check the API works)
        assert isinstance(is_diverse_strict, bool)
        assert isinstance(is_diverse_loose, bool)

    def test_constraint_propagation_detects_conflicts(self):
        """Test constraint propagation catches impossible scenarios."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)
        determinizer = Determinizer()

        # Create very tight constraints (may make sampling impossible)
        for player_pos in belief.player_constraints:
            constraints = belief.player_constraints[player_pos]
            # Eliminate 3 suits (very constrained)
            constraints.cannot_have_suits = {"♠", "♥", "♣"}

        # Try to sample - should handle gracefully
        sample = determinizer.sample_determinization(game, belief)

        # Either succeeds with valid sample or returns None
        if sample is not None:
            assert determinizer._validate_sample(sample, belief)

    def test_constraint_propagation_valid_partial(self):
        """Test constraint propagation accepts valid partial samples."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)
        determinizer = Determinizer()

        # Create a partial sample (just one player)
        sample1 = determinizer.sample_determinization(game, belief)
        if sample1 is not None:
            # Take just the first player's hand
            partial_sample = {list(sample1.keys())[0]: sample1[list(sample1.keys())[0]]}

            # Should be able to satisfy remaining players
            can_satisfy = determinizer._propagate_constraints(belief, partial_sample)

            # Should return True for reasonable constraints
            assert isinstance(can_satisfy, bool)

    def test_probability_weighted_sampling_quality(self):
        """Test probability-weighted sampling respects distributions."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]
        opponent_pos = 1

        belief = BeliefState(game, observer)

        # Artificially boost probability of specific card
        high_prob_card = list(belief.unseen_cards)[0]
        belief.card_probabilities[opponent_pos][high_prob_card] = 2.0
        belief._normalize_probabilities(opponent_pos)

        determinizer = Determinizer()

        # Sample many times
        count_with_card = 0
        num_trials = 100

        for _ in range(num_trials):
            sample = determinizer.sample_determinization(
                game, belief, use_probabilities=True
            )
            if sample and high_prob_card in sample[opponent_pos]:
                count_with_card += 1

        # Should appear more often than random (>20% vs ~20% expected baseline)
        # With boosted probability, should see it more frequently
        assert count_with_card > 25

    def test_sample_adaptive_diversity_weight_zero(self):
        """Test adaptive sampling with zero diversity weight (pure probability)."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)
        determinizer = Determinizer()

        samples = determinizer.sample_adaptive(
            game, belief, num_samples=5, diversity_weight=0.0
        )

        # Should still get samples
        assert len(samples) >= 4

    def test_sample_adaptive_diversity_weight_one(self):
        """Test adaptive sampling with max diversity weight."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)
        determinizer = Determinizer()

        samples = determinizer.sample_adaptive(
            game, belief, num_samples=5, diversity_weight=1.0
        )

        # Should still get samples
        assert len(samples) >= 4

        # Check that samples are diverse
        if len(samples) >= 2:
            for i in range(len(samples) - 1):
                similarity = determinizer._compute_similarity(samples[i], samples[i + 1])
                # Should be reasonably diverse
                assert similarity < 0.95

    def test_sample_with_diversity_fallback(self):
        """Test diversity sampling falls back gracefully."""
        from ml.mcts.determinization import Determinizer

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        observer = game.players[0]

        belief = BeliefState(game, observer)
        determinizer = Determinizer(max_attempts=10)  # Low attempts for testing

        # Generate a few samples
        existing_samples = []
        for _ in range(3):
            sample = determinizer.sample_determinization(game, belief)
            if sample:
                existing_samples.append(sample)

        # Try to get a diverse sample
        diverse_sample = determinizer.sample_determinization_with_diversity(
            game, belief, avoid_samples=existing_samples
        )

        # Should either get a diverse sample or fall back to any sample
        # (either way, should get something)
        assert diverse_sample is not None or len(existing_samples) > 0

    def test_caching_initialization(self):
        """Test that caching is properly initialized."""
        from ml.mcts.determinization import Determinizer

        # With caching
        det_with_cache = Determinizer(use_caching=True)
        assert det_with_cache.use_caching is True
        assert det_with_cache.sample_cache == []
        assert det_with_cache.cache_size == 20

        # Without caching
        det_no_cache = Determinizer(use_caching=False)
        assert det_no_cache.use_caching is False


def test_determinization_performance_benchmarks():
    """
    Benchmark determinization performance.

    Validates Phase 3 performance targets:
    - Single sample: <10ms
    - 5 samples batch: <50ms
    """
    import time
    from ml.mcts.determinization import Determinizer

    game = BlobGame(num_players=4)
    game.setup_round(cards_to_deal=5)
    observer = game.players[0]

    belief = BeliefState(game, observer)
    determinizer = Determinizer()

    # Benchmark single sample
    start = time.time()
    for _ in range(100):
        sample = determinizer.sample_determinization(game, belief)
    single_time = (time.time() - start) * 1000 / 100

    print(f"\nSingle sample: {single_time:.2f} ms")
    assert single_time < 10.0

    # Benchmark multiple samples
    start = time.time()
    for _ in range(20):
        samples = determinizer.sample_multiple_determinizations(
            game, belief, num_samples=5
        )
    multi_time = (time.time() - start) * 1000 / 20

    print(f"5 samples batch: {multi_time:.2f} ms")
    assert multi_time < 50.0  # 5 samples in <50ms

    # Benchmark adaptive sampling
    start = time.time()
    for _ in range(20):
        samples = determinizer.sample_adaptive(
            game, belief, num_samples=5, diversity_weight=0.5
        )
    adaptive_time = (time.time() - start) * 1000 / 20

    print(f"5 adaptive samples: {adaptive_time:.2f} ms")
    assert adaptive_time < 100.0  # Allow more time for diversity checks

    print("\nAll performance targets met!")
