"""
Unit tests for Blob game engine core classes.

Tests Card, Deck, Player, and Trick classes with comprehensive coverage
of normal operations, edge cases, and anti-cheat functionality.
"""

import pytest
from ml.game.blob import (
    Card,
    Deck,
    Player,
    Trick,
    BlobGame,
    BlobGameException,
    IllegalPlayException,
    InvalidBidException,
    GameStateException,
)
from ml.game.constants import SUITS, RANKS, SCORE_BASE


# ============================================================================
# Test Card Class
# ============================================================================


class TestCard:
    """Test Card class functionality."""

    def test_card_creation(self):
        """Test Card initialization and properties."""
        card = Card("A", "♠")
        assert card.rank == "A"
        assert card.suit == "♠"
        assert card.value == 14  # Ace is highest

    def test_card_value_mapping(self):
        """Test that all ranks map to correct values."""
        assert Card("2", "♠").value == 2
        assert Card("10", "♠").value == 10
        assert Card("J", "♠").value == 11
        assert Card("Q", "♠").value == 12
        assert Card("K", "♠").value == 13
        assert Card("A", "♠").value == 14

    def test_card_invalid_suit(self):
        """Test that invalid suit raises ValueError."""
        with pytest.raises(ValueError, match="Invalid suit"):
            Card("A", "X")

    def test_card_invalid_rank(self):
        """Test that invalid rank raises ValueError."""
        with pytest.raises(ValueError, match="Invalid rank"):
            Card("1", "♠")

    def test_card_equality(self):
        """Test Card comparison."""
        card1 = Card("A", "♠")
        card2 = Card("A", "♠")
        card3 = Card("K", "♠")

        assert card1 == card2
        assert card1 != card3

    def test_card_sorting(self):
        """Test Cards sort by value correctly."""
        cards = [Card("A", "♠"), Card("2", "♠"), Card("K", "♠"), Card("5", "♠")]
        sorted_cards = sorted(cards)

        assert sorted_cards[0].rank == "2"
        assert sorted_cards[1].rank == "5"
        assert sorted_cards[2].rank == "K"
        assert sorted_cards[3].rank == "A"

    def test_card_sorting_by_suit(self):
        """Test Cards with same value sort by suit."""
        cards = [Card("A", "♦"), Card("A", "♠"), Card("A", "♥"), Card("A", "♣")]
        sorted_cards = sorted(cards)

        # Should sort by SUITS order: ♠, ♥, ♣, ♦
        assert sorted_cards[0].suit == "♠"
        assert sorted_cards[1].suit == "♥"
        assert sorted_cards[2].suit == "♣"
        assert sorted_cards[3].suit == "♦"

    def test_card_string_representation(self):
        """Test __str__ and __repr__."""
        card = Card("A", "♠")
        assert str(card) == "A♠"
        assert repr(card) == "Card('A', '♠')"

    def test_card_hashable(self):
        """Test Cards can be used in sets and dicts."""
        card1 = Card("A", "♠")
        card2 = Card("A", "♠")
        card3 = Card("K", "♠")

        card_set = {card1, card2, card3}
        assert len(card_set) == 2  # card1 and card2 are same

        card_dict = {card1: "value"}
        assert card_dict[card2] == "value"  # card2 should match card1

    def test_card_lt_not_implemented(self):
        """Test __lt__ returns NotImplemented for non-Card comparison."""
        card = Card("A", "♠")
        result = card.__lt__("not a card")
        assert result == NotImplemented


# ============================================================================
# Test Deck Class
# ============================================================================


class TestDeck:
    """Test Deck class functionality."""

    def test_deck_initialization(self):
        """Deck has 52 unique cards."""
        deck = Deck()
        assert len(deck.cards) == 52

        # Check all cards are unique
        unique_cards = set(deck.cards)
        assert len(unique_cards) == 52

        # Check all suits and ranks present
        suits_found = set(card.suit for card in deck.cards)
        ranks_found = set(card.rank for card in deck.cards)

        assert suits_found == set(SUITS)
        assert ranks_found == set(RANKS)

    def test_deck_shuffle(self):
        """Shuffling changes order."""
        deck1 = Deck()
        original_order = deck1.cards.copy()

        deck1.shuffle()

        # Very unlikely to have same order after shuffle
        # (probability is 1/52! which is effectively 0)
        assert deck1.cards != original_order

    def test_deck_deal(self):
        """Dealing distributes cards correctly."""
        deck = Deck()
        hands = deck.deal(5, 4)

        assert len(hands) == 4  # 4 players
        assert all(len(hand) == 5 for hand in hands)  # Each has 5 cards

        # Check all dealt cards are unique
        all_dealt = [card for hand in hands for card in hand]
        assert len(all_dealt) == 20
        assert len(set(all_dealt)) == 20

        # Check dealt_cards tracking
        assert len(deck.dealt_cards) == 20

        # Check remaining cards
        assert deck.remaining_cards() == 32

    def test_deck_deal_validation(self):
        """Cannot deal more cards than available."""
        deck = Deck()

        # Try to deal too many cards
        with pytest.raises(ValueError, match="Cannot deal"):
            deck.deal(14, 4)  # 14 * 4 = 56 > 52

    def test_deck_deal_all_cards(self):
        """Can deal all 52 cards."""
        deck = Deck()
        _hands = deck.deal(13, 4)  # noqa: F841 - 13 * 4 = 52

        assert deck.remaining_cards() == 0
        assert len(deck.dealt_cards) == 52

    def test_deck_reset(self):
        """Reset returns all cards."""
        deck = Deck()
        deck.deal(5, 4)

        assert deck.remaining_cards() == 32
        assert len(deck.dealt_cards) == 20

        deck.reset()

        assert deck.remaining_cards() == 52
        assert len(deck.dealt_cards) == 0


# ============================================================================
# Test Player Class
# ============================================================================


class TestPlayer:
    """Test Player class functionality."""

    def test_player_creation(self):
        """Player initializes correctly."""
        player = Player("Alice", 0)

        assert player.name == "Alice"
        assert player.position == 0
        assert player.hand == []
        assert player.bid is None
        assert player.tricks_won == 0
        assert player.total_score == 0
        assert player.known_void_suits == set()
        assert player.cards_played == []

    def test_receive_cards(self):
        """Can receive cards."""
        player = Player("Alice", 0)
        cards = [Card("A", "♠"), Card("K", "♠"), Card("Q", "♠")]

        player.receive_cards(cards)

        assert len(player.hand) == 3
        assert Card("A", "♠") in player.hand

    def test_play_card(self):
        """Can play cards from hand."""
        player = Player("Alice", 0)
        card = Card("A", "♠")
        player.receive_cards([card, Card("K", "♠")])

        played = player.play_card(card)

        assert played == card
        assert card not in player.hand
        assert len(player.hand) == 1
        assert card in player.cards_played

    def test_play_card_not_in_hand(self):
        """Cannot play card not in hand."""
        player = Player("Alice", 0)
        player.receive_cards([Card("K", "♠")])

        with pytest.raises(ValueError, match="does not have"):
            player.play_card(Card("A", "♠"))

    def test_make_bid(self):
        """Can make a bid."""
        player = Player("Alice", 0)
        player.make_bid(3)

        assert player.bid == 3

    def test_win_trick(self):
        """Can win tricks."""
        player = Player("Alice", 0)

        assert player.tricks_won == 0

        player.win_trick()
        assert player.tricks_won == 1

        player.win_trick()
        assert player.tricks_won == 2

    def test_score_calculation_exact_bid(self):
        """Correct score when bid matches tricks."""
        player = Player("Alice", 0)
        player.make_bid(3)
        player.tricks_won = 3

        score = player.calculate_round_score()
        assert score == SCORE_BASE + 3  # 10 + 3 = 13

    def test_score_calculation_missed_bid_over(self):
        """Zero score when won too many tricks."""
        player = Player("Alice", 0)
        player.make_bid(2)
        player.tricks_won = 3

        score = player.calculate_round_score()
        assert score == 0

    def test_score_calculation_missed_bid_under(self):
        """Zero score when won too few tricks."""
        player = Player("Alice", 0)
        player.make_bid(3)
        player.tricks_won = 2

        score = player.calculate_round_score()
        assert score == 0

    def test_score_calculation_zero_bid_success(self):
        """10 points for bidding and making 0."""
        player = Player("Alice", 0)
        player.make_bid(0)
        player.tricks_won = 0

        score = player.calculate_round_score()
        assert score == SCORE_BASE  # 10 points

    def test_score_calculation_no_bid(self):
        """Zero score if no bid made."""
        player = Player("Alice", 0)
        player.tricks_won = 3

        score = player.calculate_round_score()
        assert score == 0

    def test_reset_round(self):
        """Reset clears round-specific state."""
        player = Player("Alice", 0)
        player.receive_cards([Card("A", "♠"), Card("K", "♠")])
        player.make_bid(2)
        player.tricks_won = 1
        player.known_void_suits.add("♥")
        player.cards_played.append(Card("A", "♠"))

        player.reset_round()

        assert player.hand == []
        assert player.bid is None
        assert player.tricks_won == 0
        assert player.known_void_suits == set()
        assert player.cards_played == []

    def test_sort_hand(self):
        """Hand sorts correctly."""
        player = Player("Alice", 0)
        player.receive_cards(
            [Card("A", "♠"), Card("2", "♠"), Card("K", "♠"), Card("5", "♠")]
        )

        player.sort_hand()

        assert player.hand[0].rank == "2"
        assert player.hand[1].rank == "5"
        assert player.hand[2].rank == "K"
        assert player.hand[3].rank == "A"

    def test_mark_void_suit(self):
        """Can mark void suits."""
        player = Player("Alice", 0)

        player.mark_void_suit("♥")
        assert "♥" in player.known_void_suits

        player.mark_void_suit("♣")
        assert "♣" in player.known_void_suits
        assert len(player.known_void_suits) == 2

    def test_has_suit(self):
        """Can check if player has suit."""
        player = Player("Alice", 0)
        player.receive_cards([Card("A", "♠"), Card("K", "♠"), Card("Q", "♥")])

        assert player.has_suit("♠") is True
        assert player.has_suit("♥") is True
        assert player.has_suit("♣") is False
        assert player.has_suit("♦") is False

    def test_player_string_representation(self):
        """Test __str__ and __repr__."""
        player = Player("Alice", 0)
        player.total_score = 25

        s = str(player)
        assert "Alice" in s
        assert "pos=0" in s
        assert "score=25" in s

        # Test __repr__ returns same as __str__
        assert repr(player) == str(player)


# ============================================================================
# Test Trick Class
# ============================================================================


class TestTrick:
    """Test Trick class functionality."""

    def test_trick_initialization(self):
        """Trick initializes correctly."""
        trick = Trick("♠")

        assert trick.trump_suit == "♠"
        assert trick.cards_played == []
        assert trick.led_suit is None
        assert trick.winner is None

    def test_trick_no_trump(self):
        """Can create no-trump trick."""
        trick = Trick(None)
        assert trick.trump_suit is None

    def test_add_card_sets_led_suit(self):
        """First card sets led suit."""
        trick = Trick("♠")
        player = Player("Alice", 0)
        card = Card("A", "♥")

        trick.add_card(player, card)

        assert trick.led_suit == "♥"
        assert len(trick.cards_played) == 1

    def test_trick_winner_no_trump(self):
        """Highest card in led suit wins (no trump)."""
        trick = Trick(None)
        p1 = Player("Alice", 0)
        p2 = Player("Bob", 1)
        p3 = Player("Carol", 2)

        # Hearts led, Alice plays highest
        trick.add_card(p1, Card("A", "♥"))
        trick.add_card(p2, Card("K", "♥"))
        trick.add_card(p3, Card("Q", "♥"))

        winner = trick.determine_winner()
        assert winner == p1

    def test_trick_winner_off_suit_loses(self):
        """Off-suit card cannot win if led suit played."""
        trick = Trick(None)
        p1 = Player("Alice", 0)
        p2 = Player("Bob", 1)
        p3 = Player("Carol", 2)

        # Hearts led
        trick.add_card(p1, Card("2", "♥"))
        trick.add_card(p2, Card("A", "♠"))  # Off-suit, even though Ace
        trick.add_card(p3, Card("3", "♥"))

        winner = trick.determine_winner()
        assert winner == p3  # 3♥ beats 2♥ (A♠ doesn't count)

    def test_trick_winner_with_trump(self):
        """Trump card beats non-trump."""
        trick = Trick("♠")  # Spades are trump
        p1 = Player("Alice", 0)
        p2 = Player("Bob", 1)
        p3 = Player("Carol", 2)

        # Hearts led, but spade trump wins
        trick.add_card(p1, Card("A", "♥"))  # Led suit, high card
        trick.add_card(p2, Card("2", "♠"))  # Trump (lowest spade)
        trick.add_card(p3, Card("K", "♥"))  # Led suit, but not trump

        winner = trick.determine_winner()
        assert winner == p2  # Lowest trump beats highest non-trump

    def test_trick_winner_multiple_trumps(self):
        """Highest trump wins."""
        trick = Trick("♠")
        p1 = Player("Alice", 0)
        p2 = Player("Bob", 1)
        p3 = Player("Carol", 2)

        trick.add_card(p1, Card("5", "♠"))  # Trump
        trick.add_card(p2, Card("K", "♠"))  # Higher trump
        trick.add_card(p3, Card("A", "♥"))  # Not trump

        winner = trick.determine_winner()
        assert winner == p2  # K♠ is highest trump

    def test_trick_winner_trump_is_led_suit(self):
        """Trump suit can be led suit."""
        trick = Trick("♠")
        p1 = Player("Alice", 0)
        p2 = Player("Bob", 1)

        trick.add_card(p1, Card("K", "♠"))  # Led trump
        trick.add_card(p2, Card("A", "♠"))  # Higher trump

        winner = trick.determine_winner()
        assert winner == p2

    def test_determine_winner_no_cards(self):
        """Cannot determine winner with no cards."""
        trick = Trick("♠")

        with pytest.raises(GameStateException, match="no cards played"):
            trick.determine_winner()

    def test_get_winning_card(self):
        """Can get the winning card."""
        trick = Trick("♠")
        p1 = Player("Alice", 0)
        p2 = Player("Bob", 1)

        card1 = Card("K", "♠")
        card2 = Card("A", "♠")

        trick.add_card(p1, card1)
        trick.add_card(p2, card2)

        trick.determine_winner()
        winning_card = trick.get_winning_card()

        assert winning_card == card2

    def test_get_winning_card_before_determine(self):
        """Cannot get winning card before determining winner."""
        trick = Trick("♠")
        p1 = Player("Alice", 0)

        trick.add_card(p1, Card("A", "♠"))

        with pytest.raises(GameStateException, match="Winner not yet determined"):
            trick.get_winning_card()

    def test_is_complete(self):
        """Check if all players have played."""
        trick = Trick("♠")
        p1 = Player("Alice", 0)
        p2 = Player("Bob", 1)
        p3 = Player("Carol", 2)

        assert trick.is_complete(3) is False

        trick.add_card(p1, Card("A", "♠"))
        assert trick.is_complete(3) is False

        trick.add_card(p2, Card("K", "♠"))
        assert trick.is_complete(3) is False

        trick.add_card(p3, Card("Q", "♠"))
        assert trick.is_complete(3) is True

    def test_clear(self):
        """Clear resets trick state."""
        trick = Trick("♠")
        p1 = Player("Alice", 0)

        trick.add_card(p1, Card("A", "♠"))
        trick.determine_winner()

        assert len(trick.cards_played) > 0
        assert trick.led_suit is not None
        assert trick.winner is not None

        trick.clear()

        assert trick.cards_played == []
        assert trick.led_suit is None
        assert trick.winner is None
        assert trick.trump_suit == "♠"  # Trump doesn't change

    def test_trick_string_representation(self):
        """Test __str__ and __repr__."""
        trick = Trick("♠")
        p1 = Player("Alice", 0)

        trick.add_card(p1, Card("A", "♥"))

        s = str(trick)
        assert "led=♥" in s
        assert "trump=♠" in s
        assert "Alice" in s

        # Test __repr__ returns same as __str__
        assert repr(trick) == str(trick)

    def test_trick_winner_no_led_suit_cards(self):
        """Test edge case where no cards in led suit exist (should not happen in valid game)."""
        trick = Trick(None)
        p1 = Player("Alice", 0)

        # Add card directly without going through proper game flow
        # to test the edge case
        trick.cards_played.append((p1, Card("A", "♠")))
        trick.led_suit = "♥"  # Set led suit that doesn't match any cards

        # This should raise exception
        with pytest.raises(GameStateException, match="No cards in led suit"):
            trick.determine_winner()

    def test_get_winning_card_edge_case(self):
        """Test get_winning_card when winner's card somehow not in trick (edge case)."""
        trick = Trick("♠")
        p1 = Player("Alice", 0)
        p2 = Player("Bob", 1)

        trick.add_card(p1, Card("K", "♠"))
        trick.add_card(p2, Card("A", "♠"))

        # Determine winner normally
        winner = trick.determine_winner()
        assert winner == p2

        # Now artificially remove the winning card to test edge case
        trick.cards_played = [(p1, Card("K", "♠"))]  # Only keep p1's card

        # This should raise exception since winner's card is not found
        with pytest.raises(GameStateException, match="Winner's card not found"):
            trick.get_winning_card()


# ============================================================================
# Test Custom Exceptions
# ============================================================================


class TestExceptions:
    """Test custom exception classes."""

    def test_blob_game_exception(self):
        """BlobGameException is base exception."""
        exc = BlobGameException("Test error")
        assert isinstance(exc, Exception)
        assert str(exc) == "Test error"

    def test_illegal_play_exception(self):
        """IllegalPlayException stores context."""
        card = Card("A", "♠")
        exc = IllegalPlayException("Alice", card, "card not in hand")

        assert exc.player_name == "Alice"
        assert exc.card == card
        assert exc.reason == "card not in hand"
        assert "Alice" in str(exc)
        assert "A♠" in str(exc)
        assert "card not in hand" in str(exc)

    def test_invalid_bid_exception(self):
        """InvalidBidException can be raised."""
        exc = InvalidBidException("Invalid bid: 5")
        assert isinstance(exc, BlobGameException)
        assert str(exc) == "Invalid bid: 5"

    def test_game_state_exception(self):
        """GameStateException can be raised."""
        exc = GameStateException("Game not in correct state")
        assert isinstance(exc, BlobGameException)
        assert str(exc) == "Game not in correct state"


# ============================================================================
# Test BlobGame Class
# ============================================================================


class TestBlobGame:
    """Test BlobGame class functionality."""

    def test_game_initialization(self):
        """Game initializes with correct player count and default state."""
        game = BlobGame(num_players=4)

        assert game.num_players == 4
        assert len(game.players) == 4
        assert game.current_round == 0
        assert game.trump_suit is None
        assert game.dealer_position == 0
        assert game.current_trick is None
        assert game.tricks_history == []
        assert game.game_phase == "setup"
        assert game.cards_played_this_round == []
        assert game.cards_remaining_by_suit == {}

        # Check default player names
        assert game.players[0].name == "Player 1"
        assert game.players[1].name == "Player 2"
        assert game.players[2].name == "Player 3"
        assert game.players[3].name == "Player 4"

        # Check player positions
        assert game.players[0].position == 0
        assert game.players[1].position == 1
        assert game.players[2].position == 2
        assert game.players[3].position == 3

        # Check deck initialization
        assert isinstance(game.deck, Deck)
        assert game.deck.remaining_cards() == 52

    def test_game_initialization_invalid_player_count(self):
        """Game initialization raises ValueError for invalid player counts."""
        # Too few players
        with pytest.raises(ValueError, match="must be between"):
            BlobGame(num_players=2)

        # Too many players
        with pytest.raises(ValueError, match="must be between"):
            BlobGame(num_players=9)

        # Zero players
        with pytest.raises(ValueError, match="must be between"):
            BlobGame(num_players=0)

        # Negative players
        with pytest.raises(ValueError, match="must be between"):
            BlobGame(num_players=-1)

    def test_game_initialization_custom_names(self):
        """Game initializes with custom player names."""
        names = ["Alice", "Bob", "Carol"]
        game = BlobGame(num_players=3, player_names=names)

        assert game.num_players == 3
        assert len(game.players) == 3
        assert game.players[0].name == "Alice"
        assert game.players[1].name == "Bob"
        assert game.players[2].name == "Carol"

    def test_game_initialization_mismatched_names(self):
        """Game initialization raises ValueError if player_names length mismatches."""
        # Too few names
        with pytest.raises(ValueError, match="must match"):
            BlobGame(num_players=4, player_names=["Alice", "Bob"])

        # Too many names
        with pytest.raises(ValueError, match="must match"):
            BlobGame(num_players=3, player_names=["Alice", "Bob", "Carol", "Dave"])

    def test_trump_rotation(self):
        """Trump cycles through suits correctly."""
        game = BlobGame(num_players=4)

        # Round 0: Spades
        game.current_round = 0
        assert game.determine_trump() == "♠"

        # Round 1: Hearts
        game.current_round = 1
        assert game.determine_trump() == "♥"

        # Round 2: Clubs
        game.current_round = 2
        assert game.determine_trump() == "♣"

        # Round 3: Diamonds
        game.current_round = 3
        assert game.determine_trump() == "♦"

        # Round 4: No trump
        game.current_round = 4
        assert game.determine_trump() is None

        # Round 5: Cycle repeats - Spades
        game.current_round = 5
        assert game.determine_trump() == "♠"

        # Round 10: Hearts (10 % 5 = 0 → ♠, wait no... 10 % 5 = 0)
        game.current_round = 10
        assert game.determine_trump() == "♠"

        # Round 11: Hearts
        game.current_round = 11
        assert game.determine_trump() == "♥"

    def test_setup_round(self):
        """setup_round() correctly prepares game for a round."""
        game = BlobGame(num_players=4)

        # Setup a round with 5 cards each
        game.setup_round(5)

        # Check trump is set
        assert game.trump_suit == "♠"  # Round 0 → Spades

        # Check game phase
        assert game.game_phase == "bidding"

        # Check all players have 5 cards
        for player in game.players:
            assert len(player.hand) == 5

        # Check cards are sorted
        for player in game.players:
            sorted_hand = sorted(player.hand)
            assert player.hand == sorted_hand

        # Check card counting initialized
        total_cards_counted = sum(game.cards_remaining_by_suit.values())
        assert total_cards_counted == 20  # 4 players × 5 cards

        # Check all suits tracked
        assert set(game.cards_remaining_by_suit.keys()) == set(SUITS)

        # Check trick state reset
        assert game.current_trick is None
        assert game.tricks_history == []
        assert game.cards_played_this_round == []

    def test_setup_round_resets_state(self):
        """setup_round() resets previous round state."""
        game = BlobGame(num_players=3, player_names=["Alice", "Bob", "Carol"])

        # Setup first round
        game.setup_round(5)

        # Simulate some game state changes
        game.players[0].make_bid(2)
        game.players[0].tricks_won = 1
        game.players[0].known_void_suits.add("♥")
        game.cards_played_this_round.append(Card("A", "♠"))

        # Setup second round
        game.current_round = 1
        game.setup_round(4)

        # Check player state is reset
        for player in game.players:
            assert player.bid is None
            assert player.tricks_won == 0
            assert player.known_void_suits == set()
            assert player.cards_played == []
            assert len(player.hand) == 4  # New cards dealt

        # Check trump rotated
        assert game.trump_suit == "♥"  # Round 1 → Hearts

        # Check card tracking reset
        assert game.cards_played_this_round == []
        total_cards = sum(game.cards_remaining_by_suit.values())
        assert total_cards == 12  # 3 players × 4 cards

        # Check game phase
        assert game.game_phase == "bidding"

    def test_setup_round_different_player_counts(self):
        """setup_round() works with different player counts."""
        # 3 players, 7 cards
        game3 = BlobGame(num_players=3)
        game3.setup_round(7)
        assert all(len(p.hand) == 7 for p in game3.players)
        assert sum(game3.cards_remaining_by_suit.values()) == 21

        # 8 players, 6 cards
        game8 = BlobGame(num_players=8)
        game8.setup_round(6)
        assert all(len(p.hand) == 6 for p in game8.players)
        assert sum(game8.cards_remaining_by_suit.values()) == 48

    def test_setup_round_max_cards(self):
        """setup_round() handles maximum card dealing."""
        # 4 players, 13 cards each = 52 cards (full deck)
        game = BlobGame(num_players=4)
        game.setup_round(13)

        assert all(len(p.hand) == 13 for p in game.players)
        assert sum(game.cards_remaining_by_suit.values()) == 52
        assert game.deck.remaining_cards() == 0

    def test_setup_round_too_many_cards(self):
        """setup_round() raises ValueError if too many cards requested."""
        game = BlobGame(num_players=4)

        # Try to deal 14 cards to 4 players (56 cards needed, only 52 available)
        with pytest.raises(ValueError, match="Cannot deal"):
            game.setup_round(14)

    # ============================================================================
    # Test Bidding Phase
    # ============================================================================

    def test_get_forbidden_bid_basic(self):
        """Forbidden bid calculation works correctly."""
        game = BlobGame(num_players=3)

        # 5 cards dealt, others bid total of 3 → forbidden is 2
        forbidden = game.get_forbidden_bid(current_total_bids=3, cards_dealt=5)
        assert forbidden == 2

        # 5 cards dealt, others bid total of 0 → forbidden is 5
        forbidden = game.get_forbidden_bid(current_total_bids=0, cards_dealt=5)
        assert forbidden == 5

        # 5 cards dealt, others bid total of 5 → forbidden is 0
        forbidden = game.get_forbidden_bid(current_total_bids=5, cards_dealt=5)
        assert forbidden == 0

    def test_get_forbidden_bid_out_of_range_negative(self):
        """Forbidden bid returns None if negative."""
        game = BlobGame(num_players=3)

        # Others bid more than cards dealt → forbidden would be negative
        forbidden = game.get_forbidden_bid(current_total_bids=6, cards_dealt=5)
        assert forbidden is None

    def test_get_forbidden_bid_out_of_range_too_high(self):
        """Forbidden bid returns None if greater than cards_dealt."""
        game = BlobGame(num_players=3)

        # Others bid negative (shouldn't happen, but test boundary)
        forbidden = game.get_forbidden_bid(current_total_bids=-2, cards_dealt=5)
        assert forbidden is None  # Would be 7, which is > 5

    def test_get_forbidden_bid_edge_cases(self):
        """Test forbidden bid edge cases."""
        game = BlobGame(num_players=4)

        # 1 card dealt, others bid 0 → forbidden is 1
        assert game.get_forbidden_bid(0, 1) == 1

        # 1 card dealt, others bid 1 → forbidden is 0
        assert game.get_forbidden_bid(1, 1) == 0

        # 1 card dealt, others bid 2 → forbidden is None (would be -1)
        assert game.get_forbidden_bid(2, 1) is None

    def test_is_valid_bid_non_dealer(self):
        """Non-dealer can bid any value in range."""
        game = BlobGame(num_players=4)

        # All bids in range [0, cards_dealt] are valid for non-dealer
        for bid in range(6):  # 0 to 5
            assert game.is_valid_bid(
                bid, is_dealer=False, current_total_bids=0, cards_dealt=5
            )

    def test_is_valid_bid_non_dealer_out_of_range(self):
        """Non-dealer cannot bid out of range."""
        game = BlobGame(num_players=4)

        # Negative bid invalid
        assert (
            game.is_valid_bid(-1, is_dealer=False, current_total_bids=0, cards_dealt=5)
            is False
        )

        # Too high bid invalid
        assert (
            game.is_valid_bid(6, is_dealer=False, current_total_bids=0, cards_dealt=5)
            is False
        )

    def test_is_valid_bid_dealer_forbidden(self):
        """Dealer cannot bid forbidden value."""
        game = BlobGame(num_players=3)

        # 5 cards dealt, others bid 3 → forbidden is 2
        # Dealer can bid 0, 1, 3, 4, 5 but NOT 2
        assert (
            game.is_valid_bid(0, is_dealer=True, current_total_bids=3, cards_dealt=5)
            is True
        )
        assert (
            game.is_valid_bid(1, is_dealer=True, current_total_bids=3, cards_dealt=5)
            is True
        )
        assert (
            game.is_valid_bid(2, is_dealer=True, current_total_bids=3, cards_dealt=5)
            is False
        )  # Forbidden!
        assert (
            game.is_valid_bid(3, is_dealer=True, current_total_bids=3, cards_dealt=5)
            is True
        )
        assert (
            game.is_valid_bid(4, is_dealer=True, current_total_bids=3, cards_dealt=5)
            is True
        )
        assert (
            game.is_valid_bid(5, is_dealer=True, current_total_bids=3, cards_dealt=5)
            is True
        )

    def test_is_valid_bid_dealer_forbidden_zero(self):
        """Dealer cannot bid 0 if that's the forbidden value."""
        game = BlobGame(num_players=3)

        # 5 cards dealt, others bid 5 → forbidden is 0
        assert (
            game.is_valid_bid(0, is_dealer=True, current_total_bids=5, cards_dealt=5)
            is False
        )  # Forbidden!
        assert (
            game.is_valid_bid(1, is_dealer=True, current_total_bids=5, cards_dealt=5)
            is True
        )

    def test_is_valid_bid_dealer_forbidden_max(self):
        """Dealer cannot bid max if that's the forbidden value."""
        game = BlobGame(num_players=3)

        # 5 cards dealt, others bid 0 → forbidden is 5
        assert (
            game.is_valid_bid(5, is_dealer=True, current_total_bids=0, cards_dealt=5)
            is False
        )  # Forbidden!
        assert (
            game.is_valid_bid(4, is_dealer=True, current_total_bids=0, cards_dealt=5)
            is True
        )

    def test_is_valid_bid_dealer_no_forbidden(self):
        """Dealer has no forbidden bid if calculation is out of range."""
        game = BlobGame(num_players=3)

        # 5 cards dealt, others bid 6 → forbidden would be -1 (invalid), so all bids in range are valid
        assert (
            game.is_valid_bid(0, is_dealer=True, current_total_bids=6, cards_dealt=5)
            is True
        )
        assert (
            game.is_valid_bid(5, is_dealer=True, current_total_bids=6, cards_dealt=5)
            is True
        )

    def test_is_valid_bid_dealer_out_of_range(self):
        """Dealer still cannot bid out of range."""
        game = BlobGame(num_players=3)

        # Even if not forbidden, out of range bids are invalid
        assert (
            game.is_valid_bid(-1, is_dealer=True, current_total_bids=3, cards_dealt=5)
            is False
        )
        assert (
            game.is_valid_bid(6, is_dealer=True, current_total_bids=3, cards_dealt=5)
            is False
        )

    def test_bidding_phase_wrong_game_state(self):
        """bidding_phase() raises exception if not in bidding phase."""
        game = BlobGame(num_players=3)

        # Game starts in 'setup' phase
        assert game.game_phase == "setup"

        with pytest.raises(GameStateException, match="Cannot start bidding phase"):
            game.bidding_phase()

    def test_bidding_phase_structure(self):
        """bidding_phase() has correct structure and raises NotImplementedError."""
        game = BlobGame(num_players=3)
        game.setup_round(5)

        # Game should be in bidding phase after setup
        assert game.game_phase == "bidding"

        # Method should raise NotImplementedError since no bid mechanism provided
        with pytest.raises(NotImplementedError, match="bid selection mechanism"):
            game.bidding_phase()

    # ============================================================================
    # Test Playing Phase
    # ============================================================================

    def test_get_legal_plays_first_card(self):
        """First card of trick can be any card in hand."""
        game = BlobGame(num_players=3)
        game.setup_round(5)
        player = game.players[0]

        # Led suit is None (first card)
        legal_plays = game.get_legal_plays(player, led_suit=None)

        # Should be able to play any card
        assert len(legal_plays) == 5
        assert set(legal_plays) == set(player.hand)

    def test_get_legal_plays_must_follow_suit(self):
        """Player must follow suit if they have cards in that suit."""
        game = BlobGame(num_players=3)
        player = game.players[0]

        # Give player specific cards
        player.receive_cards(
            [Card("5", "♥"), Card("8", "♥"), Card("3", "♣"), Card("K", "♦")]
        )

        # Led suit is hearts
        legal_plays = game.get_legal_plays(player, led_suit="♥")

        # Should only be able to play hearts
        assert len(legal_plays) == 2
        assert Card("5", "♥") in legal_plays
        assert Card("8", "♥") in legal_plays

    def test_get_legal_plays_no_led_suit(self):
        """Player can play any card if they don't have led suit."""
        game = BlobGame(num_players=3)
        player = game.players[0]

        # Give player cards with no hearts
        player.receive_cards([Card("3", "♣"), Card("K", "♦"), Card("7", "♠")])

        # Led suit is hearts (player has none)
        legal_plays = game.get_legal_plays(player, led_suit="♥")

        # Should be able to play any card
        assert len(legal_plays) == 3
        assert set(legal_plays) == set(player.hand)

    def test_is_valid_play_card_in_hand(self):
        """is_valid_play() returns True for valid play."""
        game = BlobGame(num_players=3)
        player = game.players[0]

        card = Card("5", "♥")
        player.receive_cards([card, Card("3", "♣")])

        # First card (no led suit)
        assert game.is_valid_play(card, player, led_suit=None) is True

    def test_is_valid_play_card_not_in_hand(self):
        """is_valid_play() returns False for card not in hand."""
        game = BlobGame(num_players=3)
        player = game.players[0]

        player.receive_cards([Card("3", "♣")])

        # Try to play card not in hand
        assert game.is_valid_play(Card("5", "♥"), player, led_suit=None) is False

    def test_is_valid_play_must_follow_suit(self):
        """is_valid_play() enforces follow-suit rule."""
        game = BlobGame(num_players=3)
        player = game.players[0]

        player.receive_cards([Card("5", "♥"), Card("3", "♣")])

        # Led suit is hearts, trying to play clubs
        assert game.is_valid_play(Card("3", "♣"), player, led_suit="♥") is False

        # Led suit is hearts, playing hearts
        assert game.is_valid_play(Card("5", "♥"), player, led_suit="♥") is True

    def test_is_valid_play_no_led_suit_can_play_any(self):
        """is_valid_play() allows any card if player has no led suit."""
        game = BlobGame(num_players=3)
        player = game.players[0]

        player.receive_cards([Card("3", "♣"), Card("K", "♦")])

        # Led suit is hearts (player has none), can play clubs
        assert game.is_valid_play(Card("3", "♣"), player, led_suit="♥") is True

    def test_validate_play_with_anti_cheat_valid_play(self):
        """validate_play_with_anti_cheat() allows valid plays."""
        game = BlobGame(num_players=3)
        player = game.players[0]

        card = Card("5", "♥")
        player.receive_cards([card, Card("3", "♣")])

        # Should not raise exception for valid play
        game.validate_play_with_anti_cheat(card, player, led_suit=None)

    def test_validate_play_with_anti_cheat_card_not_in_hand(self):
        """validate_play_with_anti_cheat() raises exception for card not in hand."""
        game = BlobGame(num_players=3)
        player = game.players[0]

        player.receive_cards([Card("3", "♣")])

        # Should raise IllegalPlayException
        with pytest.raises(IllegalPlayException, match="card not in hand"):
            game.validate_play_with_anti_cheat(Card("5", "♥"), player, led_suit=None)

    def test_validate_play_with_anti_cheat_illegal_suit_violation(self):
        """validate_play_with_anti_cheat() detects illegal suit violations."""
        game = BlobGame(num_players=3)
        player = game.players[0]

        player.receive_cards([Card("5", "♥"), Card("3", "♣")])

        # Player has hearts but tries to play clubs when hearts led
        with pytest.raises(IllegalPlayException, match="must follow suit"):
            game.validate_play_with_anti_cheat(Card("3", "♣"), player, led_suit="♥")

    def test_validate_play_with_anti_cheat_valid_different_suit(self):
        """validate_play_with_anti_cheat() allows different suit if player has none of led suit."""
        game = BlobGame(num_players=3)
        player = game.players[0]

        player.receive_cards([Card("3", "♣"), Card("K", "♦")])

        # Player has no hearts, can play clubs
        # Should not raise exception
        game.validate_play_with_anti_cheat(Card("3", "♣"), player, led_suit="♥")

    def test_update_card_counting_adds_to_history(self):
        """update_card_counting() adds card to round history."""
        game = BlobGame(num_players=3)
        game.setup_round(5)
        player = game.players[0]

        card = player.hand[0]

        assert len(game.cards_played_this_round) == 0

        game.update_card_counting(card, player, led_suit=None)

        assert len(game.cards_played_this_round) == 1
        assert card in game.cards_played_this_round

    def test_update_card_counting_decrements_suit_count(self):
        """update_card_counting() decrements cards_remaining_by_suit."""
        game = BlobGame(num_players=3)
        game.setup_round(5)
        player = game.players[0]

        # Find a heart in player's hand
        heart_card = None
        for card in player.hand:
            if card.suit == "♥":
                heart_card = card
                break

        if heart_card:
            initial_hearts = game.cards_remaining_by_suit["♥"]
            game.update_card_counting(heart_card, player, led_suit=None)
            assert game.cards_remaining_by_suit["♥"] == initial_hearts - 1

    def test_update_card_counting_marks_void_suit(self):
        """update_card_counting() marks player void when they don't follow suit."""
        game = BlobGame(num_players=3)
        game.setup_round(5)  # Initialize card counting state
        player = game.players[0]

        # Replace player's hand with specific card
        player.hand = [Card("3", "♣")]

        assert "♥" not in player.known_void_suits

        # Player plays clubs when hearts led (they have no hearts)
        game.update_card_counting(Card("3", "♣"), player, led_suit="♥")

        # Player should be marked void in hearts
        assert "♥" in player.known_void_suits

    def test_update_card_counting_no_void_when_following_suit(self):
        """update_card_counting() doesn't mark void when following suit."""
        game = BlobGame(num_players=3)
        game.setup_round(5)
        player = game.players[0]

        card = Card("5", "♥")
        player.receive_cards([card])

        assert len(player.known_void_suits) == 0

        # Player plays hearts when hearts led
        game.update_card_counting(card, player, led_suit="♥")

        # Player should NOT be marked void in hearts
        assert "♥" not in player.known_void_suits

    def test_play_trick_wrong_phase(self):
        """play_trick() raises exception if not in playing phase."""
        game = BlobGame(num_players=3)
        game.setup_round(5)

        # Game is in 'bidding' phase
        assert game.game_phase == "bidding"

        with pytest.raises(GameStateException, match="Cannot play trick"):
            game.play_trick()

    def test_play_trick_structure(self):
        """play_trick() has correct structure and raises NotImplementedError."""
        game = BlobGame(num_players=3)
        game.setup_round(5)

        # Transition to playing phase
        game.game_phase = "playing"

        # Method should raise NotImplementedError since no card selection mechanism
        with pytest.raises(NotImplementedError, match="card selection mechanism"):
            game.play_trick()

    def test_playing_phase_wrong_phase(self):
        """playing_phase() raises exception if not in playing phase."""
        game = BlobGame(num_players=3)
        game.setup_round(5)

        # Game is in 'bidding' phase
        assert game.game_phase == "bidding"

        with pytest.raises(GameStateException, match="Cannot start playing phase"):
            game.playing_phase()

    def test_playing_phase_structure(self):
        """playing_phase() has correct structure."""
        game = BlobGame(num_players=3)
        game.setup_round(5)

        # Transition to playing phase
        game.game_phase = "playing"

        # Should try to play tricks (will fail due to NotImplementedError in play_trick)
        with pytest.raises(NotImplementedError):
            game.playing_phase()

    def test_scoring_phase_basic(self):
        """scoring_phase() calculates and updates scores correctly."""
        game = BlobGame(num_players=3)
        game.setup_round(5)

        # Manually set up player states for scoring
        game.players[0].bid = 2
        game.players[0].tricks_won = 2  # Made bid

        game.players[1].bid = 1
        game.players[1].tricks_won = 0  # Missed bid

        game.players[2].bid = 3
        game.players[2].tricks_won = 3  # Made bid

        # Transition to scoring phase
        game.game_phase = "scoring"

        # Execute scoring
        results = game.scoring_phase()

        # Verify results structure
        assert "round" in results
        assert "trump_suit" in results
        assert "player_scores" in results
        assert results["round"] == 0

        # Verify player scores
        assert len(results["player_scores"]) == 3

        # Player 0: bid 2, won 2 → 10 + 2 = 12 points
        assert results["player_scores"][0]["name"] == "Player 1"
        assert results["player_scores"][0]["bid"] == 2
        assert results["player_scores"][0]["tricks_won"] == 2
        assert results["player_scores"][0]["round_score"] == 12
        assert results["player_scores"][0]["total_score"] == 12
        assert game.players[0].total_score == 12

        # Player 1: bid 1, won 0 → 0 points
        assert results["player_scores"][1]["name"] == "Player 2"
        assert results["player_scores"][1]["bid"] == 1
        assert results["player_scores"][1]["tricks_won"] == 0
        assert results["player_scores"][1]["round_score"] == 0
        assert results["player_scores"][1]["total_score"] == 0
        assert game.players[1].total_score == 0

        # Player 2: bid 3, won 3 → 10 + 3 = 13 points
        assert results["player_scores"][2]["name"] == "Player 3"
        assert results["player_scores"][2]["bid"] == 3
        assert results["player_scores"][2]["tricks_won"] == 3
        assert results["player_scores"][2]["round_score"] == 13
        assert results["player_scores"][2]["total_score"] == 13
        assert game.players[2].total_score == 13

        # Verify game state transitions
        assert game.game_phase == "complete"
        assert game.current_round == 1  # Incremented
        assert game.dealer_position == 1  # Rotated

    def test_scoring_phase_all_players_make_bid(self):
        """scoring_phase() handles all players making their bids."""
        game = BlobGame(num_players=4)
        game.setup_round(3)

        # All players make their bids
        for i, player in enumerate(game.players):
            player.bid = i  # 0, 1, 2, 3
            player.tricks_won = i

        game.game_phase = "scoring"
        results = game.scoring_phase()

        # Verify all players scored
        expected_scores = [10, 11, 12, 13]  # 10+0, 10+1, 10+2, 10+3
        for i, player_result in enumerate(results["player_scores"]):
            assert player_result["round_score"] == expected_scores[i]
            assert player_result["total_score"] == expected_scores[i]

    def test_scoring_phase_no_players_make_bid(self):
        """scoring_phase() handles no players making their bids."""
        game = BlobGame(num_players=4)
        game.setup_round(3)

        # All players miss their bids
        game.players[0].bid = 2
        game.players[0].tricks_won = 1  # Missed

        game.players[1].bid = 1
        game.players[1].tricks_won = 2  # Missed

        game.players[2].bid = 0
        game.players[2].tricks_won = 1  # Missed

        game.players[3].bid = 3
        game.players[3].tricks_won = 2  # Missed

        game.game_phase = "scoring"
        results = game.scoring_phase()

        # Verify all players scored 0
        for player_result in results["player_scores"]:
            assert player_result["round_score"] == 0
            assert player_result["total_score"] == 0

    def test_scoring_phase_zero_bid_success(self):
        """scoring_phase() correctly scores player who bid 0 and won 0."""
        game = BlobGame(num_players=3)
        game.setup_round(5)

        # Player bids 0 and wins 0 tricks
        game.players[0].bid = 0
        game.players[0].tricks_won = 0  # Made bid!

        game.players[1].bid = 2
        game.players[1].tricks_won = 2

        game.players[2].bid = 3
        game.players[2].tricks_won = 3

        game.game_phase = "scoring"
        results = game.scoring_phase()

        # Player 0: bid 0, won 0 → 10 + 0 = 10 points
        assert results["player_scores"][0]["round_score"] == 10
        assert game.players[0].total_score == 10

    def test_scoring_phase_wrong_phase(self):
        """scoring_phase() raises exception if not in scoring phase."""
        game = BlobGame(num_players=3)
        game.setup_round(5)

        # Game is in 'bidding' phase
        assert game.game_phase == "bidding"

        with pytest.raises(GameStateException, match="Cannot start scoring phase"):
            game.scoring_phase()

    def test_scoring_phase_dealer_rotation(self):
        """scoring_phase() rotates dealer position correctly."""
        game = BlobGame(num_players=4)
        game.setup_round(3)

        # Set initial dealer
        game.dealer_position = 0

        # Set up minimal scoring state
        for player in game.players:
            player.bid = 0
            player.tricks_won = 0

        game.game_phase = "scoring"
        game.scoring_phase()

        # Dealer should have rotated to position 1
        assert game.dealer_position == 1

        # After another round, should rotate to 2
        game.game_phase = "scoring"
        game.scoring_phase()
        assert game.dealer_position == 2

        # After another round, should rotate to 3
        game.game_phase = "scoring"
        game.scoring_phase()
        assert game.dealer_position == 3

        # After another round, should wrap to 0
        game.game_phase = "scoring"
        game.scoring_phase()
        assert game.dealer_position == 0

    def test_scoring_phase_round_counter_increment(self):
        """scoring_phase() increments round counter correctly."""
        game = BlobGame(num_players=3)
        game.setup_round(5)

        # Set up minimal scoring state
        for player in game.players:
            player.bid = 1
            player.tricks_won = 1

        assert game.current_round == 0

        game.game_phase = "scoring"
        game.scoring_phase()

        assert game.current_round == 1

        # Another round
        game.game_phase = "scoring"
        game.scoring_phase()

        assert game.current_round == 2

    def test_scoring_phase_cumulative_scores(self):
        """scoring_phase() accumulates scores across multiple rounds."""
        game = BlobGame(num_players=3)

        # Round 1
        game.setup_round(3)
        game.players[0].bid = 2
        game.players[0].tricks_won = 2
        game.game_phase = "scoring"
        game.scoring_phase()

        # Player 0 should have 12 points
        assert game.players[0].total_score == 12

        # Round 2
        game.setup_round(3)
        game.players[0].bid = 1
        game.players[0].tricks_won = 1
        game.game_phase = "scoring"
        game.scoring_phase()

        # Player 0 should now have 12 + 11 = 23 points
        assert game.players[0].total_score == 23


# ============================================================================
# Test Round Structure Generation
# ============================================================================


class TestRoundStructure:
    """Test generate_round_structure function from constants."""

    def test_generate_round_structure_basic(self):
        """Generate correct round structure for valid inputs."""
        from ml.game.constants import generate_round_structure

        # 4 players, 5 cards: should generate proper structure
        structure = generate_round_structure(5, 4)

        # Should start at 5, descend to 1, have 4 one-card rounds, ascend back to 5
        expected = [5, 4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4, 5]
        assert structure == expected

    def test_generate_round_structure_3_players(self):
        """3 players, 7 cards: [7,6,5,4,3,2,1,1,1,2,3,4,5,6,7]"""
        from ml.game.constants import generate_round_structure

        structure = generate_round_structure(7, 3)
        expected = [7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7]
        assert structure == expected

    def test_generate_round_structure_4_players(self):
        """4 players, 5 cards: [5,4,3,2,1,1,1,1,1,2,3,4,5]"""
        from ml.game.constants import generate_round_structure

        structure = generate_round_structure(5, 4)
        expected = [5, 4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4, 5]
        assert structure == expected

    def test_generate_round_structure_exceeds_deck(self):
        """Raise ValueError if starting_cards * num_players > 52."""
        from ml.game.constants import generate_round_structure

        # 8 players × 7 cards = 56 > 52
        with pytest.raises(ValueError, match="Cannot deal"):
            generate_round_structure(7, 8)

        # 4 players × 14 cards = 56 > 52
        with pytest.raises(ValueError, match="Cannot deal"):
            generate_round_structure(14, 4)

    def test_generate_round_structure_max_valid(self):
        """Find max starting cards for each player count."""
        from ml.game.constants import generate_round_structure

        # 3 players: max 17 cards (17*3=51)
        structure = generate_round_structure(17, 3)
        assert structure[0] == 17

        # 4 players: max 13 cards (13*4=52)
        structure = generate_round_structure(13, 4)
        assert structure[0] == 13

        # 6 players: max 8 cards (8*6=48)
        structure = generate_round_structure(8, 6)
        assert structure[0] == 8

    def test_generate_round_structure_invalid_player_count(self):
        """Raise ValueError for invalid player counts."""
        from ml.game.constants import generate_round_structure

        # Too few players
        with pytest.raises(ValueError, match="num_players must be between"):
            generate_round_structure(5, 2)

        # Too many players
        with pytest.raises(ValueError, match="num_players must be between"):
            generate_round_structure(5, 9)

    def test_generate_round_structure_invalid_starting_cards(self):
        """Raise ValueError for invalid starting_cards."""
        from ml.game.constants import generate_round_structure

        # Zero cards
        with pytest.raises(ValueError, match="starting_cards must be at least 1"):
            generate_round_structure(0, 4)

        # Negative cards
        with pytest.raises(ValueError, match="starting_cards must be at least 1"):
            generate_round_structure(-1, 4)


# ============================================================================
# Test Anti-Cheat System
# ============================================================================


class TestAntiCheat:
    """Test anti-cheat detection and card counting."""

    def test_detect_illegal_card_not_in_hand(self):
        """Raise exception if player plays card they don't have."""
        game = BlobGame(num_players=3)
        player = game.players[0]

        player.receive_cards([Card("3", "♣")])

        # Try to play a card not in hand
        with pytest.raises(IllegalPlayException) as exc_info:
            game.validate_play_with_anti_cheat(Card("A", "♠"), player, led_suit=None)

        assert exc_info.value.player_name == player.name
        assert exc_info.value.card == Card("A", "♠")
        assert "card not in hand" in exc_info.value.reason

    def test_detect_illegal_suit_violation(self):
        """Raise exception if player has led suit but plays different suit."""
        game = BlobGame(num_players=3)
        player = game.players[0]

        player.receive_cards([Card("5", "♥"), Card("8", "♥"), Card("3", "♣")])

        # Player has hearts but tries to play clubs when hearts led
        with pytest.raises(IllegalPlayException) as exc_info:
            game.validate_play_with_anti_cheat(Card("3", "♣"), player, led_suit="♥")

        assert exc_info.value.player_name == player.name
        assert exc_info.value.card == Card("3", "♣")
        assert "must follow suit" in exc_info.value.reason
        assert "♥" in exc_info.value.reason

    def test_suit_elimination_tracking(self):
        """Mark player void in suit when they don't follow suit."""
        game = BlobGame(num_players=3)
        game.setup_round(5)
        player = game.players[0]

        # Replace player's hand with specific cards (no hearts)
        player.hand = [Card("3", "♣"), Card("K", "♦")]

        assert "♥" not in player.known_void_suits

        # Player plays clubs when hearts led
        game.update_card_counting(Card("3", "♣"), player, led_suit="♥")

        # Player should now be marked void in hearts
        assert "♥" in player.known_void_suits

    def test_suit_elimination_multiple_suits(self):
        """Track multiple void suits."""
        game = BlobGame(num_players=3)
        game.setup_round(5)  # Initialize card counting state
        player = game.players[0]

        # Replace hand with specific cards
        player.hand = [Card("A", "♠"), Card("K", "♣")]

        # Player plays spades when hearts led
        game.update_card_counting(Card("A", "♠"), player, led_suit="♥")
        assert "♥" in player.known_void_suits

        # Player plays clubs when diamonds led
        game.update_card_counting(Card("K", "♣"), player, led_suit="♦")
        assert "♦" in player.known_void_suits

        # Should have both hearts and diamonds marked as void
        assert len(player.known_void_suits) == 2
        assert "♥" in player.known_void_suits
        assert "♦" in player.known_void_suits

    def test_card_counting_updates(self):
        """cards_remaining_by_suit updates correctly after plays."""
        game = BlobGame(num_players=3)
        game.setup_round(5)

        # Get initial suit counts
        initial_hearts = game.cards_remaining_by_suit["♥"]
        initial_clubs = game.cards_remaining_by_suit["♣"]

        # Find a heart and a club in players' hands
        heart_card = None
        club_card = None

        for player in game.players:
            for card in player.hand:
                if card.suit == "♥" and heart_card is None:
                    heart_card = (player, card)
                if card.suit == "♣" and club_card is None:
                    club_card = (player, card)

        # Play a heart
        if heart_card:
            player, card = heart_card
            game.update_card_counting(card, player, led_suit=None)
            assert game.cards_remaining_by_suit["♥"] == initial_hearts - 1

        # Play a club
        if club_card:
            player, card = club_card
            game.update_card_counting(card, player, led_suit="♥")
            assert game.cards_remaining_by_suit["♣"] == initial_clubs - 1

    def test_valid_play_different_suit_when_void(self):
        """Allow different suit if player has none of led suit."""
        game = BlobGame(num_players=3)
        player = game.players[0]

        # Player has no hearts
        player.receive_cards([Card("3", "♣"), Card("K", "♦"), Card("7", "♠")])

        # All cards should be valid when hearts led (player has no hearts)
        for card in player.hand:
            # Should not raise exception
            game.validate_play_with_anti_cheat(card, player, led_suit="♥")

    def test_known_void_suits_prevent_false_positives(self):
        """Don't raise exception if player is known to be void in suit."""
        game = BlobGame(num_players=3)
        player = game.players[0]

        player.receive_cards([Card("3", "♣")])

        # Mark player as void in hearts (from previous play)
        player.mark_void_suit("♥")

        # Player playing clubs when hearts led should not raise exception
        # (already known they have no hearts)
        game.validate_play_with_anti_cheat(Card("3", "♣"), player, led_suit="♥")


# ============================================================================
# Test Game Flow Methods
# ============================================================================


class TestGameFlow:
    """Test high-level game flow orchestration methods."""

    def simple_bid_always_zero(
        self, player, cards_dealt, is_dealer, current_total_bids, cards_dealt_param
    ):
        """Bot that always bids 0 (unless dealer and 0 is forbidden)."""
        if is_dealer:
            forbidden = cards_dealt - current_total_bids
            if forbidden == 0:
                return 1  # Bid 1 if 0 is forbidden
        return 0

    def simple_play_first_legal(self, player, legal_cards, trick):
        """Bot that always plays first legal card."""
        return legal_cards[0]

    def test_play_round_complete_flow(self):
        """play_round() executes full round flow successfully."""
        game = BlobGame(num_players=3)

        # Play a round with simple bot functions
        result = game.play_round(
            cards_to_deal=5,
            get_bid_func=self.simple_bid_always_zero,
            get_card_func=self.simple_play_first_legal,
        )

        # Verify result structure
        assert "round" in result
        assert "trump_suit" in result
        assert "player_scores" in result
        assert result["round"] == 0
        assert len(result["player_scores"]) == 3

        # Verify game state after round
        assert game.game_phase == "complete"
        assert game.current_round == 1  # Incremented
        assert len(game.tricks_history) == 5  # 5 tricks played

        # Verify all players have bids
        for player in game.players:
            assert player.bid is not None
            # Players bid 0 in simple strategy (unless forbidden)
            # Tricks won can vary based on card play

    def test_play_round_scores_calculated(self):
        """play_round() calculates scores correctly."""

        def controlled_bid(
            player, cards_dealt, is_dealer, current_total_bids, cards_dealt_param
        ):
            """Controlled bidding: Players bid based on position, avoiding forbidden."""
            bid = player.position  # Player 0 bids 0, Player 1 bids 1, etc.
            # Check if dealer and if bid is forbidden
            if is_dealer:
                forbidden = cards_dealt - current_total_bids
                if bid == forbidden:
                    # Choose different valid bid
                    bid = (bid + 1) % (cards_dealt + 1)
                    if bid == forbidden:
                        bid = (bid + 1) % (cards_dealt + 1)
            return bid

        def play_winning_cards(player, legal_cards, trick):
            """Try to play highest card to win tricks."""
            return max(legal_cards, key=lambda c: c.value)

        game = BlobGame(num_players=3, player_names=["Alice", "Bob", "Charlie"])

        result = game.play_round(
            cards_to_deal=3,
            get_bid_func=controlled_bid,
            get_card_func=play_winning_cards,
        )

        # Verify scores calculated
        for player_result in result["player_scores"]:
            # Score should be either 0 (missed bid) or 10+bid (made bid)
            score = player_result["round_score"]
            assert score >= 0
            if score > 0:
                expected_score = 10 + player_result["bid"]
                assert score == expected_score

    def test_play_round_invalid_bid_raises_exception(self):
        """play_round() raises InvalidBidException for invalid bid."""

        def invalid_bid(
            player, cards_dealt, is_dealer, current_total_bids, cards_dealt_param
        ):
            """Always return invalid bid (out of range)."""
            return 999  # Way out of range

        game = BlobGame(num_players=3)

        with pytest.raises(InvalidBidException, match="bid 999 is invalid"):
            game.play_round(
                cards_to_deal=5,
                get_bid_func=invalid_bid,
                get_card_func=self.simple_play_first_legal,
            )

    def test_play_round_dealer_constraint_enforced(self):
        """play_round() enforces dealer constraint on forbidden bid."""
        call_count = [0]

        def forbidden_bid(
            player, cards_dealt, is_dealer, current_total_bids, cards_dealt_param
        ):
            """Non-dealers bid 2, dealer tries forbidden bid."""
            call_count[0] += 1
            if is_dealer:
                # Dealer should not be able to bid forbidden value
                # If current_total is 4 and cards_dealt is 5, forbidden is 1
                forbidden = cards_dealt - current_total_bids
                return forbidden  # Try to bid forbidden value
            return 2  # Non-dealers bid 2

        game = BlobGame(num_players=3)

        with pytest.raises(InvalidBidException, match="forbidden"):
            game.play_round(
                cards_to_deal=5,
                get_bid_func=forbidden_bid,
                get_card_func=self.simple_play_first_legal,
            )

    def test_play_full_game_multiple_rounds(self):
        """play_full_game() executes multiple rounds correctly."""
        from ml.game.constants import generate_round_structure

        game = BlobGame(num_players=4, player_names=["P1", "P2", "P3", "P4"])

        # Generate round structure for 4 players, 3 starting cards
        round_structure = generate_round_structure(3, 4)

        # Play full game
        results = game.play_full_game(
            round_structure=round_structure,
            get_bid_func=self.simple_bid_always_zero,
            get_card_func=self.simple_play_first_legal,
        )

        # Verify results structure
        assert "num_rounds" in results
        assert "num_players" in results
        assert "round_results" in results
        assert "final_scores" in results
        assert "winner" in results

        # Verify correct number of rounds
        expected_rounds = len(round_structure)
        assert results["num_rounds"] == expected_rounds
        assert len(results["round_results"]) == expected_rounds

        # Verify winner structure
        assert "name" in results["winner"]
        assert "score" in results["winner"]
        assert results["winner"]["name"] in ["P1", "P2", "P3", "P4"]

        # Verify final scores are sorted
        scores = [s["total_score"] for s in results["final_scores"]]
        assert scores == sorted(scores, reverse=True)

    def test_play_full_game_dealer_rotation(self):
        """play_full_game() rotates dealer correctly across rounds."""
        game = BlobGame(num_players=3)

        # Play 5 rounds (more than number of players)
        round_structure = [3, 3, 3, 3, 3]

        # Track initial dealer
        initial_dealer = game.dealer_position

        _results = game.play_full_game(  # noqa: F841
            round_structure=round_structure,
            get_bid_func=self.simple_bid_always_zero,
            get_card_func=self.simple_play_first_legal,
        )

        # After 5 rounds with 3 players, dealer should have rotated
        # Initial: 0, after round 1: 1, after round 2: 2, after round 3: 0, etc.
        # After 5 rounds: (0 + 5) % 3 = 2
        expected_dealer = (initial_dealer + 5) % game.num_players
        assert game.dealer_position == expected_dealer

    def test_play_full_game_trump_rotation(self):
        """play_full_game() rotates trump correctly across rounds."""
        game = BlobGame(num_players=4)

        # Play 6 rounds (to cover full trump rotation: ♠,♥,♣,♦,None,♠)
        round_structure = [3] * 6

        results = game.play_full_game(
            round_structure=round_structure,
            get_bid_func=self.simple_bid_always_zero,
            get_card_func=self.simple_play_first_legal,
        )

        # Verify trump changed across rounds by checking round results
        trump_suits = [r["trump_suit"] for r in results["round_results"]]

        # Should cycle through: ♠, ♥, ♣, ♦, None, then repeat ♠
        from ml.game.constants import TRUMP_ROTATION

        # TRUMP_ROTATION has 5 elements, so for 6 rounds we expect the first one to repeat
        expected_trumps = [TRUMP_ROTATION[i % len(TRUMP_ROTATION)] for i in range(6)]
        assert trump_suits == expected_trumps

    def test_get_game_state_structure(self):
        """get_game_state() returns correct structure."""
        game = BlobGame(num_players=3, player_names=["Alice", "Bob", "Charlie"])
        game.setup_round(5)

        state = game.get_game_state()

        # Verify all expected keys present
        assert "phase" in state
        assert "round" in state
        assert "trump" in state
        assert "dealer_position" in state
        assert "num_players" in state
        assert "players" in state
        assert "current_trick" in state
        assert "tricks_history" in state
        assert "cards_remaining_by_suit" in state

        # Verify types
        assert isinstance(state["phase"], str)
        assert isinstance(state["round"], int)
        assert state["trump"] in ["♠", "♥", "♣", "♦", None]
        assert isinstance(state["dealer_position"], int)
        assert isinstance(state["num_players"], int)
        assert isinstance(state["players"], list)
        assert len(state["players"]) == 3

        # Verify player structure
        for player_state in state["players"]:
            assert "name" in player_state
            assert "position" in player_state
            assert "hand" in player_state
            assert "hand_size" in player_state
            assert "bid" in player_state
            assert "tricks_won" in player_state
            assert "total_score" in player_state
            assert "known_void_suits" in player_state

    def test_get_game_state_hidden_hands(self):
        """get_game_state() hides opponent hands when perspective provided."""
        game = BlobGame(num_players=3, player_names=["Alice", "Bob", "Charlie"])
        game.setup_round(5)

        # Get state from Alice's perspective (player 0)
        state = game.get_game_state(player_perspective=game.players[0])

        # Alice's hand should be visible
        assert state["players"][0]["hand"] is not None
        assert isinstance(state["players"][0]["hand"], list)
        assert len(state["players"][0]["hand"]) == 5

        # Bob's and Charlie's hands should be hidden
        assert state["players"][1]["hand"] is None
        assert state["players"][2]["hand"] is None

        # But hand sizes should be visible
        assert state["players"][1]["hand_size"] == 5
        assert state["players"][2]["hand_size"] == 5

    def test_get_game_state_all_visible(self):
        """get_game_state() shows all hands when no perspective given."""
        game = BlobGame(num_players=3, player_names=["Alice", "Bob", "Charlie"])
        game.setup_round(5)

        # Get state with no perspective (debug mode)
        state = game.get_game_state(player_perspective=None)

        # All hands should be visible
        for player_state in state["players"]:
            assert player_state["hand"] is not None
            assert isinstance(player_state["hand"], list)
            assert len(player_state["hand"]) == 5

    def test_get_game_state_serializable(self):
        """get_game_state() returns JSON-serializable data."""
        import json

        game = BlobGame(num_players=3)
        game.setup_round(5)

        state = game.get_game_state()

        # Should be able to serialize to JSON without errors
        try:
            json_str = json.dumps(state)
            # And deserialize back
            decoded_state = json.loads(json_str)
            assert decoded_state["num_players"] == 3
        except (TypeError, ValueError) as e:
            pytest.fail(f"State not JSON-serializable: {e}")

    def test_get_game_state_current_trick(self):
        """get_game_state() includes current trick information."""
        game = BlobGame(num_players=3)

        # Before any tricks, current_trick should be None
        state_before = game.get_game_state()
        assert state_before["current_trick"] is None

        # Start a round and create a trick
        game.setup_round(3)
        game.game_phase = "playing"
        game.current_trick = Trick(game.trump_suit)

        # Add some cards to the trick
        player1 = game.players[0]
        card1 = player1.hand[0]
        game.current_trick.add_card(player1, card1)

        player2 = game.players[1]
        card2 = player2.hand[0]
        game.current_trick.add_card(player2, card2)

        # Get state with trick in progress
        state_with_trick = game.get_game_state()

        assert state_with_trick["current_trick"] is not None
        assert "cards_played" in state_with_trick["current_trick"]
        assert "led_suit" in state_with_trick["current_trick"]
        assert "trump_suit" in state_with_trick["current_trick"]

        # Should show 2 cards played
        assert len(state_with_trick["current_trick"]["cards_played"]) == 2

        # Cards should be serialized as strings
        for player_name, card_str in state_with_trick["current_trick"]["cards_played"]:
            assert isinstance(player_name, str)
            assert isinstance(card_str, str)

    def test_get_game_state_tricks_history(self):
        """get_game_state() includes tricks_history with proper serialization."""
        game = BlobGame(num_players=3)
        game.setup_round(3)
        game.game_phase = "playing"

        # Complete a trick
        trick1 = Trick(game.trump_suit)
        for player in game.players:
            card = player.hand[0]
            trick1.add_card(player, card)

        trick1.determine_winner()
        game.tricks_history.append(trick1)

        # Complete another trick
        trick2 = Trick(game.trump_suit)
        for player in game.players:
            card = player.hand[0]
            trick2.add_card(player, card)

        trick2.determine_winner()
        game.tricks_history.append(trick2)

        # Get game state
        state = game.get_game_state()

        # Check tricks_history is serialized
        assert "tricks_history" in state
        assert len(state["tricks_history"]) == 2

        # Check each trick has proper structure
        for trick_state in state["tricks_history"]:
            assert "winner" in trick_state
            assert "cards" in trick_state
            assert isinstance(trick_state["winner"], str)
            assert isinstance(trick_state["cards"], list)
            assert len(trick_state["cards"]) == 3  # 3 players

            # Check cards are serialized as (player_name, card_str) tuples
            for player_name, card_str in trick_state["cards"]:
                assert isinstance(player_name, str)
                assert isinstance(card_str, str)

    def test_get_legal_actions_bidding_phase(self):
        """get_legal_actions() returns valid bids in bidding phase."""
        game = BlobGame(num_players=3)
        game.setup_round(5)

        # Game is in bidding phase
        assert game.game_phase == "bidding"

        # Get legal actions for first player (not dealer)
        player = game.players[1]  # Player 1 bids first (left of dealer)
        actions = game.get_legal_actions(player)

        # Should return list of integers (valid bids)
        assert isinstance(actions, list)
        assert all(isinstance(bid, int) for bid in actions)
        assert 0 in actions
        assert 5 in actions  # Max bid for 5 cards
        assert len(actions) == 6  # [0, 1, 2, 3, 4, 5]

    def test_get_legal_actions_dealer_constraint(self):
        """get_legal_actions() enforces dealer constraint."""
        game = BlobGame(num_players=3)
        game.setup_round(5)

        # Make non-dealer players bid first
        game.players[1].bid = 2
        game.players[2].bid = 1

        # Now get actions for dealer (player 0)
        dealer = game.players[0]
        actions = game.get_legal_actions(dealer)

        # Dealer should not be able to bid 2 (2 + 1 + 2 = 5)
        forbidden = 5 - 3  # forbidden = cards_dealt - current_total_bids
        assert forbidden == 2
        assert 2 not in actions
        assert 0 in actions
        assert 1 in actions
        assert 3 in actions
        assert 4 in actions
        assert 5 in actions

    def test_get_legal_actions_playing_phase(self):
        """get_legal_actions() returns legal cards in playing phase."""
        game = BlobGame(num_players=3)
        game.setup_round(5)

        # Transition to playing phase
        game.game_phase = "playing"
        game.current_trick = Trick(game.trump_suit)

        player = game.players[0]
        actions = game.get_legal_actions(player)

        # Should return list of Card objects
        assert isinstance(actions, list)
        assert all(isinstance(card, Card) for card in actions)
        assert len(actions) == 5  # Player has 5 cards

    def test_get_legal_actions_must_follow_suit(self):
        """get_legal_actions() enforces follow-suit rule."""
        game = BlobGame(num_players=3)
        game.setup_round(5)

        # Manually set player's hand with specific cards
        player = game.players[0]
        player.hand = [Card("5", "♥"), Card("8", "♥"), Card("3", "♣"), Card("K", "♦")]

        # Transition to playing phase with hearts led
        game.game_phase = "playing"
        game.current_trick = Trick(game.trump_suit)
        game.current_trick.led_suit = "♥"

        actions = game.get_legal_actions(player)

        # Should only return hearts (player must follow suit)
        assert len(actions) == 2
        assert all(card.suit == "♥" for card in actions)
        assert Card("5", "♥") in actions
        assert Card("8", "♥") in actions

    def test_get_legal_actions_no_suit_all_cards_legal(self):
        """get_legal_actions() returns all cards if player can't follow suit."""
        game = BlobGame(num_players=3)
        game.setup_round(5)

        # Manually set player's hand with no hearts
        player = game.players[0]
        player.hand = [Card("3", "♣"), Card("K", "♦"), Card("7", "♠")]

        # Transition to playing phase with hearts led
        game.game_phase = "playing"
        game.current_trick = Trick(game.trump_suit)
        game.current_trick.led_suit = "♥"

        actions = game.get_legal_actions(player)

        # Player has no hearts, so all cards are legal
        assert len(actions) == 3
        assert Card("3", "♣") in actions
        assert Card("K", "♦") in actions
        assert Card("7", "♠") in actions

    def test_get_legal_actions_other_phases(self):
        """get_legal_actions() returns empty list in non-action phases."""
        game = BlobGame(num_players=3)
        player = game.players[0]

        # In setup phase
        game.game_phase = "setup"
        actions = game.get_legal_actions(player)
        assert actions == []

        # In scoring phase
        game.game_phase = "scoring"
        actions = game.get_legal_actions(player)
        assert actions == []

        # In complete phase
        game.game_phase = "complete"
        actions = game.get_legal_actions(player)
        assert actions == []


# ============================================================================
# TestGetCurrentPlayer: Test canonical turn tracking
# ============================================================================


class TestGetCurrentPlayer:
    """
    Test canonical turn tracking via get_current_player().

    This is the single source of truth for turn order throughout the game.
    Used by MCTS and AI components to determine whose turn it is.
    """

    def test_bidding_phase_first_player(self):
        """First bidder is player left of dealer."""
        game = BlobGame(num_players=4)
        game.setup_round(5)

        current = game.get_current_player()
        expected_idx = (game.dealer_position + 1) % 4

        assert current is not None
        assert current == game.players[expected_idx]
        assert current.bid is None

    def test_bidding_phase_sequence(self):
        """Turn advances through all players in bidding order."""
        game = BlobGame(num_players=4)
        game.setup_round(5)

        bidders = []
        for i in range(4):
            current = game.get_current_player()
            assert current is not None, f"No current player at step {i}"

            bidders.append(current.position)
            current.make_bid(0 if i < 3 else 1)  # Dealer can't bid 0 (forbidden)

        # Should cycle through all 4 players exactly once
        assert len(bidders) == 4
        assert len(set(bidders)) == 4, "Should have 4 unique players"

        # Verify bidding order starts left of dealer
        expected_first = (game.dealer_position + 1) % 4
        assert bidders[0] == expected_first

    def test_bidding_complete_returns_none(self):
        """Returns None when all players have bid."""
        game = BlobGame(num_players=4)
        game.setup_round(5)

        # Make all players bid
        for i, player in enumerate(game.players):
            if player.position == game.dealer_position:
                player.make_bid(1)  # Dealer can't bid 0 in this scenario
            else:
                player.make_bid(0)

        # No current player (all have bid)
        assert game.get_current_player() is None

    def test_playing_phase_first_trick_lead(self):
        """First trick led by player left of dealer."""
        game = BlobGame(num_players=4)
        game.setup_round(5)

        # Complete bidding phase
        for i, player in enumerate(game.players):
            player.make_bid(0)

        # Transition to playing phase
        game.game_phase = "playing"
        from ml.game.blob import Trick

        game.current_trick = Trick(game.trump_suit)

        current = game.get_current_player()
        expected_idx = (game.dealer_position + 1) % 4

        assert current is not None
        assert current == game.players[expected_idx]

    def test_playing_phase_turn_rotation(self):
        """Turn rotates through players during trick."""
        game = BlobGame(num_players=4)
        game.setup_round(5)

        # Complete bidding
        for player in game.players:
            player.make_bid(0)

        # Setup playing phase
        game.game_phase = "playing"
        from ml.game.blob import Trick

        game.current_trick = Trick(game.trump_suit)

        play_order = []
        for i in range(4):
            current = game.get_current_player()
            assert current is not None, f"No current player at position {i}"

            play_order.append(current.position)

            # Simulate card play
            card = current.hand[0]
            game.current_trick.add_card(current, card)

        # Should have 4 different players in order
        assert len(play_order) == 4
        assert len(set(play_order)) == 4

        # Verify rotation is correct (sequential from lead player)
        lead_idx = (game.dealer_position + 1) % 4
        for i, pos in enumerate(play_order):
            expected = (lead_idx + i) % 4
            assert pos == expected, f"Position {i}: expected {expected}, got {pos}"

    def test_playing_phase_subsequent_trick_winner_leads(self):
        """Winner of previous trick leads next trick."""
        game = BlobGame(num_players=4)
        game.setup_round(5)

        # Complete bidding
        for player in game.players:
            player.make_bid(0)

        game.game_phase = "playing"
        from ml.game.blob import Trick

        # Play first trick completely
        first_trick = Trick(game.trump_suit)
        game.current_trick = first_trick
        lead_idx = (game.dealer_position + 1) % 4

        for i in range(4):
            player_idx = (lead_idx + i) % 4
            player = game.players[player_idx]
            card = player.hand[0]
            first_trick.add_card(player, card)
            player.hand.remove(card)

        # Determine winner of first trick
        winner = first_trick.determine_winner()
        winner.win_trick()
        game.tricks_history.append(first_trick)

        # Start second trick
        game.current_trick = Trick(game.trump_suit)
        current = game.get_current_player()

        # Winner of first trick should lead second trick
        assert current == winner

    def test_no_current_player_in_other_phases(self):
        """Returns None in setup/scoring/complete phases."""
        game = BlobGame(num_players=4)

        # Setup phase (initial state)
        assert game.game_phase == "setup"
        assert game.get_current_player() is None

        # Scoring phase
        game.game_phase = "scoring"
        assert game.get_current_player() is None

        # Complete phase
        game.game_phase = "complete"
        assert game.get_current_player() is None

    def test_playing_phase_trick_complete_returns_none(self):
        """Returns None when trick is complete (all players played)."""
        game = BlobGame(num_players=4)
        game.setup_round(5)

        # Complete bidding
        for player in game.players:
            player.make_bid(0)

        game.game_phase = "playing"
        from ml.game.blob import Trick

        game.current_trick = Trick(game.trump_suit)
        lead_idx = (game.dealer_position + 1) % 4

        # All 4 players play
        for i in range(4):
            player_idx = (lead_idx + i) % 4
            player = game.players[player_idx]
            card = player.hand[0]
            game.current_trick.add_card(player, card)

        # Trick complete, no current player
        assert game.get_current_player() is None
