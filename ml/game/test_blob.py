"""
Unit tests for Blob game engine core classes.

Tests Card, Deck, Player, and Trick classes with comprehensive coverage
of normal operations, edge cases, and anti-cheat functionality.
"""

import pytest
from ml.game.blob import (
    Card, Deck, Player, Trick,
    BlobGameException, IllegalPlayException, InvalidBidException, GameStateException
)
from ml.game.constants import SUITS, RANKS, RANK_VALUES, SCORE_BASE


# ============================================================================
# Test Card Class
# ============================================================================

class TestCard:
    """Test Card class functionality."""

    def test_card_creation(self):
        """Test Card initialization and properties."""
        card = Card('A', '♠')
        assert card.rank == 'A'
        assert card.suit == '♠'
        assert card.value == 14  # Ace is highest

    def test_card_value_mapping(self):
        """Test that all ranks map to correct values."""
        assert Card('2', '♠').value == 2
        assert Card('10', '♠').value == 10
        assert Card('J', '♠').value == 11
        assert Card('Q', '♠').value == 12
        assert Card('K', '♠').value == 13
        assert Card('A', '♠').value == 14

    def test_card_invalid_suit(self):
        """Test that invalid suit raises ValueError."""
        with pytest.raises(ValueError, match="Invalid suit"):
            Card('A', 'X')

    def test_card_invalid_rank(self):
        """Test that invalid rank raises ValueError."""
        with pytest.raises(ValueError, match="Invalid rank"):
            Card('1', '♠')

    def test_card_equality(self):
        """Test Card comparison."""
        card1 = Card('A', '♠')
        card2 = Card('A', '♠')
        card3 = Card('K', '♠')

        assert card1 == card2
        assert card1 != card3

    def test_card_sorting(self):
        """Test Cards sort by value correctly."""
        cards = [Card('A', '♠'), Card('2', '♠'), Card('K', '♠'), Card('5', '♠')]
        sorted_cards = sorted(cards)

        assert sorted_cards[0].rank == '2'
        assert sorted_cards[1].rank == '5'
        assert sorted_cards[2].rank == 'K'
        assert sorted_cards[3].rank == 'A'

    def test_card_sorting_by_suit(self):
        """Test Cards with same value sort by suit."""
        cards = [Card('A', '♦'), Card('A', '♠'), Card('A', '♥'), Card('A', '♣')]
        sorted_cards = sorted(cards)

        # Should sort by SUITS order: ♠, ♥, ♣, ♦
        assert sorted_cards[0].suit == '♠'
        assert sorted_cards[1].suit == '♥'
        assert sorted_cards[2].suit == '♣'
        assert sorted_cards[3].suit == '♦'

    def test_card_string_representation(self):
        """Test __str__ and __repr__."""
        card = Card('A', '♠')
        assert str(card) == 'A♠'
        assert repr(card) == "Card('A', '♠')"

    def test_card_hashable(self):
        """Test Cards can be used in sets and dicts."""
        card1 = Card('A', '♠')
        card2 = Card('A', '♠')
        card3 = Card('K', '♠')

        card_set = {card1, card2, card3}
        assert len(card_set) == 2  # card1 and card2 are same

        card_dict = {card1: "value"}
        assert card_dict[card2] == "value"  # card2 should match card1


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
        hands = deck.deal(13, 4)  # 13 * 4 = 52

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
        cards = [Card('A', '♠'), Card('K', '♠'), Card('Q', '♠')]

        player.receive_cards(cards)

        assert len(player.hand) == 3
        assert Card('A', '♠') in player.hand

    def test_play_card(self):
        """Can play cards from hand."""
        player = Player("Alice", 0)
        card = Card('A', '♠')
        player.receive_cards([card, Card('K', '♠')])

        played = player.play_card(card)

        assert played == card
        assert card not in player.hand
        assert len(player.hand) == 1
        assert card in player.cards_played

    def test_play_card_not_in_hand(self):
        """Cannot play card not in hand."""
        player = Player("Alice", 0)
        player.receive_cards([Card('K', '♠')])

        with pytest.raises(ValueError, match="does not have"):
            player.play_card(Card('A', '♠'))

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
        player.receive_cards([Card('A', '♠'), Card('K', '♠')])
        player.make_bid(2)
        player.tricks_won = 1
        player.known_void_suits.add('♥')
        player.cards_played.append(Card('A', '♠'))

        player.reset_round()

        assert player.hand == []
        assert player.bid is None
        assert player.tricks_won == 0
        assert player.known_void_suits == set()
        assert player.cards_played == []

    def test_sort_hand(self):
        """Hand sorts correctly."""
        player = Player("Alice", 0)
        player.receive_cards([
            Card('A', '♠'),
            Card('2', '♠'),
            Card('K', '♠'),
            Card('5', '♠')
        ])

        player.sort_hand()

        assert player.hand[0].rank == '2'
        assert player.hand[1].rank == '5'
        assert player.hand[2].rank == 'K'
        assert player.hand[3].rank == 'A'

    def test_mark_void_suit(self):
        """Can mark void suits."""
        player = Player("Alice", 0)

        player.mark_void_suit('♥')
        assert '♥' in player.known_void_suits

        player.mark_void_suit('♣')
        assert '♣' in player.known_void_suits
        assert len(player.known_void_suits) == 2

    def test_has_suit(self):
        """Can check if player has suit."""
        player = Player("Alice", 0)
        player.receive_cards([
            Card('A', '♠'),
            Card('K', '♠'),
            Card('Q', '♥')
        ])

        assert player.has_suit('♠') is True
        assert player.has_suit('♥') is True
        assert player.has_suit('♣') is False
        assert player.has_suit('♦') is False

    def test_player_string_representation(self):
        """Test __str__ and __repr__."""
        player = Player("Alice", 0)
        player.total_score = 25

        s = str(player)
        assert "Alice" in s
        assert "pos=0" in s
        assert "score=25" in s


# ============================================================================
# Test Trick Class
# ============================================================================

class TestTrick:
    """Test Trick class functionality."""

    def test_trick_initialization(self):
        """Trick initializes correctly."""
        trick = Trick('♠')

        assert trick.trump_suit == '♠'
        assert trick.cards_played == []
        assert trick.led_suit is None
        assert trick.winner is None

    def test_trick_no_trump(self):
        """Can create no-trump trick."""
        trick = Trick(None)
        assert trick.trump_suit is None

    def test_add_card_sets_led_suit(self):
        """First card sets led suit."""
        trick = Trick('♠')
        player = Player("Alice", 0)
        card = Card('A', '♥')

        trick.add_card(player, card)

        assert trick.led_suit == '♥'
        assert len(trick.cards_played) == 1

    def test_trick_winner_no_trump(self):
        """Highest card in led suit wins (no trump)."""
        trick = Trick(None)
        p1 = Player("Alice", 0)
        p2 = Player("Bob", 1)
        p3 = Player("Carol", 2)

        # Hearts led, Alice plays highest
        trick.add_card(p1, Card('A', '♥'))
        trick.add_card(p2, Card('K', '♥'))
        trick.add_card(p3, Card('Q', '♥'))

        winner = trick.determine_winner()
        assert winner == p1

    def test_trick_winner_off_suit_loses(self):
        """Off-suit card cannot win if led suit played."""
        trick = Trick(None)
        p1 = Player("Alice", 0)
        p2 = Player("Bob", 1)
        p3 = Player("Carol", 2)

        # Hearts led
        trick.add_card(p1, Card('2', '♥'))
        trick.add_card(p2, Card('A', '♠'))  # Off-suit, even though Ace
        trick.add_card(p3, Card('3', '♥'))

        winner = trick.determine_winner()
        assert winner == p3  # 3♥ beats 2♥ (A♠ doesn't count)

    def test_trick_winner_with_trump(self):
        """Trump card beats non-trump."""
        trick = Trick('♠')  # Spades are trump
        p1 = Player("Alice", 0)
        p2 = Player("Bob", 1)
        p3 = Player("Carol", 2)

        # Hearts led, but spade trump wins
        trick.add_card(p1, Card('A', '♥'))  # Led suit, high card
        trick.add_card(p2, Card('2', '♠'))  # Trump (lowest spade)
        trick.add_card(p3, Card('K', '♥'))  # Led suit, but not trump

        winner = trick.determine_winner()
        assert winner == p2  # Lowest trump beats highest non-trump

    def test_trick_winner_multiple_trumps(self):
        """Highest trump wins."""
        trick = Trick('♠')
        p1 = Player("Alice", 0)
        p2 = Player("Bob", 1)
        p3 = Player("Carol", 2)

        trick.add_card(p1, Card('5', '♠'))  # Trump
        trick.add_card(p2, Card('K', '♠'))  # Higher trump
        trick.add_card(p3, Card('A', '♥'))  # Not trump

        winner = trick.determine_winner()
        assert winner == p2  # K♠ is highest trump

    def test_trick_winner_trump_is_led_suit(self):
        """Trump suit can be led suit."""
        trick = Trick('♠')
        p1 = Player("Alice", 0)
        p2 = Player("Bob", 1)

        trick.add_card(p1, Card('K', '♠'))  # Led trump
        trick.add_card(p2, Card('A', '♠'))  # Higher trump

        winner = trick.determine_winner()
        assert winner == p2

    def test_determine_winner_no_cards(self):
        """Cannot determine winner with no cards."""
        trick = Trick('♠')

        with pytest.raises(GameStateException, match="no cards played"):
            trick.determine_winner()

    def test_get_winning_card(self):
        """Can get the winning card."""
        trick = Trick('♠')
        p1 = Player("Alice", 0)
        p2 = Player("Bob", 1)

        card1 = Card('K', '♠')
        card2 = Card('A', '♠')

        trick.add_card(p1, card1)
        trick.add_card(p2, card2)

        trick.determine_winner()
        winning_card = trick.get_winning_card()

        assert winning_card == card2

    def test_get_winning_card_before_determine(self):
        """Cannot get winning card before determining winner."""
        trick = Trick('♠')
        p1 = Player("Alice", 0)

        trick.add_card(p1, Card('A', '♠'))

        with pytest.raises(GameStateException, match="Winner not yet determined"):
            trick.get_winning_card()

    def test_is_complete(self):
        """Check if all players have played."""
        trick = Trick('♠')
        p1 = Player("Alice", 0)
        p2 = Player("Bob", 1)
        p3 = Player("Carol", 2)

        assert trick.is_complete(3) is False

        trick.add_card(p1, Card('A', '♠'))
        assert trick.is_complete(3) is False

        trick.add_card(p2, Card('K', '♠'))
        assert trick.is_complete(3) is False

        trick.add_card(p3, Card('Q', '♠'))
        assert trick.is_complete(3) is True

    def test_clear(self):
        """Clear resets trick state."""
        trick = Trick('♠')
        p1 = Player("Alice", 0)

        trick.add_card(p1, Card('A', '♠'))
        trick.determine_winner()

        assert len(trick.cards_played) > 0
        assert trick.led_suit is not None
        assert trick.winner is not None

        trick.clear()

        assert trick.cards_played == []
        assert trick.led_suit is None
        assert trick.winner is None
        assert trick.trump_suit == '♠'  # Trump doesn't change

    def test_trick_string_representation(self):
        """Test __str__ and __repr__."""
        trick = Trick('♠')
        p1 = Player("Alice", 0)

        trick.add_card(p1, Card('A', '♥'))

        s = str(trick)
        assert 'led=♥' in s
        assert 'trump=♠' in s
        assert 'Alice' in s


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
        card = Card('A', '♠')
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
