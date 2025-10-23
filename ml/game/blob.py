"""
Core game logic for Blob card game.

This module implements the complete game engine including Card, Deck, Player,
Trick, and BlobGame classes with full anti-cheat validation.
"""

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Dict, Union
import random
from ml.game.constants import SUITS, RANKS, RANK_VALUES, TRUMP_ROTATION, DECK_SIZE


# ============================================================================
# Custom Exceptions
# ============================================================================

class BlobGameException(Exception):
    """Base exception for Blob game errors."""
    pass


class IllegalPlayException(BlobGameException):
    """Raised when player attempts an illegal card play."""

    def __init__(self, player_name: str, card: 'Card', reason: str):
        self.player_name = player_name
        self.card = card
        self.reason = reason
        super().__init__(f"{player_name} played {card} illegally: {reason}")


class InvalidBidException(BlobGameException):
    """Raised when player attempts an invalid bid."""
    pass


class GameStateException(BlobGameException):
    """Raised when game is in invalid state for requested action."""
    pass


# ============================================================================
# Card Class
# ============================================================================

@dataclass(frozen=True)
class Card:
    """
    Immutable playing card with suit and rank.

    Attributes:
        rank: Card rank (2-10, J, Q, K, A)
        suit: Card suit (♠, ♥, ♣, ♦)
        value: Numeric value for comparison (2=2, J=11, Q=12, K=13, A=14)
    """
    rank: str
    suit: str

    def __post_init__(self):
        """Validate card creation."""
        if self.suit not in SUITS:
            raise ValueError(f"Invalid suit: {self.suit}. Must be one of {SUITS}")
        if self.rank not in RANKS:
            raise ValueError(f"Invalid rank: {self.rank}. Must be one of {RANKS}")

    @property
    def value(self) -> int:
        """Numeric value for card comparison."""
        return RANK_VALUES[self.rank]

    def __str__(self) -> str:
        """String representation: 'A♠'"""
        return f"{self.rank}{self.suit}"

    def __repr__(self) -> str:
        """Developer representation: Card('A', '♠')"""
        return f"Card('{self.rank}', '{self.suit}')"

    def __lt__(self, other: 'Card') -> bool:
        """Enable sorting by value, then by suit."""
        if not isinstance(other, Card):
            return NotImplemented
        if self.value != other.value:
            return self.value < other.value
        return SUITS.index(self.suit) < SUITS.index(other.suit)

    def __hash__(self) -> int:
        """Enable use in sets and dicts."""
        return hash((self.rank, self.suit))


# ============================================================================
# Deck Class
# ============================================================================

class Deck:
    """
    Standard 52-card deck with dealing and shuffling.

    Attributes:
        cards: List of all cards in deck
        dealt_cards: Set of cards that have been dealt
    """

    def __init__(self):
        """Create standard 52-card deck."""
        self.cards: List[Card] = []
        self.dealt_cards: Set[Card] = set()
        self.reset()

    def reset(self) -> None:
        """Return all cards to deck and clear dealt tracking."""
        self.cards = [Card(rank, suit) for suit in SUITS for rank in RANKS]
        self.dealt_cards = set()

    def shuffle(self) -> None:
        """Randomize card order."""
        random.shuffle(self.cards)

    def deal(self, num_cards: int, num_players: int) -> List[List[Card]]:
        """
        Deal cards to players.

        Args:
            num_cards: Number of cards to deal to each player
            num_players: Number of players receiving cards

        Returns:
            List of hands (each hand is a list of Cards)

        Raises:
            ValueError: If not enough cards available
        """
        total_needed = num_cards * num_players

        if total_needed > len(self.cards):
            raise ValueError(
                f"Cannot deal {num_cards} cards to {num_players} players "
                f"(need {total_needed}, only {len(self.cards)} available)"
            )

        hands: List[List[Card]] = [[] for _ in range(num_players)]

        for i in range(num_cards):
            for player_idx in range(num_players):
                card = self.cards.pop(0)
                hands[player_idx].append(card)
                self.dealt_cards.add(card)

        return hands

    def remaining_cards(self) -> int:
        """Return number of cards left in deck."""
        return len(self.cards)


# ============================================================================
# Player Class
# ============================================================================

class Player:
    """
    Player in a Blob game.

    Attributes:
        name: Player identifier
        position: Seat position (0-indexed)
        hand: Current cards in hand
        bid: Bid for current round (None if not yet bid)
        tricks_won: Number of tricks won this round
        total_score: Cumulative score across all rounds
        known_void_suits: Suits player has revealed they don't have
        cards_played: All cards this player has played this round
    """

    def __init__(self, name: str, position: int):
        """
        Initialize a player.

        Args:
            name: Player identifier
            position: Seat position (0-indexed)
        """
        self.name = name
        self.position = position
        self.hand: List[Card] = []
        self.bid: Optional[int] = None
        self.tricks_won: int = 0
        self.total_score: int = 0
        self.known_void_suits: Set[str] = set()
        self.cards_played: List[Card] = []

    def receive_cards(self, cards: List[Card]) -> None:
        """
        Add cards to player's hand.

        Args:
            cards: Cards to add to hand
        """
        self.hand.extend(cards)

    def play_card(self, card: Card) -> Card:
        """
        Remove and return card from hand.

        Args:
            card: Card to play

        Returns:
            The card that was played

        Raises:
            ValueError: If card not in hand
        """
        if card not in self.hand:
            raise ValueError(f"{self.name} does not have {card} in hand")

        self.hand.remove(card)
        self.cards_played.append(card)
        return card

    def make_bid(self, bid: int) -> None:
        """
        Set bid for current round.

        Args:
            bid: Number of tricks player expects to win
        """
        self.bid = bid

    def win_trick(self) -> None:
        """Increment tricks won counter."""
        self.tricks_won += 1

    def calculate_round_score(self) -> int:
        """
        Calculate score for current round.

        Returns:
            Score: 10 + bid if exact match, 0 otherwise
        """
        from ml.game.constants import SCORE_BASE

        if self.bid is None:
            return 0

        if self.tricks_won == self.bid:
            return SCORE_BASE + self.bid
        else:
            return 0

    def reset_round(self) -> None:
        """Clear round-specific state."""
        self.hand = []
        self.bid = None
        self.tricks_won = 0
        self.known_void_suits = set()
        self.cards_played = []

    def sort_hand(self) -> None:
        """Sort cards in hand by suit then rank."""
        self.hand.sort()

    def mark_void_suit(self, suit: str) -> None:
        """
        Mark that player doesn't have cards in this suit.

        Args:
            suit: Suit to mark as void
        """
        self.known_void_suits.add(suit)

    def has_suit(self, suit: str) -> bool:
        """
        Check if player has any cards in suit.

        Args:
            suit: Suit to check

        Returns:
            True if player has at least one card in suit
        """
        return any(card.suit == suit for card in self.hand)

    def __str__(self) -> str:
        """String representation."""
        return f"Player({self.name}, pos={self.position}, score={self.total_score})"

    def __repr__(self) -> str:
        """Developer representation."""
        return self.__str__()


# ============================================================================
# Trick Class
# ============================================================================

class Trick:
    """
    A single trick in a Blob game.

    Attributes:
        trump_suit: Current round's trump suit (or None for no-trump)
        cards_played: List of (player, card) tuples in play order
        led_suit: Suit of first card played (None if no cards yet)
        winner: Trick winner (None until determined)
    """

    def __init__(self, trump_suit: Optional[str]):
        """
        Initialize a new trick.

        Args:
            trump_suit: Trump suit for this round (or None)
        """
        self.trump_suit = trump_suit
        self.cards_played: List[Tuple[Player, Card]] = []
        self.led_suit: Optional[str] = None
        self.winner: Optional[Player] = None

    def add_card(self, player: Player, card: Card) -> None:
        """
        Add a card to the trick.

        Args:
            player: Player who played the card
            card: Card that was played
        """
        self.cards_played.append((player, card))

        # Set led suit on first card
        if len(self.cards_played) == 1:
            self.led_suit = card.suit

    def determine_winner(self) -> Player:
        """
        Determine the winner of this trick.

        Winner logic:
        1. If trump cards played: highest trump wins
        2. Otherwise: highest card in led suit wins

        Returns:
            Player who won the trick

        Raises:
            GameStateException: If no cards have been played
        """
        if not self.cards_played:
            raise GameStateException("Cannot determine winner: no cards played")

        # Filter trump cards if trump suit exists
        if self.trump_suit is not None:
            trump_cards = [(p, c) for p, c in self.cards_played
                          if c.suit == self.trump_suit]

            if trump_cards:
                # Highest trump wins
                winner = max(trump_cards, key=lambda x: x[1].value)
                self.winner = winner[0]
                return self.winner

        # No trump cards played (or no-trump round): highest card in led suit wins
        led_suit_cards = [(p, c) for p, c in self.cards_played
                         if c.suit == self.led_suit]

        if not led_suit_cards:
            # Shouldn't happen in valid game, but handle it
            raise GameStateException(
                f"No cards in led suit {self.led_suit} found in trick"
            )

        winner = max(led_suit_cards, key=lambda x: x[1].value)
        self.winner = winner[0]
        return self.winner

    def get_winning_card(self) -> Card:
        """
        Get the winning card of this trick.

        Returns:
            The card that won the trick

        Raises:
            GameStateException: If winner not yet determined
        """
        if self.winner is None:
            raise GameStateException("Winner not yet determined")

        for player, card in self.cards_played:
            if player == self.winner:
                return card

        raise GameStateException("Winner's card not found in trick")

    def is_complete(self, num_players: int) -> bool:
        """
        Check if all players have played.

        Args:
            num_players: Expected number of players

        Returns:
            True if all players have played
        """
        return len(self.cards_played) == num_players

    def clear(self) -> None:
        """Reset trick for reuse."""
        self.cards_played = []
        self.led_suit = None
        self.winner = None

    def __str__(self) -> str:
        """String representation."""
        cards_str = ", ".join([f"{p.name}:{c}" for p, c in self.cards_played])
        return f"Trick(led={self.led_suit}, trump={self.trump_suit}, [{cards_str}])"

    def __repr__(self) -> str:
        """Developer representation."""
        return self.__str__()
