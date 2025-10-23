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


# ============================================================================
# BlobGame Class (Main Game Orchestrator)
# ============================================================================

class BlobGame:
    """
    Main game orchestrator for Blob card game.

    Manages complete game flow including setup, bidding, playing, and scoring
    across multiple rounds with 3-8 players.

    Attributes:
        num_players: Number of players (3-8)
        players: List of Player objects
        deck: Game deck
        current_round: Round counter (0-indexed)
        trump_suit: Current trump suit (or None for no-trump)
        dealer_position: Current dealer position (rotates each round)
        current_trick: Active trick being played
        tricks_history: Completed tricks this round
        game_phase: Current phase ('setup', 'bidding', 'playing', 'scoring', 'complete')
        cards_played_this_round: All cards played this round (for card counting)
        cards_remaining_by_suit: Count of unplayed cards per suit
    """

    def __init__(self, num_players: int, player_names: Optional[List[str]] = None):
        """
        Initialize game with N players.

        Args:
            num_players: Number of players (must be 3-8)
            player_names: Optional list of player names. If None, generates default names.

        Raises:
            ValueError: If num_players not in valid range [3, 8]
            ValueError: If player_names provided but length doesn't match num_players
        """
        from ml.game.constants import MIN_PLAYERS, MAX_PLAYERS

        # Validate num_players
        if num_players < MIN_PLAYERS or num_players > MAX_PLAYERS:
            raise ValueError(
                f"num_players must be between {MIN_PLAYERS} and {MAX_PLAYERS}, "
                f"got {num_players}"
            )

        self.num_players = num_players

        # Generate player names if not provided
        if player_names is None:
            player_names = [f"Player {i+1}" for i in range(num_players)]
        else:
            if len(player_names) != num_players:
                raise ValueError(
                    f"player_names length ({len(player_names)}) must match "
                    f"num_players ({num_players})"
                )

        # Create Player objects
        self.players: List[Player] = [
            Player(name, position)
            for position, name in enumerate(player_names)
        ]

        # Initialize deck
        self.deck = Deck()

        # Game state
        self.current_round: int = 0
        self.trump_suit: Optional[str] = None
        self.dealer_position: int = 0
        self.current_trick: Optional[Trick] = None
        self.tricks_history: List[Trick] = []
        self.game_phase: str = 'setup'

        # Anti-cheat and card counting state
        self.cards_played_this_round: List[Card] = []
        self.cards_remaining_by_suit: Dict[str, int] = {}

    def determine_trump(self) -> Optional[str]:
        """
        Determine trump suit for current round based on round number.

        Trump rotates through: ♠ → ♥ → ♣ → ♦ → None (no-trump) → repeat

        Returns:
            Trump suit for current round, or None for no-trump rounds

        Examples:
            Round 0 → ♠ (Spades)
            Round 1 → ♥ (Hearts)
            Round 2 → ♣ (Clubs)
            Round 3 → ♦ (Diamonds)
            Round 4 → None (no-trump)
            Round 5 → ♠ (cycle repeats)
        """
        trump_index = self.current_round % len(TRUMP_ROTATION)
        return TRUMP_ROTATION[trump_index]

    def setup_round(self, cards_to_deal: int) -> None:
        """
        Prepare for a new round.

        Sets up all necessary state for playing a round:
        1. Reset deck and shuffle
        2. Reset all player round state
        3. Determine trump suit
        4. Deal cards to players
        5. Sort player hands
        6. Initialize card counting state
        7. Set game phase to 'bidding'

        Args:
            cards_to_deal: Number of cards to deal to each player

        Raises:
            ValueError: If cards_to_deal * num_players > 52

        Example:
            >>> game = BlobGame(num_players=4)
            >>> game.setup_round(5)  # Deal 5 cards to each of 4 players
            >>> game.trump_suit  # Will be ♠ for round 0
            >>> game.game_phase  # Will be 'bidding'
        """
        # Reset and shuffle deck
        self.deck.reset()
        self.deck.shuffle()

        # Reset player round state
        for player in self.players:
            player.reset_round()

        # Determine trump for this round
        self.trump_suit = self.determine_trump()

        # Deal cards to all players
        hands = self.deck.deal(cards_to_deal, self.num_players)
        for player, hand in zip(self.players, hands):
            player.receive_cards(hand)
            player.sort_hand()

        # Initialize card counting state
        self.cards_played_this_round = []
        self.cards_remaining_by_suit = {suit: 0 for suit in SUITS}

        # Count cards dealt by suit
        for player in self.players:
            for card in player.hand:
                self.cards_remaining_by_suit[card.suit] += 1

        # Reset trick state
        self.current_trick = None
        self.tricks_history = []

        # Set game phase to bidding
        self.game_phase = 'bidding'

    def get_forbidden_bid(self, current_total_bids: int, cards_dealt: int) -> Optional[int]:
        """
        Calculate dealer's forbidden bid.

        The dealer cannot bid such that the sum of all bids equals the number
        of cards dealt. This creates strategic tension and ensures someone will
        either over-bid or under-bid.

        Args:
            current_total_bids: Sum of all bids made by non-dealer players
            cards_dealt: Number of cards dealt to each player this round

        Returns:
            The forbidden bid value, or None if the forbidden value is out of
            valid range (i.e., < 0 or > cards_dealt)

        Examples:
            >>> # 3 players, 5 cards dealt, others bid 2 and 1
            >>> game.get_forbidden_bid(3, 5)
            2  # Dealer cannot bid 2 (would make total = 5)

            >>> # If current_total_bids already exceeds cards_dealt
            >>> game.get_forbidden_bid(6, 5)
            None  # Forbidden bid would be -1, which is invalid
        """
        forbidden = cards_dealt - current_total_bids

        # If forbidden bid is out of valid range, return None
        if forbidden < 0 or forbidden > cards_dealt:
            return None

        return forbidden

    def is_valid_bid(self, bid: int, is_dealer: bool,
                     current_total_bids: int, cards_dealt: int) -> bool:
        """
        Validate if a bid is legal for a player.

        Validation rules:
        1. General: 0 <= bid <= cards_dealt
        2. Dealer only: bid != forbidden_bid (where forbidden_bid = cards_dealt - current_total_bids)

        Args:
            bid: The bid value to validate
            is_dealer: Whether the player is the dealer
            current_total_bids: Sum of all bids made by previous players
            cards_dealt: Number of cards dealt to each player

        Returns:
            True if bid is valid, False otherwise

        Examples:
            >>> # Non-dealer can bid anything in range
            >>> game.is_valid_bid(3, is_dealer=False, current_total_bids=0, cards_dealt=5)
            True

            >>> # Dealer cannot bid forbidden value
            >>> game.is_valid_bid(2, is_dealer=True, current_total_bids=3, cards_dealt=5)
            False  # 2 is forbidden (3 + 2 = 5)

            >>> # Bid out of range is invalid for everyone
            >>> game.is_valid_bid(6, is_dealer=False, current_total_bids=0, cards_dealt=5)
            False
        """
        # Check general constraint: 0 <= bid <= cards_dealt
        if bid < 0 or bid > cards_dealt:
            return False

        # Check dealer constraint
        if is_dealer:
            forbidden_bid = self.get_forbidden_bid(current_total_bids, cards_dealt)
            if forbidden_bid is not None and bid == forbidden_bid:
                return False

        return True

    def bidding_phase(self) -> None:
        """
        Execute the bidding phase for current round.

        Collects bids from all players sequentially, starting with the player
        to the left of the dealer. Enforces the dealer constraint that prevents
        the total bids from equaling the number of cards dealt.

        The method updates each player's bid and transitions the game phase
        to 'playing' when complete.

        Raises:
            GameStateException: If game is not in 'bidding' phase
            InvalidBidException: If a player attempts an invalid bid

        Example:
            >>> game = BlobGame(num_players=4)
            >>> game.setup_round(5)
            >>> game.bidding_phase()  # Would normally prompt for bids
            >>> game.game_phase
            'playing'
            >>> all(p.bid is not None for p in game.players)
            True

        Note:
            In actual usage, this method would need to be integrated with
            a bid selection mechanism (human input, bot AI, etc.). For now,
            it serves as a placeholder for the bidding phase structure.
        """
        if self.game_phase != 'bidding':
            raise GameStateException(
                f"Cannot start bidding phase: game is in '{self.game_phase}' phase"
            )

        # Calculate cards dealt (all players should have same number)
        cards_dealt = len(self.players[0].hand) if self.players else 0

        # Determine bidding order: start with player left of dealer
        bidding_order = []
        for i in range(self.num_players):
            player_idx = (self.dealer_position + 1 + i) % self.num_players
            bidding_order.append(player_idx)

        # Collect bids from all players
        current_total_bids = 0

        for i, player_idx in enumerate(bidding_order):
            player = self.players[player_idx]
            is_dealer = (player_idx == self.dealer_position)

            # For now, this is a structural placeholder
            # In actual implementation, this would call a bid selection function
            # that returns a valid bid from the player (human or AI)
            #
            # Example usage:
            #   bid = get_player_bid(player, cards_dealt, is_dealer, current_total_bids)
            #   if not self.is_valid_bid(bid, is_dealer, current_total_bids, cards_dealt):
            #       raise InvalidBidException(f"Invalid bid {bid} for {player.name}")
            #   player.make_bid(bid)
            #   current_total_bids += bid

            # For structural completeness, we'll raise an error if called directly
            # without a bid selection mechanism
            raise NotImplementedError(
                "bidding_phase() requires integration with a bid selection mechanism. "
                "Use this method as a template and provide player bids via a "
                "callback function or override this method."
            )

        # Transition to playing phase
        self.game_phase = 'playing'
