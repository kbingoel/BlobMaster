"""
Core game logic for Blob card game.

This module implements the complete game engine including Card, Deck, Player,
Trick, and BlobGame classes with full anti-cheat validation.

Code Quality Status (as of Phase 1a.10 - 2025-10-23):
- Test Coverage: 97% total (blob.py: 88%, constants.py: 100%)
  * 135 tests, all passing
  * Missing coverage: NotImplementedError template methods, some new Phase 2 methods
- Black: PASS (code reformatted successfully)
- Flake8: PASS (no linting errors with --max-line-length=100)
- Mypy: 12 type warnings (non-critical, mostly pre-existing union types)
  * All warnings relate to optional types and duck typing
  * Code runs correctly despite warnings
"""

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Dict, Union
import random
import copy
from ml.game.constants import SUITS, RANKS, RANK_VALUES, TRUMP_ROTATION


# ============================================================================
# Custom Exceptions
# ============================================================================


class BlobGameException(Exception):
    """Base exception for Blob game errors."""

    pass


class IllegalPlayException(BlobGameException):
    """Raised when player attempts an illegal card play."""

    def __init__(self, player_name: str, card: "Card", reason: str):
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

    def __lt__(self, other: "Card") -> bool:
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

        Winner logic (priority order):
        1. If trump cards played: highest trump wins
        2. Otherwise: highest card in led suit wins

        Returns:
            Player who won the trick

        Raises:
            GameStateException: If no cards have been played
        """
        if not self.cards_played:
            raise GameStateException("Cannot determine winner: no cards played")

        # PRIORITY 1: Check for trump cards (if trump suit exists this round)
        # Trump cards beat any non-trump card, regardless of led suit
        if self.trump_suit is not None:
            trump_cards = [
                (p, c) for p, c in self.cards_played if c.suit == self.trump_suit
            ]

            if trump_cards:
                # Multiple trump cards played: highest value trump wins
                winner = max(trump_cards, key=lambda x: x[1].value)
                self.winner = winner[0]
                return self.winner

        # PRIORITY 2: No trump cards played (or no-trump round)
        # Highest card in led suit wins
        # Note: In no-trump rounds, led suit functions like trump for that trick
        led_suit_cards = [
            (p, c) for p, c in self.cards_played if c.suit == self.led_suit
        ]

        if not led_suit_cards:
            # Edge case: shouldn't happen in valid game, but handle it
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
            Player(name, position) for position, name in enumerate(player_names)
        ]

        # Initialize deck
        self.deck = Deck()

        # Game state
        self.current_round: int = 0
        self.trump_suit: Optional[str] = None
        self.dealer_position: int = 0
        self.current_trick: Optional[Trick] = None
        self.tricks_history: List[Trick] = []
        self.game_phase: str = "setup"

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
        self.game_phase = "bidding"

    def get_forbidden_bid(
        self, current_total_bids: int, cards_dealt: int
    ) -> Optional[int]:
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
        # Calculate what bid would make total equal to cards dealt
        # This is the "forbidden bid" - the one value the dealer can't choose
        # Example: 5 cards, others bid 3 total → forbidden = 5 - 3 = 2
        forbidden = cards_dealt - current_total_bids

        # Validate forbidden bid is in legal range [0, cards_dealt]
        # If it's out of range, there's no actual forbidden bid
        # (e.g., if others already bid more than cards dealt)
        if forbidden < 0 or forbidden > cards_dealt:
            return None

        return forbidden

    def is_valid_bid(
        self, bid: int, is_dealer: bool, current_total_bids: int, cards_dealt: int
    ) -> bool:
        """
        Validate if a bid is legal for a player.

        Validation rules:
        1. General: 0 <= bid <= cards_dealt
        2. Dealer only: bid != forbidden_bid
           (where forbidden_bid = cards_dealt - current_total_bids)

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

        **NOTE: This is a structural template method.**
        For actual game play, use `play_round()` with callback functions for bid selection.
        This method raises NotImplementedError when called directly without a bid mechanism.

        Collects bids from all players sequentially, starting with the player
        to the left of the dealer. Enforces the dealer constraint that prevents
        the total bids from equaling the number of cards dealt.

        The method updates each player's bid and transitions the game phase
        to 'playing' when complete.

        Raises:
            GameStateException: If game is not in 'bidding' phase
            InvalidBidException: If a player attempts an invalid bid
            NotImplementedError: If called without bid selection mechanism

        Example Usage:
            See `play_round()` method for complete implementation with callbacks.

        Structural Template Example:
            >>> game = BlobGame(num_players=4)
            >>> game.setup_round(5)
            >>> # This would raise NotImplementedError:
            >>> # game.bidding_phase()
            >>> # Instead use:
            >>> result = game.play_round(5, get_bid_func, get_card_func)
        """
        if self.game_phase != "bidding":
            raise GameStateException(
                f"Cannot start bidding phase: game is in '{self.game_phase}' phase"
            )

        # Calculate cards dealt (all players should have same number)
        _cards_dealt = len(self.players[0].hand) if self.players else 0  # noqa: F841

        # Determine bidding order: start with player left of dealer
        bidding_order = []
        for i in range(self.num_players):
            player_idx = (self.dealer_position + 1 + i) % self.num_players
            bidding_order.append(player_idx)

        # Collect bids from all players
        _current_total_bids = 0  # noqa: F841

        for i, player_idx in enumerate(bidding_order):
            _player = self.players[player_idx]  # noqa: F841
            _is_dealer = player_idx == self.dealer_position  # noqa: F841

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
        self.game_phase = "playing"

    def get_legal_plays(self, player: Player, led_suit: Optional[str]) -> List[Card]:
        """
        Get legal cards that a player can play.

        Rules:
        - If no suit has been led (first card of trick): can play any card
        - If player has cards in led suit: must play one of those cards
        - If player has no cards in led suit: can play any card

        Args:
            player: Player whose legal plays to check
            led_suit: Suit of the first card played in trick (None if first card)

        Returns:
            List of cards the player can legally play

        Examples:
            >>> # Player has [5♥, 8♥, 3♣, K♦]
            >>> # Led suit is ♥
            >>> game.get_legal_plays(player, '♥')
            [5♥, 8♥]  # Must follow suit

            >>> # Player has [3♣, K♦, 7♠]
            >>> # Led suit is ♥ (player has no hearts)
            >>> game.get_legal_plays(player, '♥')
            [3♣, K♦, 7♠]  # Can play any card
        """
        # If no suit led yet (first card of trick), can play any card
        if led_suit is None:
            return list(player.hand)

        # Get cards in led suit
        cards_in_led_suit = [card for card in player.hand if card.suit == led_suit]

        # If player has cards in led suit, must play one of them
        if cards_in_led_suit:
            return cards_in_led_suit

        # Player has no cards in led suit, can play any card
        return list(player.hand)

    def is_valid_play(
        self, card: Card, player: Player, led_suit: Optional[str]
    ) -> bool:
        """
        Check if a card play is legal.

        Validation rules:
        1. Card must be in player's hand
        2. If led_suit exists and player has cards in that suit, card must match led_suit

        Args:
            card: Card to validate
            player: Player attempting to play the card
            led_suit: Suit of first card in trick (None if first card)

        Returns:
            True if play is legal, False otherwise

        Examples:
            >>> # Player has [5♥, 3♣, K♦], led suit is ♥
            >>> game.is_valid_play(Card('5', '♥'), player, '♥')
            True

            >>> # Player has [5♥, 3♣], led suit is ♥, trying to play ♣
            >>> game.is_valid_play(Card('3', '♣'), player, '♥')
            False  # Must follow suit
        """
        # Check if card is in player's hand
        if card not in player.hand:
            return False

        # Get legal plays
        legal_cards = self.get_legal_plays(player, led_suit)

        # Check if card is in legal plays
        return card in legal_cards

    def validate_play_with_anti_cheat(
        self, card: Card, player: Player, led_suit: Optional[str]
    ) -> None:
        """
        Strictly validate card play with anti-cheat detection.

        This method performs thorough validation and raises exceptions for
        illegal plays. It's designed to catch both honest mistakes and
        deliberate cheating attempts.

        Validation steps:
        1. Verify card is in player's hand
        2. If led_suit exists and player has cards in that suit:
           - If card doesn't match led_suit: RAISE IllegalPlayException
        3. If player doesn't have led_suit, mark them as void in that suit

        Args:
            card: Card being played
            player: Player attempting to play the card
            led_suit: Suit of first card in trick (None if first card)

        Raises:
            IllegalPlayException: If player attempts an illegal move

        Examples:
            >>> # Player has [5♥, 3♣], tries to play ♣ when ♥ is led
            >>> game.validate_play_with_anti_cheat(Card('3', '♣'), player, '♥')
            IllegalPlayException: Player played 3♣ illegally: must follow suit ♥

            >>> # Player has [3♣, K♦] (no hearts), plays ♣ when ♥ is led
            >>> game.validate_play_with_anti_cheat(Card('3', '♣'), player, '♥')
            # No exception, and player.known_void_suits now contains ♥
        """
        # STEP 1: Verify card is in player's hand
        # This catches attempts to play cards the player doesn't have
        if card not in player.hand:
            raise IllegalPlayException(
                player.name, card, f"card not in hand (hand: {sorted(player.hand)})"
            )

        # STEP 2: Check follow-suit rule (if a suit has been led)
        if led_suit is not None:
            # Check if player has any cards in the led suit
            has_led_suit = player.has_suit(led_suit)

            if has_led_suit and card.suit != led_suit:
                # ANTI-CHEAT DETECTION: Player has led suit but played different suit
                # This is either a mistake or deliberate cheating
                raise IllegalPlayException(
                    player.name,
                    card,
                    f"must follow suit {led_suit} (player has cards in that suit)",
                )

    def update_card_counting(
        self, card: Card, player: Player, led_suit: Optional[str]
    ) -> None:
        """
        Update card counting state after a valid play.

        Tracks:
        1. Cards played this round (global history)
        2. Cards remaining by suit (for card counting)
        3. Void suits (when player doesn't follow suit)

        Args:
            card: Card that was played
            player: Player who played the card
            led_suit: Suit that was led (None if first card)

        Examples:
            >>> # Before: cards_remaining_by_suit = {'♥': 5, '♣': 8, ...}
            >>> game.update_card_counting(Card('5', '♥'), player, None)
            >>> # After: cards_remaining_by_suit = {'♥': 4, '♣': 8, ...}
            >>> # And cards_played_this_round contains Card('5', '♥')
        """
        # Add card to round history
        self.cards_played_this_round.append(card)

        # Decrement cards remaining in that suit
        self.cards_remaining_by_suit[card.suit] -= 1

        # If player didn't follow suit and a suit was led, mark them as void
        if led_suit is not None and card.suit != led_suit:
            player.mark_void_suit(led_suit)

    def play_trick(self) -> Player:
        """
        Execute one complete trick.

        Flow:
        1. Create new Trick with current trump
        2. Loop through players (starting with lead player)
        3. For each player:
           - Get legal plays
           - Get player's card selection (must be provided externally)
           - Validate play with anti-cheat
           - Add card to trick
           - Update card counting
        4. Determine winner after all cards played
        5. Award trick to winner
        6. Add trick to history
        7. Return winner for next trick's lead

        Returns:
            Player who won the trick (will lead next trick)

        Raises:
            GameStateException: If game is not in 'playing' phase
            NotImplementedError: If called without card selection mechanism

        Note:
            This method requires integration with a card selection mechanism
            (human input, AI decision, etc.). It serves as a structural template.
        """
        if self.game_phase != "playing":
            raise GameStateException(
                f"Cannot play trick: game is in '{self.game_phase}' phase"
            )

        # Create new trick
        self.current_trick = Trick(self.trump_suit)

        # Determine lead player
        # First trick: player left of dealer leads
        # Subsequent tricks: winner of last trick leads
        if not self.tricks_history:
            lead_player_idx = (self.dealer_position + 1) % self.num_players
        else:
            # Last trick's winner leads
            last_winner = self.tricks_history[-1].winner
            lead_player_idx = last_winner.position

        # Play order: start with lead player, go clockwise
        play_order = []
        for i in range(self.num_players):
            player_idx = (lead_player_idx + i) % self.num_players
            play_order.append(player_idx)

        # Each player plays a card
        for player_idx in play_order:
            player = self.players[player_idx]

            # Get legal plays for this player
            _legal_cards = self.get_legal_plays(  # noqa: F841
                player, self.current_trick.led_suit
            )

            # This is where we'd integrate with card selection mechanism
            # Example:
            #   card = get_player_card_choice(player, legal_cards, self.current_trick)
            #   self.validate_play_with_anti_cheat(card, player, self.current_trick.led_suit)
            #   played_card = player.play_card(card)
            #   self.current_trick.add_card(player, played_card)
            #   self.update_card_counting(played_card, player, self.current_trick.led_suit)

            raise NotImplementedError(
                "play_trick() requires integration with a card selection mechanism. "
                "Use this method as a template and provide card choices via a "
                "callback function or override this method."
            )

        # Determine winner
        winner = self.current_trick.determine_winner()
        winner.win_trick()

        # Add to history
        self.tricks_history.append(self.current_trick)

        return winner

    def playing_phase(self) -> None:
        """
        Execute the playing phase for current round.

        **NOTE: This is a structural template method.**
        For actual game play, use `play_round()` with callback functions for card selection.
        This method calls play_trick() which raises NotImplementedError without a card mechanism.

        Plays all tricks for the round. Each trick is played in sequence,
        with the winner of each trick leading the next trick.

        The method continues until all cards have been played (number of
        tricks equals the number of cards dealt to each player).

        Raises:
            GameStateException: If game is not in 'playing' phase
            NotImplementedError: If called without card selection mechanism (via play_trick)

        Example Usage:
            See `play_round()` method for complete implementation with callbacks.

        Structural Template Example:
            >>> game = BlobGame(num_players=4)
            >>> game.setup_round(5)
            >>> # This would raise NotImplementedError:
            >>> # game.playing_phase()
            >>> # Instead use:
            >>> result = game.play_round(5, get_bid_func, get_card_func)
        """
        if self.game_phase != "playing":
            raise GameStateException(
                f"Cannot start playing phase: game is in '{self.game_phase}' phase"
            )

        # Calculate number of tricks to play (equals cards dealt)
        cards_dealt = len(self.players[0].hand) if self.players else 0

        # Play all tricks
        for trick_num in range(cards_dealt):
            self.play_trick()

        # Transition to scoring phase
        self.game_phase = "scoring"

    def scoring_phase(self) -> Dict[str, Union[int, List[Dict]]]:
        """
        Execute the scoring phase for current round.

        Calculates scores for all players based on whether they met their bids,
        updates total scores, and prepares for the next round.

        Scoring rule: If tricks_won == bid, score = 10 + bid. Otherwise, score = 0.

        Returns:
            Dictionary containing round results:
            {
                'round': int,
                'trump_suit': Optional[str],
                'player_scores': [
                    {
                        'name': str,
                        'bid': int,
                        'tricks_won': int,
                        'round_score': int,
                        'total_score': int
                    }, ...
                ]
            }

        Raises:
            GameStateException: If game is not in 'scoring' phase

        Example:
            >>> game = BlobGame(num_players=4)
            >>> game.setup_round(5)
            >>> # ... bidding and playing happens ...
            >>> results = game.scoring_phase()
            >>> results['round']
            0
            >>> results['player_scores'][0]['total_score']
            12  # Example: if player bid 2 and won 2 tricks
        """
        if self.game_phase != "scoring":
            raise GameStateException(
                f"Cannot start scoring phase: game is in '{self.game_phase}' phase"
            )

        # Prepare results dictionary
        results = {
            "round": self.current_round,
            "trump_suit": self.trump_suit,
            "player_scores": [],
        }

        # Calculate and update scores for all players
        for player in self.players:
            # Calculate round score
            round_score = player.calculate_round_score()

            # Update total score
            player.total_score += round_score

            # Record results for this player
            player_result = {
                "name": player.name,
                "bid": player.bid if player.bid is not None else 0,
                "tricks_won": player.tricks_won,
                "round_score": round_score,
                "total_score": player.total_score,
            }
            results["player_scores"].append(player_result)

        # Increment round counter
        self.current_round += 1

        # Rotate dealer position for next round
        self.dealer_position = (self.dealer_position + 1) % self.num_players

        # Transition to complete phase (can setup next round or end game)
        self.game_phase = "complete"

        return results

    def play_round(self, cards_to_deal: int, get_bid_func, get_card_func) -> Dict:
        """
        Execute a complete round of play.

        Orchestrates the full round flow: setup → bidding → playing → scoring.
        Uses provided callback functions to get player decisions.

        Args:
            cards_to_deal: Number of cards to deal to each player
            get_bid_func: Callback function with signature:
                (player: Player, cards_dealt: int, is_dealer: bool,
                 current_total_bids: int, cards_dealt: int) -> int
                Returns the bid for a player.
            get_card_func: Callback function with signature:
                (player: Player, legal_cards: List[Card], trick: Trick) -> Card
                Returns the card choice for a player.

        Returns:
            Dictionary with round results (same format as scoring_phase())

        Raises:
            ValueError: If cards_to_deal invalid
            InvalidBidException: If callback returns invalid bid
            IllegalPlayException: If callback returns illegal card

        Example:
            >>> def simple_bid(player, cards, is_dealer, total, cards_dealt):
            ...     return 0  # Always bid 0
            >>> def simple_play(player, legal_cards, trick):
            ...     return legal_cards[0]  # Play first legal card
            >>> game = BlobGame(num_players=4)
            >>> result = game.play_round(5, simple_bid, simple_play)
            >>> result['round']
            0
        """
        # Phase 1: Setup
        self.setup_round(cards_to_deal)

        # Phase 2: Bidding
        # Collect bids from all players sequentially
        bidding_order = []
        for i in range(self.num_players):
            player_idx = (self.dealer_position + 1 + i) % self.num_players
            bidding_order.append(player_idx)

        current_total_bids = 0

        for player_idx in bidding_order:
            player = self.players[player_idx]
            is_dealer = player_idx == self.dealer_position

            # Get bid from callback
            bid = get_bid_func(
                player, cards_to_deal, is_dealer, current_total_bids, cards_to_deal
            )

            # Validate bid
            if not self.is_valid_bid(bid, is_dealer, current_total_bids, cards_to_deal):
                forbidden = self.get_forbidden_bid(current_total_bids, cards_to_deal)
                raise InvalidBidException(
                    f"{player.name} bid {bid} is invalid. "
                    f"Valid range: 0-{cards_to_deal}"
                    f"{f', forbidden: {forbidden}' if is_dealer and forbidden is not None else ''}"
                )

            # Accept bid
            player.make_bid(bid)
            current_total_bids += bid

        # Transition to playing phase
        self.game_phase = "playing"

        # Phase 3: Playing
        # Play all tricks for the round
        for trick_num in range(cards_to_deal):
            # Create new trick
            self.current_trick = Trick(self.trump_suit)

            # Determine lead player
            if not self.tricks_history:
                # First trick: player left of dealer leads
                lead_player_idx = (self.dealer_position + 1) % self.num_players
            else:
                # Subsequent tricks: winner of last trick leads
                last_winner = self.tricks_history[-1].winner
                lead_player_idx = last_winner.position

            # Play order: start with lead player, go clockwise
            play_order = []
            for i in range(self.num_players):
                player_idx = (lead_player_idx + i) % self.num_players
                play_order.append(player_idx)

            # Each player plays a card
            for player_idx in play_order:
                player = self.players[player_idx]

                # Get legal plays for this player
                legal_cards = self.get_legal_plays(player, self.current_trick.led_suit)

                # Get card choice from callback
                card = get_card_func(player, legal_cards, self.current_trick)

                # Validate play with anti-cheat
                self.validate_play_with_anti_cheat(
                    card, player, self.current_trick.led_suit
                )

                # Play the card
                played_card = player.play_card(card)
                self.current_trick.add_card(player, played_card)

                # Update card counting
                self.update_card_counting(
                    played_card, player, self.current_trick.led_suit
                )

            # Determine winner
            winner = self.current_trick.determine_winner()
            winner.win_trick()

            # Add to history
            self.tricks_history.append(self.current_trick)

        # Transition to scoring phase
        self.game_phase = "scoring"

        # Phase 4: Scoring
        results = self.scoring_phase()

        return results

    def play_full_game(
        self, round_structure: List[int], get_bid_func, get_card_func
    ) -> Dict:
        """
        Play a complete multi-round game.

        Executes multiple rounds with varying card counts, tracking all results
        and final scores.

        Args:
            round_structure: List of cards to deal per round
                Example: [5, 4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4, 5]
            get_bid_func: Callback for getting player bids (see play_round())
            get_card_func: Callback for getting card choices (see play_round())

        Returns:
            Dictionary containing complete game results:
            {
                'num_rounds': int,
                'num_players': int,
                'round_results': [result_dict_per_round],
                'final_scores': [
                    {'name': str, 'total_score': int, 'position': int},
                    ...
                ],
                'winner': {'name': str, 'score': int}
            }

        Example:
            >>> from ml.game.constants import generate_round_structure
            >>> game = BlobGame(num_players=4)
            >>> rounds = generate_round_structure(5, 4)
            >>> results = game.play_full_game(rounds, simple_bid, simple_play)
            >>> results['num_rounds']
            13
            >>> results['winner']['name']
            'Player 1'
        """
        # Initialize game history
        round_results = []

        # Play each round
        for round_idx, cards_to_deal in enumerate(round_structure):
            result = self.play_round(cards_to_deal, get_bid_func, get_card_func)
            round_results.append(result)

        # Compile final results
        final_scores = []
        for player in self.players:
            final_scores.append(
                {
                    "name": player.name,
                    "position": player.position,
                    "total_score": player.total_score,
                }
            )

        # Sort by score descending to find winner
        final_scores_sorted = sorted(
            final_scores, key=lambda x: x["total_score"], reverse=True
        )

        # Determine winner
        winner = final_scores_sorted[0]

        return {
            "num_rounds": len(round_structure),
            "num_players": self.num_players,
            "round_results": round_results,
            "final_scores": final_scores_sorted,
            "winner": {"name": winner["name"], "score": winner["total_score"]},
        }

    def get_game_state(self, player_perspective: Optional[Player] = None) -> Dict:
        """
        Get complete observable game state.

        Returns all game information needed for AI decision-making and UI display.
        Supports imperfect information by hiding opponent hands when a player
        perspective is specified.

        Args:
            player_perspective: If provided, only show this player's hand (imperfect info).
                               If None, show all hands (debug/UI mode).

        Returns:
            Dictionary containing complete game state:
            {
                'phase': str,  # 'setup', 'bidding', 'playing', 'scoring', 'complete'
                'round': int,
                'trump': Optional[str],
                'dealer_position': int,
                'num_players': int,
                'players': [
                    {
                        'name': str,
                        'position': int,
                        'hand': List[str] or None,  # Card strings like 'A♠'
                        'hand_size': int,
                        'bid': Optional[int],
                        'tricks_won': int,
                        'total_score': int,
                        'known_void_suits': List[str]
                    }, ...
                ],
                'current_trick': {
                    'cards_played': [(player_name, card_str), ...],
                    'led_suit': Optional[str],
                    'trump_suit': Optional[str]
                } or None,
                'tricks_history': [
                    {
                        'winner': str,  # player name
                        'cards': [(player_name, card_str), ...]
                    }, ...
                ],
                'cards_remaining_by_suit': Dict[str, int]
            }

        Example:
            >>> game = BlobGame(num_players=4)
            >>> game.setup_round(5)
            >>> state = game.get_game_state()
            >>> state['phase']
            'bidding'
            >>> state['trump']
            '♠'
            >>> len(state['players'])
            4

            >>> # Get state from player 0's perspective (hide other hands)
            >>> state = game.get_game_state(game.players[0])
            >>> state['players'][0]['hand']  # Player 0's hand visible
            ['2♠', '5♥', ...]
            >>> state['players'][1]['hand']  # Other players' hands hidden
            None
        """
        # Build players state
        players_state = []
        for player in self.players:
            # Determine if this player's hand should be visible
            show_hand = player_perspective is None or player == player_perspective

            player_state = {
                "name": player.name,
                "position": player.position,
                "hand": (
                    [str(card) for card in sorted(player.hand)] if show_hand else None
                ),
                "hand_size": len(player.hand),
                "bid": player.bid,
                "tricks_won": player.tricks_won,
                "total_score": player.total_score,
                "known_void_suits": sorted(list(player.known_void_suits)),
            }
            players_state.append(player_state)

        # Build current trick state
        current_trick_state = None
        if self.current_trick is not None:
            current_trick_state = {
                "cards_played": [
                    (player.name, str(card))
                    for player, card in self.current_trick.cards_played
                ],
                "led_suit": self.current_trick.led_suit,
                "trump_suit": self.current_trick.trump_suit,
            }

        # Build tricks history state
        tricks_history_state = []
        for trick in self.tricks_history:
            trick_state = {
                "winner": trick.winner.name if trick.winner else None,
                "cards": [
                    (player.name, str(card)) for player, card in trick.cards_played
                ],
            }
            tricks_history_state.append(trick_state)

        # Assemble complete state
        state = {
            "phase": self.game_phase,
            "round": self.current_round,
            "trump": self.trump_suit,
            "dealer_position": self.dealer_position,
            "num_players": self.num_players,
            "players": players_state,
            "current_trick": current_trick_state,
            "tricks_history": tricks_history_state,
            "cards_remaining_by_suit": dict(self.cards_remaining_by_suit),
        }

        return state

    def get_legal_actions(self, player: Player) -> Union[List[int], List[Card]]:
        """
        Get valid actions for a player based on current game phase.

        Used by AI agents to determine what actions are available.

        Args:
            player: Player whose legal actions to check

        Returns:
            - If 'bidding' phase: List of valid bid integers
            - If 'playing' phase: List of legal Card objects
            - Otherwise: Empty list

        Raises:
            GameStateException: If player not found or invalid state

        Example:
            >>> game = BlobGame(num_players=4)
            >>> game.setup_round(5)
            >>> # In bidding phase
            >>> actions = game.get_legal_actions(game.players[0])
            >>> actions
            [0, 1, 2, 3, 4, 5]

            >>> # After dealing, in playing phase with hearts led
            >>> game.game_phase = 'playing'
            >>> game.current_trick = Trick('♠')
            >>> game.current_trick.led_suit = '♥'
            >>> actions = game.get_legal_actions(game.players[0])
            >>> # Returns only hearts if player has hearts, else all cards
        """
        if player not in self.players:
            raise GameStateException(f"Player {player.name} not in game")

        if self.game_phase == "bidding":
            # Return valid bids
            cards_dealt = len(player.hand) if player.hand else 0
            is_dealer = player.position == self.dealer_position

            # Calculate current total bids (all players who have bid before this one)
            current_total_bids = 0
            bidding_order_start = (self.dealer_position + 1) % self.num_players

            # Count bids up to this player in bidding order
            for i in range(self.num_players):
                check_idx = (bidding_order_start + i) % self.num_players
                check_player = self.players[check_idx]

                if check_player == player:
                    break  # Stop when we reach the player in question

                if check_player.bid is not None:
                    current_total_bids += check_player.bid

            # Generate list of valid bids
            valid_bids = []
            for bid in range(cards_dealt + 1):
                if self.is_valid_bid(bid, is_dealer, current_total_bids, cards_dealt):
                    valid_bids.append(bid)

            return valid_bids

        elif self.game_phase == "playing":
            # Return legal cards
            if self.current_trick is None:
                # No trick in progress, shouldn't happen but return all cards
                return list(player.hand)

            return self.get_legal_plays(player, self.current_trick.led_suit)

        else:
            # No actions available in other phases
            return []

    def copy(self) -> "BlobGame":
        """
        Create deep copy of game state for MCTS simulation.

        Creates a completely independent copy of the entire game state,
        including all players, deck, tricks, and history. Modifications to
        the copy will not affect the original game state.

        Returns:
            Deep copy of the game state

        Note:
            Used by MCTS to simulate game states without modifying original.
            All game state including players, deck, tricks is recursively copied.

        Example:
            >>> game = BlobGame(num_players=4)
            >>> game.setup_round(5)
            >>> game_copy = game.copy()
            >>> game_copy.players[0].make_bid(3)
            >>> game.players[0].bid is None  # Original unchanged
            True
            >>> game_copy.players[0].bid == 3  # Copy modified
            True
        """
        return copy.deepcopy(self)

    def apply_action(self, action: int, player: Player) -> None:
        """
        Apply an action to the game state (for MCTS simulation).

        This method is designed for MCTS to simulate game moves by taking
        action indices and applying them to the game state. The action
        interpretation depends on the current game phase:

        - **Bidding phase**: action is the bid value (0-13)
        - **Playing phase**: action is card index (0-51) mapping to specific card

        Card Index Mapping (0-51):
            - 0-12: ♠ (Spades) 2-A
            - 13-25: ♥ (Hearts) 2-A
            - 26-38: ♣ (Clubs) 2-A
            - 39-51: ♦ (Diamonds) 2-A

        Args:
            action: Action index (bid value or card index)
            player: Player taking the action

        Raises:
            GameStateException: If action cannot be applied in current game phase
            InvalidBidException: If bid action is invalid
            IllegalPlayException: If card play action is invalid

        Example:
            >>> game = BlobGame(num_players=4)
            >>> game.setup_round(5)
            >>> player = game.players[0]
            >>> # Bidding: action is bid value
            >>> game.apply_action(3, player)  # Player bids 3
            >>> player.bid
            3
            >>> # Playing: action is card index
            >>> game.game_phase = 'playing'
            >>> game.current_trick = Trick(game.trump_suit)
            >>> game.apply_action(0, player)  # Play card at index 0 (♠2)
        """
        if self.game_phase == "bidding":
            # BIDDING PHASE: action is the bid value
            if not isinstance(action, int) or action < 0:
                raise InvalidBidException(f"Invalid bid action: {action}")

            # Validate bid is legal (check dealer constraint, range)
            cards_dealt = len(player.hand) if player.hand else 0
            is_dealer = player.position == self.dealer_position

            # Calculate current total bids
            current_total_bids = 0
            bidding_order_start = (self.dealer_position + 1) % self.num_players
            for i in range(self.num_players):
                check_idx = (bidding_order_start + i) % self.num_players
                check_player = self.players[check_idx]
                if check_player == player:
                    break
                if check_player.bid is not None:
                    current_total_bids += check_player.bid

            # Validate bid
            if not self.is_valid_bid(
                action, is_dealer, current_total_bids, cards_dealt
            ):
                forbidden_bid = self.get_forbidden_bid(current_total_bids, cards_dealt)
                raise InvalidBidException(
                    f"Invalid bid {action} for {player.name}: "
                    f"dealer={is_dealer}, forbidden_bid={forbidden_bid}"
                )

            # Apply bid
            player.make_bid(action)

        elif self.game_phase == "playing":
            # PLAYING PHASE: action is card index (0-51)
            # Convert card index to Card object using the standard mapping

            # Validate action is in valid range
            if not isinstance(action, int) or action < 0 or action >= 52:
                raise IllegalPlayException(
                    player.name,
                    None,
                    f"Invalid card index: {action} (must be 0-51)",
                )

            # Calculate which card this index represents
            suit_idx = action // 13  # 0=♠, 1=♥, 2=♣, 3=♦
            rank_idx = action % 13  # 0-12 maps to 2-A

            if suit_idx >= len(SUITS) or rank_idx >= len(RANKS):
                raise IllegalPlayException(
                    player.name,
                    None,
                    f"Invalid card index: {action} (suit_idx={suit_idx}, rank_idx={rank_idx})",
                )

            target_suit = SUITS[suit_idx]
            target_rank = RANKS[rank_idx]

            # Find this card in player's hand
            target_card = None
            for card in player.hand:
                if card.suit == target_suit and card.rank == target_rank:
                    target_card = card
                    break

            if target_card is None:
                raise IllegalPlayException(
                    player.name,
                    Card(target_rank, target_suit),
                    f"Card not in hand: {target_rank}{target_suit} (index {action})",
                )

            # Validate play is legal (follow suit rules, anti-cheat)
            led_suit = self.current_trick.led_suit if self.current_trick else None
            self.validate_play_with_anti_cheat(target_card, player, led_suit)

            # Apply the play
            played_card = player.play_card(target_card)
            self.current_trick.add_card(player, played_card)
            self.update_card_counting(played_card, player, led_suit)

        else:
            raise GameStateException(
                f"Cannot apply action in phase '{self.game_phase}' "
                f"(valid phases: 'bidding', 'playing')"
            )
