"""
Belief state tracking for imperfect information games.

This module implements belief tracking to handle hidden opponent hands in
the Blob card game. It maintains probability distributions and constraints
on what cards each opponent could possibly have, updating beliefs as
information is revealed through gameplay.

Core Concepts:
    - We don't know opponent hands, but we know:
      1. Which cards were dealt to us
      2. Which cards have been played
      3. Information revealed by opponent actions (e.g., can't follow suit)

    - Maintain probability distribution over possible opponent hands
    - Update beliefs incrementally as information is revealed

Belief State Components:
    1. Known cards (52-dim binary): Which cards have been seen
    2. Unseen cards: Set of cards not yet observed
    3. Player constraints (per player):
       - Cards they've played (known)
       - Suits they don't have (revealed by failing to follow suit)
       - Number of cards in hand (known from game rules)
    4. Card probability distributions (per player, per card):
       - P(player has card | observations)

Information Updates:
    1. Initial dealing: We see our hand, others are uniform distribution
    2. Card played: Remove card from unseen pool, mark as known
    3. Fail to follow suit: Zero out probability for all cards in that suit
    4. Follow suit: Increase probability for cards in that suit
    5. Trick won: Update probabilities based on play patterns

Mathematical Model:
    - Bayesian inference: P(hand | observations) ∝ P(observations | hand) × P(hand)
    - Constraint satisfaction: Only consider consistent hands
    - Sampling: Generate random hands that satisfy all constraints
"""

import numpy as np
from typing import Dict, Set, List, Optional
from dataclasses import dataclass, field
from ml.game.blob import BlobGame, Player, Card
from ml.game.constants import SUITS, RANKS


@dataclass
class PlayerConstraints:
    """
    Constraints on what cards a player can have.

    Used for belief tracking in imperfect information games.
    Tracks both positive (must have) and negative (cannot have)
    information about a player's hand.

    Attributes:
        player_position: Position of this player in game
        cards_in_hand: Number of cards player currently has
        cards_played: Cards we've seen them play
        cannot_have_suits: Suits they've revealed they don't have
        must_have_suits: Suits they've revealed they have
    """

    player_position: int
    cards_in_hand: int  # Number of cards player currently has
    cards_played: Set[Card] = field(
        default_factory=set
    )  # Cards we've seen them play
    cannot_have_suits: Set[str] = field(
        default_factory=set
    )  # Suits they've revealed they don't have
    must_have_suits: Set[str] = field(
        default_factory=set
    )  # Suits they've revealed they have

    def can_have_card(self, card: Card) -> bool:
        """
        Check if player can possibly have this card.

        Args:
            card: Card to check

        Returns:
            True if card is consistent with all constraints
        """
        # Already played this card
        if card in self.cards_played:
            return False

        # Revealed they don't have this suit
        if card.suit in self.cannot_have_suits:
            return False

        return True


class BeliefState:
    """
    Tracks belief state about opponent hands in imperfect information game.

    Maintains probability distributions and constraints for each player's hand.
    Updates beliefs as information is revealed through card plays and trick outcomes.

    The belief state is from a specific observer's perspective - we know our own
    hand perfectly but must infer opponent hands from observations.

    Attributes:
        game: Current game state
        observer: Player whose perspective we're tracking
        num_players: Number of players in game
        known_cards: Cards we know (observer's hand)
        played_cards: Cards that have been played and are visible
        unseen_cards: Cards not yet observed (candidates for opponent hands)
        player_constraints: Constraints on each opponent's possible hand
    """

    def __init__(self, game: BlobGame, observer: Player):
        """
        Initialize belief state from observer's perspective.

        Args:
            game: Current game state
            observer: Player whose perspective we're tracking
        """
        self.game = game
        self.observer = observer
        self.num_players = len(game.players)

        # Track which cards are known
        self.known_cards: Set[Card] = set(observer.hand)  # We know our own hand

        # Track cards that have been played (visible to everyone)
        self.played_cards: Set[Card] = set()
        for trick in game.tricks_history:
            # Trick.cards_played is a list of (Player, Card) tuples
            self.played_cards.update(card for player, card in trick.cards_played)

        # Cards currently in the current trick (partially visible)
        if hasattr(game, "current_trick") and game.current_trick:
            self.played_cards.update(
                card for player, card in game.current_trick.cards_played
            )

        # Unseen cards (candidates for opponent hands)
        all_cards = set(Card(rank, suit) for suit in SUITS for rank in RANKS)
        self.unseen_cards = all_cards - self.known_cards - self.played_cards

        # Constraints for each player
        self.player_constraints: Dict[int, PlayerConstraints] = {}
        for player in game.players:
            if player.position != observer.position:
                self.player_constraints[player.position] = PlayerConstraints(
                    player_position=player.position,
                    cards_in_hand=len(player.hand),
                )

        # Initialize constraint tracking from game history
        self._initialize_constraints_from_history()

    def _initialize_constraints_from_history(self):
        """
        Initialize constraints by analyzing past tricks.

        Look for situations where players revealed information:
        - Failed to follow suit → don't have that suit
        - Followed suit → have at least one card in that suit
        """
        for trick in self.game.tricks_history:
            if not trick.led_suit:
                continue

            # Trick.cards_played is a list of (Player, Card) tuples
            for player, card in trick.cards_played:
                if player.position == self.observer.position:
                    continue  # Skip observer (we know their hand)

                constraints = self.player_constraints[player.position]

                # Check if player followed suit
                if card.suit != trick.led_suit:
                    # Player didn't follow suit → they don't have that suit
                    constraints.cannot_have_suits.add(trick.led_suit)
                else:
                    # Player followed suit → they had that suit
                    constraints.must_have_suits.add(trick.led_suit)

                # Record card as played
                constraints.cards_played.add(card)

    def update_on_card_played(
        self, player: Player, card: Card, led_suit: Optional[str]
    ):
        """
        Update belief state when a card is played.

        Args:
            player: Player who played the card
            card: Card that was played
            led_suit: Suit that was led (None if first card of trick)
        """
        if player.position == self.observer.position:
            # Observer played a card - update known cards
            self.known_cards.discard(card)
        else:
            # Opponent played a card - update constraints
            constraints = self.player_constraints[player.position]
            constraints.cards_played.add(card)

            # Check for suit information
            if led_suit and card.suit != led_suit:
                # Player didn't follow suit → they don't have that suit
                constraints.cannot_have_suits.add(led_suit)
            elif led_suit and card.suit == led_suit:
                # Player followed suit → they have that suit
                constraints.must_have_suits.add(led_suit)

        # Move card to played set
        self.played_cards.add(card)
        self.unseen_cards.discard(card)

    def get_possible_cards(self, player_position: int) -> Set[Card]:
        """
        Get set of cards that a player could possibly have.

        Args:
            player_position: Position of player to query

        Returns:
            Set of cards consistent with all constraints
        """
        if player_position == self.observer.position:
            return self.known_cards

        constraints = self.player_constraints[player_position]
        possible_cards = set()

        for card in self.unseen_cards:
            if constraints.can_have_card(card):
                possible_cards.add(card)

        return possible_cards

    def is_consistent_hand(self, player_position: int, hand: List[Card]) -> bool:
        """
        Check if a proposed hand is consistent with constraints.

        Args:
            player_position: Position of player
            hand: Proposed hand to validate

        Returns:
            True if hand satisfies all constraints
        """
        if player_position == self.observer.position:
            return set(hand) == self.known_cards

        constraints = self.player_constraints[player_position]

        # Check hand size
        if len(hand) != constraints.cards_in_hand:
            return False

        # Check all cards are possible
        for card in hand:
            if not constraints.can_have_card(card):
                return False

        # Check must-have suits
        hand_suits = set(card.suit for card in hand)
        if not constraints.must_have_suits.issubset(hand_suits):
            return False

        return True

    def copy(self) -> "BeliefState":
        """
        Create a copy of this belief state.

        Returns:
            New BeliefState instance with copied data
        """
        new_belief = BeliefState(self.game, self.observer)
        new_belief.known_cards = self.known_cards.copy()
        new_belief.played_cards = self.played_cards.copy()
        new_belief.unseen_cards = self.unseen_cards.copy()

        # Deep copy constraints
        new_belief.player_constraints = {}
        for pos, constraints in self.player_constraints.items():
            new_belief.player_constraints[pos] = PlayerConstraints(
                player_position=constraints.player_position,
                cards_in_hand=constraints.cards_in_hand,
                cards_played=constraints.cards_played.copy(),
                cannot_have_suits=constraints.cannot_have_suits.copy(),
                must_have_suits=constraints.must_have_suits.copy(),
            )

        return new_belief
