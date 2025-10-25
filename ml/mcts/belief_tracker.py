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

        # Probability distributions: P(player has card | observations)
        # Shape: Dict[player_position, Dict[Card, float]]
        self.card_probabilities: Dict[int, Dict[Card, float]] = {}
        self._initialize_probabilities()

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

    def _initialize_probabilities(self):
        """
        Initialize uniform probability distributions for opponent hands.

        Each opponent has equal probability of having any unseen card that
        satisfies their constraints.
        """
        num_unseen = len(self.unseen_cards)

        if num_unseen == 0:
            # No unseen cards, nothing to initialize
            return

        for player_pos, constraints in self.player_constraints.items():
            self.card_probabilities[player_pos] = {}

            # Uniform distribution over unseen cards
            for card in self.unseen_cards:
                if constraints.can_have_card(card):
                    # P(card | player) = cards_in_hand / total_unseen
                    self.card_probabilities[player_pos][card] = (
                        constraints.cards_in_hand / num_unseen
                    )
                else:
                    self.card_probabilities[player_pos][card] = 0.0

    def update_probabilities_on_card_played(
        self, player_position: int, card: Card, led_suit: Optional[str]
    ):
        """
        Update probabilities using Bayesian inference.

        When a player plays a card, update beliefs about their remaining hand.

        Args:
            player_position: Position of player who played the card
            card: Card that was played
            led_suit: Suit that was led (None if first card of trick)
        """
        if player_position == self.observer.position:
            return  # We know our own hand

        # Zero out probability for the played card (now known)
        if card in self.card_probabilities[player_position]:
            self.card_probabilities[player_position][card] = 0.0

        # If player didn't follow suit, zero out all cards in that suit
        if led_suit and card.suit != led_suit:
            for unseen_card in self.unseen_cards:
                if unseen_card.suit == led_suit:
                    self.card_probabilities[player_position][unseen_card] = 0.0

        # Re-normalize probabilities
        self._normalize_probabilities(player_position)

    def _normalize_probabilities(self, player_position: int):
        """
        Normalize probabilities to sum to cards_in_hand.

        Args:
            player_position: Player whose probabilities to normalize
        """
        if player_position not in self.player_constraints:
            return

        constraints = self.player_constraints[player_position]
        total_prob = sum(self.card_probabilities[player_position].values())

        if total_prob > 0:
            scale = constraints.cards_in_hand / total_prob
            for card in self.card_probabilities[player_position]:
                self.card_probabilities[player_position][card] *= scale

    def get_card_probability(self, player_position: int, card: Card) -> float:
        """
        Get probability that a specific player has a specific card.

        Args:
            player_position: Player to query
            card: Card to query

        Returns:
            Probability [0.0, 1.0]
        """
        if player_position == self.observer.position:
            return 1.0 if card in self.known_cards else 0.0

        return self.card_probabilities[player_position].get(card, 0.0)

    def get_most_likely_holder(self, card: Card) -> tuple[int, float]:
        """
        Get the player most likely to hold a specific card.

        Args:
            card: Card to query

        Returns:
            Tuple of (player_position, probability)
        """
        if card in self.known_cards:
            return (self.observer.position, 1.0)

        best_player = None
        best_prob = 0.0

        for player_pos in self.player_constraints:
            prob = self.get_card_probability(player_pos, card)
            if prob > best_prob:
                best_prob = prob
                best_player = player_pos

        if best_player is None:
            # Return observer with 0 probability if no one else can have it
            return (self.observer.position, 0.0)

        return (best_player, best_prob)

    def get_entropy(self, player_position: int) -> float:
        """
        Calculate information entropy of belief about a player's hand.

        High entropy = uncertain, Low entropy = confident

        Args:
            player_position: Player to query

        Returns:
            Shannon entropy in bits
        """
        if player_position == self.observer.position:
            return 0.0  # Perfect information about own hand

        probs = self.card_probabilities[player_position].values()
        entropy = 0.0

        for p in probs:
            if p > 0:
                entropy -= p * np.log2(p)

        return entropy

    def update_from_trick(self, trick):
        """
        Update belief state from a completed trick.

        Args:
            trick: Trick object with all cards played
        """
        led_suit = trick.led_suit

        for player, card in trick.cards_played:
            self.update_on_card_played(player, card, led_suit)
            self.update_probabilities_on_card_played(player.position, card, led_suit)

    def get_belief_summary(self) -> str:
        """
        Get human-readable summary of current beliefs.

        Useful for debugging and explainability.

        Returns:
            Multi-line string describing belief state
        """
        lines = ["Belief State Summary:"]
        lines.append(f"Observer: Player {self.observer.position}")
        lines.append(f"Unseen cards: {len(self.unseen_cards)}")

        for player_pos, constraints in self.player_constraints.items():
            lines.append(f"\nPlayer {player_pos}:")
            lines.append(f"  Cards in hand: {constraints.cards_in_hand}")
            lines.append(f"  Cannot have suits: {constraints.cannot_have_suits}")
            lines.append(f"  Must have suits: {constraints.must_have_suits}")
            lines.append(f"  Cards played: {len(constraints.cards_played)}")

            # Entropy
            entropy = self.get_entropy(player_pos)
            lines.append(f"  Belief entropy: {entropy:.2f} bits")

        return "\n".join(lines)

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

        # Deep copy card probabilities
        new_belief.card_probabilities = {}
        for pos, card_probs in self.card_probabilities.items():
            new_belief.card_probabilities[pos] = card_probs.copy()

        return new_belief
