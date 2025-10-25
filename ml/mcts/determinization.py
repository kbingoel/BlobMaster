"""
Determinization sampling for imperfect information MCTS.

This module implements determinization - the process of sampling consistent
opponent hands to convert an imperfect information game into multiple perfect
information game states. This enables standard MCTS to be applied by running
searches on multiple determinized worlds and aggregating results.

Determinization Sampling Strategy:

Goal: Generate complete game states where opponent hands are revealed
      in a way that's consistent with all observations and constraints.

Approach: Constraint Satisfaction Sampling
    1. Start with unseen card pool
    2. For each opponent:
        - Filter cards to only those they can have (constraints)
        - Sample N cards randomly from filtered pool
        - Remove sampled cards from pool
    3. Validate consistency (all constraints satisfied)
    4. Repeat if inconsistent

Challenges:
    - Sampling must respect suit constraints
    - Must handle cases where constraints are tight
    - Need to avoid over-sampling from limited pools

Optimizations:
    - Rejection sampling with early termination
    - Constraint propagation to reduce search space
    - Caching of valid samples

Quality Metrics:
    - Sample diversity (avoid always sampling same hands)
    - Constraint satisfaction rate (% of samples valid)
    - Sampling speed (<10ms per sample)
"""

import random
import numpy as np
from typing import Dict, List, Optional, Set
from ml.game.blob import BlobGame, Card, Player
from ml.mcts.belief_tracker import BeliefState


class Determinizer:
    """
    Samples consistent opponent hands for determinization in MCTS.

    Generates complete game states by assigning cards to opponent hands
    in a way that satisfies all constraints from belief tracking.

    The determinizer uses rejection sampling with constraint checking to
    efficiently generate valid hand assignments. It supports both uniform
    and probability-weighted sampling based on belief state distributions.

    Attributes:
        max_attempts: Maximum sampling attempts before giving up
    """

    def __init__(self, max_attempts: int = 100):
        """
        Initialize determinizer.

        Args:
            max_attempts: Maximum sampling attempts before giving up
        """
        self.max_attempts = max_attempts

    def sample_determinization(
        self,
        game: BlobGame,
        belief: BeliefState,
        use_probabilities: bool = True,
    ) -> Optional[Dict[int, List[Card]]]:
        """
        Sample a consistent assignment of unseen cards to opponent hands.

        Uses rejection sampling to generate hand assignments that satisfy
        all belief state constraints. Can use uniform or probability-weighted
        sampling based on the use_probabilities parameter.

        Args:
            game: Current game state
            belief: Belief state with constraints
            use_probabilities: If True, use probability-weighted sampling

        Returns:
            Dictionary mapping player_position → List[Card] (hand)
            Returns None if no consistent sample found
        """
        for attempt in range(self.max_attempts):
            sampled_hands = self._attempt_sample(game, belief, use_probabilities)

            if sampled_hands is not None:
                # Validate consistency
                if self._validate_sample(sampled_hands, belief):
                    return sampled_hands

        # Failed to find consistent sample
        return None

    def _attempt_sample(
        self, game: BlobGame, belief: BeliefState, use_probabilities: bool
    ) -> Optional[Dict[int, List[Card]]]:
        """
        Attempt to sample one determinization.

        Samples cards for each opponent from the unseen card pool, respecting
        constraints. Uses rejection sampling - if we can't satisfy constraints,
        returns None and caller will retry.

        Args:
            game: Current game state
            belief: Belief state with constraints
            use_probabilities: Whether to use probability-weighted sampling

        Returns:
            Sampled hands or None if sampling failed
        """
        # Create pool of unseen cards
        unseen_pool = list(belief.unseen_cards.copy())
        random.shuffle(unseen_pool)

        sampled_hands = {}

        # Sample for each opponent (in sorted order for consistency)
        for player_pos in sorted(belief.player_constraints.keys()):
            constraints = belief.player_constraints[player_pos]
            cards_needed = constraints.cards_in_hand

            # Filter cards this player can have
            available_cards = [
                card for card in unseen_pool if constraints.can_have_card(card)
            ]

            # Not enough cards available
            if len(available_cards) < cards_needed:
                return None

            # Sample cards
            if use_probabilities:
                # Probability-weighted sampling
                sampled_cards = self._sample_with_probabilities(
                    available_cards, cards_needed, belief, player_pos
                )
            else:
                # Uniform sampling
                sampled_cards = random.sample(available_cards, cards_needed)

            sampled_hands[player_pos] = sampled_cards

            # Remove sampled cards from pool
            for card in sampled_cards:
                unseen_pool.remove(card)

        return sampled_hands

    def _sample_with_probabilities(
        self,
        available_cards: List[Card],
        num_cards: int,
        belief: BeliefState,
        player_pos: int,
    ) -> List[Card]:
        """
        Sample cards using probability distribution from belief state.

        Uses the belief state's probability distribution to weight card
        selection, making more likely cards more likely to be sampled.

        Args:
            available_cards: Cards to sample from
            num_cards: Number of cards to sample
            belief: Belief state with probabilities
            player_pos: Player position to sample for

        Returns:
            List of sampled cards
        """
        # Get probabilities for each card
        probs = np.array(
            [belief.get_card_probability(player_pos, card) for card in available_cards]
        )

        # Normalize
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            # Fall back to uniform if all probabilities are 0
            probs = np.ones(len(available_cards)) / len(available_cards)

        # Sample without replacement
        sampled_indices = np.random.choice(
            len(available_cards), size=num_cards, replace=False, p=probs
        )

        return [available_cards[i] for i in sampled_indices]

    def _validate_sample(
        self, sampled_hands: Dict[int, List[Card]], belief: BeliefState
    ) -> bool:
        """
        Validate that sampled hands satisfy all constraints.

        Checks that:
        1. Each hand satisfies player-specific constraints
        2. No duplicate cards across hands
        3. Hand sizes are correct

        Args:
            sampled_hands: Sampled hands for each player
            belief: Belief state with constraints

        Returns:
            True if sample is valid
        """
        # Validate each hand individually
        for player_pos, hand in sampled_hands.items():
            if not belief.is_consistent_hand(player_pos, hand):
                return False

        # Check no duplicate cards
        all_cards = []
        for hand in sampled_hands.values():
            all_cards.extend(hand)

        if len(all_cards) != len(set(all_cards)):
            return False  # Duplicate cards

        return True

    def sample_multiple_determinizations(
        self,
        game: BlobGame,
        belief: BeliefState,
        num_samples: int = 5,
        use_probabilities: bool = True,
    ) -> List[Dict[int, List[Card]]]:
        """
        Sample multiple determinizations for MCTS.

        Generates multiple independent samples of opponent hands. These can
        be used for multi-world MCTS where searches are run on multiple
        determinized game states.

        Args:
            game: Current game state
            belief: Belief state
            num_samples: Number of determinizations to generate
            use_probabilities: Use probability-weighted sampling

        Returns:
            List of sampled hand assignments
        """
        samples = []

        for _ in range(num_samples):
            sample = self.sample_determinization(game, belief, use_probabilities)
            if sample is not None:
                samples.append(sample)

        return samples

    def create_determinized_game(
        self,
        game: BlobGame,
        belief: BeliefState,
        sampled_hands: Dict[int, List[Card]],
    ) -> BlobGame:
        """
        Create a complete game state with determinized opponent hands.

        Takes the original game (with hidden hands) and creates a new game
        state where opponent hands are filled in with the sampled cards.
        The observer's hand remains unchanged (we know it perfectly).

        Args:
            game: Original game state (with hidden hands)
            belief: Belief state
            sampled_hands: Sampled hands for opponents

        Returns:
            New game state with revealed hands
        """
        # Create a copy of the game
        det_game = game.copy()

        # Assign sampled hands to opponent players
        for player_pos, hand in sampled_hands.items():
            det_game.players[player_pos].hand = hand.copy()

        # Observer's hand stays the same
        observer_pos = belief.observer.position
        det_game.players[observer_pos].hand = list(belief.known_cards)

        return det_game
