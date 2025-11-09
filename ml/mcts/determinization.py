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
import time
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

    def __init__(self, max_attempts: int = 100, use_caching: bool = True, must_have_bias: float = 1.0):
        """
        Initialize determinizer.

        Args:
            max_attempts: Maximum sampling attempts before giving up
            use_caching: Whether to cache recent samples for diversity
            must_have_bias: Probability multiplier for must-have suits during sampling
                           1.0 = no bias (maximum entropy), higher values = stronger preference
        """
        self.max_attempts = max_attempts
        self.use_caching = use_caching
        self.must_have_bias = must_have_bias
        self.sample_cache: List[Dict[int, List[Card]]] = []
        self.cache_size = 20
        # Instrumentation toggle (module-level control functions below)
        self._profiling_enabled = _DET_PROFILING_ENABLED

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
        start_t = time.perf_counter() if self._profiling_enabled else 0.0
        if self._profiling_enabled:
            _DET_METRICS['sample_determinization_calls'] += 1

        for attempt in range(self.max_attempts):
            if self._profiling_enabled:
                _DET_METRICS['attempts_total'] += 1
            sampled_hands = self._attempt_sample(game, belief, use_probabilities)

            if sampled_hands is not None:
                # Validate consistency
                valid_start = time.perf_counter() if self._profiling_enabled else 0.0
                is_valid = self._validate_sample(sampled_hands, belief)
                if self._profiling_enabled:
                    _DET_METRICS['validate_calls'] += 1
                    _DET_METRICS['validate_total_sec'] += time.perf_counter() - valid_start
                if is_valid:
                    if self._profiling_enabled:
                        _DET_METRICS['samples_succeeded'] += 1
                        _DET_METRICS['sample_determinization_total_sec'] += (
                            time.perf_counter() - start_t
                        )
                    return sampled_hands

        # Failed to find consistent sample
        if self._profiling_enabled:
            _DET_METRICS['sample_determinization_total_sec'] += (
                time.perf_counter() - start_t
            )
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
        constraints = belief.player_constraints[player_pos]

        # Get probabilities for each card
        probs = np.array(
            [belief.get_card_probability(player_pos, card) for card in available_cards]
        )

        # Apply soft prior bias for must-have suits
        for i, card in enumerate(available_cards):
            if card.suit in constraints.must_have_suits:
                probs[i] *= self.must_have_bias  # Boost probability for must-have suits

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

    def _propagate_constraints(
        self, belief: BeliefState, sampled_hands: Dict[int, List[Card]]
    ) -> bool:
        """
        Propagate constraints forward to check for conflicts.

        Detects early if a partial sample will lead to inconsistency by
        checking if remaining players can still be satisfied with the
        available card pool.

        Args:
            belief: Belief state
            sampled_hands: Partially sampled hands

        Returns:
            True if constraints are satisfiable, False if conflict detected
        """
        # Count cards remaining per suit
        unseen_pool = set(belief.unseen_cards)
        for hand in sampled_hands.values():
            for card in hand:
                unseen_pool.discard(card)

        # Check remaining players can be satisfied
        remaining_players = [
            pos for pos in belief.player_constraints if pos not in sampled_hands
        ]

        for player_pos in remaining_players:
            constraints = belief.player_constraints[player_pos]
            cards_needed = constraints.cards_in_hand

            # Count available cards for this player
            available = [
                card for card in unseen_pool if constraints.can_have_card(card)
            ]

            if len(available) < cards_needed:
                return False  # Not enough cards available

        return True

    def sample_determinization_with_diversity(
        self,
        game: BlobGame,
        belief: BeliefState,
        avoid_samples: Optional[List[Dict[int, List[Card]]]] = None,
    ) -> Optional[Dict[int, List[Card]]]:
        """
        Sample a determinization that's diverse from previous samples.

        Attempts to generate samples that are sufficiently different from
        existing samples to provide better coverage of the belief space.

        Args:
            game: Current game state
            belief: Belief state
            avoid_samples: List of samples to avoid (for diversity)

        Returns:
            Sampled hands that are different from avoid_samples
        """
        for attempt in range(self.max_attempts):
            sample = self.sample_determinization(game, belief, use_probabilities=True)

            if sample is None:
                continue

            # Check diversity
            if avoid_samples is None or self._is_diverse(sample, avoid_samples):
                return sample

        # Fall back to any valid sample
        return self.sample_determinization(game, belief, use_probabilities=True)

    def _is_diverse(
        self,
        sample: Dict[int, List[Card]],
        existing_samples: List[Dict[int, List[Card]]],
        threshold: float = 0.3,
    ) -> bool:
        """
        Check if sample is sufficiently different from existing samples.

        Uses Jaccard similarity to measure how similar two hand assignments are.
        A sample is considered diverse if it differs from all existing samples
        by at least the threshold percentage.

        Args:
            sample: New sample to check
            existing_samples: Existing samples
            threshold: Minimum difference ratio (0.3 = 30% different cards)

        Returns:
            True if sample is diverse enough
        """
        for existing in existing_samples:
            similarity = self._compute_similarity(sample, existing)
            if similarity > (1.0 - threshold):
                return False  # Too similar

        return True

    def _compute_similarity(
        self, sample1: Dict[int, List[Card]], sample2: Dict[int, List[Card]]
    ) -> float:
        """
        Compute Jaccard similarity between two samples.

        Jaccard similarity is |intersection| / |union| of the card sets.
        Returns 1.0 if samples are identical, 0.0 if completely different.

        Args:
            sample1: First hand assignment
            sample2: Second hand assignment

        Returns:
            Similarity in [0, 1] where 1 = identical
        """
        all_cards1 = set()
        all_cards2 = set()

        for hand in sample1.values():
            all_cards1.update(hand)

        for hand in sample2.values():
            all_cards2.update(hand)

        intersection = len(all_cards1 & all_cards2)
        union = len(all_cards1 | all_cards2)

        return intersection / union if union > 0 else 0.0

    def sample_adaptive(
        self,
        game: BlobGame,
        belief: BeliefState,
        num_samples: int,
        diversity_weight: float = 0.5,
    ) -> List[Dict[int, List[Card]]]:
        """
        Sample determinizations with adaptive diversity control.

        Balances between probability-weighted sampling (which respects
        belief distributions) and diversity (which ensures good coverage
        of the belief space).

        Args:
            game: Current game state
            belief: Belief state
            num_samples: Number of samples to generate
            diversity_weight: Weight for diversity vs probability (0-1)
                            0 = pure probability sampling
                            1 = pure diversity sampling

        Returns:
            List of diverse determinizations
        """
        samples = []
        start_t = time.perf_counter() if self._profiling_enabled else 0.0
        if self._profiling_enabled:
            _DET_METRICS['sample_adaptive_calls'] += 1

        for i in range(num_samples):
            if i == 0 or random.random() > diversity_weight:
                # First sample or probability-weighted sampling
                sample = self.sample_determinization(game, belief, use_probabilities=True)
            else:
                # Diversity-focused sampling
                sample = self.sample_determinization_with_diversity(
                    game, belief, avoid_samples=samples
                )

            if sample is not None:
                samples.append(sample)

        if self._profiling_enabled:
            _DET_METRICS['sample_adaptive_total_sec'] += time.perf_counter() - start_t
        return samples


# -------------------
# Lightweight metrics
# -------------------

# Module-level flag and store for determinization profiling metrics
_DET_PROFILING_ENABLED = False
_DET_METRICS = {
    'sample_determinization_calls': 0,
    'sample_determinization_total_sec': 0.0,
    'attempts_total': 0,
    'samples_succeeded': 0,
    'validate_calls': 0,
    'validate_total_sec': 0.0,
    'sample_adaptive_calls': 0,
    'sample_adaptive_total_sec': 0.0,
}


def enable_metrics(enabled: bool = True) -> None:
    """Enable or disable determinization instrumentation for this process."""
    global _DET_PROFILING_ENABLED
    _DET_PROFILING_ENABLED = bool(enabled)


def reset_metrics() -> None:
    """Reset determinization metrics counters for this process."""
    for k in list(_DET_METRICS.keys()):
        _DET_METRICS[k] = 0.0 if k.endswith('_sec') else 0


def get_metrics() -> dict:
    """Return a shallow copy of current determinization metrics."""
    # Derive helper KPI without recomputing on the hot path
    m = dict(_DET_METRICS)
    attempts = m.get('attempts_total', 0) or 0
    successes = m.get('samples_succeeded', 0) or 0
    calls = m.get('sample_determinization_calls', 0) or 0
    m['avg_attempts_per_call'] = (attempts / calls) if calls else 0.0
    m['avg_attempts_per_success'] = (attempts / successes) if successes else 0.0
    m['avg_validate_ms'] = (
        (m.get('validate_total_sec', 0.0) / m.get('validate_calls', 1)) * 1000.0
        if m.get('validate_calls', 0) > 0 else 0.0
    )
    m['avg_sample_determinization_ms'] = (
        (m.get('sample_determinization_total_sec', 0.0) / (calls or 1)) * 1000.0
    )
    return m
