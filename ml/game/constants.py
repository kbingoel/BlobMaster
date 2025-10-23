"""
Game constants for Blob card game.

This module defines all the core constants used throughout the game,
including card definitions, trump rotation, game constraints, and
scoring rules.
"""

from typing import List

# Card definitions
SUITS = ['♠', '♥', '♣', '♦']
SUIT_NAMES = {
    '♠': 'Spades',
    '♥': 'Hearts',
    '♣': 'Clubs',
    '♦': 'Diamonds'
}

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
RANK_VALUES = {rank: idx for idx, rank in enumerate(RANKS, start=2)}

# Trump rotation cycle
TRUMP_ROTATION = ['♠', '♥', '♣', '♦', None]  # None = no-trump rounds

# Game constraints
MIN_PLAYERS = 3
MAX_PLAYERS = 8
DECK_SIZE = 52

# Scoring
SCORE_BASE = 10  # Base score for making exact bid


def generate_round_structure(starting_cards: int, num_players: int) -> List[int]:
    """
    Generate round structure for a game.

    Creates a symmetric round structure that:
    1. Descends from starting_cards down to 1
    2. Has num_players rounds of 1 card each
    3. Ascends from 2 back up to starting_cards

    Args:
        starting_cards: Number of cards dealt in first round (e.g., 7)
        num_players: Number of players in game (3-8)

    Returns:
        List of cards to deal per round
        Example: starting_cards=5, num_players=3 → [5,4,3,2,1,1,1,2,3,4,5]

    Raises:
        ValueError: If starting_cards * num_players > 52 (exceeds deck size)
        ValueError: If starting_cards < 1
        ValueError: If num_players not in valid range [MIN_PLAYERS, MAX_PLAYERS]

    Examples:
        >>> generate_round_structure(5, 4)
        [5, 4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4, 5]

        >>> generate_round_structure(7, 3)
        [7, 6, 5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 5, 6, 7]
    """
    # Validate inputs
    if num_players < MIN_PLAYERS or num_players > MAX_PLAYERS:
        raise ValueError(
            f"num_players must be between {MIN_PLAYERS} and {MAX_PLAYERS}, "
            f"got {num_players}"
        )

    if starting_cards < 1:
        raise ValueError(
            f"starting_cards must be at least 1, got {starting_cards}"
        )

    # Validate we don't exceed deck size
    if starting_cards * num_players > DECK_SIZE:
        raise ValueError(
            f"Cannot deal {starting_cards} cards to {num_players} players "
            f"(requires {starting_cards * num_players} cards, deck has {DECK_SIZE})"
        )

    # Descending phase: starting_cards down to 1
    descending = list(range(starting_cards, 0, -1))

    # One-card rounds: num_players rounds with 1 card each
    one_card_rounds = [1] * num_players

    # Ascending phase: 2 back up to starting_cards
    ascending = list(range(2, starting_cards + 1))

    return descending + one_card_rounds + ascending
