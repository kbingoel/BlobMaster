"""
BlobMaster Game Engine Package.

This package contains the core game logic for the Blob card game,
including game rules, state management, and validation.
"""

# Import constants (always available)
from ml.game.constants import (
    SUITS,
    SUIT_NAMES,
    RANKS,
    RANK_VALUES,
    TRUMP_ROTATION,
    MIN_PLAYERS,
    MAX_PLAYERS,
    DECK_SIZE,
    SCORE_BASE,
    generate_round_structure,
)

# Import game classes (will be available after blob.py is created)
try:
    from ml.game.blob import Card, Deck, Player, Trick, BlobGame
    _blob_available = True
except ImportError:
    _blob_available = False

__all__ = [
    "SUITS",
    "SUIT_NAMES",
    "RANKS",
    "RANK_VALUES",
    "TRUMP_ROTATION",
    "MIN_PLAYERS",
    "MAX_PLAYERS",
    "DECK_SIZE",
    "SCORE_BASE",
    "generate_round_structure",
]

# Add blob classes to __all__ if available
if _blob_available:
    __all__.extend(["Card", "Deck", "Player", "Trick", "BlobGame"])
