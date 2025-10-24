"""
Monte Carlo Tree Search (MCTS) implementation for BlobMaster.

This module provides MCTS infrastructure for decision-making in the Blob card game:
- MCTSNode: Tree node with UCB1 selection and backpropagation
- MCTS: Search algorithm integrated with neural network

The MCTS implementation uses:
- UCB1 formula for exploration-exploitation balance
- Neural network policy for prior probabilities
- Neural network value for leaf evaluation
- Tree reuse for efficient search across moves (Session 8)

Example:
    >>> from ml.mcts import MCTS, MCTSNode
    >>> from ml.network import BlobNet, StateEncoder, ActionMasker
    >>> from ml.game.blob import BlobGame
    >>>
    >>> # Initialize components
    >>> network = BlobNet()
    >>> encoder = StateEncoder()
    >>> masker = ActionMasker()
    >>> mcts = MCTS(network, encoder, masker, num_simulations=100)
    >>>
    >>> # Run search
    >>> game = BlobGame(num_players=4)
    >>> game.setup_round(cards_to_deal=5)
    >>> action_probs = mcts.search(game, game.players[0])
    >>> best_action = max(action_probs, key=action_probs.get)
"""

from ml.mcts.node import MCTSNode
from ml.mcts.search import MCTS

__all__ = [
    "MCTSNode",
    "MCTS",
]
