"""
Monte Carlo Tree Search (MCTS) implementation for BlobMaster.

This module provides MCTS infrastructure for decision-making in the Blob card game:
- MCTSNode: Tree node with UCB1 selection and backpropagation
- MCTS: Search algorithm integrated with neural network (coming in Session 7)

The MCTS implementation uses:
- UCB1 formula for exploration-exploitation balance
- Neural network policy for prior probabilities
- Neural network value for leaf evaluation
- Tree reuse for efficient search across moves
"""

from ml.mcts.node import MCTSNode

__all__ = [
    "MCTSNode",
]
