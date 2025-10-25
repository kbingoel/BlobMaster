"""
Monte Carlo Tree Search (MCTS) implementation for BlobMaster.

This module provides MCTS infrastructure for decision-making in the Blob card game:
- MCTSNode: Tree node with UCB1 selection and backpropagation
- MCTS: Search algorithm integrated with neural network
- BeliefState: Belief tracking for imperfect information (Phase 3)
- Determinizer: Determinization sampling for hidden opponent hands (Phase 3)

The MCTS implementation uses:
- UCB1 formula for exploration-exploitation balance
- Neural network policy for prior probabilities
- Neural network value for leaf evaluation
- Tree reuse for efficient search across moves (Session 8)
- Belief tracking and determinization for imperfect information (Phase 3)

Example (Perfect Information):
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

Example (Imperfect Information with Determinization):
    >>> from ml.mcts import BeliefState, Determinizer
    >>> from ml.game.blob import BlobGame
    >>>
    >>> # Create game and belief state
    >>> game = BlobGame(num_players=4)
    >>> game.setup_round(cards_to_deal=5)
    >>> observer = game.players[0]
    >>> belief = BeliefState(game, observer)
    >>>
    >>> # Sample determinizations
    >>> determinizer = Determinizer()
    >>> samples = determinizer.sample_multiple_determinizations(
    ...     game, belief, num_samples=5
    ... )
    >>>
    >>> # Create determinized games for MCTS
    >>> det_games = [
    ...     determinizer.create_determinized_game(game, belief, sample)
    ...     for sample in samples
    ... ]
"""

from ml.mcts.node import MCTSNode
from ml.mcts.search import MCTS
from ml.mcts.belief_tracker import BeliefState, PlayerConstraints
from ml.mcts.determinization import Determinizer

__all__ = [
    "MCTSNode",
    "MCTS",
    "BeliefState",
    "PlayerConstraints",
    "Determinizer",
]
