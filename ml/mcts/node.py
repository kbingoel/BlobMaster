"""
MCTS Node implementation with UCB1 selection.

This module implements the tree node structure for Monte Carlo Tree Search,
including UCB1-based child selection, tree expansion, and backpropagation.

Architecture Note (Session 7.5 - 2025-10-25):
    Turn tracking delegates to BlobGame.get_current_player() as the canonical
    source of truth. This ensures game logic stays centralized in the game
    engine rather than being duplicated in MCTS, which is critical for Phase 3
    imperfect information handling where we'll need to create multiple possible
    game world copies with consistent turn tracking.
"""

from typing import Dict, Optional, List
import numpy as np
from ml.game.blob import BlobGame, Player


class MCTSNode:
    """
    Node in the MCTS tree.

    Represents a game state and stores statistics for UCB1 selection.
    Each node maintains visit counts, values, and references to parent/children
    for efficient tree traversal and backpropagation.

    The node implements the four key MCTS operations:
    1. Selection: Choose child with highest UCB1 score
    2. Expansion: Create children for all legal actions
    3. Simulation: Handled externally via neural network evaluation
    4. Backpropagation: Update statistics up to root

    Attributes:
        game_state: Current game state at this node
        player: Player whose turn it is at this node
        parent: Parent node (None for root)
        action_taken: Action that led to this node from parent
        prior_prob: Prior probability from neural network policy
        visit_count: Number of times this node was visited
        total_value: Sum of backpropagated values
        mean_value: Average value (total_value / visit_count)
        children: Dictionary mapping action → child node
        is_expanded: Whether this node has been expanded
    """

    def __init__(
        self,
        game_state: BlobGame,
        player: Player,
        parent: Optional["MCTSNode"] = None,
        action_taken: Optional[int] = None,
        prior_prob: float = 0.0,
    ):
        """
        Initialize MCTS node.

        Args:
            game_state: Current game state
            player: Player whose turn it is
            parent: Parent node (None for root)
            action_taken: Action that led to this node
            prior_prob: Prior probability from neural network policy

        Example:
            >>> game = BlobGame(num_players=4)
            >>> game.setup_round(5)
            >>> player = game.players[0]
            >>> root = MCTSNode(game, player)
            >>> root.is_root()
            True
            >>> root.is_leaf()
            True
        """
        self.game_state = game_state
        self.player = player
        self.parent = parent
        self.action_taken = action_taken
        self.prior_prob = prior_prob

        # MCTS statistics
        self.visit_count = 0
        self.total_value = 0.0
        self.mean_value = 0.0

        # Children: action_index → MCTSNode
        self.children: Dict[int, MCTSNode] = {}

        # Has this node been expanded?
        self.is_expanded = False

    def is_leaf(self) -> bool:
        """
        Check if node is a leaf (not yet expanded).

        Returns:
            True if node has not been expanded yet

        Example:
            >>> node = MCTSNode(game, player)
            >>> node.is_leaf()
            True
            >>> node.expand({0: 0.5, 1: 0.5}, [0, 1])
            >>> node.is_leaf()
            False
        """
        return not self.is_expanded

    def is_root(self) -> bool:
        """
        Check if node is root (no parent).

        Returns:
            True if node has no parent

        Example:
            >>> root = MCTSNode(game, player)
            >>> root.is_root()
            True
            >>> child = MCTSNode(game, player, parent=root, action_taken=0)
            >>> child.is_root()
            False
        """
        return self.parent is None

    def select_child(self, c_puct: float = 1.5) -> "MCTSNode":
        """
        Select child with highest UCB1 score.

        Uses the UCB1 formula to balance exploration and exploitation:
        UCB(child) = Q(child) + c_puct * P(child) * sqrt(N_parent) / (1 + N_child)

        Args:
            c_puct: Exploration constant (higher = more exploration)

        Returns:
            Child node with highest UCB1 value

        Raises:
            ValueError: If node has no children to select from

        Example:
            >>> root.expand({0: 0.5, 1: 0.5}, [0, 1])
            >>> child = root.select_child(c_puct=1.5)
            >>> isinstance(child, MCTSNode)
            True
        """
        if not self.children:
            raise ValueError("Cannot select child: node has no children")

        best_score = -float("inf")
        best_child = None

        for action, child in self.children.items():
            ucb_score = self._ucb1_score(child, c_puct)

            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child

        if best_child is None:
            raise ValueError("No children to select from")

        return best_child

    def _ucb1_score(self, child: "MCTSNode", c_puct: float) -> float:
        """
        Compute UCB1 score for child node.

        Implements the UCB1 formula from AlphaZero:
        UCB(child) = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)

        Where:
        - Q: Average value of child (exploitation term)
        - P: Prior probability from neural network
        - N_parent: Visit count of parent
        - N_child: Visit count of child
        - c_puct: Exploration constant

        Args:
            child: Child node to compute score for
            c_puct: Exploration constant

        Returns:
            UCB1 score (higher = should be explored more)

        Example:
            >>> score = parent._ucb1_score(child, c_puct=1.5)
            >>> isinstance(score, float)
            True
        """
        # Q: Average value (exploitation)
        q_value = child.mean_value

        # U: Exploration bonus
        u_value = c_puct * child.prior_prob * (
            np.sqrt(self.visit_count) / (1 + child.visit_count)
        )

        return q_value + u_value

    def expand(
        self,
        action_probs: Dict[int, float],
        legal_actions: List[int],
    ) -> None:
        """
        Expand node by creating children for all legal actions.

        Creates child nodes by simulating each legal action from the current
        game state. Uses BlobGame.copy() and apply_action() to create new
        game states for each child.

        Args:
            action_probs: Prior probabilities from neural network (action → prob)
            legal_actions: List of legal action indices

        Raises:
            ValueError: If legal_actions is empty

        Example:
            >>> node = MCTSNode(game, player)
            >>> action_probs = {0: 0.2, 1: 0.3, 2: 0.5}
            >>> legal_actions = [0, 1, 2]
            >>> node.expand(action_probs, legal_actions)
            >>> len(node.children)
            3
            >>> node.is_expanded
            True
        """
        if not legal_actions:
            raise ValueError("Cannot expand: no legal actions provided")

        for action in legal_actions:
            if action not in self.children:
                # Create copy of game state for child
                child_state = self._simulate_action(action)

                # Get prior probability (default to uniform if not provided)
                prior = action_probs.get(action, 1.0 / len(legal_actions))

                # Determine which player's turn it is after this action
                # For now, keep same player (will be updated in MCTS search)
                child_player = self._get_next_player(child_state)

                # Create child node
                child = MCTSNode(
                    game_state=child_state,
                    player=child_player,
                    parent=self,
                    action_taken=action,
                    prior_prob=prior,
                )

                self.children[action] = child

        self.is_expanded = True

    def _simulate_action(self, action: int) -> BlobGame:
        """
        Simulate taking an action and return new game state.

        Uses the existing BlobGame.copy() and apply_action() methods
        implemented in Phase 1a.10.

        Args:
            action: Action index to simulate

        Returns:
            New game state after applying action

        Note:
            - Uses BlobGame.copy() for deep copy (prevents mutation)
            - Uses BlobGame.apply_action() for both bidding and playing
            - Action interpretation depends on game phase:
                * Bidding: action is bid value (0-13)
                * Playing: action is card index (0-51)

        Example:
            >>> new_state = node._simulate_action(action=3)
            >>> new_state is not node.game_state  # Different objects
            True
        """
        # Create deep copy of game state
        new_game = self.game_state.copy()

        # Find the corresponding player in the copied game
        # (can't use self.player directly as it belongs to original game)
        copied_player = new_game.players[self.player.position]

        # Apply action to the copy
        new_game.apply_action(action, copied_player)

        return new_game

    def _get_next_player(self, game_state: BlobGame) -> Player:
        """
        Determine which player's turn it is in the given game state.

        Delegates to BlobGame.get_current_player() as the canonical
        source of truth for turn tracking. This ensures turn logic
        is centralized in the game engine rather than duplicated
        in MCTS.

        Args:
            game_state: Game state to check

        Returns:
            Player whose turn it is

        Note:
            Falls back to self.player if no current player found
            (e.g., at end of round/trick). This ensures MCTS node
            always has a valid player reference.
        """
        # Use canonical turn tracking from BlobGame
        current_player = game_state.get_current_player()

        # Fallback: if no current player (end of round/trick), use self.player
        if current_player is None:
            return self.player

        return current_player

    def backpropagate(self, value: float) -> None:
        """
        Backpropagate value up the tree.

        Updates visit count and value statistics for this node and all
        ancestors up to the root. This is called after a simulation
        reaches a leaf node.

        Args:
            value: Value to backpropagate (from neural network or terminal state)

        Example:
            >>> child.backpropagate(0.5)
            >>> child.visit_count
            1
            >>> child.total_value
            0.5
            >>> parent.visit_count  # Parent also updated
            1
        """
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count

        # Recursively backpropagate to parent
        if self.parent is not None:
            self.parent.backpropagate(value)

    def get_action_probabilities(self, temperature: float = 1.0) -> Dict[int, float]:
        """
        Get action probabilities based on visit counts.

        Converts child visit counts into a probability distribution over
        actions. Higher visit counts indicate better actions according to MCTS.

        Args:
            temperature: Temperature for sampling
                - 0.0: Greedy (select most visited)
                - 1.0: Proportional to visit counts
                - >1.0: More uniform (more exploration)

        Returns:
            Dictionary mapping action → probability

        Example:
            >>> node.expand({0: 0.5, 1: 0.5}, [0, 1])
            >>> node.children[0].backpropagate(1.0)
            >>> node.children[1].backpropagate(0.5)
            >>> probs = node.get_action_probabilities(temperature=1.0)
            >>> probs[0] > probs[1]  # Action 0 visited more
            True
        """
        if not self.children:
            return {}

        # Get visit counts for all children
        actions = list(self.children.keys())
        visits = np.array([self.children[a].visit_count for a in actions])

        if temperature == 0:
            # Greedy: select most visited
            probs = np.zeros(len(visits))
            probs[np.argmax(visits)] = 1.0
        else:
            # Apply temperature to visit counts
            # Higher temperature → more uniform distribution
            visits_temp = visits ** (1.0 / temperature)
            total = visits_temp.sum()
            if total > 0:
                probs = visits_temp / total
            else:
                # If no visits, return uniform
                probs = np.ones(len(visits)) / len(visits)

        return {action: prob for action, prob in zip(actions, probs)}

    def select_action(self, temperature: float = 1.0) -> int:
        """
        Select action based on visit counts.

        Samples an action from the distribution based on visit counts.
        Used to select the final action after MCTS search completes.

        Args:
            temperature: Temperature for sampling (see get_action_probabilities)

        Returns:
            Action index

        Raises:
            ValueError: If node has no children

        Example:
            >>> node.expand({0: 0.5, 1: 0.5}, [0, 1])
            >>> action = node.select_action(temperature=1.0)
            >>> action in [0, 1]
            True
        """
        if not self.children:
            raise ValueError("Cannot select action: node has no children")

        action_probs = self.get_action_probabilities(temperature)

        actions = list(action_probs.keys())
        probs = list(action_probs.values())

        # Sample action from distribution
        action = np.random.choice(actions, p=probs)

        return action

    def __repr__(self) -> str:
        """String representation of node for debugging."""
        return (
            f"MCTSNode(action={self.action_taken}, "
            f"visits={self.visit_count}, "
            f"value={self.mean_value:.3f}, "
            f"prior={self.prior_prob:.3f}, "
            f"children={len(self.children)})"
        )
