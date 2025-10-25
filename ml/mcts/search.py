"""
Monte Carlo Tree Search (MCTS) implementation for Blob card game.

This module implements the AlphaZero-style MCTS algorithm that integrates
with a neural network for leaf evaluation and action selection. The MCTS
search performs lookahead planning by simulating games and building a
search tree guided by the neural network's policy and value predictions.

Main Components:
    - MCTS: Main search class that orchestrates simulations
    - Four-phase MCTS loop: Selection, Expansion, Evaluation, Backpropagation
    - Terminal state handling for completed games
    - Legal action masking for both bidding and playing phases

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
    >>> # Create game and get action probabilities
    >>> game = BlobGame(num_players=4)
    >>> game.setup_round(cards_to_deal=5)
    >>> player = game.players[0]
    >>>
    >>> # Run MCTS search
    >>> action_probs = mcts.search(game, player)
    >>> # Returns: {0: 0.1, 1: 0.3, 2: 0.6} - probabilities for each legal action
    >>>
    >>> # Select best action
    >>> best_action = max(action_probs, key=action_probs.get)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

from ml.mcts.node import MCTSNode
from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.game.blob import BlobGame, Player, Card
from ml.game.constants import SUITS, RANKS


class MCTS:
    """
    Monte Carlo Tree Search implementation for Blob card game.

    Integrates with neural network for leaf evaluation and uses UCB1-based
    tree traversal to build a search tree. After running simulations, returns
    action probabilities based on visit counts.

    The search performs N simulations, each consisting of:
        1. Selection: Traverse tree using UCB1 until reaching a leaf
        2. Expansion: Create children for legal actions with network policy priors
        3. Evaluation: Use neural network to evaluate leaf (or terminal value)
        4. Backpropagation: Update visit counts and values back to root

    Attributes:
        network: Neural network for policy and value prediction
        encoder: StateEncoder for converting game state to tensor
        masker: ActionMasker for creating legal action masks
        num_simulations: Number of MCTS simulations per search
        c_puct: Exploration constant for UCB1 (typically 1.5)
        temperature: Temperature for action selection (1.0 = proportional to visits)

    Example:
        >>> mcts = MCTS(network, encoder, masker, num_simulations=100)
        >>> action_probs = mcts.search(game_state, current_player)
        >>> best_action = max(action_probs, key=action_probs.get)
    """

    def __init__(
        self,
        network: BlobNet,
        encoder: StateEncoder,
        masker: ActionMasker,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        temperature: float = 1.0,
    ):
        """
        Initialize MCTS search.

        Args:
            network: Neural network for evaluation (BlobNet instance)
            encoder: State encoder for converting game state to tensor
            masker: Action masker for creating legal action masks
            num_simulations: Number of MCTS simulations per move (default: 100)
                            Higher = better quality but slower
            c_puct: Exploration constant for UCB1 (default: 1.5)
                   Higher = more exploration, lower = more exploitation
            temperature: Temperature for action selection (default: 1.0)
                        0.0 = greedy (best action only)
                        1.0 = proportional to visit counts
                        >1.0 = more uniform (more exploration)
        """
        self.network = network
        self.encoder = encoder
        self.masker = masker
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature

        # Tree reuse: Store root node for next search
        self.root: Optional[MCTSNode] = None

        # Set network to evaluation mode (disable dropout, etc.)
        self.network.eval()

    def search(
        self,
        game_state: BlobGame,
        player: Player,
    ) -> Dict[int, float]:
        """
        Run MCTS search from current game state.

        Performs num_simulations iterations of the MCTS algorithm, building
        a search tree rooted at the current game state. Returns action
        probabilities derived from visit counts.

        Args:
            game_state: Current game state to search from
            player: Player whose turn it is (perspective for search)

        Returns:
            Dictionary mapping action index → probability
            - Bidding phase: {0: 0.1, 1: 0.3, 2: 0.6} (bid values)
            - Playing phase: {13: 0.2, 26: 0.8} (card indices)

        Example:
            >>> game = BlobGame(num_players=4)
            >>> game.setup_round(5)
            >>> player = game.players[0]
            >>> action_probs = mcts.search(game, player)
            >>> print(action_probs)
            {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.35, 4: 0.10, 5: 0.05}
        """
        # Create root node for this search
        root = MCTSNode(
            game_state=game_state,
            player=player,
            parent=None,
            action_taken=None,
            prior_prob=0.0,
        )

        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root)

        # Get action probabilities from visit counts
        action_probs = root.get_action_probabilities(self.temperature)

        return action_probs

    def _simulate(self, node: MCTSNode) -> float:
        """
        Run one MCTS simulation (all 4 phases).

        Performs one iteration of the MCTS algorithm:
            1. Selection: Traverse tree using UCB1 until leaf
            2. Expansion: Add children for legal actions
            3. Evaluation: Get value from network or terminal state
            4. Backpropagation: Update statistics back to root

        Args:
            node: Node to start simulation from (typically root)

        Returns:
            Value of the leaf node reached (for debugging)
        """
        # PHASE 1: SELECTION
        # Traverse tree using UCB1 until we reach a leaf node
        current = node
        while not current.is_leaf():
            current = current.select_child(self.c_puct)

        # PHASE 2 & 3: EXPANSION & EVALUATION
        # Check if we've reached a terminal state
        if self._is_terminal(current.game_state):
            # Use actual game outcome instead of network evaluation
            value = self._get_terminal_value(current.game_state, current.player)
        else:
            # Expand node and evaluate with neural network
            value = self._expand_and_evaluate(current)

        # PHASE 4: BACKPROPAGATION
        # Update visit counts and values back to root
        current.backpropagate(value)

        return value

    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """
        Expand leaf node and evaluate with neural network.

        Creates children for all legal actions, using the neural network's
        policy output as prior probabilities. Returns the neural network's
        value prediction for this state.

        Args:
            node: Leaf node to expand

        Returns:
            Value prediction from neural network (normalized to [-1, 1])
        """
        # Encode current state to tensor
        state_tensor = self.encoder.encode(node.game_state, node.player)

        # Get legal actions and mask for current game phase
        legal_actions, legal_mask = self._get_legal_actions_and_mask(
            node.game_state, node.player
        )

        # Neural network evaluation
        with torch.no_grad():
            policy, value = self.network(state_tensor, legal_mask)

        # Convert policy tensor to dictionary {action: probability}
        policy_np = policy.cpu().numpy()
        action_probs = {action: float(policy_np[action]) for action in legal_actions}

        # Expand node with action priors
        node.expand(action_probs, legal_actions)

        # Return value prediction
        return value.item()

    def _get_legal_actions_and_mask(
        self,
        game: BlobGame,
        player: Player,
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Get legal actions and mask for current game state.

        Determines legal actions based on game phase and creates a mask
        for the neural network to prevent illegal move predictions.

        Args:
            game: Current game state
            player: Player whose turn it is

        Returns:
            Tuple of (legal_actions, legal_mask):
                - legal_actions: List of legal action indices
                - legal_mask: Tensor of shape (action_dim,) with 1=legal, 0=illegal

        Raises:
            ValueError: If game phase is not 'bidding' or 'playing'
        """
        if game.game_phase == "bidding":
            # BIDDING PHASE: Legal actions are valid bids [0, cards_dealt]
            cards_dealt = len(player.hand)
            is_dealer = player.position == game.dealer_position
            forbidden_bid = None

            # Calculate dealer's forbidden bid
            if is_dealer:
                total_bids = sum(p.bid for p in game.players if p.bid is not None)
                forbidden_bid = cards_dealt - total_bids

            # Create bidding mask
            mask = self.masker.create_bidding_mask(
                cards_dealt=cards_dealt,
                is_dealer=is_dealer,
                forbidden_bid=forbidden_bid,
            )

            # Get legal action indices (0 to cards_dealt, excluding forbidden)
            legal_actions = []
            for bid in range(cards_dealt + 1):
                if mask[bid] == 1.0:
                    legal_actions.append(bid)

        elif game.game_phase == "playing":
            # PLAYING PHASE: Legal actions are cards in hand (following suit rules)
            led_suit = game.current_trick.led_suit if game.current_trick else None

            # Create playing mask
            mask = self.masker.create_playing_mask(
                hand=player.hand,
                led_suit=led_suit,
                encoder=self.encoder,
            )

            # Get legal card indices
            legal_actions = []
            for card in player.hand:
                card_idx = self.encoder._card_to_index(card)
                if mask[card_idx] == 1.0:
                    legal_actions.append(card_idx)

        else:
            raise ValueError(
                f"Cannot get legal actions for game phase: {game.game_phase}"
            )

        return legal_actions, mask

    def _is_terminal(self, game: BlobGame) -> bool:
        """
        Check if game state is terminal (round completed).

        Args:
            game: Game state to check

        Returns:
            True if round is complete, False otherwise
        """
        return game.game_phase in ["complete", "scoring"]

    def _get_terminal_value(self, game: BlobGame, player: Player) -> float:
        """
        Get value of terminal game state.

        Calculates the player's score for the completed round and normalizes
        it to [-1, 1] range for neural network training.

        Args:
            game: Terminal game state
            player: Player to calculate score for

        Returns:
            Normalized score in [-1, 1]:
                - 1.0: Maximum possible score (perfect round)
                - 0.0: Zero score (failed to make bid)
                - -1.0: Minimum score (not used, 0 is min)

        Note:
            Score formula: (tricks_won == bid) ? (10 + bid) : 0
            Max score: 10 + 13 = 23 (make bid of 13 tricks)
        """
        # Calculate round score
        score = player.calculate_round_score()

        # Normalize to [-1, 1]
        # Max score: 10 + max_cards (typically 10 + 13 = 23)
        max_score = 23.0

        # Normalize score
        normalized_score = score / max_score

        return normalized_score

    def search_with_tree_reuse(
        self,
        game_state: BlobGame,
        player: Player,
        previous_action: Optional[int] = None,
    ) -> Dict[int, float]:
        """
        Run MCTS with tree reuse.

        If previous_action is provided, navigate to that child node
        and make it the new root (keeping its subtree). This allows
        reusing explored nodes from previous searches, significantly
        improving performance.

        Args:
            game_state: Current game state
            player: Current player
            previous_action: Action taken to reach this state (from previous search)

        Returns:
            Action probabilities from visit counts

        Example:
            >>> # First move: build tree from scratch
            >>> action_probs = mcts.search_with_tree_reuse(game, player)
            >>> best_action = max(action_probs, key=action_probs.get)
            >>>
            >>> # Apply action to game
            >>> game.apply_action(best_action, player)
            >>>
            >>> # Second move: reuse subtree
            >>> action_probs = mcts.search_with_tree_reuse(
            ...     game, next_player, previous_action=best_action
            ... )
        """
        # Tree reuse: Navigate to child node from previous search
        if self.root is not None and previous_action is not None:
            if previous_action in self.root.children:
                # Reuse subtree: make child the new root
                self.root = self.root.children[previous_action]
                self.root.parent = None  # Detach from old parent
            else:
                # Action not in tree (unexpected), create new root
                self.root = None

        # Create new root if needed (first search or reuse failed)
        if self.root is None:
            self.root = MCTSNode(
                game_state=game_state,
                player=player,
                parent=None,
                action_taken=None,
                prior_prob=0.0,
            )

        # Run simulations from root
        for _ in range(self.num_simulations):
            self._simulate(self.root)

        # Get action probabilities
        action_probs = self.root.get_action_probabilities(self.temperature)

        return action_probs

    def reset_tree(self) -> None:
        """
        Clear tree (for new game or round).

        Call this when starting a new game or when tree reuse is not possible
        (e.g., after opponent's move that we didn't predict).

        Example:
            >>> # Start new game
            >>> mcts.reset_tree()
            >>> action_probs = mcts.search_with_tree_reuse(game, player)
        """
        self.root = None

    def search_batched(
        self,
        game_state: BlobGame,
        player: Player,
        batch_size: int = 8,
    ) -> Dict[int, float]:
        """
        Run MCTS with batched neural network inference.

        Accumulates leaf nodes and evaluates them in batches, improving
        GPU utilization and overall performance. Particularly effective
        when running on GPU with larger batch sizes.

        Args:
            game_state: Current game state
            player: Current player
            batch_size: Number of leaf nodes to evaluate per batch (default: 8)
                       Larger = better GPU utilization but more memory
                       Recommended: 8-16 for training, 4-8 for inference

        Returns:
            Action probabilities from visit counts

        Example:
            >>> # Use batched inference for better GPU utilization
            >>> action_probs = mcts.search_batched(game, player, batch_size=16)
        """
        # Create root node
        root = MCTSNode(
            game_state=game_state,
            player=player,
            parent=None,
            action_taken=None,
            prior_prob=0.0,
        )

        # Calculate number of batches needed
        num_batches = (self.num_simulations + batch_size - 1) // batch_size

        for _ in range(num_batches):
            # Collect leaf nodes for batch evaluation
            leaf_nodes = []
            current_batch_size = min(batch_size, self.num_simulations - len(leaf_nodes))

            for _ in range(current_batch_size):
                # Traverse to leaf
                leaf = self._traverse_to_leaf(root)
                if leaf is not None:
                    leaf_nodes.append(leaf)

            # Batch evaluate all collected leaves
            if leaf_nodes:
                self._batch_expand_and_evaluate(leaf_nodes)

        # Get action probabilities from visit counts
        return root.get_action_probabilities(self.temperature)

    def _traverse_to_leaf(self, root: MCTSNode) -> Optional[MCTSNode]:
        """
        Traverse from root to leaf using UCB1.

        Helper method for batched inference. Traverses tree until
        reaching a leaf node (unexpanded node).

        Args:
            root: Node to start traversal from

        Returns:
            Leaf node reached, or None if root is terminal
        """
        node = root

        # Selection: traverse until leaf
        while not node.is_leaf():
            node = node.select_child(self.c_puct)

        # Check if terminal (don't return terminal nodes for batch evaluation)
        if self._is_terminal(node.game_state):
            # Handle terminal immediately
            value = self._get_terminal_value(node.game_state, node.player)
            node.backpropagate(value)
            return None

        return node

    def _batch_expand_and_evaluate(self, nodes: List[MCTSNode]) -> None:
        """
        Expand and evaluate multiple nodes in batch.

        Performs batch neural network inference on multiple leaf nodes,
        then expands each node with action priors and backpropagates values.

        Args:
            nodes: List of leaf nodes to evaluate

        Example:
            >>> leaves = [node1, node2, node3]
            >>> mcts._batch_expand_and_evaluate(leaves)
            >>> # All leaves now expanded with children
        """
        if not nodes:
            return

        # Encode all states
        states = []
        masks = []
        legal_actions_list = []

        for node in nodes:
            state = self.encoder.encode(node.game_state, node.player)
            legal_actions, mask = self._get_legal_actions_and_mask(
                node.game_state, node.player
            )

            states.append(state)
            masks.append(mask)
            legal_actions_list.append(legal_actions)

        # Stack into batch tensors
        state_batch = torch.stack(states)
        mask_batch = torch.stack(masks)

        # Batch inference
        with torch.no_grad():
            policy_batch, value_batch = self.network(state_batch, mask_batch)

        # Expand and backpropagate each node
        for i, node in enumerate(nodes):
            policy = policy_batch[i].cpu().numpy()
            value = value_batch[i].item()

            # Create action probs dictionary
            action_probs = {
                action: float(policy[action])
                for action in legal_actions_list[i]
            }

            # Expand node
            node.expand(action_probs, legal_actions_list[i])

            # Backpropagate value
            node.backpropagate(value)


class ImperfectInfoMCTS:
    """
    Monte Carlo Tree Search for imperfect information games.

    Uses determinization to handle hidden opponent hands by sampling multiple
    possible worlds consistent with observations, running MCTS on each, and
    aggregating results.

    Multi-World MCTS Algorithm:
        1. Generate N determinizations (sampled opponent hands)
        2. For each determinization:
            - Run MCTS with K simulations
            - Get action visit counts
        3. Aggregate visit counts across all determinizations
        4. Select action based on aggregated counts

    Key Insight:
        - Running MCTS on multiple possible worlds averages out uncertainty
        - Actions that are good across many scenarios are more robust
        - Equivalent to doing expectation over belief distribution

    Attributes:
        network: Neural network for evaluation
        encoder: StateEncoder for converting game state to tensor
        masker: ActionMasker for creating legal action masks
        num_determinizations: Number of worlds to sample (default: 5)
        simulations_per_determinization: MCTS simulations per world (default: 50)
        c_puct: Exploration constant for UCB1
        temperature: Temperature for action selection

    Example:
        >>> imperfect_mcts = ImperfectInfoMCTS(
        ...     network, encoder, masker,
        ...     num_determinizations=5,
        ...     simulations_per_determinization=50
        ... )
        >>> action_probs = imperfect_mcts.search(game_state, player)
    """

    def __init__(
        self,
        network: BlobNet,
        encoder: StateEncoder,
        masker: ActionMasker,
        num_determinizations: int = 5,
        simulations_per_determinization: int = 50,
        c_puct: float = 1.5,
        temperature: float = 1.0,
    ):
        """
        Initialize imperfect information MCTS.

        Args:
            network: Neural network for evaluation
            encoder: State encoder
            masker: Action masker
            num_determinizations: Number of worlds to sample (default: 5)
                                 More = better coverage but slower
            simulations_per_determinization: MCTS simulations per world (default: 50)
                                            More = better per-world evaluation
            c_puct: Exploration constant (default: 1.5)
            temperature: Temperature for action selection (default: 1.0)

        Note:
            Total budget = num_determinizations × simulations_per_determinization
            Example: 5 determinizations × 50 simulations = 250 total simulations
        """
        self.network = network
        self.encoder = encoder
        self.masker = masker
        self.num_determinizations = num_determinizations
        self.simulations_per_determinization = simulations_per_determinization
        self.c_puct = c_puct
        self.temperature = temperature

        # Import determinization components
        from ml.mcts.belief_tracker import BeliefState
        from ml.mcts.determinization import Determinizer

        # Create determinizer
        self.determinizer = Determinizer()

        # Perfect info MCTS for each determinization
        self.perfect_info_mcts = MCTS(
            network=network,
            encoder=encoder,
            masker=masker,
            num_simulations=simulations_per_determinization,
            c_puct=c_puct,
            temperature=temperature,
        )

    def search(
        self,
        game_state: BlobGame,
        player: Player,
        belief: Optional["BeliefState"] = None,
    ) -> Dict[int, float]:
        """
        Run imperfect information MCTS search.

        Samples multiple determinizations of hidden opponent hands, runs
        MCTS on each determinized world, and aggregates results to get
        robust action probabilities.

        Args:
            game_state: Current game state (with hidden hands)
            player: Player whose turn it is
            belief: Belief state (will be created if None)

        Returns:
            Dictionary mapping action → probability
            Example: {0: 0.1, 1: 0.3, 2: 0.6}

        Example:
            >>> game = BlobGame(num_players=4)
            >>> game.setup_round(5)
            >>> player = game.players[0]
            >>> action_probs = imperfect_mcts.search(game, player)
        """
        from ml.mcts.belief_tracker import BeliefState

        # Create belief state if not provided
        if belief is None:
            belief = BeliefState(game_state, player)

        # Sample determinizations
        determinizations = self.determinizer.sample_adaptive(
            game_state,
            belief,
            num_samples=self.num_determinizations,
            diversity_weight=0.5,
        )

        if not determinizations:
            # Fall back to single MCTS on original state if sampling fails
            return self.perfect_info_mcts.search(game_state, player)

        # Aggregate action counts across determinizations
        aggregated_counts: Dict[int, int] = {}

        for det_hands in determinizations:
            # Create determinized game
            det_game = self.determinizer.create_determinized_game(
                game_state, belief, det_hands
            )

            # Run MCTS on this determinization
            action_probs = self.perfect_info_mcts.search(det_game, player)

            # Accumulate visit counts (approximate from probabilities)
            for action, prob in action_probs.items():
                # Convert prob back to visit count estimate
                visit_count = int(prob * self.simulations_per_determinization)
                aggregated_counts[action] = aggregated_counts.get(action, 0) + visit_count

        # Convert aggregated counts to probabilities
        total_counts = sum(aggregated_counts.values())

        if total_counts == 0:
            # No valid actions found, fall back
            return self.perfect_info_mcts.search(game_state, player)

        action_probs = {
            action: count / total_counts
            for action, count in aggregated_counts.items()
        }

        # Apply temperature
        if self.temperature != 1.0:
            action_probs = self._apply_temperature(action_probs, self.temperature)

        return action_probs

    def _apply_temperature(
        self,
        action_probs: Dict[int, float],
        temperature: float
    ) -> Dict[int, float]:
        """
        Apply temperature scaling to action probabilities.

        Temperature controls exploration vs exploitation:
        - temperature = 0: Greedy (select best action with prob 1.0)
        - temperature = 1: No change (proportional to visit counts)
        - temperature > 1: More uniform (more exploration)
        - temperature < 1: More peaked (more exploitation)

        Args:
            action_probs: Action probabilities
            temperature: Temperature (1.0 = no change, <1 = more greedy, >1 = more random)

        Returns:
            Temperature-scaled probabilities
        """
        if temperature == 0:
            # Greedy: select max
            best_action = max(action_probs, key=action_probs.get)
            return {best_action: 1.0}

        # Convert to logits, apply temperature, convert back
        actions = list(action_probs.keys())
        probs = np.array([action_probs[a] for a in actions])

        logits = np.log(probs + 1e-10)
        logits = logits / temperature

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        new_probs = exp_logits / exp_logits.sum()

        return {action: prob for action, prob in zip(actions, new_probs)}

    def search_with_action_details(
        self,
        game_state: BlobGame,
        player: Player,
        belief: Optional["BeliefState"] = None,
    ) -> Tuple[Dict[int, float], Dict[str, any]]:
        """
        Run search and return detailed information.

        Useful for debugging, analysis, and explainability. Returns action
        probabilities along with metadata about the search process.

        Args:
            game_state: Current game state
            player: Player whose turn it is
            belief: Belief state (will be created if None)

        Returns:
            Tuple of (action_probs, details_dict) where details contains:
            - num_determinizations: Number of worlds sampled
            - action_entropy: Entropy of action distribution (uncertainty measure)
            - belief_entropy: Uncertainty in beliefs about opponent hands
            - num_actions: Number of legal actions available

        Example:
            >>> action_probs, details = imperfect_mcts.search_with_action_details(
            ...     game, player
            ... )
            >>> print(f"Action entropy: {details['action_entropy']:.2f}")
            >>> print(f"Belief entropy: {details['belief_entropy']:.2f}")
        """
        from ml.mcts.belief_tracker import BeliefState

        if belief is None:
            belief = BeliefState(game_state, player)

        action_probs = self.search(game_state, player, belief)

        # Compute agreement metric
        # (How consistent are action preferences across determinizations?)
        entropy = -sum(p * np.log(p + 1e-10) for p in action_probs.values())

        details = {
            'num_determinizations': self.num_determinizations,
            'action_entropy': entropy,
            'belief_entropy': belief.get_entropy(player.position),
            'num_actions': len(action_probs),
        }

        return action_probs, details


# Export main classes
__all__ = ["MCTS", "ImperfectInfoMCTS"]
