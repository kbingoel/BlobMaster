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


# Export main class
__all__ = ["MCTS"]
