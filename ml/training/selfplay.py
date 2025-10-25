"""
Self-Play Game Generation for AlphaZero-Style Training

This module implements the self-play engine that generates training data by
playing games using MCTS with the current neural network. The generated games
provide (state, policy, value) tuples for supervised learning.

Core Concept:
    - Generate games using current model + MCTS
    - Store (state, MCTS_policy, final_outcome) tuples for training
    - Use exploration noise to ensure diverse gameplay
    - Run multiple games in parallel for efficiency

Game Generation Flow:
    1. Initialize game with random setup
    2. For each decision point (bid or card play):
        - Run MCTS with current model
        - Get action probabilities (visit counts)
        - Sample action with temperature-based exploration
        - Store (state, policy, None) for training
    3. Play until game ends
    4. Back-propagate final outcome to all stored positions
    5. Return training examples

Key Design Decisions:
    - Use imperfect info MCTS (realistic hidden information)
    - Temperature schedule: high early (exploration), low late (exploitation)
    - Store full game history for later analysis
    - Support variable player counts (3-8 players)

Training Example Format:
    {
        'state': encoded_state,          # 256-dim numpy array
        'policy': action_probabilities,  # MCTS visit counts (65-dim)
        'value': final_score,            # Outcome from this player's perspective
        'player_position': int,          # Which player made this decision
        'game_id': str,                  # For tracking game history
        'move_number': int,              # Position in game
    }
"""

import torch
import numpy as np
import uuid
from typing import Dict, List, Any, Optional, Callable

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.game.blob import BlobGame, Card
from ml.mcts.search import ImperfectInfoMCTS


class SelfPlayWorker:
    """
    Worker that generates self-play games for training.

    Runs MCTS-guided game generation and collects training examples.
    Each decision point in the game produces a training example with:
    - Current state encoding
    - MCTS policy (action probabilities)
    - Final game outcome (back-propagated after game ends)
    """

    def __init__(
        self,
        network: BlobNet,
        encoder: StateEncoder,
        masker: ActionMasker,
        num_determinizations: int = 3,
        simulations_per_determinization: int = 30,
        temperature_schedule: Optional[Callable[[int], float]] = None,
        use_imperfect_info: bool = True,
    ):
        """
        Initialize self-play worker.

        Args:
            network: Neural network for MCTS
            encoder: State encoder
            masker: Action masker
            num_determinizations: Determinizations per MCTS search (default: 3)
            simulations_per_determinization: MCTS simulations per world (default: 30)
            temperature_schedule: Function mapping move_number -> temperature
                                 If None, uses default schedule
            use_imperfect_info: Use imperfect info MCTS (vs perfect info)
        """
        self.network = network
        self.encoder = encoder
        self.masker = masker
        self.num_determinizations = num_determinizations
        self.simulations_per_determinization = simulations_per_determinization
        self.use_imperfect_info = use_imperfect_info

        # Temperature schedule for exploration
        if temperature_schedule is None:
            self.temperature_schedule = self.get_default_temperature_schedule()
        else:
            self.temperature_schedule = temperature_schedule

        # Create MCTS instance
        if use_imperfect_info:
            self.mcts = ImperfectInfoMCTS(
                network=network,
                encoder=encoder,
                masker=masker,
                num_determinizations=num_determinizations,
                simulations_per_determinization=simulations_per_determinization,
            )
        else:
            # For testing or comparison, can use perfect info MCTS
            from ml.mcts.search import MCTS

            total_sims = num_determinizations * simulations_per_determinization
            self.mcts = MCTS(
                network=network,
                encoder=encoder,
                masker=masker,
                num_simulations=total_sims,
            )

    def generate_game(
        self,
        num_players: int = 4,
        cards_to_deal: int = 5,
        game_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate a single self-play game.

        Plays a complete game using MCTS for all decisions, collecting training
        examples at each decision point. Final outcomes are back-propagated to
        all examples after the game completes.

        Args:
            num_players: Number of players in the game (3-8)
            cards_to_deal: Cards to deal per player (1-13)
            game_id: Optional game identifier (generates UUID if None)

        Returns:
            List of training examples (one per decision point)
            Each example is a dict with keys: state, policy, value,
            player_position, game_id, move_number
        """
        if game_id is None:
            game_id = str(uuid.uuid4())

        # Initialize game
        game = BlobGame(num_players=num_players)

        # Storage for training examples
        examples = []
        move_number = 0

        # Play the round using MCTS for all decisions
        def get_bid(player, hand, is_dealer, total_bids, cards_dealt):
            """Callback to get bid using MCTS."""
            nonlocal move_number

            # Run MCTS to get action probabilities
            action_probs = self.mcts.search(game, player)

            # Get temperature for this move
            temperature = self.temperature_schedule(move_number)

            # Select action with temperature
            bid = self._select_action(action_probs, temperature)

            # Store training example (value will be filled in later)
            state_tensor = self.encoder.encode(game, player)
            policy_vector = self._action_probs_to_vector(action_probs, is_bidding=True)

            examples.append(
                {
                    "state": state_tensor.cpu().numpy(),
                    "policy": policy_vector,
                    "value": None,  # Will be back-propagated
                    "player_position": player.position,
                    "game_id": game_id,
                    "move_number": move_number,
                }
            )

            move_number += 1
            return bid

        def get_card(player, legal_cards, trick):
            """Callback to get card to play using MCTS."""
            nonlocal move_number

            # Run MCTS to get action probabilities
            action_probs = self.mcts.search(game, player)

            # Get temperature for this move
            temperature = self.temperature_schedule(move_number)

            # Select action with temperature
            card_idx = self._select_action(action_probs, temperature)

            # Find the card in legal_cards that matches the selected index
            card = None
            for legal_card in legal_cards:
                if self.encoder._card_to_index(legal_card) == card_idx:
                    card = legal_card
                    break

            # Fallback to first legal card if MCTS selected an illegal card
            if card is None:
                card = legal_cards[0]

            # Store training example
            state_tensor = self.encoder.encode(game, player)
            policy_vector = self._action_probs_to_vector(
                action_probs, is_bidding=False
            )

            examples.append(
                {
                    "state": state_tensor.cpu().numpy(),
                    "policy": policy_vector,
                    "value": None,  # Will be back-propagated
                    "player_position": player.position,
                    "game_id": game_id,
                    "move_number": move_number,
                }
            )

            move_number += 1
            return card

        # Play the round
        result = game.play_round(cards_to_deal, get_bid, get_card)

        # Back-propagate final outcomes to all examples
        # Extract round scores from result
        final_scores = {}
        for player_result in result["player_scores"]:
            # Find the player by name to get their position
            for player in game.players:
                if player.name == player_result["name"]:
                    final_scores[player.position] = player_result["round_score"]
                    break

        self._backpropagate_outcome(examples, final_scores)

        return examples

    def _select_action(
        self,
        action_probs: Dict[int, float],
        temperature: float,
    ) -> int:
        """
        Select action using temperature-based sampling.

        Temperature controls exploration vs exploitation:
        - temperature = 0: Greedy (always pick best action)
        - temperature = 1: Proportional to probabilities
        - temperature > 1: More exploration

        Args:
            action_probs: Action probabilities from MCTS (dict: action_idx -> prob)
            temperature: Exploration temperature (0=greedy, 1=stochastic)

        Returns:
            Selected action index
        """
        if len(action_probs) == 0:
            raise ValueError("No legal actions available")

        # Extract actions and probabilities
        actions = list(action_probs.keys())
        probs = np.array([action_probs[a] for a in actions])

        # Apply temperature
        if temperature == 0:
            # Greedy: select action with highest probability
            best_idx = np.argmax(probs)
            return actions[best_idx]
        else:
            # Apply temperature scaling
            # Higher temperature = more uniform distribution
            # Lower temperature = more peaked distribution
            probs = probs ** (1.0 / temperature)
            probs = probs / probs.sum()  # Renormalize

            # Sample action
            selected_idx = np.random.choice(len(actions), p=probs)
            return actions[selected_idx]

    def _action_probs_to_vector(
        self, action_probs: Dict[int, float], is_bidding: bool
    ) -> np.ndarray:
        """
        Convert action probabilities dict to fixed-size vector.

        Creates a 65-dimensional vector (max of 14 bids + 52 cards - 1 extra for safety).
        For bidding: indices 0-13 are bids
        For playing: indices 0-51 are card indices

        Args:
            action_probs: Dict mapping action_idx -> probability
            is_bidding: True if this is a bidding action

        Returns:
            65-dim numpy array with probabilities (zeros for illegal actions)
        """
        policy_vector = np.zeros(65, dtype=np.float32)

        for action_idx, prob in action_probs.items():
            if action_idx < 65:  # Safety check
                policy_vector[action_idx] = prob

        return policy_vector

    def _backpropagate_outcome(
        self,
        examples: List[Dict[str, Any]],
        final_scores: Dict[int, int],
    ):
        """
        Back-propagate final game outcome to all training examples.

        Updates the 'value' field of each example with the final score
        achieved by the player who made that decision.

        Value normalization:
        - Scores in Blob range from 0 (failed bid) to 23 (bid 13 and made it)
        - Normalize to [-1, 1] range for neural network training
        - 0 points -> -1.0, 23 points -> 1.0

        Args:
            examples: List of training examples from the game
            final_scores: Final scores for each player {position: score}
        """
        # Find max possible score for normalization
        # Max score is 10 + 13 = 23 (bid 13 and make it)
        max_score = 23.0

        for example in examples:
            player_position = example["player_position"]
            score = final_scores.get(player_position, 0)

            # Normalize score to [-1, 1] range
            # 0 -> -1.0, 23 -> 1.0
            normalized_value = (score / max_score) * 2.0 - 1.0

            example["value"] = float(normalized_value)

    def get_default_temperature_schedule(self) -> Callable[[int], float]:
        """
        Get default temperature schedule.

        Temperature controls exploration during action selection:
        - Early game (moves 0-10): temperature = 1.0 (high exploration)
        - Mid game (moves 11-20): temperature = 0.5 (moderate)
        - Late game (moves 21+): temperature = 0.1 (near-greedy)

        This encourages diverse play early (to explore different strategies)
        and more deterministic play later (to demonstrate learned skill).

        Returns:
            Function mapping move_number -> temperature
        """

        def schedule(move_number: int) -> float:
            if move_number < 10:
                return 1.0  # High exploration
            elif move_number < 20:
                return 0.5  # Moderate exploration
            else:
                return 0.1  # Near-greedy

        return schedule
