"""
Arena system for model vs model evaluation.

This module implements a tournament system where two models play against each
other to determine which one performs better. Used to track model improvement
during training and decide when to promote a new model to "best model" status.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

from ml.game.blob import BlobGame
from ml.network.encode import StateEncoder, ActionMasker
from ml.network.model import BlobNet
from ml.mcts.search import ImperfectInfoMCTS


@dataclass
class GameContext:
    """Game context for full multi-round games (Session 4)."""
    cumulative_scores: List[int]
    rounds_completed: int
    total_rounds: int
    previous_cards: List[int]
    num_players: int
    start_cards: int
    phase: str  # 'descending', 'ones', 'ascending'


class Arena:
    """
    Tournament system for model vs model evaluation.

    Plays games between two models and records results. Models are evaluated
    by having them compete in multiple games, with fair position rotation to
    ensure neither model has an advantage from starting position.
    """

    def __init__(
        self,
        encoder: StateEncoder,
        masker: ActionMasker,
        num_determinizations: int = 3,
        simulations_per_determinization: int = 50,
        device: str = 'cpu',
        full_game_mode: bool = False,
    ):
        """
        Initialize arena for model tournaments.

        Args:
            encoder: State encoder for MCTS
            masker: Action masker for MCTS
            num_determinizations: Determinizations per MCTS search
            simulations_per_determinization: MCTS simulations per world
            device: Device to run models on ('cpu' or 'cuda')
            full_game_mode: Enable full multi-round game evaluation (Session 4)
        """
        self.encoder = encoder
        self.masker = masker
        self.num_determinizations = num_determinizations
        self.simulations_per_determinization = simulations_per_determinization
        self.device = device
        self.full_game_mode = full_game_mode

    def play_match(
        self,
        model1: BlobNet,
        model2: BlobNet,
        num_games: int = 400,
        num_players: int = 4,
        cards_to_deal: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Play a match between two models.

        Both models play as each player position equally to ensure fairness.
        Games are distributed evenly across all position combinations.

        Args:
            model1: First model (challenger)
            model2: Second model (champion)
            num_games: Number of games to play (sequences if full_game_mode=True, rounds otherwise)
            num_players: Players per game (ignored if full_game_mode=True)
            cards_to_deal: Cards to deal per player (ignored if full_game_mode=True)
            verbose: Print progress messages

        Returns:
            Match results:
            - model1_wins: Number of games model1 won
            - model2_wins: Number of games model2 won
            - draws: Number of draws
            - model1_avg_score: Average score for model1
            - model2_avg_score: Average score for model2
            - model1_total_score: Total score across all games
            - model2_total_score: Total score across all games
            - win_rate: model1 win rate
            - games_played: Total games played
        """
        # Route to full-game evaluation if enabled (Session 4)
        if self.full_game_mode:
            return self._evaluate_full_games(
                model1=model1,
                model2=model2,
                num_sequences=num_games,
                verbose=verbose,
            )

        # Single-round evaluation (original behavior)
        if verbose:
            print(f"Starting match: {num_games} games between models")
            print(f"  Players: {num_players}, Cards: {cards_to_deal}")

        # Set models to eval mode
        model1.eval()
        model2.eval()

        # Track results
        model1_scores = []
        model2_scores = []
        model1_wins = 0
        model2_wins = 0
        draws = 0

        # For fairness, each model should play as each player position equally
        # We'll cycle through having model1 as each position
        games_per_position = max(1, num_games // num_players)

        games_played = 0
        for position in range(num_players):
            for game_idx in range(games_per_position):
                if games_played >= num_games:
                    break

                # Create player assignments (which model controls which player)
                # Model1 plays as 'position', model2 plays as all others
                player_assignments = [1] * num_players  # 1 = model2
                player_assignments[position] = 0  # 0 = model1

                # Play the game
                scores = self._play_single_game(
                    [model1, model2],
                    player_assignments,
                    num_players,
                    cards_to_deal,
                )

                # Record scores
                model1_score = scores[position]
                # Average of other players' scores for model2
                model2_score = sum(
                    scores[p] for p in range(num_players) if p != position
                ) / (num_players - 1)

                model1_scores.append(model1_score)
                model2_scores.append(model2_score)

                # Determine winner (model1 vs best of model2 players)
                max_model2_score = max(
                    scores[p] for p in range(num_players) if p != position
                )
                if model1_score > max_model2_score:
                    model1_wins += 1
                elif model1_score < max_model2_score:
                    model2_wins += 1
                else:
                    draws += 1

                games_played += 1

                if verbose and games_played % 50 == 0:
                    current_win_rate = model1_wins / games_played
                    print(
                        f"  Progress: {games_played}/{num_games} games, "
                        f"Model1 win rate: {current_win_rate:.1%}"
                    )

        # Calculate final statistics
        model1_avg_score = np.mean(model1_scores)
        model2_avg_score = np.mean(model2_scores)
        model1_total_score = sum(model1_scores)
        model2_total_score = sum(model2_scores)
        win_rate = model1_wins / games_played if games_played > 0 else 0.0

        results = {
            'model1_wins': model1_wins,
            'model2_wins': model2_wins,
            'draws': draws,
            'model1_avg_score': float(model1_avg_score),
            'model2_avg_score': float(model2_avg_score),
            'model1_total_score': int(model1_total_score),
            'model2_total_score': int(model2_total_score),
            'win_rate': float(win_rate),
            'games_played': games_played,
        }

        if verbose:
            print(f"\nMatch complete!")
            print(f"  Model1 wins: {model1_wins}")
            print(f"  Model2 wins: {model2_wins}")
            print(f"  Draws: {draws}")
            print(f"  Model1 win rate: {win_rate:.1%}")
            print(f"  Model1 avg score: {model1_avg_score:.1f}")
            print(f"  Model2 avg score: {model2_avg_score:.1f}")

        return results

    def _play_single_game(
        self,
        models: List[BlobNet],
        player_assignments: List[int],
        num_players: int,
        cards_to_deal: int,
    ) -> Dict[int, int]:
        """
        Play a single game with specified model assignments.

        Args:
            models: List of models (index 0 = model1, index 1 = model2)
            player_assignments: Which model controls each player (list of 0s and 1s)
            num_players: Number of players
            cards_to_deal: Cards to deal per player

        Returns:
            Dictionary mapping player_position -> final_score
        """
        # Initialize game
        game = BlobGame(num_players=num_players)

        # Create MCTS agents for each player
        # IMPORTANT: Disable exploration noise during evaluation (Session 3)
        # Evaluation should be deterministic for fair comparison
        mcts_agents = {}
        for player_idx in range(num_players):
            model_idx = player_assignments[player_idx]
            model = models[model_idx]
            mcts_agents[player_idx] = ImperfectInfoMCTS(
                network=model,
                encoder=self.encoder,
                masker=self.masker,
                num_determinizations=self.num_determinizations,
                simulations_per_determinization=self.simulations_per_determinization,
                exploration_noise_epsilon=0.0,  # NO NOISE in evaluation
                exploration_noise_alpha=0.3,    # Unused when epsilon=0, but set for consistency
            )

        # Define callbacks for bidding and card playing
        def get_bid(player, cards_dealt, is_dealer, total_bids, num_cards):
            """Callback to get bid from MCTS agent."""
            mcts = mcts_agents[player.position]
            action_probs = mcts.search(game, player)  # Pass player object, not position
            # Choose best bid (greedy for evaluation)
            bid = max(action_probs.keys(), key=action_probs.get)
            return bid

        def get_card(player, legal_cards, trick):
            """Callback to get card to play from MCTS agent."""
            mcts = mcts_agents[player.position]
            action_probs = mcts.search(game, player)  # Pass player object, not position
            # Choose best card (greedy for evaluation)
            # Map action index back to card
            best_action_idx = max(action_probs.keys(), key=action_probs.get)
            # Find matching card in legal_cards
            for card in legal_cards:
                if self.encoder._card_to_index(card) == best_action_idx:
                    return card
            # Fallback to first legal card if no match
            return legal_cards[0]

        # Play the round
        result = game.play_round(cards_to_deal, get_bid, get_card)

        # Extract final scores by player position
        scores = {}
        for player_result in result["player_scores"]:
            # Find player by name to get position
            for player in game.players:
                if player.name == player_result["name"]:
                    scores[player.position] = player_result["round_score"]
                    break

        return scores

    def calculate_win_rate(
        self,
        match_results: Dict[str, Any],
    ) -> float:
        """
        Calculate win rate from match results.

        Args:
            match_results: Results from play_match

        Returns:
            Win rate (0.0 to 1.0)
        """
        return match_results['win_rate']

    def head_to_head_tournament(
        self,
        models: List[BlobNet],
        model_names: List[str],
        games_per_matchup: int = 100,
        num_players: int = 4,
        cards_to_deal: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a round-robin tournament between multiple models.

        Each model plays against every other model. Useful for comparing
        multiple model checkpoints at once.

        Args:
            models: List of models to compete
            model_names: Names for each model
            games_per_matchup: Games to play per model pair
            num_players: Players per game
            cards_to_deal: Cards to deal
            verbose: Print progress messages

        Returns:
            Tournament results with win matrix and rankings
        """
        num_models = len(models)
        if num_models < 2:
            raise ValueError("Need at least 2 models for a tournament")

        if len(model_names) != num_models:
            raise ValueError("Number of model names must match number of models")

        if verbose:
            print(f"Starting round-robin tournament with {num_models} models")

        # Win matrix: wins[i][j] = games model i won against model j
        wins = defaultdict(lambda: defaultdict(int))
        total_scores = defaultdict(float)
        games_played = defaultdict(lambda: defaultdict(int))

        # Play all matchups
        for i in range(num_models):
            for j in range(i + 1, num_models):
                if verbose:
                    print(f"\nMatchup: {model_names[i]} vs {model_names[j]}")

                results = self.play_match(
                    models[i],
                    models[j],
                    num_games=games_per_matchup,
                    num_players=num_players,
                    cards_to_deal=cards_to_deal,
                    verbose=verbose,
                )

                # Record results
                wins[i][j] = results['model1_wins']
                wins[j][i] = results['model2_wins']
                games_played[i][j] = results['games_played']
                games_played[j][i] = results['games_played']
                total_scores[i] += results['model1_avg_score'] * results['games_played']
                total_scores[j] += results['model2_avg_score'] * results['games_played']

        # Calculate rankings
        rankings = []
        for i in range(num_models):
            total_wins = sum(wins[i].values())
            total_games = sum(games_played[i].values())
            win_rate = total_wins / total_games if total_games > 0 else 0.0
            avg_score = (
                total_scores[i] / total_games if total_games > 0 else 0.0
            )

            rankings.append({
                'model_name': model_names[i],
                'model_idx': i,
                'total_wins': total_wins,
                'total_games': total_games,
                'win_rate': win_rate,
                'avg_score': avg_score,
            })

        # Sort by win rate
        rankings.sort(key=lambda x: x['win_rate'], reverse=True)

        if verbose:
            print("\n" + "=" * 60)
            print("TOURNAMENT RESULTS")
            print("=" * 60)
            for rank, result in enumerate(rankings, 1):
                print(
                    f"{rank}. {result['model_name']}: "
                    f"{result['total_wins']}/{result['total_games']} wins "
                    f"({result['win_rate']:.1%}), "
                    f"avg score: {result['avg_score']:.1f}"
                )

        return {
            'rankings': rankings,
            'win_matrix': dict(wins),
            'games_played': dict(games_played),
        }

    def _generate_round_sequence(self, num_players: int, start_cards: int) -> List[int]:
        """
        Generate P-conditional round sequence for full games (Session 4).

        Sequences follow the pattern: descending → ones → ascending
        - 5p/C=7: [7,6,5,4,3,2, 1,1,1,1,1, 2,3,4,5,6,7] (17 rounds)
        - 4p/C=8: [8,7,6,5,4,3,2, 1,1,1,1, 2,3,4,5,6,7,8] (18 rounds)
        - 4p/C=7: [7,6,5,4,3,2, 1,1,1,1, 2,3,4,5,6,7] (16 rounds)

        Args:
            num_players: Number of players in the game
            start_cards: Starting number of cards

        Returns:
            List of card counts for each round in sequence
        """
        # Descending phase: start_cards down to 2
        descending = list(range(start_cards, 1, -1))

        # Ones phase: P rounds of 1 card each
        ones = [1] * num_players

        # Ascending phase: 2 up to start_cards
        ascending = list(range(2, start_cards + 1))

        # Combine: descending + ones + ascending
        sequence = descending + ones + ascending

        return sequence

    def _get_phase(self, round_idx: int, round_sequence: List[int]) -> str:
        """
        Determine which phase a round is in (Session 4).

        Args:
            round_idx: Index of the round in the sequence (0-indexed)
            round_sequence: Full round sequence

        Returns:
            Phase name: 'descending', 'ones', or 'ascending'
        """
        cards = round_sequence[round_idx]

        # Check if we're ascending or descending by looking at neighbors
        if round_idx > 0:
            prev_cards = round_sequence[round_idx - 1]
            if cards > prev_cards:
                # Ascending: cards increase
                return 'ascending'
            elif cards < prev_cards:
                # Descending: cards decrease
                return 'descending'
            else:
                # Same number of cards (multiple 1s)
                # Check if next round increases (we're at end of ones, start of ascending)
                if round_idx < len(round_sequence) - 1:
                    next_cards = round_sequence[round_idx + 1]
                    if next_cards > cards:
                        return 'ascending'
                # Otherwise we're in ones phase
                return 'ones'
        else:
            # First round is always descending
            return 'descending'

    def _sample_full_game_configs(
        self,
        num_sequences: int,
        player_distribution: Dict[int, float] = None,
        start_card_distribution_4p: Dict[int, float] = None,
    ) -> List[Tuple[int, int]]:
        """
        Sample (num_players, start_cards) configurations for full games (Session 4).

        Args:
            num_sequences: Number of configurations to sample
            player_distribution: Distribution over player counts (default: 4:15%, 5:70%, 6:15%)
            start_card_distribution_4p: Distribution over start cards for 4p (default: 7:60%, 8:40%)

        Returns:
            List of (num_players, start_cards) tuples
        """
        # Default distributions from TrainingConfig
        if player_distribution is None:
            player_distribution = {4: 0.15, 5: 0.70, 6: 0.15}
        if start_card_distribution_4p is None:
            start_card_distribution_4p = {7: 0.60, 8: 0.40}

        configs = []
        for _ in range(num_sequences):
            # Sample number of players
            players = np.random.choice(
                list(player_distribution.keys()),
                p=list(player_distribution.values())
            )

            # Sample start cards based on player count
            if players == 4:
                start_cards = np.random.choice(
                    list(start_card_distribution_4p.keys()),
                    p=list(start_card_distribution_4p.values())
                )
            else:
                # For 5p and 6p, always use 7 cards
                start_cards = 7

            configs.append((players, start_cards))

        return configs

    def _play_single_round(
        self,
        num_players: int,
        cards_to_deal: int,
        game_context: GameContext,
        model1: BlobNet,
        model2: BlobNet,
        player_assignments: List[int],
    ) -> Dict[int, int]:
        """
        Play a single round with GameContext for full-game evaluation (Session 4).

        Args:
            num_players: Number of players
            cards_to_deal: Cards to deal per player
            game_context: Game context with cumulative scores and history
            model1: First model
            model2: Second model
            player_assignments: Which model controls each player (list of 0s and 1s)

        Returns:
            Dictionary mapping player_position -> round_score
        """
        # Initialize game
        game = BlobGame(num_players=num_players)

        # Create MCTS agents for each player (no exploration noise)
        mcts_agents = {}
        models = [model1, model2]
        for player_idx in range(num_players):
            model_idx = player_assignments[player_idx]
            model = models[model_idx]
            mcts_agents[player_idx] = ImperfectInfoMCTS(
                network=model,
                encoder=self.encoder,
                masker=self.masker,
                num_determinizations=self.num_determinizations,
                simulations_per_determinization=self.simulations_per_determinization,
                exploration_noise_epsilon=0.0,  # NO NOISE in evaluation
                exploration_noise_alpha=0.3,
            )

        # Define callbacks that pass game_context to MCTS
        def get_bid(player, cards_dealt, is_dealer, total_bids, num_cards):
            """Callback to get bid from MCTS agent with game context."""
            mcts = mcts_agents[player.position]
            # Pass game_context to MCTS search
            action_probs = mcts.search(game, player, game_context=game_context)
            # Choose best bid (greedy for evaluation)
            bid = max(action_probs.keys(), key=action_probs.get)
            return bid

        def get_card(player, legal_cards, trick):
            """Callback to get card to play from MCTS agent with game context."""
            mcts = mcts_agents[player.position]
            # Pass game_context to MCTS search
            action_probs = mcts.search(game, player, game_context=game_context)
            # Choose best card (greedy for evaluation)
            best_action_idx = max(action_probs.keys(), key=action_probs.get)
            # Map action index back to card
            for card in legal_cards:
                if self.encoder._card_to_index(card) == best_action_idx:
                    return card
            # Fallback to first legal card
            return legal_cards[0]

        # Play the round
        result = game.play_round(cards_to_deal, get_bid, get_card)

        # Extract round scores by player position
        scores = {}
        for player_result in result["player_scores"]:
            for player in game.players:
                if player.name == player_result["name"]:
                    scores[player.position] = player_result["round_score"]
                    break

        return scores

    def _evaluate_full_games(
        self,
        model1: BlobNet,
        model2: BlobNet,
        num_sequences: int = 100,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate models on full multi-round game sequences (Session 4).

        Plays complete games with P-conditional round sequences and cumulative scoring.
        Models are evaluated based on total game scores across all rounds.

        Args:
            model1: First model (challenger)
            model2: Second model (champion)
            num_sequences: Number of full game sequences to play
            verbose: Print progress messages

        Returns:
            Match results with total game scores:
            - model1_wins: Games where model1's total score was highest
            - model2_wins: Games where model2's total score was highest
            - draws: Games with tied total scores
            - model1_avg_score: Average total game score for model1
            - model2_avg_score: Average total game score for model2
            - model1_total_score: Sum of all total game scores
            - model2_total_score: Sum of all total game scores
            - win_rate: model1 win rate
            - games_played: Total game sequences played
        """
        if verbose:
            print(f"Starting full-game evaluation: {num_sequences} game sequences")

        # Set models to eval mode
        model1.eval()
        model2.eval()

        # Track results
        model1_scores = []
        model2_scores = []
        model1_wins = 0
        model2_wins = 0
        draws = 0

        # Sample game configurations
        configs = self._sample_full_game_configs(num_sequences)

        games_played = 0
        for config_idx, (num_players, start_cards) in enumerate(configs):
            # Generate round sequence for this configuration
            round_sequence = self._generate_round_sequence(num_players, start_cards)
            total_rounds = len(round_sequence)

            if verbose and games_played % 10 == 0:
                print(f"  Game {games_played + 1}/{num_sequences}: "
                      f"{num_players}p/{start_cards}c ({total_rounds} rounds)")

            # Initialize cumulative scores for this game
            cumulative_scores = [0] * num_players

            # Rotate which model plays each position for fairness
            # Model1 plays as position (games_played % num_players)
            model1_position = games_played % num_players
            player_assignments = [1] * num_players  # 1 = model2
            player_assignments[model1_position] = 0  # 0 = model1

            # Play all rounds in the sequence
            for round_idx, cards_to_deal in enumerate(round_sequence):
                # Determine phase
                phase = self._get_phase(round_idx, round_sequence)

                # Build GameContext
                game_context = GameContext(
                    cumulative_scores=cumulative_scores.copy(),
                    rounds_completed=round_idx,
                    total_rounds=total_rounds,
                    previous_cards=[round_sequence[i] for i in range(round_idx)],
                    num_players=num_players,
                    start_cards=start_cards,
                    phase=phase,
                )

                # Play this round
                round_scores = self._play_single_round(
                    num_players=num_players,
                    cards_to_deal=cards_to_deal,
                    game_context=game_context,
                    model1=model1,
                    model2=model2,
                    player_assignments=player_assignments,
                )

                # Update cumulative scores
                for player_idx in range(num_players):
                    cumulative_scores[player_idx] += round_scores[player_idx]

            # Record total game scores
            model1_total_score = cumulative_scores[model1_position]
            # Average of other players' scores for model2
            model2_total_score = sum(
                cumulative_scores[p] for p in range(num_players) if p != model1_position
            ) / (num_players - 1)

            model1_scores.append(model1_total_score)
            model2_scores.append(model2_total_score)

            # Determine winner (model1 vs best of model2 players)
            max_model2_score = max(
                cumulative_scores[p] for p in range(num_players) if p != model1_position
            )
            if model1_total_score > max_model2_score:
                model1_wins += 1
            elif model1_total_score < max_model2_score:
                model2_wins += 1
            else:
                draws += 1

            games_played += 1

            if verbose and games_played % 10 == 0:
                current_win_rate = model1_wins / games_played
                print(
                    f"  Progress: {games_played}/{num_sequences} games, "
                    f"Model1 win rate: {current_win_rate:.1%}"
                )

        # Calculate final statistics
        model1_avg_score = np.mean(model1_scores)
        model2_avg_score = np.mean(model2_scores)
        model1_total_score_sum = sum(model1_scores)
        model2_total_score_sum = sum(model2_scores)
        win_rate = model1_wins / games_played if games_played > 0 else 0.0

        results = {
            'model1_wins': model1_wins,
            'model2_wins': model2_wins,
            'draws': draws,
            'model1_avg_score': float(model1_avg_score),
            'model2_avg_score': float(model2_avg_score),
            'model1_total_score': int(model1_total_score_sum),
            'model2_total_score': int(model2_total_score_sum),
            'win_rate': float(win_rate),
            'games_played': games_played,
        }

        if verbose:
            print(f"\nFull-game evaluation complete!")
            print(f"  Model1 wins: {model1_wins}")
            print(f"  Model2 wins: {model2_wins}")
            print(f"  Draws: {draws}")
            print(f"  Model1 win rate: {win_rate:.1%}")
            print(f"  Model1 avg total score: {model1_avg_score:.1f}")
            print(f"  Model2 avg total score: {model2_avg_score:.1f}")

        return results
