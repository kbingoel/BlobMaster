"""
Arena system for model vs model evaluation.

This module implements a tournament system where two models play against each
other to determine which one performs better. Used to track model improvement
during training and decide when to promote a new model to "best model" status.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict

from ml.game.blob import BlobGame
from ml.network.encode import StateEncoder, ActionMasker
from ml.network.model import BlobNet
from ml.mcts.search import ImperfectInfoMCTS


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
            full_game_mode: Enable full multi-round game evaluation (Session 4, not yet implemented)

        Raises:
            NotImplementedError: If full_game_mode=True (Session 4 feature)
        """
        self.encoder = encoder
        self.masker = masker
        self.num_determinizations = num_determinizations
        self.simulations_per_determinization = simulations_per_determinization
        self.device = device
        self.full_game_mode = full_game_mode

        # Session 0: Stub for full-game evaluation (Session 4 will implement)
        if full_game_mode:
            raise NotImplementedError(
                "Full-game evaluation not yet implemented. "
                "Session 4 will add P-conditional sequences and total-game scoring. "
                "For now, use full_game_mode=False to evaluate on single rounds."
            )

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
            num_games: Number of games to play
            num_players: Players per game
            cards_to_deal: Cards to deal per player
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
