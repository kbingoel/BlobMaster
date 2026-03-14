"""
ELO rating system for tracking model improvement.

This module implements an ELO rating system to quantitatively track how models
improve over training iterations. Each time a new model is evaluated against
the previous best, ELO ratings are updated based on the outcome.
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path


class ELOTracker:
    """
    Tracks ELO ratings for model generations.

    Maintains history of model performance over training. Uses standard
    ELO rating formula to calculate relative strength of models.

    ELO System:
        - Expected score: E = 1 / (1 + 10^((opponent_elo - player_elo) / 400))
        - Rating update: new_elo = old_elo + K * (actual_score - expected_score)
        - K-factor: Controls how quickly ratings change (higher = more volatile)
    """

    def __init__(
        self,
        initial_elo: int = 1000,
        k_factor: int = 32,
    ):
        """
        Initialize ELO tracker.

        Args:
            initial_elo: Starting ELO rating for new models
            k_factor: ELO update rate (32 is standard for chess)
        """
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.history: List[Dict[str, Any]] = []

    def calculate_expected_score(
        self,
        player_elo: int,
        opponent_elo: int,
    ) -> float:
        """
        Calculate expected score for a player.

        Uses standard ELO formula to predict the expected outcome of a match
        based on the rating difference between two players.

        Args:
            player_elo: Player's current ELO rating
            opponent_elo: Opponent's current ELO rating

        Returns:
            Expected score (0.0 to 1.0)
            - 0.5 means players are equally strong
            - Higher values mean player is expected to win
            - Lower values mean player is expected to lose
        """
        rating_diff = opponent_elo - player_elo
        expected = 1.0 / (1.0 + 10.0 ** (rating_diff / 400.0))
        return expected

    def update_elo(
        self,
        player_elo: int,
        opponent_elo: int,
        actual_score: float,
    ) -> int:
        """
        Update player's ELO based on match result.

        Args:
            player_elo: Player's current ELO rating
            opponent_elo: Opponent's current ELO rating
            actual_score: Actual match score
                - 0.0 = loss
                - 0.5 = draw
                - 1.0 = win
                - Can use win rate (e.g., 0.6 for 60% wins)

        Returns:
            Updated player ELO (rounded to nearest integer)
        """
        expected_score = self.calculate_expected_score(player_elo, opponent_elo)
        elo_change = self.k_factor * (actual_score - expected_score)
        new_elo = player_elo + elo_change
        return round(new_elo)

    def add_match_result(
        self,
        iteration: int,
        model_elo: int,
        opponent_elo: int,
        win_rate: float,
        games_played: int,
        model_avg_score: float,
        opponent_avg_score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Record a match result and update ELO.

        Args:
            iteration: Training iteration number
            model_elo: Current model's ELO before this match
            opponent_elo: Opponent's ELO before this match
            win_rate: Win rate achieved (0.0 to 1.0)
            games_played: Number of games played
            model_avg_score: Average score achieved by model
            opponent_avg_score: Average score achieved by opponent
            metadata: Optional additional metadata to store

        Returns:
            Updated model ELO
        """
        # Calculate new ELO based on win rate
        new_elo = self.update_elo(model_elo, opponent_elo, win_rate)

        # Calculate opponent's new ELO (from their perspective)
        # Opponent's actual score is inverse of win rate
        opponent_new_elo = self.update_elo(
            opponent_elo, model_elo, 1.0 - win_rate
        )

        # Record in history
        history_entry = {
            'iteration': iteration,
            'model_elo_before': model_elo,
            'model_elo_after': new_elo,
            'opponent_elo_before': opponent_elo,
            'opponent_elo_after': opponent_new_elo,
            'win_rate': win_rate,
            'games_played': games_played,
            'model_avg_score': model_avg_score,
            'opponent_avg_score': opponent_avg_score,
            'elo_change': new_elo - model_elo,
        }

        if metadata:
            history_entry['metadata'] = metadata

        self.history.append(history_entry)

        return new_elo

    def get_current_elo(self) -> int:
        """
        Get the most recent ELO rating.

        Returns:
            Current ELO rating, or initial_elo if no matches yet
        """
        if not self.history:
            return self.initial_elo
        return self.history[-1]['model_elo_after']

    def get_elo_history(self) -> List[Dict[str, Any]]:
        """
        Get full ELO history.

        Returns:
            List of match result dictionaries with ELO progression
        """
        return self.history.copy()

    def save_history(self, filepath: str):
        """
        Save ELO history to JSON file.

        Args:
            filepath: Path to save history to
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        data = {
            'initial_elo': self.initial_elo,
            'k_factor': self.k_factor,
            'history': self.history,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_history(self, filepath: str):
        """
        Load ELO history from JSON file.

        Args:
            filepath: Path to load history from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.initial_elo = data.get('initial_elo', self.initial_elo)
        self.k_factor = data.get('k_factor', self.k_factor)
        self.history = data.get('history', [])

    def should_promote_model(
        self,
        win_rate: float,
        threshold: float = 0.55,
    ) -> bool:
        """
        Determine if new model should replace best model.

        A model is promoted if it achieves a win rate above the threshold.
        The threshold is typically set slightly above 50% to require clear
        improvement and avoid promoting models due to random variation.

        Args:
            win_rate: Win rate against current best (0.0 to 1.0)
            threshold: Minimum win rate for promotion (default: 0.55 = 55%)

        Returns:
            True if model should be promoted
        """
        return win_rate >= threshold

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about ELO progression.

        Returns:
            Dictionary with:
            - num_evaluations: Total number of evaluations
            - current_elo: Most recent ELO
            - initial_elo: Starting ELO
            - max_elo: Highest ELO achieved
            - min_elo: Lowest ELO (after initial)
            - avg_elo_change: Average ELO change per evaluation
            - total_elo_gain: Total ELO gained from start
            - promotions: Number of times model was promoted (win_rate >= 0.55)
        """
        if not self.history:
            return {
                'num_evaluations': 0,
                'current_elo': self.initial_elo,
                'initial_elo': self.initial_elo,
                'max_elo': self.initial_elo,
                'min_elo': self.initial_elo,
                'avg_elo_change': 0.0,
                'total_elo_gain': 0,
                'promotions': 0,
            }

        current_elo = self.get_current_elo()
        elo_values = [entry['model_elo_after'] for entry in self.history]
        elo_changes = [entry['elo_change'] for entry in self.history]
        promotions = sum(
            1 for entry in self.history if entry['win_rate'] >= 0.55
        )

        return {
            'num_evaluations': len(self.history),
            'current_elo': current_elo,
            'initial_elo': self.initial_elo,
            'max_elo': max(elo_values),
            'min_elo': min(elo_values),
            'avg_elo_change': sum(elo_changes) / len(elo_changes),
            'total_elo_gain': current_elo - self.initial_elo,
            'promotions': promotions,
        }

    def plot_elo_progression(self, save_path: Optional[str] = None):
        """
        Plot ELO rating progression over iterations.

        Requires matplotlib. If save_path is provided, saves the plot to file.
        Otherwise, displays it.

        Args:
            save_path: Optional path to save plot to
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed, cannot plot ELO progression")
            return

        if not self.history:
            print("No ELO history to plot")
            return

        iterations = [entry['iteration'] for entry in self.history]
        elo_values = [entry['model_elo_after'] for entry in self.history]
        win_rates = [entry['win_rate'] for entry in self.history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot ELO progression
        ax1.plot(iterations, elo_values, 'b-o', linewidth=2, markersize=4)
        ax1.axhline(
            y=self.initial_elo, color='r', linestyle='--', label='Initial ELO'
        )
        ax1.set_xlabel('Training Iteration')
        ax1.set_ylabel('ELO Rating')
        ax1.set_title('Model ELO Progression')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot win rates
        ax2.plot(iterations, win_rates, 'g-o', linewidth=2, markersize=4)
        ax2.axhline(
            y=0.5, color='r', linestyle='--', label='50% (Even)'
        )
        ax2.axhline(
            y=0.55, color='orange', linestyle='--', label='55% (Promotion threshold)'
        )
        ax2.set_xlabel('Training Iteration')
        ax2.set_ylabel('Win Rate')
        ax2.set_title('Win Rate vs Previous Best')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"ELO progression plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def print_summary(self):
        """Print a human-readable summary of ELO progression."""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("ELO RATING SUMMARY")
        print("=" * 60)
        print(f"Total Evaluations: {stats['num_evaluations']}")
        print(f"Current ELO: {stats['current_elo']}")
        print(f"Initial ELO: {stats['initial_elo']}")
        print(f"Total Gain: {stats['total_elo_gain']:+d}")
        print(f"Max ELO: {stats['max_elo']}")
        print(f"Min ELO: {stats['min_elo']}")
        print(f"Avg ELO Change: {stats['avg_elo_change']:+.1f}")
        print(f"Promotions: {stats['promotions']}/{stats['num_evaluations']}")
        print("=" * 60)

        if self.history:
            print("\nRecent Evaluations:")
            for entry in self.history[-5:]:
                elo_change = entry['elo_change']
                sign = '+' if elo_change >= 0 else ''
                print(
                    f"  Iteration {entry['iteration']}: "
                    f"{entry['model_elo_before']} -> {entry['model_elo_after']} "
                    f"({sign}{elo_change}) | "
                    f"Win Rate: {entry['win_rate']:.1%}"
                )
