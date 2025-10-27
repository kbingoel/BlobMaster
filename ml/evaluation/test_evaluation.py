"""
Tests for model evaluation and ELO tracking.

Tests the Arena system for model vs model evaluation and the ELO rating
system for tracking model improvement over training.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from ml.evaluation.arena import Arena
from ml.evaluation.elo import ELOTracker
from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker


@pytest.fixture
def small_network():
    """Create a small network for testing."""
    return BlobNet(
        embedding_dim=64,
        num_layers=2,
        num_heads=2,
        dropout=0.1,
    )


@pytest.fixture
def encoder():
    """Create state encoder."""
    return StateEncoder()


@pytest.fixture
def masker():
    """Create action masker."""
    return ActionMasker()


@pytest.fixture
def arena(encoder, masker):
    """Create arena for testing."""
    return Arena(
        encoder=encoder,
        masker=masker,
        num_determinizations=2,
        simulations_per_determinization=10,
        device='cpu',
    )


class TestArena:
    """Tests for Arena tournament system."""

    def test_arena_initialization(self, encoder, masker):
        """Test arena initializes correctly."""
        arena = Arena(
            encoder=encoder,
            masker=masker,
            num_determinizations=3,
            simulations_per_determinization=50,
            device='cpu',
        )

        assert arena.encoder is encoder
        assert arena.masker is masker
        assert arena.num_determinizations == 3
        assert arena.simulations_per_determinization == 50
        assert arena.device == 'cpu'

    def test_single_game_evaluation(self, arena, small_network):
        """Test playing a single evaluation game."""
        model1 = small_network
        model2 = small_network

        # Play a single game
        scores = arena._play_single_game(
            models=[model1, model2],
            player_assignments=[0, 1, 1, 1],  # model1 is player 0
            num_players=4,
            cards_to_deal=5,
        )

        # Check scores are returned for all players
        assert len(scores) == 4
        for player_idx in range(4):
            assert player_idx in scores
            assert isinstance(scores[player_idx], (int, float))
            assert scores[player_idx] >= 0  # Scores should be non-negative

    def test_match_between_models(self, arena, small_network):
        """Test playing full match between two models."""
        model1 = small_network
        model2 = small_network

        # Play a small match
        results = arena.play_match(
            model1=model1,
            model2=model2,
            num_games=4,  # Small number for testing
            num_players=4,
            cards_to_deal=5,
            verbose=False,
        )

        # Check all required fields are present
        assert 'model1_wins' in results
        assert 'model2_wins' in results
        assert 'draws' in results
        assert 'model1_avg_score' in results
        assert 'model2_avg_score' in results
        assert 'win_rate' in results
        assert 'games_played' in results

        # Check values are reasonable
        assert results['games_played'] == 4
        assert results['model1_wins'] + results['model2_wins'] + results['draws'] == 4
        assert 0.0 <= results['win_rate'] <= 1.0

    def test_win_rate_calculation(self, arena):
        """Test win rate calculation is correct."""
        match_results = {
            'model1_wins': 60,
            'model2_wins': 40,
            'draws': 0,
            'win_rate': 0.6,
            'games_played': 100,
        }

        win_rate = arena.calculate_win_rate(match_results)
        assert win_rate == 0.6

    def test_match_with_different_player_counts(self, arena, small_network):
        """Test matches work with different player counts."""
        model1 = small_network
        model2 = small_network

        # Test 3 players
        results_3p = arena.play_match(
            model1=model1,
            model2=model2,
            num_games=3,
            num_players=3,
            cards_to_deal=5,
            verbose=False,
        )
        assert results_3p['games_played'] == 3

        # Test 6 players
        results_6p = arena.play_match(
            model1=model1,
            model2=model2,
            num_games=6,
            num_players=6,
            cards_to_deal=5,
            verbose=False,
        )
        assert results_6p['games_played'] == 6

    def test_head_to_head_tournament(self, arena, small_network):
        """Test round-robin tournament between multiple models."""
        # Create 3 models (same architecture for testing)
        models = [small_network for _ in range(3)]
        model_names = ['Model A', 'Model B', 'Model C']

        # Run tournament
        results = arena.head_to_head_tournament(
            models=models,
            model_names=model_names,
            games_per_matchup=2,  # Small number for testing
            num_players=4,
            cards_to_deal=5,
            verbose=False,
        )

        # Check results structure
        assert 'rankings' in results
        assert 'win_matrix' in results
        assert 'games_played' in results

        # Check rankings
        assert len(results['rankings']) == 3
        for rank in results['rankings']:
            assert 'model_name' in rank
            assert 'win_rate' in rank
            assert 'total_wins' in rank
            assert 'total_games' in rank

        # Rankings should be sorted by win rate
        win_rates = [r['win_rate'] for r in results['rankings']]
        assert win_rates == sorted(win_rates, reverse=True)


class TestELOTracker:
    """Tests for ELO rating system."""

    def test_elo_tracker_initialization(self):
        """Test ELO tracker initializes correctly."""
        tracker = ELOTracker(initial_elo=1000, k_factor=32)

        assert tracker.initial_elo == 1000
        assert tracker.k_factor == 32
        assert len(tracker.history) == 0
        assert tracker.get_current_elo() == 1000

    def test_expected_score_calculation(self):
        """Test expected score formula."""
        tracker = ELOTracker()

        # Equal players should have 0.5 expected score
        expected = tracker.calculate_expected_score(1000, 1000)
        assert abs(expected - 0.5) < 0.01

        # Higher rated player should have >0.5 expected score
        expected_high = tracker.calculate_expected_score(1200, 1000)
        assert expected_high > 0.5

        # Lower rated player should have <0.5 expected score
        expected_low = tracker.calculate_expected_score(1000, 1200)
        assert expected_low < 0.5

        # Complement rule: E(A vs B) + E(B vs A) = 1
        assert abs(expected_high + expected_low - 1.0) < 0.01

    def test_elo_update(self):
        """Test ELO updates correctly."""
        tracker = ELOTracker(k_factor=32)

        # Player wins when expected to draw
        new_elo = tracker.update_elo(
            player_elo=1000,
            opponent_elo=1000,
            actual_score=1.0,  # Win
        )

        # Should gain ELO (expected 0.5, got 1.0)
        assert new_elo > 1000
        assert new_elo == 1000 + 32 * (1.0 - 0.5)  # K * (actual - expected)

        # Player loses when expected to draw
        new_elo_loss = tracker.update_elo(
            player_elo=1000,
            opponent_elo=1000,
            actual_score=0.0,  # Loss
        )

        # Should lose ELO
        assert new_elo_loss < 1000

    def test_add_match_result(self):
        """Test recording match results."""
        tracker = ELOTracker(initial_elo=1000)

        # Add first match result
        new_elo = tracker.add_match_result(
            iteration=0,
            model_elo=1000,
            opponent_elo=1000,
            win_rate=0.6,
            games_played=100,
            model_avg_score=15.5,
            opponent_avg_score=14.2,
        )

        # ELO should increase (won 60% when expected 50%)
        assert new_elo > 1000

        # Check history was recorded
        assert len(tracker.history) == 1
        assert tracker.history[0]['iteration'] == 0
        assert tracker.history[0]['model_elo_before'] == 1000
        assert tracker.history[0]['model_elo_after'] == new_elo
        assert tracker.history[0]['win_rate'] == 0.6

        # Current ELO should update
        assert tracker.get_current_elo() == new_elo

    def test_promotion_threshold(self):
        """Test model promotion logic."""
        tracker = ELOTracker()

        # Should promote at 55% win rate (default threshold)
        assert tracker.should_promote_model(0.55) is True
        assert tracker.should_promote_model(0.60) is True

        # Should not promote below threshold
        assert tracker.should_promote_model(0.54) is False
        assert tracker.should_promote_model(0.50) is False

        # Custom threshold
        assert tracker.should_promote_model(0.52, threshold=0.52) is True
        assert tracker.should_promote_model(0.51, threshold=0.52) is False

    def test_elo_history_persistence(self):
        """Test saving and loading ELO history."""
        tracker = ELOTracker(initial_elo=1000, k_factor=32)

        # Add some history
        tracker.add_match_result(0, 1000, 1000, 0.6, 100, 15.0, 14.0)
        tracker.add_match_result(1, tracker.get_current_elo(), 1000, 0.55, 100, 15.2, 14.8)

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            tracker.save_history(temp_path)

            # Load into new tracker
            new_tracker = ELOTracker()
            new_tracker.load_history(temp_path)

            # Check values match
            assert new_tracker.initial_elo == tracker.initial_elo
            assert new_tracker.k_factor == tracker.k_factor
            assert len(new_tracker.history) == len(tracker.history)
            assert new_tracker.get_current_elo() == tracker.get_current_elo()

        finally:
            # Clean up
            Path(temp_path).unlink()

    def test_get_statistics(self):
        """Test ELO statistics calculation."""
        tracker = ELOTracker(initial_elo=1000)

        # Empty tracker
        stats = tracker.get_statistics()
        assert stats['num_evaluations'] == 0
        assert stats['current_elo'] == 1000
        assert stats['total_elo_gain'] == 0
        assert stats['promotions'] == 0

        # Add matches with varying outcomes
        tracker.add_match_result(0, 1000, 1000, 0.6, 100, 15.0, 14.0)  # Promotion (>55%)
        tracker.add_match_result(1, tracker.get_current_elo(), 1000, 0.5, 100, 15.0, 15.0)  # No promotion
        tracker.add_match_result(2, tracker.get_current_elo(), 1000, 0.7, 100, 16.0, 13.0)  # Promotion

        stats = tracker.get_statistics()
        assert stats['num_evaluations'] == 3
        assert stats['current_elo'] == tracker.get_current_elo()
        assert stats['promotions'] == 2  # Two matches with win_rate >= 0.55

    def test_elo_progression_with_multiple_matches(self):
        """Test ELO progression over multiple matches."""
        tracker = ELOTracker(initial_elo=1000, k_factor=32)

        # Simulate improving model (increasing win rates)
        win_rates = [0.52, 0.55, 0.58, 0.60, 0.62]
        elos = [1000]

        for i, win_rate in enumerate(win_rates):
            current_elo = tracker.get_current_elo()
            new_elo = tracker.add_match_result(
                iteration=i,
                model_elo=current_elo,
                opponent_elo=1000,
                win_rate=win_rate,
                games_played=100,
                model_avg_score=15.0,
                opponent_avg_score=14.0,
            )
            elos.append(new_elo)

        # ELO should generally increase
        assert elos[-1] > elos[0]

        # Check history length
        assert len(tracker.history) == 5

    def test_print_summary(self, capsys):
        """Test printing ELO summary."""
        tracker = ELOTracker(initial_elo=1000)

        # Add some matches
        tracker.add_match_result(0, 1000, 1000, 0.6, 100, 15.0, 14.0)
        tracker.add_match_result(1, tracker.get_current_elo(), 1000, 0.55, 100, 15.0, 15.0)

        # Print summary
        tracker.print_summary()

        # Capture output
        captured = capsys.readouterr()

        # Check output contains expected information
        assert "ELO RATING SUMMARY" in captured.out
        assert "Total Evaluations: 2" in captured.out
        assert "Recent Evaluations:" in captured.out


class TestEvaluationIntegration:
    """Integration tests for evaluation system."""

    def test_arena_with_elo_tracking(self, arena, small_network):
        """Test using Arena with ELO tracker together."""
        model1 = small_network
        model2 = small_network
        tracker = ELOTracker(initial_elo=1000)

        # Play match
        results = arena.play_match(
            model1=model1,
            model2=model2,
            num_games=10,
            verbose=False,
        )

        # Update ELO based on results
        new_elo = tracker.add_match_result(
            iteration=0,
            model_elo=1000,
            opponent_elo=1000,
            win_rate=results['win_rate'],
            games_played=results['games_played'],
            model_avg_score=results['model1_avg_score'],
            opponent_avg_score=results['model2_avg_score'],
        )

        # Check ELO was updated
        assert tracker.get_current_elo() == new_elo
        assert len(tracker.history) == 1

    def test_multiple_evaluation_rounds(self, arena, small_network):
        """Test multiple rounds of evaluation."""
        tracker = ELOTracker(initial_elo=1000)

        # Simulate 5 evaluation rounds
        for i in range(5):
            model = small_network

            results = arena.play_match(
                model1=model,
                model2=model,
                num_games=4,
                verbose=False,
            )

            current_elo = tracker.get_current_elo()
            new_elo = tracker.add_match_result(
                iteration=i,
                model_elo=current_elo,
                opponent_elo=current_elo,
                win_rate=results['win_rate'],
                games_played=results['games_played'],
                model_avg_score=results['model1_avg_score'],
                opponent_avg_score=results['model2_avg_score'],
            )

        # Check history
        assert len(tracker.history) == 5
        stats = tracker.get_statistics()
        assert stats['num_evaluations'] == 5
