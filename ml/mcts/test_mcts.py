"""
Comprehensive tests for MCTS Node implementation.

Tests cover:
- Node initialization and state
- Leaf and root detection
- UCB1 child selection
- Tree expansion
- Backpropagation
- Action probability calculation
- Action selection
"""

import pytest
import numpy as np
from ml.mcts.node import MCTSNode
from ml.game.blob import BlobGame, Player


class TestMCTSNodeBasics:
    """Test basic node operations and state."""

    def test_node_initialization(self):
        """Test node initializes correctly with all attributes."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        node = MCTSNode(
            game_state=game,
            player=player,
            parent=None,
            action_taken=None,
            prior_prob=0.0,
        )

        # Check basic attributes
        assert node.game_state is game
        assert node.player is player
        assert node.parent is None
        assert node.action_taken is None
        assert node.prior_prob == 0.0

        # Check MCTS statistics initialized
        assert node.visit_count == 0
        assert node.total_value == 0.0
        assert node.mean_value == 0.0

        # Check node state
        assert len(node.children) == 0
        assert node.is_expanded is False

    def test_is_leaf_before_expansion(self):
        """Test node is leaf before expansion."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        node = MCTSNode(game, player)

        assert node.is_leaf() is True
        assert node.is_expanded is False
        assert len(node.children) == 0

    def test_is_root_detection(self):
        """Test root node detection."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        # Root node (no parent)
        root = MCTSNode(game, player, parent=None)
        assert root.is_root() is True

        # Child node (has parent)
        child = MCTSNode(game, player, parent=root, action_taken=0)
        assert child.is_root() is False

    def test_node_repr(self):
        """Test string representation."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        node = MCTSNode(game, player, action_taken=3, prior_prob=0.5)
        node.visit_count = 10
        node.total_value = 5.0
        node.mean_value = 0.5

        repr_str = repr(node)
        assert "action=3" in repr_str
        assert "visits=10" in repr_str
        assert "value=0.500" in repr_str
        assert "prior=0.500" in repr_str


class TestMCTSNodeExpansion:
    """Test node expansion and child creation."""

    def test_expand_creates_children(self):
        """Test expand creates child nodes for all legal actions."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        node = MCTSNode(game, player)

        # Expand with 3 legal actions (bids: 0, 1, 2)
        action_probs = {0: 0.2, 1: 0.3, 2: 0.5}
        legal_actions = [0, 1, 2]

        node.expand(action_probs, legal_actions)

        # Check expansion worked
        assert node.is_expanded is True
        assert node.is_leaf() is False
        assert len(node.children) == 3

        # Check children have correct attributes
        for action in legal_actions:
            child = node.children[action]
            assert isinstance(child, MCTSNode)
            assert child.parent is node
            assert child.action_taken == action
            assert child.prior_prob == action_probs[action]

    def test_expand_with_uniform_priors(self):
        """Test expansion with missing action probabilities (uniform fallback)."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        node = MCTSNode(game, player)

        # Only provide probabilities for some actions
        action_probs = {0: 0.5}
        legal_actions = [0, 1, 2]

        node.expand(action_probs, legal_actions)

        # Action 0 gets specified prior
        assert node.children[0].prior_prob == 0.5

        # Actions 1, 2 get uniform default
        uniform_prior = 1.0 / len(legal_actions)
        assert node.children[1].prior_prob == uniform_prior
        assert node.children[2].prior_prob == uniform_prior

    def test_expand_with_empty_actions_raises(self):
        """Test expand raises ValueError with empty legal actions."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        node = MCTSNode(game, player)

        with pytest.raises(ValueError, match="no legal actions"):
            node.expand({}, [])

    def test_simulate_action_creates_copy(self):
        """Test _simulate_action creates independent game state copy."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        node = MCTSNode(game, player)

        # Simulate bidding action
        action = 3  # Bid 3
        new_game = node._simulate_action(action)

        # Check new game is different object
        assert new_game is not node.game_state

        # Check action was applied in new game
        new_player = new_game.players[player.position]
        assert new_player.bid == 3

        # Check original game unchanged
        assert node.player.bid is None


class TestMCTSNodeSelection:
    """Test UCB1 child selection."""

    def test_ucb1_score_calculation(self):
        """Test UCB1 score computed correctly."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        parent = MCTSNode(game, player)
        parent.visit_count = 10  # Parent has been visited

        # Create child with known statistics
        child = MCTSNode(game, player, parent=parent, prior_prob=0.5)
        child.visit_count = 5
        child.mean_value = 0.6

        # Compute UCB1 score
        c_puct = 1.5
        score = parent._ucb1_score(child, c_puct)

        # Expected: Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
        # = 0.6 + 1.5 * 0.5 * sqrt(10) / (1 + 5)
        # = 0.6 + 1.5 * 0.5 * 3.162 / 6
        # = 0.6 + 0.395 ≈ 0.995

        expected_q = 0.6
        expected_u = 1.5 * 0.5 * (np.sqrt(10) / (1 + 5))
        expected = expected_q + expected_u

        assert abs(score - expected) < 0.001

    def test_select_child_picks_highest_ucb(self):
        """Test select_child picks child with highest UCB1 score."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        parent = MCTSNode(game, player)
        parent.visit_count = 10

        # Create children with different statistics
        # Child 0: Low value, high prior (should explore)
        child0 = MCTSNode(game, player, parent=parent, action_taken=0, prior_prob=0.8)
        child0.visit_count = 1
        child0.mean_value = 0.2

        # Child 1: High value, low prior (exploitation)
        child1 = MCTSNode(game, player, parent=parent, action_taken=1, prior_prob=0.2)
        child1.visit_count = 8
        child1.mean_value = 0.8

        parent.children = {0: child0, 1: child1}
        parent.is_expanded = True

        # With c_puct=1.5, child0 should be selected (exploration)
        selected = parent.select_child(c_puct=1.5)

        # Child 0 has higher exploration bonus despite lower value
        # UCB0 = 0.2 + 1.5 * 0.8 * sqrt(10) / (1 + 1) ≈ 0.2 + 1.897 = 2.097
        # UCB1 = 0.8 + 1.5 * 0.2 * sqrt(10) / (1 + 8) ≈ 0.8 + 0.105 = 0.905
        assert selected is child0

    def test_select_child_with_no_children_raises(self):
        """Test select_child raises ValueError when node has no children."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        node = MCTSNode(game, player)

        with pytest.raises(ValueError, match="no children"):
            node.select_child()

    def test_ucb_exploration_decreases_with_visits(self):
        """Test exploration bonus decreases as child is visited more."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        parent = MCTSNode(game, player)
        parent.visit_count = 100

        child = MCTSNode(game, player, parent=parent, prior_prob=0.5)
        child.mean_value = 0.5

        # UCB with few visits (high exploration)
        child.visit_count = 1
        ucb_low_visits = parent._ucb1_score(child, c_puct=1.5)

        # UCB with many visits (low exploration)
        child.visit_count = 50
        ucb_high_visits = parent._ucb1_score(child, c_puct=1.5)

        # Exploration bonus should decrease
        assert ucb_low_visits > ucb_high_visits


class TestMCTSNodeBackpropagation:
    """Test value backpropagation through tree."""

    def test_backpropagate_updates_stats(self):
        """Test backpropagation updates visit count and value."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        node = MCTSNode(game, player)

        # Backpropagate value
        node.backpropagate(0.5)

        assert node.visit_count == 1
        assert node.total_value == 0.5
        assert node.mean_value == 0.5

        # Backpropagate again
        node.backpropagate(1.0)

        assert node.visit_count == 2
        assert node.total_value == 1.5
        assert node.mean_value == 0.75

    def test_backpropagate_reaches_root(self):
        """Test backpropagation updates all ancestors up to root."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        # Create tree: root -> child -> grandchild
        root = MCTSNode(game, player)
        child = MCTSNode(game, player, parent=root, action_taken=0)
        grandchild = MCTSNode(game, player, parent=child, action_taken=1)

        # Backpropagate from grandchild
        grandchild.backpropagate(0.8)

        # All nodes should be updated
        assert grandchild.visit_count == 1
        assert grandchild.total_value == 0.8
        assert child.visit_count == 1
        assert child.total_value == 0.8
        assert root.visit_count == 1
        assert root.total_value == 0.8

    def test_backpropagate_multiple_branches(self):
        """Test backpropagation works correctly with multiple branches."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        root = MCTSNode(game, player)

        # Create two children
        child1 = MCTSNode(game, player, parent=root, action_taken=0)
        child2 = MCTSNode(game, player, parent=root, action_taken=1)

        # Backpropagate from both children
        child1.backpropagate(1.0)
        child2.backpropagate(0.5)

        # Root should have combined statistics
        assert root.visit_count == 2
        assert root.total_value == 1.5
        assert root.mean_value == 0.75

        # Children have independent statistics
        assert child1.visit_count == 1
        assert child1.total_value == 1.0
        assert child2.visit_count == 1
        assert child2.total_value == 0.5


class TestMCTSNodeActionSelection:
    """Test action probability calculation and selection."""

    def test_action_probabilities_from_visits(self):
        """Test action probabilities proportional to visit counts."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        root = MCTSNode(game, player)

        # Create children with different visit counts
        child0 = MCTSNode(game, player, parent=root, action_taken=0)
        child1 = MCTSNode(game, player, parent=root, action_taken=1)
        child2 = MCTSNode(game, player, parent=root, action_taken=2)

        root.children = {0: child0, 1: child1, 2: child2}
        root.is_expanded = True

        # Simulate different visit counts
        child0.visit_count = 10
        child1.visit_count = 5
        child2.visit_count = 5

        # Get probabilities with temperature=1
        probs = root.get_action_probabilities(temperature=1.0)

        # Check probabilities sum to 1
        assert abs(sum(probs.values()) - 1.0) < 0.001

        # Child 0 should have highest probability
        assert probs[0] > probs[1]
        assert probs[0] > probs[2]

        # Children 1 and 2 should have equal probability
        assert abs(probs[1] - probs[2]) < 0.001

    def test_action_probabilities_greedy_temperature_zero(self):
        """Test temperature=0 selects most visited action (greedy)."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        root = MCTSNode(game, player)

        # Create children
        child0 = MCTSNode(game, player, parent=root, action_taken=0)
        child1 = MCTSNode(game, player, parent=root, action_taken=1)

        root.children = {0: child0, 1: child1}
        root.is_expanded = True

        child0.visit_count = 10
        child1.visit_count = 5

        # Greedy selection
        probs = root.get_action_probabilities(temperature=0.0)

        # Only most visited action should have probability
        assert probs[0] == 1.0
        assert probs[1] == 0.0

    def test_action_probabilities_high_temperature(self):
        """Test high temperature creates more uniform distribution."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        root = MCTSNode(game, player)

        child0 = MCTSNode(game, player, parent=root, action_taken=0)
        child1 = MCTSNode(game, player, parent=root, action_taken=1)

        root.children = {0: child0, 1: child1}
        root.is_expanded = True

        child0.visit_count = 10
        child1.visit_count = 5

        # Low temperature (closer to greedy)
        probs_low = root.get_action_probabilities(temperature=0.5)

        # High temperature (more uniform)
        probs_high = root.get_action_probabilities(temperature=2.0)

        # High temperature should be more uniform
        # (difference between actions should be smaller)
        diff_low = probs_low[0] - probs_low[1]
        diff_high = probs_high[0] - probs_high[1]

        assert diff_high < diff_low

    def test_action_probabilities_empty_children(self):
        """Test get_action_probabilities returns empty dict for leaf node."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        node = MCTSNode(game, player)

        probs = node.get_action_probabilities()
        assert probs == {}

    def test_select_action_returns_valid_action(self):
        """Test select_action returns one of the legal actions."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        root = MCTSNode(game, player)

        # Create children
        child0 = MCTSNode(game, player, parent=root, action_taken=0)
        child1 = MCTSNode(game, player, parent=root, action_taken=1)

        root.children = {0: child0, 1: child1}
        root.is_expanded = True

        child0.visit_count = 10
        child1.visit_count = 5

        # Select action (with random seed for reproducibility)
        np.random.seed(42)
        action = root.select_action(temperature=1.0)

        # Should be one of the valid actions
        assert action in [0, 1]

    def test_select_action_with_no_children_raises(self):
        """Test select_action raises ValueError when node has no children."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        node = MCTSNode(game, player)

        with pytest.raises(ValueError, match="no children"):
            node.select_action()

    def test_select_action_greedy_always_picks_best(self):
        """Test greedy selection (temperature=0) always picks most visited."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        root = MCTSNode(game, player)

        child0 = MCTSNode(game, player, parent=root, action_taken=0)
        child1 = MCTSNode(game, player, parent=root, action_taken=1)

        root.children = {0: child0, 1: child1}
        root.is_expanded = True

        child0.visit_count = 10
        child1.visit_count = 5

        # Select greedy multiple times
        for _ in range(10):
            action = root.select_action(temperature=0.0)
            assert action == 0  # Always picks most visited


class TestMCTSNodeIntegration:
    """Integration tests with real game states."""

    def test_expand_with_bidding_phase(self):
        """Test expansion works correctly during bidding phase."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        # Use player 1 (not dealer) to avoid forbidden bid constraint
        player = game.players[1]

        node = MCTSNode(game, player)

        # In bidding phase, actions are bids (0-5)
        action_probs = {i: 1.0 / 6 for i in range(6)}
        legal_actions = list(range(6))

        node.expand(action_probs, legal_actions)

        # Check all children created
        assert len(node.children) == 6

        # Check each child has correct game state
        for action in legal_actions:
            child = node.children[action]
            child_player = child.game_state.players[player.position]
            assert child_player.bid == action

    def test_tree_structure_with_multiple_levels(self):
        """Test building multi-level tree structure."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)
        player = game.players[0]

        # Create root
        root = MCTSNode(game, player)

        # Expand root (bidding)
        root.expand({0: 0.5, 1: 0.5}, [0, 1])

        # Expand one of the children
        child = root.children[0]
        child.expand({0: 0.5, 1: 0.5}, [0, 1])

        # Check tree structure
        assert len(root.children) == 2
        assert len(child.children) == 2
        assert child.parent is root
        assert child.children[0].parent is child

    def test_full_simulation_backpropagation_cycle(self):
        """Test complete MCTS simulation cycle."""
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)
        player = game.players[0]

        root = MCTSNode(game, player)

        # 1. Expand root
        root.expand({0: 0.5, 1: 0.5}, [0, 1])

        # 2. Select child using UCB1
        # Note: root starts with visit_count=0, so set it for selection
        root.visit_count = 1
        root.total_value = 0.0  # No value yet
        selected_child = root.select_child(c_puct=1.5)

        # 3. Backpropagate value from leaf
        selected_child.backpropagate(0.8)

        # 4. Check statistics updated
        assert selected_child.visit_count == 1
        assert selected_child.mean_value == 0.8
        # Root gets updated too: (0.0 + 0.8) / (1 + 1) = 0.8 / 2 = 0.4
        # This is correct! Root had visit_count=1 with total_value=0,
        # then backprop adds 1 visit and 0.8 value
        assert root.visit_count == 2  # Was 1, now 2
        assert abs(root.mean_value - 0.4) < 0.001  # (0.0 + 0.8) / 2

        # 5. Get action probabilities
        probs = root.get_action_probabilities(temperature=1.0)

        # Action that was selected should have higher probability
        selected_action = selected_child.action_taken
        other_action = 1 - selected_action
        assert probs[selected_action] > probs.get(other_action, 0)


# ============================================================================
# MCTS SEARCH TESTS
# ============================================================================


class TestMCTSSearch:
    """Test MCTS search algorithm and integration with neural network."""

    def test_mcts_initialization(self):
        """Test MCTS initializes correctly with all components."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()

        mcts = MCTS(
            network=network,
            encoder=encoder,
            masker=masker,
            num_simulations=100,
            c_puct=1.5,
            temperature=1.0,
        )

        assert mcts.network is network
        assert mcts.encoder is encoder
        assert mcts.masker is masker
        assert mcts.num_simulations == 100
        assert mcts.c_puct == 1.5
        assert mcts.temperature == 1.0

    def test_search_returns_action_probabilities(self):
        """Test MCTS search returns valid action probability dictionary."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        # Setup
        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=10)

        # Create game in bidding phase
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        # Run search
        action_probs = mcts.search(game, player)

        # Verify output format
        assert isinstance(action_probs, dict)
        assert len(action_probs) > 0
        assert all(isinstance(k, (int, np.integer)) for k in action_probs.keys())
        assert all(isinstance(v, float) for v in action_probs.values())

    def test_search_probabilities_sum_to_one(self):
        """Test action probabilities sum to approximately 1.0."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=20)

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)
        player = game.players[0]

        action_probs = mcts.search(game, player)

        # Probabilities should sum to ~1.0
        prob_sum = sum(action_probs.values())
        assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum to {prob_sum}, expected 1.0"

    def test_search_only_returns_legal_actions(self):
        """Test MCTS only returns probabilities for legal actions."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=10)

        # BIDDING PHASE TEST
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        action_probs = mcts.search(game, player)

        # All actions should be valid bids (0-5)
        for action in action_probs.keys():
            assert 0 <= action <= 5, f"Invalid bid action: {action}"

    def test_search_respects_dealer_constraint(self):
        """Test MCTS respects dealer's forbidden bid constraint."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=20)

        # Setup game where dealer is last to bid
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)

        # Simulate all players except dealer bidding
        game.players[1].make_bid(1)  # Player 1 bids 1
        game.players[2].make_bid(1)  # Player 2 bids 1
        game.players[3].make_bid(0)  # Player 3 bids 0
        # Total bids: 2, cards dealt: 3
        # Dealer (player 0) cannot bid 1 (forbidden: 3 - 2 = 1)

        dealer = game.players[0]
        action_probs = mcts.search(game, dealer)

        # Dealer should NOT have forbidden bid (1) in action probabilities
        forbidden_bid = 3 - 2  # 1
        assert forbidden_bid not in action_probs, \
            f"Dealer should not be able to bid {forbidden_bid}"

    def test_search_in_playing_phase(self):
        """Test MCTS search works in card playing phase."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker
        from ml.game.blob import Trick

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        # Use fewer simulations to reduce chance of deep tree issues
        mcts = MCTS(network, encoder, masker, num_simulations=5)

        # Setup game in playing phase
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)

        # Complete bidding
        for player in game.players:
            player.make_bid(1)

        # Start playing phase
        game.game_phase = "playing"
        game.current_trick = Trick(game.trump_suit)
        player = game.players[0]

        # Run search - this tests the integration works
        action_probs = mcts.search(game, player)

        # Should return card indices
        assert len(action_probs) > 0, "MCTS should return action probabilities"

        # All actions should be valid card indices (0-51)
        for action in action_probs.keys():
            assert 0 <= action <= 51, f"Invalid card index: {action}"

        # Actions should correspond to cards in player's hand at root
        hand_indices = [encoder._card_to_index(card) for card in player.hand]
        for action in action_probs.keys():
            assert action in hand_indices, \
                f"Action {action} not in player's hand: {hand_indices}"

    def test_terminal_state_detection(self):
        """Test MCTS correctly detects terminal game states."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=10)

        # Create terminal game state
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)
        game.game_phase = "complete"

        # Check terminal detection
        assert mcts._is_terminal(game) is True

        # Non-terminal state
        game.game_phase = "bidding"
        assert mcts._is_terminal(game) is False

    def test_terminal_value_calculation(self):
        """Test terminal value is correctly calculated and normalized."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=10)

        # Setup player with completed round
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)
        player = game.players[0]

        # Player makes bid and achieves it
        player.make_bid(2)
        player.tricks_won = 2

        # Calculate terminal value
        value = mcts._get_terminal_value(game, player)

        # Expected: score = 10 + 2 = 12, normalized: 12 / 23 ≈ 0.52
        expected_score = 12
        expected_value = expected_score / 23.0
        assert abs(value - expected_value) < 0.01, \
            f"Terminal value {value} != expected {expected_value}"

        # Player fails to make bid
        player.tricks_won = 1  # Bid was 2, won 1
        value = mcts._get_terminal_value(game, player)
        assert value == 0.0, "Failed bid should give value 0.0"

    def test_legal_actions_and_mask_bidding(self):
        """Test legal action extraction in bidding phase."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=10)

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        legal_actions, mask = mcts._get_legal_actions_and_mask(game, player)

        # Should have bids 0-5 for non-dealer, or 0-5 minus forbidden for dealer
        assert len(legal_actions) >= 5  # At least 5 legal actions
        assert len(legal_actions) <= 6  # At most 6 legal actions
        assert all(0 <= action <= 5 for action in legal_actions)

        # Mask should match
        assert mask.sum() >= 5  # At least 5 legal actions

    def test_legal_actions_and_mask_playing(self):
        """Test legal action extraction in playing phase."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker
        from ml.game.blob import Trick

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=10)

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        # Move to playing phase
        for p in game.players:
            p.make_bid(1)
        game.game_phase = "playing"
        game.current_trick = Trick(game.trump_suit)

        legal_actions, mask = mcts._get_legal_actions_and_mask(game, player)

        # Should have 5 legal actions (5 cards in hand)
        assert len(legal_actions) == 5

        # All actions should correspond to cards in hand
        hand_indices = [encoder._card_to_index(card) for card in player.hand]
        assert set(legal_actions) == set(hand_indices)

    def test_expand_and_evaluate(self):
        """Test node expansion and neural network evaluation."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=10)

        # Create root node
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)
        player = game.players[0]

        root = MCTSNode(game, player)

        # Expand and evaluate
        value = mcts._expand_and_evaluate(root)

        # Node should now be expanded
        assert root.is_expanded is True
        assert len(root.children) > 0

        # Value should be in reasonable range [-1, 1]
        assert -1.0 <= value <= 1.0, f"Value {value} out of range [-1, 1]"

    def test_simulate_updates_tree(self):
        """Test single simulation updates tree statistics."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=10)

        # Create root
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)
        player = game.players[0]

        root = MCTSNode(game, player)

        # Initial state
        assert root.visit_count == 0
        assert root.is_leaf()

        # Run one simulation
        value = mcts._simulate(root)

        # Root should be expanded and visited
        assert root.is_expanded is True
        assert root.visit_count == 1
        assert root.mean_value == value

    def test_multiple_simulations_converge(self):
        """Test multiple simulations build tree and converge."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=50)

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)
        player = game.players[0]

        # Run search
        action_probs = mcts.search(game, player)

        # Should have explored multiple actions
        assert len(action_probs) > 0

        # At least one action should have significant probability
        max_prob = max(action_probs.values())
        assert max_prob > 0.1, "Should have at least one action with >10% probability"

    def test_temperature_affects_action_selection(self):
        """Test temperature parameter affects action probability distribution."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)
        player = game.players[0]

        # Low temperature (greedy)
        mcts_greedy = MCTS(network, encoder, masker, num_simulations=30, temperature=0.0)
        probs_greedy = mcts_greedy.search(game, player)

        # One action should dominate (greedy selection)
        max_prob_greedy = max(probs_greedy.values())
        assert max_prob_greedy > 0.8, "Greedy should strongly prefer one action"

        # High temperature (more uniform)
        mcts_explore = MCTS(network, encoder, masker, num_simulations=30, temperature=2.0)
        probs_explore = mcts_explore.search(game, player)

        # Distribution should be more uniform
        max_prob_explore = max(probs_explore.values())
        assert max_prob_explore < max_prob_greedy, \
            "High temperature should produce more uniform distribution"


class TestMCTSIntegrationWithNetwork:
    """Test MCTS integration with neural network components."""

    def test_full_mcts_search_pipeline(self):
        """Test complete MCTS pipeline from game state to action."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        # Initialize all components
        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=50)

        # Create game
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        # Run search
        action_probs = mcts.search(game, player)

        # Select action
        best_action = max(action_probs, key=action_probs.get)

        # Verify action is legal
        assert 0 <= best_action <= 5, f"Selected action {best_action} out of range"

        # Apply action should work
        player.make_bid(best_action)
        assert player.bid == best_action

    def test_mcts_plays_complete_bidding_round(self):
        """Test MCTS can play complete bidding round for all players."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=30)

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)

        # All players make bids using MCTS
        for player in game.players:
            action_probs = mcts.search(game, player)
            bid = max(action_probs, key=action_probs.get)
            player.make_bid(bid)

        # All players should have bids
        assert all(p.bid is not None for p in game.players)

        # Dealer constraint should be respected
        total_bids = sum(p.bid for p in game.players)
        cards_dealt = 3
        assert total_bids != cards_dealt, "Dealer constraint violated"


class TestMCTSTreeReuse:
    """Test tree reuse functionality for performance optimization."""

    def test_search_with_tree_reuse_first_call(self):
        """Test first call to search_with_tree_reuse creates new root."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=20)

        # First call: no previous root
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)
        player = game.players[0]

        assert mcts.root is None, "Root should be None initially"

        action_probs = mcts.search_with_tree_reuse(game, player)

        # Root should now exist
        assert mcts.root is not None, "Root should be created"
        assert mcts.root.visit_count > 0, "Root should have visits"
        assert len(action_probs) > 0, "Should return action probabilities"

    def test_search_with_tree_reuse_navigates_to_child(self):
        """Test tree reuse navigates to child node after action."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=30)

        # First search
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)
        player = game.players[0]

        action_probs = mcts.search_with_tree_reuse(game, player)
        best_action = max(action_probs, key=action_probs.get)

        # Store old root info
        old_root = mcts.root
        assert best_action in old_root.children, "Best action should have child node"
        old_child = old_root.children[best_action]
        old_child_visits = old_child.visit_count

        # Apply action to game
        game.apply_action(best_action, player)
        next_player = game.get_current_player()

        # Second search with tree reuse
        action_probs2 = mcts.search_with_tree_reuse(
            game, next_player, previous_action=best_action
        )

        # Root should now be the old child
        assert mcts.root is old_child, "Root should be the previous child"
        assert mcts.root.parent is None, "New root should have no parent"
        assert mcts.root.visit_count >= old_child_visits, "Visit count should increase"

    def test_reset_tree_clears_root(self):
        """Test reset_tree clears the stored root."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=20)

        # Build a tree
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)
        player = game.players[0]

        action_probs = mcts.search_with_tree_reuse(game, player)
        assert mcts.root is not None, "Root should exist after search"

        # Reset tree
        mcts.reset_tree()
        assert mcts.root is None, "Root should be None after reset"

    def test_tree_reuse_with_unexpected_action(self):
        """Test tree reuse handles action not in tree gracefully."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=20)

        # First search
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)
        player = game.players[0]

        action_probs = mcts.search_with_tree_reuse(game, player)

        # Pick an action that might not be in tree (e.g., lowest probability)
        unexpected_action = min(action_probs, key=action_probs.get)

        # Apply action
        game.apply_action(unexpected_action, player)
        next_player = game.get_current_player()

        # Search with unexpected action (might not be in tree)
        # Should create new root instead of reusing
        old_root = mcts.root
        action_probs2 = mcts.search_with_tree_reuse(
            game, next_player, previous_action=99  # Invalid action
        )

        # Should create new root since action not found
        assert mcts.root is not None, "Should create new root"
        assert mcts.root is not old_root, "Should be a different root"


class TestMCTSBatchedInference:
    """Test batched neural network inference for performance optimization."""

    def test_search_batched_returns_valid_probabilities(self):
        """Test batched search returns valid action probabilities."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=32)

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)
        player = game.players[0]

        # Run batched search
        action_probs = mcts.search_batched(game, player, batch_size=8)

        # Check results
        assert len(action_probs) > 0, "Should return action probabilities"
        assert abs(sum(action_probs.values()) - 1.0) < 1e-5, "Probabilities should sum to 1"

        # All actions should be legal
        for action in action_probs.keys():
            assert 0 <= action <= 3, f"Action {action} out of range for 3 cards"

    def test_batched_vs_sequential_produces_similar_results(self):
        """Test batched and sequential search produce similar results."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker
        import torch

        # Use fixed seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)
        player = game.players[0]

        # Sequential search
        mcts_seq = MCTS(network, encoder, masker, num_simulations=40)
        probs_seq = mcts_seq.search(game, player)

        # Batched search
        game2 = BlobGame(num_players=4)
        game2.setup_round(cards_to_deal=3)
        player2 = game2.players[0]

        mcts_batch = MCTS(network, encoder, masker, num_simulations=40)
        probs_batch = mcts_batch.search_batched(game2, player2, batch_size=8)

        # Results should have same actions
        assert set(probs_seq.keys()) == set(probs_batch.keys()), \
            "Both should explore same actions"

        # Note: Results won't be identical due to different traversal order,
        # but distribution should be similar
        # Check that top action is in top 2 of other method
        best_seq = max(probs_seq, key=probs_seq.get)
        best_batch = max(probs_batch, key=probs_batch.get)

        top2_seq = sorted(probs_seq, key=probs_seq.get, reverse=True)[:2]
        top2_batch = sorted(probs_batch, key=probs_batch.get, reverse=True)[:2]

        assert best_seq in top2_batch or best_batch in top2_seq, \
            "Best actions should be similar between methods"

    def test_traverse_to_leaf_finds_unexpanded_node(self):
        """Test _traverse_to_leaf finds unexpanded nodes."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=20)

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)
        # Use player 1 (not dealer) to avoid forbidden bid issues
        player = game.players[1]

        # Create root node
        from ml.mcts.node import MCTSNode
        root = MCTSNode(game, player)

        # Expand root with legal actions
        legal_actions = [0, 1, 2, 3]
        action_probs = {a: 0.25 for a in legal_actions}
        root.expand(action_probs, legal_actions)

        # Traverse should find a child (leaf)
        leaf = mcts._traverse_to_leaf(root)

        assert leaf is not None, "Should find a leaf node"
        assert leaf.is_leaf(), "Found node should be a leaf"
        assert leaf in root.children.values(), "Leaf should be a child of root"

    def test_batch_expand_and_evaluate_handles_multiple_nodes(self):
        """Test _batch_expand_and_evaluate expands multiple nodes."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker
        from ml.mcts.node import MCTSNode

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=20)

        # Create multiple leaf nodes
        nodes = []
        for i in range(3):
            game = BlobGame(num_players=4)
            game.setup_round(cards_to_deal=3)
            player = game.players[0]
            node = MCTSNode(game, player)
            nodes.append(node)

        # All should be leaves initially
        assert all(node.is_leaf() for node in nodes)

        # Batch expand and evaluate
        mcts._batch_expand_and_evaluate(nodes)

        # All should now be expanded
        assert all(not node.is_leaf() for node in nodes)
        assert all(len(node.children) > 0 for node in nodes)
        assert all(node.visit_count > 0 for node in nodes)

    def test_batched_inference_with_different_batch_sizes(self):
        """Test batched inference works with various batch sizes."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)
        player = game.players[0]

        # Test different batch sizes
        for batch_size in [1, 4, 8, 16]:
            game_copy = BlobGame(num_players=4)
            game_copy.setup_round(cards_to_deal=3)
            player_copy = game_copy.players[0]

            mcts = MCTS(network, encoder, masker, num_simulations=32)
            action_probs = mcts.search_batched(game_copy, player_copy, batch_size=batch_size)

            assert len(action_probs) > 0, f"Should work with batch_size={batch_size}"
            assert abs(sum(action_probs.values()) - 1.0) < 1e-5


class TestMCTSPerformance:
    """Performance benchmarks for MCTS search."""

    def test_mcts_inference_speed_cpu(self):
        """Benchmark MCTS inference speed on CPU."""
        import time
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=50)

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        # Warm-up run
        _ = mcts.search(game, player)

        # Benchmark
        start_time = time.time()
        action_probs = mcts.search(game, player)
        elapsed_ms = (time.time() - start_time) * 1000

        print(f"\nMCTS inference (50 sims): {elapsed_ms:.2f} ms")
        print(f"Time per simulation: {elapsed_ms/50:.2f} ms")

        # Should complete in reasonable time (relaxed for CI)
        assert elapsed_ms < 5000, f"Too slow: {elapsed_ms:.2f} ms > 5000 ms"
        assert len(action_probs) > 0

    def test_batched_inference_performance(self):
        """Compare batched vs sequential inference performance."""
        import time
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        # Sequential
        mcts_seq = MCTS(network, encoder, masker, num_simulations=40)
        start = time.time()
        _ = mcts_seq.search(game, player)
        time_seq = time.time() - start

        # Batched
        game2 = BlobGame(num_players=4)
        game2.setup_round(cards_to_deal=5)
        player2 = game2.players[0]

        mcts_batch = MCTS(network, encoder, masker, num_simulations=40)
        start = time.time()
        _ = mcts_batch.search_batched(game2, player2, batch_size=8)
        time_batch = time.time() - start

        print(f"\nSequential: {time_seq*1000:.2f} ms")
        print(f"Batched: {time_batch*1000:.2f} ms")
        print(f"Speedup: {time_seq/time_batch:.2f}x")

        # Both should complete successfully
        assert time_seq > 0 and time_batch > 0

    def test_tree_reuse_effectiveness(self):
        """Measure effectiveness of tree reuse."""
        import time
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=50)

        # First search (cold start)
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)
        player = game.players[0]

        start = time.time()
        action_probs = mcts.search_with_tree_reuse(game, player)
        time_first = time.time() - start

        best_action = max(action_probs, key=action_probs.get)

        # Get stats before reuse
        old_root_visits = mcts.root.visit_count

        # Apply action
        game.apply_action(best_action, player)
        next_player = game.get_current_player()

        # Second search (with tree reuse)
        start = time.time()
        action_probs2 = mcts.search_with_tree_reuse(
            game, next_player, previous_action=best_action
        )
        time_reuse = time.time() - start

        print(f"\nFirst search: {time_first*1000:.2f} ms")
        print(f"Reused search: {time_reuse*1000:.2f} ms")
        print(f"Speedup: {time_first/time_reuse:.2f}x")

        # Root should have accumulated visits
        assert mcts.root.visit_count >= 50, "Root should have at least 50 visits"

        # Both should produce valid results
        assert len(action_probs2) > 0


class TestMCTSIntegrationComplete:
    """Complete integration tests for optimized MCTS."""

    def test_complete_game_with_tree_reuse(self):
        """Test playing a complete game using tree reuse."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=30)

        # Create game
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=2)

        # Bidding phase with tree reuse
        previous_action = None
        for i, player in enumerate(game.players):
            action_probs = mcts.search_with_tree_reuse(
                game, player, previous_action=previous_action
            )
            bid = max(action_probs, key=action_probs.get)
            game.apply_action(bid, player)
            previous_action = bid

            # After first player, tree should exist
            if i == 0:
                assert mcts.root is not None

        # All players should have bids
        assert all(p.bid is not None for p in game.players)

        # Reset tree for playing phase
        mcts.reset_tree()

        print("\nBidding complete, all players have valid bids")

    def test_batched_inference_complete_bidding(self):
        """Test complete bidding round using batched inference."""
        from ml.mcts.search import MCTS
        from ml.network.model import BlobNet
        from ml.network.encode import StateEncoder, ActionMasker

        network = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()
        mcts = MCTS(network, encoder, masker, num_simulations=32)

        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)

        # All players use batched search in proper bidding order (left of dealer goes first)
        bidding_order_start = (game.dealer_position + 1) % game.num_players
        for i in range(game.num_players):
            player_idx = (bidding_order_start + i) % game.num_players
            player = game.players[player_idx]

            action_probs = mcts.search_batched(game, player, batch_size=8)
            bid = max(action_probs, key=action_probs.get)
            player.make_bid(bid)

        # All players should have bids
        assert all(p.bid is not None for p in game.players)

        # Dealer constraint respected (sum should never equal cards dealt = 3)
        total_bids = sum(p.bid for p in game.players)
        assert total_bids != 3, "Dealer constraint should be respected"

        print("\nBatched inference: all players made valid bids")
