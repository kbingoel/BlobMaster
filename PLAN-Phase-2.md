# PLAN-Phase-2.md
# Implementation Plan: MCTS + Neural Network

**Phase**: 2 - MCTS + Neural Network
**Timeline**: Midweek (9 sessions × 2 hours = 18 hours)
**Goal**: Basic AI that can play legal moves and improve with training

---

## Overview

Build the core AI infrastructure for BlobMaster by implementing:
1. **State Encoding**: Convert complex game state into neural network input tensors
2. **Neural Network**: Lightweight Transformer that predicts action probabilities and game value
3. **MCTS**: Monte Carlo Tree Search for lookahead planning using the neural network
4. **Integration**: Connect all components and validate performance

This phase builds on Phase 1's game engine to create an AI agent capable of playing Blob through self-play.

---

## File Structure to Create

```
ml/
├── network/
│   ├── __init__.py          # Package initialization
│   ├── encode.py            # State → tensor encoding (SESSION 1-2)
│   ├── model.py             # Transformer architecture (SESSION 3-5)
│   └── test_network.py      # Network tests
│
├── mcts/
│   ├── __init__.py          # Package initialization
│   ├── node.py              # MCTS tree node (SESSION 6)
│   ├── search.py            # MCTS algorithm (SESSION 7-8)
│   └── test_mcts.py         # MCTS tests
│
└── requirements.txt         # Add PyTorch and dependencies
```

---

## Prerequisites

Before starting Phase 2, ensure Phase 1 is complete:
- ✅ `ml/game/blob.py` fully implemented and tested (COMPLETE - Phase 1a.10)
- ✅ All unit tests passing - 135 tests, 97% coverage (COMPLETE - Phase 1a.9)
- ⚠️ CLI version playable (SKIPPED - not needed, use `play_round()` with callbacks)
- ✅ Game state queries working: `get_game_state()`, `get_legal_actions()` (COMPLETE)
- ✅ **BONUS**: `copy()` and `apply_action()` methods ready for MCTS (Phase 1a.10)

---

## Detailed Session Breakdown

### SESSION 1: State Encoding - Basic Structure (2 hours)

**Goal**: Encode game state into tensor representation suitable for neural network input.

#### 1.1 Setup (15 min)
- [ ] Create `ml/network/` directory
- [ ] Create `ml/network/__init__.py`
- [ ] Create `ml/network/encode.py`
- [ ] Update `ml/requirements.txt`:
```txt
torch>=2.0.0
numpy>=1.24.0
pytest>=7.3.0
```
- [ ] Install dependencies: `pip install -r ml/requirements.txt`

#### 1.2 Design State Vector (15 min)

Document the encoding scheme (**256 dimensions - optimized for CPU inference**):
```python
"""
State Encoding Dimensions (256 total):

1. My Hand (52-dim binary):
   - One-hot encoding: 1 if I have this card, 0 otherwise
   - Ordered by suit (♠♥♣♦) then rank (2-A)

2. Cards Played This Trick (52-dim sequential):
   - 0 if not played
   - 1-8 for play order (which player position played it)

3. All Cards Played This Round (52-dim binary):
   - 1 if card has been played in any trick this round, 0 otherwise

4. Player Bids (8-dim, padded):
   - Normalized bid value for each player position
   - -1 for absent players or players who haven't bid yet

5. Player Tricks Won (8-dim, padded):
   - Normalized tricks won for each player position
   - 0 for absent players

6. My Bid (1-dim scalar):
   - Normalized bid value (-1 if not yet bid)

7. My Tricks Won (1-dim scalar):
   - Normalized tricks won

8. Round Metadata (8-dim):
   - Cards dealt this round (normalized)
   - Current trick number (normalized)
   - My position relative to dealer (normalized)
   - Number of active players (normalized)
   - Trump suit (one-hot: 4-dim for ♠♥♣♦, all zeros for None)
   - Am I the dealer? (binary)

9. Bidding Constraint (1-dim):
   - Is forbidden bid calculation active for me? (binary)

10. Game Phase (3-dim one-hot):
    - [bidding, playing_trick, scoring]

11. Positional Encoding (16-dim):
    - Additional features for position awareness
    - Relative position to current lead player
    - Cards remaining in hand
    - Rounds completed
    - etc.

Total: 52 + 52 + 52 + 8 + 8 + 1 + 1 + 8 + 1 + 3 + 16 = 202 base dimensions
Padded to 256 with zeros for future extensibility (54 dimensions spare)

Architecture Decision: Why 256 vs 512?
---------------------------------------
- 202 required dimensions + 54 spare = 26% overhead (vs 60% with 512)
- 2x less memory per state vector (1KB vs 2KB)
- 4x fewer parameters in embedding layer (65K vs 262K)
- Faster CPU inference on Intel laptop (deployment target)
- Better cache utilization (fits in L1/L2 cache)
- Still power-of-2 for hardware memory alignment benefits
- Results in ~60% smaller model overall (2-3M vs 5-8M parameters)
"""
```

#### 1.3 Implement StateEncoder Class (60 min)

```python
# ml/network/encode.py

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from ml.game.blob import BlobGame, Player, Card
from ml.game.constants import SUITS, RANKS, RANK_VALUES, MAX_PLAYERS


class StateEncoder:
    """
    Encodes game state into tensor representation for neural network.

    Output: torch.Tensor of shape (512,) containing:
        - Card representations (one-hot, sequential)
        - Player state (bids, tricks, positions)
        - Game metadata (trump, phase, constraints)
        - Positional encoding
    """

    def __init__(self):
        self.state_dim = 256  # Optimized for CPU inference
        self.card_dim = 52
        self.max_players = MAX_PLAYERS

    def encode(self, game: BlobGame, player: Player) -> torch.Tensor:
        """
        Encode current game state from perspective of given player.

        Args:
            game: BlobGame instance
            player: Player whose perspective to encode

        Returns:
            torch.Tensor of shape (256,) with normalized state features
        """
        # Initialize full state vector
        state = torch.zeros(self.state_dim, dtype=torch.float32)

        offset = 0

        # 1. My Hand (52-dim binary)
        hand_vector = self._encode_hand(player.hand)
        state[offset:offset+52] = hand_vector
        offset += 52

        # 2. Cards Played This Trick (52-dim sequential)
        trick_vector = self._encode_current_trick(game)
        state[offset:offset+52] = trick_vector
        offset += 52

        # 3. All Cards Played This Round (52-dim binary)
        played_vector = self._encode_cards_played(game)
        state[offset:offset+52] = played_vector
        offset += 52

        # 4. Player Bids (8-dim)
        bids_vector = self._encode_player_bids(game)
        state[offset:offset+8] = bids_vector
        offset += 8

        # 5. Player Tricks Won (8-dim)
        tricks_vector = self._encode_player_tricks(game)
        state[offset:offset+8] = tricks_vector
        offset += 8

        # 6. My Bid (1-dim)
        state[offset] = self._normalize_bid(player.bid, game)
        offset += 1

        # 7. My Tricks Won (1-dim)
        state[offset] = self._normalize_tricks(player.tricks_won, game)
        offset += 1

        # 8. Round Metadata (8-dim)
        metadata_vector = self._encode_metadata(game, player)
        state[offset:offset+8] = metadata_vector
        offset += 8

        # 9. Bidding Constraint (1-dim)
        state[offset] = self._encode_bidding_constraint(game, player)
        offset += 1

        # 10. Game Phase (3-dim one-hot)
        phase_vector = self._encode_game_phase(game)
        state[offset:offset+3] = phase_vector
        offset += 3

        # 11. Positional Encoding (16-dim)
        pos_vector = self._encode_positional_features(game, player)
        state[offset:offset+16] = pos_vector
        offset += 16

        # Remaining dimensions padded with zeros (for future features)
        # offset should be 202, rest is zero-padded to 256 (54 dims spare)

        return state

    def _encode_hand(self, hand: List[Card]) -> torch.Tensor:
        """Encode player's hand as 52-dim binary vector."""
        # Implementation details...
        pass

    def _encode_current_trick(self, game: BlobGame) -> torch.Tensor:
        """Encode cards played in current trick with play order."""
        # Implementation details...
        pass

    def _encode_cards_played(self, game: BlobGame) -> torch.Tensor:
        """Encode all cards played this round as binary vector."""
        # Implementation details...
        pass

    # ... [implement remaining helper methods]

    def _card_to_index(self, card: Card) -> int:
        """
        Convert card to index (0-51).

        Ordering: ♠2-♠A (0-12), ♥2-♥A (13-25), ♣2-♣A (26-38), ♦2-♦A (39-51)
        """
        suit_idx = SUITS.index(card.suit)
        rank_idx = RANKS.index(card.rank)
        return suit_idx * 13 + rank_idx
```

#### 1.4 Basic Tests (30 min)
- [ ] Create `ml/network/test_network.py`
- [ ] Test `_card_to_index()` mapping
- [ ] Test hand encoding (verify correct cards marked)
- [ ] Test state shape (should be 256-dim)
- [ ] Test with simple game state

**Deliverable**: Basic StateEncoder that produces 256-dim tensors

---

### SESSION 2: State Encoding - Complete Implementation (2 hours)

**Goal**: Finish all encoding methods and comprehensive testing.

#### 2.1 Implement Remaining Encoders (60 min)

Complete all helper methods:
- [ ] `_encode_hand()` - Binary one-hot for cards in hand
- [ ] `_encode_current_trick()` - Sequential play order (1-8)
- [ ] `_encode_cards_played()` - Binary for all played cards
- [ ] `_encode_player_bids()` - Normalized bids with -1 padding
- [ ] `_encode_player_tricks()` - Normalized tricks with 0 padding
- [ ] `_normalize_bid()` - Scale bid to [0, 1] or -1 if not bid
- [ ] `_normalize_tricks()` - Scale tricks to [0, 1]
- [ ] `_encode_metadata()` - Round info, trump, position, etc.
- [ ] `_encode_bidding_constraint()` - Am I dealer with constraint?
- [ ] `_encode_game_phase()` - One-hot for [bidding, playing, scoring]
- [ ] `_encode_positional_features()` - Position-aware features

**Key Implementation Details**:
```python
def _normalize_bid(self, bid: Optional[int], game: BlobGame) -> float:
    """Normalize bid to [0, 1] range, -1 if not yet bid."""
    if bid is None:
        return -1.0
    # Normalize by max possible bid (cards dealt this round)
    max_bid = len(game.players[0].hand) + len(game.cards_played_this_round) // len(game.players)
    return bid / max_bid if max_bid > 0 else 0.0

def _encode_metadata(self, game: BlobGame, player: Player) -> torch.Tensor:
    """
    Encode round metadata (8-dim):
    [cards_dealt, trick_num, my_position, num_players, trump_one_hot×4, is_dealer]
    """
    metadata = torch.zeros(8)

    # Cards dealt this round (normalized by 13, max per player)
    cards_dealt = len(player.hand) + len(player.cards_played)
    metadata[0] = cards_dealt / 13.0

    # Current trick number (normalized by cards dealt)
    trick_num = len(game.tricks_history)
    metadata[1] = trick_num / cards_dealt if cards_dealt > 0 else 0

    # My position relative to dealer (normalized by num_players)
    my_pos = (player.position - game.dealer_position) % len(game.players)
    metadata[2] = my_pos / len(game.players)

    # Number of active players (normalized by MAX_PLAYERS)
    metadata[3] = len(game.players) / MAX_PLAYERS

    # Trump suit one-hot (4-dim): ♠♥♣♦
    # Note: metadata[4:8] reserved for trump, but only use 4 bits
    # We'll handle None (no-trump) by all zeros
    if game.trump_suit is not None:
        trump_idx = SUITS.index(game.trump_suit)
        metadata[4 + trump_idx] = 1.0

    return metadata
```

#### 2.2 Variable Player Count Handling (30 min)

Ensure encoding handles 3-8 players correctly:
- [ ] Test with 3 players
- [ ] Test with 8 players
- [ ] Verify padding works (unused player slots = -1 or 0)
- [ ] Test position normalization

#### 2.3 Comprehensive Testing (30 min)

Test suite for state encoding:
```python
class TestStateEncoder:
    def test_encoder_initialization(self):
        """Test StateEncoder creates correct dimensions."""

    def test_encode_full_game_state(self):
        """Test encoding complete game state."""

    def test_hand_encoding(self):
        """Verify hand cards are correctly marked in tensor."""

    def test_trick_encoding_order(self):
        """Verify trick play order is sequential."""

    def test_played_cards_tracking(self):
        """Verify all played cards marked correctly."""

    def test_bid_normalization(self):
        """Test bid values normalized to [0,1] or -1."""

    def test_trump_encoding(self):
        """Test trump suit one-hot encoding."""

    def test_no_trump_encoding(self):
        """Test no-trump rounds (all zeros)."""

    def test_dealer_constraint_detection(self):
        """Test bidding constraint flag for dealer."""

    def test_variable_player_counts(self):
        """Test encoding works for 3-8 players."""

    def test_state_determinism(self):
        """Same game state should produce identical tensor."""

    def test_state_shape_always_256(self):
        """State tensor always 256-dim regardless of game config."""
```

**Deliverable**: Fully functional StateEncoder with >95% test coverage

---

### SESSION 3: Neural Network - Basic Transformer (2 hours)

**Goal**: Implement basic Transformer architecture with embedding layer.

#### 3.1 Network Architecture Planning (15 min)

Design document:
```python
"""
BlobNet: Lightweight Transformer for Blob Card Game

Architecture:
    Input (256) → Embedding (256) → Transformer Layers (4-6) → Dual Heads

    Dual Heads:
    1. Policy Head:
       - Bidding phase: Softmax over valid bids [0, cards_dealt]
       - Playing phase: Softmax over cards in hand
       - Legal action masking applied before output

    2. Value Head:
       - Single scalar: Expected final score (normalized [-1, 1])
       - Predicts how well current player will do

Hyperparameters (Optimized for CPU Inference):
    - State dim: 256 (reduced from 512)
    - Embedding dim: 256 (reduced from 512)
    - Transformer layers: 6
    - Attention heads: 8
    - Feedforward dim: 1024 (reduced from 2048)
    - Dropout: 0.1

Total Parameters: ~2-3M (optimized for laptop CPU inference)

Performance Benefits:
    - 4x fewer parameters in embedding layer (65K vs 262K)
    - 2x smaller feedforward layers (1024 vs 2048)
    - ~60% smaller model overall (2-3M vs 5-8M parameters)
    - Faster inference on Intel i5-1135G7 iGPU
    - Better memory efficiency for 8GB RTX 4060 training
"""
```

#### 3.2 Implement BlobNet Class (75 min)

```python
# ml/network/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BlobNet(nn.Module):
    """
    Transformer-based neural network for Blob card game.

    Dual-head architecture:
    - Policy head: Action probabilities (bids or card plays)
    - Value head: Expected score prediction
    """

    def __init__(
        self,
        state_dim: int = 256,  # Optimized for CPU inference
        embedding_dim: int = 256,  # Optimized for CPU inference
        num_layers: int = 6,
        num_heads: int = 8,
        feedforward_dim: int = 1024,  # Reduced from 2048
        dropout: float = 0.1,
        max_bid: int = 13,  # Max cards per player
        max_cards: int = 52,  # Full deck
    ):
        super(BlobNet, self).__init__()

        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.max_bid = max_bid
        self.max_cards = max_cards

        # Input embedding
        self.input_embedding = nn.Linear(state_dim, embedding_dim)
        self.input_norm = nn.LayerNorm(embedding_dim)

        # Positional encoding (learned)
        self.positional_encoding = nn.Parameter(
            torch.randn(1, 1, embedding_dim)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Policy head (for both bidding and card playing)
        self.policy_fc = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, max(max_bid + 1, max_cards)),  # Max action space
        )

        # Value head
        self.value_fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(
        self,
        state: torch.Tensor,
        legal_actions_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: (batch_size, state_dim) or (state_dim,)
            legal_actions_mask: (batch_size, action_dim) binary mask
                                1 = legal action, 0 = illegal

        Returns:
            policy: (batch_size, action_dim) action probabilities
            value: (batch_size, 1) expected score
        """
        # Handle single state (add batch dimension)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single_input = True
        else:
            single_input = False

        batch_size = state.size(0)

        # Embed input
        x = self.input_embedding(state)  # (batch, embedding_dim)
        x = self.input_norm(x)

        # Add positional encoding
        x = x.unsqueeze(1)  # (batch, 1, embedding_dim)
        x = x + self.positional_encoding

        # Transformer encoding
        x = self.transformer(x)  # (batch, 1, embedding_dim)
        x = x.squeeze(1)  # (batch, embedding_dim)

        # Policy head (action logits)
        policy_logits = self.policy_fc(x)  # (batch, action_dim)

        # Apply legal action masking if provided
        if legal_actions_mask is not None:
            # Set illegal actions to very negative value before softmax
            policy_logits = policy_logits.masked_fill(
                legal_actions_mask == 0,
                float('-inf')
            )

        # Softmax to get probabilities
        policy = F.softmax(policy_logits, dim=-1)

        # Value head
        value = self.value_fc(x)  # (batch, 1)

        # Remove batch dimension if single input
        if single_input:
            policy = policy.squeeze(0)
            value = value.squeeze(0)

        return policy, value
```

#### 3.3 Basic Tests (30 min)

```python
class TestBlobNet:
    def test_network_initialization(self):
        """Test network creates with correct architecture."""

    def test_forward_pass_shape(self):
        """Test output shapes are correct."""

    def test_single_state_inference(self):
        """Test forward pass with single state (no batch)."""

    def test_batch_inference(self):
        """Test forward pass with batched states."""

    def test_value_output_range(self):
        """Test value head outputs in [-1, 1] due to tanh."""

    def test_policy_sums_to_one(self):
        """Test policy probabilities sum to 1."""
```

**Deliverable**: Basic Transformer network that produces policy and value outputs

---

### SESSION 4: Neural Network - Legal Action Masking (2 hours)

**Goal**: Implement legal action masking for bidding and card playing phases.

#### 4.1 Action Space Design (20 min)

Document action representation:
```python
"""
Action Space Encoding:

The policy head outputs a vector that represents BOTH bidding and card playing:
- Size: max(max_bid + 1, 52) = 52 dimensions

Bidding Phase:
    - First 14 dimensions (indices 0-13): Bid values [0, 1, 2, ..., 13]
    - Mask out bids > cards_dealt
    - Mask out dealer's forbidden bid

Playing Phase:
    - All 52 dimensions: Each card in deck
    - Mask out cards not in player's hand
    - Mask out cards already played

Masking Strategy:
    - Create legal_actions_mask tensor before forward pass
    - Network receives mask and applies it before softmax
    - Illegal actions get -inf logits → 0 probability after softmax
"""
```

#### 4.2 Implement ActionMasker Class (60 min)

```python
# ml/network/encode.py (add to existing file)

class ActionMasker:
    """
    Creates legal action masks for bidding and card playing phases.
    """

    def __init__(self, max_bid: int = 13, deck_size: int = 52):
        self.max_bid = max_bid
        self.deck_size = deck_size
        self.action_dim = max(max_bid + 1, deck_size)

    def create_bidding_mask(
        self,
        cards_dealt: int,
        is_dealer: bool,
        forbidden_bid: Optional[int],
    ) -> torch.Tensor:
        """
        Create mask for valid bids.

        Args:
            cards_dealt: Number of cards dealt this round
            is_dealer: Is this player the dealer?
            forbidden_bid: Dealer's forbidden bid (or None)

        Returns:
            torch.Tensor of shape (action_dim,) with 1 for legal, 0 for illegal
        """
        mask = torch.zeros(self.action_dim, dtype=torch.float32)

        # Valid bids: 0 to cards_dealt
        mask[0:cards_dealt + 1] = 1.0

        # Dealer constraint: mask out forbidden bid
        if is_dealer and forbidden_bid is not None:
            if 0 <= forbidden_bid <= cards_dealt:
                mask[forbidden_bid] = 0.0

        # Ensure at least one legal bid exists
        if mask.sum() == 0:
            raise ValueError(f"No legal bids available: cards={cards_dealt}, forbidden={forbidden_bid}")

        return mask

    def create_playing_mask(
        self,
        hand: List[Card],
        led_suit: Optional[str],
        encoder: StateEncoder,
    ) -> torch.Tensor:
        """
        Create mask for valid card plays.

        Args:
            hand: Player's current hand
            led_suit: Suit that was led (or None if first card)
            encoder: StateEncoder to map cards to indices

        Returns:
            torch.Tensor of shape (action_dim,) with 1 for legal, 0 for illegal
        """
        mask = torch.zeros(self.action_dim, dtype=torch.float32)

        # Get legal plays using game logic
        # If led_suit exists and player has that suit: only those cards legal
        # Otherwise: all cards in hand are legal

        legal_cards = hand
        if led_suit is not None:
            cards_in_led_suit = [c for c in hand if c.suit == led_suit]
            if cards_in_led_suit:
                legal_cards = cards_in_led_suit

        # Mark legal cards
        for card in legal_cards:
            card_idx = encoder._card_to_index(card)
            mask[card_idx] = 1.0

        # Ensure at least one legal card
        if mask.sum() == 0:
            raise ValueError(f"No legal cards available from hand: {hand}")

        return mask

    def sample_action(
        self,
        policy: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> int:
        """
        Sample action from policy with temperature.

        Args:
            policy: Action probabilities (already masked)
            mask: Legal action mask
            temperature: Sampling temperature (1.0 = neutral, >1 = more random)

        Returns:
            int: Selected action index
        """
        # Apply temperature scaling
        if temperature != 1.0:
            # Re-compute logits from probabilities, scale, re-normalize
            logits = torch.log(policy + 1e-8)  # Add epsilon for numerical stability
            logits = logits / temperature
            policy = F.softmax(logits, dim=-1)

        # Ensure policy is still masked (should already be, but double-check)
        policy = policy * mask
        policy = policy / policy.sum()  # Re-normalize

        # Sample from distribution
        action = torch.multinomial(policy, num_samples=1).item()

        return action

    def greedy_action(self, policy: torch.Tensor, mask: torch.Tensor) -> int:
        """
        Select highest probability legal action.

        Args:
            policy: Action probabilities (already masked)
            mask: Legal action mask

        Returns:
            int: Action index with highest probability
        """
        # Mask out illegal actions (set to -inf for argmax)
        masked_policy = policy.clone()
        masked_policy[mask == 0] = -float('inf')

        action = torch.argmax(masked_policy).item()

        return action
```

#### 4.3 Integration with BlobNet (20 min)

Update network to use masking:
- [ ] Test forward pass with bidding mask
- [ ] Test forward pass with playing mask
- [ ] Verify illegal actions get 0 probability

#### 4.4 Tests (20 min)

```python
class TestActionMasker:
    def test_bidding_mask_normal_player(self):
        """Test bidding mask for non-dealer."""

    def test_bidding_mask_dealer_constraint(self):
        """Test dealer's forbidden bid is masked out."""

    def test_playing_mask_must_follow_suit(self):
        """Test only cards in led suit are legal."""

    def test_playing_mask_no_led_suit(self):
        """Test all hand cards legal when no led suit."""

    def test_playing_mask_cant_follow_suit(self):
        """Test all hand cards legal when can't follow suit."""

    def test_sample_action_respects_mask(self):
        """Test sampled actions are always legal."""

    def test_greedy_action_selects_best(self):
        """Test greedy selection picks highest legal probability."""
```

**Deliverable**: Legal action masking fully integrated with neural network

---

### SESSION 5: Neural Network - Training Infrastructure (2 hours)

**Goal**: Add loss functions, optimization, and basic training capability.

#### 5.1 Loss Functions (30 min)

```python
# ml/network/model.py (add to existing file)

class BlobNetTrainer:
    """
    Training utilities for BlobNet.
    """

    def __init__(
        self,
        model: BlobNet,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        value_loss_weight: float = 1.0,
        policy_loss_weight: float = 1.0,
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.value_loss_weight = value_loss_weight
        self.policy_loss_weight = policy_loss_weight

    def compute_loss(
        self,
        state: torch.Tensor,
        target_policy: torch.Tensor,
        target_value: torch.Tensor,
        legal_actions_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            state: (batch_size, state_dim)
            target_policy: (batch_size, action_dim) target action probs (from MCTS)
            target_value: (batch_size, 1) target score (from game outcome)
            legal_actions_mask: (batch_size, action_dim) legal action mask

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary with individual loss components
        """
        # Forward pass
        pred_policy, pred_value = self.model(state, legal_actions_mask)

        # Policy loss: Cross-entropy between MCTS policy and network policy
        # Only consider legal actions
        policy_loss = -torch.sum(
            target_policy * torch.log(pred_policy + 1e-8),
            dim=-1
        ).mean()

        # Value loss: MSE between predicted value and actual outcome
        value_loss = F.mse_loss(pred_value, target_value)

        # Combined loss
        total_loss = (
            self.policy_loss_weight * policy_loss +
            self.value_loss_weight * value_loss
        )

        loss_dict = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
        }

        return total_loss, loss_dict

    def train_step(
        self,
        state: torch.Tensor,
        target_policy: torch.Tensor,
        target_value: torch.Tensor,
        legal_actions_mask: torch.Tensor,
    ) -> dict:
        """
        Single training step.

        Returns:
            loss_dict: Dictionary with loss values
        """
        self.optimizer.zero_grad()

        loss, loss_dict = self.compute_loss(
            state, target_policy, target_value, legal_actions_mask
        )

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss_dict
```

#### 5.2 Model Checkpointing (20 min)

```python
def save_checkpoint(
    model: BlobNet,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    filepath: str,
):
    """Save model checkpoint."""
    torch.save({
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)

def load_checkpoint(
    filepath: str,
    model: BlobNet,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['iteration']
```

#### 5.3 Integration Test (40 min)

Create end-to-end test:
```python
def test_network_training_pipeline():
    """
    Test complete training pipeline:
    1. Create random game states
    2. Encode to tensors
    3. Create action masks
    4. Forward pass through network
    5. Compute loss with random targets
    6. Backward pass
    7. Verify gradients exist
    8. Verify loss decreases over iterations
    """
    # Create encoder and network
    encoder = StateEncoder()
    model = BlobNet()
    trainer = BlobNetTrainer(model)

    # Create dummy game
    game = BlobGame(num_players=4)
    game.setup_round(cards_to_deal=5)
    player = game.players[0]

    # Encode state
    state = encoder.encode(game, player)

    # Create action mask (bidding phase)
    masker = ActionMasker()
    mask = masker.create_bidding_mask(
        cards_dealt=5,
        is_dealer=False,
        forbidden_bid=None,
    )

    # Create random target (simulate MCTS output)
    target_policy = torch.zeros(52)
    target_policy[0:6] = torch.rand(6)  # Random bid preferences
    target_policy = target_policy / target_policy.sum()

    target_value = torch.tensor([[0.5]])  # Random value

    # Train for a few steps
    initial_loss = None
    for i in range(10):
        loss_dict = trainer.train_step(
            state.unsqueeze(0),
            target_policy.unsqueeze(0),
            target_value,
            mask.unsqueeze(0),
        )

        if i == 0:
            initial_loss = loss_dict['total_loss']

    final_loss = loss_dict['total_loss']

    # Loss should decrease
    assert final_loss < initial_loss, "Loss should decrease during training"

    print(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
```

#### 5.4 Documentation (30 min)

- [ ] Add comprehensive docstrings
- [ ] Document hyperparameter choices
- [ ] Create usage examples in comments

**Deliverable**: Complete neural network with training capability

---

### SESSION 6: MCTS - Node Implementation (2 hours)

**Goal**: Implement MCTS tree node structure with UCB1 selection.

#### 6.1 Setup (10 min)
- [ ] Create `ml/mcts/` directory
- [ ] Create `ml/mcts/__init__.py`
- [ ] Create `ml/mcts/node.py`
- [ ] Create `ml/mcts/test_mcts.py`

#### 6.2 MCTS Node Design (20 min)

Document MCTS algorithm:
```python
"""
Monte Carlo Tree Search (MCTS) for Blob

Algorithm:
    For N simulations:
        1. Selection: Traverse tree using UCB1 until leaf node
        2. Expansion: Add new child nodes for legal actions
        3. Evaluation: Use neural network to get (policy, value)
        4. Backpropagation: Update visit counts and values up the tree

UCB1 Formula:
    UCB(child) = Q(child) + c_puct * P(child) * sqrt(N_parent) / (1 + N_child)

    Where:
    - Q(child): Average value of child node
    - P(child): Prior probability from neural network policy
    - N_parent: Visit count of parent
    - N_child: Visit count of child
    - c_puct: Exploration constant (typically 1.5)

Tree Structure:
    Each node represents a game state after an action
    - Stores: visit count, total value, prior probability
    - Children: Map from action → child node
    - Parent: Reference to parent node
"""
```

#### 6.3 Implement MCTSNode Class (70 min)

```python
# ml/mcts/node.py

from typing import Dict, Optional, List
import numpy as np
from ml.game.blob import BlobGame, Player


class MCTSNode:
    """
    Node in the MCTS tree.

    Represents a game state and stores statistics for UCB1 selection.
    """

    def __init__(
        self,
        game_state: BlobGame,
        player: Player,
        parent: Optional['MCTSNode'] = None,
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
        """Check if node is a leaf (not yet expanded)."""
        return not self.is_expanded

    def is_root(self) -> bool:
        """Check if node is root (no parent)."""
        return self.parent is None

    def select_child(self, c_puct: float = 1.5) -> 'MCTSNode':
        """
        Select child with highest UCB1 score.

        Args:
            c_puct: Exploration constant

        Returns:
            Child node with highest UCB1 value
        """
        best_score = -float('inf')
        best_child = None

        for action, child in self.children.items():
            ucb_score = self._ucb1_score(child, c_puct)

            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child

        if best_child is None:
            raise ValueError("No children to select from")

        return best_child

    def _ucb1_score(self, child: 'MCTSNode', c_puct: float) -> float:
        """
        Compute UCB1 score for child node.

        UCB(child) = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
        """
        # Q: Average value
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

        Args:
            action_probs: Prior probabilities from neural network
            legal_actions: List of legal action indices
        """
        for action in legal_actions:
            if action not in self.children:
                # Create copy of game state for child
                child_state = self._simulate_action(action)

                # Get prior probability
                prior = action_probs.get(action, 1.0 / len(legal_actions))

                # Create child node
                child = MCTSNode(
                    game_state=child_state,
                    player=self.player,  # Will be updated based on game phase
                    parent=self,
                    action_taken=action,
                    prior_prob=prior,
                )

                self.children[action] = child

        self.is_expanded = True

    def _simulate_action(self, action: int) -> BlobGame:
        """
        Simulate taking an action and return new game state.

        NOTE: ✅ BlobGame.copy() and apply_action() were implemented in Phase 1a.10
        """
        # Create copy of game state
        new_game = self.game_state.copy()

        # Apply action to the copy
        new_game.apply_action(action, self.player)

        return new_game

    def backpropagate(self, value: float) -> None:
        """
        Backpropagate value up the tree.

        Args:
            value: Value to backpropagate (from neural network or terminal state)
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

        Args:
            temperature: Temperature for sampling (1.0 = proportional to visits)

        Returns:
            Dictionary mapping action → probability
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
            # Apply temperature
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / visits_temp.sum()

        return {action: prob for action, prob in zip(actions, probs)}

    def select_action(self, temperature: float = 1.0) -> int:
        """
        Select action based on visit counts.

        Returns:
            Action index
        """
        action_probs = self.get_action_probabilities(temperature)

        actions = list(action_probs.keys())
        probs = list(action_probs.values())

        # Sample action
        action = np.random.choice(actions, p=probs)

        return action
```

#### 6.4 Tests (20 min)

```python
class TestMCTSNode:
    def test_node_initialization(self):
        """Test node initializes correctly."""

    def test_is_leaf_before_expansion(self):
        """Test node is leaf before expansion."""

    def test_expand_creates_children(self):
        """Test expand creates child nodes."""

    def test_ucb1_score_calculation(self):
        """Test UCB1 score is computed correctly."""

    def test_select_child_picks_highest_ucb(self):
        """Test select_child picks highest UCB1 score."""

    def test_backpropagate_updates_stats(self):
        """Test backpropagation updates visit count and value."""

    def test_backpropagate_to_root(self):
        """Test backpropagation reaches root."""

    def test_action_probabilities_from_visits(self):
        """Test action probs proportional to visit counts."""
```

**Deliverable**: MCTSNode class with UCB1 selection and backpropagation

---

### SESSION 7: MCTS - Search Algorithm (2 hours)

**Goal**: Implement MCTS search algorithm integrated with neural network.

#### 7.1 Game State Cloning (30 min)

**✅ ALREADY IMPLEMENTED in Phase 1a.10**

The `BlobGame.copy()` and `BlobGame.apply_action()` methods have been implemented in Phase 1 to prepare for MCTS integration.

**What was implemented:**
- `copy()`: Uses `copy.deepcopy()` to create independent game state copies
- `apply_action()`: Handles both bidding (action = bid value) and playing (action = card index 0-51)
- Full validation with anti-cheat detection
- Card index mapping: 0-12 (♠), 13-25 (♥), 26-38 (♣), 39-51 (♦)

**Tasks for this session:**
- [ ] Read and understand the existing implementation in `ml/game/blob.py` (lines ~1535-1685)
- [ ] Write unit tests for `copy()` method:
  - Test that copy is independent (modify copy doesn't affect original)
  - Test all game state is copied (players, deck, tricks, history)
  - Test with different game phases
- [ ] Write unit tests for `apply_action()` method:
  - Test bidding actions (valid bids, dealer constraints)
  - Test playing actions (card index mapping, follow-suit rules)
  - Test error cases (invalid indices, illegal plays)
- [ ] Verify integration with existing game logic

**Time estimate:** 30 min (now focused on testing, not implementation)

#### 7.2 Implement MCTS Class (60 min)

```python
# ml/mcts/search.py

import torch
import numpy as np
from typing import Dict, Optional
from ml.mcts.node import MCTSNode
from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.game.blob import BlobGame, Player


class MCTS:
    """
    Monte Carlo Tree Search implementation for Blob.

    Integrates with neural network for leaf evaluation.
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
        Initialize MCTS.

        Args:
            network: Neural network for evaluation
            encoder: State encoder
            masker: Action masker
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant for UCB1
            temperature: Temperature for action selection
        """
        self.network = network
        self.encoder = encoder
        self.masker = masker
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature

    def search(
        self,
        game_state: BlobGame,
        player: Player,
    ) -> Dict[int, float]:
        """
        Run MCTS search from current game state.

        Args:
            game_state: Current game state
            player: Player whose turn it is

        Returns:
            Dictionary mapping action → probability (from visit counts)
        """
        # Create root node
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
        Run one MCTS simulation.

        Returns:
            Value of the leaf node (from neural network)
        """
        # 1. Selection: Traverse tree until leaf node
        while not node.is_leaf():
            node = node.select_child(self.c_puct)

        # 2. Check if game is terminal
        if self._is_terminal(node.game_state):
            # Return actual game outcome
            value = self._get_terminal_value(node.game_state, node.player)
            node.backpropagate(value)
            return value

        # 3. Expansion + Evaluation: Use neural network
        value = self._expand_and_evaluate(node)

        # 4. Backpropagation
        node.backpropagate(value)

        return value

    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """
        Expand node and evaluate with neural network.

        Returns:
            Value from neural network
        """
        # Encode state
        state_tensor = self.encoder.encode(node.game_state, node.player)

        # Get legal actions
        legal_actions, legal_mask = self._get_legal_actions_and_mask(
            node.game_state, node.player
        )

        # Neural network evaluation
        with torch.no_grad():
            policy, value = self.network(state_tensor, legal_mask)

        # Convert policy to dictionary
        policy_np = policy.cpu().numpy()
        action_probs = {action: policy_np[action] for action in legal_actions}

        # Expand node
        node.expand(action_probs, legal_actions)

        # Return value
        return value.item()

    def _get_legal_actions_and_mask(
        self,
        game: BlobGame,
        player: Player,
    ) -> tuple[List[int], torch.Tensor]:
        """
        Get legal actions and mask for current game state.

        Returns:
            legal_actions: List of legal action indices
            legal_mask: Tensor mask for neural network
        """
        if game.game_phase == 'bidding':
            # Get legal bids
            cards_dealt = len(player.hand)
            is_dealer = (player.position == game.dealer_position)
            forbidden_bid = None

            if is_dealer:
                # Calculate forbidden bid
                total_bids = sum(p.bid for p in game.players if p.bid is not None)
                forbidden_bid = cards_dealt - total_bids

            # Create mask
            mask = self.masker.create_bidding_mask(
                cards_dealt, is_dealer, forbidden_bid
            )

            # Get legal actions
            legal_actions = [i for i in range(cards_dealt + 1)
                           if mask[i] == 1.0 and (not is_dealer or i != forbidden_bid)]

        elif game.game_phase == 'playing':
            # Get legal card plays
            led_suit = game.current_trick.led_suit if game.current_trick else None

            mask = self.masker.create_playing_mask(
                player.hand, led_suit, self.encoder
            )

            # Get legal actions
            legal_actions = [self.encoder._card_to_index(card)
                           for card in player.hand
                           if mask[self.encoder._card_to_index(card)] == 1.0]

        else:
            raise ValueError(f"Invalid game phase for MCTS: {game.game_phase}")

        return legal_actions, mask

    def _is_terminal(self, game: BlobGame) -> bool:
        """Check if game state is terminal (round over)."""
        return game.game_phase == 'complete' or game.game_phase == 'scoring'

    def _get_terminal_value(self, game: BlobGame, player: Player) -> float:
        """
        Get value of terminal state.

        Returns:
            Normalized score in [-1, 1]
        """
        # Calculate score for this round
        score = player.calculate_round_score()

        # Normalize to [-1, 1]
        # Max score is 10 + max_cards (typically 10 + 13 = 23)
        max_score = 23
        normalized_score = score / max_score

        return normalized_score
```

#### 7.3 Tests (30 min)

```python
class TestMCTS:
    def test_mcts_initialization(self):
        """Test MCTS initializes correctly."""

    def test_search_returns_probabilities(self):
        """Test search returns action probabilities."""

    def test_search_probabilities_sum_to_one(self):
        """Test action probabilities sum to 1."""

    def test_more_simulations_improve_quality(self):
        """Test more simulations converge to better actions."""

    def test_legal_actions_only(self):
        """Test MCTS only considers legal actions."""

    def test_terminal_state_handling(self):
        """Test MCTS handles terminal states."""
```

**Deliverable**: Complete MCTS search algorithm integrated with neural network

---

### SESSION 8: MCTS - Tree Reuse & Optimization (2 hours)

**Goal**: Add tree reuse, performance optimizations, and batched inference.

#### 8.1 Tree Reuse Implementation (40 min)

```python
# ml/mcts/search.py (add to MCTS class)

class MCTS:
    # ... existing code ...

    def __init__(self, ...):
        # ... existing init ...
        self.root = None  # Store root for tree reuse

    def search_with_tree_reuse(
        self,
        game_state: BlobGame,
        player: Player,
        previous_action: Optional[int] = None,
    ) -> Dict[int, float]:
        """
        Run MCTS with tree reuse.

        If previous_action is provided, navigate to that child node
        and make it the new root (keeping its subtree).

        Args:
            game_state: Current game state
            player: Current player
            previous_action: Action taken to reach this state

        Returns:
            Action probabilities
        """
        # Tree reuse: Navigate to child node from previous search
        if self.root is not None and previous_action is not None:
            if previous_action in self.root.children:
                # Reuse subtree
                self.root = self.root.children[previous_action]
                self.root.parent = None  # Make it new root
            else:
                # Action not in tree, create new root
                self.root = None

        # Create new root if needed
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

    def reset_tree(self):
        """Clear tree (for new game)."""
        self.root = None
```

#### 8.2 Batched Neural Network Inference (40 min)

Optimize MCTS by batching network evaluations:
```python
class MCTS:
    # ... existing code ...

    def search_batched(
        self,
        game_state: BlobGame,
        player: Player,
        batch_size: int = 8,
    ) -> Dict[int, float]:
        """
        Run MCTS with batched neural network inference.

        Accumulate leaf nodes, evaluate in batch, then backpropagate.
        Improves GPU utilization.

        Args:
            game_state: Current game state
            player: Current player
            batch_size: Number of leaf nodes to evaluate per batch

        Returns:
            Action probabilities
        """
        root = MCTSNode(game_state, player)

        num_batches = (self.num_simulations + batch_size - 1) // batch_size

        for _ in range(num_batches):
            # Collect leaf nodes for batch
            leaf_nodes = []
            for _ in range(min(batch_size, self.num_simulations)):
                leaf = self._traverse_to_leaf(root)
                leaf_nodes.append(leaf)

            # Batch evaluate
            if leaf_nodes:
                self._batch_expand_and_evaluate(leaf_nodes)

        return root.get_action_probabilities(self.temperature)

    def _traverse_to_leaf(self, root: MCTSNode) -> MCTSNode:
        """Traverse from root to leaf using UCB1."""
        node = root
        while not node.is_leaf():
            node = node.select_child(self.c_puct)
        return node

    def _batch_expand_and_evaluate(self, nodes: List[MCTSNode]) -> None:
        """
        Expand and evaluate multiple nodes in batch.

        Args:
            nodes: List of leaf nodes to evaluate
        """
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

        # Stack into batch
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
                action: policy[action]
                for action in legal_actions_list[i]
            }

            # Expand
            node.expand(action_probs, legal_actions_list[i])

            # Backpropagate
            node.backpropagate(value)
```

#### 8.3 Performance Benchmarks (20 min)

```python
def test_mcts_inference_speed():
    """
    Benchmark MCTS inference speed.

    Target: <200ms per move on CPU
    """
    import time

    # Setup
    network = BlobNet()
    encoder = StateEncoder()
    masker = ActionMasker()
    mcts = MCTS(network, encoder, masker, num_simulations=100)

    # Create game
    game = BlobGame(num_players=4)
    game.setup_round(cards_to_deal=5)
    player = game.players[0]

    # Benchmark
    start_time = time.time()
    action_probs = mcts.search(game, player)
    elapsed_ms = (time.time() - start_time) * 1000

    print(f"MCTS search time: {elapsed_ms:.2f} ms")
    print(f"Simulations: 100")
    print(f"Time per simulation: {elapsed_ms/100:.2f} ms")

    assert elapsed_ms < 200, f"Too slow: {elapsed_ms:.2f} ms > 200 ms"
```

#### 8.4 Integration Tests (20 min)

Complete end-to-end test:
```python
def test_full_game_with_mcts():
    """
    Test playing complete game using MCTS.

    Verifies:
    - MCTS makes legal moves throughout game
    - Game completes without errors
    - Both bidding and playing phases work
    """
    # Create network and MCTS
    network = BlobNet()
    encoder = StateEncoder()
    masker = ActionMasker()
    mcts = MCTS(network, encoder, masker, num_simulations=50)

    # Create game
    game = BlobGame(num_players=4)
    game.setup_round(cards_to_deal=5)

    # Bidding phase
    for player in game.players:
        action_probs = mcts.search(game, player)
        bid = max(action_probs, key=action_probs.get)
        player.make_bid(bid)
        print(f"{player.name} bids: {bid}")

    # Playing phase
    for trick_num in range(5):
        print(f"\nTrick {trick_num + 1}:")
        for player in game.players:
            action_probs = mcts.search(game, player)
            card_idx = max(action_probs, key=action_probs.get)

            # Find card corresponding to index
            # Play card...
            # (Need to implement action → card mapping)

    # Verify game completed
    assert game.game_phase == 'scoring', "Game should reach scoring phase"
```

**Deliverable**: Optimized MCTS with tree reuse and batching

---

### SESSION 9: Integration Testing & Validation (2 hours)

**Goal**: Comprehensive testing and validation of complete system.

#### 9.1 End-to-End Integration (40 min)

Create comprehensive integration test:
```python
# ml/tests/test_integration.py

def test_phase2_complete_pipeline():
    """
    Test complete Phase 2 pipeline:
    1. Create game
    2. Encode state
    3. Get legal actions
    4. Run MCTS
    5. Select action
    6. Apply to game
    7. Repeat until game over
    """
    from ml.game.blob import BlobGame
    from ml.network.model import BlobNet
    from ml.network.encode import StateEncoder, ActionMasker
    from ml.mcts.search import MCTS

    # Initialize components
    network = BlobNet()
    encoder = StateEncoder()
    masker = ActionMasker()
    mcts = MCTS(network, encoder, masker, num_simulations=100)

    # Create game
    game = BlobGame(num_players=4, player_names=["MCTS1", "MCTS2", "MCTS3", "MCTS4"])

    # Play 3-card round
    game.setup_round(cards_to_deal=3)

    print("=== BIDDING PHASE ===")
    # Bidding phase
    for i, player in enumerate(game.players):
        print(f"\n{player.name}'s turn to bid:")
        print(f"Hand: {sorted(player.hand)}")

        # MCTS search
        action_probs = mcts.search(game, player)

        # Select bid
        bid = max(action_probs, key=action_probs.get)
        print(f"Action probabilities: {action_probs}")
        print(f"Selected bid: {bid}")

        # Apply bid
        player.make_bid(bid)

    print("\n=== PLAYING PHASE ===")
    # Playing phase
    lead_player_idx = (game.dealer_position + 1) % len(game.players)

    for trick_num in range(3):
        print(f"\nTrick {trick_num + 1}:")
        trick = game.create_trick()

        for i in range(len(game.players)):
            player_idx = (lead_player_idx + i) % len(game.players)
            player = game.players[player_idx]

            print(f"\n{player.name}'s turn:")
            print(f"Hand: {sorted(player.hand)}")

            # MCTS search
            action_probs = mcts.search(game, player)

            # Select card (action is card index)
            card_idx = max(action_probs, key=action_probs.get)

            # Map index back to card
            # Find card in hand matching index
            card_to_play = None
            for card in player.hand:
                if encoder._card_to_index(card) == card_idx:
                    card_to_play = card
                    break

            if card_to_play is None:
                # Fallback: play first legal card
                legal_cards = game.get_legal_plays(player, trick.led_suit)
                card_to_play = legal_cards[0]

            print(f"Plays: {card_to_play}")

            # Apply action
            trick.add_card(player, card_to_play)
            player.play_card(card_to_play)

        # Determine winner
        winner = trick.determine_winner()
        winner.win_trick()
        lead_player_idx = winner.position

        print(f"\n{winner.name} wins the trick!")

    print("\n=== SCORING PHASE ===")
    # Scoring
    for player in game.players:
        score = player.calculate_round_score()
        print(f"{player.name}: Bid {player.bid}, Won {player.tricks_won}, Score {score}")

    print("\nTest completed successfully!")
```

#### 9.2 Performance Validation (30 min)

Benchmark all components:
```python
def test_performance_benchmarks():
    """
    Validate performance targets:
    - State encoding: <1ms
    - Neural network forward pass: <10ms (CPU)
    - MCTS search (100 sims): <200ms (CPU)
    - Full move decision: <250ms (CPU)
    """
    import time

    # Setup
    network = BlobNet()
    encoder = StateEncoder()
    masker = ActionMasker()
    mcts = MCTS(network, encoder, masker, num_simulations=100)

    game = BlobGame(num_players=4)
    game.setup_round(cards_to_deal=5)
    player = game.players[0]

    # Benchmark state encoding
    start = time.time()
    for _ in range(100):
        state = encoder.encode(game, player)
    encoding_time = (time.time() - start) * 1000 / 100
    print(f"State encoding: {encoding_time:.2f} ms")
    assert encoding_time < 1.0

    # Benchmark network inference
    state = encoder.encode(game, player)
    legal_actions, mask = mcts._get_legal_actions_and_mask(game, player)

    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            policy, value = network(state, mask)
    inference_time = (time.time() - start) * 1000 / 100
    print(f"Network inference: {inference_time:.2f} ms")
    assert inference_time < 10.0

    # Benchmark MCTS
    start = time.time()
    action_probs = mcts.search(game, player)
    mcts_time = (time.time() - start) * 1000
    print(f"MCTS search (100 sims): {mcts_time:.2f} ms")
    assert mcts_time < 200.0

    print("All performance targets met!")
```

#### 9.3 Improvement over Random (30 min)

Test that MCTS plays better than random:
```python
def test_mcts_beats_random():
    """
    Test MCTS agent beats random agent over multiple games.

    Play 20 games: MCTS vs 3 random players
    MCTS should win more points than average.
    """
    from ml.game.blob import BlobGame
    import random

    network = BlobNet()
    encoder = StateEncoder()
    masker = ActionMasker()
    mcts = MCTS(network, encoder, masker, num_simulations=50)

    mcts_scores = []
    random_scores = []

    for game_num in range(20):
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=3)

        # Player 0 uses MCTS, others random
        # Bidding
        for i, player in enumerate(game.players):
            if i == 0:
                # MCTS bid
                action_probs = mcts.search(game, player)
                bid = max(action_probs, key=action_probs.get)
            else:
                # Random bid
                legal_bids = range(4)  # 0-3 for 3 cards
                bid = random.choice(legal_bids)

            player.make_bid(bid)

        # Playing (simplified)
        # ... play tricks using MCTS for player 0, random for others

        # Score
        mcts_scores.append(game.players[0].calculate_round_score())
        random_scores.extend([
            p.calculate_round_score() for p in game.players[1:]
        ])

    avg_mcts = sum(mcts_scores) / len(mcts_scores)
    avg_random = sum(random_scores) / len(random_scores)

    print(f"MCTS average score: {avg_mcts:.2f}")
    print(f"Random average score: {avg_random:.2f}")

    # With random network, MCTS should still be slightly better
    # (due to lookahead)
    assert avg_mcts >= avg_random * 0.8, "MCTS should not be much worse than random"
```

#### 9.4 Documentation & README Updates (20 min)

- [ ] Update main README with Phase 2 completion status
- [ ] Document neural network architecture
- [ ] Document MCTS parameters
- [ ] Add usage examples
- [ ] Update roadmap

**Deliverable**: Fully tested and validated Phase 2 implementation

---

## Success Criteria

### Functional Requirements
✅ State encoder converts game state → 512-dim tensor
✅ Neural network produces policy and value outputs
✅ Legal action masking prevents illegal moves
✅ MCTS integrates with neural network
✅ MCTS makes legal moves for bidding and playing
✅ Complete games can be played with MCTS agents
✅ Inference speed: <200ms per move (CPU)

### Code Quality
✅ All pytest tests pass
✅ Code coverage >85%
✅ Type hints on all function signatures
✅ Comprehensive docstrings

### Performance Targets
✅ State encoding: <1ms
✅ Network inference: <10ms (CPU)
✅ MCTS search (100 sims): <200ms (CPU)
✅ Full move decision: <250ms (CPU)

### Ready for Phase 3
✅ MCTS handles imperfect information states
✅ Belief state representation in encoder
✅ Network can be trained with dummy data
✅ Integration tests validate complete pipeline

---

## Deliverables

After Phase 2 completion:

1. **ml/network/encode.py**: State encoding (~300 lines)
2. **ml/network/model.py**: Neural network + trainer (~400 lines)
3. **ml/network/test_network.py**: Network tests (~200 lines)
4. **ml/mcts/node.py**: MCTS node (~200 lines)
5. **ml/mcts/search.py**: MCTS search (~300 lines)
6. **ml/mcts/test_mcts.py**: MCTS tests (~200 lines)
7. **ml/tests/test_integration.py**: Integration tests (~300 lines)

**Total**: ~1900 lines of new code

---

## Timeline Summary

| Session | Duration | Component | Deliverable |
|---------|----------|-----------|-------------|
| 1 | 2h | State Encoding - Basic | StateEncoder class with basic structure |
| 2 | 2h | State Encoding - Complete | Full state encoding with tests |
| 3 | 2h | Neural Network - Transformer | Basic Transformer architecture |
| 4 | 2h | Neural Network - Masking | Legal action masking integrated |
| 5 | 2h | Neural Network - Training | Loss functions and training loop |
| 6 | 2h | MCTS - Node | MCTSNode with UCB1 |
| 7 | 2h | MCTS - Search | Complete MCTS algorithm |
| 8 | 2h | MCTS - Optimization | Tree reuse and batching |
| 9 | 2h | Integration Testing | Full pipeline validation |

**Total**: 18 hours (9 sessions × 2 hours)

---

## Next Steps (Phase 3)

After Phase 2:

1. **Code Review**: Validate network architecture and MCTS implementation
2. **Performance Tuning**: Optimize slow components
3. **Hyperparameter Experiments**: Test different network sizes, MCTS simulations
4. **Proceed to Phase 3**: Imperfect Information Handling
   - Belief state tracking
   - Determinization sampling
   - MCTS with multiple determinizations

---

## Common Issues & Solutions

### Issue: MCTS too slow
**Solution**:
- Reduce num_simulations (try 50 instead of 100)
- Use batched inference
- Profile code to find bottlenecks

### Issue: Network outputs NaN
**Solution**:
- Check for division by zero in normalization
- Add epsilon to log computations
- Verify input tensors are valid

### Issue: MCTS only picks one action
**Solution**:
- Check temperature setting (should be >0)
- Verify UCB1 exploration is working
- Increase c_puct constant

### Issue: Game state cloning slow
**Solution**:
- Implement shallow copy for immutable fields
- Cache encoded states
- Use copy-on-write data structures

---

## References

- **AlphaZero Paper**: [Mastering Chess and Shogi](https://arxiv.org/abs/1712.01815)
- **MCTS Survey**: [A Survey of Monte Carlo Tree Search Methods](https://ieeexplore.ieee.org/document/6145622)
- **Transformers**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **PyTorch Docs**: [pytorch.org/docs](https://pytorch.org/docs/stable/index.html)

---

**Last Updated**: 2025-10-23
**Status**: Ready for implementation
**Version**: 1.0
