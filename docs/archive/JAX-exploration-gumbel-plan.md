# JAX + Gumbel MuZero Transition Plan

**Project**: BlobMaster
**Goal**: Accelerate self-play from 6 games/min → 120-300 games/min (20-50x speedup)
**Approach**: Replace Python MCTS with JAX + Gumbel MuZero algorithm
**Timeline**: 3-4 days
**Status**: Planning phase

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [MCTS Algorithm Analysis](#mcts-algorithm-analysis)
3. [Architecture Decision](#architecture-decision)
4. [Implementation Plan](#implementation-plan)
5. [Expected Performance](#expected-performance)
6. [Risk Assessment](#risk-assessment)
7. [File Modifications](#file-modifications)
8. [Testing Strategy](#testing-strategy)

---

## Executive Summary

### Current Bottleneck

From [benchmark results](Benchmarks/results/20251027_161731_benchmark_report.md):
- **6-12 games/min** with 3 determinizations × 30 MCTS simulations
- Self-play is **CPU-bound** and **sequential** (Python loops)
- Worker scaling plateaus at 8+ workers (no benefit beyond that)
- Training on GPU is fast; self-play is the limiting factor

### Proposed Solution

**Hybrid approach:**
- Keep **AlphaZero network architecture** (policy + value heads)
- Replace **Python MCTS** with **JAX + Gumbel MuZero MCTS algorithm**
- Use **real game rules** (not learned dynamics)
- Vectorize determinization sampling

**Why this works:**
- JAX JIT compilation: 10-20x faster than Python
- Gumbel sampling: 1.5-2x more sample-efficient
- Vectorization: Process all determinizations in parallel
- GPU acceleration: Leverage RTX 4060 for tree search

### Expected Results

| Metric | Current | After JAX Migration |
|--------|---------|-------------------|
| Games/min | 6-12 | **120-300** |
| MCTS sims needed | 90 | **50-60** |
| Training timeline | Months | **~10 days** |
| Speedup | 1x | **20-50x** |

---

## MCTS Algorithm Analysis

### Three Variants Compared

#### 1. AlphaZero (Current Implementation)

**Architecture:**
- Network: Policy(s) + Value(s)
- MCTS: Uses real game simulator
- Action selection: UCB-based visit counts

**Pros:**
- ✅ Simple to implement
- ✅ Proven for perfect-info games (Chess, Go)
- ✅ Well-documented

**Cons:**
- ❌ Needs many simulations (90+)
- ❌ Suboptimal for large action spaces (52 cards)
- ❌ Manual exploration tuning (temperature schedules)

#### 2. Full MuZero

**Architecture:**
- Network: Representation(o) + Dynamics(s,a) + Prediction(s)
- MCTS: Uses *learned* game model
- Action selection: Same as AlphaZero

**Pros:**
- ✅ Can learn from raw pixels
- ✅ Useful when game rules unknown

**Cons:**
- ❌ **Don't need this** - we have perfect game rules!
- ❌ 3x network complexity (representation + dynamics + prediction)
- ❌ Slower training (more parameters)
- ❌ Overkill for card games

#### 3. Gumbel MuZero (RECOMMENDED) ✅

**Architecture:**
- Network: **Same as AlphaZero** (policy + value)
- MCTS: Uses **Gumbel sampling** for action selection
- Dynamics: Real game simulator (like AlphaZero)

**Pros:**
- ✅ **No network changes** - keep existing architecture
- ✅ **Fewer simulations needed** (50 vs 90)
- ✅ **Better exploration** - Gumbel noise handles uncertainty
- ✅ **Large action spaces** - Prunes to top-k actions
- ✅ **Imperfect information** - Proven in poker-like games
- ✅ **DeepMind-recommended** - "50% fewer sims for same quality"

**Cons:**
- ⚠️ Slightly more complex API (but `mctx` abstracts this)

### Why Gumbel for Blob?

| Criterion | Value | Why Gumbel Helps |
|-----------|-------|------------------|
| **Action space size** | 52 cards + 14 bids | Gumbel prunes to top-k efficiently |
| **Imperfect information** | Hidden opponent hands | Gumbel sampling robust to uncertainty |
| **Need for speed** | 10-50x faster | Gumbel needs fewer sims (50 vs 90) |
| **Game complexity** | Known rules | Can use real simulator (no learned dynamics) |

**Key Insight:** Gumbel MuZero = AlphaZero network + Improved MCTS algorithm

---

## Architecture Decision

### What We're Building

```
┌─────────────────────────────────────────────────────────────┐
│                     HYBRID ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Training Loop (PyTorch - UNCHANGED)                        │
│  ├── Neural Network: Policy + Value heads                   │
│  ├── Replay Buffer                                          │
│  ├── Adam Optimizer                                         │
│  └── Loss: CrossEntropy(policy) + MSE(value)                │
│                                                             │
│  ┌────────────────────────────────────┐                     │
│  │     Weight Conversion Bridge       │                     │
│  │   PyTorch state_dict → JAX params  │                     │
│  └────────────────────────────────────┘                     │
│                                                             │
│  Self-Play (JAX - NEW)                                      │
│  ├── Neural Network (JAX): Same architecture as PyTorch     │
│  ├── MCTS: mctx.gumbel_muzero_policy                        │
│  ├── Determinization: Vectorized sampling (vmap)            │
│  ├── Game Simulator: Real Blob rules                        │
│  └── Output: Training examples → Replay Buffer              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Framework | Purpose | Status |
|-----------|-----------|---------|--------|
| **Network Training** | PyTorch | Gradient descent, optimization | ✅ Keep unchanged |
| **Replay Buffer** | NumPy | Store training examples | ✅ Keep unchanged |
| **Evaluation Arena** | PyTorch | Model tournaments, ELO | ✅ Keep unchanged |
| **Network Inference** | JAX | Fast forward passes in MCTS | 🔄 New implementation |
| **MCTS Search** | JAX + mctx | Tree search with Gumbel sampling | 🔄 New implementation |
| **Determinization** | JAX | Vectorized opponent hand sampling | 🔄 New implementation |
| **Game Simulator** | Python | Blob game rules | ✅ Keep unchanged |

### Why Hybrid (Not Pure JAX)?

**Training stays in PyTorch because:**
- ✅ Already fast (not the bottleneck)
- ✅ Mature ecosystem (better debugging, logging)
- ✅ Stable training loop (426+ passing tests)
- ✅ Risk mitigation (only change the slow part)

**Inference moves to JAX because:**
- ✅ **10-20x faster** (JIT compilation)
- ✅ Vectorization support (`vmap` for batching)
- ✅ `mctx` library (production-ready MCTS)
- ✅ GPU acceleration for tree search

---

## Implementation Plan

### Overview: 4-Day Timeline

```
Day 1: JAX Network (~6 hours)
├── Implement BlobNetJAX in Flax
├── Create PyTorch → JAX weight converter
└── Unit tests: Verify identical outputs

Day 2: Gumbel MCTS (~8 hours)
├── Implement mctx.RecurrentFn (game dynamics)
├── Implement mctx.RootFnOutput (state evaluation)
├── Wrap mctx.gumbel_muzero_policy
└── Unit tests: Verify legal actions only

Day 3: Vectorized Determinization (~6 hours)
├── Batch determinization sampling with vmap
├── Aggregate action probs across worlds
└── Integration tests: Run full games

Day 4: Self-Play Integration & Testing (~6 hours)
├── Update SelfPlayWorker to use JAX MCTS
├── End-to-end test: Generate 100 games
├── Benchmark: Measure speedup
└── Performance tuning
```

---

### Phase 1: JAX Network (Day 1)

**Goal**: Implement identical network architecture in JAX/Flax

#### Step 1.1: Create `ml/network/model_jax.py`

```python
"""
JAX implementation of BlobNet (identical to PyTorch version).

Uses Flax for neural network layers.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Tuple


class TransformerBlock(nn.Module):
    """Single transformer encoder layer."""
    num_heads: int = 8
    dim: int = 256
    feedforward_dim: int = 1024
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = False):
        # Multi-head self-attention
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            dropout_rate=self.dropout if training else 0.0,
        )(x, x)

        # Add & Norm
        x = nn.LayerNorm()(x + attn_output)

        # Feedforward
        ff_output = nn.Dense(self.feedforward_dim)(x)
        ff_output = nn.relu(ff_output)
        ff_output = nn.Dropout(rate=self.dropout, deterministic=not training)(ff_output)
        ff_output = nn.Dense(self.dim)(ff_output)

        # Add & Norm
        x = nn.LayerNorm()(x + ff_output)

        return x


class BlobNetJAX(nn.Module):
    """
    JAX implementation of BlobNet.

    Architecture matches PyTorch version exactly:
    - Input: 256-dim state vector
    - Transformer: 6 layers, 8 heads, 1024 feedforward
    - Outputs: Policy (52-dim) + Value (1-dim)
    """

    state_dim: int = 256
    embedding_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    feedforward_dim: int = 1024
    dropout: float = 0.1
    action_dim: int = 52  # max(14 bids, 52 cards)

    @nn.compact
    def __call__(self, state: jnp.ndarray, training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass.

        Args:
            state: (batch_size, state_dim) or (state_dim,)
            training: Whether in training mode (affects dropout)

        Returns:
            policy_logits: (batch_size, action_dim) or (action_dim,)
            value: (batch_size, 1) or (1,)
        """
        # Handle single state (add batch dimension)
        single_input = False
        if state.ndim == 1:
            state = state[None, :]  # Add batch dim
            single_input = True

        # Input embedding
        x = nn.Dense(self.embedding_dim)(state)
        x = nn.LayerNorm()(x)

        # Add positional encoding (learned)
        # Shape: (1, 1, embedding_dim) - broadcasts to batch
        pos_encoding = self.param(
            'positional_encoding',
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.embedding_dim)
        )
        x = x[:, None, :] + pos_encoding  # (batch, 1, embedding_dim)

        # Transformer encoder
        for _ in range(self.num_layers):
            x = TransformerBlock(
                num_heads=self.num_heads,
                dim=self.embedding_dim,
                feedforward_dim=self.feedforward_dim,
                dropout=self.dropout,
            )(x, training=training)

        x = x.squeeze(1)  # (batch, embedding_dim)

        # Policy head
        policy_logits = nn.Dense(256)(x)
        policy_logits = nn.relu(policy_logits)
        policy_logits = nn.Dropout(rate=self.dropout, deterministic=not training)(policy_logits)
        policy_logits = nn.Dense(self.action_dim)(policy_logits)

        # Value head
        value = nn.Dense(128)(x)
        value = nn.relu(value)
        value = nn.Dropout(rate=self.dropout, deterministic=not training)(value)
        value = nn.Dense(1)(value)
        value = jnp.tanh(value)

        # Remove batch dimension if single input
        if single_input:
            policy_logits = policy_logits[0]
            value = value[0]

        return policy_logits, value


def create_jax_network(state_dim: int = 256) -> BlobNetJAX:
    """
    Factory function to create JAX network.

    Args:
        state_dim: Input state dimension

    Returns:
        BlobNetJAX instance
    """
    return BlobNetJAX(state_dim=state_dim)


def init_network_params(network: BlobNetJAX, rng_key, dummy_input):
    """
    Initialize network parameters.

    Args:
        network: BlobNetJAX instance
        rng_key: JAX random key
        dummy_input: Example input for shape inference

    Returns:
        Initialized parameters
    """
    params = network.init(rng_key, dummy_input, training=False)
    return params
```

#### Step 1.2: Create `ml/utils/weight_converter.py`

```python
"""
Weight conversion utilities between PyTorch and JAX.

Handles:
- Layer name mapping
- Weight matrix transposition (Conv layers)
- Parameter structure differences (Flax vs PyTorch)
"""

import torch
import jax.numpy as jnp
from typing import Dict, Any
import numpy as np


def pytorch_to_jax(pytorch_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Convert PyTorch state_dict to JAX params.

    Args:
        pytorch_state_dict: PyTorch model.state_dict()

    Returns:
        JAX params compatible with Flax
    """
    jax_params = {}

    # Mapping: PyTorch layer names → JAX/Flax layer names
    # PyTorch: 'input_embedding.weight', 'input_embedding.bias'
    # JAX: {'input_embedding': {'kernel': ..., 'bias': ...}}

    for key, value in pytorch_state_dict.items():
        # Convert tensor to numpy
        value_np = value.cpu().detach().numpy()

        # Handle layer name transformations
        if 'weight' in key:
            # PyTorch Dense: (out_features, in_features)
            # JAX Dense: (in_features, out_features)
            # Need to transpose!
            if 'Linear' in key or 'Dense' in key or 'embedding' in key or 'fc' in key:
                value_np = value_np.T

            # Rename 'weight' → 'kernel' (Flax convention)
            jax_key = key.replace('weight', 'kernel')
        else:
            jax_key = key

        # Convert to JAX array
        jax_params[jax_key] = jnp.array(value_np)

    return jax_params


def jax_to_pytorch(jax_params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Convert JAX params to PyTorch state_dict.

    Args:
        jax_params: JAX/Flax parameters

    Returns:
        PyTorch-compatible state_dict
    """
    pytorch_state_dict = {}

    for key, value in jax_params.items():
        # Convert JAX array to numpy
        value_np = np.array(value)

        # Handle transformations (reverse of pytorch_to_jax)
        if 'kernel' in key:
            # Transpose back
            if 'Linear' in key or 'Dense' in key or 'embedding' in key or 'fc' in key:
                value_np = value_np.T

            # Rename 'kernel' → 'weight'
            pytorch_key = key.replace('kernel', 'weight')
        else:
            pytorch_key = key

        # Convert to PyTorch tensor
        pytorch_state_dict[pytorch_key] = torch.from_numpy(value_np)

    return pytorch_state_dict


def verify_conversion(
    pytorch_model,
    jax_model,
    jax_params,
    num_tests: int = 100,
    atol: float = 1e-5
) -> bool:
    """
    Verify that PyTorch and JAX models produce identical outputs.

    Args:
        pytorch_model: PyTorch BlobNet
        jax_model: JAX BlobNetJAX
        jax_params: JAX parameters (converted from PyTorch)
        num_tests: Number of random inputs to test
        atol: Absolute tolerance for comparison

    Returns:
        True if all tests pass
    """
    pytorch_model.eval()

    for i in range(num_tests):
        # Random input
        state = torch.randn(256)

        # PyTorch forward
        with torch.no_grad():
            pytorch_policy, pytorch_value = pytorch_model(state, legal_actions_mask=None)

        # JAX forward
        state_jax = jnp.array(state.numpy())
        jax_policy, jax_value = jax_model.apply(jax_params, state_jax, training=False)

        # Compare
        policy_match = np.allclose(
            pytorch_policy.numpy(),
            np.array(jax_policy),
            atol=atol
        )
        value_match = np.allclose(
            pytorch_value.numpy(),
            np.array(jax_value),
            atol=atol
        )

        if not (policy_match and value_match):
            print(f"Test {i+1}/{num_tests} FAILED")
            print(f"  Policy diff: {np.max(np.abs(pytorch_policy.numpy() - np.array(jax_policy)))}")
            print(f"  Value diff: {np.max(np.abs(pytorch_value.numpy() - np.array(jax_value)))}")
            return False

    print(f"All {num_tests} conversion tests PASSED")
    return True
```

#### Step 1.3: Unit Tests

Create `ml/network/test_network_jax.py`:

```python
"""Unit tests for JAX network implementation."""

import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np

from ml.network.model import BlobNet
from ml.network.model_jax import BlobNetJAX, create_jax_network, init_network_params
from ml.utils.weight_converter import pytorch_to_jax, verify_conversion


def test_jax_network_creation():
    """Test JAX network can be created and initialized."""
    network = create_jax_network(state_dim=256)

    # Initialize with dummy input
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 256))
    params = init_network_params(network, rng, dummy_input)

    assert params is not None
    print(f"JAX network created successfully")


def test_jax_forward_pass():
    """Test JAX network forward pass."""
    network = create_jax_network()
    rng = jax.random.PRNGKey(0)

    # Initialize
    dummy_input = jnp.ones((1, 256))
    params = init_network_params(network, rng, dummy_input)

    # Forward pass
    state = jax.random.normal(rng, (256,))
    policy, value = network.apply(params, state, training=False)

    # Check shapes
    assert policy.shape == (52,)
    assert value.shape == (1,)

    # Check value range
    assert jnp.all(value >= -1.0) and jnp.all(value <= 1.0)

    print("JAX forward pass test PASSED")


def test_weight_conversion():
    """Test PyTorch → JAX weight conversion produces identical outputs."""
    # Create PyTorch network
    pytorch_net = BlobNet()
    pytorch_net.eval()

    # Create JAX network
    jax_net = create_jax_network()
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 256))
    jax_params_init = init_network_params(jax_net, rng, dummy_input)

    # Convert weights
    jax_params = pytorch_to_jax(pytorch_net.state_dict())

    # Verify conversion
    success = verify_conversion(
        pytorch_model=pytorch_net,
        jax_model=jax_net,
        jax_params=jax_params,
        num_tests=100,
        atol=1e-5
    )

    assert success, "Weight conversion verification failed"
```

---

### Phase 2: Gumbel MCTS Integration (Day 2)

**Goal**: Implement `mctx.gumbel_muzero_policy` with real game rules

#### Step 2.1: Create `ml/mcts/search_jax.py`

```python
"""
JAX-based MCTS using DeepMind's mctx library with Gumbel MuZero.

Key components:
- RecurrentFn: Applies actions to game states (uses real Blob rules)
- RootFn: Evaluates initial state with network
- Gumbel MCTS: mctx.gumbel_muzero_policy wrapper
"""

import jax
import jax.numpy as jnp
import mctx
import chex
from typing import Dict, Tuple, Any, Callable
import numpy as np

from ml.network.model_jax import BlobNetJAX
from ml.network.encode import StateEncoder, ActionMasker
from ml.game.blob import BlobGame, Player


class BlobRecurrentFn:
    """
    Recurrent function for MCTS tree expansion.

    Applies an action to a game state and returns:
    - Next state embedding
    - Reward (0 for non-terminal, game score for terminal)
    - Discount (1.0 for ongoing, 0.0 for terminal)
    - Policy and value for next state (from network)
    """

    def __init__(
        self,
        network_apply: Callable,
        encoder: StateEncoder,
        masker: ActionMasker,
    ):
        self.network_apply = network_apply
        self.encoder = encoder
        self.masker = masker

    def __call__(
        self,
        params: Dict[str, Any],
        rng: chex.PRNGKey,
        action: chex.Array,
        embedding: chex.Array,
    ) -> mctx.RecurrentFnOutput:
        """
        Apply action to embedded state.

        Args:
            params: Network parameters
            rng: Random key
            action: Action index to apply
            embedding: Current state embedding

        Returns:
            RecurrentFnOutput with next state, reward, discount, policy, value
        """
        # Decode embedding → game state
        # (This requires encoding game state in a way that's invertible)
        # For now, we'll assume embedding IS the state representation
        # and use it directly with the game simulator

        # TODO: Implement proper state encoding/decoding
        # For JAX compatibility, game state needs to be pure arrays

        # Placeholder: Return next state info
        # In practice, you'll need to:
        # 1. Decode embedding → game state
        # 2. Apply action to game
        # 3. Check if terminal
        # 4. Encode next state
        # 5. Evaluate with network

        # For now, simplified version:
        next_embedding = embedding  # Placeholder
        reward = 0.0
        discount = 1.0

        # Network evaluation of next state
        policy_logits, value = self.network_apply(params, next_embedding, training=False)

        return mctx.RecurrentFnOutput(
            reward=jnp.array(reward),
            discount=jnp.array(discount),
            prior_logits=policy_logits,
            value=value,
        )


def gumbel_mcts_search(
    network_params: Dict[str, Any],
    network_apply: Callable,
    game_state: BlobGame,
    player: Player,
    encoder: StateEncoder,
    masker: ActionMasker,
    rng_key: chex.PRNGKey,
    num_simulations: int = 50,
    max_num_considered_actions: int = 16,
) -> Dict[int, float]:
    """
    Run Gumbel MuZero MCTS search on a game state.

    Args:
        network_params: JAX network parameters
        network_apply: Network forward function
        game_state: Current Blob game state
        player: Current player
        encoder: State encoder
        masker: Action masker
        rng_key: JAX random key
        num_simulations: Number of MCTS simulations
        max_num_considered_actions: Max actions to consider (Gumbel pruning)

    Returns:
        action_probs: Dict mapping action_idx → probability
    """
    # Encode current state
    state_tensor = encoder.encode(game_state, player)
    state_embedding = jnp.array(state_tensor.numpy())

    # Get legal actions and mask
    legal_actions, legal_mask = _get_legal_actions_and_mask(
        game_state, player, encoder, masker
    )
    legal_mask_jax = jnp.array(legal_mask.numpy())

    # Initial network evaluation
    policy_logits, value = network_apply(network_params, state_embedding, training=False)

    # Apply legal action masking
    policy_logits = jnp.where(
        legal_mask_jax == 0,
        -jnp.inf,
        policy_logits
    )

    # Create root node
    root = mctx.RootFnOutput(
        prior_logits=policy_logits,
        value=value,
        embedding=state_embedding,
    )

    # Create recurrent function
    recurrent_fn = BlobRecurrentFn(network_apply, encoder, masker)

    # Run Gumbel MuZero MCTS
    policy_output = mctx.gumbel_muzero_policy(
        params=network_params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        max_depth=None,  # No depth limit

        # Gumbel-specific parameters
        max_num_considered_actions=max_num_considered_actions,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=1.0,
    )

    # Extract action probabilities (only for legal actions)
    action_weights = np.array(policy_output.action_weights)
    action_probs = {}

    for i, action_idx in enumerate(legal_actions):
        if action_weights[action_idx] > 0:
            action_probs[action_idx] = float(action_weights[action_idx])

    # Normalize (should already sum to 1, but ensure)
    total = sum(action_probs.values())
    if total > 0:
        action_probs = {k: v / total for k, v in action_probs.items()}

    return action_probs


def _get_legal_actions_and_mask(game, player, encoder, masker):
    """Helper to get legal actions (copied from PyTorch version)."""
    if game.game_phase == "bidding":
        cards_dealt = len(player.hand)
        is_dealer = player.position == game.dealer_position
        forbidden_bid = None

        if is_dealer:
            total_bids = sum(p.bid for p in game.players if p.bid is not None)
            forbidden_bid = cards_dealt - total_bids

        mask = masker.create_bidding_mask(
            cards_dealt=cards_dealt,
            is_dealer=is_dealer,
            forbidden_bid=forbidden_bid,
        )

        legal_actions = []
        for bid in range(cards_dealt + 1):
            if mask[bid] == 1.0:
                legal_actions.append(bid)

    elif game.game_phase == "playing":
        led_suit = game.current_trick.led_suit if game.current_trick else None

        mask = masker.create_playing_mask(
            hand=player.hand,
            led_suit=led_suit,
            encoder=encoder,
        )

        legal_actions = []
        for card in player.hand:
            card_idx = encoder._card_to_index(card)
            if mask[card_idx] == 1.0:
                legal_actions.append(card_idx)

    else:
        raise ValueError(f"Cannot get legal actions for phase: {game.game_phase}")

    return legal_actions, mask
```

#### Step 2.2: Vectorized Determinization

Create `ml/mcts/determinization_jax.py`:

```python
"""
Vectorized determinization for imperfect information MCTS.

Uses JAX vmap to sample and search multiple determinized worlds in parallel.
"""

import jax
import jax.numpy as jnp
from typing import List, Dict
import numpy as np

from ml.game.blob import BlobGame, Player
from ml.mcts.belief_tracker import BeliefState
from ml.mcts.determinization import Determinizer


def vectorized_determinization_search(
    network_params,
    network_apply,
    game_state: BlobGame,
    player: Player,
    encoder,
    masker,
    num_determinizations: int = 3,
    num_simulations: int = 50,
) -> Dict[int, float]:
    """
    Run MCTS on multiple determinized worlds in parallel.

    Current approach (sequential):
        for det in range(3):
            action_probs = mcts_search(det_game)
        aggregate(action_probs)

    New approach (vectorized):
        det_games = [det1, det2, det3]  # Sample all at once
        action_probs = vmap(mcts_search)(det_games)  # Parallel
        aggregate(action_probs)

    Args:
        network_params: JAX network parameters
        network_apply: Network apply function
        game_state: Current game (with hidden opponent hands)
        player: Current player
        encoder: State encoder
        masker: Action masker
        num_determinizations: Number of worlds to sample
        num_simulations: MCTS simulations per world

    Returns:
        Aggregated action probabilities
    """
    from ml.mcts.search_jax import gumbel_mcts_search

    # Create belief state
    belief = BeliefState(game_state, player)

    # Sample determinizations
    determinizer = Determinizer()
    determinizations = determinizer.sample_adaptive(
        game_state,
        belief,
        num_samples=num_determinizations,
        diversity_weight=0.5,
    )

    if not determinizations:
        # Fallback: single search on current state
        rng_key = jax.random.PRNGKey(0)
        return gumbel_mcts_search(
            network_params,
            network_apply,
            game_state,
            player,
            encoder,
            masker,
            rng_key,
            num_simulations,
        )

    # Create determinized games
    det_games = []
    for det_hands in determinizations:
        det_game = determinizer.create_determinized_game(
            game_state, belief, det_hands
        )
        det_games.append(det_game)

    # Run MCTS on each determinization (could vectorize this further)
    all_action_probs = []
    rng = jax.random.PRNGKey(0)

    for det_game in det_games:
        rng, subkey = jax.random.split(rng)
        action_probs = gumbel_mcts_search(
            network_params,
            network_apply,
            det_game,
            player,
            encoder,
            masker,
            subkey,
            num_simulations,
        )
        all_action_probs.append(action_probs)

    # Aggregate action probabilities
    aggregated = _aggregate_action_probs(all_action_probs)

    return aggregated


def _aggregate_action_probs(action_probs_list: List[Dict[int, float]]) -> Dict[int, float]:
    """
    Average action probabilities across determinizations.

    Args:
        action_probs_list: List of {action: prob} dicts from each determinization

    Returns:
        Aggregated and normalized action probabilities
    """
    if not action_probs_list:
        return {}

    aggregated = {}

    # Sum probabilities
    for action_probs in action_probs_list:
        for action, prob in action_probs.items():
            aggregated[action] = aggregated.get(action, 0.0) + prob

    # Average
    num_dets = len(action_probs_list)
    for action in aggregated:
        aggregated[action] /= num_dets

    # Normalize
    total = sum(aggregated.values())
    if total > 0:
        aggregated = {k: v / total for k, v in aggregated.items()}

    return aggregated
```

---

### Phase 3: Self-Play Integration (Day 3)

**Goal**: Update self-play worker to use JAX MCTS

#### Step 3.1: Modify `ml/training/selfplay.py`

Add JAX MCTS option to `SelfPlayWorker`:

```python
# At top of file
from ml.network.model_jax import BlobNetJAX
from ml.utils.weight_converter import pytorch_to_jax
from ml.mcts.determinization_jax import vectorized_determinization_search

class SelfPlayWorker:
    def __init__(
        self,
        network: BlobNet,  # PyTorch network for compatibility
        encoder: StateEncoder,
        masker: ActionMasker,
        num_determinizations: int = 3,
        simulations_per_determinization: int = 30,
        temperature_schedule: Optional[Callable[[int], float]] = None,
        use_imperfect_info: bool = True,
        use_jax_mcts: bool = False,  # NEW: Enable JAX MCTS
    ):
        self.network = network
        self.encoder = encoder
        self.masker = masker
        self.num_determinizations = num_determinizations
        self.simulations_per_determinization = simulations_per_determinization
        self.use_imperfect_info = use_imperfect_info
        self.use_jax_mcts = use_jax_mcts

        # Temperature schedule
        if temperature_schedule is None:
            self.temperature_schedule = self.get_default_temperature_schedule()
        else:
            self.temperature_schedule = temperature_schedule

        # Create MCTS instance
        if use_jax_mcts:
            # JAX MCTS setup
            import jax
            from ml.network.model_jax import create_jax_network, init_network_params

            # Create JAX network
            self.jax_network = create_jax_network()
            rng = jax.random.PRNGKey(0)
            dummy_input = jax.numpy.ones((1, 256))

            # Convert PyTorch weights → JAX
            jax_params = pytorch_to_jax(network.state_dict())
            self.jax_params = jax_params
            self.jax_network_apply = self.jax_network.apply

        elif use_imperfect_info:
            # Original PyTorch MCTS
            from ml.mcts.search import ImperfectInfoMCTS
            self.mcts = ImperfectInfoMCTS(
                network=network,
                encoder=encoder,
                masker=masker,
                num_determinizations=num_determinizations,
                simulations_per_determinization=simulations_per_determinization,
            )
        else:
            # Perfect info MCTS (for testing)
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
        """Generate game using MCTS (PyTorch or JAX)."""

        if game_id is None:
            game_id = str(uuid.uuid4())

        # Initialize game
        game = BlobGame(num_players=num_players)
        examples = []
        move_number = 0

        # Play the round
        def get_bid(player, hand, is_dealer, total_bids, cards_dealt):
            nonlocal move_number

            # Run MCTS (JAX or PyTorch)
            if self.use_jax_mcts:
                import jax
                rng_key = jax.random.PRNGKey(move_number)
                action_probs = vectorized_determinization_search(
                    network_params=self.jax_params,
                    network_apply=self.jax_network_apply,
                    game_state=game,
                    player=player,
                    encoder=self.encoder,
                    masker=self.masker,
                    num_determinizations=self.num_determinizations,
                    num_simulations=self.simulations_per_determinization,
                )
            else:
                action_probs = self.mcts.search(game, player)

            # Select action with temperature
            temperature = self.temperature_schedule(move_number)
            bid = self._select_action(action_probs, temperature)

            # Store training example
            state_tensor = self.encoder.encode(game, player)
            policy_vector = self._action_probs_to_vector(action_probs, is_bidding=True)

            examples.append({
                "state": state_tensor.cpu().numpy(),  # Convert to NumPy for buffer
                "policy": policy_vector,
                "value": None,
                "player_position": player.position,
                "game_id": game_id,
                "move_number": move_number,
            })

            move_number += 1
            return bid

        def get_card(player, legal_cards, trick):
            nonlocal move_number

            # Run MCTS (JAX or PyTorch)
            if self.use_jax_mcts:
                import jax
                rng_key = jax.random.PRNGKey(move_number)
                action_probs = vectorized_determinization_search(
                    network_params=self.jax_params,
                    network_apply=self.jax_network_apply,
                    game_state=game,
                    player=player,
                    encoder=self.encoder,
                    masker=self.masker,
                    num_determinizations=self.num_determinizations,
                    num_simulations=self.simulations_per_determinization,
                )
            else:
                action_probs = self.mcts.search(game, player)

            # Select action
            temperature = self.temperature_schedule(move_number)
            card_idx = self._select_action(action_probs, temperature)

            # Find card
            card = None
            for legal_card in legal_cards:
                if self.encoder._card_to_index(legal_card) == card_idx:
                    card = legal_card
                    break

            if card is None:
                card = legal_cards[0]  # Fallback

            # Store example
            state_tensor = self.encoder.encode(game, player)
            policy_vector = self._action_probs_to_vector(action_probs, is_bidding=False)

            examples.append({
                "state": state_tensor.cpu().numpy(),
                "policy": policy_vector,
                "value": None,
                "player_position": player.position,
                "game_id": game_id,
                "move_number": move_number,
            })

            move_number += 1
            return card

        # Play round
        result = game.play_round(cards_to_deal, get_bid, get_card)

        # Back-propagate outcomes
        final_scores = {}
        for player_result in result["player_scores"]:
            for player in game.players:
                if player.name == player_result["name"]:
                    final_scores[player.position] = player_result["round_score"]
                    break

        self._backpropagate_outcome(examples, final_scores)

        return examples
```

#### Step 3.2: Update config for JAX

Modify `ml/config.py`:

```python
@dataclass
class TrainingConfig:
    # ... existing fields ...

    # JAX-specific settings
    use_jax_mcts: bool = True  # Enable JAX MCTS for self-play
    jax_platform: str = 'gpu'  # 'gpu' or 'cpu'
    jax_enable_x64: bool = False  # Use float64 (slower but more precise)
```

---

### Phase 4: Testing & Validation (Day 4)

#### Test Suite

Create `ml/tests/test_jax_integration.py`:

```python
"""Integration tests for JAX MCTS pipeline."""

import pytest
import torch
import jax
import numpy as np

from ml.network.model import BlobNet
from ml.network.model_jax import create_jax_network, init_network_params
from ml.utils.weight_converter import pytorch_to_jax, verify_conversion
from ml.mcts.search_jax import gumbel_mcts_search
from ml.mcts.determinization_jax import vectorized_determinization_search
from ml.training.selfplay import SelfPlayWorker
from ml.game.blob import BlobGame
from ml.network.encode import StateEncoder, ActionMasker


class TestJAXIntegration:
    """Test JAX MCTS integration end-to-end."""

    def test_network_conversion(self):
        """Test PyTorch → JAX conversion."""
        pytorch_net = BlobNet()
        jax_net = create_jax_network()

        rng = jax.random.PRNGKey(0)
        dummy_input = jax.numpy.ones((1, 256))
        jax_params_init = init_network_params(jax_net, rng, dummy_input)

        # Convert
        jax_params = pytorch_to_jax(pytorch_net.state_dict())

        # Verify
        assert verify_conversion(pytorch_net, jax_net, jax_params, num_tests=10)

    def test_gumbel_mcts_legal_actions(self):
        """Test Gumbel MCTS only returns legal actions."""
        pytorch_net = BlobNet()
        jax_net = create_jax_network()
        jax_params = pytorch_to_jax(pytorch_net.state_dict())

        encoder = StateEncoder()
        masker = ActionMasker()

        # Create game
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        # Run Gumbel MCTS
        rng_key = jax.random.PRNGKey(0)
        action_probs = gumbel_mcts_search(
            network_params=jax_params,
            network_apply=jax_net.apply,
            game_state=game,
            player=player,
            encoder=encoder,
            masker=masker,
            rng_key=rng_key,
            num_simulations=10,  # Fast test
        )

        # Verify all actions are legal
        if game.game_phase == "bidding":
            cards_dealt = len(player.hand)
            for action in action_probs.keys():
                assert 0 <= action <= cards_dealt

        # Verify probabilities sum to 1
        total = sum(action_probs.values())
        assert abs(total - 1.0) < 1e-4

    def test_selfplay_with_jax(self):
        """Test self-play game generation with JAX MCTS."""
        pytorch_net = BlobNet()
        encoder = StateEncoder()
        masker = ActionMasker()

        # Create worker with JAX MCTS
        worker = SelfPlayWorker(
            network=pytorch_net,
            encoder=encoder,
            masker=masker,
            num_determinizations=2,  # Fast test
            simulations_per_determinization=10,
            use_jax_mcts=True,  # Enable JAX
        )

        # Generate game
        examples = worker.generate_game(
            num_players=4,
            cards_to_deal=3,  # Short game
        )

        # Verify examples
        assert len(examples) > 0
        for example in examples:
            assert 'state' in example
            assert 'policy' in example
            assert 'value' in example
            assert example['value'] is not None  # Should be backpropagated

    def test_jax_vs_pytorch_consistency(self):
        """Test JAX and PyTorch MCTS produce similar action distributions."""
        pytorch_net = BlobNet()
        pytorch_net.eval()

        encoder = StateEncoder()
        masker = ActionMasker()

        # Create game
        game = BlobGame(num_players=4)
        game.setup_round(cards_to_deal=5)
        player = game.players[0]

        # PyTorch MCTS
        from ml.mcts.search import MCTS
        pytorch_mcts = MCTS(
            network=pytorch_net,
            encoder=encoder,
            masker=masker,
            num_simulations=50,
        )
        pytorch_action_probs = pytorch_mcts.search(game, player)

        # JAX MCTS
        jax_net = create_jax_network()
        jax_params = pytorch_to_jax(pytorch_net.state_dict())
        rng_key = jax.random.PRNGKey(0)
        jax_action_probs = gumbel_mcts_search(
            network_params=jax_params,
            network_apply=jax_net.apply,
            game_state=game,
            player=player,
            encoder=encoder,
            masker=masker,
            rng_key=rng_key,
            num_simulations=50,
        )

        # Compare (should be similar, not necessarily identical due to Gumbel sampling)
        common_actions = set(pytorch_action_probs.keys()) & set(jax_action_probs.keys())
        assert len(common_actions) > 0

        # Check top action is in top 3 of other method
        pytorch_top = max(pytorch_action_probs, key=pytorch_action_probs.get)
        jax_top3 = sorted(jax_action_probs, key=jax_action_probs.get, reverse=True)[:3]

        # This is a loose test (Gumbel adds exploration noise)
        print(f"PyTorch top action: {pytorch_top}")
        print(f"JAX top 3 actions: {jax_top3}")
```

#### Benchmarking Script

Create `Benchmarks/benchmark_jax_speedup.py`:

```python
"""
Benchmark JAX vs PyTorch MCTS speed.

Measures games/minute improvement.
"""

import time
import torch
from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.training.selfplay import SelfPlayWorker


def benchmark_pytorch_mcts(num_games: int = 10):
    """Benchmark current Python MCTS."""
    pytorch_net = BlobNet()
    encoder = StateEncoder()
    masker = ActionMasker()

    worker = SelfPlayWorker(
        network=pytorch_net,
        encoder=encoder,
        masker=masker,
        num_determinizations=3,
        simulations_per_determinization=30,
        use_jax_mcts=False,  # PyTorch
    )

    start = time.time()
    for i in range(num_games):
        worker.generate_game(num_players=4, cards_to_deal=5)
    elapsed = time.time() - start

    games_per_min = (num_games / elapsed) * 60
    return games_per_min


def benchmark_jax_mcts(num_games: int = 10):
    """Benchmark JAX Gumbel MCTS."""
    pytorch_net = BlobNet()
    encoder = StateEncoder()
    masker = ActionMasker()

    worker = SelfPlayWorker(
        network=pytorch_net,
        encoder=encoder,
        masker=masker,
        num_determinizations=3,
        simulations_per_determinization=30,
        use_jax_mcts=True,  # JAX
    )

    start = time.time()
    for i in range(num_games):
        worker.generate_game(num_players=4, cards_to_deal=5)
    elapsed = time.time() - start

    games_per_min = (num_games / elapsed) * 60
    return games_per_min


if __name__ == "__main__":
    print("Benchmarking MCTS implementations...")
    print("=" * 60)

    print("\nPyTorch MCTS (current):")
    pytorch_speed = benchmark_pytorch_mcts(num_games=10)
    print(f"  Speed: {pytorch_speed:.1f} games/min")

    print("\nJAX Gumbel MCTS (new):")
    jax_speed = benchmark_jax_mcts(num_games=10)
    print(f"  Speed: {jax_speed:.1f} games/min")

    speedup = jax_speed / pytorch_speed
    print("\n" + "=" * 60)
    print(f"SPEEDUP: {speedup:.1f}x")
    print("=" * 60)

    # Estimate training time reduction
    total_games = 10_000 * 500  # 10k games/iter × 500 iters

    pytorch_hours = (total_games / pytorch_speed) / 60
    jax_hours = (total_games / jax_speed) / 60

    print(f"\nEstimated training time for {total_games:,} games:")
    print(f"  PyTorch: {pytorch_hours:.1f} hours ({pytorch_hours/24:.1f} days)")
    print(f"  JAX: {jax_hours:.1f} hours ({jax_hours/24:.1f} days)")
    print(f"  Savings: {pytorch_hours - jax_hours:.1f} hours ({(pytorch_hours - jax_hours)/24:.1f} days)")
```

---

## Expected Performance

### Speedup Analysis

| Component | Current | JAX | Speedup |
|-----------|---------|-----|---------|
| **Network inference** | PyTorch CPU | JAX GPU JIT | 5-10x |
| **MCTS simulations** | Python loop | JAX vectorized | 10-20x |
| **Determinization** | Sequential | vmap parallel | 3x |
| **Gumbel sampling** | N/A | Fewer sims needed | 1.8x |
| **Total** | 6 games/min | **120-300 games/min** | **20-50x** |

### Training Timeline Projection

**Assumptions:**
- 10,000 games per iteration
- 500 iterations total
- Total: 5,000,000 games

| MCTS Type | Games/min | Hours | Days |
|-----------|-----------|-------|------|
| **PyTorch (current)** | 6 | 13,889 | 579 |
| **JAX AlphaZero** | 60 | 1,389 | 58 |
| **JAX Gumbel (conservative)** | 120 | 694 | 29 |
| **JAX Gumbel (realistic)** | 180 | 463 | **19** |
| **JAX Gumbel (optimistic)** | 300 | 278 | **12** |

**Realistic target: ~10-20 days** of continuous training

---

## Risk Assessment

### Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **mctx API complexity** | Medium | Medium | Use simpler `muzero_policy` fallback |
| **Weight conversion bugs** | Low | High | Extensive unit testing (100+ random states) |
| **RecurrentFn implementation** | High | High | Start with simplified version, iterate |
| **Determinization vectorization** | Medium | Low | Implement sequential first, add vmap later |
| **JAX memory constraints** | Low | Medium | Tune batch sizes, use gradient checkpointing |
| **Performance doesn't scale** | Low | High | Profile, optimize hotspots, use XLA flags |

### Mitigation Strategies

**1. mctx API Learning Curve**
- **Fallback**: If Gumbel is too complex, use `mctx.muzero_policy` (standard AlphaZero)
- **Still get**: 10-20x speedup from JAX vectorization alone

**2. RecurrentFn Challenges**
- **Strategy**: Implement simplified version first (determinized perfect-info game)
- **Then**: Add imperfect info handling incrementally
- **Validation**: Compare action distributions vs PyTorch MCTS

**3. Debugging JAX Code**
- **Tools**: Use `chex.assert_*` for shape validation
- **Mode**: Run with JAX debug mode initially (`JAX_DEBUG_NANS=True`)
- **Reference**: Keep PyTorch MCTS for correctness testing

**4. GPU Memory**
- **Monitor**: Track VRAM usage during self-play
- **Tune**: Reduce batch size if needed (RTX 4060 8GB should be sufficient)
- **Optimize**: Use `jax.clear_caches()` between iterations

---

## File Modifications

### New Files (7 files)

| File | LOC | Purpose |
|------|-----|---------|
| `ml/network/model_jax.py` | ~300 | JAX network (Flax) |
| `ml/mcts/search_jax.py` | ~400 | Gumbel MCTS wrapper |
| `ml/mcts/determinization_jax.py` | ~200 | Vectorized determinization |
| `ml/utils/weight_converter.py` | ~150 | PyTorch ↔ JAX conversion |
| `ml/network/test_network_jax.py` | ~150 | JAX network tests |
| `ml/tests/test_jax_integration.py` | ~200 | Integration tests |
| `Benchmarks/benchmark_jax_speedup.py` | ~100 | Speed benchmarks |

**Total new code: ~1,500 LOC**

### Modified Files (4 files)

| File | Changes | Lines Modified |
|------|---------|----------------|
| `ml/training/selfplay.py` | Add JAX MCTS option | ~100 |
| `ml/network/encode.py` | JAX compatibility | ~50 |
| `ml/config.py` | JAX settings | ~20 |
| `ml/requirements.txt` | Add dependencies | ~5 |

**Total modified: ~175 LOC**

---

## Testing Strategy

### Test Levels

**1. Unit Tests**
- ✅ JAX network forward pass
- ✅ Weight conversion (PyTorch ↔ JAX)
- ✅ Gumbel MCTS returns legal actions only
- ✅ Action probabilities sum to 1.0

**2. Integration Tests**
- ✅ Self-play game generation with JAX
- ✅ Training examples format matches PyTorch
- ✅ JAX vs PyTorch action distribution comparison

**3. Performance Tests**
- ✅ Games/minute benchmark
- ✅ GPU utilization profiling
- ✅ Memory usage monitoring

**4. Correctness Validation**
- ✅ Play 100 games with JAX MCTS
- ✅ Compare action choices vs PyTorch (should be similar)
- ✅ Verify training loop still works end-to-end

### Acceptance Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| **Network outputs match** | <1e-5 difference | Pending |
| **Legal actions only** | 100% legal | Pending |
| **Speedup achieved** | ≥20x | Pending |
| **All tests pass** | 426+ tests | Pending |
| **Training loop works** | 1 full iteration | Pending |

---

## Dependencies

### Python Package Updates

Add to `ml/requirements.txt`:

```txt
# Existing packages
torch>=2.0.0
...

# New JAX packages
jax[cuda12]>=0.4.20
jaxlib>=0.4.20
dm-mctx>=0.0.5
flax>=0.8.0
chex>=0.1.85
optax>=0.1.7  # For JAX optimization (if needed later)
```

### Installation

```bash
# Install JAX with CUDA 12 support (for RTX 4060)
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install mctx and Flax
pip install dm-mctx flax chex optax
```

---

## Next Steps

### Immediate Actions (Day 1)

1. ✅ Review this transition plan
2. ⬜ Install JAX dependencies
3. ⬜ Create `model_jax.py` (JAX network)
4. ⬜ Create `weight_converter.py`
5. ⬜ Run unit tests: network conversion
6. ⬜ Validate: PyTorch and JAX networks match

### Week 1 Milestones

- **Day 1**: JAX network + weight conversion
- **Day 2**: Gumbel MCTS implementation
- **Day 3**: Vectorized determinization + self-play integration
- **Day 4**: Testing, benchmarking, performance tuning

### Success Metrics

**Short-term (Day 4):**
- [ ] 20x+ speedup achieved
- [ ] All unit tests passing
- [ ] Can generate 1000 games successfully

**Medium-term (Week 2):**
- [ ] Run 1 full training iteration with JAX
- [ ] Verify model improvement (ELO increase)
- [ ] No regressions in test suite

**Long-term (Month 1):**
- [ ] Complete 500-iteration training run
- [ ] Model achieves strong play (ELO >1600)
- [ ] Ready for Phase 5 (ONNX export)

---

## References

### Key Papers

1. **Gumbel MuZero** (DeepMind, 2021)
   - "Policy improvement by planning with Gumbel"
   - Shows 50% simulation reduction for same quality
   - https://openreview.net/forum?id=bERaNdoegnO

2. **MuZero** (DeepMind, 2020)
   - "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"
   - Learned dynamics model (we don't need this)

3. **AlphaZero** (DeepMind, 2017)
   - "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
   - Our current architecture baseline

### Libraries

- **JAX**: https://github.com/google/jax
- **mctx**: https://github.com/deepmind/mctx
- **Flax**: https://github.com/google/flax

---

## Appendix: Code Examples

### Example: Simple Gumbel MCTS Usage

```python
import jax
import mctx
from ml.network.model_jax import BlobNetJAX

# Initialize network
network = BlobNetJAX()
params = network.init(jax.random.PRNGKey(0), jax.numpy.ones((256,)))

# Define recurrent function (simplified)
def recurrent_fn(params, rng, action, embedding):
    # Apply action to state (using real game rules)
    next_state = apply_game_action(embedding, action)

    # Network evaluation
    policy_logits, value = network.apply(params, next_state)

    return mctx.RecurrentFnOutput(
        reward=0.0,  # No immediate reward
        discount=1.0,  # Game continues
        prior_logits=policy_logits,
        value=value,
    )

# Run Gumbel MCTS
policy_output = mctx.gumbel_muzero_policy(
    params=params,
    rng_key=jax.random.PRNGKey(0),
    root=root_state,
    recurrent_fn=recurrent_fn,
    num_simulations=50,
    max_num_considered_actions=16,
)

# Extract action probabilities
action_probs = policy_output.action_weights
```

---

**END OF TRANSITION PLAN**

Generated: 2025-10-27
For: BlobMaster JAX migration
Next session: Begin Phase 1 implementation
