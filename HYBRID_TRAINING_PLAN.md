# Hybrid Training Plan - Implementation Guide

**Goal**: Train Blob AI optimized for 5-player/7-card games using hybrid curriculum approach

**Total Implementation**: 20 hours (5 sessions × 4 hours)
**Total Training Time**: 4.6-14.6 days (vs 48 days full-game-only or 5.3 days narrow 5-card-only)

**⚠️ CRITICAL - Current State**: This plan is **documentation-only**. The codebase currently:
- ❌ Plays fixed (5p, 5c) rounds only - no (P,C,c) sampling
- ❌ No game context encoding (scores, round position) - encoder uses 202 dims, not 256
- ❌ No full-game evaluation - arena only plays single rounds
- ❌ No MCTS curriculum - fixed 3×30 throughout training

**Session 0 is PREREQUISITE** - builds scaffolding for Sessions 1-4 to populate.

---

## Terminology

**IMPORTANT:** This document uses specific terminology consistently:

- **Round**: A single bidding + trick-taking cycle with a fixed number of cards dealt (e.g., one 5-card round)
- **Game**: ALWAYS refers to a complete Blob game with the full sequence of rounds (e.g., 17 rounds for 5p/C=7)

**Notation (P,C,c):**
- **P** = Number of players (e.g., 4, 5, or 6 players)
- **C** = Starting card count / Maximum cards dealt (e.g., C=7 or C=8)
- **c** = Current round's card count (e.g., c=1, c=2, ... c=7)
- Example: "5p/C=7" means 5 players starting with 7 cards, playing rounds: 7→6→5→4→3→2→1→1→1→1→1→2→3→4→5→6→7

**Throughput Metrics:**
- **rounds/min**: Independent single-round throughput (Phase 1: ~360 rounds/min with optimized MCTS curriculum)
- **games/min**: Full multi-round game throughput (Phase 2: ~73 games/min, varies with MCTS settings)

**Training Phases:**
- **Phase 1**: Independent rounds with decision-weighted sampling and synthetic game context
- **Phase 2**: Full multi-round game sequences with natural game context

---

## Core Strategy

### The Problem
- Current: Trains on fixed 5-card rounds only → potentially catastrophic failure on 1-3 card rounds
- Actual games vary by player count:
  - 5p/C=7: 7,6,5,4,3,2,1,1,1,1,1,2,3,4,5,6,7 (47% of rounds are 1c)
  - 4p/C=8: 8,7,6,5,4,3,2,1,1,1,1,2,3,4,5,6,7,8 (28% are 1c)
  - But decision points ≠ round counts: 7c has P×(7+1) decisions, 1c has P×(1+1)
- All-or-nothing scoring: 1-card round = 11 pts max, 7-card round = 17 pts max (difference minimal compared to complexity increase)

### The Solution: Hybrid Curriculum with Decision-Weighted Sampling
**Phase 1** (400-500 iter, ~4.6-5.8 days): Independent rounds + game context → fast foundation
**Phase 2** (100 iter, ~10 days, optional): Full games → multi-round polish

**Key innovations vs naive approaches**:
1. **Decision-weighted sampling**: Weight card counts by `rounds × P × (c+1)`, not arbitrary percentages
   - 5p/C=7: 7c gets 21.1% (not 50% by round-count or 35% by guess)
   - 1c gets 13.2% despite being "trivial" (P repeats = high decision contribution)
2. **(P,C)-conditional**: 4p can start C=7 or C=8; sampling respects real game structure
3. **Game context encoding**: 29 dims of cumulative scores, round position, trajectory → solves reward alignment
4. **Synthetic context generation**: Early/mid/late game scenarios for independent rounds
5. **Evaluation on full games**: Extend `ml/evaluation/arena.py` (not custom scripts)

---

## Implementation Roadmap

**Session 0 (4h) - PREREQUISITE**: Build core scaffolding
- Add distribution parameters to `TrainingConfig`
- Implement `sample_game_config()` sampler with decision weights
- Wire `game_context` parameter through `StateEncoder.encode()` (54 dims reserved)
- Add MCTS curriculum schedule `get_mcts_params(iteration)`
- Stub `ModelArena(full_game_mode=True)` for Session 4

**Session 1 (4h)**: Populate context encoding
- Fill 29 context dims in encoder (scores, round position, previous cards, config)
- Implement `generate_synthetic_context()` for early/mid/late game scenarios

**Session 2 (4h)**: Wire into training loop
- Call `sample_game_config()` in `SelfPlayWorker.play_round()`
- Thread `game_context` through MCTS and replay buffer
- Update all `encoder.encode()` call sites to pass context

**Session 3 (4h)**: MCTS curriculum & CLI
- Wire `config.get_mcts_params(iteration)` into `TrainingPipeline`
- Add `--training-on`, `--iterations` CLI args to `train.py`
- Validate end-to-end pipeline with `--fast` mode

**Session 4 (4h) - OPTIONAL**: Full games & evaluation
- Implement `FullGameGenerator` for P-conditional sequences
- Remove `NotImplementedError` from `ModelArena(full_game_mode=True)`
- Support Phase 2 training and full-game ELO evaluation

---

## Implementation Sessions

### Session 0: Core Infrastructure (4 hours) [PREREQUISITE]

**Goal**: Build minimal scaffolding for (P,C,c) sampling, context passing, and full-game evaluation

**Current state**: Code plays fixed (5p, 5c) rounds only. No infrastructure for:
- Sampling player count, start cards, or card count by distribution
- Passing game context (scores, round index) to encoder
- Full-game evaluation with P-conditional sequences

**Files to modify**:
- `ml/config.py` - Add distribution parameters and schedule hooks
- `ml/training/selfplay.py` - Add minimal sampler and context struct
- `ml/network/encode.py` - Add optional `game_context` parameter (no features yet, just wiring)
- `ml/evaluation/arena.py` - Add `full_game_mode` stub (returns NotImplementedError for now)

**Changes to ml/config.py**:
```python
@dataclass
class TrainingConfig:
    # ... existing fields ...

    # NEW: Player distribution (matches your table)
    player_distribution: dict = field(default_factory=lambda: {
        4: 0.15,  # 15% - occasional
        5: 0.70,  # 70% - YOUR STANDARD
        6: 0.15,  # 15% - occasional
    })

    # NEW: 4p starting cards split
    start_card_distribution_4p: dict = field(default_factory=lambda: {
        7: 0.60,  # 60% - standard
        8: 0.40,  # 40% - when fewer players
    })

    # NEW: Use decision-weighted sampling vs fixed (5p, 5c)
    use_decision_weighted_sampling: bool = True

    # NEW: Training mode - controls what games_per_iteration counts
    # TODO: Add this parameter to ml/config.py
    training_on: str = "rounds"  # "rounds" (Phase 1) or "games" (Phase 2)
    # - "rounds": games_per_iteration counts independent rounds (~360 rounds/min)
    # - "games": games_per_iteration counts full multi-round games (~73 games/min)

    # NEW: MCTS curriculum schedule (iteration → (det, sims))
    mcts_schedule: dict = field(default_factory=lambda: {
        50: (1, 15),
        150: (2, 25),
        300: (3, 35),
        450: (4, 45),
        500: (5, 50),
    })

    def get_mcts_params(self, iteration: int) -> tuple[int, int]:
        """Return (num_determinizations, simulations_per_det) for iteration."""
        for threshold, (det, sims) in sorted(self.mcts_schedule.items()):
            if iteration <= threshold:
                return (det, sims)
        # Default to highest if beyond schedule
        return (5, 50)
```

**Changes to ml/training/selfplay.py** (minimal sampler):
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class GameContext:
    """Synthetic game context for independent rounds."""
    cumulative_scores: list[int]
    rounds_completed: int
    total_rounds: int
    previous_cards: list[int]
    num_players: int
    start_cards: int
    phase: str  # 'descending', 'ones', 'ascending'

# Precomputed decision weights (computed once at module load)
DECISION_WEIGHTS = {
    (5, 7): {1: 0.132, 2: 0.079, 3: 0.105, 4: 0.132, 5: 0.158, 6: 0.184, 7: 0.211},
    (4, 8): {1: 0.091, 2: 0.061, 3: 0.073, 4: 0.091, 5: 0.110, 6: 0.128, 7: 0.146, 8: 0.300},
    (4, 7): {1: 0.125, 2: 0.083, 3: 0.111, 4: 0.139, 5: 0.167, 6: 0.194, 7: 0.181},
    (6, 7): {1: 0.130, 2: 0.078, 3: 0.104, 4: 0.130, 5: 0.156, 6: 0.182, 7: 0.220},
}

class SelfPlayWorker:
    # ... existing methods ...

    def sample_game_config(self) -> tuple[int, int, int, Optional[GameContext]]:
        """Sample (num_players, start_cards, num_cards, context) using decision weights.

        Returns:
            (num_players, start_cards, num_cards, game_context or None)
        """
        if not self.config.use_decision_weighted_sampling:
            # Fallback to fixed config
            return (self.config.num_players, self.config.cards_to_deal,
                    self.config.cards_to_deal, None)

        # 1. Sample player count
        players = list(self.config.player_distribution.keys())
        probs = [self.config.player_distribution[p] for p in players]
        num_players = np.random.choice(players, p=probs)

        # 2. Sample starting cards (conditional on P)
        if num_players == 4:
            start_cards = np.random.choice(
                [7, 8],
                p=[self.config.start_card_distribution_4p[7],
                   self.config.start_card_distribution_4p[8]]
            )
        else:
            start_cards = 7  # 5p/6p always C=7

        # 3. Sample card count by decision weights
        weights = DECISION_WEIGHTS.get((num_players, start_cards))
        if weights is None:
            # Fallback if combination not precomputed
            num_cards = start_cards
        else:
            card_sizes = list(range(1, start_cards + 1))
            probs = [weights[c] for c in card_sizes]
            num_cards = np.random.choice(card_sizes, p=probs)

        # 4. Generate synthetic context (stub for now - Session 1 fills this in)
        context = None  # TODO Session 1: generate_synthetic_context(num_players, num_cards, start_cards)

        return (num_players, start_cards, num_cards, context)
```

**Changes to ml/network/encode.py** (wiring only, no features yet):
```python
class StateEncoder:
    def encode(
        self,
        game: BlobGame,
        player: int,
        game_context: Optional[GameContext] = None  # NEW: optional context
    ) -> torch.Tensor:
        """Encode game state to 256-dim tensor.

        Args:
            game: BlobGame instance
            player: Player index
            game_context: Optional game context (scores, round index, etc.)
        """
        # ... existing 202-dim encoding ...

        # NEW: Reserve 54 dims for game context (Session 1 populates these)
        if game_context is not None:
            context_features = torch.zeros(54, dtype=torch.float32)
            # TODO Session 1: populate context_features from game_context
        else:
            context_features = torch.zeros(54, dtype=torch.float32)

        # Concatenate: [existing 202 dims] + [context 54 dims] = 256 dims
        return torch.cat([features, context_features])
```

**Changes to ml/evaluation/arena.py** (stub only):
```python
class ModelArena:
    def __init__(self, config: TrainingConfig, full_game_mode: bool = False):
        self.config = config
        self.full_game_mode = full_game_mode

        if full_game_mode:
            raise NotImplementedError(
                "Full-game evaluation not yet implemented. "
                "Session 4 will add P-conditional sequences and total-game scoring."
            )
```

**Testing**:
```bash
# Verify sampling works (use --seed for reproducibility in tests)
python -c "
import numpy as np
np.random.seed(42)  # Deterministic for testing

from ml.training.selfplay import SelfPlayWorker
from ml.config import TrainingConfig

config = TrainingConfig(use_decision_weighted_sampling=True)
worker = SelfPlayWorker(config, model=None, worker_id=0)

# Sample 1000 games
samples = [worker.sample_game_config() for _ in range(1000)]
players = [s[0] for s in samples]
cards = [s[2] for s in samples]

# Validate player distribution
print(f'5p: {players.count(5)/10:.1f}% (target 70%)')
print(f'4p: {players.count(4)/10:.1f}% (target 15%)')
print(f'6p: {players.count(6)/10:.1f}% (target 15%)')

# Validate 4p start-card split
four_p_samples = [(s[1], s[2]) for s in samples if s[0] == 4]
c7_count = sum(1 for start_c, _ in four_p_samples if start_c == 7)
c8_count = sum(1 for start_c, _ in four_p_samples if start_c == 8)
print(f'4p C=7: {c7_count/len(four_p_samples)*100:.1f}% (target 60%)')
print(f'4p C=8: {c8_count/len(four_p_samples)*100:.1f}% (target 40%)')
"

# Verify MCTS schedule
python -c "
from ml.config import TrainingConfig

config = TrainingConfig()
print('MCTS schedule:')
for i in [1, 50, 100, 250, 400, 500]:
    det, sims = config.get_mcts_params(i)
    print(f'  Iter {i}: {det} det × {sims} sims')
"

# TODO: Verify training_on parameter validation
python -c "
from ml.config import TrainingConfig

# Test default value
config = TrainingConfig()
assert config.training_on == 'rounds', f'Expected default \"rounds\", got {config.training_on}'
print(f'Default training_on: {config.training_on} ✓')

# Test valid values
config.training_on = 'games'
assert config.training_on == 'games'
print(f'training_on=\"games\" accepted ✓')

# TODO: Add validation in config that raises ValueError for invalid values
# try:
#     config.training_on = 'invalid'
#     assert False, 'Should have raised ValueError'
# except ValueError:
#     print('Invalid training_on rejected ✓')
"

# Verify encoder accepts context parameter (returns zeros for now)
python -c "
from ml.network.encode import StateEncoder
from ml.game.blob import BlobGame

encoder = StateEncoder()
game = BlobGame(num_players=5, cards_to_deal=7)
game.start_game()

encoding = encoder.encode(game, player=0, game_context=None)
assert encoding.shape == (256,), f'Expected 256 dims, got {encoding.shape}'
print(f'Encoding shape: {encoding.shape} ✓')
print(f'Context dims [202:256]: all zeros = {(encoding[202:] == 0).all()} ✓')
"
```

**Acceptance criteria**:
- ✅ `TrainingConfig` has `player_distribution`, `start_card_distribution_4p`, `use_decision_weighted_sampling`
- ✅ `TrainingConfig` has `training_on` parameter (default "rounds", accepts "rounds" or "games")
- ✅ `SelfPlayWorker.sample_game_config()` returns (P, C, c, context) sampled by decision weights
- ✅ Distribution tests pass: 70% 5p ±5%, 15% 4p ±5%, 15% 6p ±5% over 1000 samples
- ✅ `StateEncoder.encode()` accepts optional `game_context` parameter
- ✅ `StateEncoder.encode()` returns 256-dim tensor (202 existing + 54 zeros for context)
- ✅ `ModelArena(full_game_mode=True)` raises NotImplementedError with helpful message
- ✅ MCTS schedule returns correct (det, sims) for each iteration range

**Why this session is critical**:
Without this infrastructure, Sessions 1-4 cannot work:
- Session 1 needs `game_context` parameter to populate
- Session 2 needs `sample_game_config()` to call
- Session 3 needs `get_mcts_params()` for curriculum
- Session 4 needs `full_game_mode` hook in arena

---

### Session 1: Game Context Encoding (4 hours)

**Goal**: Populate the 54 context dims in StateEncoder (wired in Session 0) and generate synthetic context

**Prerequisite**: Session 0 complete - `StateEncoder.encode()` accepts `game_context` parameter

**Files to modify**:
- `ml/network/encode.py` - StateEncoder.encode() (populate context_features)
- `ml/training/selfplay.py` - Add `generate_synthetic_context()` function

**Changes to StateEncoder (uses 54 spare dims)**:
```python
# Existing: 202 dims (hand, trick history, bids, tricks won, belief state)
# Add context features (54 dims total, slotted at end → no reshape):

# 1. Per-player cumulative scores (8 slots, normalized)
#    scores[i] / (rounds_completed × (10 + start_cards))
#    Example: 25 pts after 3 rounds in 5p/C=7 → 25/(3×17) = 0.49
#    IMPORTANT: No hard clamp at 1.0! Preserves signal when player far ahead late-game

# 2. Round position (4 slots)
#    - rounds_completed / total_rounds (e.g., 5/17 = 0.29)
#    - remaining_rounds / total_rounds (e.g., 12/17 = 0.71)
#    - num_1card_rounds_remaining / P (e.g., 3/5 = 0.60)
#    - phase: 0=descending, 0.5=in 1s, 1=ascending

# 3. Previous round cards (max 15 slots, normalized by start_cards)
#    Store MOST RECENT 15 rounds (truncate older, pad with 0 if <15)
#    Example: 5p/C=7 game, rounds [7,6,5,4,3,2,1,1,1,1,1,2,3,4,5,6,7] → last 15
#    [1,1,1,1,2,3,4,5,6,7] / 7 → [0.143, 0.143, ..., 1.0]
#    Example: 4p/C=8 game → normalize by 8 instead
#    IMPORTANT: Recent context more valuable; use start_cards for scaling consistency

# 4. Game config (2 slots)
#    - num_players / 8 (e.g., 5/8 = 0.625)
#    - start_cards / 8 (e.g., 7/8 = 0.875)

# Total: 8 + 4 + 15 + 2 = 29 dims used, 25 reserved for future
```

**Context generation for independent rounds**:
```python
def generate_synthetic_context(num_players, num_cards):
    """Create realistic early/mid/late game context for independent rounds."""
    # Sample game phase: 30% early, 40% mid, 30% late
    phase = np.random.choice(['early', 'mid', 'late'], p=[0.3, 0.4, 0.3])

    # Generate consistent trajectory for (P, C)
    start_C = 7 if num_players in [5,6] else (8 if np.random.rand()<0.4 else 7)
    total_rounds = 2*start_C - 1 + num_players  # e.g., 5p/C=7 → 17 rounds

    if phase == 'early':
        rounds_done = np.random.randint(0, total_rounds//3)
    elif phase == 'mid':
        rounds_done = np.random.randint(total_rounds//3, 2*total_rounds//3)
    else:
        rounds_done = np.random.randint(2*total_rounds//3, total_rounds)

    # Generate scores consistent with rounds_done and all-or-nothing scoring
    scores = [np.random.randint(0, rounds_done * 17) for _ in range(num_players)]

    # Build previous round sequence consistent with (P, C, rounds_done)
    full_seq = list(range(start_C, 0, -1)) + [1]*num_players + list(range(2, start_C+1))
    prev_cards = full_seq[:rounds_done] if rounds_done > 0 else []

    return {
        'scores': scores,
        'rounds_completed': rounds_done,
        'total_rounds': total_rounds,
        'previous_cards': prev_cards,
        'num_players': num_players,
        'start_cards': start_C
    }
```

**Testing**:
```bash
python -m pytest ml/network/test_network.py::TestStateEncoder::test_game_context_encoding
```

**Acceptance**: StateEncoder.encode() returns 256-dim tensor including game context, all tests pass

---

### Session 2: Wire Sampling Into Training Pipeline (4 hours)

**Goal**: Connect Session 0+1 pieces into actual training loop

**Prerequisite**: Session 0+1 complete - `sample_game_config()` and `generate_synthetic_context()` exist

**Note on training_on parameter**: This session implements `training_on="rounds"` mode (Phase 1). Session 4 implements `training_on="games"` mode (Phase 2).

**Files to modify**:
- `ml/training/selfplay.py` - SelfPlayWorker.play_round() (call sample_game_config)
- `ml/training/trainer.py` - Pass sampled config through pipeline
- All encoder call sites - Pass `game_context` parameter

**TODO**: When implementing, add conditional logic based on `config.training_on`:
- `training_on="rounds"`: Use `play_round()` method (this session)
- `training_on="games"`: Use `FullGameGenerator` (Session 4)

**Wire sampling into training loop**:
```python
# In ml/training/selfplay.py - SelfPlayWorker.play_round()

def play_round(self):
    """Play a single round and return training positions."""
    # Sample game configuration (Session 0 implemented this)
    num_players, start_cards, num_cards, game_context = self.sample_game_config()

    # Create game with sampled parameters
    game = BlobGame(num_players=num_players, cards_to_deal=num_cards)
    game.start_game()

    positions = []
    while not game.is_terminal():
        current_player = game.current_player

        # Get MCTS policy (pass context to encoder inside MCTS)
        mcts_policy = self.get_mcts_policy(game, game_context)

        # Store position for training
        positions.append({
            'state': game.copy(),
            'player': current_player,
            'policy': mcts_policy,
            'game_context': game_context,  # NEW: include context
        })

        # Sample action from policy and apply
        action = sample_action(mcts_policy)
        game.apply_action(action)

    # Add final scores to all positions
    final_scores = game.get_scores()
    for pos in positions:
        pos['value'] = final_scores[pos['player']]

    return positions


# In ml/mcts/search.py - Pass context through to encoder

def evaluate_leaf(node, game, game_context):
    """Evaluate leaf node using neural network."""
    state_encoding = self.encoder.encode(game, node.player, game_context)  # NEW: pass context
    policy_logits, value = self.network(state_encoding)
    return policy_logits, value
```

**Update all encoder call sites** (CRITICAL for correctness):
```python
# IMPORTANT: ALL encoder invocations must pass game_context, including:
# - Top-level in SelfPlayWorker.play_round() ✓
# - Inside MCTS search paths (ml/mcts/search.py - every leaf evaluation) ✓
# - Training loop batch encoding (ml/training/trainer.py) ✓
# - Evaluation/arena (ml/evaluation/arena.py) ✓
# - Test files that call encode() ✓

# The context must thread through:
#   SelfPlayWorker → MCTS.search() → Node.evaluate_leaf() → encoder.encode()
#
# Example threading pattern:

# In SelfPlayWorker.get_mcts_policy():
def get_mcts_policy(self, game, game_context):
    # Pass context into MCTS search
    return self.mcts.search(game, game_context)

# In MCTSSearch.search():
def search(self, game, game_context):
    for _ in range(self.num_simulations):
        node = self.select(root)
        if node.is_leaf():
            # Pass context to leaf evaluation
            policy, value = self.evaluate_leaf(node, game, game_context)

# In MCTSSearch.evaluate_leaf():
def evaluate_leaf(self, node, game, game_context):
    # Finally reaches encoder with context
    encoding = self.encoder.encode(game, node.player, game_context)
    return self.network(encoding)

# ⚠️ Missing context at ANY level → network sees zeros → training failure!
```

**Testing**:
```bash
# Verify (P,C)-conditional distributions match decision weights over 10K samples
python -m pytest ml/training/test_training.py::TestSelfPlay::test_decision_weighted_sampling
python -m pytest ml/training/test_training.py::TestSelfPlay::test_player_distribution
python -m pytest ml/training/test_training.py::TestSelfPlay::test_4p_start_cards
```

**Acceptance**:
- 10K samples match player distribution (70% 5p, 15% 4p, 15% 6p) ±2%
- Card distributions match decision weights per (P,C) combination ±3%
- 4p splits correctly between C=7 (60%) and C=8 (40%) ±3%

---

### Session 3: MCTS Curriculum & CLI Integration (4 hours)

**Goal**: Wire MCTS schedule (from Session 0) into training loop and add CLI args

**Prerequisite**: Session 0-2 complete - sampling + encoding working

**Files to modify**:
- `ml/training/trainer.py` - Call `config.get_mcts_params(iteration)` to vary MCTS intensity
- `ml/train.py` - Add `--training-on`, `--iterations` CLI args

**TODO**: Update references to use `training_on` parameter instead of `--phase`

**Wire MCTS curriculum into training loop**:
```python
# In ml/training/trainer.py - TrainingPipeline.run()

class TrainingPipeline:
    def run(self, num_iterations: int):
        for iteration in range(1, num_iterations + 1):
            # Get MCTS params for this iteration (Session 0 implemented get_mcts_params)
            num_det, sims_per_det = self.config.get_mcts_params(iteration)

            logger.info(f"Iteration {iteration}: {num_det} det × {sims_per_det} sims")

            # Update config for this iteration
            self.config.num_determinizations = num_det
            self.config.simulations_per_determinization = sims_per_det

            # Run self-play with updated MCTS params
            # TODO: Add conditional based on config.training_on:
            #   - training_on="rounds": Generate independent rounds (games_per_iteration = rounds)
            #   - training_on="games": Generate full multi-round games (games_per_iteration = games)
            games = self.self_play_engine.generate_games(self.config.games_per_iteration)

            # NEW: Distribution sanity logging (every 10 iterations)
            if iteration % 10 == 0:
                self._log_distribution_sanity(games)

            # ... rest of training loop (replay buffer, network training, etc.)

    def _log_distribution_sanity(self, games):
        """Log and validate sampled distributions match target."""
        player_counts = [g.num_players for g in games]
        card_counts = [g.cards_to_deal for g in games]

        # Histogram
        from collections import Counter
        p_hist = Counter(player_counts)
        logger.info(f"Player distribution: {dict(p_hist)}")

        # Validate against target (±2-3% tolerance)
        total = len(player_counts)
        for p, target_pct in self.config.player_distribution.items():
            actual_pct = p_hist.get(p, 0) / total
            assert abs(actual_pct - target_pct) < 0.03, \
                f"Player {p}: {actual_pct:.1%} vs target {target_pct:.1%}"
```

**Add training_on CLI arg**:
```python
# In ml/train.py
# TODO: Replace --phase with --training-on for clarity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--training-on',
        type=str,
        choices=['rounds', 'games'],
        default='rounds',
        help='Train on independent rounds (Phase 1) or full multi-round games (Phase 2)'
    )
    parser.add_argument('--iterations', type=int, default=400,
                        help='Number of training iterations')
    parser.add_argument('--fast', action='store_true',
                        help='Fast test mode (5 iterations, reduced games)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (optional)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Random seed set to {args.seed}")

    # Load config
    config = get_production_config() if not args.fast else get_fast_config()

    # Override training_on from CLI (Session 4 implements "games" mode)
    if args.training_on:
        config.training_on = args.training_on

    # Run training
    pipeline = TrainingPipeline(config)
    pipeline.run(num_iterations=args.iterations, resume_from=args.resume)
```

**Testing**:
```bash
# Fast integration test (validates pipeline end-to-end)
python ml/train.py --fast --training-on rounds --iterations 5
```

**Acceptance**: 5-iteration test run completes successfully with varied card counts in logs

---

### Session 4: Full Games Mode (training_on="games") & Evaluation (4 hours) [OPTIONAL]

**Goal**: Implement `training_on="games"` mode for full multi-round game generation and evaluation

**Prerequisite**: Session 0-3 complete - `training_on="rounds"` mode working

**Files to modify**:
- `ml/training/selfplay.py` - Implement `FullGameGenerator` class
- `ml/evaluation/arena.py` - Implement `full_game_mode=True` (remove NotImplementedError)

**TODO**: When `config.training_on="games"`, use `FullGameGenerator` instead of `play_round()`

**Implementation** (P-conditional sequences):
```python
class FullGameGenerator:
    def generate_full_game(self, num_players, start_cards):
        """Generate full game with P-conditional sequence.

        Examples:
          5p, C=7: 7,6,5,4,3,2,1,1,1,1,1,2,3,4,5,6,7 (17 rounds)
          4p, C=8: 8,7,6,5,4,3,2,1,1,1,1,2,3,4,5,6,7,8 (18 rounds)
          4p, C=7: 7,6,5,4,3,2,1,1,1,1,2,3,4,5,6,7 (16 rounds)
        """
        # Descending phase: C, C-1, ..., 2, 1
        descending = list(range(start_cards, 0, -1))

        # 1-card repeats: P times (critical scoring rounds!)
        ones = [1] * num_players

        # Ascending phase: 2, 3, ..., C-1, C
        ascending = list(range(2, start_cards + 1))

        full_sequence = descending + ones + ascending

        # Play all rounds with cumulative scoring
        game_positions = []
        cumulative_scores = [0] * num_players

        for round_idx, card_count in enumerate(full_sequence):
            round_positions = self.play_round(
                num_players, card_count, cumulative_scores, round_idx, len(full_sequence)
            )
            game_positions.extend(round_positions)

            # Update cumulative scores after round (for next round's context)
            cumulative_scores = self._get_updated_scores(round_positions)

        return game_positions  # All positions with game-level reward signal
```

**Performance**:
- Phase 2 speed: ~73 games/min (full 16-round games)
- 100 iterations × 10K games = 1M games = ~10 days

**Testing**:
```bash
python ml/train.py --fast --training-on games --iterations 2  # Test full game generation
```

**Acceptance**: Full games generated correctly with proper score accumulation

---

## Training Timeline

### Phase 1: Independent Rounds + Context (REQUIRED)
```bash
python ml/train.py --training-on rounds --iterations 400
```
- **Time**: ~4.6 days
- **Output**: `models/checkpoints/phase1_best.pth`
- **Expected ELO**: ~1400-1600 (solid play across all card counts)
- **Note**: `--training-on rounds` means `games_per_iteration` counts individual rounds (~360 rounds/min)

### Evaluation Checkpoint
Test model on full games using existing arena:
```bash
# Extend ml/evaluation/arena.py to support full-game mode
python -c "
from ml.evaluation.arena import ModelArena
from ml.config import TrainingConfig

config = TrainingConfig()
arena = ModelArena(config, full_game_mode=True)  # NEW: full-game evaluation
results = arena.evaluate_checkpoint(
    'models/checkpoints/phase1_best.pth',
    games=400,
    player_dist={4: 0.15, 5: 0.70, 6: 0.15}  # Match your table
)
print(f'Full-game ELO: {results.elo}')
print(f'Context-aware: {results.shows_context_strategy}')
"
```
- If model shows context-dependent strategy (conservative when ahead) → **DONE!**
- If model ignores game context or struggles with multi-round optimization → Phase 2

**Arena enhancement needed** (Session 4):
- Add `full_game_mode` parameter to ModelArena
- Generate P-conditional sequences (5p/C=7, 4p/C=7 or C=8, etc.)
- Measure ELO on total game scores (not round scores)
- Detect context-aware behavior (bid variance by score position)

### Phase 2: Full Games (OPTIONAL)
```bash
python ml/train.py --training-on games --iterations 100 --resume models/checkpoints/phase1_best.pth
```
- **Time**: ~10 days
- **Output**: `models/checkpoints/phase2_best.pth`
- **Expected ELO**: ~1700-1900 (near-optimal game score maximization)
- **Note**: `--training-on games` means `games_per_iteration` counts full multi-round games (~73 games/min)

---

## Target Distributions (Built into Config)

### Player Distribution (matches your table composition)
```
4 players: 15% (occasional - 60% start C=7, 40% start C=8)
5 players: 70% (YOUR STANDARD - always start C=7)
6 players: 15% (occasional - always start C=7)
```

### Card Distribution (decision-weighted, P-conditional)

**Not static!** Card counts sampled conditional on (P,C) weighted by decision contribution.

**5p, C=7** (70% of training data):
```
1c: 13.2%  (P×1 rounds × 5p × 2 decisions = high weight despite "trivial")
2c:  7.9%  (2 rounds × 5p × 3 decisions)
3c: 10.5%  (2 rounds × 5p × 4 decisions)
4c: 13.2%  (2 rounds × 5p × 5 decisions)
5c: 15.8%  (2 rounds × 5p × 6 decisions)
6c: 18.4%  (2 rounds × 5p × 7 decisions)
7c: 21.1%  (2 rounds × 5p × 8 decisions - highest decision density)
```

**4p, C=8** (6% of training data - 15% × 40%):
```
1c:  9.1%  (4 rounds × 4p × 2 decisions)
2c:  6.1%  (2 rounds × 4p × 3 decisions)
3c:  7.3%  (2 rounds × 4p × 4 decisions)
4c:  9.1%  (2 rounds × 4p × 5 decisions)
5c: 11.0%  (2 rounds × 4p × 6 decisions)
6c: 12.8%  (2 rounds × 4p × 7 decisions)
7c: 14.6%  (2 rounds × 4p × 8 decisions)
8c: 30.0%  (2 rounds × 4p × 9 decisions - dominant!)
```

**4p, C=7** (9% of training data - 15% × 60%):
```
Similar to 5p/C=7 but with 4 players instead of 5
```

**Key insight**: This mirrors actual full-game decision exposure, NOT naive round counts!

---

## Success Metrics

### After Phase 1 (4.6 days):
- ✅ Handles all card counts (1-7) without catastrophic failures
- ✅ Shows context-dependent bidding (conservative when ahead)
- ✅ ELO ~1400-1600 across varied scenarios
- ✅ >50% win rate against random baseline on full games

### After Phase 2 (14.6 days total, if needed):
- ✅ Near-optimal game score maximization
- ✅ Multi-round strategy (e.g., "take risks early, protect lead late")
- ✅ ELO ~1700-1900
- ✅ >70% win rate against Phase 1 model on full games

---

## Why This Approach Wins

| Metric | Fixed 5-card | Full Games Only | **Hybrid (Decision-Weighted)** |
|--------|-------------|-----------------|-------------------------------|
| Training time | 5.3 days | 48 days | **4.6-14.6 days** |
| Card coverage | ❌ 5 only | ✅ 1-8 (P-conditional) | ✅ 1-8 (P-conditional) |
| Game context | ❌ None | ✅ Natural | ✅ Encoded + synthetic |
| Reward alignment | ❌ Round-only | ✅ Perfect | ✅ Context-based → game-optimal |
| Decision distribution | ❌ 100% at 5c | ✅ Matches real games | ✅ Matches real games |
| Your scenario (5p/7c) | ❌ 0% exposure | 21.1% × 70% = 14.8% | **21.1% × 70% = 14.8%** |
| Handles 4p C=8 | ❌ No | ✅ Yes (6% exposure) | ✅ Yes (6% exposure) |
| Prevents 1c failures | ❌ No | ✅ Yes (13% exposure) | ✅ Yes (13% exposure) |

**Key advantage**: Decision-weighted sampling matches actual full-game exposure at the **position level**, not just round counts. Training on 5p/7c rounds at 21.1% mirrors their decision contribution in real games (not 50% naive round-count weighting or 35% arbitrary static weight). Context encoding enables game-optimal strategy without slow full-game training.

---

## Quick Reference Commands

```bash
# Setup (one-time)
source venv/bin/activate
pip install -r ml/requirements.txt

# Session 0-4 testing (incremental)
python -c "..."  # See Session 0 Testing section - verify sampling/schedule/encoder wiring
python -m pytest ml/network/test_network.py  # After Session 1 - context features populated
python -m pytest ml/training/test_training.py  # After Session 2 - distribution tests pass
python ml/train.py --fast --training-on rounds --iterations 5  # After Session 3 - end-to-end pipeline

# Full training (after all sessions)
python ml/train.py --training-on rounds --iterations 400  # Short Phase 1 (~4.6 days)
python ml/train.py --training-on rounds --iterations 500  # Long Phase 1 (~5.8 days)

# Checkpoint evaluation (using extended arena.py)
python -c "from ml.evaluation.arena import ModelArena; ..."  # See Evaluation section

# Phase 2 (optional, only if Phase 1 evaluation shows need)
python ml/train.py --training-on games --iterations 100  # ~10 days

# Monitoring (External Monitor approach)
python ml/monitor.py  # Live progress dashboard (press 'p' to pause, 'q' to quit)
tail -f runs/training.log  # Alternative: watch raw logs
```

---

## Optional: Data Curriculum Tweaks

For faster strategic depth learning, consider **early Phase 1 boost** for complex rounds:

**Iterations 1-100** (foundation building):
- Boost c≥6 by +10% relative weight (multiply 6c/7c/8c weights by 1.1, renormalize)
- Accelerates complex-play pattern learning while network is most plastic

**Iterations 101+** (curriculum annealing):
- Gradually return to decision-weight neutral by iteration 200
- Ensures final model aligns with true game decision distribution

**Implementation**: Add `curriculum_phase` parameter to `compute_decision_weights()`:
```python
def compute_decision_weights(num_players, start_cards, iteration=0):
    base_weights = {...}  # Standard decision weights

    # Early curriculum boost for c≥6
    if iteration < 100:
        boost = 1.10  # 10% boost
        for c in [6, 7, 8]:
            if c in base_weights:
                base_weights[c] *= boost
    elif iteration < 200:
        # Linear anneal: 1.10 → 1.00 over iterations 100-200
        boost = 1.10 - 0.10 * (iteration - 100) / 100
        for c in [6, 7, 8]:
            if c in base_weights:
                base_weights[c] *= boost

    # Renormalize to probabilities
    total = sum(base_weights.values())
    return {c: w/total for c, w in base_weights.items()}
```

**Tradeoff**: Faster strategic depth vs slight distribution shift early on. Recommended if time-limited or seeking maximum performance per training hour.
```

---

## External Monitor & Checkpoint Management (Optional Enhancement)

**Status**: Documentation only - not yet implemented

**Goal**: Add external monitoring with live progress display and improve checkpoint management with iteration-boundary pause/resume capability. Core training works without this; it enhances operational visibility and checkpoint efficiency.

**Approach**: External Monitor (two-process, file-based communication)
- Training writes status.json for progress updates
- Monitor reads status.json and displays live dashboard
- Control signals via control.signal file for pause requests
- No web server, no threading, no IPC complexity

**Estimated Effort**: ~8 hours implementation

---

### A. Checkpoint Rotation & ELO (Permanent + Cache)

**Problem**: Current implementation saves checkpoints every 10 iterations with no cleanup, leading to unbounded disk growth (~25-50GB for 500 iterations).

**Solution**: Two-tier checkpoint system with automatic rotation.

**Permanent Checkpoints** (iterations 5, 10, 15, 20, ...):
- Path: `models/checkpoints/permanent/checkpoint_iter_{N}.pth`
- Saved when `(iteration + 1) % 5 == 0`
- **Filename convention**: 1-based (checkpoint_iter_5.pth, checkpoint_iter_10.pth, etc.)
  - Internal iteration index is 0-based (0, 1, 2, ...)
  - Example: internal iteration index 4 → filename checkpoint_iter_5.pth
  - Example: internal iteration index 19 → filename checkpoint_iter_20.pth
- Never deleted automatically
- Triggers ELO evaluation
- Contents: model + optimizer + scheduler state

**Cached Checkpoints** (iterations 1, 2, 3, 4, 6, 7, 8, 9, ...):
- Path: `models/checkpoints/cache/checkpoint_iter_{N}.pth`
- Max 4 files at any time
- **Pruning timing**: When saving a cache checkpoint, prune oldest immediately (FIFO behavior)
  - Cache never exceeds 4 files, even between permanent saves
- When permanent checkpoint created: clear entire cache
- Contents: model + optimizer + scheduler state (same as permanent)

**Replay Buffer Handling:**
- Remove replay buffer persistence from `ml/training/trainer.py:906`
- On resume: buffer starts empty and refills naturally during self-play
- Code already handles missing buffer gracefully (verified in agent exploration)

**Storage Efficiency:**
- 500 iterations = 100 permanent + max 4 cache (~50GB total)
  - **Note**: Estimate depends on model size (~4.9M parameters) and optimizer state
  - Allow users to strip optimizer from older permanents later if disk becomes tight
- vs current: 50 checkpoints saved every 10 iterations (unbounded growth)
- More granular resume capability (every iteration vs every 10)
- More frequent ELO tracking (every 5 vs every 10)

---

### B. Status File for Progress Tracking

**File**: `models/checkpoints/status.json`

**Purpose**: Real-time progress updates for external monitor

**Fields**:
```json
{
  "iteration": 42,
  "phase": "selfplay",
  "games_done": 2500,
  "games_total": 10000,
  "progress_pct": 25.0,
  "started_at": "2025-11-12T14:00:00",
  "elapsed_sec": 90,
  "eta_sec": 270,
  "status": "running",
  "latest_permanent_iter": 40,
  "latest_permanent_elo": 1245.0,
  "elo_delta": 12.0,
  "games_per_min": 36.7,
  "buffer_warming": false
}
```

**Update Cadence**:
- On every phase transition (atomic write to .tmp then os.replace())
- During self-play via progress callback (throttle to every 5-10 sec)
- Before/after pause operations

**StatusWriter class** (~40 lines in `ml/training/trainer.py`):
- Atomic writes (write to .tmp, then os.replace() to status.json)
  - **Pattern**: Write to `.json.tmp`, then use `os.replace()` (atomic within same filesystem)
  - **Apply to all JSON files**: status.json, elo_history.json, metrics_history.json
- Thread-safe updates from progress callbacks
- Handles missing fields gracefully

---

### C. Control Signal for Pause/Resume

**File**: `models/checkpoints/control.signal`

**Format**: JSON with action command
```json
{"action": "pause"}
```

**Handling** (in `TrainingPipeline.run_training()`):
- Check for control.signal at end of each iteration
- If exists and contains `{"action": "pause"}`: delete file immediately and exit loop gracefully
- Only act at iteration boundary (never mid-iteration)
- Training saves checkpoint before exiting

**Why Iteration-Boundary Only**:
- No complex worker coordination required
- No mid-iteration state persistence needed
- Acceptable wait time: max ~4 minutes per iteration (Medium MCTS)
- Clean checkpoint state

---

### D. External Monitor Script

**File**: `ml/monitor.py` (~200 lines)

**Dependencies**:
- **Preferred**: `rich` library for beautiful terminal UI
- **Fallback**: Plain text output if rich not installed

**Data Sources** (file-based, no IPC):
1. **Primary**: `models/checkpoints/status.json` (phase, progress, ETA)
2. **Metrics**: `models/checkpoints/metrics_history.json` (last 5 iterations)
3. **ELO**: `models/checkpoints/elo_history.json` (progression)
4. **Logs**: Optional tail of `runs/training.log`
   - **Note**: Many progress messages use `print()` (not logging module)
   - `runs/training.log` will not contain print() output unless redirected
   - Rely primarily on status.json and metrics_history.json for monitoring

**Display Format** (live-updating, overwriting):
```
═══════════════════════════════════════════════════
 BlobMaster Training Monitor
═══════════════════════════════════════════════════
 Status: Running | Iteration: 42/500 | Phase: selfplay
 Progress: [████████████░░░░░░░░] 60% (6,000/10,000 games)
 Elapsed: 1m 30s | ETA: 1m 00s

 Latest Permanent: Iter 40 | ELO: 1,245 (+12 since iter 35)

 Recent Metrics (last 5 iterations):
   Iter 41: Loss=2.34 | Win Rate=0.56 | Time=2.1m
   Iter 40: Loss=2.41 | Win Rate=0.54 | Time=2.3m (PROMOTED)
   ...

 Controls: [p]ause after iteration | [q]uit monitor
═══════════════════════════════════════════════════
```

**Keyboard Controls** (non-blocking):
- `p`: Write `{"action": "pause"}` to control.signal
- `q`: Exit monitor (training continues)
- Non-blocking input: Use `select`/`termios` (Unix) or `msvcrt` (Windows)
- UI refreshes every 1 second regardless

**Atomic Reads**:
- Handle missing/partial files gracefully (retry next refresh)
- Cache last valid state if file temporarily unavailable

---

### E. ELO Evaluation (Every 5th Iteration Only)

**Current State**: ELO evaluation runs every 5 iterations (`eval_frequency=5` in config).

**Change**: Evaluation only runs for **permanent checkpoints** (when checkpoint will be kept long-term).

**Modify `ml/training/trainer.py._evaluation_phase()`:**
```python
def _evaluation_phase(self, iteration: int) -> dict:
    """Run evaluation only on permanent checkpoint iterations."""
    if (iteration + 1) % 5 != 0:
        logger.info(f"Skipping evaluation (not a permanent checkpoint iteration)")
        return {"eval_performed": False, "reason": "skipped_non_permanent"}

    # Run evaluation arena, update ELO, check promotion
    # ... existing evaluation logic ...
```

**ELO History:**
- Only permanent checkpoints have ELO ratings
- `models/checkpoints/elo_history.json` contains permanent checkpoint progression only
- Monitor displays latest permanent ELO + delta from previous permanent
- Cached checkpoints have no ELO (they're deleted within 5 iterations anyway)

**Why This Works:**
- Aligns ELO tracking with checkpoint retention policy
- Reduces evaluation overhead (still every 5 iterations, just more explicit)
- Cleaner metrics history (no ELO for transient checkpoints)

---

### F. Implementation Changes

**New Files to Create:**

1. **`ml/monitor.py`** (~200 lines)
   - Main monitor script with rich UI or plain fallback
   - File watching for status.json, metrics_history.json, elo_history.json
   - Non-blocking keyboard input handler
   - Control signal writer (on 'p' keypress)
   - Auto-refresh loop (1 second intervals)

**Files to Modify:**

1. **`ml/training/trainer.py`** (TrainingPipeline class):

   **Add StatusWriter class** (~40 lines):
   ```python
   class StatusWriter:
       """Atomic status file writer for external monitoring."""
       def __init__(self, checkpoint_dir: str):
           self.status_path = Path(checkpoint_dir) / "status.json"

       def update(self, **fields):
           """Atomic write to status.json."""
           # Write to temp file then atomic replace
           tmp_path = self.status_path.with_suffix('.json.tmp')
           with open(tmp_path, 'w') as f:
               json.dump(fields, f, indent=2)
           os.replace(tmp_path, self.status_path)
   ```

   **Modify `run_training()` method**:
   - Add control signal check at end of each iteration
   - Update status.json at phase boundaries
   - Call `_check_pause_signal()` after checkpoint phase
   - Exit loop gracefully if pause requested

   **Add `_check_pause_signal()` method** (~15 lines):
   - Read `control.signal` file
   - If exists and valid JSON with `{"action": "pause"}`: delete and return True
   - Handle missing/malformed files gracefully

   **Update `_selfplay_phase()` method**:
   - Enhance progress callback to update status.json every 5-10 sec
   - Track `last_update` timestamp to throttle
   - Update fields: games_done, progress_pct, eta_sec

   **Update `_checkpoint_phase()` method**:
   - Use (iteration+1) % 5 == 0 for permanent checkpoints
   - Cache checkpoints for iterations not divisible by 5
   - Prune oldest cache to keep ≤4 files
   - Don't save replay buffer (remove line ~906)

2. **`CLAUDE.md`** (Training section):
   - Add External Monitor usage examples
   - Document two-terminal workflow (tmux + monitor)

3. **`HYBRID_TRAINING_PLAN.md`** (this file):
   - Replace entire "Web Monitoring" section with External Monitor approach
   - Update usage examples to show monitor.py
   - Update testing section

**No Changes Needed**:
- `ml/training/selfplay.py` (workers unaffected by iteration-boundary pause)
- `ml/game/blob.py`, `ml/mcts/`, `ml/network/` (core logic unchanged)
- `ml/evaluation/arena.py` (evaluation logic unchanged)
- `ml/train.py` (no Flask integration, no --web flag)

---

### G. Usage Examples

**Start training in tmux:**
```bash
# Terminal 1: Start training
tmux new -s blob_training
python ml/train.py --iterations 500
# Ctrl+B, D to detach
```

**Attach monitor anytime:**
```bash
# Terminal 2: Monitor progress
python ml/monitor.py
# Press 'p' to pause after current iteration
# Press 'q' to exit monitor (training continues)
```

**Reconnect after SSH drop:**
```bash
# SSH dropped? Reconnect and attach to tmux
tmux attach -t blob_training

# Or just attach monitor to running training
python ml/monitor.py  # Picks up status immediately
```

**Resume from paused state:**
```bash
# Training paused itself after seeing control signal
# Simply restart with --resume
python ml/train.py --iterations 500 --resume models/checkpoints/checkpoint_iter_42.pth
# This resumes from internal iteration index 42 (the next iteration after file label 42)
# Matches ml/training/trainer.py:980 resume logic
```

---

### H. Implementation Order

**Phase 1: Backend Infrastructure** (~4 hours)

1. **StatusWriter class** (~1 hour)
   - Create StatusWriter in trainer.py
   - Implement atomic write logic
   - Test with manual updates

2. **Update TrainingPipeline** (~2 hours)
   - Add status.json updates at phase boundaries
   - Integrate progress callback throttling
   - Add control signal checking
   - Remove replay buffer save
   - Test with --fast --iterations 5

3. **Checkpoint rotation** (~1 hour)
   - Implement (iteration+1) % 5 logic
   - Add cache pruning
   - Test with --fast --iterations 15

**Phase 2: External Monitor** (~3 hours)

4. **Monitor script MVP** (~2 hours)
   - Create ml/monitor.py with plain text output
   - Implement file watching (status.json, metrics_history.json)
   - Add basic display loop (no rich yet)
   - Test with running training

5. **Rich UI enhancement** (~1 hour)
   - Add rich library conditional import
   - Implement fancy progress bars and panels
   - Add keyboard input handling (p/q keys)
   - Test pause/resume workflow

**Phase 3: Documentation & Testing** (~1 hour)

6. **Documentation updates**
   - Update CLAUDE.md with monitor examples
   - Update HYBRID_TRAINING_PLAN.md (this section)
   - Add docstrings to monitor.py

7. **Integration testing**
   - Test 10-iteration run with monitor attached/detached
   - Test pause at various points
   - Test tmux detach/reattach workflow

**Total Estimated Effort:** ~8 hours

---

### I. Testing Plan

**Unit Tests**:
1. **StatusWriter atomic writes**:
   - Test that partial writes are never visible
   - Test concurrent updates from multiple threads
   - Test handling of missing directory

2. **Control signal handling**:
   - Test pause signal read and delete
   - Test malformed JSON handling
   - Test missing file handling

**Integration Tests**:

3. **Checkpoint rotation over 20 iterations:**
   ```bash
   python ml/train.py --fast --iterations 20
   # Verify:
   # - Internal iteration indices 4, 9, 14, 19 produce files checkpoint_iter_5.pth, 10, 15, 20 (permanent)
   # - Filenames are 1-based while internal iteration is 0-based
   # - Cache has max 4 files between permanent saves
   # - No replay buffer files exist
   ```

4. **Monitor attachment/detachment:**
   ```bash
   # Terminal 1: Start training
   python ml/train.py --iterations 100 &

   # Terminal 2: Attach monitor
   python ml/monitor.py
   # Verify live updates

   # Kill monitor (Ctrl+C)
   # Restart monitor
   python ml/monitor.py
   # Verify reconnects and displays current state
   ```

5. **Pause/resume flow:**
   ```bash
   # Start training
   python ml/train.py --iterations 100 &

   # Attach monitor and press 'p' after iteration 5
   python ml/monitor.py
   # Verify: training completes iteration 5, then pauses

   # Resume from checkpoint
   python ml/train.py --iterations 100 --resume models/checkpoints/checkpoint_iter_6.pth
   # Verify: continues from iteration 6
   # Note: Resuming from checkpoint_iter_6.pth starts from internal iteration index 6
   #       (i.e., the next iteration after file label 6)
   ```

6. **tmux workflow:**
   ```bash
   # Start in tmux
   tmux new -s test
   python ml/train.py --fast --iterations 10
   # Detach: Ctrl+B, D

   # Reattach and monitor
   tmux attach -t test
   python ml/monitor.py  # Should show progress
   ```

---

### J. Success Criteria

**Checkpoint System:**
- ✅ Permanent checkpoints use (iteration+1) % 5 == 0
- ✅ Filenames: checkpoint_iter_{iteration+1}.pth
- ✅ Cache pruned to ≤4 files
- ✅ No replay buffer files saved
- ✅ ELO evaluation only on permanent checkpoints

**Status File:**
- ✅ status.json updated at all phase boundaries
- ✅ Progress throttled to 5-10 sec during self-play
- ✅ Atomic writes (no partial reads)
- ✅ All required fields present and valid

**Control Signals:**
- ✅ Pause signal read and deleted immediately
- ✅ Training completes current iteration before pausing
- ✅ Checkpoint saved before pause
- ✅ Graceful exit (no crashes)

**External Monitor:**
- ✅ Displays all metrics in real-time
- ✅ Updates every 1 second
- ✅ Handles missing files gracefully
- ✅ Keyboard controls work (p/q)
- ✅ Works with tmux detach/reattach
- ✅ Can start/stop independently of training

**Operational:**
- ✅ Training works without monitor (status.json written but not required)
- ✅ Monitor works with already-running training
- ✅ SSH drop doesn't kill training
- ✅ Resume from checkpoint continues correctly

---

### K. Advantages Over Web Interface

**Simplicity:**
- No Flask, no threading, no web server
- File-based communication (OS-level atomic writes)
- Independent processes (failure isolation)

**Reconnectability:**
- Attach monitor from any terminal
- SSH drop ≠ training death
- tmux-friendly workflow

**Resource Efficiency:**
- No web server overhead
- No threading complexity in training process
- Monitor uses negligible CPU when training running

**Development:**
- Monitor improvements don't require training changes
- No API versioning concerns
- Easier to debug (two separate processes)

**Deployment:**
- No port management
- No firewall configuration
- Works identically locally and over SSH

---

## Green Light Criteria

**After Session 0** (foundation complete):
- ✅ `TrainingConfig` has all distribution params and MCTS schedule
- ✅ `sample_game_config()` returns (P, C, c, None) with correct distributions
- ✅ `StateEncoder.encode(game, player, game_context)` returns 256 dims (202 + 54 zeros)
- ✅ `ModelArena(full_game_mode=True)` raises NotImplementedError
- **Can proceed to Session 1**

**After Session 1** (context encoding complete):
- ✅ `generate_synthetic_context()` creates realistic early/mid/late scenarios
- ✅ Context features populate 29 of 54 reserved dims correctly
- ✅ Normalization verified: scores clamped [0,1], round position in [0,1], etc.
- **Can proceed to Session 2**

**After Session 2** (wired into training):
- ✅ `SelfPlayWorker.play_round()` calls `sample_game_config()`
- ✅ All encoder call sites pass `game_context` parameter
- ✅ Distribution tests pass: 70% 5p ±2%, card distributions match decision weights ±3%
- **Can proceed to Session 3**

**After Session 3** (curriculum & CLI complete):
- ✅ `TrainingPipeline` varies MCTS params by iteration
- ✅ `--training-on rounds --iterations 5` completes successfully with varied (P,C,c) in logs
- ✅ MCTS schedule applied correctly (verify logs show 1×15 early, increasing later)
- **Phase 1 training ready! Can optionally proceed to Session 4**

**After Session 4** (full games complete):
- ✅ `FullGameGenerator` creates P-conditional sequences (5p: 17 rounds, 4p: 16-18)
- ✅ `ModelArena(full_game_mode=True)` evaluates on total game scores
- ✅ `--training-on games` training works, using full-game generation
- **Phase 2 training ready!**

**Implementation-complete threshold**: Sessions 0-3 (16 hours)
- Phase 1 training (independent rounds + context) fully functional
- Decision-weighted sampling operational
- Game context encoding working
- MCTS curriculum applied
- Ready for 400-500 iteration training run

**Full-featured threshold**: Sessions 0-4 (20 hours)
- Phase 2 training (full games) functional
- Full-game evaluation operational
- Can run complete hybrid curriculum (Phase 1 → eval → Phase 2)

---

## Execution Checklist (Critical Details)

**1. Score Normalization** (Session 1):
- ✅ Use `scores[i] / (rounds_completed × (10 + start_cards))`
- ❌ **NOT** `scores[i] / (10 + start_cards)` with hard clamp at 1.0
- **Why**: Preserves signal when player far ahead late-game

**2. Previous-Cards Truncation** (Session 1):
- ✅ Store **most recent 15 rounds** (truncate older, pad with 0)
- ✅ Normalize by `start_cards` (not fixed 8) for scaling consistency
- ❌ **NOT** oldest-first truncation or fixed /8 normalization
- **Why**: Recent context more valuable; consistent scaling across C=7 vs C=8

**3. Encoder Threading** (Session 2):
- ✅ Pass `game_context` through **entire call chain**: SelfPlayWorker → MCTS → evaluate_leaf → encoder
- ❌ **NOT** just top-level encode() call
- **Why**: Missing context at any level → network sees zeros → silent training failure

**4. Distribution Logging** (Session 3):
- ✅ Log histograms every 10 iterations
- ✅ Assert drift within ±2-3% tolerance
- **Why**: Catches config issues early (e.g., broken RNG, weight typos)

**5. Reproducibility** (Session 3):
- ✅ Add `--seed` CLI arg for deterministic sampling
- ✅ Seed np.random, torch, torch.cuda
- **Why**: Distribution tests are stable, debugging is reproducible

**6. Context Generation** (Session 1):
- ✅ Generate synthetic context consistent with (P, C, current_card_count)
- ✅ Sample early/mid/late game phase (30/40/30 split)
- **Why**: Network learns context-dependent strategy from varied scenarios
