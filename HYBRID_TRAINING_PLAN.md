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

## Core Strategy

### The Problem
- Current: Trains on fixed 5-card rounds only → catastrophic failure on 1-3 card rounds
- Actual games vary by player count:
  - 5p/C=7: 7,6,5,4,3,2,1,1,1,1,1,2,3,4,5,6,7 (47% of rounds are 1c)
  - 4p/C=8: 8,7,6,5,4,3,2,1,1,1,1,2,3,4,5,6,7,8 (28% are 1c)
  - But decision points ≠ round counts: 7c has P×(7+1) decisions, 1c has P×(1+1)
- All-or-nothing scoring: 1-card round = 11 pts max, 7-card round = 17 pts max (minimal difference)

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
- Add `--phase`, `--iterations` CLI args to `train.py`
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

**Files to modify**:
- `ml/training/selfplay.py` - SelfPlayWorker.play_round() (call sample_game_config)
- `ml/training/trainer.py` - Pass sampled config through pipeline
- All encoder call sites - Pass `game_context` parameter

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
- `ml/train.py` - Add `--phase`, `--iterations` CLI args

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

**Add Phase 1/2 CLI args**:
```python
# In ml/train.py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, choices=[1, 2], default=1,
                        help='Phase 1: independent rounds, Phase 2: full games')
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

    # Phase 2: Use full-game generation (Session 4 implements this)
    if args.phase == 2:
        config.use_full_game_mode = True  # Session 4 adds this flag

    # Run training
    pipeline = TrainingPipeline(config)
    pipeline.run(num_iterations=args.iterations, resume_from=args.resume)
```

**Testing**:
```bash
# Fast integration test (validates pipeline end-to-end)
python ml/train.py --fast --phase 1 --iterations 5
```

**Acceptance**: 5-iteration test run completes successfully with varied card counts in logs

---

### Session 4: Phase 2 Full Games & Evaluation (4 hours) [OPTIONAL]

**Goal**: Implement full-game generation and evaluation (stubbed in Session 0)

**Prerequisite**: Session 0-3 complete - Phase 1 training working

**Files to modify**:
- `ml/training/selfplay.py` - Implement `FullGameGenerator` class
- `ml/evaluation/arena.py` - Implement `full_game_mode=True` (remove NotImplementedError)
- `ml/config.py` - Add `use_full_game_mode` flag

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
python ml/train.py --fast --phase 2 --iterations 2  # Test full game generation
```

**Acceptance**: Full games generated correctly with proper score accumulation

---

## Training Timeline

### Phase 1: Independent Rounds + Context (REQUIRED)
```bash
python ml/train.py --phase 1 --iterations 400
```
- **Time**: ~4.6 days
- **Output**: `models/checkpoints/phase1_best.pth`
- **Expected ELO**: ~1400-1600 (solid play across all card counts)

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
python ml/train.py --phase 2 --iterations 100 --resume models/checkpoints/phase1_best.pth
```
- **Time**: ~10 days
- **Output**: `models/checkpoints/phase2_best.pth`
- **Expected ELO**: ~1700-1900 (near-optimal game score maximization)

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
python ml/train.py --fast --phase 1 --iterations 5  # After Session 3 - end-to-end pipeline

# Full training (after all sessions)
python ml/train.py --phase 1 --iterations 400  # Short Phase 1 (~4.6 days)
python ml/train.py --phase 1 --iterations 500  # Long Phase 1 (~5.8 days)

# Checkpoint evaluation (using extended arena.py)
python -c "from ml.evaluation.arena import ModelArena; ..."  # See Evaluation section

# Phase 2 (optional, only if Phase 1 evaluation shows need)
python ml/train.py --phase 2 --iterations 100  # ~10 days

# Monitoring
tensorboard --logdir=runs/  # If tensorboard installed
tail -f logs/training.log  # Watch progress
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
- ✅ `--phase 1 --iterations 5` completes successfully with varied (P,C,c) in logs
- ✅ MCTS schedule applied correctly (verify logs show 1×15 early, increasing later)
- **Phase 1 training ready! Can optionally proceed to Session 4**

**After Session 4** (full games complete):
- ✅ `FullGameGenerator` creates P-conditional sequences (5p: 17 rounds, 4p: 16-18)
- ✅ `ModelArena(full_game_mode=True)` evaluates on total game scores
- ✅ `--phase 2` training works, using full-game generation
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
