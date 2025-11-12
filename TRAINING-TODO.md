# Training TODO - Final Implementation Plan

**Created**: 2025-11-12
**Approved**: 2025-11-12 ✅
**Status**: Approved - Ready for implementation
**Total Effort**: 22 hours core + 8 hours optional = 30 hours
**Expected Training Time**: ~10-12 days Phase 1 + ~10 days Phase 2 (optional)

---

## Current Project State

**Verified Status** (as of 2025-11-12):

✅ **Complete:**
- HYBRID Session 0: Infrastructure (TrainingConfig with distributions, GameContext, DECISION_WEIGHTS)
- HYBRID Session 1: Context encoding (256-dim StateEncoder with game_context parameter)
- HYBRID Session 2: Decision-weighted sampling (sample_game_config() implemented)
- Performance Session 1: Determinization fix (364 games/min achieved, ~10x speedup)

❌ **Incomplete** (blockers for progressive training):
- HYBRID Session 3: MCTS curriculum NOT wired into TrainingPipeline.run_training()
- CLI flag `--training-on` does NOT exist in ml/train.py
- Distribution sanity logging NOT implemented
- HYBRID Session 4: Full-game mode NOT implemented

---

## Prerequisites: Complete Missing Baseline

### Session 0: Wire MCTS Curriculum & CLI (4 hours)

**Goal**: Complete HYBRID Session 3 infrastructure before progressive enhancements.

**Status**: REQUIRED - Must complete before any other sessions.

#### Changes Needed:

**1. Update `ml/training/trainer.py`** (~100 lines):

```python
class TrainingPipeline:
    def run_training(self, num_iterations: int, resume_from: Optional[str] = None):
        """Run training with MCTS curriculum."""

        for iteration in range(start_iteration, num_iterations):
            # NEW: Get MCTS params for this iteration
            num_det, sims_per_det = self.config.get_mcts_params(iteration)

            logger.info(
                f"Iteration {iteration + 1}: MCTS curriculum "
                f"{num_det} det × {sims_per_det} sims"
            )

            # Update self-play engine config
            self.selfplay_engine.num_determinizations = num_det
            self.selfplay_engine.simulations_per_determinization = sims_per_det

            # Run iteration with updated MCTS params
            metrics = self.run_iteration(iteration)

            # NEW: Distribution sanity logging (every 10 iterations)
            if (iteration + 1) % 10 == 0:
                self._log_distribution_sanity(metrics)

    def _log_distribution_sanity(self, metrics: Dict[str, Any]):
        """Validate sampled distributions match targets."""
        if 'distribution_stats' not in metrics:
            return

        stats = metrics['distribution_stats']

        # Check player distribution
        total_games = sum(stats['player_counts'].values())
        for num_players, target_pct in self.config.player_distribution.items():
            actual_count = stats['player_counts'].get(num_players, 0)
            actual_pct = actual_count / total_games if total_games > 0 else 0.0

            logger.info(
                f"  Player {num_players}: {actual_pct:.1%} "
                f"(target {target_pct:.1%})"
            )

            # Assert within ±3% tolerance
            if abs(actual_pct - target_pct) > 0.03:
                logger.warning(
                    f"Distribution drift detected for {num_players}p: "
                    f"{actual_pct:.1%} vs target {target_pct:.1%}"
                )

        # Check card count distributions per (P, C) combination
        for (P, C), card_dist in stats['card_distributions'].items():
            logger.info(f"  Card distribution for {P}p/C={C}:")
            for c, count in sorted(card_dist.items()):
                pct = count / total_games if total_games > 0 else 0.0
                logger.info(f"    c={c}: {pct:.1%}")
```

**2. Update `ml/training/selfplay.py`** (~50 lines):

```python
class SelfPlayEngine:
    def generate_games(self, num_games: int, progress_callback=None):
        """Generate games and collect distribution stats."""

        # Generate games (existing logic)
        all_examples = self._generate_games_multiprocess(num_games, progress_callback)

        # NEW: Collect distribution statistics
        if self.config.use_decision_weighted_sampling:
            distribution_stats = self._collect_distribution_stats(all_examples)
        else:
            distribution_stats = None

        return all_examples, distribution_stats

    def _collect_distribution_stats(self, examples: List[Dict]) -> Dict[str, Any]:
        """Collect player/card distribution statistics."""
        from collections import Counter

        player_counts = Counter()
        card_distributions = {}  # (P, C) -> Counter({c: count})

        for example in examples:
            # Extract from game metadata (stored in example)
            num_players = example.get('num_players')
            start_cards = example.get('start_cards')
            cards_dealt = example.get('cards_dealt')

            if num_players is not None:
                player_counts[num_players] += 1

            if num_players and start_cards and cards_dealt:
                key = (num_players, start_cards)
                if key not in card_distributions:
                    card_distributions[key] = Counter()
                card_distributions[key][cards_dealt] += 1

        return {
            'player_counts': dict(player_counts),
            'card_distributions': {
                f"{P}p_C{C}": dict(dist)
                for (P, C), dist in card_distributions.items()
            },
        }
```

**3. Update `ml/train.py`** (~30 lines):

```python
def parse_args():
    parser = argparse.ArgumentParser(...)

    # NEW: Add --training-on flag
    parser.add_argument(
        '--training-on',
        type=str,
        choices=['rounds', 'games'],
        default='rounds',
        help='Training mode: "rounds" (independent rounds, Phase 1) or '
             '"games" (full multi-round games, Phase 2)'
    )

    # NEW: Add --enable-curriculum flag
    parser.add_argument(
        '--enable-curriculum',
        action='store_true',
        help='Enable MCTS curriculum (progressively increase search depth)'
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Load config
    if args.fast:
        config = get_fast_config()
    else:
        config = get_production_config()

    # Override training mode from CLI
    config.training_on = args.training_on

    # Enable MCTS curriculum if requested
    if args.enable_curriculum:
        logger.info("MCTS curriculum enabled")
        # Config already has mcts_schedule, just needs to be applied
    else:
        # Use fixed MCTS params (no curriculum)
        logger.info("Using fixed MCTS params (no curriculum)")
        config.mcts_schedule = {}  # Empty schedule = use defaults

    # Continue with training...
```

**4. Update `ml/training/selfplay.py` (worker metadata)** (~20 lines):

```python
class SelfPlayWorker:
    def generate_game(self, ...):
        """Generate game and include metadata in examples."""

        # Sample config
        if self.config and self.config.use_decision_weighted_sampling:
            num_players, start_cards, cards_to_deal, game_context = self.sample_game_config()

        # Generate game...
        positions = []

        while not game.is_terminal():
            # ... MCTS and action selection ...

            positions.append({
                'state': state_encoding,
                'policy': mcts_policy,
                'value': None,  # filled later
                'num_players': num_players,       # NEW: metadata
                'start_cards': start_cards,       # NEW: metadata
                'cards_dealt': cards_to_deal,     # NEW: metadata
                'game_context': game_context,
            })

        return positions
```

#### Testing:

```bash
# Test MCTS curriculum schedule
python -c "
from ml.config import TrainingConfig

config = TrainingConfig()
for i in [1, 50, 100, 150, 250, 300, 400, 450, 500]:
    det, sims = config.get_mcts_params(i)
    print(f'Iter {i:3d}: {det} det × {sims} sims')
"

# Test CLI flags
python ml/train.py --help | grep -E "(training-on|enable-curriculum)"

# Test distribution logging with fast run
python ml/train.py --fast --training-on rounds --enable-curriculum --iterations 5
# Verify logs show:
# - "MCTS curriculum enabled"
# - "Iteration N: MCTS curriculum X det × Y sims"
# - Distribution stats at iteration 10 (if running 10+ iterations)
```

#### Acceptance Criteria:

- ✅ `TrainingPipeline.run_training()` calls `config.get_mcts_params(iteration)`
- ✅ MCTS params update each iteration (logged)
- ✅ `--training-on` CLI flag works (accepts 'rounds' or 'games')
- ✅ `--enable-curriculum` CLI flag works
- ✅ Distribution sanity logging runs every 10 iterations
- ✅ Player distribution logged and validated (±3% tolerance)
- ✅ Card distribution logged per (P, C) combination
- ✅ Fast test run completes without errors

---

## Phase 1: Core Training Enhancements

### Session 1: Zero-Choice Fast Path (2 hours)

**Goal**: Skip MCTS and training examples for forced last-card plays (14% of decisions).

**Justification**: User correctly identified that EVERY round ends with forced last-card plays. Real savings: ~14% of playing decisions.

#### Analysis:

**Forced decision scenarios:**
1. **Last card in hand** (playing phase): ~14% of all decisions
   - 5p/7c game: 17 rounds → 17 forced last-cards per player
   - Total decisions per player: ~118 (59 bids + 59 plays)
   - Forced plays: 17/118 ≈ 14.4%

2. **Dealer no-choice bid** (bidding phase): <1% of all decisions
   - Only when dealer has single legal bid (rare)
   - Implementation complexity: moderate (need to compute forbidden bid)

**Scope decision**: Implement #1 only (last-card fast path), skip #2 (edge case, low ROI).

#### Implementation:

**File**: `ml/training/selfplay.py` (SelfPlayWorker.generate_game) (~30 lines):

```python
def generate_game(self, num_players: int, cards_to_deal: int, game_id: Optional[str] = None):
    """Generate game with zero-choice fast path."""

    # ... initialization ...

    positions = []
    forced_skips = 0  # NEW: track skipped forced actions

    while not game.is_terminal():
        current_player = game.current_player

        # Get legal actions
        if game.phase == 'bidding':
            legal_bids = game.get_legal_bids(current_player)
            legal_actions = legal_bids
        else:  # playing phase
            legal_cards = game.get_legal_cards(current_player)
            legal_actions = legal_cards

        # NEW: Zero-choice fast path
        if len(legal_actions) == 1:
            # Only one legal action - forced move
            action = legal_actions[0]
            game.apply_action(action)
            forced_skips += 1
            continue  # Don't store training example

        # Regular MCTS path (when choice exists)
        mcts_policy = self._run_mcts(game, game_context)

        # Store training example
        positions.append({
            'state': self.encoder.encode(game, current_player, game_context),
            'policy': mcts_policy,
            'value': None,
            # ... metadata ...
        })

        # Sample and apply action
        action = self._select_action(mcts_policy, legal_actions, move_number)
        game.apply_action(action)

    # Log forced skip rate
    if forced_skips > 0:
        total_decisions = len(positions) + forced_skips
        skip_rate = forced_skips / total_decisions
        # Optionally log to metrics

    return positions
```

#### Metrics to Track:

Add to iteration metrics:
```python
{
    'forced_action_skips': total_forced_skips,
    'total_decisions': total_decisions,
    'skip_rate': forced_skips / total_decisions,  # Expect ~14%
    'avg_policy_entropy': avg_entropy_of_stored_policies,  # Should increase slightly
}
```

#### Testing:

```bash
# Unit test
python -c "
from ml.training.selfplay import SelfPlayWorker
from ml.config import TrainingConfig
from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker

config = TrainingConfig()
network = BlobNet(config)
encoder = StateEncoder()
masker = ActionMasker()

worker = SelfPlayWorker(
    network=network,
    encoder=encoder,
    masker=masker,
    config=config,
)

# Generate a 7-card game (should skip 7 forced last-cards)
positions = worker.generate_game(num_players=5, cards_to_deal=7)

# Count decisions (expect fewer than 5 players × 7 bids + 5 × 7 plays = 70)
# because last 7 playing decisions are skipped
print(f'Positions stored: {len(positions)}')
print(f'Expected: ~63 (70 total - 7 forced last-cards)')
"

# Integration test with fast training
python ml/train.py --fast --iterations 2 --training-on rounds
# Check logs for:
# - "forced_action_skips: X"
# - "skip_rate: ~0.14" (14%)
```

#### Acceptance Criteria:

- ✅ Last-card plays skip MCTS (len(legal_cards) == 1)
- ✅ No training examples stored for forced actions
- ✅ `forced_action_skips` tracked per iteration
- ✅ Skip rate ~14% (±3%) for typical games
- ✅ Avg policy entropy of stored moves increases (forced moves excluded)
- ✅ Training runs without errors
- ✅ Self-play throughput increases (expect ~16% speedup = 422 games/min)

---

### Session 2: EMA Stabilization & Progressive Targets (4 hours)

**Goal**: Stabilize self-play with EMA model and sharpen policy targets over training.

**Justification**: Proven technique from MuZero/AlphaZero - prevents overfitting to noisy early self-play data.

#### Key Concepts:

1. **EMA (Exponential Moving Average) Model**:
   - Maintain smoothed copy of network weights
   - Use EMA model for self-play (stable policy)
   - Train online model (adapts quickly)
   - Prevents catastrophic forgetting

2. **Progressive Target Sharpening**:
   - Build policy targets from MCTS visit counts with temperature: `π_target ∝ N^(1/τ)`
   - Anneal τ over training: 1.0 (uniform) → 0.7 (sharp) over ~200 iterations
   - Early: explore diverse strategies
   - Late: commit to best actions

#### Implementation:

**File**: `ml/training/trainer.py` (~120 lines):

```python
class TrainingPipeline:
    def __init__(self, network, encoder, masker, config):
        # ... existing init ...

        # NEW: EMA model
        self.ema_model = copy.deepcopy(network)
        self.ema_decay = 0.997  # Fixed value from MuZero
        self.use_ema_for_selfplay = True

        # NEW: Progressive target temperature schedule
        self.pi_target_tau_start = 1.0
        self.pi_target_tau_end = 0.7
        self.tau_anneal_iters = 200

    def get_pi_target_tau(self, iteration: int) -> float:
        """Get policy target temperature for this iteration."""
        if iteration >= self.tau_anneal_iters:
            return self.pi_target_tau_end

        # Linear anneal
        progress = iteration / self.tau_anneal_iters
        tau = self.pi_target_tau_start - progress * (
            self.pi_target_tau_start - self.pi_target_tau_end
        )
        return tau

    def update_ema_model(self):
        """Update EMA model weights."""
        with torch.no_grad():
            for ema_param, online_param in zip(
                self.ema_model.parameters(),
                self.network.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    online_param.data, alpha=1 - self.ema_decay
                )

    def run_iteration(self, iteration: int):
        """Run iteration with EMA and progressive targets."""

        # Phase 1: Self-play (use EMA model if enabled)
        model_for_selfplay = self.ema_model if self.use_ema_for_selfplay else self.network

        # Update self-play engine to use EMA model
        self.selfplay_engine.update_model(model_for_selfplay)

        examples, dist_stats = self.selfplay_engine.generate_games(
            self.config.games_per_iteration
        )

        # Phase 2: Training
        # Get policy target temperature for this iteration
        tau = self.get_pi_target_tau(iteration)
        logger.info(f"Policy target τ = {tau:.3f}")

        # Transform MCTS visit counts to policy targets with temperature
        for example in examples:
            visit_counts = example['policy']  # Raw visit counts from MCTS

            # Apply temperature: π_target ∝ N^(1/τ)
            if tau != 1.0:
                visit_counts_tempered = np.power(visit_counts, 1.0 / tau)
            else:
                visit_counts_tempered = visit_counts

            # Normalize to probabilities
            total = visit_counts_tempered.sum()
            if total > 0:
                example['policy_target'] = visit_counts_tempered / total
            else:
                example['policy_target'] = visit_counts  # Fallback (shouldn't happen)

        # Train online model (not EMA)
        train_metrics = self.trainer.train_epoch(
            examples,
            use_policy_targets=True  # Use tempered targets
        )

        # Phase 3: Update EMA model
        self.update_ema_model()
        logger.info(f"EMA model updated (decay={self.ema_decay})")

        # Phase 4: Evaluation (use online model, not EMA)
        eval_metrics = self.evaluator.evaluate(
            self.network,  # Evaluate online model
            num_games=self.config.eval_games
        )

        return {
            **train_metrics,
            **eval_metrics,
            'pi_target_tau': tau,
            'ema_decay': self.ema_decay,
        }
```

**File**: `ml/training/selfplay.py` (~20 lines):

```python
class SelfPlayEngine:
    def update_model(self, new_model: BlobNet):
        """Update the model used for self-play."""
        # Update in-process model (for threaded mode)
        self.network = new_model

        # For multiprocessing: workers need to reload model
        # This is tricky - may need to save to temp file and reload
        # OR just use in-process threading for now
```

**Note**: Multiprocessing with updated models requires either:
1. Saving EMA model to temp checkpoint and reloading in workers
2. Using threading instead of multiprocessing (slower but simpler)

For now, recommend **Option 1**: Save EMA model to `models/checkpoints/ema_current.pth` and reload in workers each iteration.

#### Testing:

```bash
# Test EMA update
python -c "
import torch
from ml.network.model import BlobNet
from ml.config import TrainingConfig

config = TrainingConfig()
network = BlobNet(config)
ema_model = copy.deepcopy(network)

# Simulate training update
with torch.no_grad():
    for param in network.parameters():
        param.data.add_(torch.randn_like(param) * 0.1)  # Random update

# Apply EMA
ema_decay = 0.997
with torch.no_grad():
    for ema_param, online_param in zip(ema_model.parameters(), network.parameters()):
        ema_param.data.mul_(ema_decay).add_(online_param.data, alpha=1 - ema_decay)

# Verify EMA is between old and new (smoothed)
print('EMA update successful')
"

# Test progressive tau schedule
python -c "
from ml.training.trainer import TrainingPipeline

# Simulate schedule
for i in [0, 50, 100, 150, 200, 250, 300]:
    progress = min(i / 200, 1.0)
    tau = 1.0 - progress * (1.0 - 0.7)
    print(f'Iter {i:3d}: τ = {tau:.3f}')
"

# Integration test
python ml/train.py --fast --iterations 5 --training-on rounds
# Check logs for:
# - "Policy target τ = X.XXX"
# - "EMA model updated"
```

#### Acceptance Criteria:

- ✅ EMA model maintained (decay=0.997)
- ✅ Self-play uses EMA model
- ✅ Training updates online model
- ✅ Policy target τ anneals: 1.0 → 0.7 over 200 iterations
- ✅ Tempered targets applied before training
- ✅ Evaluation uses online model (not EMA)
- ✅ Logs show τ value each iteration
- ✅ Training stable (no divergence)

---

### Session 3: Fixed Exploration (Root Dirichlet Noise) (4 hours)

**Goal**: Add Dirichlet noise at MCTS root for exploration during self-play.

**Justification**: Standard AlphaZero technique - ensures diverse self-play data. Uses FIXED parameters validated in literature (not adaptive schedules).

#### Key Concepts:

**Dirichlet Noise at Root**:
- Add noise to prior policy at root node: `P'(a) = (1-ε)·P(a) + ε·Dir(α)`
- Parameters from AlphaZero paper:
  - ε (epsilon) = 0.25 (25% noise weight)
  - α (alpha) = 0.30 (Dirichlet concentration)
- Apply ONLY in self-play, NEVER in evaluation/arena

**Why fixed parameters?**
- AlphaZero used constant noise throughout training (no schedule)
- Proven to work across chess, Go, shogi
- Simpler than adaptive schedules
- Avoids hyperparameter hell

#### Implementation:

**File**: `ml/mcts/search.py` (~40 lines):

```python
class ImperfectInfoMCTS:
    def __init__(self, ..., exploration_noise_epsilon=0.0, exploration_noise_alpha=0.3):
        """
        Args:
            exploration_noise_epsilon: Weight of Dirichlet noise (0.0 = no noise, 0.25 = AlphaZero default)
            exploration_noise_alpha: Dirichlet concentration parameter (0.3 = AlphaZero default)
        """
        self.exploration_noise_epsilon = exploration_noise_epsilon
        self.exploration_noise_alpha = exploration_noise_alpha

    def search(self, game, game_context=None):
        """Run MCTS with optional root noise."""

        # Create root node
        root = MCTSNode(state=game, player=game.current_player)

        # Get prior policy from network
        state_encoding = self.encoder.encode(game, root.player, game_context)
        policy_logits, value = self.network(state_encoding)
        prior_probs = self.masker.apply_policy_mask(game, root.player, policy_logits)

        # Apply Dirichlet noise at root (if enabled)
        if self.exploration_noise_epsilon > 0:
            prior_probs = self._add_dirichlet_noise(
                prior_probs,
                epsilon=self.exploration_noise_epsilon,
                alpha=self.exploration_noise_alpha,
            )

        root.expand(prior_probs, value)

        # Run MCTS simulations
        for _ in range(self.num_simulations):
            self._run_simulation(root, game_context)

        # Return visit count policy (no noise in output)
        return root.get_action_probs()

    def _add_dirichlet_noise(
        self,
        prior_probs: torch.Tensor,
        epsilon: float,
        alpha: float,
    ) -> torch.Tensor:
        """
        Add Dirichlet noise to prior probabilities at root.

        P'(a) = (1-ε)·P(a) + ε·Dir(α)

        Args:
            prior_probs: Prior policy probabilities (52-dim, includes masked zeros)
            epsilon: Noise weight (0.25 = 25% noise)
            alpha: Dirichlet concentration (0.3 = AlphaZero default)

        Returns:
            Noisy prior probabilities (same shape as input)
        """
        # Sample Dirichlet noise only for legal actions (non-zero probs)
        legal_mask = prior_probs > 0
        num_legal = legal_mask.sum().item()

        if num_legal == 0:
            return prior_probs  # No legal actions (shouldn't happen)

        # Sample Dirichlet noise
        noise = np.random.dirichlet([alpha] * num_legal)

        # Apply noise only to legal actions
        noisy_probs = prior_probs.clone()
        noisy_probs[legal_mask] = (
            (1 - epsilon) * prior_probs[legal_mask]
            + epsilon * torch.from_numpy(noise).float().to(prior_probs.device)
        )

        # Renormalize (should already sum to 1, but ensure numerical stability)
        noisy_probs = noisy_probs / noisy_probs.sum()

        return noisy_probs
```

**File**: `ml/training/selfplay.py` (~20 lines):

```python
class SelfPlayWorker:
    def __init__(self, ..., enable_exploration_noise=True):
        """
        Args:
            enable_exploration_noise: Add Dirichlet noise at root (self-play only)
        """
        self.enable_exploration_noise = enable_exploration_noise

        # Create MCTS with exploration noise
        self.mcts = ImperfectInfoMCTS(
            network=network,
            encoder=encoder,
            masker=masker,
            num_determinizations=num_determinizations,
            simulations_per_determinization=simulations_per_determinization,
            exploration_noise_epsilon=0.25 if enable_exploration_noise else 0.0,
            exploration_noise_alpha=0.30,
            # ... other params ...
        )
```

**File**: `ml/evaluation/arena.py` (~10 lines):

```python
class ModelArena:
    def evaluate(self, model, num_games):
        """Evaluate model WITHOUT exploration noise."""

        # Create workers with noise DISABLED
        workers = [
            SelfPlayWorker(
                network=model,
                encoder=self.encoder,
                masker=self.masker,
                enable_exploration_noise=False,  # NO NOISE in evaluation
                # ... other params ...
            )
            for _ in range(num_workers)
        ]

        # Run evaluation games...
```

#### Testing:

```bash
# Test Dirichlet noise
python -c "
import numpy as np
import torch

# Test noise generation
prior_probs = torch.tensor([0.5, 0.3, 0.2, 0.0, 0.0])  # 3 legal actions
legal_mask = prior_probs > 0
num_legal = legal_mask.sum().item()

noise = np.random.dirichlet([0.3] * num_legal)
print(f'Prior: {prior_probs}')
print(f'Noise: {noise}')

epsilon = 0.25
noisy_probs = prior_probs.clone()
noisy_probs[legal_mask] = (1 - epsilon) * prior_probs[legal_mask] + epsilon * torch.from_numpy(noise).float()
noisy_probs = noisy_probs / noisy_probs.sum()

print(f'Noisy: {noisy_probs}')
print(f'Sum: {noisy_probs.sum()}')
"

# Test self-play with noise
python ml/train.py --fast --iterations 2 --training-on rounds
# Check logs for:
# - No "exploration_noise_epsilon" errors
# - Policy entropy higher than without noise
```

#### Acceptance Criteria:

- ✅ Dirichlet noise applied at MCTS root (ε=0.25, α=0.30)
- ✅ Noise applied ONLY in self-play (not evaluation)
- ✅ Noise applied ONLY to legal actions
- ✅ Prior probabilities sum to 1.0 after noise
- ✅ Policy entropy increases with noise (vs no-noise baseline)
- ✅ Training stable (no NaN/Inf values)
- ✅ Evaluation runs without noise
- ✅ Fast test completes successfully

---

## Phase 2: Full-Game Training

### Session 4: Full-Game Evaluation Infrastructure (4 hours)

**Goal**: Extend ModelArena to support full multi-round game evaluation.

**Justification**: Need to validate that Phase 1 models show context-aware strategy before investing in full-game training (Session 5).

#### Key Concepts:

**Full-game sequences**:
- 5p/C=7: 7,6,5,4,3,2,1,1,1,1,1,2,3,4,5,6,7 (17 rounds)
- 4p/C=8: 8,7,6,5,4,3,2,1,1,1,1,2,3,4,5,6,7,8 (18 rounds)
- 4p/C=7: 7,6,5,4,3,2,1,1,1,1,2,3,4,5,6,7 (16 rounds)

**Evaluation criteria**:
- Total game score (sum of all rounds)
- ELO rating on full games (not individual rounds)
- Context-aware behavior detection (conservative when ahead?)

#### Implementation:

**File**: `ml/evaluation/arena.py` (~150 lines):

```python
class ModelArena:
    def __init__(self, ..., full_game_mode=False):
        """
        Args:
            full_game_mode: If True, evaluate on full multi-round games
        """
        self.full_game_mode = full_game_mode

    def evaluate(self, model1, model2=None, num_games=400):
        """
        Evaluate model(s) on games or full game sequences.

        Args:
            model1: Primary model to evaluate
            model2: Optional second model (if None, evaluate vs random)
            num_games: Number of games/sequences to play

        Returns:
            Evaluation metrics (ELO, win rate, scores)
        """
        if self.full_game_mode:
            return self._evaluate_full_games(model1, model2, num_games)
        else:
            return self._evaluate_rounds(model1, model2, num_games)

    def _evaluate_full_games(self, model1, model2, num_sequences):
        """Evaluate on full multi-round game sequences."""

        results = {
            'model1_total_scores': [],
            'model2_total_scores': [],
            'sequences_played': 0,
        }

        # Sample game configurations (P, C) from player distribution
        configs = self._sample_full_game_configs(num_sequences)

        for config_idx, (num_players, start_cards) in enumerate(configs):
            # Generate full round sequence for this (P, C)
            round_sequence = self._generate_round_sequence(num_players, start_cards)

            # Initialize cumulative scores
            cumulative_scores = [0] * num_players

            # Play all rounds in sequence
            for round_idx, cards_to_deal in enumerate(round_sequence):
                # Build game context
                game_context = GameContext(
                    cumulative_scores=list(cumulative_scores),
                    rounds_completed=round_idx,
                    total_rounds=len(round_sequence),
                    previous_cards=round_sequence[:round_idx],
                    num_players=num_players,
                    start_cards=start_cards,
                    phase=self._get_phase(round_idx, round_sequence),
                )

                # Play single round with context
                round_scores = self._play_single_round(
                    num_players,
                    cards_to_deal,
                    game_context,
                    model1,
                    model2,
                )

                # Update cumulative scores
                for i in range(num_players):
                    cumulative_scores[i] += round_scores[i]

            # Record final total scores
            results['model1_total_scores'].append(cumulative_scores[0])  # Assuming player 0 uses model1
            results['model2_total_scores'].append(cumulative_scores[1])  # Assuming player 1 uses model2
            results['sequences_played'] += 1

        # Compute win rate and ELO
        model1_wins = sum(
            1 for s1, s2 in zip(results['model1_total_scores'], results['model2_total_scores'])
            if s1 > s2
        )
        win_rate = model1_wins / num_sequences if num_sequences > 0 else 0.0

        # Calculate ELO change
        elo_change = self._calculate_elo_change(win_rate)

        return {
            'win_rate': win_rate,
            'elo_change': elo_change,
            'avg_score_model1': np.mean(results['model1_total_scores']),
            'avg_score_model2': np.mean(results['model2_total_scores']),
            'sequences_played': results['sequences_played'],
        }

    def _generate_round_sequence(self, num_players, start_cards):
        """Generate P-conditional round sequence."""
        # Descending: C, C-1, ..., 2, 1
        descending = list(range(start_cards, 0, -1))

        # Ones: P repetitions of 1-card rounds
        ones = [1] * num_players

        # Ascending: 2, 3, ..., C
        ascending = list(range(2, start_cards + 1))

        return descending + ones + ascending

    def _get_phase(self, round_idx, round_sequence):
        """Determine game phase for this round."""
        start_cards = round_sequence[0]
        num_players = round_sequence.count(1)

        descending_len = start_cards
        ones_start = descending_len
        ones_end = ones_start + num_players

        if round_idx < ones_start:
            return 'descending'
        elif round_idx < ones_end:
            return 'ones'
        else:
            return 'ascending'

    def _sample_full_game_configs(self, num_sequences):
        """Sample (P, C) configurations from player distribution."""
        configs = []

        for _ in range(num_sequences):
            # Sample player count
            num_players = np.random.choice(
                list(self.config.player_distribution.keys()),
                p=list(self.config.player_distribution.values())
            )

            # Sample start cards (conditional on P)
            if num_players == 4:
                start_cards = np.random.choice(
                    [7, 8],
                    p=[
                        self.config.start_card_distribution_4p[7],
                        self.config.start_card_distribution_4p[8]
                    ]
                )
            else:
                start_cards = 7

            configs.append((num_players, start_cards))

        return configs
```

#### Testing:

```bash
# Test round sequence generation
python -c "
def generate_round_sequence(num_players, start_cards):
    descending = list(range(start_cards, 0, -1))
    ones = [1] * num_players
    ascending = list(range(2, start_cards + 1))
    return descending + ones + ascending

# Test cases
print('5p/C=7:', generate_round_sequence(5, 7))
print('4p/C=8:', generate_round_sequence(4, 8))
print('4p/C=7:', generate_round_sequence(4, 7))
"

# Integration test
python -c "
from ml.evaluation.arena import ModelArena
from ml.config import TrainingConfig
from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker

config = TrainingConfig()
network = BlobNet(config)
encoder = StateEncoder()
masker = ActionMasker()

arena = ModelArena(
    config=config,
    encoder=encoder,
    masker=masker,
    full_game_mode=True
)

# Evaluate on 10 full-game sequences
results = arena.evaluate(network, num_games=10)
print(f'Results: {results}')
"
```

#### Acceptance Criteria:

- ✅ `ModelArena(full_game_mode=True)` works
- ✅ Generates correct round sequences for (P, C) combinations
- ✅ Plays all rounds in sequence with cumulative scoring
- ✅ GameContext updated between rounds
- ✅ Total game scores computed correctly
- ✅ Win rate and ELO calculated on total scores
- ✅ Evaluation completes without errors
- ✅ Can evaluate any checkpoint on full games

---

### Session 5: Full-Game Training Mode (4 hours)

**Goal**: Implement `training_on="games"` mode for full multi-round game generation.

**Justification**: Necessary for Phase 2 training - learn game-optimal strategy (not just round-optimal).

#### Key Changes:

1. **Self-play generates full game sequences** (not independent rounds)
2. **Value targets = normalized total game score** (not round score)
3. **Examples collected across all rounds** with proper context

#### Implementation:

**File**: `ml/training/selfplay.py` (~200 lines):

```python
class SelfPlayWorker:
    def generate_game(self, num_players=None, cards_to_deal=None, game_id=None):
        """
        Generate game or full game sequence based on config.training_on.

        Returns:
            List of training examples (positions)
        """
        if self.config.training_on == 'games':
            return self._generate_full_game_sequence(num_players, game_id)
        else:
            return self._generate_independent_round(num_players, cards_to_deal, game_id)

    def _generate_full_game_sequence(self, num_players=None, game_id=None):
        """Generate full multi-round game sequence."""

        if game_id is None:
            game_id = str(uuid.uuid4())

        # Sample game configuration
        if num_players is None:
            # Sample (P, C) from distribution
            num_players = np.random.choice(
                list(self.config.player_distribution.keys()),
                p=list(self.config.player_distribution.values())
            )

        if num_players == 4:
            start_cards = np.random.choice(
                [7, 8],
                p=[
                    self.config.start_card_distribution_4p[7],
                    self.config.start_card_distribution_4p[8]
                ]
            )
        else:
            start_cards = 7

        # Generate round sequence
        round_sequence = self._generate_round_sequence(num_players, start_cards)

        # Initialize cumulative scores
        cumulative_scores = [0] * num_players

        # Collect all positions across all rounds
        all_positions = []

        # Play all rounds in sequence
        for round_idx, cards_to_deal in enumerate(round_sequence):
            # Build game context
            game_context = GameContext(
                cumulative_scores=list(cumulative_scores),
                rounds_completed=round_idx,
                total_rounds=len(round_sequence),
                previous_cards=round_sequence[:round_idx],
                num_players=num_players,
                start_cards=start_cards,
                phase=self._get_phase(round_idx, round_sequence, num_players, start_cards),
            )

            # Play single round
            round_positions = self._play_single_round_with_context(
                num_players,
                cards_to_deal,
                game_context,
                game_id,
            )

            # Extract round scores
            round_scores = self._extract_round_scores(round_positions)

            # Update cumulative scores
            for i in range(num_players):
                cumulative_scores[i] += round_scores[i]

            # Add round positions to collection
            all_positions.extend(round_positions)

        # Backpropagate TOTAL GAME SCORES to all positions
        max_possible_score = len(round_sequence) * (10 + start_cards)  # Upper bound

        for pos in all_positions:
            player = pos['player']
            # Normalize total game score to [-1, 1] or [0, 1]
            normalized_score = cumulative_scores[player] / max_possible_score
            pos['value'] = normalized_score
            pos['total_game_score'] = cumulative_scores[player]  # Store raw score too

        return all_positions

    def _generate_independent_round(self, num_players, cards_to_deal, game_id):
        """Generate independent round (existing logic)."""
        # Sample config if needed
        if self.config.use_decision_weighted_sampling:
            num_players, start_cards, cards_to_deal, game_context = self.sample_game_config()
        else:
            game_context = None

        # Play single round (existing implementation)
        positions = self._play_single_round_with_context(
            num_players,
            cards_to_deal,
            game_context,
            game_id,
        )

        # Backpropagate round scores (existing logic)
        return positions

    def _play_single_round_with_context(self, num_players, cards_to_deal, game_context, game_id):
        """Play single round and return positions (helper for both modes)."""
        # Initialize game
        game = BlobGame(num_players=num_players, cards_to_deal=cards_to_deal)
        game.start_game()

        positions = []
        move_number = 0

        while not game.is_terminal():
            current_player = game.current_player

            # Get legal actions
            legal_actions = self._get_legal_actions(game)

            # Zero-choice fast path
            if len(legal_actions) == 1:
                game.apply_action(legal_actions[0])
                continue

            # Run MCTS
            mcts_policy = self._run_mcts(game, game_context)

            # Store position
            positions.append({
                'state': self.encoder.encode(game, current_player, game_context),
                'policy': mcts_policy,
                'value': None,  # Filled later
                'player': current_player,
                'game_id': game_id,
                'move_number': move_number,
                'num_players': num_players,
                'cards_dealt': cards_to_deal,
                'game_context': game_context,
            })

            # Select and apply action
            action = self._select_action(mcts_policy, legal_actions, move_number)
            game.apply_action(action)
            move_number += 1

        # Extract round scores
        round_scores = game.get_scores()

        # Store round scores in positions (for independent round mode)
        for pos in positions:
            pos['round_score'] = round_scores[pos['player']]

        return positions

    def _generate_round_sequence(self, num_players, start_cards):
        """Generate P-conditional round sequence."""
        descending = list(range(start_cards, 0, -1))
        ones = [1] * num_players
        ascending = list(range(2, start_cards + 1))
        return descending + ones + ascending

    def _get_phase(self, round_idx, round_sequence, num_players, start_cards):
        """Determine game phase."""
        descending_len = start_cards
        ones_start = descending_len
        ones_end = ones_start + num_players

        if round_idx < ones_start:
            return 'descending'
        elif round_idx < ones_end:
            return 'ones'
        else:
            return 'ascending'
```

**File**: `ml/training/trainer.py` (~30 lines):

```python
class TrainingPipeline:
    def run_iteration(self, iteration: int):
        """Run iteration in rounds or games mode."""

        # Log mode
        mode = self.config.training_on
        logger.info(f"Training mode: {mode}")

        if mode == 'games':
            logger.info(f"Generating {self.config.games_per_iteration} full game sequences...")
            # Self-play engine will use _generate_full_game_sequence()
        else:
            logger.info(f"Generating {self.config.games_per_iteration} independent rounds...")
            # Self-play engine will use _generate_independent_round()

        # Generate games (method dispatches based on config.training_on)
        examples, dist_stats = self.selfplay_engine.generate_games(
            self.config.games_per_iteration
        )

        # Log statistics
        if mode == 'games':
            avg_rounds_per_game = np.mean([
                len(set(ex['game_id'] for ex in examples))  # Approximate
            ])
            logger.info(f"Avg rounds per game: {avg_rounds_per_game:.1f}")

        # Continue with training...
```

#### Testing:

```bash
# Test full-game sequence generation
python -c "
from ml.training.selfplay import SelfPlayWorker
from ml.config import TrainingConfig
from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker

config = TrainingConfig()
config.training_on = 'games'  # Enable full-game mode

network = BlobNet(config)
encoder = StateEncoder()
masker = ActionMasker()

worker = SelfPlayWorker(network, encoder, masker, config=config)

# Generate one full 5p/C=7 game
positions = worker.generate_game(num_players=5)

# Verify structure
print(f'Total positions: {len(positions)}')
print(f'Expect: ~118 (5 players × (7+6+...+1+1+1+1+1+2+...+7) rounds × ~1.4 decisions/round)')

# Check value targets
values = [p['value'] for p in positions if p['value'] is not None]
print(f'Value range: [{min(values):.3f}, {max(values):.3f}]')
"

# Integration test
python ml/train.py --fast --training-on games --iterations 2
# Check logs for:
# - "Training mode: games"
# - "Generating N full game sequences"
# - "Avg rounds per game: ~17" (for 5p/C=7)
```

#### Acceptance Criteria:

- ✅ `config.training_on='games'` triggers full-game mode
- ✅ Full game sequences generated with correct round order
- ✅ Cumulative scores tracked across rounds
- ✅ GameContext updated between rounds
- ✅ Value targets = normalized total game score
- ✅ All examples have proper metadata
- ✅ Training completes without errors
- ✅ Can switch between 'rounds' and 'games' mode via CLI

---

## Phase 3: Operational Tooling (Optional)

### Session 6: External Monitor & Checkpoint Management (8 hours)

**Goal**: Add operational tooling for long training runs.

**Justification**: Training runs are 14+ days - need pause/resume, progress visibility, and disk management.

**Status**: OPTIONAL - Implement only after validating core training works (Phase 1-2 complete).

#### Key Features:

1. **Checkpoint Rotation**:
   - Permanent: every 5 iterations (keep forever)
   - Cache: other iterations (keep last 4, rotate FIFO)
   - Saves disk space: ~50GB vs ~250GB for 500 iterations

2. **Atomic Status File**:
   - `models/checkpoints/status.json` (atomic writes)
   - Fields: iteration, phase, progress, ETA, ELO, knobs

3. **External Monitor Script**:
   - `python ml/monitor.py` (attach/detach anytime)
   - Live TUI with Rich library
   - Keyboard controls: `p` = pause, `q` = quit

4. **Pause/Resume**:
   - Control signal: `models/checkpoints/control.signal`
   - Iteration-boundary pause (safe, no mid-iteration state)
   - Resume via `--resume` flag

#### Implementation:

See PROGRESSIVE_TRAINING_STRATEGY Session 6 for detailed implementation (lines 79-143).

**Files to create**:
- `ml/monitor.py` (~200 lines)

**Files to modify**:
- `ml/training/trainer.py` (StatusWriter class, checkpoint rotation, control signal)

#### Acceptance Criteria:

- ✅ Permanent checkpoints every 5 iterations
- ✅ Cache checkpoints rotated (max 4 files)
- ✅ status.json updated atomically
- ✅ External monitor displays live progress
- ✅ Pause signal works (iteration-boundary)
- ✅ Resume from checkpoint works
- ✅ Works with tmux detach/reattach

---

## Testing & Validation

### Unit Tests

Add tests for each session:

```bash
# Session 0: MCTS curriculum
python -m pytest ml/training/test_trainer.py::test_mcts_curriculum

# Session 1: Zero-choice fast path
python -m pytest ml/training/test_selfplay.py::test_zero_choice_fast_path

# Session 2: EMA update
python -m pytest ml/training/test_trainer.py::test_ema_update

# Session 3: Dirichlet noise
python -m pytest ml/mcts/test_search.py::test_dirichlet_noise_at_root

# Session 4: Full-game evaluation
python -m pytest ml/evaluation/test_arena.py::test_full_game_evaluation

# Session 5: Full-game training
python -m pytest ml/training/test_selfplay.py::test_full_game_generation
```

### Integration Tests

```bash
# Test complete pipeline (Phase 1)
python ml/train.py \
    --fast \
    --training-on rounds \
    --enable-curriculum \
    --iterations 5

# Test full-game mode (Phase 2)
python ml/train.py \
    --fast \
    --training-on games \
    --iterations 2

# Test checkpoint resume
python ml/train.py \
    --fast \
    --training-on rounds \
    --iterations 10 \
    --resume models/checkpoints/checkpoint_5.pth
```

### Performance Benchmarks

```bash
# Baseline throughput (after Session 1)
python benchmarks/performance/benchmark_selfplay.py \
    --workers 32 \
    --games 100 \
    --config medium

# Expected: ~420 games/min (364 baseline + 16% from zero-choice)

# Full-game throughput (Session 5)
python benchmarks/performance/benchmark_selfplay.py \
    --workers 32 \
    --games 100 \
    --mode games

# Expected: ~73 games/min (full 17-round sequences)
```

---

## Training Execution Plan

### Phase 1: Independent Rounds (400-500 iterations)

```bash
# Start Phase 1 training
python ml/train.py \
    --training-on rounds \
    --enable-curriculum \
    --iterations 500 \
    --device cuda \
    --workers 32

# Expected duration: ~10-12 days @ 420 games/min
# Output: models/checkpoints/phase1_best.pth
```

**Checkpoints of interest**:
- Iteration 50: First MCTS bump (2 det × 25 sims)
- Iteration 150: Second bump (3 det × 35 sims)
- Iteration 300: Third bump (4 det × 45 sims)
- Iteration 450-500: Final strength (5 det × 50 sims)

### Phase 1 Evaluation (after iteration 400-500)

```bash
# Evaluate best Phase 1 model on full games
python -c "
from ml.evaluation.arena import ModelArena
from ml.config import TrainingConfig
import torch

config = TrainingConfig()
arena = ModelArena(config, full_game_mode=True)

# Load Phase 1 best checkpoint
checkpoint = torch.load('models/checkpoints/phase1_best.pth')
network = BlobNet(config)
network.load_state_dict(checkpoint['model_state_dict'])

# Evaluate on 400 full-game sequences
results = arena.evaluate(network, num_games=400)

print(f'Full-game ELO: {results[\"elo\"]}')
print(f'Avg total score: {results[\"avg_score_model1\"]:.1f}')
print(f'Win rate: {results[\"win_rate\"]:.1%}')
"
```

**Decision point**:
- If model shows context-aware strategy (conservative when ahead): ✅ **Phase 1 SUCCESS, can stop here**
- If model ignores game context or plays inconsistently: ⚠️ **Proceed to Phase 2**

### Phase 2: Full Multi-Round Games (100-200 iterations)

```bash
# Resume from Phase 1 best, train on full games
python ml/train.py \
    --training-on games \
    --enable-curriculum \
    --iterations 100 \
    --resume models/checkpoints/phase1_best.pth \
    --device cuda \
    --workers 32

# Expected duration: ~10 days @ 73 games/min
# Output: models/checkpoints/phase2_best.pth
```

**Note**: Full-game mode is SLOWER (73 games/min vs 420 rounds/min) because each "game" is now a 17-round sequence.

---

## Success Metrics

### After Session 0 (Baseline Complete):
- ✅ MCTS curriculum wired and functional
- ✅ `--training-on` CLI flag works
- ✅ Distribution logging shows correct percentages
- ✅ Fast test run completes without errors

### After Session 1 (Zero-Choice):
- ✅ Skip rate ~14% (±3%)
- ✅ Throughput: ~420 games/min (16% speedup)
- ✅ Policy entropy increases (forced moves excluded)

### After Session 2 (EMA):
- ✅ Training stable (no divergence)
- ✅ EMA model used for self-play
- ✅ Policy targets sharpen over training (τ: 1.0 → 0.7)

### After Session 3 (Exploration):
- ✅ Dirichlet noise applied at root
- ✅ Policy entropy higher in self-play
- ✅ Evaluation runs without noise

### After Session 4 (Full-Game Eval):
- ✅ Full-game sequences generated correctly
- ✅ Total game scores computed
- ✅ ELO calculated on full games

### After Session 5 (Full-Game Training):
- ✅ Full-game mode generates 17-round sequences
- ✅ Value targets = total game score
- ✅ Can train for 100+ iterations in games mode

### Phase 1 Training Complete:
- ✅ 400-500 iterations @ ~420 rounds/min
- ✅ ELO progression shows improvement
- ✅ Model handles all card counts (1-8)
- ✅ Context-aware bidding emerges
- ✅ Full-game evaluation: ELO ~1400-1600

### Phase 2 Training Complete (Optional):
- ✅ 100 iterations @ ~73 games/min
- ✅ Game-optimal strategy learned
- ✅ Multi-round planning visible
- ✅ Full-game evaluation: ELO ~1700-1900

---

## Timeline & Effort Summary

### Implementation (26 hours):

| Session | Effort | Status |
|---------|--------|--------|
| Session 0: Wire MCTS Curriculum | 4h | Required |
| Session 1: Zero-Choice Fast Path | 2h | Core |
| Session 2: EMA + Progressive Targets | 4h | Core |
| Session 3: Fixed Exploration | 4h | Core |
| Session 4: Full-Game Evaluation | 4h | Core |
| Session 5: Full-Game Training | 4h | Core |
| Session 6: External Monitor | 8h | Optional |
| **Total** | **26h** | **22h core + 4h baseline** |

### Training (14-24 days):

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 1: Independent Rounds | 10-12 days | Required |
| Phase 1: Evaluation | 4 hours | Required |
| Phase 2: Full Games | 10 days | Optional |
| **Total** | **10-22 days** | **Depends on Phase 1 results** |

---

## Next Steps

1. ✅ **Approve this plan** - Review and confirm approach
2. 🔨 **Implement Session 0** - Complete missing baseline (4h)
3. 🔨 **Implement Sessions 1-3** - Core training enhancements (10h)
4. 🔨 **Implement Sessions 4-5** - Full-game infrastructure (8h)
5. 🧪 **Validate pipeline** - Run fast tests, fix bugs (2-4h)
6. 🚀 **Start Phase 1 training** - 500 iterations (~10-12 days)
7. 📊 **Evaluate Phase 1** - Test on full games (4h)
8. 🎯 **Decision: Phase 2?** - If needed, train on full games (~10 days)
9. 🎉 **Training complete!** - Export ONNX, proceed to Phase 5

---

## Document History

- **2025-11-12**: Initial creation (consolidates PROGRESSIVE + HYBRID plans)
- **Supersedes**: PROGRESSIVE_TRAINING_STRATEGY.md, HYBRID_TRAINING_PLAN.md Sessions 3-4
- **Status**: Final plan, ready for implementation
