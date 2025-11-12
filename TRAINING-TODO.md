# Training TODO - Final Implementation Plan

**Created**: 2025-11-12
**Updated**: 2025-11-12 (added LR fix + linear curriculum + checkpoint naming)
**Approved**: 2025-11-12 ✅
**Status**: Approved - Ready for implementation
**Total Effort**: 23 hours core + 8 hours optional = 31 hours
**Expected Training Time**: **~7-9 days Phase 1** (with linear curriculum) + ~35-40 days Phase 2 (optional)

---

## Terminology Clarification

**Performance Metrics:**
- **Round**: Single deal (e.g., 5-card round). Phase 1 trains on independent rounds.
  - Measured in: **rounds/min** (~360 rounds/min on RTX 4060)
- **Game**: Full multi-round sequence (e.g., 17 rounds for 5p/C=7). Phase 2 trains on complete games.
  - Measured in: **games/min** (~40 games/min on RTX 4060)

**Why the difference?** Phase 1 plays single rounds independently (faster). Phase 2 plays complete 17-round game sequences (slower but learns multi-round strategy).

See CLAUDE.md "Terminology" section for complete definitions.

---

## Current Project State

**Verified Status** (as of 2025-11-12):

✅ **Complete:**
- HYBRID Session 0: Infrastructure (TrainingConfig with distributions, GameContext, DECISION_WEIGHTS)
- HYBRID Session 1: Context encoding (256-dim StateEncoder with game_context parameter)
- HYBRID Session 2: Decision-weighted sampling (sample_game_config() implemented)
- Performance Session 1: Determinization fix (~360 rounds/min achieved for Phase 1 independent rounds, ~10x speedup)

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
            # NOTE: get_mcts_params() expects 1-indexed iteration (docstring),
            # but Python loop is 0-indexed, so we pass iteration + 1
            num_det, sims_per_det = self.config.get_mcts_params(iteration + 1)

            logger.info(
                f"Iteration {iteration + 1}: MCTS curriculum "
                f"{num_det} det × {sims_per_det} sims"
            )

            # Update self-play engine config
            # NOTE: These params must flow through task creation in multiprocess dispatch
            # Simply mutating engine attributes won't affect in-flight workers
            # Ensure _generate_games_multiprocess passes these to worker tasks
            self.selfplay_engine.num_determinizations = num_det
            self.selfplay_engine.simulations_per_determinization = sims_per_det

            # Run iteration with updated MCTS params
            metrics = self.run_iteration(iteration)

            # NEW: Distribution sanity logging (every 10 iterations)
            if (iteration + 1) % 10 == 0 and 'distribution_stats' in metrics:
                self._log_distribution_sanity(metrics)

    def _log_distribution_sanity(self, metrics: Dict[str, Any]):
        """Validate sampled distributions match targets."""
        if 'distribution_stats' not in metrics:
            return

        stats = metrics['distribution_stats']

        # Check player distribution
        total_games = sum(stats['player_counts'].values())

        # Require minimum sample size before validating distribution
        MIN_SAMPLE_SIZE = 100  # Need at least 100 rounds for valid statistics
        if total_games < MIN_SAMPLE_SIZE:
            logger.info(f"  Sample size too small ({total_games} < {MIN_SAMPLE_SIZE}), skipping distribution check")
            return

        for num_players, target_pct in self.config.player_distribution.items():
            actual_count = stats['player_counts'].get(num_players, 0)
            actual_pct = actual_count / total_games if total_games > 0 else 0.0

            logger.info(
                f"  Player {num_players}: {actual_pct:.1%} "
                f"(target {target_pct:.1%})"
            )

            # Assert within ±3% tolerance (only for sufficient sample size)
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
        """
        Generate games and collect distribution stats.

        Returns:
            List[Dict]: Training examples from self-play games
        """

        # Generate games (existing logic)
        all_examples = self._generate_games_multiprocess(num_games, progress_callback)

        # NEW: Collect distribution statistics and attach to examples as metadata
        if self.config.use_decision_weighted_sampling:
            distribution_stats = self._collect_distribution_stats(all_examples)
            # Store stats in a global tracker or log them here
            logger.info(f"Distribution stats: {distribution_stats}")

        return all_examples  # Returns list of examples only

    def _collect_distribution_stats(self, examples: List[Dict]) -> Dict[str, Any]:
        """
        Collect player/card distribution statistics.

        NOTE: Stats should be tracked per-round (not per-example) to avoid
        inflating counts by decision density. Track unique game_ids instead.
        """
        from collections import Counter

        player_counts = Counter()
        card_distributions = {}  # (P, C) -> Counter({c: count})
        seen_game_ids = set()  # Track unique rounds to count once per round

        for example in examples:
            game_id = example.get('game_id')
            if not game_id or game_id in seen_game_ids:
                continue  # Skip if already counted this round

            seen_game_ids.add(game_id)

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
                (P, C): dict(dist)  # Use tuple keys (not string) for consistency
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
    def __init__(self, ..., num_determinizations=None, simulations_per_determinization=None):
        """
        Workers receive MCTS params at task creation time (not via engine mutation).

        Args:
            num_determinizations: Number of determinizations for MCTS (passed per-task)
            simulations_per_determinization: Simulations per determinization (passed per-task)
        """
        # Store params received from task creation
        self.num_determinizations = num_determinizations or self.config.num_determinizations
        self.simulations_per_determinization = simulations_per_determinization or self.config.simulations_per_determinization

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

**5. Implement standardized checkpoint naming** (~80 lines):

**File**: `ml/training/trainer.py` (TrainingPipeline class)

**Checkpoint naming convention:**
```
YYYYMMDD-Blobmaster-v1-{workers}w-{mode}-{mcts}-iter{XXX}[-elo{YYYY}].pth
```

**Components:**
- `YYYYMMDD`: Training start date (e.g., `20251115`)
- `Blobmaster`: Project name
- `v1`: Model architecture version
- `{workers}w`: Number of workers (e.g., `32w`)
- `{mode}`: Training mode (`rounds` or `games`)
- `{mcts}`: MCTS config at this iteration (e.g., `3x35` = 3 det × 35 sims)
- `iter{XXX}`: Iteration number, zero-padded to 3 digits (e.g., `iter023`, `iter150`)
- `[-elo{YYYY}]`: **Optional**, only for permanent checkpoints (e.g., `-elo1450`)

**Implementation:**

```python
class TrainingPipeline:
    def __init__(self, network, encoder, masker, config):
        # ... existing init ...

        # Store training start date for checkpoint naming
        from datetime import datetime
        self.training_start_date = datetime.now().strftime("%Y%m%d")

        logger.info(f"Training session started: {self.training_start_date}")

    def _get_checkpoint_filename(
        self,
        iteration: int,
        elo: Optional[int] = None,
        is_permanent: bool = False
    ) -> str:
        """
        Generate standardized checkpoint filename.

        Args:
            iteration: Current training iteration (0-indexed)
            elo: Optional ELO rating (only for permanent checkpoints)
            is_permanent: Whether this is a permanent checkpoint (every 5 iters)

        Returns:
            Filename following convention: YYYYMMDD-Blobmaster-v1-{params}-iter{XXX}[-elo{YYYY}].pth

        Examples:
            >>> self._get_checkpoint_filename(23, elo=None, is_permanent=False)
            '20251115-Blobmaster-v1-32w-rounds-1x15-iter023.pth'

            >>> self._get_checkpoint_filename(150, elo=1420, is_permanent=True)
            '20251115-Blobmaster-v1-32w-rounds-3x35-iter150-elo1420.pth'
        """
        # Get MCTS params for this iteration (expects 1-indexed)
        num_det, sims = self.config.get_mcts_params(iteration + 1)

        # Build components
        date = self.training_start_date
        project = "Blobmaster"
        version = "v1"
        workers = f"{self.config.num_workers}w"
        mode = self.config.training_on  # 'rounds' or 'games'
        mcts = f"{num_det}x{sims}"
        iter_str = f"iter{iteration:03d}"  # Zero-padded to 3 digits

        # Build filename
        parts = [date, project, version, workers, mode, mcts, iter_str]

        # Add ELO if provided (permanent checkpoints only)
        if is_permanent and elo is not None:
            parts.append(f"elo{elo}")

        filename = "-".join(parts) + ".pth"
        return filename

    def save_checkpoint(self, iteration: int, is_permanent: bool = False):
        """
        Save checkpoint with standardized naming.

        Args:
            iteration: Current iteration
            is_permanent: If True, save to permanent/ subdir with ELO in filename
        """
        # Get ELO if available (for permanent checkpoints)
        elo = None
        if is_permanent and hasattr(self, 'current_elo'):
            elo = int(self.current_elo)

        # Generate filename
        filename = self._get_checkpoint_filename(iteration, elo, is_permanent)

        # Determine save directory
        if is_permanent:
            save_dir = "models/checkpoints/permanent"
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = "models/checkpoints/cache"
            os.makedirs(save_dir, exist_ok=True)

        filepath = os.path.join(save_dir, filename)

        # Save checkpoint
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'scheduler_state_dict': self.trainer.scheduler.state_dict(),
            'elo': elo,
            'config': self.config,
            'training_start_date': self.training_start_date,
        }, filepath)

        logger.info(f"Saved {'permanent' if is_permanent else 'cache'} checkpoint: {filename}")

        return filepath
```

**Checkpoint rotation logic** (detailed in Session 6):
- **Permanent**: Every 5 iterations (iter 5, 10, 15, ..., 500) → `models/checkpoints/permanent/`
- **Cache**: All other iterations → `models/checkpoints/cache/` (keep last 4, rotate FIFO)

**Example filenames:**

**Cache checkpoints:**
```
models/checkpoints/cache/20251115-Blobmaster-v1-32w-rounds-1x15-iter001.pth
models/checkpoints/cache/20251115-Blobmaster-v1-32w-rounds-1x15-iter023.pth
models/checkpoints/cache/20251115-Blobmaster-v1-32w-rounds-2x25-iter087.pth
```

**Permanent checkpoints (with ELO):**
```
models/checkpoints/permanent/20251115-Blobmaster-v1-32w-rounds-1x15-iter005-elo0980.pth
models/checkpoints/permanent/20251115-Blobmaster-v1-32w-rounds-2x25-iter050-elo1120.pth
models/checkpoints/permanent/20251115-Blobmaster-v1-32w-rounds-2x25-iter100-elo1280.pth
models/checkpoints/permanent/20251115-Blobmaster-v1-32w-rounds-3x35-iter150-elo1420.pth
models/checkpoints/permanent/20251115-Blobmaster-v1-32w-rounds-5x50-iter500-elo1650.pth
```

**Phase 2 checkpoints:**
```
models/checkpoints/permanent/20251125-Blobmaster-v1-32w-games-3x35-iter050-elo1720.pth
models/checkpoints/permanent/20251201-Blobmaster-v1-32w-games-5x50-iter100-elo1890.pth
```

#### Testing:

```bash
# Test MCTS curriculum schedule
venv/bin/python -c "
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
- ✅ **Checkpoint naming convention implemented**
- ✅ Filenames include date, version, workers, mode, MCTS params, iteration
- ✅ Permanent checkpoints include ELO in filename
- ✅ Cache vs permanent directories created correctly
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
            legal_cards = game.get_legal_plays(current_player)
            legal_actions = legal_cards

        # NEW: Zero-choice fast path (ONLY for playing phase last-card)
        # Restrict to playing phase to avoid edge cases with dealer bidding constraints
        if len(legal_actions) == 1 and game.game_phase == 'playing':
            # Only one legal action - forced last-card play
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
venv/bin/python -c "
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

- ✅ Last-card plays skip MCTS (len(legal_cards) == 1 AND game.phase == 'playing')
- ✅ Bidding phase is NOT skipped (to avoid edge cases with dealer constraints)
- ✅ No training examples stored for forced last-card plays
- ✅ `forced_action_skips` tracked per iteration
- ✅ Skip rate ~14% (±3%) for typical games
- ✅ Avg policy entropy of stored moves increases (forced moves excluded)
- ✅ Training runs without errors
- ✅ Self-play throughput increases (expect ~16% speedup = ~420 rounds/min for Phase 1)

---

### Session 2: Training Stabilization & Curriculum (5 hours)

**Goal**: Stabilize training with proper LR scheduling, EMA model, progressive targets, and adaptive game curriculum.

**Justification**: Proven techniques from MuZero/AlphaZero - prevents overfitting, fixes critical learning rate bug, and saves ~40 days of training time through adaptive curriculum.

#### Key Concepts:

This session combines four complementary techniques:

**A. Cosine Annealing Learning Rate** (improvement over step decay):
   - **IMPROVEMENT**: Replace StepLR (abrupt jumps) with CosineAnnealingLR (smooth decay)
   - Current StepLR works correctly but has sharp drops at iteration 100, 200, etc.
   - CosineAnnealingLR provides smooth decay from 0.001 → 0.0001 over 500 iterations
   - Standard practice for long training runs (better convergence, fewer oscillations)

**B. Adaptive Training Curriculum** (saves ~3-4 days):
   - **Linear ramp**: 2,000 → 10,000 training units over 500 iterations
   - Formula: `units = 2000 + (iteration × 16)`
   - Smooth increase (no jumps) prevents interaction with MCTS curriculum bumps
   - Average: ~6,000 units/iteration (vs 10,000 fixed baseline)
   - **Phase 1**: Returns "rounds" (independent rounds, `training_on='rounds'`)
   - **Phase 2**: Returns "games" (full game sequences, `training_on='games'`)

**C. EMA (Exponential Moving Average) Model**:
   - Maintain smoothed copy of network weights
   - Use EMA model for self-play (stable policy)
   - Train online model (adapts quickly)
   - Prevents catastrophic forgetting

**D. Progressive Target Sharpening**:
   - Build policy targets from MCTS visit counts with temperature: `π_target ∝ N^(1/τ)`
   - Anneal τ over training: 1.0 (uniform) → 0.7 (sharp) over ~200 iterations
   - Early: explore diverse strategies
   - Late: commit to best actions
   - **IMPORTANT**: MCTS returns untempered probabilities (T=1.0), pipeline applies target τ
     - Avoids double-tempering (MCTS node.get_action_probabilities has T parameter)
     - Separation of concerns: MCTS provides raw visit counts, training controls target sharpness

#### Implementation:

**File**: `ml/config.py` (~30 lines):

```python
class TrainingConfig:
    # ... existing config ...

    def get_training_units_per_iteration(self, iteration: int) -> int:
        """
        Adaptive training curriculum with linear ramp (saves ~3-4 days Phase 1 training).

        Returns different units depending on training mode:
        - Phase 1 (training_on='rounds'): Returns number of ROUNDS (independent rounds)
        - Phase 2 (training_on='games'): Returns number of GAMES (full 17-round sequences)

        Linear ramp from 2,000 → 10,000 over 500 iterations.
        Smooth increase prevents sharp jumps that would interact with MCTS curriculum.

        Args:
            iteration: Current training iteration (0-indexed, so iter 0 = first iteration)

        Returns:
            Number of training units (rounds or games) for this iteration

        Examples:
            >>> config = TrainingConfig()
            >>> config.get_training_units_per_iteration(0)    # First iteration
            2000
            >>> config.get_training_units_per_iteration(50)
            2800
            >>> config.get_training_units_per_iteration(250)
            6000
            >>> config.get_training_units_per_iteration(499)  # Last iteration
            9984
            >>> config.get_training_units_per_iteration(500)
            10000  # Capped at max
        """
        # Linear ramp: +16 units per iteration, capped at 10,000
        return min(2000 + (iteration * 16), 10000)
```

**File**: `ml/training/trainer.py` (~150 lines):

```python
class NetworkTrainer:
    def __init__(self, network, config):
        # ... existing init ...

        self.optimizer = optim.Adam(
            network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # IMPROVEMENT: Replace StepLR with CosineAnnealingLR for smoother decay
        # OLD CODE (works, but has abrupt jumps):
        # self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.1)
        # ^ Steps per-iteration (not per-epoch), causing sharp drops at:
        #   - Iteration 100: 0.001 → 0.0001 (sharp drop)
        #   - Iteration 200: 0.0001 → 0.00001 (sharp drop)
        #   These abrupt changes can cause training instability

        # NEW: Cosine annealing (smooth, gradual decay)
        from torch.optim.lr_scheduler import CosineAnnealingLR

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=500,  # Total iterations (matches training plan)
            eta_min=0.0001,  # Minimum LR (10x smaller than initial 0.001)
        )

        logger.info("Using CosineAnnealingLR scheduler (per-iteration, not per-epoch)")

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
        """Run iteration with adaptive curriculum, EMA, and progressive targets."""

        # NEW: Get adaptive training unit count for this iteration
        training_units = self.config.get_training_units_per_iteration(iteration)
        unit_type = "rounds" if self.config.training_on == 'rounds' else "games"
        logger.info(
            f"Iteration {iteration + 1}: Generating {training_units:,} {unit_type} "
            f"(adaptive curriculum: linear ramp)"
        )

        # Phase 1: Self-play (use EMA model if enabled)
        model_for_selfplay = self.ema_model if self.use_ema_for_selfplay else self.network

        # Update self-play engine to use EMA model
        self.selfplay_engine.update_model(model_for_selfplay)

        # NEW: Generate adaptive number of training units (rounds or games)
        examples = self.selfplay_engine.generate_games(training_units)
        # Distribution stats are logged inside generate_games() if enabled

        # Phase 2: Training
        # Get policy target temperature for this iteration
        tau = self.get_pi_target_tau(iteration)
        logger.info(f"Policy target τ = {tau:.3f}")

        # Transform MCTS visit counts to policy targets with temperature
        # NOTE: Tempering happens BEFORE adding to replay buffer
        # IMPORTANT: MCTS returns untempered probabilities (temperature=1.0)
        # We apply target sharpening here to control training signal
        for example in examples:
            visit_counts = example['policy']  # Raw visit counts from MCTS (untempered)

            # Apply temperature: π_target ∝ N^(1/τ)
            if tau != 1.0:
                visit_counts_tempered = np.power(visit_counts, 1.0 / tau)
            else:
                visit_counts_tempered = visit_counts

            # Normalize to probabilities and REPLACE policy field
            total = visit_counts_tempered.sum()
            if total > 0:
                example['policy'] = visit_counts_tempered / total
            else:
                example['policy'] = visit_counts  # Fallback (shouldn't happen)

        # Add tempered examples to replay buffer
        for example in examples:
            self.replay_buffer.add(example)

        # Train online model (not EMA) using replay buffer
        train_metrics = self.trainer.train_epoch(
            replay_buffer=self.replay_buffer,
            batch_size=self.config.batch_size
        )

        # Phase 3: Update LR scheduler (per-iteration, not per-epoch!)
        self.trainer.scheduler.step()
        current_lr = self.trainer.scheduler.get_last_lr()[0]
        logger.info(f"Learning rate: {current_lr:.6f}")

        # Phase 4: Update EMA model
        self.update_ema_model()
        logger.info(f"EMA model updated (decay={self.ema_decay})")

        # Phase 5: Evaluation (use online model, not EMA)
        eval_metrics = self.evaluator.evaluate(
            self.network,  # Evaluate online model
            num_games=self.config.eval_games
        )

        return {
            **train_metrics,
            **eval_metrics,
            'pi_target_tau': tau,
            'ema_decay': self.ema_decay,
            'learning_rate': current_lr,
            'training_units_generated': training_units,
            'unit_type': unit_type,
        }
```

**File**: `ml/training/selfplay.py` (~40 lines):

```python
class SelfPlayEngine:
    def update_model(self, new_model: BlobNet):
        """
        Update the model used for self-play.

        IMPORTANT: For multiprocessing mode, workers receive a static network_state
        snapshot at creation (ml/training/selfplay.py:731). They won't see EMA
        updates unless we refresh them. Three approaches:

        1. Save EMA to disk, pass checkpoint path to workers (disk I/O overhead)
        2. Recreate Pool each iteration with fresh network_state (pool overhead)
        3. Use ThreadPoolExecutor instead (shares network automatically)

        Current implementation uses approach #1 for multiprocessing compatibility.
        """
        # Save EMA model to temporary checkpoint
        checkpoint_path = "models/checkpoints/ema_current.pth"
        torch.save({
            'model_state_dict': new_model.state_dict(),
        }, checkpoint_path)
        logger.info(f"EMA model saved to {checkpoint_path} for worker reload")

        # Update internal reference (for threading mode)
        self.network = new_model
        self.ema_checkpoint_path = checkpoint_path

        # For multiprocessing: update network_state so next batch uses EMA
        self.network_state = new_model.state_dict()

    def _worker_init(self, worker_id: int):
        """
        Worker initialization function (called once per worker at pool creation).

        NOTE: In multiprocessing mode, workers are initialized once and reused.
        To get EMA updates, workers must reload from checkpoint before each task.
        """
        # Load latest EMA checkpoint if available
        if hasattr(self, 'ema_checkpoint_path') and os.path.exists(self.ema_checkpoint_path):
            checkpoint = torch.load(self.ema_checkpoint_path)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            logger.debug(f"Worker {worker_id} loaded EMA model from {self.ema_checkpoint_path}")
```

**Implementation Note**:
- **Multiprocessing Challenge**: Workers get snapshot at init, don't auto-update
- **Option 1** (Recommended): Save EMA to disk each iteration, update network_state
  - Workers created with fresh network_state get latest EMA automatically
  - Disk I/O overhead: ~50-100ms per save/load
- **Option 2** (Alternative): Recreate Pool each iteration
  - Higher overhead: ~500ms-1s per recreation
  - Simpler: no checkpoint file management
- **Option 3** (Best for CUDA): Use ThreadPoolExecutor
  - Workers share network automatically (no reload needed)
  - Already implemented for GPU mode (use_thread_pool=True when device='cuda')

**Chosen Approach**: Option 1 for CPU mode, Option 3 (threads) for GPU mode.

#### Testing:

```bash
# Test 1: Cosine annealing LR schedule
venv/bin/python -c "
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Create dummy optimizer
optimizer = Adam([torch.randn(10, 10)], lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=0.0001)

# Test schedule at key iterations
print('Cosine Annealing LR Schedule:')
for i in [0, 50, 100, 250, 400, 499]:
    for _ in range(i):
        scheduler.step()
    lr = scheduler.get_last_lr()[0]
    print(f'Iter {i:3d}: LR = {lr:.6f}')
    # Reset for next test
    optimizer = Adam([torch.randn(10, 10)], lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=0.0001)

# Expected:
# Iter 0:   LR ≈ 0.001000 (start)
# Iter 50:  LR ≈ 0.000900 (slight decay)
# Iter 100: LR ≈ 0.000700 (moderate)
# Iter 250: LR ≈ 0.000300 (halfway)
# Iter 400: LR ≈ 0.000150 (near end)
# Iter 499: LR ≈ 0.000100 (minimum)
"

# Test 2: Adaptive linear curriculum
venv/bin/python -c "
from ml.config import TrainingConfig

config = TrainingConfig()

# Test linear ramp schedule
print('Adaptive Training Curriculum (Linear Ramp):')
for i in [0, 50, 100, 150, 250, 300, 400, 499, 500]:
    units = config.get_training_units_per_iteration(i)
    print(f'Iter {i:3d}: {units:,} training units')

# Expected (linear ramp: 2000 + iter*16):
# Iter 0:    2,000 (start)
# Iter 50:   2,800
# Iter 100:  3,600
# Iter 150:  4,400
# Iter 250:  6,000 (midpoint)
# Iter 300:  6,800
# Iter 400:  8,400
# Iter 499:  9,984
# Iter 500: 10,000 (capped)
"

# Test 3: EMA update
venv/bin/python -c "
import torch
import copy
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

# Test 4: Progressive tau schedule
venv/bin/python -c "
# Simulate tau schedule
print('Progressive Target Sharpening (τ):')
for i in [0, 50, 100, 150, 200, 250, 300]:
    progress = min(i / 200, 1.0)
    tau = 1.0 - progress * (1.0 - 0.7)
    print(f'Iter {i:3d}: τ = {tau:.3f}')

# Expected:
# Iter 0-100:   τ decreases from 1.0 → 0.85
# Iter 100-200: τ decreases from 0.85 → 0.70
# Iter 200+:    τ stays at 0.70
"

# Test 5: Integration test with all features
python ml/train.py --fast --iterations 5 --training-on rounds --enable-curriculum
# Check logs for:
# - "Using CosineAnnealingLR scheduler (per-iteration, not per-epoch)"
# - "Iteration N: Generating X rounds (adaptive curriculum: linear ramp)"
# - "Learning rate: 0.XXXXXX"
# - "Policy target τ = X.XXX"
# - "EMA model updated"
# Verify unit counts match linear ramp (iter 1: 2016, iter 2: 2032, ...)
```

#### Acceptance Criteria:

**A. Learning Rate (Bug Fix):**
- ✅ StepLR removed (old buggy scheduler eliminated)
- ✅ CosineAnnealingLR active (steps per-iteration, not per-epoch)
- ✅ LR logged each iteration
- ✅ LR smoothly decreases: 0.001 → 0.0001 over 500 iterations
- ✅ No rapid LR decay at iteration 10 (bug is fixed)

**B. Adaptive Training Curriculum:**
- ✅ `get_training_units_per_iteration()` method implemented in TrainingConfig
- ✅ Training units follow linear ramp: 2k → 10k (formula: 2000 + iter*16)
- ✅ Unit type ("rounds" or "games") logged correctly per training mode
- ✅ Linear schedule verified via test script
- ✅ No sharp jumps (smooth increase every iteration)

**C. EMA Model:**
- ✅ EMA model maintained (decay=0.997)
- ✅ Self-play uses EMA model
- ✅ Training updates online model (not EMA)
- ✅ EMA update logged each iteration
- ✅ Evaluation uses online model (not EMA)

**D. Progressive Target Sharpening:**
- ✅ Policy target τ anneals: 1.0 → 0.7 over 200 iterations
- ✅ Tempered targets applied before adding to replay buffer
- ✅ τ value logged each iteration
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
class Arena:
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
venv/bin/python -c "
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
class Arena:
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
venv/bin/python -c "
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
venv/bin/python -c "
from ml.evaluation.arena import Arena
from ml.config import TrainingConfig
from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker

config = TrainingConfig()
network = BlobNet(config)
encoder = StateEncoder()
masker = ActionMasker()

arena = Arena(
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

- ✅ `Arena(full_game_mode=True)` works
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
        examples = self.selfplay_engine.generate_games(
            self.config.games_per_iteration
        )
        # Distribution stats are logged inside generate_games() if enabled

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
venv/bin/python -c "
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

# Expected: ~420 rounds/min (360 baseline + 16% from zero-choice, Phase 1 independent rounds)

# Full-game throughput (Session 5)
python benchmarks/performance/benchmark_selfplay.py \
    --workers 32 \
    --games 100 \
    --mode games

# Expected: ~40 games/min (full 17-round sequences, Phase 2 multi-round games)
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

# Expected duration: ~7-9 days with linear curriculum (down from ~10-12 days!)
# Breakdown (Phase 1 trains on ROUNDS, not full games):
#   - Average rounds/iter: 6,000 (linear ramp from 2k → 10k)
#   - Average throughput: ~360-420 rounds/min (varies with MCTS curriculum)
#   - Time per iteration: ~15-20 minutes (average)
#   - Total: 500 iters × 17.5 min = ~145 hours = ~6 days continuous
#   - Buffer for slower MCTS late: +1-3 days → 7-9 days total
# Savings: ~3-4 days vs fixed 10k rounds throughout
# Output: models/checkpoints/permanent/YYYYMMDD-Blobmaster-v1-32w-rounds-5x50-iter500-eloXXXX.pth
```

**Checkpoints of interest**:
- Iteration 50: 2,800 rounds/iter + MCTS bump (1×15 → 2×25 sims)
- Iteration 150: 4,400 rounds/iter + MCTS bump (2×25 → 3×35 sims)
- Iteration 300: 6,800 rounds/iter + MCTS bump (3×35 → 4×45 sims)
- Iteration 450: 9,200 games/iter + MCTS bump (4×45 → 5×50 sims)
- Iteration 500: 10,000 games/iter (final strength, 5×50 sims MCTS)

### Phase 1 Evaluation (after iteration 400-500)

```bash
# Evaluate best Phase 1 model on full games
venv/bin/python -c "
from ml.evaluation.arena import Arena
from ml.config import TrainingConfig
import torch

config = TrainingConfig()
arena = Arena(config, full_game_mode=True)

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

# Expected duration: ~35-40 days @ ~40 games/min (Phase 2 full multi-round games)
# Output: models/checkpoints/phase2_best.pth
```

**Note**: Full-game mode is SIGNIFICANTLY SLOWER (~40 games/min vs ~420 rounds/min) because each "game" is now a complete 17-round sequence instead of a single independent round.

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

### After Session 2 (Training Stabilization & Curriculum):
- ✅ **LR Bug Fixed**: No rapid decay at iteration 10 (cosine annealing working)
- ✅ Learning rate smoothly decreases: 0.001 → 0.0001 over 500 iterations
- ✅ **Adaptive curriculum active**: Linear ramp 2k → 10k rounds/iter
- ✅ Iteration times decrease early (fewer games = faster), increase late (more games)
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

### Implementation (27 hours):

| Session | Effort | Status |
|---------|--------|--------|
| Session 0: Wire MCTS Curriculum | 4h | Required |
| Session 1: Zero-Choice Fast Path | 2h | Core |
| Session 2: Training Stabilization & Curriculum | 5h | Core (includes LR fix + game curriculum) |
| Session 3: Fixed Exploration | 4h | Core |
| Session 4: Full-Game Evaluation | 4h | Core |
| Session 5: Full-Game Training | 4h | Core |
| Session 6: External Monitor | 8h | Optional |
| **Total** | **27h** | **23h core + 4h baseline** |

### Training (7-47 days):

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 1: Independent Rounds | **7-9 days** (with adaptive curriculum) | Required |
| Phase 1: Evaluation | 4 hours | Required |
| Phase 2: Full Games | 35-40 days | Optional |
| **Total** | **7-49 days** | **Depends on Phase 1 results** |

**Key Improvement**: Adaptive game curriculum saves ~3-4 days in Phase 1 (down from 10-12 days to 7-9 days)!

---

## Next Steps

1. ✅ **Approved this plan** - Review and confirm approach (20251112)
2. 🔨 **Implement Session 0** - Complete missing baseline (4h)
3. 🔨 **Implement Sessions 1-3** - Core training enhancements (11h, includes LR fix + curriculum)
4. 🔨 **Implement Sessions 4-5** - Full-game infrastructure (8h)
5. 🧪 **Validate pipeline** - Run fast tests, fix bugs (2-4h)
6. 🚀 **Start Phase 1 training** - 500 iterations (~7-9 days with adaptive curriculum)
7. 📊 **Evaluate Phase 1** - Test on full games (4h)
8. 🎯 **Decision: Phase 2?** - If needed, train on full games (~35-40 days)
9. 🎉 **Training complete!** - Export ONNX, proceed to Phase 5

---

## Document History

- **2025-11-12**: Initial creation (consolidates PROGRESSIVE + HYBRID plans)
- **2025-11-12 (Update 1)**: Major enhancements - LR improvement, linear curriculum, standardized naming
  - **Scheduler improvement**: Replace StepLR with CosineAnnealingLR for smoother decay
  - **Efficiency gain**: Linear adaptive curriculum saves ~3-4 days in Phase 1 training
  - **Standardized naming**: Checkpoint filenames now include date, version, MCTS params, ELO
  - **Terminology fix**: Phase 1 uses "rounds" (independent), Phase 2 uses "games" (full sequences)
  - Updated Session 2: "EMA + Progressive Targets" → "Training Stabilization & Curriculum" (5h)
  - Added checkpoint naming to Session 0 (infrastructure setup)
  - Linear ramp (2k → 10k) prevents sharp jumps that would interact with MCTS curriculum
- **2025-11-12 (Update 2)**: Critical bug fixes from external review
  - **FIXED**: Iteration indexing off-by-one (use `iteration + 1` for get_mcts_params)
  - **FIXED**: Distribution stats key type mismatch (tuple keys, not strings)
  - **FIXED**: API naming `game.phase` → `game.game_phase`
  - **FIXED**: Distribution stats now tracked per-round (not per-example)
  - **ADDED**: Minimum sample size check (100 rounds) before distribution validation
  - **ADDED**: EMA model refresh documentation for multiprocessing mode
  - **CLARIFIED**: Temperature conventions (MCTS returns T=1.0, pipeline applies target τ)
  - **CORRECTED**: LR scheduler rationale (improvement, not bug fix)
- **Supersedes**: PROGRESSIVE_TRAINING_STRATEGY.md, HYBRID_TRAINING_PLAN.md Sessions 3-4
- **Status**: Final plan with critical fixes, ready for implementation
