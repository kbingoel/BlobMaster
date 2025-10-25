# PLAN-Phase-4.md
# Implementation Plan: Self-Play Training Pipeline

**Phase**: 4 - Self-Play Training Pipeline
**Goal**: Automated training loop generating strong models through self-play

---

## Overview

Implement the AlphaZero-style self-play training pipeline that generates games using the current model, stores experience in a replay buffer, and trains the neural network to improve its play quality. This is the core training infrastructure that will produce strong AI models.

**Key Components**:
1. **Self-Play Engine**: Generate games using MCTS + current model
2. **Replay Buffer**: Store and sample training experiences
3. **Training Loop**: Update network from replay buffer data
4. **Evaluation Pipeline**: Test new models vs previous best
5. **ELO Tracking**: Measure model improvement over time

This phase transforms the AI from a static model to a continuously learning system.

---

## File Structure to Create

```
ml/
├── training/
│   ├── __init__.py              # Package initialization (NEW)
│   ├── selfplay.py              # Parallel game generation (SESSION 1-2)
│   ├── replay_buffer.py         # Experience storage (SESSION 3)
│   ├── trainer.py               # Training orchestration (SESSION 4-5)
│   └── test_training.py         # Training tests (ALL SESSIONS)
│
├── evaluation/
│   ├── __init__.py              # Package initialization (NEW)
│   ├── arena.py                 # Model vs model tournaments (SESSION 6)
│   ├── elo.py                   # ELO calculation (SESSION 6)
│   └── test_evaluation.py      # Evaluation tests (SESSION 6)
│
├── train.py                     # Main training entry point (SESSION 7)
└── config.py                    # Training configuration (SESSION 7)
```

---

## Prerequisites

Before starting Phase 4, ensure Phase 3 is complete:
- ✅ `ml/mcts/belief_tracker.py` fully implemented (COMPLETE - Phase 3)
- ✅ `ml/mcts/determinization.py` fully implemented (COMPLETE - Phase 3)
- ✅ `ml/mcts/search.py` with ImperfectInfoMCTS (COMPLETE - Phase 3)
- ✅ All Phase 3 tests passing - 333 tests total (COMPLETE)
- ✅ Imperfect info MCTS can play complete games (COMPLETE)

---

## Detailed Session Breakdown

### SESSION 1: Self-Play Engine - Foundation

**Goal**: Implement basic self-play game generation using MCTS.

#### 1.1 Setup (10 min)
- [ ] Create `ml/training/` directory
- [ ] Create `ml/training/__init__.py`
- [ ] Create `ml/training/selfplay.py`
- [ ] Create `ml/training/test_training.py`
- [ ] Review existing MCTS code to understand integration

#### 1.2 Design Self-Play Architecture (20 min)

Document the self-play approach:
```python
"""
Self-Play Game Generation for AlphaZero-Style Training

Core Concept:
    - Generate games using current model + MCTS
    - Store (state, MCTS_policy, final_outcome) tuples for training
    - Use exploration noise to ensure diverse gameplay
    - Run multiple games in parallel for efficiency

Game Generation Flow:
    1. Initialize game with random setup
    2. For each decision point:
        - Run MCTS with current model
        - Get action probabilities (visit counts)
        - Sample action with temperature-based exploration
        - Store (state, policy, None) for training
    3. Play until game ends
    4. Back-propagate final outcome to all stored positions
    5. Return training examples

Key Design Decisions:
    - Use imperfect info MCTS (realistic hidden information)
    - Temperature schedule: high early (exploration), low late (exploitation)
    - Store full game history for later analysis
    - Support variable player counts (3-8 players)

Training Example Format:
    {
        'state': encoded_state,          # 256-dim tensor
        'policy': action_probabilities,  # MCTS visit counts
        'value': final_score,            # Outcome from this player's perspective
        'player_position': int,          # Which player made this decision
        'game_id': str,                  # For tracking game history
        'move_number': int,              # Position in game
    }
"""
```

#### 1.3 Implement SelfPlayWorker Class (60 min)

Function signatures and descriptions:

```python
# ml/training/selfplay.py

class SelfPlayWorker:
    """
    Worker that generates self-play games for training.

    Runs MCTS-guided game generation and collects training examples.
    """

    def __init__(
        self,
        network: BlobNet,
        encoder: StateEncoder,
        masker: ActionMasker,
        num_determinizations: int = 3,
        simulations_per_determinization: int = 30,
        temperature_schedule: Optional[Callable[[int], float]] = None,
        use_imperfect_info: bool = True,
    ):
        """
        Initialize self-play worker.

        Args:
            network: Neural network for MCTS
            encoder: State encoder
            masker: Action masker
            num_determinizations: Determinizations per MCTS search
            simulations_per_determinization: MCTS simulations per world
            temperature_schedule: Function mapping move_number -> temperature
            use_imperfect_info: Use imperfect info MCTS (vs perfect info)
        """
        pass

    def generate_game(
        self,
        num_players: int = 4,
        cards_to_deal: int = 5,
        game_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate a single self-play game.

        Args:
            num_players: Number of players in the game
            cards_to_deal: Cards to deal per player
            game_id: Optional game identifier

        Returns:
            List of training examples (one per decision point)
        """
        pass

    def _play_bidding_phase(
        self,
        game: BlobGame,
        examples: List[Dict[str, Any]],
        move_number: int,
    ) -> int:
        """
        Play the bidding phase and collect examples.

        Args:
            game: Current game state
            examples: List to append training examples to
            move_number: Current move number (for temperature)

        Returns:
            Updated move number
        """
        pass

    def _play_trick_taking_phase(
        self,
        game: BlobGame,
        examples: List[Dict[str, Any]],
        move_number: int,
    ) -> int:
        """
        Play the trick-taking phase and collect examples.

        Args:
            game: Current game state
            examples: List to append training examples to
            move_number: Current move number

        Returns:
            Updated move number
        """
        pass

    def _select_action(
        self,
        action_probs: Dict[int, float],
        temperature: float,
    ) -> int:
        """
        Select action using temperature-based sampling.

        Args:
            action_probs: Action probabilities from MCTS
            temperature: Exploration temperature (0=greedy, 1=stochastic)

        Returns:
            Selected action index
        """
        pass

    def _backpropagate_outcome(
        self,
        examples: List[Dict[str, Any]],
        final_scores: Dict[int, int],
    ):
        """
        Back-propagate final game outcome to all training examples.

        Args:
            examples: List of training examples from the game
            final_scores: Final scores for each player
        """
        pass

    def get_default_temperature_schedule(self) -> Callable[[int], float]:
        """
        Get default temperature schedule.

        Returns:
            Function mapping move_number -> temperature
            - Early game (moves 0-10): temperature = 1.0 (high exploration)
            - Mid game (moves 11-20): temperature = 0.5 (moderate)
            - Late game (moves 21+): temperature = 0.1 (near-greedy)
        """
        pass
```

#### 1.4 Basic Tests (30 min)

Test function signatures:

```python
# ml/training/test_training.py

class TestSelfPlayWorker:
    def test_selfplay_worker_initialization(self):
        """Test self-play worker initializes correctly."""
        pass

    def test_generate_single_game(self):
        """Test generating a single self-play game."""
        pass

    def test_training_examples_format(self):
        """Test training examples have correct format."""
        pass

    def test_temperature_schedule(self):
        """Test temperature schedule changes over moves."""
        pass

    def test_outcome_backpropagation(self):
        """Test outcomes are correctly back-propagated."""
        pass
```

**Deliverable**: SelfPlayWorker generating training examples from single games

---

### SESSION 2: Self-Play Engine - Parallel Execution

**Goal**: Enable parallel self-play game generation for efficiency.

#### 2.1 Parallel Self-Play Design (20 min)

Document parallelization strategy:
```python
"""
Parallel Self-Play for Training Efficiency

Goal: Generate 10,000+ games per training iteration

Approaches:
    1. Multiprocessing: Separate Python processes (avoids GIL)
    2. Process Pool: Fixed number of workers
    3. Shared Network: Workers share read-only network weights
    4. Independent Randomness: Each worker has different seed

Architecture:
    Main Process:
        - Loads neural network
        - Creates process pool (16-32 workers)
        - Distributes game generation tasks
        - Collects training examples

    Worker Processes:
        - Copy of neural network (inference only)
        - Generate N games independently
        - Return training examples to main process

Synchronization:
    - Network weights: Shared memory (read-only)
    - Training examples: Queue or return values
    - No inter-worker communication needed

Performance Targets:
    - 16 workers on Ryzen 7950X (16 cores)
    - ~20 games/minute/worker = 320 games/minute total
    - 10,000 games in ~30 minutes
"""
```

#### 2.2 Implement SelfPlayEngine Class (70 min)

Function signatures:

```python
# ml/training/selfplay.py

class SelfPlayEngine:
    """
    Manages parallel self-play game generation.

    Orchestrates multiple workers to generate games efficiently.
    """

    def __init__(
        self,
        network: BlobNet,
        encoder: StateEncoder,
        masker: ActionMasker,
        num_workers: int = 16,
        num_determinizations: int = 3,
        simulations_per_determinization: int = 30,
        temperature_schedule: Optional[Callable[[int], float]] = None,
    ):
        """
        Initialize self-play engine.

        Args:
            network: Neural network for MCTS
            encoder: State encoder
            masker: Action masker
            num_workers: Number of parallel workers
            num_determinizations: Determinizations per MCTS search
            simulations_per_determinization: MCTS simulations per world
            temperature_schedule: Temperature schedule function
        """
        pass

    def generate_games(
        self,
        num_games: int,
        num_players: int = 4,
        cards_to_deal: int = 5,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple games in parallel.

        Args:
            num_games: Total number of games to generate
            num_players: Players per game
            cards_to_deal: Cards to deal per player
            progress_callback: Optional callback for progress updates

        Returns:
            Flat list of all training examples from all games
        """
        pass

    def _worker_generate_games(
        self,
        worker_id: int,
        num_games: int,
        num_players: int,
        cards_to_deal: int,
        network_state: Dict[str, torch.Tensor],
    ) -> List[Dict[str, Any]]:
        """
        Worker function for parallel game generation.

        Args:
            worker_id: Unique worker identifier
            num_games: Number of games this worker should generate
            num_players: Players per game
            cards_to_deal: Cards to deal
            network_state: Network weights (state dict)

        Returns:
            List of training examples from all games
        """
        pass

    def generate_games_async(
        self,
        num_games: int,
        num_players: int = 4,
        cards_to_deal: int = 5,
    ) -> concurrent.futures.Future:
        """
        Generate games asynchronously (non-blocking).

        Args:
            num_games: Total number of games to generate
            num_players: Players per game
            cards_to_deal: Cards to deal

        Returns:
            Future that will contain training examples when complete
        """
        pass

    def shutdown(self):
        """Shutdown the parallel workers gracefully."""
        pass
```

#### 2.3 Tests for Parallel Execution (30 min)

Test function signatures:

```python
class TestSelfPlayEngine:
    def test_selfplay_engine_initialization(self):
        """Test self-play engine initializes with worker pool."""
        pass

    def test_parallel_game_generation(self):
        """Test generating multiple games in parallel."""
        pass

    def test_training_examples_aggregation(self):
        """Test examples from multiple workers are aggregated correctly."""
        pass

    def test_worker_isolation(self):
        """Test workers don't interfere with each other."""
        pass

    def test_async_generation(self):
        """Test asynchronous game generation."""
        pass

    def test_performance_scaling(self):
        """Test performance scales with number of workers."""
        pass
```

**Deliverable**: Parallel self-play engine generating thousands of games

---

### SESSION 3: Replay Buffer Implementation

**Goal**: Implement circular replay buffer for storing and sampling training data.

#### 3.1 Replay Buffer Design (15 min)

Document replay buffer architecture:
```python
"""
Replay Buffer for Experience Storage

Purpose: Store recent self-play games for training

Key Features:
    - Circular buffer (FIFO when full)
    - Efficient random sampling
    - Optional data augmentation
    - Memory-mapped storage for large buffers

Design:
    - Capacity: 500,000 positions (configurable)
    - Storage format: List of training example dicts
    - Sampling: Uniform random or prioritized
    - Persistence: Save/load to disk for resume training

Training Example Structure:
    {
        'state': np.array,      # Encoded state (256-dim)
        'policy': np.array,     # MCTS policy (max 65 actions)
        'value': float,         # Outcome (-1 to 1 normalized)
        'player_position': int,
        'game_id': str,
        'move_number': int,
    }

Operations:
    - add_examples(examples): Add new examples
    - sample_batch(batch_size): Random sample for training
    - save(path): Persist buffer to disk
    - load(path): Load buffer from disk
    - clear(): Empty buffer
    - __len__(): Current size
"""
```

#### 3.2 Implement ReplayBuffer Class (70 min)

Function signatures:

```python
# ml/training/replay_buffer.py

class ReplayBuffer:
    """
    Circular replay buffer for storing self-play training examples.

    Stores recent game experiences and provides efficient sampling.
    """

    def __init__(
        self,
        capacity: int = 500_000,
        use_prioritization: bool = False,
    ):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of examples to store
            use_prioritization: Use prioritized experience replay
        """
        pass

    def add_examples(self, examples: List[Dict[str, Any]]):
        """
        Add training examples to the buffer.

        Args:
            examples: List of training example dictionaries
        """
        pass

    def sample_batch(
        self,
        batch_size: int,
        device: str = 'cpu',
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random batch of training examples.

        Args:
            batch_size: Number of examples to sample
            device: Device to place tensors on ('cpu' or 'cuda')

        Returns:
            (states, policies, values) tuple of tensors
            - states: (batch_size, 256) tensor
            - policies: (batch_size, 65) tensor (max actions)
            - values: (batch_size,) tensor
        """
        pass

    def save(self, filepath: str):
        """
        Save replay buffer to disk.

        Args:
            filepath: Path to save buffer to
        """
        pass

    def load(self, filepath: str):
        """
        Load replay buffer from disk.

        Args:
            filepath: Path to load buffer from
        """
        pass

    def clear(self):
        """Clear all examples from the buffer."""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about buffer contents.

        Returns:
            Dictionary with:
            - size: Current number of examples
            - capacity: Maximum capacity
            - num_games: Number of unique games
            - avg_game_length: Average game length
            - value_distribution: Histogram of outcome values
        """
        pass

    def __len__(self) -> int:
        """Get current number of examples in buffer."""
        pass

    def is_ready_for_training(self, min_examples: int = 10_000) -> bool:
        """
        Check if buffer has enough examples for training.

        Args:
            min_examples: Minimum examples required

        Returns:
            True if buffer is ready for training
        """
        pass
```

#### 3.3 Data Augmentation (Optional) (20 min)

Function signatures for data augmentation:

```python
# ml/training/replay_buffer.py

def augment_example(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Apply data augmentation to a training example.

    Potential augmentations:
    - None for now (card games don't have obvious symmetries like images)
    - Could explore: hand permutations, suit remapping

    Args:
        example: Original training example

    Returns:
        List of augmented examples (may just return [example])
    """
    pass
```

#### 3.4 Tests (25 min)

Test function signatures:

```python
# ml/training/test_training.py

class TestReplayBuffer:
    def test_replay_buffer_initialization(self):
        """Test replay buffer initializes correctly."""
        pass

    def test_add_examples(self):
        """Test adding examples to buffer."""
        pass

    def test_circular_buffer_overflow(self):
        """Test buffer overwrites old examples when full."""
        pass

    def test_sample_batch(self):
        """Test sampling batches for training."""
        pass

    def test_save_and_load(self):
        """Test persisting buffer to disk."""
        pass

    def test_buffer_statistics(self):
        """Test buffer statistics calculation."""
        pass

    def test_batch_tensor_shapes(self):
        """Test sampled batch tensors have correct shapes."""
        pass
```

**Deliverable**: Replay buffer with efficient storage and sampling

---

### SESSION 4: Training Loop - Network Updates

**Goal**: Implement the core training loop that updates the network.

#### 4.1 Training Loop Design (20 min)

Document training approach:
```python
"""
Neural Network Training Loop

Goal: Update network to match MCTS policies and predict outcomes

Loss Function:
    total_loss = policy_loss + value_loss + regularization_loss

    policy_loss = CrossEntropy(predicted_policy, MCTS_policy)
    value_loss = MSE(predicted_value, actual_outcome)
    regularization_loss = L2_weight_decay

Training Process:
    1. Sample batch from replay buffer
    2. Forward pass through network
    3. Compute losses
    4. Backward pass (gradient computation)
    5. Optimizer step (weight update)
    6. Logging and metrics

Hyperparameters:
    - Batch size: 512
    - Learning rate: 0.001 (with scheduler)
    - Weight decay: 1e-4
    - Epochs per iteration: 10
    - Gradient clipping: max_norm = 1.0

Optimization:
    - Optimizer: Adam
    - LR schedule: Cosine annealing or step decay
    - Mixed precision: FP16 for faster training
"""
```

#### 4.2 Implement NetworkTrainer Class (70 min)

Function signatures:

```python
# ml/training/trainer.py

class NetworkTrainer:
    """
    Manages neural network training from replay buffer.

    Handles optimization, loss computation, and checkpointing.
    """

    def __init__(
        self,
        network: BlobNet,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        policy_loss_weight: float = 1.0,
        value_loss_weight: float = 1.0,
        use_mixed_precision: bool = False,
        device: str = 'cuda',
    ):
        """
        Initialize network trainer.

        Args:
            network: Neural network to train
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            policy_loss_weight: Weight for policy loss
            value_loss_weight: Weight for value loss
            use_mixed_precision: Use FP16 training
            device: Training device ('cuda' or 'cpu')
        """
        pass

    def train_epoch(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 512,
        num_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch on replay buffer data.

        Args:
            replay_buffer: Replay buffer to sample from
            batch_size: Training batch size
            num_batches: Number of batches (None = full epoch)

        Returns:
            Dictionary with:
            - total_loss: Average total loss
            - policy_loss: Average policy loss
            - value_loss: Average value loss
            - policy_accuracy: Top-1 policy accuracy
        """
        pass

    def train_step(
        self,
        states: torch.Tensor,
        target_policies: torch.Tensor,
        target_values: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single training step (forward + backward + optimize).

        Args:
            states: Batch of encoded states (batch_size, 256)
            target_policies: Target policies from MCTS (batch_size, 65)
            target_values: Target values from outcomes (batch_size,)

        Returns:
            Dictionary with loss components
        """
        pass

    def compute_loss(
        self,
        policy_pred: torch.Tensor,
        value_pred: torch.Tensor,
        policy_target: torch.Tensor,
        value_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training losses.

        Args:
            policy_pred: Predicted policy logits (batch_size, 65)
            value_pred: Predicted values (batch_size,)
            policy_target: Target policy distributions (batch_size, 65)
            value_target: Target values (batch_size,)

        Returns:
            (total_loss, metrics_dict)
        """
        pass

    def save_checkpoint(
        self,
        filepath: str,
        iteration: int,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Save training checkpoint.

        Args:
            filepath: Path to save checkpoint
            iteration: Training iteration number
            metrics: Optional metrics to save with checkpoint
        """
        pass

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        Load training checkpoint.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Dictionary with checkpoint metadata
        """
        pass

    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        pass

    def step_scheduler(self):
        """Step the learning rate scheduler."""
        pass
```

#### 4.3 Tests (30 min)

Test function signatures:

```python
# ml/training/test_training.py

class TestNetworkTrainer:
    def test_trainer_initialization(self):
        """Test network trainer initializes correctly."""
        pass

    def test_single_training_step(self):
        """Test a single training step."""
        pass

    def test_loss_computation(self):
        """Test loss computation is correct."""
        pass

    def test_train_epoch(self):
        """Test training for one epoch."""
        pass

    def test_checkpoint_save_load(self):
        """Test saving and loading checkpoints."""
        pass

    def test_learning_rate_scheduling(self):
        """Test learning rate scheduler works."""
        pass

    def test_gradient_clipping(self):
        """Test gradients are clipped correctly."""
        pass
```

**Deliverable**: Network trainer with loss computation and optimization

---

### SESSION 5: Training Loop - Orchestration

**Goal**: Integrate self-play, replay buffer, and training into main loop.

#### 5.1 Training Pipeline Design (15 min)

Document the full training pipeline:
```python
"""
AlphaZero Training Pipeline

Main Training Loop:
    for iteration in range(max_iterations):
        1. Self-Play Generation
           - Generate 10,000 games using current model
           - Store training examples in replay buffer

        2. Network Training
           - Train for N epochs on replay buffer
           - Update network weights

        3. Model Evaluation
           - Test new model vs previous best
           - Calculate ELO rating

        4. Checkpoint & Logging
           - Save model checkpoint
           - Log metrics to TensorBoard/W&B
           - Update best model if improved

        5. Iteration Complete
           - Move to next iteration

Iteration Schedule:
    - Total iterations: 500+
    - Games per iteration: 10,000
    - Training epochs per iteration: 10
    - Evaluation games: 400

Expected Timeline:
    - Self-play: ~30 minutes (16 workers)
    - Training: ~10 minutes (GPU)
    - Evaluation: ~5 minutes
    - Total per iteration: ~45 minutes
    - 500 iterations: ~15 days continuous training
"""
```

#### 5.2 Implement TrainingPipeline Class (70 min)

Function signatures:

```python
# ml/training/trainer.py

class TrainingPipeline:
    """
    Orchestrates the full AlphaZero training pipeline.

    Manages self-play, training, evaluation, and checkpointing.
    """

    def __init__(
        self,
        network: BlobNet,
        encoder: StateEncoder,
        masker: ActionMasker,
        config: Dict[str, Any],
    ):
        """
        Initialize training pipeline.

        Args:
            network: Neural network to train
            encoder: State encoder
            masker: Action masker
            config: Training configuration dictionary
        """
        pass

    def run_training(
        self,
        num_iterations: int,
        resume_from: Optional[str] = None,
    ):
        """
        Run the full training pipeline.

        Args:
            num_iterations: Number of training iterations
            resume_from: Optional checkpoint to resume from
        """
        pass

    def run_iteration(self, iteration: int) -> Dict[str, Any]:
        """
        Run a single training iteration.

        Args:
            iteration: Current iteration number

        Returns:
            Dictionary with iteration metrics
        """
        pass

    def _selfplay_phase(self, iteration: int) -> List[Dict[str, Any]]:
        """
        Generate self-play games for this iteration.

        Args:
            iteration: Current iteration number

        Returns:
            List of training examples
        """
        pass

    def _training_phase(
        self,
        iteration: int,
        num_epochs: int,
    ) -> Dict[str, float]:
        """
        Train network on replay buffer.

        Args:
            iteration: Current iteration number
            num_epochs: Number of training epochs

        Returns:
            Training metrics
        """
        pass

    def _evaluation_phase(self, iteration: int) -> Dict[str, Any]:
        """
        Evaluate new model vs previous best.

        Args:
            iteration: Current iteration number

        Returns:
            Evaluation metrics (will implement in Session 6)
        """
        pass

    def _checkpoint_phase(
        self,
        iteration: int,
        metrics: Dict[str, Any],
    ):
        """
        Save checkpoint and update best model if needed.

        Args:
            iteration: Current iteration number
            metrics: Iteration metrics
        """
        pass

    def _log_metrics(
        self,
        iteration: int,
        metrics: Dict[str, Any],
    ):
        """
        Log metrics to TensorBoard/W&B.

        Args:
            iteration: Current iteration number
            metrics: Metrics to log
        """
        pass
```

#### 5.3 Integration Tests (30 min)

Test function signatures:

```python
# ml/training/test_training.py

class TestTrainingPipeline:
    def test_pipeline_initialization(self):
        """Test training pipeline initializes correctly."""
        pass

    def test_single_iteration(self):
        """Test running a single training iteration."""
        pass

    def test_selfplay_training_integration(self):
        """Test self-play examples feed into training."""
        pass

    def test_checkpoint_resume(self):
        """Test resuming training from checkpoint."""
        pass

    def test_metrics_logging(self):
        """Test metrics are logged correctly."""
        pass
```

**Deliverable**: Full training pipeline orchestration

---

### SESSION 6: Evaluation & ELO Tracking

**Goal**: Implement model evaluation and ELO rating system.

#### 6.1 Evaluation Design (20 min)

Document evaluation approach:
```python
"""
Model Evaluation System

Purpose: Test new models against previous best to track improvement

Tournament Structure:
    - Play N games (400 default) between two models
    - Each model plays as each player position equally
    - Record wins, losses, final scores
    - Calculate ELO rating change

ELO Rating System:
    - Start at 1000 ELO
    - K-factor: 32 (standard)
    - Expected score: E = 1 / (1 + 10^((opponent_elo - player_elo) / 400))
    - Rating update: new_elo = old_elo + K * (actual_score - expected_score)

Model Promotion:
    - New model must win >55% of games to replace best
    - Confidence threshold prevents noise from promoting weak models
    - Track ELO history for visualization

Metrics:
    - Win rate
    - Average score differential
    - Bidding accuracy (bid vs actual tricks)
    - Move quality (vs MCTS recommendations)
"""
```

#### 6.2 Implement Arena Class (50 min)

Function signatures:

```python
# ml/evaluation/arena.py

class Arena:
    """
    Tournament system for model vs model evaluation.

    Plays games between two models and records results.
    """

    def __init__(
        self,
        encoder: StateEncoder,
        masker: ActionMasker,
        num_determinizations: int = 3,
        simulations_per_determinization: int = 50,
    ):
        """
        Initialize arena for model tournaments.

        Args:
            encoder: State encoder
            masker: Action masker
            num_determinizations: Determinizations for MCTS
            simulations_per_determinization: MCTS simulations per world
        """
        pass

    def play_match(
        self,
        model1: BlobNet,
        model2: BlobNet,
        num_games: int = 400,
        num_players: int = 4,
        cards_to_deal: int = 5,
    ) -> Dict[str, Any]:
        """
        Play a match between two models.

        Args:
            model1: First model (challenger)
            model2: Second model (champion)
            num_games: Number of games to play
            num_players: Players per game
            cards_to_deal: Cards to deal

        Returns:
            Match results:
            - model1_wins: Number of games model1 won
            - model2_wins: Number of games model2 won
            - draws: Number of draws
            - model1_avg_score: Average score for model1
            - model2_avg_score: Average score for model2
            - win_rate: model1 win rate
        """
        pass

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
            models: List of models
            player_assignments: Which model controls each player
            num_players: Number of players
            cards_to_deal: Cards to deal

        Returns:
            Dictionary mapping player_position -> final_score
        """
        pass

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
        pass
```

#### 6.3 Implement ELO System (30 min)

Function signatures:

```python
# ml/evaluation/elo.py

class ELOTracker:
    """
    Tracks ELO ratings for model generations.

    Maintains history of model performance over training.
    """

    def __init__(
        self,
        initial_elo: int = 1000,
        k_factor: int = 32,
    ):
        """
        Initialize ELO tracker.

        Args:
            initial_elo: Starting ELO rating
            k_factor: ELO update rate
        """
        pass

    def calculate_expected_score(
        self,
        player_elo: int,
        opponent_elo: int,
    ) -> float:
        """
        Calculate expected score for a player.

        Args:
            player_elo: Player's current ELO
            opponent_elo: Opponent's current ELO

        Returns:
            Expected score (0.0 to 1.0)
        """
        pass

    def update_elo(
        self,
        player_elo: int,
        opponent_elo: int,
        actual_score: float,
    ) -> int:
        """
        Update player's ELO based on match result.

        Args:
            player_elo: Player's current ELO
            opponent_elo: Opponent's current ELO
            actual_score: Actual match score (0.0 = loss, 0.5 = draw, 1.0 = win)

        Returns:
            Updated player ELO
        """
        pass

    def add_match_result(
        self,
        iteration: int,
        model_elo: int,
        opponent_elo: int,
        win_rate: float,
    ) -> int:
        """
        Record a match result and update ELO.

        Args:
            iteration: Training iteration number
            model_elo: Current model's ELO
            opponent_elo: Opponent's ELO
            win_rate: Win rate achieved

        Returns:
            Updated model ELO
        """
        pass

    def get_elo_history(self) -> List[Dict[str, Any]]:
        """
        Get full ELO history.

        Returns:
            List of {iteration, elo, opponent_elo, win_rate} dicts
        """
        pass

    def save_history(self, filepath: str):
        """Save ELO history to JSON file."""
        pass

    def load_history(self, filepath: str):
        """Load ELO history from JSON file."""
        pass

    def should_promote_model(
        self,
        win_rate: float,
        threshold: float = 0.55,
    ) -> bool:
        """
        Determine if new model should replace best model.

        Args:
            win_rate: Win rate against current best
            threshold: Minimum win rate for promotion

        Returns:
            True if model should be promoted
        """
        pass
```

#### 6.4 Tests (20 min)

Test function signatures:

```python
# ml/evaluation/test_evaluation.py

class TestArena:
    def test_arena_initialization(self):
        """Test arena initializes correctly."""
        pass

    def test_single_game_evaluation(self):
        """Test playing a single evaluation game."""
        pass

    def test_match_between_models(self):
        """Test playing full match between two models."""
        pass

    def test_win_rate_calculation(self):
        """Test win rate calculation is correct."""
        pass


class TestELOTracker:
    def test_elo_tracker_initialization(self):
        """Test ELO tracker initializes correctly."""
        pass

    def test_expected_score_calculation(self):
        """Test expected score formula."""
        pass

    def test_elo_update(self):
        """Test ELO updates correctly."""
        pass

    def test_promotion_threshold(self):
        """Test model promotion logic."""
        pass

    def test_elo_history_persistence(self):
        """Test saving and loading ELO history."""
        pass
```

**Deliverable**: Model evaluation system with ELO tracking

---

### SESSION 7: Main Training Script & Configuration

**Goal**: Create the main training entry point and configuration system.

#### 7.1 Configuration System (30 min)

Function signatures:

```python
# ml/config.py

@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""

    # Self-play settings
    num_workers: int = 16
    games_per_iteration: int = 10_000
    num_determinizations: int = 3
    simulations_per_determinization: int = 30

    # Replay buffer settings
    replay_buffer_capacity: int = 500_000
    min_buffer_size: int = 10_000

    # Training settings
    batch_size: int = 512
    epochs_per_iteration: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0

    # Evaluation settings
    eval_games: int = 400
    eval_determinizations: int = 3
    eval_simulations: int = 50
    promotion_threshold: float = 0.55

    # Network settings
    embedding_dim: int = 256
    num_transformer_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1

    # Hardware settings
    device: str = 'cuda'
    use_mixed_precision: bool = True

    # Checkpointing
    checkpoint_dir: str = 'models/checkpoints'
    save_every_n_iterations: int = 10

    # Logging
    log_dir: str = 'runs'
    use_wandb: bool = False
    wandb_project: str = 'blobmaster'

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        pass

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        pass

    @classmethod
    def from_file(cls, filepath: str) -> 'TrainingConfig':
        """Load config from JSON file."""
        pass

    def save(self, filepath: str):
        """Save config to JSON file."""
        pass
```

#### 7.2 Implement Main Training Script (50 min)

Function signatures:

```python
# ml/train.py

def main():
    """Main training entry point."""
    pass

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    pass

def setup_logging(config: TrainingConfig):
    """
    Setup logging (TensorBoard, W&B, file logging).

    Args:
        config: Training configuration
    """
    pass

def create_network(config: TrainingConfig) -> BlobNet:
    """
    Create neural network from config.

    Args:
        config: Training configuration

    Returns:
        Initialized neural network
    """
    pass

def run_training_pipeline(
    config: TrainingConfig,
    num_iterations: int,
    resume_from: Optional[str] = None,
):
    """
    Run the full training pipeline.

    Args:
        config: Training configuration
        num_iterations: Number of iterations to train
        resume_from: Optional checkpoint to resume from
    """
    pass

if __name__ == '__main__':
    main()
```

#### 7.3 Integration Testing (30 min)

Test function signatures:

```python
# ml/training/test_training.py

class TestMainTrainingScript:
    def test_config_creation(self):
        """Test creating training config."""
        pass

    def test_config_save_load(self):
        """Test saving and loading config."""
        pass

    def test_network_creation_from_config(self):
        """Test creating network from config."""
        pass

    def test_mini_training_run(self):
        """Test running a small training iteration (integration test)."""
        pass
```

#### 7.4 Documentation (10 min)

- [ ] Update README.md with Phase 4 status
- [ ] Document training command usage
- [ ] Add training configuration examples
- [ ] Document checkpoint format
- [ ] Update roadmap progress

**Deliverable**: Complete training pipeline ready to run

---

## Success Criteria

### Functional Requirements
- [ ] Self-play generates valid training examples
- [ ] Parallel workers generate games efficiently
- [ ] Replay buffer stores and samples correctly
- [ ] Network training reduces loss over time
- [ ] Evaluation system compares models accurately
- [ ] ELO ratings track model improvement
- [ ] Training pipeline runs end-to-end
- [ ] Checkpointing and resume work correctly

### Code Quality
- [ ] All pytest tests pass (target: 60+ new tests)
- [ ] Code coverage >85% for new modules
- [ ] Type hints on all function signatures
- [ ] Comprehensive docstrings

### Performance Targets
- [ ] Self-play: >300 games/minute (16 workers)
- [ ] Training: <15 min per epoch on GPU
- [ ] Evaluation: <10 minutes for 400 games
- [ ] Full iteration: <60 minutes
- [ ] Memory usage: <8GB GPU, <32GB RAM

### Ready for Training
- [ ] Can run `python ml/train.py --iterations 500`
- [ ] Training progresses smoothly
- [ ] Loss curves show learning
- [ ] Model checkpoints save correctly
- [ ] Can resume from checkpoint

---

## Deliverables

After Phase 4 completion:

1. **ml/training/selfplay.py**: Self-play game generation (~500 lines)
2. **ml/training/replay_buffer.py**: Experience storage (~300 lines)
3. **ml/training/trainer.py**: Training orchestration (~600 lines)
4. **ml/evaluation/arena.py**: Model evaluation (~300 lines)
5. **ml/evaluation/elo.py**: ELO tracking (~200 lines)
6. **ml/train.py**: Main training script (~200 lines)
7. **ml/config.py**: Configuration system (~150 lines)
8. **Tests**: Comprehensive test coverage (~500 lines)

**Total**: ~2750 lines of new code

**Total Test Count**: 333 (Phase 3) + 60 (Phase 4) = **393 tests**

---

## Timeline Summary

| Session | Component | Deliverable |
|---------|-----------|-------------|
| 1 | Self-Play Foundation | SelfPlayWorker generating games |
| 2 | Self-Play Parallelization | Parallel game generation |
| 3 | Replay Buffer | Experience storage and sampling |
| 4 | Training Loop - Updates | Network optimization |
| 5 | Training Loop - Orchestration | Full pipeline integration |
| 6 | Evaluation & ELO | Model comparison system |
| 7 | Main Script & Config | Complete training system |

**Total**: 7 sessions × 2 hours = 14 hours

---

## Next Steps (Phase 5)

After Phase 4:

1. **Start Training**: Run 3-7 day training on RTX GPU
2. **Monitor Progress**: Watch ELO curves, loss curves, game quality
3. **Hyperparameter Tuning**: Adjust based on initial results
4. **Proceed to Phase 5**: ONNX Export & Inference
   - Export best model to ONNX format
   - Validate outputs match PyTorch
   - Optimize for Intel CPU/iGPU inference
   - Test on laptop target hardware

---

## Common Issues & Solutions

### Issue: Self-play too slow
**Solution**:
- Reduce num_determinizations (try 2 instead of 3)
- Reduce simulations_per_determinization (try 20 instead of 30)
- Increase num_workers if CPU cores available
- Profile to find bottlenecks

### Issue: Training loss not decreasing
**Solution**:
- Check replay buffer has enough diversity
- Verify loss computation is correct
- Try lower learning rate (0.0005)
- Check for gradient vanishing/exploding
- Ensure MCTS policies are being stored correctly

### Issue: Model not improving (ELO stagnant)
**Solution**:
- Increase games per iteration (more data)
- Increase epochs per iteration (more training)
- Check if model is overfitting (validation loss)
- Verify evaluation is fair (equal positions)
- Consider curriculum learning (start simple)

### Issue: Out of memory during training
**Solution**:
- Reduce batch_size (try 256)
- Reduce replay_buffer_capacity (try 250k)
- Use gradient accumulation for effective larger batches
- Enable mixed precision training
- Clear GPU cache between iterations

### Issue: Evaluation results noisy
**Solution**:
- Increase eval_games (try 800)
- Ensure each model plays all positions equally
- Use higher simulations for evaluation MCTS
- Average results over multiple matches
- Use ELO confidence intervals

---

## Research Questions to Explore

During Phase 4 implementation, consider:

1. **Optimal self-play balance**: How many games vs training epochs?
2. **Replay buffer size**: Does larger buffer help or hurt?
3. **Temperature schedule**: What exploration strategy works best?
4. **Loss weighting**: Policy vs value loss importance?
5. **Determinization count**: 3 vs 5 vs 10 for training?
6. **Training stability**: How to prevent catastrophic forgetting?
7. **Sample efficiency**: Can we learn faster with better data?

---

## Performance Optimization Tips

### Self-Play Optimization
- Use process pools (not threads) to avoid GIL
- Batch neural network inference when possible
- Cache MCTS tree between moves
- Profile to identify bottlenecks

### Training Optimization
- Use mixed precision (FP16) training
- Enable cuDNN autotuner
- Use DataLoader with multiple workers
- Pin memory for faster GPU transfer
- Use gradient accumulation for larger effective batch

### Memory Optimization
- Clear MCTS trees after each game
- Use numpy for storage (more compact than Python objects)
- Compress old replay buffer data
- Limit checkpoint retention

---

## References

- **AlphaZero Paper**: [Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815)
- **AlphaGo Zero**: [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)
- **Replay Buffers**: [Experience Replay in Deep RL](https://arxiv.org/abs/1712.01275)
- **ELO Rating System**: [The Rating of Chess Players](https://en.wikipedia.org/wiki/Elo_rating_system)
- **PyTorch Training**: [PyTorch Training Best Practices](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

**Last Updated**: 2025-10-25
**Status**: ⏳ READY TO START
**Version**: 1.0
