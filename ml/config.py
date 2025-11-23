"""
Training Configuration System

Centralized configuration for the AlphaZero training pipeline.
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""

    # Self-play settings
    num_workers: int = 16
    games_per_iteration: int = 10_000
    num_determinizations: int = 3
    simulations_per_determinization: int = 30
    must_have_suit_bias: float = 1.0  # Probability multiplier for must-have suits during determinization (1.0 = no bias/maximum entropy, 1.5-3.0 = increasing preference)

    # MCTS parallelization
    use_parallel_expansion: bool = True
    parallel_batch_size: int = 42  # Default from BO Stage 3 (use get_batch_params for curriculum)
    batch_timeout_ms: float = 9.0  # Batch collection timeout (BO-tuned, use get_batch_params for curriculum)

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
    wandb_run_name: Optional[str] = None

    # Game settings
    num_players: int = 4
    cards_to_deal: int = 5

    # Hybrid Training: Player distribution (Session 0)
    player_distribution: dict = field(default_factory=lambda: {
        4: 0.15,  # 15% - occasional
        5: 0.70,  # 70% - YOUR STANDARD
        6: 0.15,  # 15% - occasional
    })

    # Hybrid Training: 4p starting cards split (Session 0)
    start_card_distribution_4p: dict = field(default_factory=lambda: {
        7: 0.60,  # 60% - standard
        8: 0.40,  # 40% - when fewer players
    })

    # Hybrid Training: Use decision-weighted sampling vs fixed (5p, 5c) (Session 0)
    use_decision_weighted_sampling: bool = False  # Default to False for backward compatibility

    # Hybrid Training: Training mode - controls what games_per_iteration counts (Session 0)
    training_on: str = "rounds"  # "rounds" (Phase 1) or "games" (Phase 2)
    # - "rounds": games_per_iteration counts independent rounds (~360 rounds/min)
    # - "games": games_per_iteration counts full multi-round games (~73 games/min)

    # Hybrid Training: MCTS curriculum schedule (Session 0)
    # Maps iteration threshold -> (num_determinizations, simulations_per_determinization)
    mcts_schedule: dict = field(default_factory=lambda: {
        50: (1, 15),
        150: (2, 25),
        300: (3, 35),
        450: (4, 45),
        500: (5, 50),
    })

    # Bayesian Optimization (Optuna TPE) auto-tuned batch configuration (2025-11-21)
    # Maps iteration threshold -> (parallel_batch_size, batch_timeout_ms)
    # 25 trials per stage with TPE sampler found optimal configs for all stages
    batch_config_schedule: dict = field(default_factory=lambda: {
        50: (40, 6),    # Stage 1 (1×15): +6.4% speedup
        150: (26, 3),   # Stage 2 (2×25): +2.1% speedup
        300: (42, 9),   # Stage 3 (3×35): +40.8% speedup
        450: (50, 8),   # Stage 4 (4×45): +45.3% speedup
        500: (53, 9),   # Stage 5 (5×50): +48.7% speedup
    })

    def get_mcts_params(self, iteration: int) -> tuple:
        """
        Return (num_determinizations, simulations_per_det) for iteration.

        Uses mcts_schedule to determine MCTS parameters based on training iteration.
        Earlier iterations use lighter MCTS (faster, less accurate), later iterations
        use heavier MCTS (slower, more accurate).

        Args:
            iteration: Current training iteration (1-indexed)

        Returns:
            Tuple of (num_determinizations, simulations_per_determinization)
        """
        for threshold, (det, sims) in sorted(self.mcts_schedule.items()):
            if iteration <= threshold:
                return (det, sims)
        # Default to highest if beyond schedule
        return (5, 50)

    def get_batch_params(self, iteration: int) -> tuple:
        """
        Return (parallel_batch_size, batch_timeout_ms) for iteration.

        Uses batch_config_schedule from Bayesian Optimization (Optuna TPE) auto-tuning.
        These parameters control MCTS parallel expansion batching and timeout for GPU inference.

        All stages show significant improvements (2-48%) over baseline with BO-tuned configs.

        Args:
            iteration: Current training iteration (1-indexed)

        Returns:
            Tuple of (parallel_batch_size, batch_timeout_ms)

        Examples:
            >>> config = TrainingConfig()
            >>> config.get_batch_params(1)    # Stage 1
            (40, 6)
            >>> config.get_batch_params(300)  # Stage 3
            (42, 9)
            >>> config.get_batch_params(500)  # Stage 5
            (53, 9)
        """
        for threshold, (batch_size, timeout_ms) in sorted(self.batch_config_schedule.items()):
            if iteration <= threshold:
                return (batch_size, timeout_ms)
        # Default to Stage 5 optimal if beyond schedule
        return (53, 9)

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.

        Returns:
            Dictionary representation of config
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """
        Create config from dictionary.

        Args:
            config_dict: Dictionary of configuration values

        Returns:
            TrainingConfig instance
        """
        # Filter out keys that aren't valid config fields
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)

    @classmethod
    def from_file(cls, filepath: str) -> 'TrainingConfig':
        """
        Load config from JSON file.

        Args:
            filepath: Path to JSON config file

        Returns:
            TrainingConfig instance
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def save(self, filepath: str):
        """
        Save config to JSON file.

        Args:
            filepath: Path to save config to
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def validate(self) -> bool:
        """
        Validate configuration values.

        Returns:
            True if config is valid

        Raises:
            ValueError: If config values are invalid
        """
        if self.num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {self.num_workers}")

        if self.games_per_iteration <= 0:
            raise ValueError(
                f"games_per_iteration must be positive, got {self.games_per_iteration}"
            )

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )

        if not 0 <= self.promotion_threshold <= 1:
            raise ValueError(
                f"promotion_threshold must be in [0, 1], got {self.promotion_threshold}"
            )

        if self.device not in ('cuda', 'cpu'):
            raise ValueError(f"device must be 'cuda' or 'cpu', got {self.device}")

        if self.num_players < 3 or self.num_players > 8:
            raise ValueError(
                f"num_players must be between 3 and 8, got {self.num_players}"
            )

        if self.cards_to_deal <= 0:
            raise ValueError(
                f"cards_to_deal must be positive, got {self.cards_to_deal}"
            )

        if self.training_on not in ('rounds', 'games'):
            raise ValueError(
                f"training_on must be 'rounds' or 'games', got {self.training_on}"
            )

        return True

    def __str__(self) -> str:
        """String representation of config."""
        lines = ["Training Configuration:"]
        lines.append(f"  Self-play: {self.games_per_iteration} games, {self.num_workers} workers")
        lines.append(f"  MCTS: {self.num_determinizations} determinizations, {self.simulations_per_determinization} sims/det")
        lines.append(f"  Training: batch={self.batch_size}, lr={self.learning_rate}, epochs={self.epochs_per_iteration}")
        lines.append(f"  Evaluation: {self.eval_games} games, threshold={self.promotion_threshold}")
        lines.append(f"  Network: {self.num_transformer_layers} layers, {self.num_heads} heads, dim={self.embedding_dim}")
        lines.append(f"  Device: {self.device}, Mixed Precision: {self.use_mixed_precision}")
        return "\n".join(lines)


def get_fast_config() -> TrainingConfig:
    """
    Get a fast training config for testing/debugging.

    Uses BO-optimized batch parameters from Stage 1 (1×15) for maximum throughput
    with light MCTS settings.

    Returns:
        TrainingConfig with reduced computational requirements
    """
    return TrainingConfig(
        num_workers=32,
        games_per_iteration=100,  # 100 games for quick testing
        num_determinizations=1,
        simulations_per_determinization=15,
        parallel_batch_size=40,  # BO-optimized from Stage 1 (1×15): +6.4% speedup
        batch_timeout_ms=6,      # BO-optimized from Stage 1: best throughput (1643 r/min)
        replay_buffer_capacity=10_000,  # Max examples stored (across multiple iterations)
        min_buffer_size=1_000,
        batch_size=64,  # Training batch size (gradient updates, not MCTS batching)
        epochs_per_iteration=2,
        eval_games=50,
        eval_determinizations=1,
        eval_simulations=15,
    )


def get_production_config() -> TrainingConfig:
    """
    Get the full production training config.

    Returns:
        TrainingConfig with full computational requirements
    """
    return TrainingConfig()  # Uses defaults
