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

    Returns:
        TrainingConfig with reduced computational requirements
    """
    return TrainingConfig(
        num_workers=2,
        games_per_iteration=100,
        num_determinizations=2,
        simulations_per_determinization=10,
        replay_buffer_capacity=10_000,
        min_buffer_size=1_000,
        batch_size=64,
        epochs_per_iteration=2,
        eval_games=50,
        eval_determinizations=2,
        eval_simulations=10,
    )


def get_production_config() -> TrainingConfig:
    """
    Get the full production training config.

    Returns:
        TrainingConfig with full computational requirements
    """
    return TrainingConfig()  # Uses defaults
