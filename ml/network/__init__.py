"""
Neural network module for BlobMaster.

This module contains:
- State encoding: Convert game state to tensor representation
- Action masking: Create legal action masks for neural network
- Neural network: Transformer-based policy/value network
- Training utilities: Loss functions, optimization, checkpointing
"""

from ml.network.encode import StateEncoder, ActionMasker
from ml.network.model import (
    BlobNet,
    BlobNetTrainer,
    create_model,
    create_trainer,
)

__all__ = [
    "StateEncoder",
    "ActionMasker",
    "BlobNet",
    "BlobNetTrainer",
    "create_model",
    "create_trainer",
]
