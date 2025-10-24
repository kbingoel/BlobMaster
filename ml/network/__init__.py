"""
Neural network module for BlobMaster.

This module contains:
- State encoding: Convert game state to tensor representation
- Neural network: Transformer-based policy/value network
- Training utilities: Loss functions, optimization, checkpointing
"""

from ml.network.encode import StateEncoder, ActionMasker

__all__ = ["StateEncoder", "ActionMasker"]
