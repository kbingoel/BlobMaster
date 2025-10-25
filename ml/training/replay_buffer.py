"""
Replay Buffer for Experience Storage

This module implements a circular replay buffer for storing self-play training
examples. The buffer stores recent game experiences and provides efficient
random sampling for neural network training.

Purpose: Store recent self-play games for training

Key Features:
    - Circular buffer (FIFO when full)
    - Efficient random sampling
    - Optional data augmentation
    - Disk persistence for resume training

Design:
    - Capacity: 500,000 positions (configurable)
    - Storage format: List of training example dicts
    - Sampling: Uniform random (no prioritization in v1)
    - Persistence: Save/load to disk via pickle

Training Example Structure:
    {
        'state': np.array,      # Encoded state (256-dim)
        'policy': np.array,     # MCTS policy (65-dim)
        'value': float,         # Outcome (-1 to 1 normalized)
        'player_position': int,
        'game_id': str,
        'move_number': int,
    }

Operations:
    - add_examples(examples): Add new examples (FIFO replacement when full)
    - sample_batch(batch_size): Random sample for training
    - save(path): Persist buffer to disk
    - load(path): Load buffer from disk
    - clear(): Empty buffer
    - __len__(): Current size
"""

import numpy as np
import torch
import pickle
from typing import Dict, List, Any, Tuple
from collections import defaultdict


class ReplayBuffer:
    """
    Circular replay buffer for storing self-play training examples.

    Stores recent game experiences and provides efficient sampling for
    neural network training. Uses FIFO policy when capacity is exceeded.
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
            use_prioritization: Use prioritized experience replay (not implemented yet)
        """
        self.capacity = capacity
        self.use_prioritization = use_prioritization

        # Storage for training examples
        self.buffer: List[Dict[str, Any]] = []

        # Track position for circular buffer
        self.position = 0

        # Flag for whether buffer is full (affects insert behavior)
        self.is_full = False

        if use_prioritization:
            raise NotImplementedError(
                "Prioritized experience replay not yet implemented"
            )

    def add_examples(self, examples: List[Dict[str, Any]]):
        """
        Add training examples to the buffer.

        Uses FIFO policy when buffer is full - oldest examples are replaced.
        Examples are added one at a time, maintaining circular buffer semantics.

        Args:
            examples: List of training example dictionaries
        """
        for example in examples:
            if not self.is_full and len(self.buffer) < self.capacity:
                # Buffer not yet full, append
                self.buffer.append(example)
            else:
                # Buffer is full, replace oldest example
                self.is_full = True
                self.buffer[self.position] = example

            # Update position (circular)
            self.position = (self.position + 1) % self.capacity

    def sample_batch(
        self,
        batch_size: int,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random batch of training examples.

        Performs uniform random sampling from the buffer and converts
        numpy arrays to PyTorch tensors.

        Args:
            batch_size: Number of examples to sample
            device: Device to place tensors on ('cpu' or 'cuda')

        Returns:
            (states, policies, values) tuple of tensors
            - states: (batch_size, 256) tensor
            - policies: (batch_size, 65) tensor
            - values: (batch_size,) tensor

        Raises:
            ValueError: If batch_size is larger than buffer size
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} examples from buffer with "
                f"{len(self.buffer)} examples"
            )

        # Randomly sample indices
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)

        # Gather examples
        batch_examples = [self.buffer[i] for i in indices]

        # Convert to tensors
        states = np.stack([ex["state"] for ex in batch_examples])
        policies = np.stack([ex["policy"] for ex in batch_examples])
        values = np.array([ex["value"] for ex in batch_examples], dtype=np.float32)

        # Convert to PyTorch tensors
        states_tensor = torch.from_numpy(states).float().to(device)
        policies_tensor = torch.from_numpy(policies).float().to(device)
        values_tensor = torch.from_numpy(values).float().to(device)

        return states_tensor, policies_tensor, values_tensor

    def save(self, filepath: str):
        """
        Save replay buffer to disk.

        Saves the buffer contents and metadata using pickle format.
        This allows resuming training from a checkpoint.

        Args:
            filepath: Path to save buffer to (e.g., 'replay_buffer.pkl')
        """
        save_data = {
            "buffer": self.buffer,
            "position": self.position,
            "is_full": self.is_full,
            "capacity": self.capacity,
            "use_prioritization": self.use_prioritization,
        }

        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)

    def load(self, filepath: str):
        """
        Load replay buffer from disk.

        Restores buffer contents and metadata from a saved file.
        Allows resuming training from where it left off.

        Args:
            filepath: Path to load buffer from

        Raises:
            FileNotFoundError: If filepath doesn't exist
            ValueError: If loaded capacity doesn't match current capacity
        """
        with open(filepath, "rb") as f:
            save_data = pickle.load(f)

        # Validate capacity matches
        if save_data["capacity"] != self.capacity:
            raise ValueError(
                f"Loaded buffer capacity ({save_data['capacity']}) doesn't match "
                f"current capacity ({self.capacity})"
            )

        # Restore state
        self.buffer = save_data["buffer"]
        self.position = save_data["position"]
        self.is_full = save_data["is_full"]
        self.use_prioritization = save_data["use_prioritization"]

    def clear(self):
        """Clear all examples from the buffer."""
        self.buffer.clear()
        self.position = 0
        self.is_full = False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about buffer contents.

        Returns:
            Dictionary with:
            - size: Current number of examples
            - capacity: Maximum capacity
            - utilization: Percentage full (0-100)
            - num_games: Number of unique games
            - avg_game_length: Average game length in moves
            - value_distribution: Dict with min/max/mean/std of values
            - player_distribution: Count of examples per player position
        """
        if len(self.buffer) == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "utilization": 0.0,
                "num_games": 0,
                "avg_game_length": 0.0,
                "value_distribution": {
                    "min": 0.0,
                    "max": 0.0,
                    "mean": 0.0,
                    "std": 0.0,
                },
                "player_distribution": {},
            }

        # Count unique games
        game_ids = set(ex["game_id"] for ex in self.buffer)
        num_games = len(game_ids)

        # Calculate average game length
        game_lengths = defaultdict(int)
        for ex in self.buffer:
            game_lengths[ex["game_id"]] += 1
        avg_game_length = np.mean(list(game_lengths.values()))

        # Value distribution
        values = np.array([ex["value"] for ex in self.buffer])
        value_dist = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
        }

        # Player position distribution
        player_counts = defaultdict(int)
        for ex in self.buffer:
            player_counts[ex["player_position"]] += 1

        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "utilization": 100.0 * len(self.buffer) / self.capacity,
            "num_games": num_games,
            "avg_game_length": float(avg_game_length),
            "value_distribution": value_dist,
            "player_distribution": dict(player_counts),
        }

    def __len__(self) -> int:
        """Get current number of examples in buffer."""
        return len(self.buffer)

    def is_ready_for_training(self, min_examples: int = 10_000) -> bool:
        """
        Check if buffer has enough examples for training.

        Args:
            min_examples: Minimum examples required

        Returns:
            True if buffer is ready for training
        """
        return len(self.buffer) >= min_examples


def augment_example(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Apply data augmentation to a training example.

    Note: Card games don't have obvious symmetries like images (rotation,
    flipping). Potential augmentations could include:
    - Hand permutations (limited value - changes semantics)
    - Suit remapping (could work for Blob since suits are mostly equivalent)

    For now, this is a placeholder that returns the original example unchanged.
    Future implementations could explore suit-based augmentations.

    Args:
        example: Original training example

    Returns:
        List of augmented examples (currently just [example])
    """
    # No augmentation for now - card games have limited symmetries
    return [example]
