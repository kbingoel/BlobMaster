"""
Neural Network Training Loop

This module implements the NetworkTrainer class that handles neural network
training from replay buffer data. It manages optimization, loss computation,
and checkpointing for the AlphaZero-style training pipeline.

Goal: Update network to match MCTS policies and predict outcomes

Loss Function:
    total_loss = policy_loss + value_loss

    policy_loss = CrossEntropy(predicted_policy, MCTS_policy)
    value_loss = MSE(predicted_value, actual_outcome)

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
    - LR schedule: Step decay or cosine annealing
    - Mixed precision: FP16 for faster training (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from ml.network.model import BlobNet
from ml.training.replay_buffer import ReplayBuffer


class NetworkTrainer:
    """
    Manages neural network training from replay buffer.

    Handles optimization, loss computation, and checkpointing for the
    AlphaZero-style training pipeline.
    """

    def __init__(
        self,
        network: BlobNet,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        policy_loss_weight: float = 1.0,
        value_loss_weight: float = 1.0,
        use_mixed_precision: bool = False,
        device: str = "cuda",
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
        self.network = network
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        self.use_mixed_precision = use_mixed_precision

        # Set device
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU instead")
            device = "cpu"
        self.device = device
        self.network.to(self.device)

        # Initialize optimizer (Adam with weight decay)
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Initialize learning rate scheduler (StepLR: decay by 0.1 every 100 epochs)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=100,
            gamma=0.1,
        )

        # Mixed precision scaler (for FP16 training)
        self.scaler = None
        if use_mixed_precision and device == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()

        # Training statistics
        self.total_steps = 0
        self.total_epochs = 0

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
            num_batches: Number of batches (None = full epoch based on buffer size)

        Returns:
            Dictionary with:
            - total_loss: Average total loss
            - policy_loss: Average policy loss
            - value_loss: Average value loss
            - policy_accuracy: Top-1 policy accuracy (%)
        """
        self.network.train()

        # Calculate number of batches if not specified
        if num_batches is None:
            # One epoch = go through entire buffer once (approximately)
            num_batches = max(1, len(replay_buffer) // batch_size)

        # Accumulate metrics
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        policy_accuracy_sum = 0.0

        for batch_idx in range(num_batches):
            # Sample batch from replay buffer
            states, target_policies, target_values = replay_buffer.sample_batch(
                batch_size, device=self.device
            )

            # Train step
            metrics = self.train_step(states, target_policies, target_values)

            # Accumulate
            total_loss_sum += metrics["total_loss"]
            policy_loss_sum += metrics["policy_loss"]
            value_loss_sum += metrics["value_loss"]
            policy_accuracy_sum += metrics["policy_accuracy"]

            self.total_steps += 1

        # Average metrics
        self.total_epochs += 1

        return {
            "total_loss": total_loss_sum / num_batches,
            "policy_loss": policy_loss_sum / num_batches,
            "value_loss": value_loss_sum / num_batches,
            "policy_accuracy": policy_accuracy_sum / num_batches,
        }

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
            target_policies: Target policies from MCTS (batch_size, action_dim)
            target_values: Target values from outcomes (batch_size,)

        Returns:
            Dictionary with loss components and metrics
        """
        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass (with optional mixed precision)
        if self.use_mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                policy_logits, value_pred = self.network(states)
                total_loss, metrics = self.compute_loss(
                    policy_logits, value_pred, target_policies, target_values
                )

            # Backward pass with gradient scaling
            self.scaler.scale(total_loss).backward()

            # Gradient clipping (unscale first for mixed precision)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

            # Optimizer step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard forward pass
            policy_logits, value_pred = self.network(states)
            total_loss, metrics = self.compute_loss(
                policy_logits, value_pred, target_policies, target_values
            )

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

            # Optimizer step
            self.optimizer.step()

        return metrics

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
            policy_pred: Predicted policy logits (batch_size, action_dim)
            value_pred: Predicted values (batch_size,)
            policy_target: Target policy distributions (batch_size, action_dim)
            value_target: Target values (batch_size,)

        Returns:
            (total_loss, metrics_dict)
            - total_loss: Combined weighted loss (scalar tensor)
            - metrics_dict: Dictionary with individual loss components and accuracy
        """
        # Policy loss: Cross-entropy between predicted and target distributions
        # Target is already a probability distribution, so we use KL divergence
        # which is equivalent to cross-entropy for probability distributions

        # Apply log_softmax to predictions
        log_policy_pred = F.log_softmax(policy_pred, dim=1)

        # KL divergence loss: sum(target * (log(target) - log(pred)))
        # Equivalent to: -sum(target * log(pred)) when target is a distribution
        policy_loss = -(policy_target * log_policy_pred).sum(dim=1).mean()

        # Value loss: MSE between predicted and target values
        value_loss = F.mse_loss(value_pred.squeeze(), value_target)

        # Total loss (weighted combination)
        total_loss = (
            self.policy_loss_weight * policy_loss +
            self.value_loss_weight * value_loss
        )

        # Calculate policy accuracy (top-1)
        # Check if the highest probability action matches the target's highest
        policy_pred_actions = policy_pred.argmax(dim=1)
        policy_target_actions = policy_target.argmax(dim=1)
        policy_accuracy = (policy_pred_actions == policy_target_actions).float().mean()

        # Return metrics as Python floats (not tensors)
        metrics = {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "policy_accuracy": policy_accuracy.item() * 100.0,  # Convert to percentage
        }

        return total_loss, metrics

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
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "iteration": iteration,
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "total_steps": self.total_steps,
            "total_epochs": self.total_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "policy_loss_weight": self.policy_loss_weight,
            "value_loss_weight": self.value_loss_weight,
            "metrics": metrics or {},
        }

        # Add scaler state if using mixed precision
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        Load training checkpoint.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Dictionary with checkpoint metadata (iteration, metrics, etc.)
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        # Restore network and optimizer state
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore training statistics
        self.total_steps = checkpoint.get("total_steps", 0)
        self.total_epochs = checkpoint.get("total_epochs", 0)

        # Restore scaler if it exists
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Return metadata for logging
        return {
            "iteration": checkpoint["iteration"],
            "total_steps": self.total_steps,
            "total_epochs": self.total_epochs,
            "metrics": checkpoint.get("metrics", {}),
        }

    def get_learning_rate(self) -> float:
        """
        Get current learning rate.

        Returns:
            Current learning rate (float)
        """
        # Get LR from first param group (all groups have same LR in our case)
        return self.optimizer.param_groups[0]["lr"]

    def step_scheduler(self):
        """Step the learning rate scheduler."""
        self.scheduler.step()
