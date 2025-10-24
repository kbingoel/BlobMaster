"""
Neural network model for Blob card game.

This module implements a Transformer-based neural network (BlobNet) that:
- Takes encoded game states as input (256-dim tensors)
- Outputs action probabilities (policy head) and value estimates (value head)
- Supports legal action masking to ensure only valid moves are considered

Architecture: Lightweight Transformer optimized for CPU inference (~2-3M parameters)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from pathlib import Path


class BlobNet(nn.Module):
    """
    Transformer-based neural network for Blob card game.

    Dual-head architecture:
    - Policy head: Action probabilities (bids or card plays)
    - Value head: Expected score prediction

    Architecture:
        Input (256) → Embedding (256) → Transformer (6 layers) → Dual Heads
        - Policy head: Softmax over action space (52 dims for max flexibility)
        - Value head: Tanh activation for [-1, 1] value estimate

    Optimized for:
        - CPU inference on Intel i5-1135G7 iGPU
        - Compact model size (~2-3M parameters)
        - Fast inference (<10ms per forward pass)
    """

    def __init__(
        self,
        state_dim: int = 256,
        embedding_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        max_bid: int = 13,
        max_cards: int = 52,
    ):
        """
        Initialize BlobNet.

        Args:
            state_dim: Dimension of input state vector (default: 256)
            embedding_dim: Dimension of embedding space (default: 256)
            num_layers: Number of Transformer encoder layers (default: 6)
            num_heads: Number of attention heads (default: 8)
            feedforward_dim: Dimension of feedforward network (default: 1024)
            dropout: Dropout rate (default: 0.1)
            max_bid: Maximum bid value (default: 13 cards)
            max_cards: Maximum number of cards in deck (default: 52)
        """
        super(BlobNet, self).__init__()

        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.max_bid = max_bid
        self.max_cards = max_cards

        # Action space size: max of bidding actions and card playing actions
        # Bidding: 0-13 (14 actions), Playing: 0-51 (52 actions)
        # Use 52 for unified action space
        self.action_dim = max(max_bid + 1, max_cards)

        # Input embedding layer
        self.input_embedding = nn.Linear(state_dim, embedding_dim)
        self.input_norm = nn.LayerNorm(embedding_dim)

        # Positional encoding (learned, not sinusoidal)
        # Single position for flattened state vector
        self.positional_encoding = nn.Parameter(
            torch.randn(1, 1, embedding_dim) * 0.02  # Small random init
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            activation='relu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Policy head (for both bidding and card playing)
        self.policy_fc = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.action_dim),
        )

        # Value head (expected score prediction)
        self.value_fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        state: torch.Tensor,
        legal_actions_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state: (batch_size, state_dim) or (state_dim,)
                   Encoded game state tensor
            legal_actions_mask: (batch_size, action_dim) or (action_dim,)
                                Binary mask where 1 = legal action, 0 = illegal
                                If provided, illegal actions get 0 probability

        Returns:
            policy: (batch_size, action_dim) or (action_dim,)
                    Action probabilities (sums to 1.0)
            value: (batch_size, 1) or (1,)
                   Expected score prediction in [-1, 1]

        Example:
            >>> net = BlobNet()
            >>> state = torch.randn(256)
            >>> mask = torch.ones(52)
            >>> mask[10:] = 0  # Only first 10 actions legal
            >>> policy, value = net(state, mask)
            >>> policy.shape
            torch.Size([52])
            >>> value.shape
            torch.Size([1])
            >>> policy.sum()  # Should be ~1.0
            tensor(1.0000)
        """
        # Handle single state (add batch dimension)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single_input = True
        else:
            single_input = False

        batch_size = state.size(0)

        # Handle mask dimension
        if legal_actions_mask is not None and legal_actions_mask.dim() == 1:
            legal_actions_mask = legal_actions_mask.unsqueeze(0)

        # Input embedding
        x = self.input_embedding(state)  # (batch, embedding_dim)
        x = self.input_norm(x)

        # Add positional encoding
        x = x.unsqueeze(1)  # (batch, 1, embedding_dim)
        x = x + self.positional_encoding  # Broadcast across batch

        # Transformer encoding
        x = self.transformer(x)  # (batch, 1, embedding_dim)
        x = x.squeeze(1)  # (batch, embedding_dim)

        # Policy head - compute logits
        policy_logits = self.policy_fc(x)  # (batch, action_dim)

        # Apply legal action masking if provided
        if legal_actions_mask is not None:
            # Set illegal actions to very negative value before softmax
            # This ensures they get ~0 probability
            policy_logits = policy_logits.masked_fill(
                legal_actions_mask == 0,
                float('-inf')
            )

        # Apply softmax to get probabilities
        policy = F.softmax(policy_logits, dim=-1)

        # Value head
        value = self.value_fc(x)  # (batch, 1)

        # Remove batch dimension if single input
        if single_input:
            policy = policy.squeeze(0)
            value = value.squeeze(0)

        return policy, value

    def get_num_parameters(self) -> int:
        """
        Get total number of trainable parameters.

        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BlobNetTrainer:
    """
    Training utilities for BlobNet.

    Handles:
    - Loss computation (policy + value)
    - Optimization (Adam with gradient clipping)
    - Training steps
    - Checkpointing

    Loss Function:
        total_loss = policy_weight * policy_loss + value_weight * value_loss

        Policy Loss: Cross-entropy between MCTS policy and network policy
        Value Loss: MSE between predicted value and actual game outcome
    """

    def __init__(
        self,
        model: BlobNet,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        value_loss_weight: float = 1.0,
        policy_loss_weight: float = 1.0,
        device: str = 'cpu',
    ):
        """
        Initialize trainer.

        Args:
            model: BlobNet model to train
            learning_rate: Learning rate for Adam optimizer (default: 0.001)
            weight_decay: L2 regularization weight (default: 1e-4)
            value_loss_weight: Weight for value loss (default: 1.0)
            policy_loss_weight: Weight for policy loss (default: 1.0)
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.to(device)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.value_loss_weight = value_loss_weight
        self.policy_loss_weight = policy_loss_weight

    def compute_loss(
        self,
        state: torch.Tensor,
        target_policy: torch.Tensor,
        target_value: torch.Tensor,
        legal_actions_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            state: (batch_size, state_dim) - Encoded game states
            target_policy: (batch_size, action_dim) - Target action probs from MCTS
            target_value: (batch_size, 1) - Target values from game outcomes
            legal_actions_mask: (batch_size, action_dim) - Legal action mask

        Returns:
            total_loss: Combined weighted loss tensor
            loss_dict: Dictionary with individual loss components
                - 'total_loss': Combined loss value
                - 'policy_loss': Policy loss value
                - 'value_loss': Value loss value

        Note:
            Policy loss uses cross-entropy, which measures how well the network's
            action probabilities match the MCTS search results.

            Value loss uses MSE, which measures how accurately the network
            predicts final game scores.
        """
        # Forward pass
        pred_policy, pred_value = self.model(state, legal_actions_mask)

        # Policy loss: Cross-entropy between target and predicted policies
        # KL divergence: sum(target * log(target/pred))
        # Cross-entropy: -sum(target * log(pred))
        # We use cross-entropy since we want to match MCTS distribution
        policy_loss = -torch.sum(
            target_policy * torch.log(pred_policy + 1e-8),  # Add epsilon for stability
            dim=-1
        ).mean()

        # Value loss: MSE between predicted value and actual outcome
        value_loss = F.mse_loss(pred_value, target_value)

        # Combined loss
        total_loss = (
            self.policy_loss_weight * policy_loss +
            self.value_loss_weight * value_loss
        )

        loss_dict = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
        }

        return total_loss, loss_dict

    def train_step(
        self,
        state: torch.Tensor,
        target_policy: torch.Tensor,
        target_value: torch.Tensor,
        legal_actions_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform single training step.

        Args:
            state: (batch_size, state_dim)
            target_policy: (batch_size, action_dim)
            target_value: (batch_size, 1)
            legal_actions_mask: (batch_size, action_dim)

        Returns:
            Dictionary with loss values

        Steps:
            1. Zero gradients
            2. Compute loss
            3. Backward pass
            4. Gradient clipping (max_norm=1.0)
            5. Optimizer step
        """
        # Ensure data is on correct device
        state = state.to(self.device)
        target_policy = target_policy.to(self.device)
        target_value = target_value.to(self.device)
        legal_actions_mask = legal_actions_mask.to(self.device)

        # Zero gradients
        self.optimizer.zero_grad()

        # Compute loss
        loss, loss_dict = self.compute_loss(
            state, target_policy, target_value, legal_actions_mask
        )

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step
        self.optimizer.step()

        return loss_dict

    def save_checkpoint(
        self,
        filepath: str,
        iteration: int,
        metadata: Optional[Dict] = None,
    ):
        """
        Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
            iteration: Current training iteration
            metadata: Optional metadata to save (e.g., ELO, win rate)

        Checkpoint contains:
            - iteration: Training iteration number
            - model_state_dict: Model weights
            - optimizer_state_dict: Optimizer state
            - metadata: Optional training metadata
        """
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        if metadata is not None:
            checkpoint['metadata'] = metadata

        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, filepath)

    def load_checkpoint(
        self,
        filepath: str,
        load_optimizer: bool = True,
    ) -> Tuple[int, Optional[Dict]]:
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint file
            load_optimizer: Whether to load optimizer state (default: True)

        Returns:
            iteration: Training iteration from checkpoint
            metadata: Optional metadata from checkpoint

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if requested
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        iteration = checkpoint.get('iteration', 0)
        metadata = checkpoint.get('metadata', None)

        return iteration, metadata


def create_model(
    state_dim: int = 256,
    device: str = 'cpu',
    **kwargs
) -> BlobNet:
    """
    Factory function to create BlobNet model.

    Args:
        state_dim: Dimension of input state (default: 256)
        device: Device to place model on (default: 'cpu')
        **kwargs: Additional arguments passed to BlobNet constructor

    Returns:
        BlobNet model instance

    Example:
        >>> model = create_model(state_dim=256, num_layers=4)
        >>> print(f"Model has {model.get_num_parameters():,} parameters")
    """
    model = BlobNet(state_dim=state_dim, **kwargs)
    model.to(device)
    return model


def create_trainer(
    model: BlobNet,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    **kwargs
) -> BlobNetTrainer:
    """
    Factory function to create BlobNetTrainer.

    Args:
        model: BlobNet model to train
        learning_rate: Learning rate (default: 0.001)
        device: Device for training (default: 'cpu')
        **kwargs: Additional arguments passed to BlobNetTrainer constructor

    Returns:
        BlobNetTrainer instance

    Example:
        >>> model = create_model()
        >>> trainer = create_trainer(model, learning_rate=0.001)
    """
    return BlobNetTrainer(
        model=model,
        learning_rate=learning_rate,
        device=device,
        **kwargs
    )
