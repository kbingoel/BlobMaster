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
import json
import time
import copy
import numpy as np
from datetime import datetime
import logging
import shutil
import glob
import os

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.training.replay_buffer import ReplayBuffer
from ml.training.selfplay import SelfPlayEngine
from ml.evaluation.arena import Arena
from ml.evaluation.elo import ELOTracker

logger = logging.getLogger(__name__)


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

        # IMPROVEMENT: Replace StepLR with CosineAnnealingLR for smoother decay
        # OLD CODE (works, but has abrupt jumps):
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        # ^ Steps per-iteration (not per-epoch), causing sharp drops at:
        #   - Iteration 100: 0.001 → 0.0001 (sharp drop)
        #   - Iteration 200: 0.0001 → 0.00001 (sharp drop)
        #   These abrupt changes can cause training instability

        # NEW: Cosine annealing (smooth, gradual decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=500,  # Total iterations (matches training plan)
            eta_min=0.0001,  # Minimum LR (10x smaller than initial 0.001)
        )

        print("Using CosineAnnealingLR scheduler (per-iteration, not per-epoch)")

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


class StatusWriter:
    """
    Atomic status file writer for external monitoring.

    Writes training status to JSON file with atomic updates
    to prevent corruption from concurrent reads.
    """

    def __init__(self, status_file: str = "models/checkpoints/status.json"):
        """
        Args:
            status_file: Path to status file (default: models/checkpoints/status.json)
        """
        self.status_file = Path(status_file)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()

    def update(
        self,
        iteration: int,
        total_iterations: int,
        metrics: Dict[str, Any],
        config: Any
    ):
        """
        Update status file with current training state.

        Args:
            iteration: Current iteration (0-indexed)
            total_iterations: Total iterations planned
            metrics: Metrics dict from iteration
            config: Config dict or object

        Note:
            Iteration is 0-indexed internally but displayed as 1-indexed in status.
        """
        # Calculate progress (use 1-indexed for display)
        display_iteration = iteration + 1
        progress = display_iteration / total_iterations if total_iterations > 0 else 0.0

        # Estimate ETA
        elapsed = time.time() - self.start_time
        if display_iteration > 0:
            avg_iter_time = elapsed / display_iteration
            remaining_iters = total_iterations - display_iteration
            eta_seconds = avg_iter_time * remaining_iters
            eta_hours = eta_seconds / 3600
            eta_days = eta_hours / 24
        else:
            eta_hours = 0
            eta_days = 0

        # Get MCTS params for current iteration (with fallback)
        mcts_str = "unknown"
        try:
            # Config can be dict or object
            if isinstance(config, dict):
                if 'num_determinizations' in config and 'simulations_per_determinization' in config:
                    mcts_str = f"{config['num_determinizations']}×{config['simulations_per_determinization']}"
            elif hasattr(config, 'get_mcts_params'):
                num_det, sims = config.get_mcts_params(display_iteration)
                mcts_str = f"{num_det}×{sims}"
            elif hasattr(config, 'num_determinizations') and hasattr(config, 'simulations_per_determinization'):
                mcts_str = f"{config.num_determinizations}×{config.simulations_per_determinization}"
        except Exception as e:
            logger.warning(f"Failed to get MCTS params: {e}")

        # Get phase (with fallback)
        if isinstance(config, dict):
            phase = config.get('training_on', 'unknown')
        else:
            phase = getattr(config, 'training_on', 'unknown')

        # Build status dict
        status = {
            'timestamp': time.time(),
            'iteration': display_iteration,  # 1-indexed for display
            'total_iterations': total_iterations,
            'progress': progress,
            'elapsed_hours': elapsed / 3600,
            'eta_hours': eta_hours,
            'eta_days': eta_days,
            'phase': phase,
            'mcts_config': mcts_str,
            'elo': metrics.get('current_elo', None),
            'elo_change': metrics.get('elo_change', None),
            'learning_rate': metrics.get('learning_rate', None),
            'loss': metrics.get('avg_total_loss', None),
            'policy_loss': metrics.get('avg_policy_loss', None),
            'value_loss': metrics.get('avg_value_loss', None),
            'training_units_generated': metrics.get('training_units_generated', None),
            'unit_type': metrics.get('unit_type', 'rounds'),
            'pi_target_tau': metrics.get('pi_target_tau', None),
        }

        # Atomic write: write to temp file, then rename
        tmp_file = self.status_file.with_suffix('.json.tmp')
        try:
            with open(tmp_file, 'w') as f:
                json.dump(status, f, indent=2)

            # Atomic rename (POSIX guarantees atomicity)
            tmp_file.replace(self.status_file)
        except Exception as e:
            logger.error(f"Failed to write status file: {e}")
            # Clean up temp file if it exists
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except:
                    pass


class TrainingPipeline:
    """
    Orchestrates the full AlphaZero training pipeline.

    Manages self-play, training, evaluation, and checkpointing in a unified
    training loop. This is the main entry point for running training.

    Training Loop:
        for iteration in range(max_iterations):
            1. Self-Play Generation - Generate games using current model
            2. Network Training - Train for N epochs on replay buffer
            3. Model Evaluation - Test new model vs previous best (Session 6)
            4. Checkpoint & Logging - Save model and log metrics

    Expected Timeline:
        - Self-play: ~30 minutes (16 workers, 10k games)
        - Training: ~10 minutes (GPU, 10 epochs)
        - Evaluation: ~5 minutes (400 games)
        - Total per iteration: ~45 minutes
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
            config: Training configuration dictionary with keys:
                - num_workers: Number of parallel self-play workers
                - games_per_iteration: Games to generate per iteration
                - num_determinizations: Determinizations for MCTS
                - simulations_per_determinization: MCTS simulations
                - replay_buffer_capacity: Max replay buffer size
                - min_buffer_size: Min examples before training
                - batch_size: Training batch size
                - epochs_per_iteration: Training epochs per iteration
                - learning_rate: Learning rate
                - weight_decay: L2 regularization
                - checkpoint_dir: Directory for checkpoints
                - save_every_n_iterations: Save frequency
                - device: 'cuda' or 'cpu'
        """
        self.network = network
        self.encoder = encoder
        self.masker = masker
        self.config = config

        # Extract config values with defaults
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "models/checkpoints"))
        self.save_every_n_iterations = config.get("save_every_n_iterations", 10)
        self.device = config.get("device", "cuda")

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        print("Initializing training pipeline components...")

        # Self-play engine
        self.selfplay_engine = SelfPlayEngine(
            network=network,
            encoder=encoder,
            masker=masker,
            num_workers=config.get("num_workers", 16),
            num_determinizations=config.get("num_determinizations", 3),
            simulations_per_determinization=config.get(
                "simulations_per_determinization", 30
            ),
            device=self.device,
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config.get("replay_buffer_capacity", 500_000),
        )

        # Network trainer
        self.trainer = NetworkTrainer(
            network=network,
            learning_rate=config.get("learning_rate", 0.001),
            weight_decay=config.get("weight_decay", 1e-4),
            device=self.device,
        )

        # Evaluation system
        self.arena = Arena(
            encoder=encoder,
            masker=masker,
            num_determinizations=config.get("eval_determinizations", 3),
            simulations_per_determinization=config.get("eval_simulations", 50),
            device=self.device,
        )

        self.elo_tracker = ELOTracker(
            initial_elo=config.get("initial_elo", 1000),
            k_factor=config.get("elo_k_factor", 32),
        )

        # Load ELO history if it exists
        elo_history_path = self.checkpoint_dir / "elo_history.json"
        if elo_history_path.exists():
            self.elo_tracker.load_history(str(elo_history_path))
            print(f"  - Loaded ELO history: Current ELO = {self.elo_tracker.get_current_elo()}")

        # Training state
        self.current_iteration = 0
        self.best_model_path = None
        self.best_model_elo = config.get("initial_elo", 1000)
        self.metrics_history = []
        self.eval_frequency = config.get("eval_frequency", 5)  # Evaluate every N iterations

        # Store training start date for checkpoint naming
        self.training_start_date = datetime.now().strftime("%Y%m%d")
        self.current_elo = self.best_model_elo  # Track current ELO for checkpoint naming

        # NEW: EMA model (Session 2)
        self.ema_model = copy.deepcopy(network)
        self.ema_decay = 0.997  # Fixed value from MuZero
        self.use_ema_for_selfplay = True

        # NEW: Progressive target temperature schedule (Session 2)
        self.pi_target_tau_start = 1.0
        self.pi_target_tau_end = 0.7
        self.tau_anneal_iters = 200

        # Status writer for external monitoring
        self.status_writer = StatusWriter()
        self.total_iterations = 0  # Set by run_training()

        # Control signal file for pause/resume
        self.control_signal_file = Path("models/checkpoints/control.signal")

        print(f"Pipeline initialized:")
        print(f"  - Training session started: {self.training_start_date}")
        print(f"  - Self-play workers: {config.get('num_workers', 16)}")
        print(f"  - Replay buffer capacity: {config.get('replay_buffer_capacity', 500_000):,}")
        print(f"  - Evaluation frequency: every {self.eval_frequency} iterations")
        print(f"  - Device: {self.device}")
        print(f"  - Checkpoint directory: {self.checkpoint_dir}")
        print(f"  - EMA model: enabled (decay={self.ema_decay})")
        print(f"  - Progressive target τ: {self.pi_target_tau_start} → {self.pi_target_tau_end} over {self.tau_anneal_iters} iterations")

    def get_pi_target_tau(self, iteration: int) -> float:
        """
        Get policy target temperature for this iteration.

        Args:
            iteration: Current iteration (0-indexed)

        Returns:
            Temperature value (1.0 = uniform, lower = sharper)
        """
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

    def _check_control_signal(self) -> str:
        """
        Check for control signals from external monitor.

        Returns:
            Control command string ('PAUSE', 'CONTINUE', or empty)
        """
        if not self.control_signal_file.exists():
            return ''

        try:
            signal = self.control_signal_file.read_text().strip().upper()
            return signal
        except Exception as e:
            logger.warning(f"Failed to read control signal: {e}")
            return ''

    def _clear_control_signal(self):
        """Clear control signal file."""
        if self.control_signal_file.exists():
            try:
                self.control_signal_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to clear control signal: {e}")

    def _shutdown_workers(self):
        """
        Gracefully shutdown self-play workers.

        Note:
            Override this method based on your worker implementation.
            This is a placeholder for proper cleanup.
        """
        if hasattr(self, 'selfplay_engine') and self.selfplay_engine is not None:
            try:
                logger.info("Shutting down self-play workers...")
                # Assuming SelfPlayEngine has a shutdown method
                if hasattr(self.selfplay_engine, 'shutdown'):
                    self.selfplay_engine.shutdown()
                else:
                    # Fallback: just set flag if shutdown method doesn't exist
                    logger.warning("SelfPlayEngine has no shutdown() method")
            except Exception as e:
                logger.error(f"Error shutting down workers: {e}")

    def run_training(
        self,
        num_iterations: int,
        resume_from: Optional[str] = None,
    ):
        """
        Run the full training pipeline.

        Args:
            num_iterations: Number of training iterations to run
            resume_from: Optional checkpoint path to resume from
        """
        # Set total iterations for status tracking
        self.total_iterations = num_iterations

        # Clear any stale control signals
        self._clear_control_signal()

        # Resume from checkpoint if specified
        if resume_from:
            self._resume_from_checkpoint(resume_from)
            print(f"Resumed training from iteration {self.current_iteration}")

        print(f"\n{'='*80}")
        print(f"Starting AlphaZero Training Pipeline")
        print(f"{'='*80}")
        print(f"Total iterations: {num_iterations}")
        print(f"Games per iteration: {self.config.get('games_per_iteration', 10_000):,}")
        print(f"Epochs per iteration: {self.config.get('epochs_per_iteration', 10)}")
        print(f"Batch size: {self.config.get('batch_size', 512)}")
        print(f"{'='*80}\n")

        # Main training loop
        start_iteration = self.current_iteration
        try:
            for iteration in range(start_iteration, num_iterations):
                # Check for pause request BEFORE starting iteration
                signal = self._check_control_signal()
                if signal == 'PAUSE':
                    logger.info(
                        f"Pause signal received at iteration {iteration} (displayed as {iteration + 1}). "
                        f"Last checkpoint saved at iteration {iteration - 1}."
                    )
                    self._clear_control_signal()
                    # Checkpoint already saved at end of previous iteration
                    # Just cleanup and exit
                    break

                self.current_iteration = iteration
                print(f"\n{'='*80}")
                print(f"ITERATION {iteration + 1}/{num_iterations}")
                print(f"{'='*80}")

                # Get MCTS params for this iteration (if config supports curriculum)
                if hasattr(self.config, 'get_mcts_params'):
                    # NOTE: get_mcts_params() expects 1-indexed iteration (docstring),
                    # but Python loop is 0-indexed, so we pass iteration + 1
                    num_det, sims_per_det = self.config.get_mcts_params(iteration + 1)
                    print(f"MCTS curriculum: {num_det} determinizations × {sims_per_det} simulations")

                    # Update self-play engine config
                    self.selfplay_engine.num_determinizations = num_det
                    self.selfplay_engine.simulations_per_determinization = sims_per_det

                iteration_start_time = time.time()

                # Run single iteration
                try:
                    metrics = self.run_iteration(iteration)

                    # Log iteration time
                    iteration_time = time.time() - iteration_start_time
                    metrics["iteration_time_minutes"] = iteration_time / 60.0

                    # Log metrics
                    self._log_metrics(iteration, metrics)

                    # Distribution sanity logging (every 10 iterations)
                    if (iteration + 1) % 10 == 0 and 'distribution_stats' in metrics:
                        self._log_distribution_sanity(metrics)

                    # Store metrics history
                    self.metrics_history.append({
                        "iteration": iteration,
                        "timestamp": datetime.now().isoformat(),
                        **metrics,
                    })

                    # Update status file (atomic write for external monitoring)
                    self.status_writer.update(
                        iteration=iteration,
                        total_iterations=self.total_iterations,
                        metrics=metrics,
                        config=self.config
                    )

                    print(f"\nIteration {iteration + 1} completed in {iteration_time / 60:.1f} minutes")

                except Exception as e:
                    print(f"\nERROR in iteration {iteration + 1}: {e}")
                    # Save emergency checkpoint before raising
                    emergency_path = self.checkpoint_dir / f"emergency_iter_{iteration}.pth"
                    self._checkpoint_phase(iteration, {"error": str(e)})
                    print(f"Emergency checkpoint saved to {emergency_path}")
                    raise

            # Training completed normally or paused
            if iteration == num_iterations - 1:
                print(f"\n{'='*80}")
                print(f"Training Complete!")
                print(f"{'='*80}")
                print(f"Total iterations: {num_iterations}")
                print(f"Best model: {self.best_model_path}")
                print(f"Metrics history saved to: {self.checkpoint_dir / 'metrics_history.json'}")
            else:
                print(f"\n{'='*80}")
                print(f"Training Paused")
                print(f"{'='*80}")
                print(f"Completed iterations: {iteration}/{num_iterations}")
                print(f"Resume with: --resume <checkpoint_path>")

        finally:
            # Always cleanup workers on exit (normal or exception)
            logger.info("Cleaning up resources...")
            self._shutdown_workers()

    def run_iteration(self, iteration: int) -> Dict[str, Any]:
        """
        Run a single training iteration.

        Args:
            iteration: Current iteration number

        Returns:
            Dictionary with iteration metrics
        """
        metrics = {}

        # Phase 1: Self-play
        print(f"\n[1/4] Self-Play Phase")
        print("-" * 40)
        selfplay_examples = self._selfplay_phase(iteration)
        metrics["num_selfplay_examples"] = len(selfplay_examples)
        metrics["replay_buffer_size"] = len(self.replay_buffer)

        # Add distribution stats if available
        if hasattr(self, 'last_distribution_stats') and self.last_distribution_stats is not None:
            metrics["distribution_stats"] = self.last_distribution_stats

        # Phase 2: Training
        print(f"\n[2/4] Training Phase")
        print("-" * 40)
        training_metrics = self._training_phase(
            iteration,
            num_epochs=self.config.get("epochs_per_iteration", 10),
        )
        metrics.update(training_metrics)

        # Phase 3: Evaluation (stub for now, will implement in Session 6)
        print(f"\n[3/4] Evaluation Phase")
        print("-" * 40)
        eval_metrics = self._evaluation_phase(iteration)
        metrics.update(eval_metrics)

        # Phase 4: Checkpoint
        print(f"\n[4/4] Checkpoint Phase")
        print("-" * 40)
        self._checkpoint_phase(iteration, metrics)

        return metrics

    def _selfplay_phase(self, iteration: int) -> list:
        """
        Generate self-play games for this iteration.

        Args:
            iteration: Current iteration number

        Returns:
            List of training examples from generated games
        """
        # NEW: Use adaptive training curriculum (Session 2)
        if hasattr(self.config, 'get_training_units_per_iteration'):
            num_games = self.config.get_training_units_per_iteration(iteration)
            unit_type = "rounds" if self.config.get('training_on', 'rounds') == 'rounds' else "games"
            print(f"Generating {num_games:,} {unit_type} (adaptive curriculum: linear ramp)...")
        else:
            num_games = self.config.get("games_per_iteration", 10_000)
            print(f"Generating {num_games:,} self-play games...")

        # NEW: Use EMA model for self-play (Session 2)
        if self.use_ema_for_selfplay:
            print(f"  - Using EMA model for self-play (stable policy)")
            model_for_selfplay = self.ema_model
            # Update self-play engine to use EMA model
            if hasattr(self.selfplay_engine, 'update_model'):
                self.selfplay_engine.update_model(model_for_selfplay)

        start_time = time.time()

        # Progress callback
        games_generated = [0]  # Use list to allow mutation in closure

        def progress_callback(count: int):
            games_generated[0] = count
            if count % 100 == 0 or count == num_games:
                elapsed = time.time() - start_time
                rate = count / elapsed if elapsed > 0 else 0
                print(f"  Generated {count:,}/{num_games:,} games ({rate:.1f} games/sec)")

        # Generate games
        examples = self.selfplay_engine.generate_games(
            num_games=num_games,
            num_players=4,
            cards_to_deal=5,
            progress_callback=progress_callback,
        )

        elapsed_time = time.time() - start_time

        print(f"Self-play complete:")
        print(f"  - Games generated: {num_games:,}")
        print(f"  - Training examples: {len(examples):,}")
        print(f"  - Time: {elapsed_time / 60:.1f} minutes")
        print(f"  - Rate: {num_games / elapsed_time:.1f} games/sec")

        # NEW: Apply progressive target sharpening (Session 2)
        # Get policy target temperature for this iteration
        tau = self.get_pi_target_tau(iteration)
        print(f"  - Policy target τ = {tau:.3f}")

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
        self.replay_buffer.add_examples(examples)
        print(f"  - Replay buffer size: {len(self.replay_buffer):,}/{self.replay_buffer.capacity:,}")

        # Collect distribution statistics (if decision-weighted sampling is enabled)
        distribution_stats = None
        if hasattr(self.config, 'use_decision_weighted_sampling') and self.config.get('use_decision_weighted_sampling'):
            distribution_stats = self.selfplay_engine._collect_distribution_stats(examples)

        # Store distribution stats for logging
        self.last_distribution_stats = distribution_stats

        return examples

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
            Training metrics (avg over epochs)
        """
        # Check if buffer has enough data
        min_buffer_size = self.config.get("min_buffer_size", 10_000)
        if len(self.replay_buffer) < min_buffer_size:
            print(f"Replay buffer too small ({len(self.replay_buffer)} < {min_buffer_size}), skipping training")
            return {
                "avg_total_loss": 0.0,
                "avg_policy_loss": 0.0,
                "avg_value_loss": 0.0,
                "avg_policy_accuracy": 0.0,
            }

        batch_size = self.config.get("batch_size", 512)

        print(f"Training for {num_epochs} epochs...")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Replay buffer size: {len(self.replay_buffer):,}")
        print(f"  - Learning rate: {self.trainer.get_learning_rate():.6f}")

        # Accumulate metrics across epochs
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        policy_accuracy_sum = 0.0

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train for one epoch
            epoch_metrics = self.trainer.train_epoch(
                self.replay_buffer,
                batch_size=batch_size,
            )

            # Accumulate
            total_loss_sum += epoch_metrics["total_loss"]
            policy_loss_sum += epoch_metrics["policy_loss"]
            value_loss_sum += epoch_metrics["value_loss"]
            policy_accuracy_sum += epoch_metrics["policy_accuracy"]

            epoch_time = time.time() - epoch_start

            # Print progress every 2 epochs or on last epoch
            if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
                print(f"  Epoch {epoch + 1}/{num_epochs}: "
                      f"loss={epoch_metrics['total_loss']:.4f}, "
                      f"policy_acc={epoch_metrics['policy_accuracy']:.1f}%, "
                      f"time={epoch_time:.1f}s")

        # Step learning rate scheduler
        self.trainer.step_scheduler()

        # Get current learning rate after stepping
        current_lr = self.trainer.get_learning_rate()

        # NEW: Update EMA model (Session 2)
        self.update_ema_model()

        elapsed_time = time.time() - start_time

        # Average metrics
        avg_metrics = {
            "avg_total_loss": total_loss_sum / num_epochs,
            "avg_policy_loss": policy_loss_sum / num_epochs,
            "avg_value_loss": value_loss_sum / num_epochs,
            "avg_policy_accuracy": policy_accuracy_sum / num_epochs,
            "training_time_minutes": elapsed_time / 60.0,
            "learning_rate": current_lr,  # NEW: Add learning rate to metrics
        }

        print(f"Training complete:")
        print(f"  - Avg total loss: {avg_metrics['avg_total_loss']:.4f}")
        print(f"  - Avg policy accuracy: {avg_metrics['avg_policy_accuracy']:.1f}%")
        print(f"  - Learning rate: {current_lr:.6f}")
        print(f"  - EMA model updated (decay={self.ema_decay})")
        print(f"  - Time: {elapsed_time / 60:.1f} minutes")

        return avg_metrics

    def _evaluation_phase(self, iteration: int) -> Dict[str, Any]:
        """
        Evaluate new model vs previous best.

        Runs evaluation every N iterations (configurable via eval_frequency).
        When evaluation runs, plays a match between current model and best model,
        updates ELO ratings, and promotes model if it wins decisively.

        Args:
            iteration: Current iteration number

        Returns:
            Evaluation metrics including:
            - eval_win_rate: Win rate of current model vs best
            - eval_games_played: Number of evaluation games
            - model_promoted: Whether model was promoted to best
            - current_elo: ELO after evaluation
            - elo_change: ELO change from evaluation
            - eval_performed: Whether evaluation actually ran this iteration
        """
        # Check if we should run evaluation this iteration
        should_evaluate = (iteration + 1) % self.eval_frequency == 0

        if not should_evaluate:
            print(f"Skipping evaluation (runs every {self.eval_frequency} iterations)")
            return {
                "eval_win_rate": 0.0,
                "eval_games_played": 0,
                "model_promoted": False,
                "current_elo": self.best_model_elo,
                "elo_change": 0,
                "eval_performed": False,
            }

        print(f"Running model evaluation (iteration {iteration + 1})")

        # If this is the first evaluation, save current model as best
        if self.best_model_path is None:
            print("  - First evaluation: Current model becomes best model")
            return {
                "eval_win_rate": 1.0,
                "eval_games_played": 0,
                "model_promoted": True,
                "current_elo": self.best_model_elo,
                "elo_change": 0,
                "eval_performed": True,
            }

        # Load best model for comparison
        print(f"  - Loading best model from {self.best_model_path.name}")
        best_model = BlobNet(
            embedding_dim=self.network.embedding_dim,
            num_layers=self.network.num_layers,
            num_heads=self.network.num_heads,
            dropout=self.network.dropout,
        ).to(self.device)

        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        best_model.load_state_dict(checkpoint['model_state_dict'])

        # Run arena match
        num_eval_games = self.config.get("eval_games", 400)
        print(f"  - Playing {num_eval_games} evaluation games...")

        match_results = self.arena.play_match(
            model1=self.network,
            model2=best_model,
            num_games=num_eval_games,
            num_players=self.config.get("num_players", 4),
            cards_to_deal=self.config.get("cards_to_deal", 5),
            verbose=False,
        )

        win_rate = match_results['win_rate']
        print(f"  - Evaluation complete:")
        print(f"    - Current model win rate: {win_rate:.1%}")
        print(f"    - Average scores: {match_results['model1_avg_score']:.1f} vs {match_results['model2_avg_score']:.1f}")

        # Update ELO ratings
        new_elo = self.elo_tracker.add_match_result(
            iteration=iteration,
            model_elo=self.best_model_elo,
            opponent_elo=self.best_model_elo,  # Playing against current best
            win_rate=win_rate,
            games_played=match_results['games_played'],
            model_avg_score=match_results['model1_avg_score'],
            opponent_avg_score=match_results['model2_avg_score'],
            metadata={
                'num_determinizations': self.config.get("eval_determinizations", 3),
                'simulations_per_determinization': self.config.get("eval_simulations", 50),
            }
        )

        elo_change = new_elo - self.best_model_elo

        # Determine if model should be promoted
        promotion_threshold = self.config.get("promotion_threshold", 0.55)
        should_promote = self.elo_tracker.should_promote_model(
            win_rate, promotion_threshold
        )

        if should_promote:
            print(f"  - Model PROMOTED! (win rate {win_rate:.1%} >= {promotion_threshold:.1%})")
            print(f"    - ELO: {self.best_model_elo} -> {new_elo} ({elo_change:+d})")
            self.best_model_elo = new_elo
        else:
            print(f"  - Model NOT promoted (win rate {win_rate:.1%} < {promotion_threshold:.1%})")
            print(f"    - ELO unchanged: {self.best_model_elo}")

        # Save ELO history
        elo_history_path = self.checkpoint_dir / "elo_history.json"
        self.elo_tracker.save_history(str(elo_history_path))

        return {
            "eval_win_rate": float(win_rate),
            "eval_games_played": match_results['games_played'],
            "model_promoted": should_promote,
            "current_elo": self.best_model_elo,
            "elo_change": elo_change,
            "eval_performed": True,
            "model1_avg_score": match_results['model1_avg_score'],
            "model2_avg_score": match_results['model2_avg_score'],
        }

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
        if hasattr(self.config, 'get_mcts_params'):
            num_det, sims = self.config.get_mcts_params(iteration + 1)
        else:
            # Fallback to config defaults
            num_det = self.config.get('num_determinizations', 3)
            sims = self.config.get('simulations_per_determinization', 30)

        # Build components
        date = self.training_start_date
        project = "Blobmaster"
        version = "v1"
        workers = f"{self.config.get('num_workers', 16)}w"
        mode = self.config.get('training_on', 'rounds')  # 'rounds' or 'games'
        mcts = f"{num_det}x{sims}"
        iter_str = f"iter{iteration:03d}"  # Zero-padded to 3 digits

        # Build filename
        parts = [date, project, version, workers, mode, mcts, iter_str]

        # Add ELO if provided (permanent checkpoints only)
        if is_permanent and elo is not None:
            parts.append(f"elo{elo}")

        filename = "-".join(parts) + ".pth"
        return filename

    def save_checkpoint_with_rotation(
        self,
        iteration: int,
        metrics: Optional[dict] = None
    ) -> str:
        """
        Save checkpoint with automatic rotation logic.

        Args:
            iteration: Current iteration (0-indexed)
            metrics: Optional metrics dict (should contain 'current_elo' if available)

        Returns:
            Path to saved checkpoint

        Note:
            - Permanent checkpoints: iterations 4, 9, 14, ... (every 5th, 0-indexed)
            - Cache checkpoints: all others (max 4 kept via FIFO rotation)
            - Iteration is 0-indexed internally
        """
        # Determine if this is a permanent checkpoint (every 5th iteration)
        # iteration 4, 9, 14, 19, ... → displayed as 5, 10, 15, 20, ...
        is_permanent = (iteration + 1) % 5 == 0

        # Get ELO if available (only for permanent checkpoints)
        elo = None
        if is_permanent and metrics and 'current_elo' in metrics:
            try:
                elo = int(metrics['current_elo'])
            except (ValueError, TypeError):
                logger.warning(f"Invalid ELO value: {metrics['current_elo']}, skipping ELO in filename")
                elo = None

        # Generate filename using existing method
        filename = self._get_checkpoint_filename(
            iteration=iteration,
            elo=elo,
            is_permanent=is_permanent
        )

        # Determine directory
        if is_permanent:
            save_dir = Path("models/checkpoints/permanent")
        else:
            save_dir = Path("models/checkpoints/cache")

        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / filename

        # Build checkpoint data
        checkpoint_data = {
            'iteration': iteration,  # 0-indexed
            'model_state_dict': self.network.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'scheduler_state_dict': self.trainer.scheduler.state_dict(),
            'replay_buffer': self.replay_buffer.get_state(),
            'elo': elo,
            'metrics': metrics,
            'config': self.config,
            'training_start_date': getattr(self, 'training_start_date', None),
        }

        # Save checkpoint with error handling
        try:
            # Check available disk space (require at least 2GB free)
            stat = shutil.disk_usage(save_dir)
            free_gb = stat.free / (1024**3)
            if free_gb < 2.0:
                logger.error(f"Low disk space: {free_gb:.1f}GB free. Checkpoint may fail.")

            torch.save(checkpoint_data, filepath)

            logger.info(
                f"Saved {'permanent' if is_permanent else 'cache'} checkpoint: "
                f"{filename} ({iteration+1}/{self.total_iterations if hasattr(self, 'total_iterations') else '?'})"
            )
        except Exception as e:
            logger.error(f"Failed to save checkpoint {filepath}: {e}")
            raise

        # Rotate cache checkpoints if needed
        if not is_permanent:
            self._rotate_cache_checkpoints(max_keep=4)

        return str(filepath)

    def _rotate_cache_checkpoints(self, max_keep: int = 4):
        """
        Keep only the most recent N cache checkpoints, delete older ones.

        Args:
            max_keep: Maximum number of cache checkpoints to keep (default: 4)

        Note:
            Permanent checkpoints are never deleted by this method.
        """
        cache_dir = Path("models/checkpoints/cache")
        if not cache_dir.exists():
            return

        # Get all cache checkpoint files
        cache_files = sorted(
            cache_dir.glob("*.pth"),
            key=lambda p: p.stat().st_mtime  # Sort by modification time
        )

        # Delete oldest files if we exceed max_keep
        num_to_delete = len(cache_files) - max_keep
        if num_to_delete > 0:
            for old_file in cache_files[:num_to_delete]:
                try:
                    file_size_mb = old_file.stat().st_size / (1024**2)
                    logger.info(
                        f"Rotating out old cache checkpoint: {old_file.name} "
                        f"({file_size_mb:.1f}MB)"
                    )
                    old_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete old checkpoint {old_file}: {e}")

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
        # Determine checkpoint type: permanent (every 5) or cache (others)
        is_permanent = (iteration + 1) % 5 == 0
        should_save = (iteration + 1) % self.save_every_n_iterations == 0

        if should_save:
            # Get ELO for permanent checkpoints
            elo = None
            if is_permanent:
                elo = int(self.current_elo) if hasattr(self, 'current_elo') else None

            # Generate standardized filename
            filename = self._get_checkpoint_filename(iteration, elo, is_permanent)

            # Determine save directory (permanent vs cache)
            if is_permanent:
                save_dir = self.checkpoint_dir / "permanent"
                save_dir.mkdir(parents=True, exist_ok=True)
            else:
                save_dir = self.checkpoint_dir / "cache"
                save_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = save_dir / filename

            checkpoint_type = "permanent" if is_permanent else "cache"
            print(f"Saving {checkpoint_type} checkpoint to {filename}...")
            self.trainer.save_checkpoint(
                str(checkpoint_path),
                iteration=iteration,
                metrics=metrics,
            )

            # Also save replay buffer
            buffer_filename = filename.replace(".pth", "_buffer.pkl")
            buffer_path = save_dir / buffer_filename
            self.replay_buffer.save(str(buffer_path))

            print(f"  - Checkpoint saved: {checkpoint_path}")
            print(f"  - Replay buffer saved: {buffer_path}")

            # Rotate cache checkpoints if this is a cache checkpoint
            if not is_permanent:
                self._rotate_cache_checkpoints(max_keep=4)

        # Save best model only if promoted or if this is the first save
        should_save_best = (
            metrics.get("model_promoted", False) or
            self.best_model_path is None
        )

        if should_save_best:
            # Update current ELO for checkpoint naming
            if 'current_elo' in metrics:
                self.current_elo = metrics['current_elo']

            # Use standardized naming for best model too
            best_filename = self._get_checkpoint_filename(
                iteration,
                elo=int(self.current_elo) if hasattr(self, 'current_elo') else None,
                is_permanent=True  # Best model is always permanent
            )
            best_model_path = self.checkpoint_dir / f"best_{best_filename}"
            self.trainer.save_checkpoint(
                str(best_model_path),
                iteration=iteration,
                metrics=metrics,
            )
            self.best_model_path = best_model_path
            print(f"  - Best model updated: {best_model_path.name}")
        else:
            print(f"  - Best model unchanged (current model not promoted)")

        # Save metrics history
        metrics_path = self.checkpoint_dir / "metrics_history.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

    def _log_metrics(
        self,
        iteration: int,
        metrics: Dict[str, Any],
    ):
        """
        Log metrics to console and files.

        Args:
            iteration: Current iteration number
            metrics: Metrics to log
        """
        # For now, just print summary (TensorBoard/W&B integration later)
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration + 1} SUMMARY")
        print(f"{'='*80}")

        # Self-play metrics
        if "num_selfplay_examples" in metrics:
            print(f"Self-Play:")
            print(f"  - Examples generated: {metrics['num_selfplay_examples']:,}")
            print(f"  - Replay buffer size: {metrics['replay_buffer_size']:,}")

        # Training metrics
        if "avg_total_loss" in metrics:
            print(f"Training:")
            print(f"  - Avg total loss: {metrics['avg_total_loss']:.4f}")
            print(f"  - Avg policy loss: {metrics['avg_policy_loss']:.4f}")
            print(f"  - Avg value loss: {metrics['avg_value_loss']:.4f}")
            print(f"  - Avg policy accuracy: {metrics['avg_policy_accuracy']:.1f}%")

        # Evaluation metrics (Session 6)
        if "eval_win_rate" in metrics and metrics["eval_win_rate"] > 0:
            print(f"Evaluation:")
            print(f"  - Win rate: {metrics['eval_win_rate']:.1%}")
            print(f"  - Games played: {metrics['eval_games_played']}")
            print(f"  - Model promoted: {metrics['model_promoted']}")

        # Timing
        if "iteration_time_minutes" in metrics:
            print(f"Timing:")
            print(f"  - Total iteration time: {metrics['iteration_time_minutes']:.1f} minutes")

        print(f"{'='*80}\n")

    def _log_distribution_sanity(self, metrics: Dict[str, Any]):
        """
        Validate sampled distributions match targets.

        Args:
            metrics: Metrics dictionary containing distribution_stats
        """
        if 'distribution_stats' not in metrics:
            return

        stats = metrics['distribution_stats']

        # Check player distribution
        total_games = sum(stats['player_counts'].values())

        # Require minimum sample size before validating distribution
        MIN_SAMPLE_SIZE = 100  # Need at least 100 rounds for valid statistics
        if total_games < MIN_SAMPLE_SIZE:
            print(f"  Sample size too small ({total_games} < {MIN_SAMPLE_SIZE}), skipping distribution check")
            return

        print(f"\n{'='*80}")
        print(f"DISTRIBUTION SANITY CHECK")
        print(f"{'='*80}")

        # Check player distribution (if config has player_distribution)
        if hasattr(self.config, 'player_distribution'):
            print(f"Player Distribution (n={total_games}):")
            for num_players, target_pct in self.config.player_distribution.items():
                actual_count = stats['player_counts'].get(num_players, 0)
                actual_pct = actual_count / total_games if total_games > 0 else 0.0

                print(
                    f"  {num_players}p: {actual_pct:.1%} "
                    f"(target {target_pct:.1%}, count={actual_count})"
                )

                # Assert within ±3% tolerance (only for sufficient sample size)
                if abs(actual_pct - target_pct) > 0.03:
                    print(
                        f"  ⚠️  Distribution drift detected for {num_players}p: "
                        f"{actual_pct:.1%} vs target {target_pct:.1%}"
                    )

        # Check card count distributions per (P, C) combination
        if 'card_distributions' in stats:
            print(f"\nCard Distribution:")
            for (P, C), card_dist in stats['card_distributions'].items():
                print(f"  {P}p/C={C}:")
                for c, count in sorted(card_dist.items()):
                    pct = count / total_games if total_games > 0 else 0.0
                    print(f"    c={c}: {pct:.1%} (count={count})")

        print(f"{'='*80}\n")

    def _resume_from_checkpoint(self, checkpoint_path: str):
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Resuming from checkpoint: {checkpoint_path}")

        # Load trainer checkpoint
        metadata = self.trainer.load_checkpoint(checkpoint_path)
        self.current_iteration = metadata["iteration"] + 1  # Start from next iteration

        # Try to load replay buffer
        buffer_path = Path(checkpoint_path).parent / f"replay_buffer_iter_{metadata['iteration'] + 1}.pkl"
        if buffer_path.exists():
            print(f"Loading replay buffer from {buffer_path.name}...")
            self.replay_buffer.load(str(buffer_path))
            print(f"  - Replay buffer size: {len(self.replay_buffer):,}")
        else:
            print(f"  - Replay buffer not found, starting with empty buffer")

        # Try to load metrics history
        metrics_path = Path(checkpoint_path).parent / "metrics_history.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                self.metrics_history = json.load(f)
            print(f"  - Loaded {len(self.metrics_history)} historical metrics")

        print(f"Resume complete. Starting from iteration {self.current_iteration + 1}")
