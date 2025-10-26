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
from datetime import datetime

from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.training.replay_buffer import ReplayBuffer
from ml.training.selfplay import SelfPlayEngine


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

        # Training state
        self.current_iteration = 0
        self.best_model_path = None
        self.metrics_history = []

        print(f"Pipeline initialized:")
        print(f"  - Self-play workers: {config.get('num_workers', 16)}")
        print(f"  - Replay buffer capacity: {config.get('replay_buffer_capacity', 500_000):,}")
        print(f"  - Device: {self.device}")
        print(f"  - Checkpoint directory: {self.checkpoint_dir}")

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
        for iteration in range(start_iteration, num_iterations):
            self.current_iteration = iteration
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration + 1}/{num_iterations}")
            print(f"{'='*80}")

            iteration_start_time = time.time()

            # Run single iteration
            try:
                metrics = self.run_iteration(iteration)

                # Log iteration time
                iteration_time = time.time() - iteration_start_time
                metrics["iteration_time_minutes"] = iteration_time / 60.0

                # Log metrics
                self._log_metrics(iteration, metrics)

                # Store metrics history
                self.metrics_history.append({
                    "iteration": iteration,
                    "timestamp": datetime.now().isoformat(),
                    **metrics,
                })

                print(f"\nIteration {iteration + 1} completed in {iteration_time / 60:.1f} minutes")

            except Exception as e:
                print(f"\nERROR in iteration {iteration + 1}: {e}")
                # Save emergency checkpoint before raising
                emergency_path = self.checkpoint_dir / f"emergency_iter_{iteration}.pth"
                self._checkpoint_phase(iteration, {"error": str(e)})
                print(f"Emergency checkpoint saved to {emergency_path}")
                raise

        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"{'='*80}")
        print(f"Total iterations: {num_iterations}")
        print(f"Best model: {self.best_model_path}")
        print(f"Metrics history saved to: {self.checkpoint_dir / 'metrics_history.json'}")

        # Shutdown self-play engine
        self.selfplay_engine.shutdown()

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
        num_games = self.config.get("games_per_iteration", 10_000)

        print(f"Generating {num_games:,} self-play games...")
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

        # Add examples to replay buffer
        self.replay_buffer.add_examples(examples)
        print(f"  - Replay buffer size: {len(self.replay_buffer):,}/{self.replay_buffer.capacity:,}")

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

        elapsed_time = time.time() - start_time

        # Average metrics
        avg_metrics = {
            "avg_total_loss": total_loss_sum / num_epochs,
            "avg_policy_loss": policy_loss_sum / num_epochs,
            "avg_value_loss": value_loss_sum / num_epochs,
            "avg_policy_accuracy": policy_accuracy_sum / num_epochs,
            "training_time_minutes": elapsed_time / 60.0,
        }

        print(f"Training complete:")
        print(f"  - Avg total loss: {avg_metrics['avg_total_loss']:.4f}")
        print(f"  - Avg policy accuracy: {avg_metrics['avg_policy_accuracy']:.1f}%")
        print(f"  - Time: {elapsed_time / 60:.1f} minutes")

        return avg_metrics

    def _evaluation_phase(self, iteration: int) -> Dict[str, Any]:
        """
        Evaluate new model vs previous best.

        Args:
            iteration: Current iteration number

        Returns:
            Evaluation metrics (stub for Session 6)
        """
        # TODO: Implement in Session 6 (Arena evaluation with ELO)
        print("Evaluation not yet implemented (will be added in Session 6)")
        print("  - Skipping model comparison")

        return {
            "eval_win_rate": 0.0,
            "eval_games_played": 0,
            "model_promoted": False,
        }

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
        # Always save checkpoint at specified intervals
        should_save = (iteration + 1) % self.save_every_n_iterations == 0

        if should_save:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_iter_{iteration + 1}.pth"

            print(f"Saving checkpoint to {checkpoint_path.name}...")
            self.trainer.save_checkpoint(
                str(checkpoint_path),
                iteration=iteration,
                metrics=metrics,
            )

            # Also save replay buffer
            buffer_path = self.checkpoint_dir / f"replay_buffer_iter_{iteration + 1}.pkl"
            self.replay_buffer.save(str(buffer_path))

            print(f"  - Checkpoint saved: {checkpoint_path.name}")
            print(f"  - Replay buffer saved: {buffer_path.name}")

        # Save best model (based on evaluation, but for now just save latest)
        # TODO: In Session 6, only save if model wins evaluation
        best_model_path = self.checkpoint_dir / "best_model.pth"
        self.trainer.save_checkpoint(
            str(best_model_path),
            iteration=iteration,
            metrics=metrics,
        )
        self.best_model_path = best_model_path

        # Save metrics history
        metrics_path = self.checkpoint_dir / "metrics_history.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        print(f"  - Best model updated: {best_model_path.name}")

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
