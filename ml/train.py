"""
Main Training Script

Entry point for running the AlphaZero training pipeline.

Usage:
    # Start fresh training
    python ml/train.py --iterations 500

    # Resume from checkpoint
    python ml/train.py --iterations 500 --resume models/checkpoints/checkpoint_100.pth

    # Use custom config
    python ml/train.py --config configs/my_config.json --iterations 100

    # Fast test run
    python ml/train.py --fast --iterations 5
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch

# Import tensorboard only if available (optional dependency)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.config import TrainingConfig, get_fast_config, get_production_config
from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker
from ml.training.trainer import TrainingPipeline


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train BlobMaster AI using AlphaZero-style self-play",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training parameters
    parser.add_argument(
        '--iterations',
        type=int,
        default=500,
        help='Number of training iterations to run',
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from',
    )

    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON config file (overrides defaults)',
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Use fast config for testing/debugging',
    )

    # Hardware
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to train on',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of self-play workers (overrides config)',
    )

    # Logging
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging',
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='blobmaster',
        help='W&B project name',
    )
    parser.add_argument(
        '--wandb-run-name',
        type=str,
        default=None,
        help='W&B run name',
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level',
    )

    # Checkpointing
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory to save checkpoints (overrides config)',
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=None,
        help='Save checkpoint every N iterations (overrides config)',
    )

    return parser.parse_args()


def setup_logging(config: TrainingConfig, log_level: str = 'INFO'):
    """
    Setup logging (file logging and console).

    Args:
        config: Training configuration
        log_level: Logging level
    """
    # Create log directory
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Setup file handler
    log_file = log_dir / 'training.log'
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")


def setup_wandb(config: TrainingConfig):
    """
    Setup Weights & Biases logging.

    Args:
        config: Training configuration
    """
    try:
        import wandb

        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.to_dict(),
        )
        logging.info("Weights & Biases logging enabled")
    except ImportError:
        logging.warning(
            "wandb not installed. Install with: pip install wandb"
        )
        logging.warning("Continuing without W&B logging")


def create_network(config: TrainingConfig, device: str) -> BlobNet:
    """
    Create neural network from config.

    Args:
        config: Training configuration
        device: Device to create network on

    Returns:
        Initialized neural network
    """
    logger = logging.getLogger(__name__)

    network = BlobNet(
        state_dim=256,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_transformer_layers,
        num_heads=config.num_heads,
        feedforward_dim=config.embedding_dim * 4,  # Standard transformer ratio
        dropout=config.dropout,
    ).to(device)

    # Log network info
    num_params = sum(p.numel() for p in network.parameters())
    num_trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)

    logger.info(f"Network created: {num_params:,} total parameters")
    logger.info(f"Trainable parameters: {num_trainable:,}")
    logger.info(f"Network device: {device}")

    return network


def determine_device(device_arg: str) -> str:
    """
    Determine which device to use for training.

    Args:
        device_arg: Device argument from command line

    Returns:
        Device string ('cuda' or 'cpu')
    """
    if device_arg == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_arg

    if device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available, falling back to CPU")
        device = 'cpu'

    return device


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
    logger = logging.getLogger(__name__)

    # Validate config
    config.validate()
    logger.info("Configuration validated successfully")
    logger.info(f"\n{config}")

    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Create network
    device = config.device
    network = create_network(config, device)

    # Create encoder and masker
    encoder = StateEncoder()
    masker = ActionMasker()
    logger.info("State encoder and action masker created")

    # Create training pipeline
    pipeline = TrainingPipeline(
        network=network,
        encoder=encoder,
        masker=masker,
        config=config.to_dict(),
    )
    logger.info("Training pipeline initialized")

    # Run training
    try:
        logger.info(f"Starting training for {num_iterations} iterations...")
        pipeline.run_training(
            num_iterations=num_iterations,
            resume_from=resume_from,
        )
        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info("Saving checkpoint...")
        # Pipeline will auto-save on interrupt
        logger.info("Checkpoint saved. Training can be resumed with --resume")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


def main():
    """Main training entry point."""
    # Parse arguments
    args = parse_args()

    # Load config
    if args.config:
        config = TrainingConfig.from_file(args.config)
        print(f"Loaded config from {args.config}")
    elif args.fast:
        config = get_fast_config()
        print("Using fast config (for testing)")
    else:
        config = get_production_config()
        print("Using production config")

    # Override config with command line args
    if args.device != 'auto':
        config.device = args.device
    else:
        config.device = determine_device('auto')

    if args.workers is not None:
        config.num_workers = args.workers

    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir

    if args.save_every is not None:
        config.save_every_n_iterations = args.save_every

    if args.wandb:
        config.use_wandb = True
        config.wandb_project = args.wandb_project
        config.wandb_run_name = args.wandb_run_name

    # Setup logging
    setup_logging(config, args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("BlobMaster Training - AlphaZero Self-Play Pipeline")
    logger.info("=" * 80)

    # Setup W&B if enabled
    if config.use_wandb:
        setup_wandb(config)

    # Save config
    config_save_path = Path(config.checkpoint_dir) / 'config.json'
    config.save(str(config_save_path))
    logger.info(f"Config saved to {config_save_path}")

    # Run training
    run_training_pipeline(
        config=config,
        num_iterations=args.iterations,
        resume_from=args.resume,
    )


if __name__ == '__main__':
    main()
