"""
Training package for BlobMaster.

This package contains the AlphaZero-style training pipeline for the Blob card game AI:
- Self-play game generation
- Replay buffer for experience storage
- Training loop orchestration
- Model evaluation and ELO tracking

Main Components:
    - SelfPlayWorker: Generates training games using MCTS (IMPLEMENTED - Phase 4 Session 1)
    - SelfPlayEngine: Parallelizes self-play across multiple workers (TODO - Session 2)
    - ReplayBuffer: Stores and samples training examples (TODO - Session 3)
    - NetworkTrainer: Updates neural network weights (TODO - Session 4)
    - TrainingPipeline: Orchestrates the full training loop (TODO - Session 5)

Phase 4 Session 1 Status: COMPLETE
- SelfPlayWorker class: ✅ Implemented
- Training example generation: ✅ Implemented
- Temperature-based exploration: ✅ Implemented
- Outcome back-propagation: ✅ Implemented
- Test coverage: 98% (18 tests, all passing)
"""

from ml.training.selfplay import SelfPlayWorker

__all__ = [
    "SelfPlayWorker",
]
