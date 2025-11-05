# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BlobMaster is an AI-powered assistant for the card game "Blob" (trick-taking bidding game) using AlphaZero-style reinforcement learning. The project implements a neural network trained through self-play with Monte Carlo Tree Search (MCTS) to master both bidding strategy and optimal card play in an imperfect information game.

**Architecture**: Training and development happen on a Linux PC (Ubuntu 24.04) with RTX 4060 GPU + Python 3.14/PyTorch. Future inference deployment (Phase 7+) may use a Windows laptop with Intel iGPU + ONNX Runtime, or may be split into a separate repository. The production stack (when implemented) uses Bun (TypeScript backend), Svelte (frontend), and SQLite (database).

## Development Environment

**IMPORTANT - Development Platform:**
- **Primary Development**: Ubuntu Linux 24.04 on Ryzen 9 7950X + RTX 4060 8GB
- **Python Version**: Python 3.14.0 (with GIL enabled)
- **Training Location**: All training happens on the Linux PC
- **Future Deployment**: Windows laptop inference is Phase 7+ (future work, may be separate repo)

### Training Environment (Python)

**Setup virtual environment**:
```bash
# Create and activate venv with Python 3.14
python3.14 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA 12.9 support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r ml/requirements.txt
```

**Testing**:
```bash
# Run all tests (460 tests across all phases)
venv/Scripts/python.exe -m pytest ml/

# Run specific module tests
venv/Scripts/python.exe -m pytest ml/game/test_blob.py  # Game engine (135 tests)
venv/Scripts/python.exe -m pytest ml/network/test_network.py  # Neural network
venv/Scripts/python.exe -m pytest ml/mcts/test_mcts.py  # MCTS implementation
venv/Scripts/python.exe -m pytest ml/training/test_training.py  # Training pipeline (93 tests)

# Run integration tests
venv/Scripts/python.exe -m pytest ml/tests/test_integration.py
venv/Scripts/python.exe -m pytest ml/tests/test_imperfect_info_integration.py

# Run with coverage
venv/Scripts/python.exe -m pytest --cov=ml ml/
```

**Training**:
```bash
# Start full training pipeline (136 days @ Medium MCTS or 72 days @ Light MCTS)
venv/Scripts/python.exe ml/train.py --iterations 500

# Fast test run (validates pipeline without long wait)
venv/Scripts/python.exe ml/train.py --fast --iterations 5

# Resume from checkpoint
venv/Scripts/python.exe ml/train.py --iterations 500 --resume models/checkpoints/checkpoint_100.pth

# Use custom config
venv/Scripts/python.exe ml/train.py --config configs/my_config.json --iterations 100

# Monitor training progress (if tensorboard installed)
tensorboard --logdir=runs/
```

**Code Quality**:
```bash
# Format code
venv/Scripts/black.exe ml/

# Lint
venv/Scripts/flake8.exe ml/

# Type checking
venv/Scripts/mypy.exe ml/
```

**ONNX Export** (Phase 5 - not yet implemented):
```bash
# Export trained model for production inference
venv/Scripts/python.exe ml/export_onnx.py --checkpoint models/checkpoints/best.pth
```

### Production Environment (Bun/TypeScript)

**Backend**:
```bash
cd backend
bun install
bun run dev  # Development server on http://localhost:3000
bun test     # Run backend tests
```

**Frontend**:
```bash
cd frontend
bun install
bun run dev   # Development server on http://localhost:5173
bun run build # Production build
```

## High-Level Architecture

### Three-Environment Design

1. **Training (Python/PyTorch)**: Heavy GPU computation for self-play and neural network training
2. **Inference (ONNX)**: Lightweight cross-platform model serving
3. **Production (Bun + Svelte)**: Web application that loads ONNX model for real-time gameplay

This separation allows expensive training on powerful hardware while keeping the production app lightweight and deployable on modest hardware.

### ML Pipeline (AlphaZero-Style)

**Self-Play Training Loop**:
```
Self-Play → Replay Buffer → Network Training → Evaluation → Checkpoint
    ↑                                                            |
    └────────────────────────────────────────────────────────────┘
```

1. **Self-Play**: Generate games using current model + MCTS (parallel workers)
2. **Replay Buffer**: Store (state, MCTS_policy, final_score) tuples
3. **Training**: Update neural network via supervised learning on self-play data
4. **Evaluation**: Test new model vs previous best using ELO ratings
5. **Checkpoint**: Save model if it beats previous best by >55% win rate

**Key Components**:

- **Game State Encoder**: Converts imperfect information game state → 256-dim tensor
  - Player's hand, cards played, bids, tricks won, belief state about opponent cards
  - Handles variable player counts (3-8 players) with masking

- **Neural Network**: Lightweight Transformer (~5-8M parameters)
  - Dual-head architecture: Policy head (action probabilities) + Value head (score prediction)
  - Policy head handles both bidding phase and card-playing phase
  - Legal action masking ensures only valid moves are considered

- **MCTS with Determinization**: Handles imperfect information (hidden opponent cards)
  - Sample 3-5 possible opponent hand distributions
  - Run MCTS on each sample (200-400 simulations during training, 50-100 during inference)
  - Aggregate visit counts across samples to get final action probabilities
  - Tree reuse: Navigate to child node after opponent move to keep explored subtree

- **Belief Tracking**: Maintain probability distribution over opponent cards
  - Update beliefs when players reveal information (e.g., can't follow suit → don't have that suit)
  - Critical for accurate determinization sampling

### Project Structure Organization

**`ml/`** - Python training code (not used in production):
- `game/`: Core game rules (Blob bidding/trick-taking logic)
  - `blob.py`: Main game engine with BlobGame class
  - `constants.py`: Card ranks, suits, scoring constants
  - `test_blob.py`: 135 comprehensive tests for game logic
- `mcts/`: Monte Carlo Tree Search with determinization for imperfect info
  - `node.py`: MCTSNode with UCB1 selection
  - `search.py`: MCTS search algorithm with tree reuse
  - `belief_tracker.py`: Belief state tracking and suit elimination
  - `determinization.py`: Sampling consistent opponent hands
  - `test_mcts.py`, `test_determinization.py`: MCTS tests
- `network/`: Transformer model, state encoding, training loop
  - `model.py`: BlobNet Transformer architecture (~4.9M params)
  - `encode.py`: StateEncoder (game → 256-dim tensor) and ActionMasker
  - `test_network.py`: Neural network tests
- `training/`: Self-play engine, replay buffer, training orchestration
  - `selfplay.py`: SelfPlayWorker and SelfPlayEngine (parallel)
  - `replay_buffer.py`: ReplayBuffer for experience storage
  - `trainer.py`: NetworkTrainer and TrainingPipeline
  - `test_training.py`: 93 training pipeline tests
- `evaluation/`: Model tournaments, ELO calculation, metrics
  - `arena.py`: ModelArena for head-to-head evaluation
  - `elo.py`: ELO rating calculation and tracking
  - `test_evaluation.py`: Evaluation system tests
- `tests/`: Integration tests
  - `test_integration.py`: End-to-end game playing tests
  - `test_imperfect_info_integration.py`: Imperfect info workflow tests
- `config.py`: Centralized training configuration system
- `train.py`: Main training entry point (CLI)

**`models/`** - Model artifacts:
- `checkpoints/`: PyTorch (.pth) model snapshots during training
- `best_model.onnx`: Production-ready exported model for inference
- `elo_history.json`: ELO progression tracking across training iterations

**`backend/`** - Bun/TypeScript API server (Phase 6 - not yet implemented):
- `api/`: REST/WebSocket endpoints for game management
- `inference/`: ONNX model loading and inference (uses ONNX Runtime with OpenVINO for Intel iGPU)
- `game/`: TypeScript port of game rules for server-side validation
- `db/`: SQLite queries (game history, user stats)

**`frontend/`** - Svelte UI (Phase 7 - not yet implemented):
- `lib/components/`: GameBoard, Hand, BidSelector, TrickHistory, ScoreBoard, AIThinking
- `lib/stores/`: Game state management (reactive stores)
- `routes/`: SvelteKit pages

**`data/`** - Persistent storage:
- `games.db`: SQLite database (game history, user stats, move analysis)
- `training_data/`: Self-play game records for training

**`docs/`** - Documentation and analysis:
- `performance/`: Performance investigation reports and findings
- `phases/`: Phase completion summaries and historical records
- `archive/`: Superseded documents and bug fixes (including phase plans and session summaries)

**`benchmarks/`** - Performance benchmarking and testing:
- `performance/`: Benchmark scripts for self-play, training, and iteration timing
- `tests/`: Ad-hoc test scripts for configuration validation

**`results/`** - CSV result files from benchmarks and tests

## Key Technical Decisions

### Why AlphaZero over PPO/A3C?
- **Sample efficiency**: MCTS + neural network converges faster than pure policy gradients
- **Proven for card games**: Similar architectures dominate poker (Pluribus), bridge
- **Interpretability**: MCTS tree visualizable to explain AI reasoning
- **Tree reuse**: Retain computed nodes when game state updates incrementally

### Why Transformer over CNN?
- Better at modeling card relationships (no spatial locality assumption)
- Handles variable-length game states naturally (3-8 players, different rounds)
- Self-attention captures interactions between cards, bids, and trick history

### Why Determinization for Imperfect Information?
- Standard MCTS assumes perfect information (all cards visible)
- Determinization: Sample multiple possible worlds consistent with observations
- Average MCTS results across samples approximates Bayesian belief state
- Belief tracking updates: When player doesn't follow suit, eliminate that suit from their possible cards

### Why Bun + Svelte (not Python Flask/FastAPI)?
- **Faster startup**: No Python interpreter overhead
- **Single binary**: Easier deployment on laptop
- **File-based storage**: SQLite in repo directory (no separate database server)
- **Better inference performance**: ONNX Runtime optimized for production, OpenVINO uses Intel iGPU
- **Modern stack**: TypeScript end-to-end for type safety

## Game Rules (Blob Variant)

Understanding the game is crucial for debugging AI behavior:

- **Bidding**: Players bid sequentially on exact number of tricks they'll win
  - **Last bidder constraint**: Dealer cannot bid such that total bids = cards dealt (creates strategic tension)
- **Playing**: Standard trick-taking with trump suit
  - Must follow suit if possible
  - Trump beats non-trump, highest card in led suit wins otherwise
  - Winner leads next trick
- **Scoring**: All-or-nothing: `score = (tricks_won == bid) ? (10 + bid) : 0`
  - Exact trick count required for points
  - Creates high-stakes decisions and rewards accurate self-assessment

## Development Workflow

### Current Project Status: Phase 4 Complete ✅

The training infrastructure is fully implemented and ready for multi-day training runs. The README outlines a 7-phase roadmap:

1. ✅ **Core Game Engine** - Complete (135 tests, 97% coverage)
2. ✅ **MCTS + Neural Network** - Complete (148 tests)
3. ✅ **Imperfect Information Handling** - Complete (333 tests total)
4. ✅ **Self-Play Training Pipeline** - Complete (93 training tests, 460 total tests)
5. 🔜 **ONNX Export & Inference** - Next phase
6. 🔜 **Backend API** - Bun server with ONNX inference
7. 🔜 **Frontend UI** - Svelte web interface

**Current capabilities:**
- Full game simulation with 3-8 players
- Neural network with Transformer architecture (~4.9M parameters)
- MCTS with determinization for imperfect information
- Belief tracking and suit elimination
- Parallel self-play workers (16-32 workers)
- Replay buffer with experience management
- Training loop with checkpointing and ELO tracking
- Evaluation arena for model tournaments

**Ready to start:**
```bash
venv/Scripts/python.exe ml/train.py --fast --iterations 5  # Quick validation
venv/Scripts/python.exe ml/train.py --iterations 500        # Full training run
```

### Training Configuration System

The project uses a centralized configuration system via `ml/config.py`:

**Built-in configs:**
```python
from ml.config import TrainingConfig, get_fast_config, get_production_config

# Production config (3-7 day training)
config = get_production_config()

# Fast config (quick testing)
config = get_fast_config()  # Fewer games, workers, simulations

# Custom config
config = TrainingConfig(
    num_workers=16,
    games_per_iteration=10000,
    batch_size=512,
    learning_rate=0.001,
    # ... see ml/config.py for all options
)
```

**Key parameters:**
- `num_workers`: Parallel self-play workers (default: 16)
- `games_per_iteration`: Games per training iteration (default: 10,000)
- `num_determinizations`: Worlds sampled for imperfect info (default: 3)
- `simulations_per_determinization`: MCTS simulations per world (default: 30)
- `batch_size`: Training batch size (default: 512)
- `replay_buffer_capacity`: Experience replay size (default: 500,000)
- `eval_games`: Games for model evaluation (default: 400)
- `promotion_threshold`: Win rate to promote new model (default: 0.55)

**Using configs:**
```bash
# Use fast config via CLI
venv/Scripts/python.exe ml/train.py --fast --iterations 5

# Load from JSON file
venv/Scripts/python.exe ml/train.py --config my_config.json --iterations 100
```

### When Modifying Game Rules
1. Update `ml/game/blob.py` (Python source of truth)
2. Add unit tests to verify rule changes
3. Re-train model from scratch (rules changes invalidate old training data)
4. Port rule changes to TypeScript backend (`backend/src/game/`) when implementing Phase 6

### When Debugging AI Behavior
1. **Check belief tracking**: Are opponent cards being deduced correctly?
2. **Visualize MCTS tree**: What actions is AI considering? Visit counts?
3. **Compare determinizations**: Are sampled worlds consistent with observations?
4. **Verify legal action masking**: Is network only outputting valid moves?
5. **ELO progression**: Is model improving over iterations? Plateauing?

### When Optimizing Inference Performance
- **Reduce MCTS simulations**: 50-100 for laptop inference vs 200-400 for training
- **Batch neural network calls**: Amortize model overhead across multiple MCTS leaf evaluations
- **OpenVINO optimization**: Use Intel-specific optimizations for iGPU
- **ONNX graph optimization**: Operator fusion, constant folding

## Performance Findings & Training Configuration

After extensive performance testing (see [PERFORMANCE-FINDINGS.md](PERFORMANCE-FINDINGS.md)), we determined the optimal configuration:

**Best Configuration** (Phase 2 - Baseline Multiprocessing):
```bash
python ml/train.py --workers 32 --device cuda
```

**Current Performance** (Ubuntu 24.04 + RTX 4060, validated 2025-11-05):
- Light MCTS (2 det × 20 sims): **69.1 games/min** ← **fastest option**
- Medium MCTS (3 det × 30 sims): **36.7 games/min** ← **recommended for training**
- Heavy MCTS (5 det × 50 sims): **16.0 games/min** ← **highest quality**

**Worker Scaling** (Medium MCTS, 50 games tested per config):
- 1 worker: 2.6 games/min (baseline)
- 4 workers: 9.5 games/min (3.7x speedup, 92% efficiency)
- 8 workers: 17.1 games/min (6.6x speedup, 82% efficiency)
- 16 workers: 26.6 games/min (10.2x speedup, 64% efficiency)
- 32 workers: 36.7 games/min (14.1x speedup, 44% efficiency)
- **48 workers: FAILED** (CUDA out of memory - GPU VRAM exhausted)
- **64 workers: FAILED** (CUDA out of memory - GPU VRAM exhausted)

**Hardware Limit Identified**: RTX 4060 8GB can support maximum **32 workers** (each worker uses ~150MB GPU memory, total ~5-6GB). Beyond 32 workers causes CUDA OOM errors.

**Training Timeline** (500 iterations × 10,000 games each):
- Light MCTS @ 69.1 games/min: **~72 days continuous training**
- Medium MCTS @ 36.7 games/min: **~136 days continuous training**
- Heavy MCTS @ 16.0 games/min: **~312 days continuous training**

**Key Findings**:
- GPU server architecture (Phase 3.5) failed: 3-5x slower due to small batch sizes
- Threading + batching failed: GIL contention and overhead exceeded benefits
- Simple multiprocessing wins: 32 workers with per-worker networks
- **32 workers is optimal**: Hardware limit for RTX 4060 8GB VRAM
- Scaling efficiency degrades beyond 16 workers (diminishing returns + memory pressure)
- Linux performance validated (36.7 games/min Medium MCTS, 15% slower than historical Windows baseline of 43.3 games/min)

**Detailed Analysis**: See [docs/performance/](docs/performance/) for comprehensive investigation

## Expected Training Progression

Based on similar AlphaZero projects and validated benchmarks:

- **Day 1** (ELO ~800): Random legal moves
- **Day 3** (ELO ~1200): Learned basic trick-taking rules
- **Day 7** (ELO ~1600): Strategic bidding and card counting
- **~136 days total** on RTX 4060 for strong model (with Medium MCTS @ 36.7 games/min)
- **~72 days total** if using Light MCTS (@ 69.1 games/min, slightly lower quality)

**Hardware Requirements**:
- **Training** (Linux PC): RTX 4060 8GB GPU (supports max 32 workers), Ryzen 9 7950X 16-core, 128GB DDR5 RAM, 50GB+ storage
- **Inference** (future, Windows laptop): Intel i5-1135G7, iGPU, 16GB RAM, <500ms latency target

## Research Questions Being Explored

1. **Strategy convergence**: Do models converge to same optimal play, or do initial conditions create different "styles"?
2. **Imperfect player exploitation**: Can AI exploit suboptimal human play patterns?
3. **Position value**: Is last bidder position advantageous (information) or disadvantageous (constraint)?
4. **Risk/reward bidding**: Conservative vs aggressive bidding strategies
5. **Belief state accuracy**: How quickly can AI deduce opponent hands?
6. **Transfer learning**: Can 4-player model adapt to 6-player games?

## Common Pitfalls

### MCTS with Imperfect Information
- **Don't** run MCTS on current state with unknown opponent cards → will make incorrect assumptions
- **Do** sample multiple determinizations first, then run MCTS on each

### Legal Action Masking
- **Don't** let neural network output probabilities for invalid actions (e.g., playing card not in hand)
- **Do** apply masking layer that zeros out invalid actions before softmax

### Belief State Updates
- **Don't** assume uniform distribution over unseen cards → ignores revealed information
- **Do** track information sets: when player doesn't follow suit, remove that suit from their possible cards

### Training Data Quality
- **Don't** train only on games from early iterations → model will overfit to weak play
- **Do** maintain replay buffer with recent games (last 500k positions)

### ONNX Export Validation
- **Don't** assume ONNX output matches PyTorch without testing
- **Do** compare outputs on 100+ random states to verify numerical equivalence
