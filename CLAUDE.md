# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BlobMaster is an AI-powered assistant for the card game "Blob" (trick-taking bidding game) using AlphaZero-style reinforcement learning. The project implements a neural network trained through self-play with Monte Carlo Tree Search (MCTS) to master both bidding strategy and optimal card play in an imperfect information game.

**Architecture**: Training happens on a desktop PC with RTX GPU + Python/PyTorch, while inference runs on an Intel laptop with CPU/iGPU using ONNX Runtime. The production stack uses Bun (TypeScript backend), Svelte (frontend), and SQLite (database).

## Development Environment

VERY IMPORTANT:
You are working in a Windows 11 environment, plan your console commands accordingly!
When checking for tools, use the correct Windows command syntax! For example, paths must use forward slashes!
Use Grep!
Avoid Unicode encoding errors (checkmarks and card symbols)

### Training Environment (Python)

**Setup virtual environment**:
```bash
# Create and activate venv
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r ml/requirements.txt
```

**Testing**:
```bash
# Run all tests
pytest ml/

# Run specific test file
pytest ml/game/test_blob.py

# Run with coverage
pytest --cov=ml ml/
```

**Training**:
```bash
# Start training pipeline
python ml/train.py --iterations 500 --gpu 0

# Monitor training progress
tensorboard --logdir=runs/
```

**Code Quality**:
```bash
# Format code
black ml/

# Lint
flake8 ml/

# Type checking
mypy ml/
```

**ONNX Export**:
```bash
# Export trained model for production inference
python ml/export_onnx.py --checkpoint models/checkpoints/best.pth
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
- `mcts/`: Monte Carlo Tree Search with determinization for imperfect info
- `network/`: Transformer model, state encoding, training loop
- `training/`: Self-play engine, replay buffer, training orchestration
- `evaluation/`: Model tournaments, ELO calculation, metrics

**`models/`** - Model artifacts:
- `checkpoints/`: PyTorch (.pth) model snapshots during training
- `best_model.onnx`: Production-ready exported model for inference
- `elo_history.json`: ELO progression tracking across training iterations

**`backend/`** - Bun/TypeScript API server:
- `api/`: REST/WebSocket endpoints for game management
- `inference/`: ONNX model loading and inference (uses ONNX Runtime with OpenVINO for Intel iGPU)
- `game/`: TypeScript port of game rules for server-side validation
- `db/`: SQLite queries (game history, user stats)

**`frontend/`** - Svelte UI:
- `lib/components/`: GameBoard, Hand, BidSelector, TrickHistory, ScoreBoard, AIThinking
- `lib/stores/`: Game state management (reactive stores)
- `routes/`: SvelteKit pages

**`data/`** - Persistent storage:
- `games.db`: SQLite database (game history, user stats, move analysis)
- `training_data/`: Self-play game records for training

**`docs/`** - Documentation and analysis:
- `performance/`: Performance investigation reports and findings
- `phases/`: Phase completion summaries and historical records
- `archive/`: Superseded documents and bug fixes

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

### Starting Fresh (Phase 1 in progress)
The project is currently in early stages. The README outlines a 7-phase roadmap:
1. **Core Game Engine** (Weekend 1) - Implement game rules in Python
2. **MCTS + Neural Network** (Midweek) - Basic AI infrastructure
3. **Imperfect Information Handling** (Weekend 2) - Belief tracking and determinization
4. **Self-Play Training Pipeline** (Week 2) - Automated training loop
5. **ONNX Export & Inference** (Day 15) - Production model export
6. **Backend API** (Weekend 3) - Bun server with ONNX inference
7. **Frontend UI** (Days 18-21) - Svelte web interface

### When Modifying Game Rules
1. Update `ml/game/blob.py` (Python source of truth)
2. Add unit tests to verify rule changes
3. Test with CLI version: `python ml/game/blob.py`
4. Re-train model from scratch (rules changes invalidate old training data)
5. Port rule changes to TypeScript backend (`backend/src/game/`)

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

**Expected Performance**:
- Light MCTS (2 det × 20 sims): 80 games/min
- Medium MCTS (3 det × 30 sims): 43 games/min ← **recommended for training**
- Heavy MCTS (5 det × 50 sims): 25 games/min

**Training Timeline** (500 iterations × 10,000 games each):
- Medium MCTS: ~54 days continuous training
- GPU utilization: 15-20% (acceptable for MCTS sequential nature)

**Key Findings**:
- GPU server architecture (Phase 3.5) failed: 3-5x slower due to small batch sizes
- Threading + batching failed: GIL contention and overhead exceeded benefits
- Simple multiprocessing wins: 32 workers with per-worker networks
- Future optimization: GPU-batched MCTS with virtual loss could achieve 300-600 games/min

**Detailed Analysis**: See [docs/performance/](docs/performance/) for comprehensive investigation

## Expected Training Progression

Based on similar AlphaZero projects:

- **Day 1** (ELO ~800): Random legal moves
- **Day 3** (ELO ~1200): Learned basic trick-taking rules
- **Day 7** (ELO ~1600): Strategic bidding and card counting
- **~50 days total** on RTX 4060 for strong model (with Medium MCTS)

**Hardware Requirements**:
- Training: RTX 4060 8GB GPU, 128GB RAM (parallel workers), 50GB+ storage
- Inference: Intel i5-1135G7 iGPU, 16GB RAM, <100ms latency target

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
