# BlobMaster

An AlphaZero-style reinforcement learning agent for the trick-taking card game **Blob**, trained through self-play with Monte Carlo Tree Search (MCTS) to master bidding strategy and card play.

**Current Status**: Phase 1 training infrastructure complete. Ready to train on independent rounds (**~3-5 days** with optimized baseline). Phase 2 (full multi-round games) not yet implemented.

---

## Terminology - READ THIS FIRST

**CRITICAL**: This project uses specific terminology to distinguish training modes:

- **Round**: A single deal with fixed cards (e.g., one 5-card bidding + trick-taking cycle). Phase 1 trains on independent rounds sampled randomly.
  - **Performance metric**: rounds/min (741-1,049 rounds/min achieved on RTX 4060, varies by MCTS complexity)

- **Game**: A complete Blob game consisting of a full sequence of rounds (e.g., 17 rounds for 5 players: 7→6→5→4→3→2→1→1→1→1→1→2→3→4→5→6→7). Phase 2 trains on complete game sequences.
  - **Performance metric**: games/min (NOT YET IMPLEMENTED - full game training is Session 4-5 future work)

**Note**: Older code and documentation may inconsistently use "game" to mean what we now define as "round". This document represents the current baseline terminology.

---

## Implementation Status

### ✅ Complete

**Phase 1-3: Core ML Infrastructure**
- Game engine with 135 tests, 97% coverage ([ml/game/blob.py](ml/game/blob.py))
- Neural network: Transformer architecture, ~4.9M parameters ([ml/network/model.py](ml/network/model.py))
- MCTS with determinization for imperfect information ([ml/mcts/](ml/mcts/))
- Belief tracking and suit elimination
- 460 tests total across all components

**Phase 4: Training Pipeline (Partial)**
- Self-play engine with multiprocessing workers ([ml/training/selfplay.py](ml/training/selfplay.py))
- Replay buffer with 500K capacity ([ml/training/replay_buffer.py](ml/training/replay_buffer.py))
- Network training loop with Adam optimizer ([ml/training/trainer.py](ml/training/trainer.py))
- ELO evaluation system ([ml/evaluation/](ml/evaluation/))
- Main training script ([ml/train.py](ml/train.py))

**Training Infrastructure Sessions (TRAINING-TODO.md)**
- ✅ Session 0: MCTS curriculum integration & CLI flags
- ✅ Session 1: Zero-choice fast path optimization
- ✅ Session 2: Training stabilization & linear curriculum
- ✅ Session 3: Exploration noise (Dirichlet α at root)
- ✅ Session 6: External monitoring & checkpoint management

### ❌ Not Implemented

**Phase 4 Remaining (Sessions 4-5)** - ~8 hours of work:
- Session 4: Full multi-round game evaluation infrastructure
- Session 5: Full multi-round game training mode (Phase 2)

**Phase 5-7: Production Deployment**
- ONNX model export for inference
- Bun/TypeScript backend API (directories exist, no code)
- Svelte frontend UI (directories exist, no code)

---

## Training Readiness

**Phase 1 (Independent Rounds)**: ✅ **READY NOW**
- Train on randomly sampled single rounds
- Performance: 741-1,049 rounds/min (32 workers, RTX 4060, Medium-Light MCTS)
- Timeline: **~3-5 days** for 5M rounds (500 iterations × 10K each)
- Command: `python ml/train.py --iterations 500 --training-on rounds`

**Phase 2 (Full Game Sequences)**: ❌ **NOT READY**
- Requires completing Sessions 4-5 (~8 hours implementation)
- Would train on complete 17-round game sequences
- Estimated timeline: TBD (needs benchmarking after implementation)

**Recommendation**: Start Phase 1 training now, implement Phase 2 later if multi-round strategy learning is needed.

---

## Game Rules - Blob Variant

### Overview
Blob is a trick-taking card game where players bid on the **exact** number of tricks they'll win, then play to meet their bid precisely. All-or-nothing scoring creates high-stakes decisions.

### Setup
- **Players**: 3-8 players (variable per game)
- **Deck**: Standard 52-card deck
- **Rounds**: Variable cards dealt (typically: 7→6→5→4→3→2→1→1→1...→2→3→4→5→6→7)
- **Trump**: Rotates through all four suits, then no-trump rounds

### Bidding Phase
- Players bid **sequentially** on how many tricks they expect to win
- **Last bidder constraint**: Dealer cannot bid such that total bids = cards dealt
  - Creates strategic tension: last position has information but a constraint

### Playing Phase
- Standard trick-taking:
  - Must follow suit if possible
  - Highest card in led suit wins (unless trump played)
  - Trump cards beat non-trump cards
  - Winner of trick leads next trick

### Scoring
- **Exact bids only**: `score = (tricks_won == bid) ? (10 + bid) : 0`
- Examples:
  - Bid 2, won 2 tricks: **12 points**
  - Bid 3, won 4 tricks: **0 points** (bust)
  - Bid 0, won 0 tricks: **10 points** (risky but rewarding)

This all-or-nothing scoring rewards accurate self-assessment and risk management.

---

## Performance Benchmarks

**Platform**: Ubuntu 24.04, RTX 4060 8GB, Ryzen 9 7950X, 128GB RAM, Python 3.14

### Phase 1 Training (Independent Rounds)

**Official Baseline** (2025-11-13): Tested with 500 rounds on fixed 5-card deals. See [benchmarks/docs/archive/BASELINE.md](benchmarks/docs/archive/BASELINE.md).

| MCTS Config | Det × Sims | Total Sims | Rounds/Min | Training Timeline (5M rounds)* |
|-------------|------------|------------|------------|--------------------------------|
| **Light**   | 2 × 20     | 40         | **1,049** 🏆 | **~3.3 days** (fastest)        |
| **Medium**  | 3 × 30     | 90         | **741** ⭐   | **~4.7 days** (recommended)    |
| **Heavy**   | 5 × 50     | 250        | **310**      | **~11.2 days** (highest quality) |

*Training time = 500 iterations × 10,000 rounds = 5M rounds

**Annotations**:
- 🏆 Light MCTS = fastest iteration, excellent quality
- ⭐ Medium MCTS = recommended balance of speed/quality
- Heavy MCTS = research-grade quality, ~3.4x slower than Light

**Configuration**: 32 workers, RTX 4060 8GB, Ubuntu 24.04, Python 3.14, PyTorch CUDA 12.4

**Hardware Limit**: RTX 4060 8GB supports maximum **32 workers** before CUDA out-of-memory. 48+ workers fail with OOM errors.

**Performance Notes**:
- Zero-choice fast path enabled (skips MCTS for forced last-card plays)
- Parallel expansion with batch size 30
- Batched neural network evaluator (512 max batch, 10ms timeout)
- 96% GPU batch efficiency, 261µs per inference
- Examples per round: ~20 (validated in baseline)

### Profiling & Optimization

For detailed performance analysis and bottleneck investigations:
- [docs/profiling/PROFILING_ANALYSIS_2025-11-11.md](docs/profiling/PROFILING_ANALYSIS_2025-11-11.md) - Detailed analysis of 368 rounds/min on 5-card test
- [benchmarks/profiling/profiling-readme.md](benchmarks/profiling/profiling-readme.md) - How to run profiling tools

**Key findings from profiling**:
- 96% GPU batch efficiency (28.9/30 avg batch size)
- 261µs per neural network inference
- 100% determinization success rate (no rejection sampling)
- Multiprocessing overhead is minimal and expected
- Performance varies 5-10x based on round complexity (card count)

---

## Quick Start

### Setup (Ubuntu Linux)

```bash
# Create virtual environment with Python 3.14
python3.14 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r ml/requirements.txt
```

### Testing

```bash
# Activate venv
source venv/bin/activate

# Run all tests (460 tests)
python -m pytest ml/

# Run specific test suites
python -m pytest ml/game/test_blob.py          # Game engine (135 tests)
python -m pytest ml/network/test_network.py    # Neural network
python -m pytest ml/mcts/test_mcts.py          # MCTS
python -m pytest ml/training/test_training.py  # Training pipeline (93 tests)

# Run with coverage
python -m pytest --cov=ml ml/
```

### Training

```bash
# Phase 1: Train on independent rounds (READY NOW)
python ml/train.py --iterations 500 --training-on rounds

# Fast test run (validates pipeline, ~5 iterations in minutes)
python ml/train.py --fast --iterations 5

# Resume from checkpoint
python ml/train.py --iterations 500 --resume models/checkpoints/checkpoint_100.pth

# Custom configuration
python ml/train.py --config my_config.json --iterations 100
```

**Training parameters** (see [ml/config.py](ml/config.py)):
- `--iterations`: Number of training iterations (default: 100)
- `--training-on`: `rounds` (Phase 1) or `games` (Phase 2, not implemented)
- `--workers`: Parallel self-play workers (default: 32, max: 32 for RTX 4060)
- `--fast`: Use fast config for testing (fewer games, smaller MCTS)
- `--resume`: Resume from checkpoint path
- `--config`: Load config from JSON file

### Code Quality

```bash
# Format code
python -m black ml/

# Lint
python -m flake8 ml/

# Type checking
python -m mypy ml/
```

---

## Project Structure

```
BlobMaster/
├── ml/                          # Python training code (active development)
│   ├── game/                    # Core Blob game engine
│   │   ├── blob.py              # Main game logic (135 tests)
│   │   ├── constants.py         # Card ranks, suits, scoring
│   │   └── test_blob.py
│   ├── mcts/                    # Monte Carlo Tree Search
│   │   ├── search.py            # MCTS with determinization
│   │   ├── node.py              # MCTS node, UCB1 selection
│   │   ├── belief_tracker.py   # Belief state tracking
│   │   └── determinization.py  # Sampling opponent hands
│   ├── network/                 # Neural network
│   │   ├── model.py             # Transformer architecture (~4.9M params)
│   │   └── encode.py            # State encoder (game → 256-dim tensor)
│   ├── training/                # Self-play & training pipeline
│   │   ├── selfplay.py          # Parallel self-play workers
│   │   ├── replay_buffer.py    # Experience storage (500K capacity)
│   │   └── trainer.py           # Training orchestration
│   ├── evaluation/              # Model evaluation
│   │   ├── arena.py             # Model tournaments
│   │   └── elo.py               # ELO rating calculation
│   ├── config.py                # Centralized configuration
│   └── train.py                 # Main training entry point
│
├── models/                      # Model checkpoints
│   └── checkpoints/             # Training snapshots (.pth files)
│
├── docs/                        # Documentation
│   ├── performance/             # Performance analysis
│   ├── profiling/               # Profiling guides
│   └── phases/                  # Phase completion summaries
│
├── benchmarks/                  # Performance testing
│   ├── profiling/               # Profiling scripts & results
│   └── results/                 # Benchmark CSV data
│
├── backend/                     # Bun/TypeScript API (planned, empty)
├── frontend/                    # Svelte UI (planned, empty)
│
├── README.md                    # Original README (outdated)
├── NEW_README.md                # This file (current baseline)
├── CLAUDE.md                    # Development guide for Claude Code
└── TRAINING-TODO.md             # Remaining implementation work
```

---

## Roadmap

### Completed ✅

1. **Phase 1-3**: Core ML infrastructure
   - Game engine, neural network, MCTS, imperfect information handling
   - 460 tests, production-ready code

2. **Phase 4 (Partial)**: Training pipeline infrastructure
   - Sessions 0-3, 6 complete
   - Self-play, replay buffer, training loop, evaluation system

### In Progress 🔨

**Phase 4 Completion** (~8 hours remaining):
- Session 4: Full-game evaluation infrastructure (4 hours)
- Session 5: Full-game training mode (4 hours)

See [TRAINING-TODO.md](TRAINING-TODO.md) for detailed implementation plan.

### Planned 🔜

3. **Phase 5**: ONNX Export (after training)
   - Convert PyTorch model → ONNX format
   - Optimize for CPU/iGPU inference
   - Target: <100ms inference latency

4. **Phase 6**: Backend API (Bun + TypeScript)
   - REST/WebSocket endpoints
   - ONNX Runtime integration
   - SQLite database for game history

5. **Phase 7**: Frontend UI (Svelte)
   - Playable web interface
   - Real-time game state visualization
   - AI move explanations

---

## Development Workflow

### When to Start Training

**Option A: Start Phase 1 Now** (recommended)
- Train on independent rounds for **~3-5 days** (Medium-Light MCTS)
- Validate that training infrastructure works end-to-end
- Get a trained model for bidding/card-play on single rounds
- Implement Phase 2 later if needed

**Option B: Complete Phase 2 First** (~8 hours implementation + TBD training)
- Implement Sessions 4-5 (full-game mode)
- Train on complete 17-round game sequences
- Learn multi-round strategy and score accumulation
- Timeline depends on benchmarking (not yet measured)

### Monitoring Training

```bash
# Check training logs
tail -f logs/training_YYYYMMDD_HHMMSS.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# TensorBoard (if installed)
tensorboard --logdir=runs/
```

### Expected Training Progression

Based on AlphaZero literature and similar projects:

| Iteration | ELO  | Capability |
|-----------|------|------------|
| 0         | ~800 | Random legal moves |
| ~50       | ~1000 | Basic trick-taking (follow suit) |
| ~150      | ~1200 | Learned bidding/scoring relationship |
| ~300      | ~1400 | Strategic bidding, card counting |
| ~500      | ~1600+ | Advanced play (suit elimination, risk management) |

**Timeline**: Iterations 0→500 in **~3-5 days** (Phase 1, Medium-Light MCTS on RTX 4060)

---

## Technical Architecture

### Why AlphaZero?

- **Sample efficiency**: MCTS + neural network converges faster than pure policy gradients
- **Proven for card games**: Similar architectures dominate poker (Pluribus), bridge
- **Interpretability**: MCTS tree is visualizable to explain AI reasoning
- **Tree reuse**: Retain computed nodes when game state updates

### Network Architecture

**Lightweight Transformer** (~4.9M parameters):

```
Input: 256-dim state vector (hand, bids, tricks, belief state)
  ↓
Transformer (6 layers, 8 attention heads)
  ↓
  ├─→ Policy Head: P(action | state) with legal masking
  │   ├─ Bidding: probabilities over valid bids [0, cards_dealt]
  │   └─ Playing: probabilities over cards in hand
  │
  └─→ Value Head: Expected final score (normalized)
```

**Design choices**:
- Transformer over CNN: better at card relationships, variable-length states
- Small size: fast inference (~1.4ms on GPU)
- Dual-phase policy: handles both bidding and card-play

### MCTS with Determinization

Handles imperfect information (hidden opponent cards):

1. **Belief tracking**: Maintain probability distribution over opponent hands
2. **Sample determinizations**: Generate 2-5 possible worlds consistent with observations
3. **Run MCTS**: 20-50 simulations per determinization (training uses more)
4. **Aggregate**: Average visit counts across samples → action probabilities

**Belief updates**: When player doesn't follow suit → eliminate that suit from their possible cards (100% success rate, no rejection sampling).

### Self-Play Training Loop

```
Loop (500 iterations):
  1. Self-Play: Generate 10,000 rounds with current model + MCTS
     - 32 parallel workers (multiprocessing)
     - Progressive MCTS curriculum (20 sims → 50 sims)
     - Store (state, MCTS_policy, final_score) tuples

  2. Training: Update neural network
     - Sample batches from replay buffer (last 500K positions)
     - Loss = policy_loss + value_loss + L2_regularization
     - Adam optimizer with cosine annealing LR

  3. Evaluation: Test new model vs previous best
     - 400 games, calculate ELO ratings
     - Promote if new model wins >55%

  4. Checkpoint: Save every iteration with standardized naming
```

---

## Configuration System

Training is controlled via [ml/config.py](ml/config.py):

```python
from ml.config import get_production_config, get_fast_config

# Production config (~3-5 days training with baseline performance)
config = get_production_config()

# Fast config (testing pipeline)
config = get_fast_config()

# Custom config
config = TrainingConfig(
    num_workers=32,
    games_per_iteration=10000,
    batch_size=512,
    learning_rate=0.001,
    # ... see ml/config.py for all options
)
```

**Key parameters**:
- `num_workers`: Parallel self-play workers (default: 32, max: 32 for RTX 4060)
- `games_per_iteration`: Rounds generated per iteration (default: 10,000)
- `num_determinizations`: Worlds sampled for MCTS (default: 2-3)
- `simulations_per_determinization`: MCTS sims per world (default: 20-50, progressive)
- `replay_buffer_capacity`: Experience storage (default: 500,000)
- `eval_games`: Games for model evaluation (default: 400)
- `promotion_threshold`: Win rate to promote new model (default: 0.55)
- `mcts_schedule`: Progressive curriculum (iteration → MCTS params)

---

## Research Questions

1. **Strategy convergence**: Do models converge to same optimal strategy, or create different "styles"?
2. **Position value**: Is last bidder position advantageous (information) or disadvantageous (constraint)?
3. **Risk management**: Conservative vs aggressive bidding - which emerges?
4. **Belief accuracy**: How quickly can AI deduce opponent hands from suit information?
5. **Transfer learning**: Can 4-player model adapt to 6-player games?
6. **Exploitation**: Can AI exploit suboptimal human play patterns?

---

## Common Issues

### CUDA Out of Memory

**Problem**: Training crashes with `CUDA out of memory` error.

**Solution**: RTX 4060 8GB supports maximum **32 workers**. Reduce workers:
```bash
python ml/train.py --workers 16  # Safer, ~270 rounds/min
```

### Slow Performance

**Problem**: Training is slower than benchmarks suggest.

**Solution**:
1. Check GPU usage: `nvidia-smi` (should be >90% utilization)
2. Verify CUDA is enabled: Check logs for `device: cuda:0`
3. Use Light MCTS for faster iteration: config has progressive curriculum
4. See profiling guide: [docs/profiling/PROFILING_ANALYSIS_2025-11-11.md](docs/profiling/PROFILING_ANALYSIS_2025-11-11.md)

### Outdated Documentation

**Problem**: README.md, CLAUDE.md, or other docs contradict this file.

**Solution**: **THIS FILE (NEW_README.md) IS THE SOURCE OF TRUTH** as of 2025-11-13. Other docs may contain outdated claims (e.g., "Phase 4 complete", "games/min" metrics for unimplemented features).

---

## License

MIT License - Feel free to learn from and extend this project.

---

## Acknowledgments

- AlphaZero team at DeepMind for game AI techniques
- Pluribus team at Facebook AI for imperfect information methods
- The Blob/Oh Hell card game community

---

**Last Updated**: 2025-11-13
**Project Version**: Phase 4 (Partial), Sessions 0-3 & 6 Complete
**Training Status**: Ready for Phase 1 (independent rounds)
