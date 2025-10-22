# BlobMaster

An AI-powered assistant for the card game **Blob** using state-of-the-art reinforcement learning techniques. This project implements an AlphaZero-style neural network trained through self-play to master both bidding strategy and optimal card play.

## Project Goals

1. **Learn modern ML/RL techniques** through a practical, scoped project
2. **Implement AlphaZero-style architecture** with MCTS + neural network
3. **Master imperfect information games** using determinization and belief tracking
4. **Build end-to-end pipeline** from game rules → self-play training → production inference
5. **Study emergent strategy** without human bias through pure self-play
6. **Track learning progression** via ELO ratings across training generations

## Game Rules - Blob Variant

### Overview
Blob is a trick-taking card game where players bid on the exact number of tricks they'll win, then play to meet their bid precisely.

### Setup
- **Players**: 3-8 players (variable per game)
- **Deck**: Standard 52-card deck
- **Rounds**: Variable cards dealt per round (typically starts with some cards, decreases, then as many 1-card rounds as there are players, then increases again to the same number of cards as in the first round)
- **Trump**: Alternaties between all four suits and then no suit. Trump in no-suit games determined by the lead card of each trick (first card played sets trump for that trick)

### Bidding Phase
- Players bid **sequentially** on how many tricks they expect to win
- **Constraint**: The last bidder (dealer) cannot bid such that the total bids equal the number of cards dealt
- This creates strategic tension: last position has information but a constraint (ending up "under" or "over")

### Playing Phase
- Standard trick-taking rules:
  - Must follow suit if possible
  - Highest card in led suit wins (unless trump played)
  - Trump cards beat non-trump cards
  - Winner of trick leads next trick (their lead card becomes trump)

### Scoring
- **Exact bids only**: Score = `(tricks_won == bid) ? (10 + bid) : 0`
- Examples:
  - Bid 2, won 2: **12 points**
  - Bid 3, won 4: **0 points** (bust)
  - Bid 0, won 0: **10 points** (risky but rewarding)

This all-or-nothing scoring creates high-stakes decisions and rewards accurate self-assessment.

## Technical Approach

### Why AlphaZero-Style Architecture?

For this project, we're using an **AlphaZero-inspired approach** rather than pure policy gradient methods (like PPO) because:

1. **Sample efficiency**: MCTS + neural network converges faster than pure RL
2. **Proven for card games**: Similar architectures dominate poker (Pluribus), bridge, and other imperfect information games
3. **Interpretability**: MCTS tree can be visualized to understand AI reasoning
4. **Handles imperfect information**: Determinization sampling works naturally with MCTS
5. **Tree reuse**: Can retain computed nodes when game state updates incrementally

### Architecture Components

#### 1. Game State Representation

The AI needs to encode a **belief state** since opponent hands are hidden:

```
State Vector (~256-512 dimensions):
├── My Hand (52-dim binary: which cards I hold)
├── Cards Played This Trick (52-dim sequential)
├── All Cards Played This Round (52-dim binary)
├── Player Bids (8-dim, padded -1 for absent players)
├── Player Tricks Won (8-dim, padded)
├── My Bid (scalar, -1 if not yet bid)
├── My Tricks Won (scalar)
├── Round Metadata
│   ├── Total cards this round
│   ├── Current trick number
│   ├── My position (relative to dealer)
│   └── Number of active players
├── Bidding Constraint (boolean: am I the last bidder?)
└── Belief State (distribution over opponent possible hands)
```

**Key Innovation**: Track **information sets** - when a player doesn't follow suit, we deduce they don't have that suit and update beliefs.

#### 2. Neural Network Architecture

**Lightweight Transformer** (~5-8M parameters for laptop inference):

```
Input Encoder (512-dim)
    ↓
Positional Encoding (handle variable game state)
    ↓
4-6 Transformer Layers (multi-head attention)
    ↓
    ├── Policy Head: P(action | state) with legal action masking
    │   ├── Bidding phase: Probability over valid bids [0, cards_dealt]
    │   └── Playing phase: Probability over cards in hand
    │
    └── Value Head: Expected final score (normalized [-1, 1])
```

**Design choices**:
- **Transformer** over CNN: Better at modeling card relationships and variable-length states
- **Small size**: Fast inference on Intel laptop CPU/iGPU (~50-100ms per evaluation)
- **Dual-phase policy**: Network learns both bidding and card-play strategies
- **Legal action masking**: Ensures network only considers valid moves

#### 3. Monte Carlo Tree Search (MCTS)

MCTS provides lookahead planning using the neural network as a heuristic:

```
For each decision point:
1. Sample 3-5 determinizations (possible opponent hand distributions)
2. For each determinization:
   - Run 200-400 MCTS simulations (training)
   - Run 50-100 simulations (inference on laptop)
3. Aggregate visit counts → action probabilities
4. Select action (sample during training, argmax during play)
```

**Tree Reuse**: After opponent moves, navigate to that child node and make it the new root, keeping the entire explored subtree.

#### 4. Determinization for Imperfect Information

Since we don't know opponent cards:

1. **Belief tracking**: Maintain probability distribution over possible opponent hands
2. **Information set updates**: When player doesn't follow suit → eliminate that suit from their possible cards
3. **Sampling**: Generate random opponent hands consistent with:
   - Cards they've played
   - Suits they've shown/not shown
   - Number of cards they should have
4. **Average across samples**: Run MCTS on each, aggregate results

#### 5. Self-Play Training Loop (AlphaZero)

```
Loop until convergence:
    ├── Self-Play: Generate 10,000+ games using latest model + MCTS
    │   ├── Run 16-32 parallel games (GPU-accelerated)
    │   ├── Store (state, MCTS_policy, final_score) tuples
    │   └── Use exploration noise for diversity
    │
    ├── Training: Update neural network
    │   ├── Sample batches from replay buffer (last 500k positions)
    │   ├── Loss = policy_loss + value_loss + L2_regularization
    │   └── Train for N epochs on GPU
    │
    ├── Evaluation: Test new model vs. previous best
    │   ├── Play 400 games (no exploration noise)
    │   ├── Calculate ELO ratings
    │   └── If new model wins >55%, promote to "best"
    │
    └── Checkpoint: Save model every iteration
```

**Expected training time**: 3-7 days on RTX GPU (continuous self-play)

#### 6. ELO Tracking

- Maintain a pool of historical model checkpoints
- Periodically run round-robin tournaments
- Calculate ELO ratings (start at 1000)
- Expected progression:
  - Day 1: ~800 (random play)
  - Day 3: ~1200 (learned basic rules)
  - Day 7: ~1600+ (strong strategic play)

## Tech Stack

### Training Environment (Desktop PC with RTX 4060 8GB GPU and 16-core Ryzen 7950X CPU, 128GB DDR5)

| Component | Technology | Justification |
|-----------|------------|---------------|
| **ML Framework** | PyTorch 2.x (CUDA) | Best flexibility for custom RL, excellent GPU support |
| **Game Engine** | Python 3.11+ | Clean implementation of game rules, easy testing |
| **MCTS** | Custom Python | ~200 lines, full control over determinization |
| **Parallelization** | Python multiprocessing or Ray | 16-32 parallel self-play games |
| **Metrics/Logging** | Weights & Biases or TensorBoard | Track ELO, loss curves, hyperparameters |
| **Model Export** | ONNX | Cross-platform inference |

### Production Environment (Intel Laptop - CPU/iGPU - Intel i5-1135G7, 16 GB shared)

| Component | Technology | Justification |
|-----------|------------|---------------|
| **Backend** | Bun (TypeScript) | Fast, single runtime, built-in SQLite, WebSocket support |
| **Frontend** | Svelte + TypeScript | Lightweight, fast compilation, simple reactivity |
| **Database** | SQLite | File-based (no browser storage issues), handles game history |
| **Inference** | ONNX Runtime (OpenVINO) | Intel iGPU acceleration, 50-100ms latency |
| **Communication** | WebSockets | Real-time game state updates |

### Why This Stack?

1. **Separation of concerns**: Heavy training (Python/GPU) separate from lightweight inference (Bun/ONNX)
2. **No Python in production**: Faster startup, easier deployment, better laptop performance
3. **File-based storage**: SQLite in repo directory (your Tauri motivation without Tauri complexity)
4. **Modern & maintainable**: Svelte + Bun are fast to develop and debug
5. **GPU → CPU pipeline**: Train on powerful hardware, run anywhere via ONNX

## Project Structure

```
blobmaster/
├── README.md                  # This file
├── frontend/                  # Svelte UI
│   ├── src/
│   │   ├── lib/
│   │   │   ├── components/    # GameBoard, Hand, BidSelector, etc.
│   │   │   └── stores/        # Game state management
│   │   ├── routes/            # SvelteKit pages
│   │   └── app.html
│   ├── package.json
│   └── svelte.config.js
│
├── backend/                   # Bun API server
│   ├── src/
│   │   ├── api/               # REST/WebSocket endpoints
│   │   ├── inference/         # ONNX model loading & inference
│   │   ├── game/              # TypeScript game state management
│   │   └── db/                # SQLite queries
│   ├── package.json
│   └── tsconfig.json
│
├── ml/                        # Python training code
│   ├── game/
│   │   ├── blob.py            # Core game rules
│   │   ├── state.py           # State representation
│   │   └── actions.py         # Action space (bid/play)
│   │
│   ├── mcts/
│   │   ├── node.py            # MCTS tree node
│   │   ├── search.py          # MCTS algorithm
│   │   └── determinization.py # Belief state sampling
│   │
│   ├── network/
│   │   ├── model.py           # Transformer architecture
│   │   ├── encode.py          # State → tensor encoding
│   │   └── train.py           # Training loop
│   │
│   ├── training/
│   │   ├── selfplay.py        # Parallel game generation
│   │   ├── replay_buffer.py   # Experience storage
│   │   └── trainer.py         # Main training orchestration
│   │
│   ├── evaluation/
│   │   ├── arena.py           # Model vs. model tournaments
│   │   ├── elo.py             # ELO calculation
│   │   └── metrics.py         # Logging & visualization
│   │
│   ├── requirements.txt
│   └── train.py               # Entry point for training
│
├── models/                    # Saved model checkpoints
│   ├── checkpoints/           # .pth files (PyTorch)
│   ├── best_model.onnx        # Production model
│   └── elo_history.json       # ELO progression data
│
├── data/
│   ├── games.db               # SQLite: game history, user stats
│   └── training_data/         # Self-play game records
│
└── docs/
    ├── training_log.md        # Training progress notes
    └── architecture.md        # Detailed technical design
```

## Implementation Roadmap

### Phase 1: Core Game Engine (Weekend 1 - Days 1-2)

**Goal**: Bulletproof game logic that handles all edge cases

- [ ] Implement `game/blob.py`:
  - Deck management, shuffling, dealing
  - Bidding phase with last-player constraint
  - Trick-taking logic (follow suit, trump, winner determination)
  - Scoring (exact trick calculation)
  - Support for 3-8 players, variable cards per round

- [ ] Write comprehensive unit tests:
  - Test all game rules independently
  - Edge cases: 0-bid, last player constraint, suit elimination
  - Multi-player scenarios (3, 4, 5, 8 players)

- [ ] Build CLI version:
  - Play manually against random bots
  - Validate rules feel correct
  - Debug any logic issues

**Deliverable**: `python ml/game/blob.py` runs a playable game

### Phase 2: MCTS + Neural Network (Midweek - Days 3-5)

**Goal**: Basic AI that can play legal moves and improve with training

- [ ] Implement state encoding (`network/encode.py`):
  - Convert game state → 512-dim tensor
  - Handle variable player counts with masking
  - Belief state representation

- [ ] Build neural network (`network/model.py`):
  - Transformer architecture (4-6 layers)
  - Policy head (bidding + card play)
  - Value head (score prediction)
  - Legal action masking

- [ ] Implement MCTS (`mcts/search.py`):
  - Basic MCTS with UCB1 selection
  - Integration with neural network for leaf evaluation
  - Action sampling based on visit counts

- [ ] Test integration:
  - Random network should make legal moves
  - MCTS improves move quality over random
  - Inference speed: <200ms per move (CPU)

**Deliverable**: AI plays valid games end-to-end

### Phase 3: Imperfect Information Handling (Weekend 2 - Days 6-7)

**Goal**: Handle hidden opponent cards via determinization

- [ ] Implement belief tracking (`mcts/determinization.py`):
  - Track which cards each player could have
  - Update beliefs when players reveal suit information
  - Sample consistent opponent hands

- [ ] MCTS with determinization:
  - Run MCTS on 3-5 sampled worlds
  - Aggregate visit counts across samples
  - Handle perfect information (known cards) vs. imperfect

- [ ] Validate on known scenarios:
  - Test with fully-revealed hands (should match perfect info MCTS)
  - Test suit elimination logic
  - Verify sampling produces valid hands

**Deliverable**: AI handles imperfect information correctly

### Phase 4: Self-Play Training Pipeline (Week 2 - Days 8-14)

**Goal**: Automated training loop generating strong models

- [ ] Build self-play engine (`training/selfplay.py`):
  - Parallel game generation (16-32 workers)
  - MCTS with exploration noise
  - Store (state, policy, value) tuples

- [ ] Implement replay buffer (`training/replay_buffer.py`):
  - Circular buffer (500k positions)
  - Efficient sampling for training batches
  - Data augmentation (if applicable)

- [ ] Training loop (`training/trainer.py`):
  - Load batch from replay buffer
  - Compute loss (policy + value)
  - Update network with Adam optimizer
  - Learning rate scheduling

- [ ] Evaluation pipeline (`evaluation/arena.py`):
  - Model vs. model tournaments
  - ELO calculation and tracking
  - Automated checkpoint promotion

- [ ] Monitoring:
  - W&B or TensorBoard logging
  - Loss curves, ELO progression
  - Move quality metrics

- [ ] **Run training**: 3-7 days on RTX GPU

**Deliverable**: Trained model with ELO progression data

### Phase 5: ONNX Export & Inference (Day 15)

**Goal**: Fast inference on Intel laptop

- [ ] Export model to ONNX:
  - Convert PyTorch model → ONNX format
  - Validate outputs match PyTorch
  - Optimize for inference (operator fusion)

- [ ] Test ONNX Runtime:
  - CPU inference speed
  - OpenVINO (Intel iGPU) acceleration
  - Benchmark: should be <100ms per evaluation

**Deliverable**: `best_model.onnx` runs on laptop

### Phase 6: Backend API (Weekend 3 - Days 16-17)

**Goal**: Bun server that loads model and serves predictions

- [ ] Setup Bun project (`backend/`):
  - Initialize with TypeScript
  - Install ONNX Runtime Node.js bindings
  - Setup SQLite database

- [ ] Implement game logic in TypeScript:
  - Port essential game rules from Python
  - Game state management
  - Move validation

- [ ] Build inference endpoint:
  - Load ONNX model at startup
  - API: `POST /predict` → { state } → { suggested_bid | suggested_card, confidence }
  - Handle MCTS on server-side (or return NN policy directly)

- [ ] WebSocket server:
  - Real-time game state updates
  - Multiplayer support (human + AI players)

- [ ] Database schema:
  - Games table (history)
  - Users table (stats, ELO vs. AI)
  - Moves table (for analysis)

**Deliverable**: `bun run backend/src/index.ts` starts API server

### Phase 7: Frontend UI (Days 18-21)

**Goal**: Playable web interface

- [ ] Setup Svelte project (`frontend/`):
  - SvelteKit with TypeScript
  - WebSocket client
  - State management stores

- [ ] Build components:
  - `GameBoard.svelte`: Full game view
  - `Hand.svelte`: Your cards, clickable to play
  - `BidSelector.svelte`: Choose bid with constraint warning
  - `TrickHistory.svelte`: Show completed tricks
  - `ScoreBoard.svelte`: Live scores, ELO
  - `AIThinking.svelte`: Show what AI is considering (MCTS tree viz?)

- [ ] Game flow:
  - Start new game (choose # players, AI players)
  - Bidding phase → Playing phase → Scoring → Next round
  - AI moves with animations
  - Highlight legal moves

- [ ] Polish:
  - Card graphics (or use Unicode card symbols)
  - Smooth animations
  - Responsive design

**Deliverable**: Full playable game in browser

## Training Configuration

### Hyperparameters (Starting Point)

```python
# Network
EMBEDDING_DIM = 512
NUM_TRANSFORMER_LAYERS = 6
NUM_HEADS = 8
DROPOUT = 0.1

# MCTS
MCTS_SIMULATIONS_TRAINING = 300
MCTS_SIMULATIONS_INFERENCE = 100
CPUCT = 1.5  # Exploration constant
DETERMINIZATIONS = 5

# Training
BATCH_SIZE = 512
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
REPLAY_BUFFER_SIZE = 500_000
SELF_PLAY_GAMES_PER_ITERATION = 10_000
TRAINING_EPOCHS_PER_ITERATION = 10

# Evaluation
EVAL_GAMES = 400
ELO_PROMOTION_THRESHOLD = 0.55  # 55% win rate to become new best
```

### Hardware Requirements

**Training**:
- **GPU**: NVIDIA RTX 4060 8GB
- **RAM**: 128GB DDR5 (for parallel self-play workers)
- **Storage**: 50GB+ (model checkpoints, replay buffer)
- **Time**: 3-7 days continuous training

**Inference**:
- **CPU**: Intel i5-1135G7 with iGPU
- **RAM**: 16GB shared
- **Storage**: 100MB (model + frontend assets)
- **Latency**: <100ms per move evaluation

### Expected ELO Progression

Based on similar projects (AlphaZero chess/Go, poker bots):

```
Iteration  |  ELO   |  Capability
-----------|--------|------------------------------------------
0          |  800   | Random legal moves
10         |  1000  | Basic trick-taking (follow suit)
50         |  1200  | Learned bidding/scoring relationship
100        |  1400  | Strategic bidding, card counting
200        |  1600  | Advanced play (suit elimination, bluffing)
500+       | 1800+  | Superhuman (optimal against imperfect players)
```

## Research Questions

Throughout this project, you'll explore:

1. **Strategy convergence**: Do all trained models converge to the same optimal strategy, or do initial conditions create different "styles"?

2. **Imperfect player exploitation**: How well does the AI adapt to suboptimal human play? (This is what you're most curious about!)

3. **Position value**: Is last bidder position advantageous (information) or disadvantageous (constraint)?

4. **Risk/reward bidding**: Does AI learn conservative (safe bids) or aggressive (maximize points) strategies?

5. **Belief state accuracy**: How quickly does the AI deduce opponent hands from suit information?

6. **Transfer learning**: Can a model trained on 4-player games adapt to 6-player games?

## Future Enhancements

After the initial version is working:

- **Explainability**: Visualize MCTS tree, show "why" AI made a bid/play
- **Difficulty levels**: Reduce MCTS simulations or add noise for easier opponents
- **Online play**: Multiplayer over internet
- **Mobile app**: React Native or Flutter version
- **Variants**: Support different Blob rule sets
- **Meta-learning**: Train on multiple simultaneous games (different player counts)
- **Human coaching**: "Tutor mode" that suggests moves and explains reasoning

## Getting Started

### Prerequisites

```bash
# Python environment (training)
python --version  # 3.11+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r ml/requirements.txt

# Bun (backend/frontend)
curl -fsSL https://bun.sh/install | bash
bun --version  # 1.0+
```

### Quick Start

```bash
# 1. Clone and setup
git clone <your-repo>
cd blobmaster

# 2. Test game engine
cd ml
python -m pytest game/test_blob.py
python game/blob.py  # Play CLI version

# 3. Train model (start this and let it run)
python train.py --iterations 500 --gpu 0

# 4. Monitor training (in another terminal)
tensorboard --logdir=runs/

# 5. After training, export model
python export_onnx.py --checkpoint models/checkpoints/best.pth

# 6. Start backend
cd ../backend
bun install
bun run dev  # Starts on http://localhost:3000

# 7. Start frontend
cd ../frontend
bun install
bun run dev  # Starts on http://localhost:5173
```

## Learning Resources

To understand the techniques used:

1. **AlphaZero Paper**: [Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815)
2. **Pluribus (Poker AI)**: [Superhuman AI for multiplayer poker](https://science.sciencemag.org/content/365/6456/885)
3. **MCTS Survey**: [A Survey of Monte Carlo Tree Search Methods](https://ieeexplore.ieee.org/document/6145622)
4. **Imperfect Information Games**: [Bayesian Opponent Modeling](https://papers.nips.cc/paper/2013/file/e2230b853516e7b05d79744fbd4c9c13-Paper.pdf)
5. **Transformers**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## License

MIT License - feel free to learn from and extend this project!

## Acknowledgments

- AlphaZero team at DeepMind for revolutionizing game AI
- Pluribus team at Facebook AI for imperfect information techniques
- The card game community for keeping Blob/Oh Hell alive!
