# BlobMaster: Holiday Challenge Project

**Build an AI that masters the card game "Blob" using AlphaZero-style reinforcement learning**

A 3-4 week challenge project to implement neural MCTS, self-play training, and a web interface for an imperfect information trick-taking card game.

---

## Overview

**Objective**: Train a neural network from scratch to play Blob competitively through self-play, then deploy it as a web app where humans can challenge the AI.

**Approach**: AlphaZero-style reinforcement learning - no human game data, the AI learns purely by playing millions of games against itself.

**Hardware Required**:
- Desktop/workstation with NVIDIA GPU (8GB+ VRAM recommended)
- 16GB+ RAM
- 50GB+ storage for training data and models
- Linux OS (Ubuntu 22.04 LTS recommended)

**Timeline**: 3-4 weeks part-time (20-30 hours/week)

**Difficulty**: Advanced (requires ML, functional programming, and game AI knowledge)

---

## What is Blob?

Blob is a trick-taking bidding card game (similar to Oh Hell, Wizard, or Spades).

### Core Mechanics

**Players**: 3-8 players (4 is standard)

**Deck**: Standard 52-card deck (13 ranks × 4 suits)

**Game Flow**:
1. **Deal**: Each player receives N cards (typically 5-13)
2. **Trump Selection**: Random suit becomes trump for the round
3. **Bidding Phase**: Players sequentially bid on exact number of tricks they'll win
   - **Dealer constraint**: Last bidder cannot bid such that total bids = cards dealt (prevents guaranteed outcomes)
4. **Playing Phase**: Standard trick-taking
   - Must follow suit if possible
   - Highest trump wins, otherwise highest card of led suit wins
   - Trick winner leads next trick
5. **Scoring**: All-or-nothing
   - Made bid: Score = 10 + bid
   - Missed bid: Score = 0

### Why Blob is Interesting for AI

- **Imperfect information**: Can't see opponent cards (requires belief tracking)
- **Strategic tension**: Dealer bidding constraint creates complex game theory
- **Risk/reward tradeoff**: Conservative vs aggressive bidding
- **Long-term planning**: Card counting, suit deduction across multiple tricks
- **Large action space**: 52 possible cards + 14 possible bids

---

## Project Goals

### What You'll Build

1. **Game Engine**: Pure functional implementation in JAX
2. **Neural Network**: Transformer architecture with policy + value heads
3. **MCTS Search**: Monte Carlo Tree Search with Gumbel sampling for imperfect information
4. **Training Pipeline**: Self-play workers, replay buffer, training loop, evaluation arena
5. **Web Interface**: Svelte frontend + Bun backend for human vs AI gameplay

### What You'll Learn

- AlphaZero algorithm and neural MCTS
- Handling imperfect information in game AI (determinization, belief tracking)
- JAX functional programming and JIT compilation
- Large-scale self-play training orchestration
- Deploying ML models for real-time inference

### Success Criteria

- AI plays legal moves 100% of the time
- AI beats random baseline by >90% win rate
- AI demonstrates strategic bidding (not always bidding 0 or max)
- Training pipeline generates 100+ games/minute
- Web interface allows smooth human gameplay

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **OS** | Ubuntu 22.04 LTS | Stable CUDA support |
| **Language** | Python 3.10 | ML ecosystem |
| **ML Framework** | JAX + Flax | JIT compilation, GPU acceleration |
| **MCTS Library** | DeepMind mctx | Production-ready MCTS |
| **Optimizer** | Optax | JAX-native gradient descent |
| **Game Engine** | Pure JAX (functional) | JIT-compatible, vectorizable |
| **Backend** | Bun + TypeScript | Fast HTTP server |
| **Frontend** | Svelte + SvelteKit | Reactive UI |
| **Database** | SQLite | Game history, stats |

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  TRAINING PIPELINE                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Self-Play Workers (Parallel)                      │
│  ├── JAX Game Engine                               │
│  ├── Neural Network (Flax Transformer)             │
│  ├── MCTS (mctx.gumbel_muzero_policy)              │
│  └── Determinization (vmap for imperfect info)     │
│                    ↓                                │
│  Replay Buffer (NumPy arrays)                      │
│                    ↓                                │
│  Network Trainer (Optax)                           │
│  ├── Sample batches from replay buffer             │
│  ├── Policy loss: CrossEntropy(MCTS → Network)     │
│  ├── Value loss: MSE(GameOutcome → Network)        │
│  └── Update network weights                        │
│                    ↓                                │
│  Evaluation Arena                                  │
│  ├── New model vs Best model tournament            │
│  ├── ELO rating calculation                        │
│  └── Promote if new model wins >55%                │
│                                                     │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                  WEB APPLICATION                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Svelte Frontend                                   │
│  ├── Game board visualization                      │
│  ├── Hand display, bid selector                    │
│  ├── Trick history, score tracking                 │
│  └── WebSocket connection to backend               │
│                    ↓                                │
│  Bun Backend (TypeScript)                          │
│  ├── REST API (game management)                    │
│  ├── WebSocket (real-time moves)                   │
│  ├── Call JAX inference server                     │
│  └── SQLite for persistence                        │
│                    ↓                                │
│  JAX Inference Server                              │
│  ├── Load trained model                            │
│  ├── Run MCTS (50 simulations)                     │
│  └── Return action + policy distribution           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
blobmaster/
├── game/
│   ├── state.py           # GameState dataclass (pure JAX)
│   ├── rules.py           # Step functions, legal moves, scoring
│   ├── encoder.py         # GameState → neural network input
│   ├── belief.py          # Belief tracking for imperfect info
│   └── test_game.py       # Game logic tests
│
├── network/
│   ├── model.py           # Flax Transformer (policy + value heads)
│   ├── architecture.py    # Network hyperparameters
│   └── test_network.py    # Network tests
│
├── mcts/
│   ├── search.py          # mctx wrapper, RecurrentFn, RootFn
│   ├── determinization.py # Sample opponent hands (vmap)
│   ├── gumbel.py          # Gumbel sampling configuration
│   └── test_mcts.py       # MCTS tests
│
├── training/
│   ├── selfplay.py        # Self-play worker (generates games)
│   ├── trainer.py         # Network training loop
│   ├── replay.py          # Replay buffer
│   ├── evaluator.py       # Model evaluation, ELO tracking
│   ├── pipeline.py        # Orchestration (main training loop)
│   └── config.py          # Hyperparameters
│
├── inference/
│   ├── server.py          # JAX model serving (gRPC or HTTP)
│   └── client.py          # Client library for backend
│
├── backend/
│   ├── src/
│   │   ├── game/          # TypeScript game state (for validation)
│   │   ├── api/           # REST endpoints
│   │   ├── websocket/     # Real-time game updates
│   │   └── db/            # SQLite queries
│   └── package.json
│
├── frontend/
│   ├── src/
│   │   ├── lib/
│   │   │   ├── components/  # GameBoard, Hand, BidSelector
│   │   │   └── stores/      # Game state management
│   │   └── routes/          # SvelteKit pages
│   └── package.json
│
├── scripts/
│   ├── train.py           # Training entry point
│   ├── evaluate.py        # Play games between models
│   └── export.py          # Model checkpointing
│
├── tests/
│   └── integration/       # End-to-end tests
│
├── data/
│   ├── checkpoints/       # Saved models (.pkl JAX params)
│   ├── training_data/     # Self-play game records
│   └── games.db           # SQLite database
│
└── docs/
    ├── game-rules.md      # Full Blob rules
    ├── architecture.md    # Technical design doc
    └── training-guide.md  # How to run training
```

---

## Implementation Timeline

### Week 1: Game Engine + Network

**Goal**: Implement game logic and neural network in pure JAX

#### Days 1-2: JAX Game State
- Define `GameState` dataclass with static-shape arrays
  - `hands: (8, 52) bool` - player hands (padded to 8 players)
  - `bids: (8,) int` - bids (-1 = not bid yet)
  - `tricks_won: (8,) int` - tricks won this round
  - `current_trick: (8,) int` - card indices in current trick (-1 = not played)
  - `phase: int` - 0=bidding, 1=playing, 2=done
  - `trump_suit: int` - 0-3 (clubs, diamonds, hearts, spades)
  - `dealer: int` - dealer position
  - `current_player: int` - whose turn
- Implement `deal_cards(rng_key, num_players, cards_per_player) -> GameState`
- JIT-compile all functions, verify speed

#### Days 3-4: Game Rules
- Legal move generation
  - Bidding: `get_legal_bids(state) -> (14,) bool mask`
  - Dealer constraint: Cannot bid X if sum(bids) + X == cards_dealt
  - Playing: `get_legal_cards(state, player) -> (52,) bool mask`
  - Must follow suit if possible
- State transitions
  - `step_bid(state, player, bid) -> new_state`
  - `step_play(state, player, card_idx) -> new_state`
  - Handle trick completion, determine winner, advance to next trick
- Scoring
  - `compute_rewards(state) -> (8,) float`
  - All-or-nothing: 10 + bid if exact, else 0
- Terminal state detection
- Write 100+ property-based tests (Hypothesis)

#### Days 5-6: Neural Network
- Flax Transformer architecture (~5M parameters)
  - Input: 256-dim state embedding
  - 6 transformer layers, 8 attention heads
  - Dual heads: Policy (52-dim logits) + Value (1-dim score)
- State encoder: `encode(state, player) -> (256,) embedding`
  - Player's hand (52-dim one-hot)
  - Bids made so far (8-dim)
  - Tricks won (8-dim)
  - Current trick cards (8×4 grid)
  - Trump suit (4-dim one-hot)
  - Dealer position (8-dim one-hot)
  - Game phase (3-dim one-hot)
  - Mask for imperfect info (hide opponent cards)
- Forward pass: `network.apply(params, state_embedding) -> (policy_logits, value)`
- Initialize with Xavier/He initialization
- Test: Verify shapes, check gradients

#### Day 7: Buffer Day
- Catch up on any Week 1 slippage
- Write integration tests
- Verify JAX JIT compilation works for all components

---

### Week 2: MCTS + Determinization

**Goal**: Implement neural MCTS with imperfect information handling

#### Days 8-9: mctx Integration
- Install `dm-mctx` library
- Implement `RecurrentFn` for tree expansion
  - Takes: `(params, rng, action, embedding)` where embedding = GameState
  - Calls: `step_play(state, action)` or `step_bid(state, action)`
  - Evaluates new state with network
  - Returns: `RecurrentFnOutput(reward, discount, prior_logits, value)`
- Implement `RootFn`
  - Evaluate initial state with network
  - Apply legal action mask (set illegal actions to -inf)
  - Return: `RootFnOutput(prior_logits, value, embedding=state)`
- Wrap `mctx.gumbel_muzero_policy`
  - Set `num_simulations=50` (will tune later)
  - Set `max_num_considered_actions=16` (Gumbel pruning)
  - Use `qtransform_completed_by_mix_value` (DeepMind default)
- Test: Run MCTS on random game states, verify legal actions only

#### Days 10-11: Belief Tracking & Determinization
- Implement belief state tracker
  - Track probability distribution over opponent cards
  - Update when information revealed (e.g., player doesn't follow suit → doesn't have that suit)
  - Store as `(num_opponents, 52) float` belief matrix
- Determinization sampling
  - `sample_determinizations(state, belief, num_samples) -> (B, GameState)`
  - Sample consistent opponent hands given belief state
  - Use constrained random permutation (no rejection loops)
  - Vectorize with `jax.vmap`
- Aggregate MCTS across determinizations
  - Run MCTS on each determinized world: `vmap(mctx.gumbel_muzero_policy)`
  - Average action visit counts across worlds
  - Return aggregated policy
- Test: Verify determinized games are consistent with observations

#### Days 12-13: Self-Play Worker
- Implement game generation loop
  - Initialize game
  - For each move:
    - Run MCTS (with determinization if imperfect info)
    - Sample action from policy (temperature-based exploration)
    - Store training example: `(state, mcts_policy, value=None)`
    - Apply action to game
  - At game end: Back-propagate final scores to all examples
- Temperature schedule
  - High temperature (T=1.5) early in training (exploration)
  - Anneal to low temperature (T=0.1) later (exploitation)
  - Position-based: Higher temp for first 10 moves of game
- Training example format
  - State: Encoded state vector (256-dim)
  - Policy: MCTS visit count distribution (52-dim, normalized)
  - Value: Final score (normalized to [-1, 1])
- Test: Generate 10 full games, verify format

#### Day 14: Buffer Day
- Integration testing: End-to-end self-play
- Profile performance bottlenecks
- Optimize JIT compilation (cache compiled functions)

---

### Week 3: Training Pipeline

**Goal**: Full self-play training loop with evaluation

#### Days 15-16: Replay Buffer
- Implement circular buffer
  - Fixed capacity: 500,000 training examples
  - Store as NumPy arrays (not JAX - easier to serialize)
  - FIFO eviction: Drop oldest examples when full
- Batch sampling
  - Sample random batches for training
  - Batch size: 512-1024
  - Ensure diversity (don't sample all from one game)
- Data augmentation (optional)
  - Rotate suits (4 symmetries)
  - Permute player positions (if applicable)

#### Days 17-18: Training Loop
- Implement Optax training step
  - Loss function: `policy_loss + value_loss`
  - Policy loss: Cross-entropy between MCTS policy and network policy
  - Value loss: MSE between game outcome and network value prediction
  - Optimizer: Adam with learning rate 0.001, decay schedule
- Training iteration
  - Sample batch from replay buffer
  - Compute gradients with `jax.value_and_grad`
  - Apply gradients with Optax
  - Log metrics: Loss, policy accuracy, value MSE
- Checkpointing
  - Save network params every N iterations
  - Save optimizer state (for resume)
  - Save ELO history

#### Days 19-20: Evaluation Arena
- Model tournament system
  - Pit new model vs current best model
  - Play 400 games (100 games per starting position in 4-player)
  - Calculate win rate
- ELO tracking
  - Update ELO ratings after each tournament
  - Track progression over iterations
  - Plot ELO curve
- Promotion logic
  - If new model wins >55% → promote to "best model"
  - Save new best checkpoint
  - Use new best for next self-play iteration
- Test: Run mini-tournament with random policies

#### Day 21: Full Pipeline Integration
- Training orchestration script
  - Self-play phase: Generate 10,000 games (16 parallel workers)
  - Training phase: Train network on replay buffer (1000 batches)
  - Evaluation phase: Tournament new vs best model
  - Repeat for 500 iterations
- Configuration system
  - Hyperparameters in JSON/YAML
  - Easy to modify: Learning rate, MCTS sims, batch size, etc.
- Logging and monitoring
  - TensorBoard integration (optional)
  - Console progress bars
  - Slack/email alerts on milestones (optional)
- Test: Run 1 full iteration end-to-end

---

### Week 4: Web Interface + Deployment

**Goal**: Deploy trained model in playable web app

#### Days 22-23: Inference Server
- JAX model serving
  - Load trained checkpoint (best model params)
  - Expose inference endpoint: `predict(game_state) -> (action, policy_distribution)`
  - Use fewer MCTS simulations (50 vs 200 for training)
  - HTTP server (Flask/FastAPI) or gRPC
- Optimize for latency
  - Pre-compile JIT functions on startup
  - Batch inference if multiple requests
  - Target: <100ms per move
- Test: Send 100 random game states, verify responses

#### Days 24-25: Backend (Bun + TypeScript)
- Game management API
  - `POST /game/create` - Start new game (human vs AI)
  - `POST /game/:id/move` - Submit human move
  - `GET /game/:id/state` - Get current game state
  - `DELETE /game/:id` - Abandon game
- WebSocket for real-time updates
  - Subscribe to game updates
  - Push AI moves to client
  - Push opponent moves in multiplayer (future)
- Call JAX inference server
  - When AI turn: Request action from inference server
  - Apply action to game state
  - Broadcast update to frontend
- SQLite integration
  - Store game history
  - User stats (games played, win rate vs AI)
- Test: Postman/curl API tests

#### Days 26-27: Frontend (Svelte)
- Game board component
  - Display 4 player positions
  - Show each player's bid and tricks won
  - Render current trick in center
  - Trump suit indicator
- Hand component
  - Display human player's cards
  - Highlight legal moves
  - Click to play card
- Bid selector
  - Buttons for valid bids
  - Show dealer constraint (forbidden bid grayed out)
- Trick history
  - Scrollable list of past tricks
  - Show who won each trick
- Scoreboard
  - Current scores
  - Running total across rounds
- AI thinking indicator
  - Show "AI thinking..." during MCTS
  - Optionally visualize policy distribution (card probabilities)
- WebSocket integration
  - Connect on game load
  - Listen for game state updates
  - Update UI reactively
- Test: Play 10 full games against AI

#### Day 28: Polish + Documentation
- UI polish
  - Card animations
  - Sound effects (optional)
  - Mobile responsive (optional)
- Documentation
  - README: Setup instructions, training guide
  - Game rules doc
  - API documentation
  - Code comments
- Performance tuning
  - Frontend bundle optimization
  - Backend query optimization
  - Inference server warm-up
- Demo video
  - Record gameplay session
  - Showcase AI strategic decisions

---

## Key Technical Challenges

### Challenge 1: JAX Functional Programming

**Problem**: JAX requires pure functions (no mutation, no side effects)

**Solution**:
```python
# Wrong (mutation)
state.hands[player][card_idx] = False

# Right (immutable update)
state = state.replace(
    hands=state.hands.at[player, card_idx].set(False)
)
```

Use `jax.tree_map` for nested updates, `chex.dataclass` for immutable structs.

### Challenge 2: Static Shape Constraints

**Problem**: JAX JIT requires static shapes, but Blob supports 3-8 players

**Solution**:
- Pad all arrays to max size (8 players, 52 cards)
- Use mask arrays to indicate active players: `(8,) bool`
- Example: 4-player game has mask `[T,T,T,T,F,F,F,F]`

### Challenge 3: Imperfect Information MCTS

**Problem**: Standard MCTS assumes perfect information (see all cards)

**Solution**:
- Sample multiple determinizations (possible opponent hands)
- Run MCTS on each determinized world
- Aggregate action probabilities across worlds
- Use belief tracking to make sampling realistic

### Challenge 4: Gumbel MuZero API

**Problem**: mctx library has complex API, sparse documentation

**Solution**:
- Follow DeepMind's examples: https://github.com/deepmind/mctx/tree/main/examples
- Start with simpler `muzero_policy` (AlphaZero), upgrade to `gumbel_muzero_policy` later
- Use `chex.assert_shape` liberally to catch shape errors

### Challenge 5: Training Stability

**Problem**: RL training can diverge (loss explodes, agent forgets skills)

**Solution**:
- Clip gradients (max norm = 1.0)
- Use large replay buffer (500k examples)
- Conservative learning rate (0.001 with decay)
- Promote new model only if clearly better (>55% win rate)
- Monitor value loss - if increasing, learning rate too high

---

## Hyperparameters

### Network Architecture
- State embedding: 256 dimensions
- Transformer layers: 6
- Attention heads: 8
- Feedforward dimension: 1024
- Dropout: 0.1
- Total parameters: ~5M

### MCTS Configuration
- Simulations (training): 50-200
- Simulations (inference): 30-50
- Gumbel max actions: 16
- Determinizations: 3-5
- Temperature (early): 1.5
- Temperature (late): 0.1

### Training
- Games per iteration: 10,000
- Batch size: 512
- Learning rate: 0.001 (cosine decay)
- Replay buffer: 500,000
- Training iterations: 500
- Parallel workers: 16

### Evaluation
- Tournament games: 400
- Promotion threshold: 55% win rate
- ELO K-factor: 32

---

## Expected Performance

### Training Throughput
- Self-play: 100-200 games/min (GPU-accelerated)
- Training: 1000 batches/min
- Evaluation: 50 games/min
- Total iteration time: ~1-2 hours

### Training Timeline
- Iterations needed: 500
- Total time: ~3-7 days continuous
- Training cost: ~500 GPU-hours on RTX 4060

### Model Strength Progression
- Day 1: Random legal moves (ELO ~800)
- Day 3: Basic trick-taking (ELO ~1200)
- Day 7: Strategic bidding (ELO ~1600)
- Final: Strong play, beats human beginners

### Inference Performance
- Latency: 50-100ms per move
- Throughput: 100+ moves/sec (batched)
- Memory: <6GB VRAM

---

## Learning Resources

### AlphaZero / MCTS
- AlphaZero paper: https://arxiv.org/abs/1712.01815
- Gumbel MuZero: https://openreview.net/forum?id=bERaNdoegnO
- MCTS survey: https://ieeexplore.ieee.org/document/6145622

### JAX
- JAX quickstart: https://jax.readthedocs.io/en/latest/quickstart.html
- JAX for deep learning: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/
- mctx library: https://github.com/deepmind/mctx

### Imperfect Information Games
- Information Set MCTS: https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.221.5658
- Belief state tracking: https://papers.nips.cc/paper/2019/hash/7d31ad5b9f8fcbc96a61f90ee98e9e93-Abstract.html

### Svelte / Bun
- Svelte tutorial: https://svelte.dev/tutorial
- Bun docs: https://bun.sh/docs

---

## Milestones & Checkpoints

### Week 1 Complete
- [ ] JAX game engine plays full 4-player games correctly
- [ ] Neural network forward pass produces valid shapes
- [ ] 100+ game logic tests pass
- [ ] JIT compilation works for all components

### Week 2 Complete
- [ ] MCTS returns legal actions only
- [ ] Determinization sampling produces valid games
- [ ] Self-play worker generates complete games
- [ ] Training examples in correct format

### Week 3 Complete
- [ ] Full training iteration completes (self-play → train → evaluate)
- [ ] Model improves over random baseline
- [ ] ELO tracking functional
- [ ] Checkpointing works

### Week 4 Complete
- [ ] Web app deployed (local or cloud)
- [ ] Human can play full games vs AI
- [ ] AI response time <500ms
- [ ] Game state persists in database

### Project Complete
- [ ] Model trained to strong play (ELO >1500)
- [ ] AI demonstrates strategic bidding
- [ ] Web interface polished and bug-free
- [ ] Documentation complete
- [ ] GitHub repo published (optional)

---

## Extensions & Future Work

Once core project complete, consider:

### Research Directions
- Ablation studies: Transformer vs MLP, Gumbel vs standard MCTS
- Transfer learning: Train on 4-player, test on 6-player
- Opponent modeling: Exploit suboptimal human play
- Meta-learning: Fast adaptation to rule variants

### Engineering Improvements
- Distributed training (Ray, multi-machine)
- Model compression (quantization, pruning)
- Mobile deployment (TFLite, ONNX)
- Multiplayer support (4 humans online)

### Game Variants
- Implement other trick-taking games (Spades, Hearts, Euchre)
- Share learned representations across games
- Benchmark learning speed differences

---

## Success Metrics

### Minimum Viable Product
- AI plays legal moves 100% of the time
- AI beats random baseline >90% win rate
- Web app allows humans to play vs AI
- Training pipeline runs end-to-end

### Stretch Goals
- AI achieves ELO >1600 (beats casual human players)
- Training completes in <5 days
- Inference latency <100ms
- Web app deployed publicly (Vercel, Railway)

### Showpiece Achievements
- Publish demo video on Twitter/HN
- Open-source repo with clear docs
- Write blog post on lessons learned
- Submit to AI game competition (if available)

---

## Getting Started

### Day 0: Setup

1. **Provision hardware**
   - Linux machine with NVIDIA GPU (RTX 3060+)
   - Install Ubuntu 22.04 LTS
   - Install CUDA 12.x drivers

2. **Install dependencies**
   ```bash
   # System packages
   sudo apt update
   sudo apt install python3.10 python3-pip git

   # Python packages
   pip install "jax[cuda12]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   pip install flax==0.8.1 dm-mctx==0.0.5 optax==0.1.8 chex==0.1.85
   pip install numpy scipy matplotlib pytest hypothesis

   # Bun (backend)
   curl -fsSL https://bun.sh/install | bash

   # Verify JAX GPU
   python -c "import jax; print(jax.devices())"  # Should show GPU
   ```

3. **Create project structure**
   ```bash
   mkdir blobmaster && cd blobmaster
   git init
   # Create directories as outlined in "Project Structure" section
   ```

4. **Read game rules**
   - Understand bidding phase, dealer constraint
   - Understand playing phase, trump suit, must-follow-suit
   - Play a few hands manually to internalize

5. **Review AlphaZero paper**
   - Understand self-play → train → evaluate loop
   - Understand how MCTS generates training data
   - Note differences for imperfect information

### First Commit
- Set up repo skeleton
- Add README with project goals
- Add .gitignore (data/, checkpoints/)
- Commit: "Initial project structure"

---

**Good luck and have fun!** This is a challenging but incredibly rewarding project. You'll come out with deep knowledge of RL, game AI, and production ML systems.

When stuck, consult the JAX docs, mctx examples, and DeepMind papers. The AlphaZero algorithm is well-studied - you're following a proven path.

**Most importantly**: Build incrementally, test thoroughly, and enjoy the journey of watching your AI learn to play from scratch.
