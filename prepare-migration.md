# BlobMaster: Rust Rewrite Migration Plan

**Date**: 2026-03-14
**Source**: Python/PyTorch AlphaZero implementation (Phases 1-4)
**Target**: Clean-sheet Rust rewrite

---

## 1. What We're Porting

A complete AlphaZero-style reinforcement learning system for the card game "Blob" (trick-taking with bidding). The Python implementation is correct and well-tested (460 tests, 97% game engine coverage) but fundamentally bottlenecked by Python's per-operation overhead for MCTS tree search and game simulation.

**Core components to rewrite in Rust:**

| Component | Python Source | Lines | Tests | Purpose |
|-----------|-------------|-------|-------|---------|
| Game Engine | `blob.py`, `constants.py` | ~1,900 | 135 | Blob rules, card/deck/trick/game logic |
| State Encoder | `encode.py` | 754 | in test_network | Game state → 256-dim tensor |
| Neural Network | `model.py` | 509 | in test_network | BlobNet Transformer (dual-head) |
| MCTS Core | `node.py`, `search.py` | ~1,200 | in test_mcts | Tree search with UCB1 selection |
| Belief Tracker | `belief_tracker.py` | ~600 | in test_mcts | Opponent hand probability tracking |
| Determinization | `determinization.py` | ~700 | dedicated | Sampling consistent opponent hands |
| Self-Play | `selfplay.py` | ~1,200 | 93 | Parallel game generation for training |
| Replay Buffer | `replay_buffer.py` | ~400 | in test_training | Experience storage & sampling |
| Trainer | `trainer.py` | ~600 | in test_training | Network training loop |
| Evaluation | `arena.py`, `elo.py` | ~1,100 | dedicated | Model tournaments & ELO tracking |
| Config | `config.py` | 344 | — | Training hyperparameters |
| CLI | `train.py` | ~400 | — | Entry point, checkpointing |

---

## 2. Legacy Folder Structure

All reference files have been copied to `legacy/` organized by component:

```
legacy/
├── game-engine/           # Blob game rules (PORT FIRST)
│   ├── blob.py            # BlobGame, Card, Deck, Player, Trick classes (1,807 lines)
│   ├── constants.py       # SUITS, RANKS, RANK_VALUES, TRUMP_ROTATION, round structure
│   └── test_blob.py       # 135 tests — use as Rust test specification
│
├── neural-network/        # BlobNet architecture
│   ├── model.py           # BlobNet Transformer + BlobNetTrainer (509 lines)
│   ├── encode.py          # StateEncoder (256-dim) + ActionMasker (754 lines)
│   └── test_network.py    # Network shape/output tests
│
├── mcts/                  # Monte Carlo Tree Search
│   ├── node.py            # MCTSNode with UCB1 selection (~400 lines)
│   ├── search.py          # MCTS + ImperfectInfoMCTS classes (~800 lines)
│   ├── belief_tracker.py  # BeliefState + PlayerConstraints (~600 lines)
│   ├── determinization.py # Determinizer for sampling opponent hands (~700 lines)
│   ├── test_mcts.py       # MCTS algorithm tests (~1,000 lines)
│   └── test_determinization.py  # Determinization tests
│
├── training/              # Self-play training pipeline
│   ├── selfplay.py        # SelfPlayWorker + SelfPlayEngine (~1,200 lines)
│   ├── replay_buffer.py   # ReplayBuffer circular FIFO (~400 lines)
│   ├── trainer.py         # NetworkTrainer with mixed precision (~600 lines)
│   └── test_training.py   # 93 training pipeline tests
│
├── evaluation/            # Model evaluation
│   ├── arena.py           # Arena tournament system (~800 lines)
│   ├── elo.py             # ELOTracker rating system (~300 lines)
│   └── test_evaluation.py # Evaluation tests
│
├── config/                # Configuration
│   ├── config.py          # TrainingConfig dataclass (344 lines)
│   ├── train.py           # CLI entry point (~400 lines)
│   ├── requirements.txt   # Python dependencies
│   └── requirements-cpu.txt
│
├── tests/                 # Integration tests
│   ├── test_integration.py              # End-to-end workflow
│   └── test_imperfect_info_integration.py  # Imperfect info workflow
│
├── specs/                 # Specifications
│   └── STATE_ENCODER_SPEC.md   # Complete 256-dim encoding spec
│
└── docs/                  # Documentation & findings
    ├── conclusion.md      # Project post-mortem with all findings
    ├── CLAUDE.md          # Full project context document
    ├── performance/       # Performance benchmarks & analysis (15 files)
    └── phases/            # Phase completion reports (Phases 1-4)
```

---

## 3. Game Rules Reference

The game engine is the foundation. Port this first and validate against all 135 tests.

### 3.1 Card System
- **Deck**: Standard 52 cards (4 suits x 13 ranks)
- **Suits**: Spades(♠), Hearts(♥), Clubs(♣), Diamonds(♦)
- **Ranks**: 2,3,4,5,6,7,8,9,10,J,Q,K,A (values 2-14)
- **Card index**: `suit_index * 13 + rank_index` (0-51)
  - ♠2=0, ♠A=12, ♥2=13, ..., ♦A=51

### 3.2 Round Structure
- **Players**: 3-8
- **Round sequence** (e.g., 5 players, C=7): `[7,6,5,4,3,2,1,1,1,1,1,2,3,4,5,6,7]` — 17 rounds
- **Trump rotation**: ♠→♥→♣→♦→None→♠→... (cycles every 5 rounds)

### 3.3 Bidding Phase
- Players bid sequentially (left of dealer first)
- Bid = exact number of tricks you expect to win (0 to cards_dealt)
- **Dealer constraint**: Dealer cannot bid such that total_bids == cards_dealt

### 3.4 Playing Phase
- Standard trick-taking: must follow suit if possible
- Trump beats non-trump; highest card in led suit wins otherwise
- Trick winner leads next trick

### 3.5 Scoring
- `score = (tricks_won == bid) ? (10 + bid) : 0`
- All-or-nothing: exact bid required for points

### 3.6 Rust Design Target
```rust
// ~200 bytes, stack-allocated, Copy
struct BlobState {
    hands: [u64; 8],         // Bitmask of 52 cards per player
    played_this_trick: u64,  // Cards in current trick
    played_this_round: u64,  // All cards played
    bids: [u8; 8],           // Player bids (255 = not yet bid)
    tricks_won: [u8; 8],
    trump_suit: u8,          // 0-3 for suits, 4 for no-trump
    current_player: u8,
    dealer: u8,
    num_players: u8,
    cards_dealt: u8,
    game_phase: u8,          // 0=bidding, 1=playing, 2=complete
    trick_leader: u8,
    trick_play_order: [u8; 8],  // Card indices played this trick in order
}
```

---

## 4. Neural Network Architecture

### 4.1 BlobNet (Transformer, ~4.9M parameters)
- **Input**: 256-dim float vector (encoded game state)
- **Embedding**: Linear(256→256) + LayerNorm
- **Transformer**: 6 layers, 8 heads, FFN dim 1024, dropout 0.1, ReLU activation
- **Policy head**: Linear(256→256) → ReLU → Dropout → Linear(256→52) → masked softmax
- **Value head**: Linear(256→128) → ReLU → Dropout → Linear(128→1) → Tanh
- **Init**: Xavier uniform for all linear layers, zeros for biases

### 4.2 State Encoding (256 dimensions)
Full spec in `legacy/specs/STATE_ENCODER_SPEC.md`. Summary:

| Offset | Dims | Feature |
|--------|------|---------|
| 0-51 | 52 | My hand (binary: 1 if I have card) |
| 52-103 | 52 | Current trick (0=not played, 1-8=play order) |
| 104-155 | 52 | All cards played this round (binary) |
| 156-163 | 8 | Player bids (normalized, -1 if not bid) |
| 164-171 | 8 | Player tricks won (normalized) |
| 172 | 1 | My bid (normalized, -1 if not bid) |
| 173 | 1 | My tricks won (normalized) |
| 174-181 | 8 | Round metadata (cards, trick#, position, players, trump one-hot) |
| 182 | 1 | Bidding constraint active (binary) |
| 183-185 | 3 | Game phase one-hot (bidding/playing/scoring) |
| 186-201 | 16 | Positional features (hand stats, suit counts, high cards, trump strength) |
| 202-255 | 54 | Game context (scores, round position, previous cards, config; 25 reserved) |

### 4.3 Action Space
- **Unified 52-dim** output (max of 14 bid actions and 52 card actions)
- **Bidding**: Actions 0-13 represent bid values 0-13
- **Playing**: Actions 0-51 represent card indices (same as card encoding)
- **Legal action mask**: Binary mask applied before softmax (illegal → -inf)

### 4.4 Loss Function
- **Policy**: Cross-entropy: `-sum(target * log(pred + 1e-8))`, averaged over batch
- **Value**: MSE between predicted and actual outcome
- **Combined**: `policy_weight * policy_loss + value_weight * value_loss` (both weights = 1.0)
- **Optimizer**: Adam(lr=0.001, weight_decay=1e-4), gradient clip max_norm=1.0

### 4.5 Rust Implementation Choice
Use `tch-rs` (libtorch bindings) for GPU training. For MCTS inference, use `ort` (ONNX Runtime) on CPU — avoids CUDA launch overhead for single-sample evaluation. Alternatively, batch 32-64 MCTS leaf evaluations and run one CPU inference call (~1ms total).

---

## 5. MCTS Implementation

### 5.1 Core Algorithm (AlphaZero-style)
Each simulation performs:
1. **Selection**: Traverse tree using UCB1: `Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))`
2. **Expansion**: Create child nodes for all legal actions, set prior = network policy
3. **Evaluation**: Run neural network on leaf state → (policy, value)
4. **Backpropagation**: Update visit counts and values up to root

### 5.2 Imperfect Information (Determinization)
Since we can't see opponent hands:
1. Sample N determinizations (consistent opponent hand assignments)
2. Run full MCTS on each determinized world
3. Aggregate visit counts across all worlds
4. Final action probabilities = normalized aggregate visits

### 5.3 Belief Tracking
- Track which suits each opponent **cannot** have (revealed by not following suit)
- Track cards already played (removed from unseen pool)
- Use constraints when sampling determinizations
- `PlayerConstraints` dataclass: `eliminated_suits: Set[str]`, `cards_played`, `cards_in_hand`

### 5.4 Determinization Sampling
- Pool = all 52 cards minus (my hand + played cards)
- For each opponent: filter pool by constraints, sample `cards_in_hand` cards, remove from pool
- Rejection sampling with early termination if constraints can't be satisfied
- Target: <10ms per sample

### 5.5 Rust Design Target
```rust
// Arena-allocated nodes, cache-friendly
struct MctsNode {
    visit_count: u32,
    total_value: f32,
    prior: f32,
    action: u8,
    children: SmallVec<[u32; 8]>,  // Indices into arena Vec
}

struct MctsArena {
    nodes: Vec<MctsNode>,  // Contiguous allocation
}
```

Key difference from Python: arena allocation instead of heap-allocated objects. All nodes contiguous in memory for cache-friendly traversal.

---

## 6. Training Pipeline

### 6.1 Self-Play Loop
```
For each iteration:
  1. Generate N rounds/games using current model + MCTS
  2. Store (state, mcts_policy, outcome) tuples in replay buffer
  3. Train network on random batches from buffer (10 epochs)
  4. Evaluate new model vs current best (every K iterations)
  5. Promote if win rate > 55%
  6. Checkpoint model and buffer
```

### 6.2 Training Example Format
```rust
struct TrainingExample {
    state: [f32; 256],    // Encoded game state
    policy: [f32; 52],    // MCTS visit count distribution
    value: f32,           // Final game outcome for this player
}
```

### 6.3 Replay Buffer
- **Capacity**: 500,000 examples, circular FIFO
- **Sampling**: Uniform random batch of 512
- **Rust design**: Three contiguous `Vec<f32>` (states, policies, values) — NOT dicts
- **Checkpoint**: `mmap` to disk (~100ms instead of 13s pickle)

### 6.4 Self-Play Workers
- **Python**: 32 multiprocessing workers with IPC serialization overhead
- **Rust**: `rayon` thread pool, shared memory, zero IPC — workers share network and buffer directly

### 6.5 Training Hyperparameters (from config.py)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Workers | 32 | Max for RTX 4060 8GB |
| Rounds per iteration | 2,000→10,000 | Linear ramp over 500 iters |
| Batch size | 512 | Training gradient updates |
| Epochs per iteration | 10 | Passes over buffer |
| Learning rate | 0.001 | Adam optimizer |
| Weight decay | 1e-4 | L2 regularization |
| Buffer capacity | 500,000 | Circular FIFO |
| Eval games | 400 | Model tournament size |
| Promotion threshold | 0.55 | Win rate to promote |
| Gradient clip | 1.0 | Max gradient norm |

### 6.6 MCTS Curriculum Schedule

| Iteration | Determinizations | Sims/Det | Total Sims/Move |
|-----------|-----------------|----------|-----------------|
| 1-50 | 1 | 15 | 15 |
| 51-150 | 2 | 25 | 50 |
| 151-300 | 3 | 35 | 105 |
| 301-450 | 4 | 45 | 180 |
| 451-500 | 5 | 50 | 250 |

**Critical finding**: 1x15 produced zero learning signal (uniform visit distributions). The Rust rewrite should start at **minimum 3x50 = 150 sims/move** since Rust self-play will be fast enough. Consider starting at 5x100 = 500 sims/move.

### 6.7 Player Distribution (for training variety)

| Players | Probability | Starting Cards |
|---------|------------|----------------|
| 4 | 15% | 7 (60%) or 8 (40%) |
| 5 | 70% | 7 |
| 6 | 15% | 7 |

---

## 7. Evaluation System

### 7.1 Arena
- Two models play N games (default 400) with fair position rotation
- Each model controls one player seat, rest filled by copies of either model
- Track win rates per model

### 7.2 ELO Rating
- Standard ELO formula: `E = 1 / (1 + 10^((R_opp - R_self) / 400))`
- Update: `new = old + K * (actual - expected)`, K=32
- Starting ELO: 1000
- Track ELO history across all iterations

---

## 8. Critical Lessons to Apply in Rust

### 8.1 MCTS Signal Strength (MUST FIX)
The Python model never learned because 15 sims/move produced uniform visit distributions. In Rust:
- Start with **5x100 sims/move minimum** (Rust speed makes this feasible)
- **Validate before training**: Run MCTS on a few states, verify visit counts are non-uniform
- If top action has <2x visits of average, increase simulation budget

### 8.2 Replay Buffer (MUST FIX)
Python used `List[Dict]` — 650MB scattered heap, 100% L3 cache miss rate. In Rust:
- Use three contiguous `Vec<f32>` (states, policies, values)
- Batch sample = `indices.iter().map(|&i| &states[i*256..(i+1)*256])` — stays in cache
- Checkpoint via `mmap` (zero-copy, ~100ms for 500K examples)

### 8.3 GPU Utilization (MUST FIX)
Python had <5% GPU utilization due to tiny kernels + sync overhead. In Rust:
- Accumulate losses on GPU, read once per epoch (no `.item()` per batch)
- Consider CPU-only training via `tch-rs` with oneDNN — for 5M params, CPU may beat GPU when eliminating all overhead
- If using GPU: fuse operations, minimize kernel launches

### 8.4 Self-Play Architecture
Python used 32 multiprocessing workers with full process isolation. In Rust:
- Use `rayon` thread pool — zero IPC, shared memory
- Workers share the neural network (read-only during self-play)
- Workers write directly to shared replay buffer (mutex or lock-free)

---

## 9. Porting Order (Recommended)

### Phase 1: Game Engine
- Port `blob.py` and `constants.py` to Rust
- Use bitwise card representation (`u64` bitmasks)
- Port all 135 tests from `test_blob.py` — they serve as the specification
- Validate: stack-allocated state, ~50ns copy, ~10ns per legal move check

### Phase 2: State Encoder + Action Masker
- Port `encode.py` encoding logic
- Validate against `STATE_ENCODER_SPEC.md` — every dimension must match
- Output: `[f32; 256]` tensor

### Phase 3: MCTS
- Port `node.py` with arena allocation
- Port `search.py` MCTS loop
- Port `belief_tracker.py` constraint tracking
- Port `determinization.py` sampling
- Validate against MCTS tests

### Phase 4: Neural Network
- Set up `tch-rs` or `candle` for the Transformer
- Match BlobNet architecture exactly (6 layers, 8 heads, 256-dim)
- Verify forward pass outputs match Python (within 1e-5 tolerance)

### Phase 5: Training Pipeline
- Replay buffer as contiguous tensors
- Self-play with rayon thread pool
- Training loop calling libtorch
- Checkpoint/resume support

### Phase 6: Evaluation + CLI
- Arena system
- ELO tracking
- CLI with clap

---

## 10. Rust Crate Recommendations

| Purpose | Crate | Notes |
|---------|-------|-------|
| ML training | `tch` (libtorch) | Most mature PyTorch-in-Rust option |
| ML inference | `ort` | ONNX Runtime for MCTS leaf eval |
| Parallelism | `rayon` | Data-parallel thread pool |
| CLI | `clap` | Argument parsing |
| Serialization | `serde` + `bincode` | Config files, checkpoints |
| Arena alloc | `typed-arena` or manual `Vec<Node>` | MCTS node storage |
| Small vectors | `smallvec` | MCTS child lists |
| Random | `rand` + `rand_xoshiro` | Fast RNG for determinization |
| Logging | `tracing` | Structured logging |
| Progress | `indicatif` | Progress bars for training |
| Testing | built-in + `proptest` | Property-based testing for game rules |

---

## 11. Files NOT Ported (and why)

| File/Artifact | Reason |
|---------------|--------|
| Model checkpoints (`.pth`) | Learned nothing (uniform policy). No weights to transfer. |
| Replay buffer saves | Training data from random play. No value. |
| `gpu_server.py` | Abandoned architecture (3-5x slower than multiprocessing) |
| `batch_evaluator.py` | Python-specific batching workaround. Rust won't need it. |
| `dashboard.py`, `monitor.py` | Python monitoring tools. Rewrite with `tracing`. |
| `benchmarks/` | Python-specific benchmarks. Invalid for Rust. |
| Backend (`backend/`) | Phase 6 stubs only. Will be Rust-native or kept as TS. |
| Frontend (`frontend/`) | Phase 7 stubs only. Svelte stays as-is. |

---

## 12. Expected Performance Targets

Based on conclusion.md analysis:

| Component | Python (actual) | Rust (projected) | Speedup |
|-----------|----------------|-------------------|---------|
| Self-play (5x100, 2500 rounds) | ~10+ min | ~15 sec | ~40x |
| Training loop (9,760 batches) | 4 min 20s | ~10-20 sec | ~15x |
| Buffer checkpoint (500K examples) | 13 sec | ~0.1 sec | ~130x |
| Full iteration | 5 min 18s | ~30-45 sec | ~7-10x |
| 500 iterations | ~44 hours | ~4-6 hours | ~8x |

The most important outcome: **5x100 MCTS in Rust will be faster than 1x15 in Python**, meaning the model will actually produce learning signal.

---

## 13. Verification Checklist

Before considering the port complete, verify:

- [ ] All 135 game engine tests pass (ported from `test_blob.py`)
- [ ] State encoder produces identical 256-dim vectors for the same game state (cross-validate with Python)
- [ ] MCTS with 5x100 sims produces non-uniform visit distributions (top action >2x average)
- [ ] Self-play generates valid training examples (state/policy/value shapes correct)
- [ ] Replay buffer samples are uniformly distributed
- [ ] Training loop reduces policy loss below `ln(avg_legal_actions)` within 10 iterations
- [ ] ELO increases above 1000 within 20 iterations
- [ ] Full iteration completes in <60 seconds
- [ ] 32-thread self-play achieves >80% scaling efficiency
- [ ] Checkpoint/resume produces identical results
