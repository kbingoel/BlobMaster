# BlobMaster Rust Rewrite — Development Outline

**Estimated total**: ~20 sessions (4h each, ~80h), organized in 6 phases
**Approach**: Bottom-up port following the dependency chain — each phase produces a standalone, tested module before moving on. Legacy Python in `legacy/` is read-only reference; no runtime dependencies on it.

---

## Phase 1: Foundation — Game Engine (Sessions 1–5)

The game engine is the root dependency for everything else. The Python version uses OOP classes (Card, Deck, Player, Trick, BlobGame) with deep copies and heap allocation. The Rust version replaces this with a flat, stack-allocated `BlobState` struct using `u64` bitmasks for card sets — enabling ~50ns state copies vs ~100us in Python. This is the single most important architectural change: every MCTS simulation copies state thousands of times, so this 2000x speedup cascades through the entire system.

**Session 1 — Project scaffold + card/deck primitives**
Set up `cargo init`, workspace layout (`src/game/`, `src/encoder/`, `src/mcts/`, `src/network/`, `src/training/`, `src/eval/`), CI basics, and implement the `u64` bitmask card representation with suit/rank encoding, iteration, and population count helpers. Using a flat module layout (single crate, multiple modules) because the components are tightly coupled and a workspace would add unnecessary build complexity at this scale.

**Session 2 — Core game state and bidding phase**
Implement `BlobState` struct (~200 bytes, `Copy` trait), round structure generation, trump rotation, dealer rotation, and the complete bidding phase including the dealer constraint (total bids != cards dealt). Port the first ~40 bidding-related tests from `test_blob.py` as the specification.

**Session 3 — Trick-taking and scoring**
Implement the playing phase: follow-suit enforcement, trump resolution, trick winner determination, trick progression, and scoring (`exact bid = 10 + bid, else 0`). Port the ~50 playing/scoring tests. This session is where the bitmask design pays off — legal move generation becomes a single bitwise AND between hand and suit mask.

**Session 4 — Full game loop and multi-round games**
Wire up the complete game lifecycle: deal → bid → play → score → next round, with round structure sequences (descending → ones plateau → ascending). Implement `get_legal_actions()` as the unified interface returning either valid bids or playable cards depending on phase. Port the remaining ~45 tests covering edge cases, multi-round games, and the full 17-round 5-player game.

**Session 5 — Validation, benchmarks, and hardening**
Run all 135 ported tests, add property-based tests with `proptest` for rule invariants (e.g., trick winner always played a card, scores sum correctly, legal actions never empty during active play). Benchmark state copy, legal move generation, and full-game simulation to validate performance targets (~50ns copy, ~10ns legal move check). Fix any edge cases found by property testing.

---

## Phase 2: State Encoder (Sessions 6–7)

The encoder translates game state into the 256-dim float vector the neural network consumes. It must match the legacy spec exactly (within 1e-6) so that any future cross-validation or weight transfer is possible. This is a pure function with no dependencies beyond the game engine — a clean module boundary.

**Session 6 — Core encoding (dims 0–201)**
Implement the 12 encoding sections: hand bitmask → binary vector (0–51), trick cards with play order (52–103), played cards history (104–155), player bids/tricks (156–173), round metadata with trump one-hot (174–181), bidding constraint (182), game phase (183–185), and positional features including suit counts, high cards, trump strength (186–201). Each section is a small deterministic function — the complexity is in getting the normalization and special values exactly right (e.g., -1.0 for "no bid yet", not 0.0).

**Session 7 — Game context features, action masking, and golden tests**
Implement dims 202–255 (cumulative scores, round position, previous round history, game config, reserved zeros). Build the `ActionMasker` that produces a 52-dim binary mask of legal actions for the policy head. Create a golden test suite: manually construct diverse game states (early bidding, mid-trick, dealer constraint active, no-trump round, 3-player vs 8-player) and verify every dimension against the spec. This is the firewall that prevents subtle encoding bugs from silently corrupting training.

---

## Phase 3: MCTS (Sessions 8–12)

MCTS is the most complex component and the primary beneficiary of the Rust rewrite. The Python version uses heap-allocated node objects with Python dicts — each node visit costs ~10us. The Rust version uses arena allocation (`Vec<MctsNode>`) where all nodes are contiguous in memory and referenced by `u32` index — each visit drops to ~10ns. This 1000x improvement is what makes 5x100 sims/move feasible (previously too slow even at 1x15).

**Session 8 — Arena-allocated MCTS nodes and tree structure**
Implement `MctsArena` (a `Vec<MctsNode>`) and `MctsNode` (~64 bytes: visit count, total value, prior, action, `SmallVec<[u32; 8]>` children). Build tree operations: create root, expand node with children, and UCB1 selection (`Q + c_puct * P * sqrt(N_parent) / (1 + N_child)`). Arena allocation chosen over `Box<Node>` because it guarantees cache-locality during tree traversal — the hot path of every MCTS simulation.

**Session 9 — Core MCTS search loop (perfect information)**
Implement the full select → expand → evaluate → backpropagate loop for perfect-information search. Use a dummy evaluator (random policy, zero value) first to validate the tree mechanics independently of the neural network. Port core MCTS tests: verify visit counts increase, UCB1 selects correctly, backpropagation updates ancestors, and the algorithm converges to the best action in simple forced-win positions.

**Session 10 — Belief tracker and suit elimination**
Implement `BeliefState` with bitwise constraint tracking: when a player doesn't follow suit, mark that suit as eliminated for them (`eliminated_suits: [u8; 8]` as bitmask). Track cards played and cards remaining per opponent. This is simpler in Rust than Python because suit elimination is a single bit operation on a `u8` rather than a `Set[str]` mutation.

**Session 11 — Determinization sampling**
Implement the determinizer: given belief constraints, sample consistent opponent hand assignments from the unseen card pool. Use `rand_xoshiro` for fast RNG. Algorithm: for each opponent, filter available cards by suit constraints, sample `cards_in_hand` cards, remove from pool. Add rejection sampling with a retry limit for rare unsatisfiable constraint combinations. Validate that sampled hands respect all known constraints and that the distribution is reasonable.

**Session 12 — Imperfect-information MCTS integration**
Wire determinization into MCTS: run N determinized searches, aggregate visit counts across worlds, produce final action distribution. Implement the `NNEvaluator` trait so MCTS can call any evaluator (random, neural network, or a test stub). Validate that with enough simulations (100+), the imperfect-info MCTS produces non-uniform visit distributions on non-trivial game states — this is the critical signal-strength check that the Python version failed.

---

## Phase 4: Neural Network (Sessions 13–15)

The network is a 6-layer Transformer with dual heads (~4.9M parameters). Using `tch-rs` (Rust bindings to libtorch) rather than `candle` or `burn` because libtorch is the most battle-tested option for GPU training and provides the exact same ops as PyTorch — minimizing translation risk. For MCTS inference, we'll use `ort` (ONNX Runtime) on CPU to avoid CUDA launch overhead on single-sample evaluations.

**Session 13 — tch-rs setup and BlobNet architecture**
Install libtorch, configure `tch-rs` build, and implement BlobNet: Linear(256→256) + LayerNorm embedding, 6 Transformer encoder layers (8 heads, 1024 FFN dim, 0.1 dropout, ReLU), policy head (Linear→ReLU→Dropout→Linear(256→52)→masked softmax), value head (Linear(256→128)→ReLU→Dropout→Linear→Tanh). Xavier uniform init for weights, zeros for biases. Verify forward pass shape: input `[batch, 256]` → policy `[batch, 52]` + value `[batch, 1]`.

**Session 14 — Training mechanics and loss functions**
Implement the training loop internals: cross-entropy policy loss with legal action masking (`-sum(target * log(pred + 1e-8))`), MSE value loss, combined loss with equal weights, Adam optimizer (lr=0.001, weight_decay=1e-4), gradient clipping (max_norm=1.0). Critically: accumulate losses on GPU and read once per epoch — the Python version called `.item()` per batch, causing 4ms CPU-GPU sync stalls on each of 9,760 batches.

**Session 15 — ONNX export and CPU inference integration**
Export trained BlobNet to ONNX format, set up `ort` for CPU inference, and implement the `NNEvaluator` trait for MCTS. Benchmark single-sample and batched (32-64) inference latency. Target: <0.5ms single-sample on CPU via ONNX vs ~1ms via CUDA (where launch overhead dominates for this small model). This dual-runtime approach — libtorch for training, ONNX for MCTS inference — is the key architectural decision that avoids the GPU utilization problem from the Python version.

---

## Phase 5: Training Pipeline (Sessions 16–18)

The training pipeline orchestrates self-play, experience storage, and network updates. The Python version used 32 multiprocessing workers with full process isolation (IPC serialization of model weights and game states). Rust replaces this with `rayon` shared-memory threading — workers read the network and write to the replay buffer through the same address space, eliminating all serialization overhead.

**Session 16 — Contiguous replay buffer**
Implement `ReplayBuffer` as three contiguous `Vec<f32>` arrays (states: 500K×256, policies: 500K×52, values: 500K×1) with circular FIFO insertion and uniform random batch sampling. Checkpoint via `mmap`/direct binary write (~100ms for 500K examples vs 13s pickle in Python). The contiguous layout means batch sampling hits L3 cache instead of chasing 500K scattered heap pointers — this alone fixes the 9ms/batch random-access penalty from the Python version.

**Session 17 — Rayon self-play workers**
Implement `SelfPlayEngine`: spawn `rayon` thread pool, each worker runs full game simulations using MCTS + ONNX evaluator, collects (state, policy, value) tuples, and writes them to the shared replay buffer (via `Mutex<ReplayBuffer>` or a lock-free concurrent append). Implement the player distribution sampling (4p: 15%, 5p: 70%, 6p: 15%) and MCTS curriculum schedule (start at 5x100 sims/move). Validate scaling efficiency: target >80% at 16 threads on the 7950X.

**Session 18 — Training loop and checkpoint/resume**
Wire everything into the main training loop: generate N rounds via self-play → sample batches from buffer → train network for 10 epochs → checkpoint model + buffer. Implement resume-from-checkpoint (reload model weights, optimizer state, buffer contents, iteration counter). Add `tracing` structured logging and `indicatif` progress bars for training visibility. Run a smoke test: 5 iterations end-to-end, verify loss decreases and the pipeline doesn't crash.

---

## Phase 6: Evaluation, CLI, and Polish (Sessions 19–20)

**Session 19 — Arena evaluation and ELO tracking**
Implement `Arena`: two models play N games (default 400) with fair seat rotation, track win rates, compute ELO updates (K=32, starting 1000). Implement model promotion logic: new model replaces current best if win rate >55%. Add ELO history persistence (JSON or bincode). This is the feedback loop that tells you whether training is actually working — the Python version's flat ELO at 1000 across 36 iterations was the clearest signal of failure.

**Session 20 — CLI, integration tests, and launch readiness**
Build `clap` CLI with subcommands: `train` (full pipeline with all hyperparameter flags), `eval` (run arena between two checkpoints), `play` (single game with MCTS for debugging), `bench` (performance benchmarks). Write integration tests covering the full pipeline: init → self-play → train → eval → checkpoint → resume. Verify against the verification checklist from `prepare-migration.md`: all 135 game tests pass, MCTS produces non-uniform visits, loss drops below `ln(avg_legal_actions)` within 10 iterations, full iteration <60s. Delete `legacy/` folder.

---

## Risk Notes

- **tch-rs / libtorch build**: libtorch C++ linkage is the most common friction point in Rust ML projects. Session 13 has buffer time for this. Fallback: `candle` (pure Rust, no C++ dependency) at the cost of less mature GPU training support.
- **MCTS signal validation**: Session 12 explicitly validates that visit distributions are non-uniform. If they aren't, increase simulation budget before proceeding to Phase 5 — do not repeat the Python mistake of building an entire training pipeline on top of broken MCTS signal.
- **GPU vs CPU training**: The 5M parameter model may train faster on CPU (7950X with oneDNN) than GPU (4060) once Python overhead is removed. Session 14 should benchmark both and pick the winner. Don't assume GPU is faster for small models.
- **Scaling ceiling**: `rayon` threading eliminates IPC overhead but introduces `Mutex` contention on the replay buffer. If scaling stalls, switch to per-thread local buffers that merge after each iteration — a simple design change that avoids lock contention entirely.
