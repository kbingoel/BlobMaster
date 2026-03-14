# BlobMaster Project Conclusion

**Date**: 2026-03-14
**Project Duration**: 2025-11-12 to 2026-03-14 (~4 months)
**Status**: Concluded after Phase 4 (Self-Play Training Pipeline)

---

## What We Built

An AlphaZero-style reinforcement learning system for the card game "Blob" (trick-taking with bidding), consisting of:

- **Game Engine**: Complete Blob implementation supporting 3-8 players (135 tests)
- **Neural Network**: BlobNet Transformer (~4.9M parameters, 6 layers, 8 heads, 256-dim embedding)
- **MCTS**: Monte Carlo Tree Search with determinization for imperfect information (belief tracking, suit elimination)
- **Training Pipeline**: Self-play engine (32 multiprocessing workers), replay buffer (500K capacity), network trainer with mixed precision, ELO evaluation arena
- **Total**: ~2,750 lines production code, 460 tests across all phases

**Phases completed**: 1-4 (Game Engine, MCTS+Network, Imperfect Information, Training Pipeline)
**Phases not started**: 5-7 (ONNX Export, Backend API, Frontend UI)

---

## Critical Findings

### 1. The Model Never Learned Anything

After 36 training iterations, all loss metrics flatlined:

| Metric | Iteration 1 | Iteration 14 | Iteration 36 | Interpretation |
|--------|-------------|--------------|--------------|----------------|
| Policy Loss | 3.902 | 3.573 | 3.567 | Converged to ln(~35) = uniform distribution over legal actions |
| Value Loss | 0.154 | 0.096 | 0.096 | Converged to predicting the mean outcome |
| ELO | 1000 | 1000 | 1000 | No improvement, slight drops during evaluation (-9 to -14) |

**Root cause**: The MCTS curriculum starts at 1 determinization x 15 simulations. With a branching factor of ~5-10 legal actions per turn, 15 simulations explore maybe 2-3 actions to depth 2. The resulting visit counts across actions are nearly uniform, so the network learns to output a uniform distribution, which produces uniform MCTS targets, creating a vicious cycle of random play training on random play.

**Key insight**: For AlphaZero to work, MCTS must be strong enough to discover better-than-random moves even with a random initial policy. For Blob (trick-taking with imperfect information), 15 simulations is far below this threshold. For comparison, AlphaZero for chess used 800 simulations per move.

### 2. Python Is the Wrong Language for This Project

The training loop analysis revealed that Python overhead completely dominates actual computation:

**Per training batch (observed: 26.6ms)**:
| Component | Time | What's happening |
|-----------|------|-----------------|
| GPU compute (forward + backward + optimizer) | <1 ms | The actual useful work |
| Random dict traversal in replay buffer | ~9 ms | 512 random lookups into 500K scattered Python dicts |
| CUDA kernel launch/sync overhead | ~9 ms | ~300 tiny kernel launches per batch, each doing microseconds of work |
| `.item()` synchronization stalls | ~4 ms | 4 calls per batch force CPU-GPU sync, preventing pipelining |
| Python interpreter overhead | ~4 ms | Function calls, context managers, dict creation |

**The GPU does <1ms of useful work per batch, then waits 25ms for Python.** For 9,760 batches per iteration (10 epochs x 976 batches), this adds up to 4 minutes 20 seconds of single-threaded CPU work per iteration, while the GPU sits idle.

**Per-iteration breakdown (total: ~5 min 18s)**:
| Phase | Time | CPU Pattern |
|-------|------|------------|
| Self-play (32 workers, MCTS) | ~38s | All cores ~90% |
| Training loop (9,760 batches) | ~4m 20s | Single core, GPU idle |
| Replay buffer pickle save (651 MB) | ~13s | Single core, disk I/O |
| Model checkpoint (57 MB) | ~2s | Single core |
| Dashboard/status updates | ~5s | Single core |

**Why Python specifically fails here**: This project is 99% fine-grained operations (MCTS tree traversal at ~10us/node in Python vs ~10ns in Rust, game state copies at ~100us vs ~50ns, dict lookups for every buffer access). Python's ~1000x per-operation overhead is negligible when operations take milliseconds (LLM training), but catastrophic when operations take nanoseconds (game tree search).

### 3. The Replay Buffer Design Is Fundamentally Inefficient

The buffer stores 500,000 training examples as a `List[Dict[str, Any]]` — half a million Python dictionaries scattered across ~650 MB of heap memory. Every training batch requires:

1. `np.random.choice(500000, 512)` — sample indices
2. `[self.buffer[i] for i in indices]` — 512 random pointer chases through 650 MB heap (far exceeds 64 MB L3 cache)
3. `np.stack([ex["state"] for ex in batch])` — gather 512 numpy arrays from random heap locations
4. `torch.from_numpy(...).float().to(device)` — convert and transfer to GPU

This should be three pre-allocated contiguous tensors with batch sampling as a single `tensor[indices]` operation.

Additionally, checkpointing uses `pickle.dump` on the entire dict-of-lists structure, writing 651 MB per iteration. With contiguous tensors, this could use `mmap` and complete in ~100ms instead of ~13 seconds.

### 4. The GPU Is Massively Underutilized

The RTX 4060 has 15 TFLOPS of compute. A forward+backward pass through a 5M parameter model with batch_size=512 requires ~15 GFLOPS — roughly 1ms of GPU time. But each batch involves ~300 CUDA kernel launches (linear layers, attention, layernorm, dropout, residual connections, gradient ops, optimizer updates, gradient clipping — each a separate kernel), and each launch has ~5-10us of fixed overhead regardless of the actual computation.

The model is too small for the GPU. Each kernel does microseconds of useful work but pays the full launch overhead. This is the opposite of LLM training where kernels run for milliseconds and launch overhead is negligible.

### 5. Hardware Limits Identified

- **RTX 4060 8GB maximum**: 32 self-play workers (each ~150MB VRAM). 48 workers causes CUDA OOM.
- **Scaling efficiency degrades**: 92% at 4 workers, 44% at 32 workers (diminishing returns from multiprocessing overhead).
- **Linux vs Windows**: ~15% performance penalty on Ubuntu vs Windows for the same workload (36.7 vs 43.3 games/min).

### 6. Performance Benchmarks Established

Self-play throughput on Ryzen 9 7950X + RTX 4060 (32 workers):

| MCTS Config | Total Sims/Move | Games/Min | Rounds/Min (Phase 1) |
|-------------|-----------------|-----------|----------------------|
| Light (2x20) | 40 | 69.1 | ~1,049 |
| Medium (3x30) | 90 | 36.7 | ~741 |
| Heavy (5x50) | 250 | 16.0 | ~310 |

Bayesian optimization (Optuna TPE, 25 trials per stage) tuned parallel batch parameters, achieving +6.4% to +48.7% speedup across curriculum stages.

---

## The Fundamental Architectural Mistake

This project used Python for a workload where Python is worst-case:

| Operation | Python | Rust/C++ | Ratio |
|-----------|--------|----------|-------|
| Function call | ~1 us | ~1 ns | 1000x |
| Game state copy | ~100 us | ~50 ns | 2000x |
| MCTS node visit | ~10 us | ~10 ns | 1000x |
| Dict/HashMap lookup | ~0.5 us | ~20 ns | 25x |
| Buffer batch sample | ~5 ms | ~5 us | 1000x |

In a project where 80% of compute time is MCTS (millions of node visits per iteration) and 15% is training loop overhead (thousands of batch-prep operations), Python's per-operation overhead IS the bottleneck. The actual math (matrix multiplications, game logic) is trivially fast on the available hardware.

**Estimated impact of a Rust rewrite**:

| Component | Current (Python) | Projected (Rust) |
|-----------|-----------------|-------------------|
| Self-play (1x15, 2500 rounds) | 38 sec | ~0.5 sec |
| Self-play (5x100, 2500 rounds) | ~10+ min (estimated) | ~15 sec |
| Training loop (9,760 batches) | 4 min 20s | ~10-20 sec |
| Buffer checkpoint | 13 sec | ~0.1 sec (mmap) |
| **Full iteration** | **5 min 18s** | **~30-45 sec** |
| **500 iterations** | **~44 hours** | **~4-6 hours** |

The most important consequence: with Rust self-play, **5x100 MCTS would be faster than current 1x15**, meaning the model would actually learn.

---

## Recommended Architecture for a Rebuild

### Language Choice

**Rust for everything performance-critical. Python only for analysis/visualization.**

- **Rust**: `tch-rs` (libtorch bindings) for GPU training, `ort` (ONNX Runtime) for MCTS inference, mature ML ecosystem (`candle`, `burn`)
- **Not Zig**: Equally fast, but Rust has better ML library ecosystem
- **Not C++**: Same performance, but Rust prevents the memory bugs that would cost days of debugging in a complex MCTS implementation
- **Not Python**: Fine when GPU compute dominates (LLM training). Fatal when per-operation overhead dominates (game tree search with a small model)

### Design Principles

**1. Game state as a compact struct (~200 bytes, stack-allocated)**
```rust
struct BlobState {
    hand: u64,           // Bitmask of 52 cards
    played: u64,         // Cards played this trick
    bids: [u8; 8],       // 8 bytes for all players
    tricks_won: [u8; 8],
    trump: u8,
    current_player: u8,
    // Copyable in ~50ns, not ~100us
}
```

**2. MCTS with arena allocation (nodes contiguous in memory)**
```rust
struct MctsNode {
    visit_count: u32,
    total_value: f32,
    children: SmallVec<[u32; 8]>,  // Indices into arena
    action: u8,
    prior: f32,
}
// ~64 bytes per node, cache-friendly traversal
```

**3. Replay buffer as contiguous tensors (not Python dicts)**
```rust
struct ReplayBuffer {
    states: Vec<[f32; 256]>,   // 500K x 256, contiguous
    policies: Vec<[f32; 61]>,  // 500K x 61, contiguous
    values: Vec<f32>,          // 500K x 1, contiguous
}
// Batch sample: memcpy a slice. Checkpoint: mmap to disk.
```

**4. MCTS inference on CPU via ONNX Runtime (not PyTorch on GPU)**

For batch_size=1 through a 5M param model:
- PyTorch on GPU: ~1ms (CUDA launch overhead)
- ONNX Runtime on CPU: ~0.2-0.5ms (no launch overhead, data stays in cache)

Better: batch 32-64 MCTS leaf evaluations, run one CPU inference call: ~1ms total.

**5. Training on GPU via tch-rs, without Python overhead**

Call libtorch directly from Rust. No `.item()` sync per batch, no Python context managers, no dict creation. Accumulate losses on GPU, read once per epoch.

Alternatively: for a 5M param model, train entirely on CPU with oneDNN. The 7950X with optimized BLAS can do ~10-20ms per batch with zero overhead — competitive with GPU+Python at 26ms per batch.

### Architecture Diagram

```
Rust Binary (single process)
├── Game Engine          [stack-allocated states, ~10ns/move]
├── MCTS                 [arena-allocated nodes, ~10ns/visit]
├── Belief Tracker       [bitwise suit elimination]
├── Replay Buffer        [contiguous tensors, mmap checkpoint]
├── Self-Play Workers    [rayon thread pool, zero IPC overhead]
├── ONNX Runtime         [CPU inference for MCTS, ~0.2ms/eval]
└── tch-rs / libtorch    [GPU training, called once per epoch]
```

Key difference from Python architecture: **no multiprocessing** (and its serialization/IPC overhead). Rust threads share memory safely via the type system. Self-play workers can share the neural network and replay buffer directly.

---

## Lessons Learned

### About AlphaZero Training
1. **MCTS simulation budget must exceed a minimum threshold** for the specific game's branching factor. Below this threshold, MCTS targets are indistinguishable from uniform random, and the network learns nothing. For Blob: minimum ~50-100 simulations per move (not 15).
2. **The curriculum approach (start weak, ramp up MCTS) only works if the starting point produces non-zero signal.** Starting at 1x15 for Blob produced zero signal, so 500 iterations of training were wasted before reaching the 2x25 stage.
3. **ELO tracking is essential for detecting stagnation early.** The flat ELO at 1000 across 36 iterations was the clearest signal that training was not working, more informative than loss curves alone.
4. **Loss plateau != convergence.** A policy loss of ln(K) where K is the average number of legal actions means the network is outputting uniform probabilities — it has "converged" to knowing nothing.

### About Performance Engineering
5. **Profile before optimizing.** The assumption was that self-play (38s, all cores busy) was the bottleneck. In reality, the training loop (4m 20s, single core) was 7x worse — but invisible without profiling because it showed no CPU utilization.
6. **Python overhead is invisible in standard profiling.** The 26ms/batch wasn't in any single function — it was spread across thousands of interpreter operations, CUDA launch overhead, and memory management. Only wall-clock analysis of file timestamps revealed the true cost.
7. **GPU utilization can be near-zero even when "using the GPU."** Every batch was dispatched to the GPU, but the GPU completed its work in <1ms and waited 25ms for the next batch. A GPU utilization monitor would have shown <5% utilization during training.
8. **For small models, CPU can beat GPU** when you account for CUDA launch overhead, CPU-GPU transfer latency, and synchronization costs. The crossover point is roughly where per-batch GPU compute exceeds ~5ms (models >50M parameters with reasonable batch sizes).

### About Language Choice for ML Projects
9. **Python is optimal when GPU compute dominates** (large models, large batches, long kernel execution times). The ~1ms of Python overhead per batch is negligible against 500ms of GPU compute for an LLM.
10. **Python is catastrophic when CPU logic dominates** (game simulation, tree search, many small operations). The ~1000x per-operation overhead turns a 4-hour Rust job into a 4-day Python job.
11. **The ML ecosystem's Python dependency is not a fundamental constraint.** libtorch, ONNX Runtime, and oneDNN all have C/Rust APIs. You can train and run neural networks without Python.
12. **Multiprocessing in Python (to work around the GIL) adds serialization overhead** that doesn't exist in languages with real threading. The 32-worker multiprocessing pool serializes model weights, game states, and training examples across process boundaries — all unnecessary with shared-memory threading in Rust.

### About Hardware Utilization
13. **The Ryzen 9 7950X (16C/32T) and 128 GB DDR5 are dramatically underutilized.** Self-play uses 32 Python workers at ~44% scaling efficiency. Training uses 1 core. The hardware could sustain 10-100x higher throughput with efficient code.
14. **The RTX 4060's 15 TFLOPS are irrelevant for a 5M parameter model.** Each forward+backward pass uses ~15 GFLOPS — 0.1% of the GPU's capacity. The GPU's value here is not compute but memory bandwidth and parallel kernel execution, neither of which is utilized by tiny per-batch workloads.
15. **Cache hierarchy matters for data structure design.** The replay buffer (500K Python dicts, ~650 MB scattered heap) vastly exceeds the 64 MB L3 cache. Every batch access pattern is random, causing near-100% L3 miss rate. Contiguous tensor storage would keep the active working set in cache.

---

## What Would Be Different Next Time

1. **Start with Rust** for game engine and MCTS from day one. Use Python only for Jupyter notebooks and data analysis.
2. **Validate MCTS signal strength before building the training pipeline.** Run MCTS with various simulation budgets on a few game states and verify that visit count distributions are meaningfully non-uniform before investing in the full training loop.
3. **Use contiguous tensor storage** for the replay buffer from the start. Never store training data as Python dicts.
4. **Consider CPU-only training** for models under ~50M parameters. The CUDA overhead tax exceeds the compute benefit at this scale.
5. **Start with a simpler game configuration** (2-3 cards, 3 players) where even light MCTS can discover optimal play, then transfer to harder configurations.
6. **Instrument GPU utilization from iteration 1.** `nvidia-smi` or PyTorch's CUDA profiler would have revealed the <5% GPU utilization immediately.

---

## Project Artifacts

### What to Keep
- **Game engine** (`ml/game/blob.py`): Well-tested (135 tests), correct implementation of Blob rules. Useful as a reference specification for a Rust port.
- **Test suites** (460 tests): Encode game rules, edge cases, and expected behavior. Essential for validating a rewrite.
- **Performance benchmarks** (`docs/performance/`): Hardware limits, scaling curves, and profiling data. Still valid for the same hardware.
- **This document**: Captures all findings and architectural decisions.

### What to Discard
- **Trained model checkpoints**: The model learned nothing (uniform policy, constant value prediction). No useful weights to transfer.
- **Replay buffer saves**: Training data generated by random play has no value.
- **Training logs/dashboard history**: Documents 36 iterations of non-learning. Useful only as evidence of the MCTS signal problem.

---

*This project was a valuable learning exercise in AlphaZero-style reinforcement learning, transformer architectures, MCTS for imperfect information games, and the performance characteristics of Python for compute-intensive ML workloads. The core game logic and test suite remain solid foundations for a future Rust-based implementation.*
