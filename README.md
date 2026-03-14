# BlobMaster

AlphaZero-style AI for the card game "Blob" (trick-taking with bidding, 3-8 players).

## Status: Rust Rewrite (starting fresh)

The original Python/PyTorch implementation (Phases 1-4) has been **concluded and archived**. It produced a correct, well-tested game engine and training pipeline, but Python's per-operation overhead made MCTS too slow to generate learning signal. The model never learned. Full post-mortem in [conclusion.md](conclusion.md).

The repository has been cleaned. All Python source is archived in `legacy/` as read-only reference for the Rust rewrite.

## What's Here

```
README.md              ← You are here
conclusion.md          ← Why Python failed, what to do differently
prepare-migration.md   ← Detailed Rust rewrite plan (specs, architecture, porting order)
legacy/                ← Organized Python reference code (game engine, MCTS, network, training)
```

## The Plan

Rewrite everything performance-critical in **Rust**. The core bottleneck was Python's ~1000x per-operation overhead on millions of MCTS node visits and game state copies per iteration. Rust eliminates this entirely.

**Porting order:**

1. **Game Engine** — Blob rules with bitwise card representation (`u64` bitmasks, ~50ns state copy)
2. **State Encoder** — Game state → 256-dim tensor (matching legacy spec exactly)
3. **MCTS** — Arena-allocated tree search with belief tracking and determinization
4. **Neural Network** — BlobNet Transformer via `tch-rs` (libtorch) or `candle`
5. **Training Pipeline** — Self-play (`rayon` threads), contiguous replay buffer, training loop
6. **Evaluation + CLI** — Arena tournaments, ELO tracking, `clap` CLI

**Key fixes over Python version:**
- MCTS starts at 5x100 sims/move (not 1x15) — enough signal to actually learn
- Replay buffer as contiguous tensors (not 500K Python dicts scattered across 650MB heap)
- Shared-memory threading via `rayon` (not 32 multiprocessing workers with IPC serialization)
- Full iteration in ~30-45s (not ~5 min), 500 iterations in ~4-6h (not ~44h)

## Hardware

- **Training**: Ubuntu 24.04, Ryzen 9 7950X (16C/32T), RTX 4060 8GB, 128GB DDR5
- **Future inference**: Windows laptop, Intel i5 iGPU, ONNX Runtime

## Detailed Plan

See [prepare-migration.md](prepare-migration.md) for the complete specification: game rules, network architecture, state encoding, MCTS algorithm, training hyperparameters, crate recommendations, and verification checklist.
