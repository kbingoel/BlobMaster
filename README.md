# BlobMaster

AlphaZero-style AI for the card game "Blob" (trick-taking with bidding, 3-8 players).

## Status: Rust Rewrite — Architecture Finalized

The original Python/PyTorch implementation (Phases 1-4) has been **concluded and archived**. It produced a correct, well-tested game engine and training pipeline, but Python's per-operation overhead made MCTS too slow to generate useful learning signal. The model never learned. Full post-mortem in [conclusion.md](conclusion.md).

The Rust rewrite is in progress. The neural network architecture has been fully specified and supersedes the legacy design.

## What's Here

```
README.md              ← You are here
conclusion.md          ← Why Python failed, what to do differently
architecture.md        ← Structured Entity Transformer specification (current)
prepare-migration.md   ← Rust rewrite plan (game engine, MCTS, training pipeline)
legacy/                ← Archived Python reference code
```

## Neural Network: Structured Entity Transformer (~1.63M params)

The legacy BlobNet was a 6-layer Transformer over a flat 256-dim state vector (~4.9M params). Self-attention over a single token is mathematically degenerate — it reduces to a linear projection. The new architecture fixes this.

**Core idea**: represent the game state as a set of typed entity tokens, each encoding domain-specific features. The Transformer then learns meaningful attention patterns across entities.

### Token Types

| Token | Count | What it encodes |
|---|---|---|
| Hand card | 1–13 | Rank, suit, trump status, relative strength, suit counts |
| Played card | 0–48 | Rank, suit, player, trick number, lead/follow, win status |
| Player state | 3–8 | Bid, tricks won, bid status, suit voids, position |
| Context | 1 | Trump suit, cards dealt, trick progress, game phase |
| CLS | 1 | Learned aggregation token, feeds value and bid heads |

**Shared embeddings**: rank (13×16) and suit (4×8) are shared between hand and played card tokens; player (8×16) is shared between played cards and player states. The model learns card identity once.

**Chronological embeddings**: played card tokens receive an additional learned positional embedding (52×128) encoding their order in the play sequence.

### Transformer Config

`d_model=128`, `num_layers=8`, `num_heads=8`, `ffn_dim=512`, Pre-norm, GeLU, dropout=0.1

### Output Heads

- **Playing head**: each hand card token → shared MLP → scalar score → softmax over legal cards. Entity-native, 4K params vs 23K for a CLS→52 head.
- **Bidding head**: CLS → Linear(128→64) → Linear(64→14) → masked softmax. Separate from playing head to eliminate gradient interference at bid-value positions.
- **Value head**: CLS → Linear(128→64) → Linear(64→1) → tanh. Target is z-scored final score: `clip((my_score − mean) / std, −1, 1)`.

### Why This Architecture

Blob's key reasoning patterns are inherently sequential and relational:
- **Suit voids**: when a player fails to follow suit, they reveal a permanent constraint. Precomputed as binary flags on player tokens.
- **Card counting with attribution**: knowing *who* played *what* in *which trick* enables hand ranging. This is a natural attention query: a hand card attends to played cards of the same suit to determine its current dominance.
- **Bid-progress tracking**: `bid_status` (met / busted / live) drives opponent modeling; `tricks_needed` provides urgency. Both are precomputed features.
- **Multi-trick planning**: the full ordered play history gives the network context for continuing or abandoning multi-trick plans.

Precomputing key derived features (void flags, `is_highest_in_suit`, `cards_above_remaining`) reduces the burden on attention while leaving the network free to learn nuanced patterns through self-attention.

### Inference Performance

At a typical 35-token sequence: ~57M MACs per forward pass, ~0.15ms on CPU (ONNX Runtime). With 5 determinizations × 100 MCTS simulations per move, one full game takes ~3s of neural network time. At 32 rayon threads on the 7950X: **~640 games/minute**, comfortably within the 2,000–10,000 games/iteration training target.

## Porting Order

1. **Game Engine** — Blob rules, bitwise card representation (`u64` bitmasks, ~50ns state copy), trick history log (`TrickRecord` struct)
2. **Entity Encoder** — `BlobState` → variable-length token sequence
3. **Structured Entity Transformer** — as specified in [architecture.md](architecture.md)
4. **MCTS** — Arena-allocated tree search with belief tracking and determinization
5. **Training Pipeline** — Self-play (`rayon`), contiguous replay buffer, training loop
6. **Evaluation + CLI** — Arena tournaments, ELO tracking, `clap` CLI

**Key fixes over Python version:**
- MCTS starts at 5×100 sims/move — enough signal to actually learn
- Strong policy prior breaks the vicious cycle: weak prior → uniform MCTS targets → weaker prior
- Replay buffer as contiguous tensors, not 500K Python dicts across a 650MB heap
- Full iteration in ~30–45s vs ~5 min; 500 iterations in ~4–6h vs ~44h

## Hardware

- **Training**: Ubuntu 24.04, Ryzen 9 7950X (16C/32T), RTX 4060 8GB, 128GB DDR5
- **Future inference**: Windows laptop, Intel i5 iGPU, ONNX Runtime
