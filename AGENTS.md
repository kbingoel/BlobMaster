# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project State

The Rust rewrite has not started yet. This repository currently contains only planning documents and archived Python reference code. **No Rust source exists.** The next step is creating the Rust workspace and beginning Phase 1 (game engine).

## Planning Documents

Read these before making architectural decisions:

- **[architecture.md](architecture.md)** — The finalized neural network specification (Structured Entity Transformer). This supersedes the legacy BlobNet described in `prepare-migration.md`. Use `architecture.md` as the authoritative source for all network design decisions.
- **[prepare-migration.md](prepare-migration.md)** — Rust rewrite plan: game rules (§3), MCTS algorithm (§5), training pipeline (§6), crate recommendations (§10), verification checklist (§13).
- **[legacy/](legacy/)** — Archived Python source. Read-only reference. Never modify.

## Architecture Overview

The system is an AlphaZero-style MCTS+neural-network pipeline for a trick-taking card game.

### Data Flow

```
BlobState (stack, ~200 bytes)
  → Entity Encoder → variable-length token sequence (~14–56 tokens)
  → Structured Entity Transformer (1.63M params)
  → Playing head (hand card tokens → per-card scores → softmax)
  → Bidding head (CLS token → 14-dim softmax)
  → Value head (CLS token → scalar ∈ [-1, 1])
```

### Key Architectural Decisions

**BlobState** must be extended from the minimal struct in `prepare-migration.md §3.6` to include a `trick_history: [TrickRecord; 13]` field — the entity encoder requires the full ordered log of who played what card in which trick (see `architecture.md §5.1`).

**Entity encoder** transforms `BlobState` into five token types:
- Hand cards (1–13), Played cards (0–48), Player states (3–8), Context (1), CLS (1)
- Rank/suit embeddings are **shared** between hand and played card tokens
- Player embeddings are **shared** between played cards and player state tokens
- Played card tokens receive additional chronological embeddings (52×128 table)

**MCTS** uses determinization: sample N consistent opponent hand assignments, run full tree search on each, aggregate visit counts. Arena-allocate nodes as contiguous `Vec<MctsNode>`. Start at **5×100 sims/move minimum** — fewer produces uniform visit distributions and zero learning signal (the root cause of the Python failure).

**Training loop**: self-play via `rayon` thread pool → replay buffer (3× contiguous `Vec<f32>`) → gradient updates. Value target is z-scored final score: `clip((my_score − mean) / std_dev, −1, 1)`.

**Output heads**: the playing head scores each hand card token directly (entity-native, ~4K params); the bidding head reads from the CLS token (14 bid values, separate to prevent gradient interference at positions 0–13 which overlap between bid values and card indices in a unified head).

### Card Encoding

Card index = `suit_index * 13 + rank_index`. Suits: ♠=0, ♥=1, ♣=2, ♦=3. Ranks: 2=0 through A=12. Hands stored as `u64` bitmasks.

### Scoring

`score = (tricks_won == bid) ? (10 + bid) : 0` — all-or-nothing.

## Porting Order

1. Game engine (`blob.py` → Rust) — port all 135 tests from `legacy/game-engine/test_blob.py`
2. Entity encoder (`architecture.md §4–5`) — replaces `legacy/neural-network/encode.py`
3. Structured Entity Transformer (`architecture.md §4.3–4.6`) — replaces `legacy/neural-network/model.py`
4. MCTS with belief tracking and determinization (`prepare-migration.md §5`)
5. Training pipeline: self-play, replay buffer, trainer (`prepare-migration.md §6`)
6. Evaluation + CLI: arena, ELO, `clap` (`prepare-migration.md §7`)

## Crate Choices

`tch` (libtorch) for GPU training, `ort` (ONNX Runtime) for MCTS inference, `rayon` for self-play parallelism, `serde`+`bincode` for checkpoints, `smallvec` for MCTS child lists, `rand`+`rand_xoshiro` for determinization sampling, `clap` for CLI, `tracing` for logging.

## Verification Gates

Before considering a phase complete:
- Game engine: all 135 ported tests pass; state copy benchmarks at ~50ns
- MCTS: with 5×100 sims, top action has >2× average visit count (non-uniform signal)
- Training: policy loss drops below `ln(avg_legal_actions)` within 10 iterations; ELO exceeds 1000 within 20 iterations
- Performance: full iteration (self-play + training) completes in <60 seconds; 32-thread self-play >80% scaling efficiency

## Hardware Target

Training: Ubuntu 24.04, Ryzen 9 7950X (16C/32T), RTX 4060 8GB, 128GB DDR5. Future inference: Windows + Intel iGPU via ONNX Runtime.
