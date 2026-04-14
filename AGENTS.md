# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project State

The Rust rewrite has not started yet. This repository currently contains only planning documents and archived Python reference code. **No Rust source exists.** The next step is creating the Rust workspace and beginning Phase 1 (game engine).

## Planning Documents

Read these before making architectural decisions:

- **[development-plan.md](development-plan.md)** — The authoritative development plan. Contains the complete Structured Entity Transformer specification (Sections 2–3), MCTS design (Section 4), training pipeline (Sections 5–7), and evaluation/deployment (Sections 6–8). Use this as the single source of truth for all design decisions.
- **[prepare-migration.md](prepare-migration.md)** — Rust rewrite plan: game rules (§3), MCTS algorithm (§5), training pipeline (§6), crate recommendations (§10), verification checklist (§13).
- **[legacy/](legacy/)** — Archived Python source. Read-only reference. Never modify.

## Architecture Overview

The system is an AlphaZero-style MCTS+neural-network pipeline for a trick-taking card game.

### Data Flow

```
BlobState (stack, ~410 bytes)
  → Entity Encoder → variable-length token sequence (~14–58 tokens)
  → Structured Entity Transformer (1.63M params)
  → Playing head (hand card tokens → per-card scores → softmax)
  → Bidding head (CLS token → 14-dim softmax)
  → Value head (CLS token → scalar ∈ [-1, 1])
```

### Key Architectural Decisions

**BlobState** must be extended from the minimal struct in `prepare-migration.md §3.6` to include a `trick_history: [TrickRecord; 13]` field — the entity encoder requires the full ordered log of who played what card in which trick (see `development-plan.md` Session 2).

**Entity encoder** transforms `BlobState` into five token types:
- Hand cards (1–13), Played cards (0–51), Player states (3–8), Context (1), CLS (1)
- Rank, suit, and player are encoded as **raw one-hots inside per-token feature vectors**, not as learned embedding tables. Each token type has its own input projection (`Linear(30,128)` for hand, `Linear(48,128)` for played, `Linear(29,128)` for player state, `Linear(13,128)` for context — see `development-plan.md` Session 3.1), so the projection's weight matrix absorbs the rank/suit/player columns. There is no weight sharing between token types
- Played card tokens additionally index a learned chronological embedding table (`nn::Embedding(52, 128)`), added on top of the input projection
- The encoder takes a `perspective: u8` argument; MCTS always passes `state.current_player`

**MCTS** uses determinization: sample N consistent opponent hand assignments, run full tree search on each, aggregate visit counts. Arena-allocate nodes as contiguous `Vec<MctsNode>`. Start at **5×100 sims/move minimum** — fewer produces uniform visit distributions and zero learning signal (the root cause of the Python failure).

**Training loop**: self-play via `rayon` thread pool → replay buffer (3× contiguous `Vec<f32>`) → gradient updates. Value target is z-scored final score: `clip((my_score − mean) / std_dev, −1, 1)`.

**Output heads**: the playing head scores each hand card token directly (entity-native, ~4K params); the bidding head reads from the CLS token (14 bid values, separate to prevent gradient interference at positions 0–13 which overlap between bid values and card indices in a unified head).

### Card Encoding

Card index = `suit_index * 13 + rank_index`. Suits: ♠=0, ♥=1, ♣=2, ♦=3. Ranks: 2=0 through A=12. Hands stored as `u64` bitmasks.

### Scoring

`score = (tricks_won == bid) ? (10 + bid) : 0` — all-or-nothing.

## Porting Order

1. Game engine (`blob.py` → Rust) — port all 143 tests from `legacy/game-engine/test_blob.py`. **Note**: the legacy `generate_round_structure` has an off-by-one (produces `2C + num_players − 1` rounds instead of the correct `2C + num_players − 2`). The Rust port uses the correct formula; any ported test asserting round counts or round-index→cards-dealt mappings must be adjusted. See `development-plan.md` Session 1.2 for details
2. Entity encoder (`development-plan.md` Section 2) — replaces `legacy/neural-network/encode.py`
3. Structured Entity Transformer (`development-plan.md` Section 3) — replaces `legacy/neural-network/model.py`
4. MCTS with belief tracking and determinization (`development-plan.md` Section 4)
5. Training pipeline: self-play, replay buffer, trainer (`development-plan.md` Section 5)
6. Evaluation + CLI: strength tracking, `clap` (`development-plan.md` Section 6)

## Crate Choices

`tch` (libtorch) for GPU training, `ort` (ONNX Runtime) for MCTS inference, `rayon` for self-play parallelism, `serde`+`bincode` for checkpoints, `smallvec` for MCTS child lists, `rand`+`rand_xoshiro` for determinization sampling, `clap` for CLI, `tracing` for logging.

## Verification Gates

Before considering a phase complete:
- Game engine: all 143 ported tests pass (adjusted for round-structure correction); `BlobState` copy benchmarks at ~100ns (~410 B across ~6 cache lines)
- MCTS: with 5×100 sims, top action has >2× average visit count (non-uniform signal)
- Training: policy loss drops below `ln(avg_legal_actions)` within 10 iterations; win rate vs random > 55% within 20 iterations
- Performance: full iteration (self-play + training, ~177 games × 10 epochs) completes in <5 minutes on 32 threads; 32-thread self-play >80% scaling efficiency; ONNX inference <0.2 ms (batch=1, ort CPU); ONNX ↔ tch output agreement within 1e-5. (The earlier "<60 seconds" figure was a small-game sanity target; Session 6.3 reconciled it with the realistic full-iteration budget in `development-plan.md` and the benches under `blob-engine/benches/`.)

## Hardware Target

Training: Ubuntu 24.04, Ryzen 9 7950X (16C/32T), RTX 4060 8GB, 128GB DDR5. Future inference: Windows + Intel iGPU via ONNX Runtime.
