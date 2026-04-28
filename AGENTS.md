# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project State

The Rust rewrite is well underway. Sections 1–6 of [development-plan.md](development-plan.md) (game engine, encoder, transformer, MCTS, training, evaluation) are complete. Section 7 (training) is at **Session 7.4 closed** as of 2026-04-28 — the 2026-04-27 overnight battery ruled out the three remaining 7.4-class levers (Muon optimizer converges to AdamW-only by iter 9; INT8 fails by 5–11pp across all 10 calibration variants; `num_determinizations > 5` is flat-to-regressive). 7.5 ships with FP32 / Stage-1 batched MCTS / AdamW / `enable_muon = false` / `target_batch = num_determinizations = 5` at ~9 min/iter ⇒ ~15 h for the 100-iter mixed-player run. Section 8 (fine-tuning) is next. See [development-plan.md](development-plan.md) for session-by-session status and [self-play-profile.md](self-play-profile.md) for performance baselines.

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

**MCTS** uses determinization: sample N consistent opponent hand assignments, run full tree search on each, aggregate visit counts. Arena-allocate nodes as contiguous `Vec<MctsNode>`. Start at **5×100 sims/move minimum** — fewer produces uniform visit distributions and zero learning signal (the root cause of the Python failure). MCTS leaf eval uses cross-determinization batching (Session 7.4c stage 1): the lockstep driver in [blob-engine/src/mcts.rs](blob-engine/src/mcts.rs) round-robins across all 5 dets and issues one `evaluate_batch(B=5)` per outer step. Within-tree virtual-loss batching (`target_batch > num_dets`) is implemented but parked — the 2026-04-27 sweep showed per-call ONNX cost rises super-linearly past `num_dets` on the 7950X, making `target_batch = num_determinizations = 5` the per-game-wall optimum. See [self-play-profile.md](self-play-profile.md) for the performance breakdown driving these defaults.

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
- Performance: ONNX inference <0.2 ms (batch=1, ort CPU, 1T); ONNX ↔ tch output agreement within 1e-5. The original "32-thread >80% scaling efficiency" gate proved unachievable on the 7950X — Session 7.4 measurements show 16T at 59% scaling and 32T at 22% under FP32 batch=1, with each ONNX call slowing 1.7×–4.5× as concurrent workers contend for AVX/cache. Realistic post-7.4 targets: 16T self-play >55% efficiency at batch=1 (Stage-1 cross-det batching B=5 changes the regime — at B=5 the GEMM is fatter and SMT helps, putting T=32 ahead of T=16 by ~10%, see [self-play-profile.md](self-play-profile.md)). Iteration wall-clock at 7.4c Stage 1 (T=32, target_batch=5): ~4.7 s/game × 118 games ≈ ~9 min/iter.

## Hardware Target

Training: Ubuntu 24.04, Ryzen 9 7950X (16C/32T), RTX 4060 8GB, 128GB DDR5. Future inference: Windows + Intel iGPU via ONNX Runtime.

## Runtime environment (training runs on this machine)

All long training runs go through `./target/release/blobmaster-train train ...` and need three things lined up. Skip any of them and you either get a "missing shared library" abort at startup, a silent CPU fallback, or `scripts/export_onnx.py` blowing up with `ModuleNotFoundError: torch`.

- **Pinned Python venv**: `.venv/` at the repo root, **Python 3.12.3** with `torch==2.5.1+cu124`, `onnxruntime==1.24.4`, `onnx==1.21.0`, `numpy==2.4.4`. System Python (`/usr/bin/python3`, also 3.12.3) has none of these — `scripts/export_onnx.py` is invoked as `python3` by [blob-train/src/main.rs](blob-train/src/main.rs) each iteration, so the venv must be first on `PATH`.
- **Downloaded libtorch**: `tch` + `download-libtorch` drops a libtorch tree under `target/<profile>/build/torch-sys-*/out/libtorch/libtorch/lib`. The `torch-sys-*` hash changes whenever `tch` rebuilds, so re-resolve the directory with `find` rather than hard-coding. Pinned `tch = 0.20.0` (Cargo.lock); ships libtorch 2.4-class, CUDA 12.x runtime.
- **CUDA preload**: without `LD_PRELOAD=libtorch_cuda.so`, libtorch loads CPU-only and the run silently falls back. Without `LD_LIBRARY_PATH=$LIBTORCH_DIR`, the binary fails to load at all. CUDA driver on box: 580.x; runtime carried by libtorch (12.4 per `torch.version.cuda`); `nvcc` is **not installed system-wide** — don't reach for it, the CUDA toolkit lives inside the libtorch and PyPI wheels.
- **Do NOT let `LD_PRELOAD` bleed into Python subshells.** Tch's vendored libtorch (~2.4) has a different C++ ABI than the venv's `torch==2.5.1+cu124` wheel; preloading the tch copy into Python crashes `import torch` with `undefined symbol: ...torch::jit::Graph::toString...`. When a script needs both the Rust binary (CUDA-preloaded) and Python helpers (e.g. `scripts/export_onnx.py`, `scripts/int8_levers.py`), wrap python invocations in `( unset LD_PRELOAD; python3 ... )`. The training driver itself is fine — `blobmaster-train` invokes the venv python without inheriting LD_PRELOAD because the export call goes through `Command::new` which builds its env explicitly.

Canonical launch template:

```bash
cd /home/kbuntu/Documents/Github/BlobMaster
LIBTORCH_DIR="$(find target/release/build -maxdepth 6 -type d -name lib -path '*/libtorch/libtorch/lib' | head -n1)"
PATH=".venv/bin:$PATH" \
LD_LIBRARY_PATH="$LIBTORCH_DIR:${LD_LIBRARY_PATH:-}" \
LD_PRELOAD="$LIBTORCH_DIR/libtorch_cuda.so" \
RUST_LOG=info \
./target/release/blobmaster-train train \
  --config blob-train/config.sample.toml \
  --checkpoint-dir checkpoints/<run-name>
```

`scripts/README.md` has the same incantation; keep them in sync when re-rooting. GPU on this box is a single RTX 4060 (`cuda:0`); `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` before launching if another run might be resident.

## `total_iterations` off-by-one (eval cadence trap)

[blob-train/src/main.rs](blob-train/src/main.rs) runs the train loop `for _ in 0..total_iterations`, processing iters `0..N-1`. Eval triggers when `iter > anchor_iter && iter % eval_interval == 0`. **Consequence: to get an eval row at "iter K", set `total_iterations = K + 1`.** With `total_iterations = 10` and `eval_interval = 5` the only eval-triggering iter inside the range is iter=5 — iter=10 is never reached, and the saved-on-disk model is `iter_000009/` (the rolling latest). The 7.3c run set `total_iterations = 15` and got eval rows at iter 5 and iter 10 because iter=10 fell inside `0..14`.

When validating a planned-iter-K trajectory, either (a) set `total_iterations = K + 1` up front, or (b) finish the run as planned and call `blobmaster-train evaluate` directly on `iter_K/model.onnx` vs the anchor — `--resume`'ing for one extra iter does *not* reproduce the in-loop eval cleanly because `try_resume` sets `anchor_iter = tl.iteration`, so the next eval-trigger boundary becomes the iter *after* the resume point, not the resume point itself.
