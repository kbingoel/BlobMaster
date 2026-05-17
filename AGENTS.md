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

## `total_iterations` is the absolute target iter (not a count)

[blob-train/src/main.rs](blob-train/src/main.rs) runs the train loop `while tl.iteration < total_iterations`. Iter K is processed iff `K < total_iterations` (i.e. `total_iterations` is the first iter NOT processed). Eval triggers when `iter > anchor_iter && iter % eval_interval == 0`. **Consequence: to get an eval row at "iter K", set `total_iterations = K + 1`** — for fresh runs the K+1 rule is the same as before. With `total_iterations = 10` and `eval_interval = 5` the only eval-triggering iter inside the range is iter=5; iter=10 is never reached and the saved-on-disk model is `iter_000009/` (the rolling latest). The 7.3c run set `total_iterations = 15` and got eval rows at iter 5 and iter 10 because iter=10 was inside `iter < 15`.

For resumes: `total_iterations` is still the absolute target. Resuming from `iter_000015/` with `total_iterations = 30` processes iters 16..29 — that's 14 *more* iters in this session, but the **LR schedule and the loop both agree** that the run is reaching iter 29. Resume scripts conventionally take "additional iters" as their user-facing arg and convert to the absolute target internally (`target = latest_iter + 1 + add`); see [scripts/sweep-2026-04-28-resume.sh](scripts/sweep-2026-04-28-resume.sh).

**Why this matters — Bug #2 (2026-04-28 anchor resume)**: previously `total_iterations` was a *count* and the loop was `for _ in 0..total`. Resume scripts set `total_iterations = 14` (count) for "14 more iters". But `LrSchedule::new(total_iterations)` keys the cosine span on the same field, and `LrSchedule::lr(iter, ...)` reads the **absolute** iter counter. After a resume from iter_15 the absolute iter was 16+, schedule span was 13, so `t = 16/13 > 1.0` clamped to 1.0 → cos = 0 → LR pinned at MIN_LR (1e-5) for every iter of the resume. The 14-iter resume window produced **no measurable strength gain** (iter_29 vs iter_15 head-to-head = 0.484 win rate, statistically inconclusive). Symptom of recurrence: `learning_rate` in metrics.jsonl stays at 1e-5 across multiple consecutive iters with no cosine motion. The `iteration complete` tracing line now logs `learning_rate=` so this is visible in the live log.

Resume-anchor caveat: `--resume`'ing does not preserve the from-scratch eval anchor. `try_resume` sets `anchor_iter = tl.iteration` (= the resume baseline), so post-resume in-loop evals in `strength.csv` are vs the resume baseline, not vs `iter_000000`. To compare the resumed model to the from-scratch starting point, run `blobmaster-train evaluate iter_K/model.onnx iter_000000/model.onnx` directly.

## MCTS sim budget is config-driven (since 2026-05-17)

[`adaptive_budget`](blob-engine/src/mcts.rs) returns `(num_determinizations, sims_per_determinization)` for every non-forced decision and **reads both directly from `MctsConfig`** — i.e. the `[mcts]` block of the training TOML or the `--config` TOML passed to `blobmaster-train evaluate`. Forced moves (`num_legal ≤ 1`) still short-circuit to `(1, 0)`. `min_sims_floor` is a safety net that can raise `sims` if the configured budget falls below it.

**Before 2026-05-17 this was hardcoded to `(5, 100)`** and the TOML fields were silently ignored. Any pre-2026-05-17 run logs / scripts that "set" `sims_per_determinization` were really running at 5×100; treat their declared budgets as decorative. Going forward, the values in `[mcts]` are load-bearing.

**To run an eval at a different sim budget** (e.g. the diagnostic that asks "is iter_X actually stronger than iter_Y at 2× sims, or are they just tied because 500 sims can't resolve them?"):

```bash
cp blob-train/run-<name>.toml /tmp/eval-5x200.toml
sed -i 's/^sims_per_determinization = 100$/sims_per_determinization = 200/' /tmp/eval-5x200.toml
./scripts/run-train.sh evaluate \
  --model-a checkpoints/<run>/iter_AAAAAA/model.onnx \
  --model-b checkpoints/<run>/iter_BBBBBB/model.onnx \
  --num-games 200 --num-players 5 --cards-dealt 7 \
  --config /tmp/eval-5x200.toml
```

Confirm the override took effect: the startup log line `evaluate — starting head-to-head ...` echoes `num_determinizations=` and `sims_per_determinization=`. If those don't match your TOML, the override didn't land.

**To run training at a different sim budget**, edit the run's TOML before launch (or before `--resume`). The cost is roughly linear in `dets × sims`: at 5×100 self-play is ~520s/iter on this box (mean from run-2026-05-14, see [self-play-profile.md](self-play-profile.md)); 5×200 ≈ ~1000s, pushing iter wall from ~1500s to ~2000s. Budget accordingly.

**Don't lower the budget below 5×100 without a strong reason.** The 7.3a "bucketed" schedule (60 sims at `nl=2`, 90 at `nl=3`) starved low-branching decisions and regressed strength — see [7.3b-analysis.md](7.3b-analysis.md) §5 / §7.1. The 5×100 floor is the empirical minimum for usable learning signal. The `adaptive_budget_reads_cfg` unit test in [blob-engine/src/mcts.rs](blob-engine/src/mcts.rs) pins the cfg-driven behavior; if you change the budget shape, update that test.

## Graceful exit (`STOP` file)

The training driver checks for `<checkpoint_dir>/STOP` at each iteration boundary. `touch checkpoints/<run-name>/STOP` to ask the loop to finish the current iteration (save checkpoint + export ONNX), then exit cleanly. The file is deleted on detection so the next `--resume` doesn't immediately stop again. There is no SIGINT/SIGTERM handler, so Ctrl-C still hard-kills mid-iter and loses the in-flight iteration's compute (~35-45 min on the 7950X mixed-player stack). Use STOP if you care about that work.

## Visualizing a run

Two Python scripts read a run's checkpoint directory + logs and produce plot folders under `logs/`. Both use the venv interpreter (matplotlib + numpy + onnx are pinned in `requirements.lock.txt`); the bare `python3` on the system likely won't have the right onnx version.

### Training-process dashboard — [scripts/visualize_strength.py](scripts/visualize_strength.py)

Three original strength plots (winrate vs anchor with Wilson CI bands, score differential, train losses) plus two diagnostics added 2026-05-14: per-iter convergence (combined/value loss, top-1 accuracies, `num_epochs_run`, LR cosine, policy KL + visit entropy) and wall-clock per iter (iter wall stacked with derived post-iter eval wall). The `--metrics` and `--stderr` flags are optional — without them you get just the original three plots.

```
.venv/bin/python scripts/visualize_strength.py \
  --csv      checkpoints/<run-name>/strength.csv \
  --metrics  checkpoints/<run-name>/metrics.jsonl \
  --stderr   logs/<run-name>.stderr \
  --out-dir  logs/<run-name>-progress
```

Eval wall is derived from stderr as `(T_{K+1} − T_K) − wall_clock_secs_{K+1}` between consecutive `iteration complete` log lines (the `iteration_complete` timer in [blob-train/src/main.rs](blob-train/src/main.rs) wraps `run_iteration` only — eval runs *after* that log line). The SP-vs-training split inside `wall_clock_secs` is **not** derivable from current logs; getting it requires adding `self_play_secs` / `training_step_secs` fields to `IterationMetrics` in [blob-nn/src/training_loop.rs](blob-nn/src/training_loop.rs)::`run_iteration`.

### Weight evolution — [scripts/visualize_weight_evolution.py](scripts/visualize_weight_evolution.py)

Reads every `iter_NNNNNN/model.onnx` in the checkpoint directory and produces five plots: per-step weight velocity (`||W_t − W_{t-1}|| / ||W_t||`), distance/cosine from init, weight histograms per representative layer, singular-value spectra, and init-vs-final heatmaps. Useful to diagnose dead/saturated layers and to check whether the trainer is still meaningfully moving weights late in a run.

```
.venv/bin/python scripts/visualize_weight_evolution.py \
  --checkpoint-dir checkpoints/<run-name> \
  --out-dir        logs/<run-name>-weight-evolution
```

The script loads every checkpoint's full state into memory (~11 MB per iter × N iters), so on a 230-iter run expect a few GB of RAM. `--max-layers-histogram` (default 12) controls how many representative layers the histogram/SVD/heatmap panels sample — increase for more detail, decrease if the plots get too crowded.
