# Async Training — V2 Design Notes

Input for a future implementation plan. The goal is a **standalone V2 training pipeline** built around an actor–learner async architecture from day one. V1 (`blob-train` + `blob-nn::training_loop`) remains untouched as a reference; V2 may copy state/encoder/buffer modules but does not depend on V1's driver, LR schedule, or eval orchestration.

## 1. Why async — measured baseline

Hardware: AM5 7950X (16C/32T) + RTX 4060 8GB. V1 run `run-2026-05-06.toml`, batch=512, buffer=500k, 10 epochs/iter, num_games=118, 32 self-play threads.

Per-iter wall clock from [logs/run-2026-05-06.stderr](logs/run-2026-05-06.stderr) (steady state, iters 8–19):

| Phase | Time | Hardware engaged |
|---|---|---|
| Self-play | ~16 min | 32 CPU threads at ~100%, GPU idle |
| Training | ~30 min | GPU at **91–98%** (sampled via `nvidia-smi`), 1 CPU thread for re-encode |
| **Total** | **~47 min** | each half idles the other half's hardware |

The two phases use disjoint hardware but execute serially in V1. CUDA's async kernel queue already pipelines the single-thread re-encode behind GPU work — the GPU is genuinely the training bottleneck, not the encode loop. So the win from async is **reclaiming the 16 min of idle-GPU + idle-31-CPU-threads**, not from speeding up training itself.

## 2. Throughput model

Per-wall-minute steady-state rates:

- **Self-play production**: 53k examples / 16 min = **3.3k examples/min** at 32 threads
- **Training consumption**: 977 steps × ~7 effective epochs × 512 samples / 30 min ≈ **117k sample-uses/min** at GPU saturation

| Metric | V1 sync | V2 async (30 self-play threads + GPU continuous) |
|---|---|---|
| Examples produced per 47 min | 53k | 145k (**2.7×**) |
| Sample-uses per 47 min | 3.5M | 5.5M (**1.57×**) |
| GPU duty cycle | 30/47 ≈ 64% | ~95% |
| Buffer life (500k cap) | 9.4 iters ≈ 7.4 hr | 161 min ≈ 2.7 hr |
| **Reuse per example** | **~66 uses** | **~38 uses** |

**Expected wall-clock learning speedup: ~2×**, bracketed by the compute-only view (1.57×) and the data-only view (2.7×). The data axis additionally improves *quality*: fewer reuses per example reduces overfit-on-stale-data risk and removes the cold-buffer regression class that motivated `cold_buffer_post_resume_epochs` (V1 [training_loop.rs:653-662](blob-nn/src/training_loop.rs#L653-L662)).

## 3. Off-policy lag is acceptable

AlphaZero-style targets (MCTS visit distributions + final z-scored outcomes) are computed by a tree search using a past model — they are effectively supervised labels, not bootstrapped from the current network's predictions. V1 already tolerates 9 iters of staleness in its buffer. Async adds at most 1–2 iters of additional lag (between an example being generated and the next model publish reaching self-play workers). No theoretical concern; matches the precedent from public AlphaZero-style implementations.

## 4. Architecture sketch

```
                         ┌────────────────────────────────┐
                         │   Shared ReplayBuffer (Arc)    │
                         │   500k capacity, FIFO, RW lock │
                         └──┬───────────────────────┬─────┘
                            │ push(state,π,z,phase) │ sample_batch(512)
                            │                       │
   ┌────────────────────┐   │                       │   ┌────────────────────┐
   │ Self-play workers  │───┘                       └───│ Trainer thread     │
   │ rayon pool, ~30T   │                               │ 1 thread + GPU     │
   │ each owns 1 ONNX   │                               │ libtorch tch-rs    │
   │ session, polls     │       ┌──────────────┐        │ loop: sample →     │
   │ AtomicU64 model    │◀──────│ Model        │◀───────│ encode → fwd/bwd → │
   │ version, hot-swaps │       │ publisher    │        │ optimizer.step →   │
   │ on game boundary   │       │ writes ONNX  │        │ maybe publish →    │
   └────────────────────┘       │ + bumps ver  │        │ maybe eval         │
                                └──────────────┘        └────────────────────┘
```

Key actors:

1. **Self-play pool** — rayon thread pool of ~30 workers. Each worker owns one `OnnxEvaluator`, plays games continuously, pushes decisions into the shared buffer. On every game boundary, checks the model-version `AtomicU64` and reloads its ONNX session if bumped. (V1 pattern in `blob-nn/src/engine.rs::self_play_iteration`.)
2. **Trainer** — one dedicated thread; owns the libtorch model + optimizer; loops `sample → encode → forward/backward → step`. Publishes a fresh ONNX every K gradient steps.
3. **Publisher** — sub-step of the trainer loop. Writes `model.onnx.tmp`, atomic rename to `model.onnx`, increments the version atomic. Workers swap on next game start, never mid-game.
4. **Eval orchestrator** — periodic, triggered by `global_step` milestones (every N steps) rather than iter boundaries. Spawns its own ONNX-session pool against the latest model and a frozen anchor.

## 5. What changes vs V1

Concept changes (not just code):

- **No iterations.** Replace iter as the schedule unit with `global_step` (gradient steps) and/or wall time. Affects:
  - LR schedule: V1 [`LrSchedule`](blob-nn/src/train.rs#L127-L185) keys cosine off `(iteration, step_in_run)`. V2: cosine over `total_global_steps`, warmup over first N steps, gated by `buffer.len() >= warmup_threshold` instead of "iter 0 only."
  - Eval cadence: every N gradient steps or every N minutes.
  - Anchor promotion: same Wilson-lower-95 logic, just keyed off step.
  - Metrics emission: `metrics.jsonl` rows tagged `global_step` not `iteration`.
- **Buffer is shared concurrently.** V1 `ReplayBuffer` ([blob-engine/src/replay.rs](blob-engine/src/replay.rs)) is single-owner. V2 needs concurrent push (many writers) + concurrent sample (one reader). Options:
  - `Arc<RwLock<ReplayBuffer>>` — simplest, fine if push is cheap. Risk: writer starvation under many self-play threads.
  - Lock-free MPMC ring (e.g., `crossbeam`) wrapping the existing FIFO — preferred. Index allocation can be a single atomic; per-slot writes are independent.
- **Pause/resume becomes trivial.** Drop a `STOP` file → trainer finishes its current step, serializes buffer + model + step counter, exits. Self-play workers drain in-flight games, then exit. On resume, no cold-buffer regression — the buffer is already at full size and the trainer just continues. The `cold_buffer_post_resume_epochs` mitigation is no longer needed.
- **Reproducibility weakens.** Thread timing affects sampling order, so two runs from the same seed don't bit-match. Document and accept.
- **No per-iter publish boundary for the model.** Model files published on a step cadence. Eval, anchor history, and "model.onnx" hot-swap all key off the version atomic.

What stays the same — copy from V1 mostly verbatim:

- `blob-engine` entirely: state, bidding, playing, MCTS, encoder, ONNX evaluator. No changes.
- `blob-nn` model definition, heads, transformer, input pipeline.
- `ReplayBuffer` storage layout (raw `BlobState` + sparse policy + value + phase) and `sample_batch` semantics — only the locking/concurrency wrapper is new.
- `bid_train_batch` / `play_train_batch` ([training_loop.rs:318-409](blob-nn/src/training_loop.rs#L318-L409)) — unchanged in semantics.
- Loss accumulators and metrics finalization — minor reshape from per-iter to per-window aggregates.
- ONNX export pipeline.

## 6. Concurrency details worth pinning down before coding

These are the design questions whose answers shape the API. They should be resolved in the implementation plan, not deferred:

1. **Buffer concurrency primitive.** RwLock vs lock-free ring. Recommend lock-free ring; benchmark sample-batch latency under 30-writer load before committing.
2. **Backpressure.** What happens if the trainer falls behind self-play and the buffer is being overwritten faster than it can be sampled? Unlike V1 (where production stops during training), in V2 production never stops. Either accept (buffer is FIFO, oldest wins) or rate-limit producers. Recommend accept — at our throughput ratio (~38 uses/example) we're nowhere near losing examples before they're trained on.
3. **Model hot-swap semantics.** Atomic file rename + `AtomicU64` version. Worker reads version at start of each game; if changed, reload ONNX session before first decision. Never mid-game (avoids inference inconsistency within a single game).
4. **Trainer warmup gate.** `buffer.len() >= 50_000` (≈one V1-iter's worth) before first gradient step. LR warmup proceeds normally on top of that gate.
5. **Eval isolation.** Eval should not steal CPU from self-play or GPU from training. Options: (a) eval on CPU only via ONNX, (b) pause training for eval window. Recommend (a) — eval already runs CPU-only via ONNX in V1.
6. **Shutdown semantics.** STOP file → set `should_stop` atomic → trainer exits after current step (serializing buffer + checkpoint) → workers exit on next game boundary. Bound on shutdown wait: ≤ one game (~5 sec at typical decision rate).
7. **Metrics cadence.** V1 emits one `metrics.jsonl` row per iter. V2 emits one row per N gradient steps. Choose N so cadence ≈ 1/min for log readability.

## 7. Code reference index (V1, for porting)

| V2 component | Copy / adapt from |
|---|---|
| Game state, bidding, playing | [blob-engine/src/state.rs](blob-engine/src/state.rs), [bidding.rs](blob-engine/src/bidding.rs), [playing.rs](blob-engine/src/playing.rs) |
| MCTS | [blob-engine/src/mcts.rs](blob-engine/src/mcts.rs) |
| Encoder (state → entity tokens) | [blob-engine/src/encoder.rs](blob-engine/src/encoder.rs) |
| ONNX inference for self-play | [blob-engine/src/onnx.rs](blob-engine/src/onnx.rs), [blob-nn/src/engine.rs](blob-nn/src/engine.rs) |
| Replay buffer (storage, not concurrency) | [blob-engine/src/replay.rs](blob-engine/src/replay.rs) |
| Model, heads, transformer | [blob-nn/src/model.rs](blob-nn/src/model.rs), [heads.rs](blob-nn/src/heads.rs), [transformer.rs](blob-nn/src/transformer.rs) |
| Train batch construction | [blob-nn/src/training_loop.rs:318-409](blob-nn/src/training_loop.rs#L318-L409) |
| Single train step (forward/backward/optimizer) | [blob-nn/src/training_loop.rs:582-632](blob-nn/src/training_loop.rs#L582-L632) (extract the GPU-step body) |
| LR schedule shape (cosine + warmup) | [blob-nn/src/train.rs:127-185](blob-nn/src/train.rs#L127-L185) — re-key off `global_step` |
| Eval vs anchor (Wilson lower-95, promotion) | [blob-train/src/main.rs](blob-train/src/main.rs) `maybe_promote_anchor` |
| STOP file pattern | [blob-train/src/main.rs:336-351](blob-train/src/main.rs#L336-L351) |
| ONNX export | existing `export_onnx` path used at iter end in V1 driver |

## 8. Risks and unknowns

- **Buffer contention under 30 writers + 1 reader.** Untested at our throughput. Bench before committing to a primitive.
- **CUDA kernel launch latency** is currently invisible (~1–4 ms / step at our shape, vs ~250 ms step time). It would become a bottleneck only if we drove batch size much smaller. Not a concern for V2 at batch=512.
- **Model-swap race.** If a worker mid-game holds an old ONNX session reference and the file is renamed under it: ONNX Runtime keeps its mmap valid; safe on Linux. Confirm on the actual platform.
- **Eval throughput interference.** Even CPU-only eval steals cores from self-play during eval windows. Acceptable if eval is short relative to inter-eval intervals; quantify before deciding.
- **Determinism.** Lost. Document that V2 runs are not bit-reproducible.

## 9. Out of scope for V2 v1

Things explicitly *not* in the first cut, to keep scope finite:

- GPU-batched self-play inference (already evaluated and abandoned — see memory `project_batching_status.md`).
- Distributed multi-machine training.
- On-policy correction (importance sampling against model version drift).
- INT8 self-play in the async loop. Add later if the FP32 ONNX self-play rate becomes the bottleneck.

## 10. Success criteria

V2 is "done" when:

- [ ] Wall-clock to a fixed loss target is ≤ 60% of V1's time on the same hardware (the ~2× target).
- [ ] GPU utilization sustained ≥ 90% over a multi-hour run (`nvidia-smi` sampling).
- [ ] CPU utilization sustained ≥ 28 of 32 threads at >80% over the same window.
- [ ] STOP file → clean exit with serialized buffer + checkpoint in ≤ 30 sec.
- [ ] Resume from STOP produces no measurable loss spike (no cold-buffer regression).
- [ ] Eval-vs-anchor pipeline produces results comparable to V1's iter-keyed eval (same Wilson methodology, just step-keyed cadence).
