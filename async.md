# Async Training — V2 Design Notes

Input for a future implementation plan. The goal is a **standalone V2 training pipeline** built around an actor–learner async architecture from day one. V1 (`blob-train` + `blob-nn::training_loop`) remains untouched as a reference; V2 may copy state/encoder/buffer modules but does not depend on V1's driver, LR schedule, or eval orchestration.

## 1. Why async — measured baseline (post-2026-05 stack)

Hardware: AM5 7950X (16C/32T) + RTX 4060 8GB. V1 baseline comes from two consecutive runs on the current stack (Cause 1–3 MCTS fixes, fixed [5,7] player distribution, `num_games=118`, `target_batch=5`, batch=512, buffer=500k, `epoch_early_stop_rel=0.005`, 32 self-play threads):

- **`run-2026-05-12`** — Cause 1–3 fixes, no forced-move shortcut. Steady-state (iters 11–15): **~895 s/iter**.
- **`run-2026-05-13`** — same config + forced-move shortcut. Steady-state (iters 11–13): **~785 s/iter**.

The forced-move shortcut saves a flat **~120 s/iter**, all coming off the self-play half (the shortcut never touches the training step). That gives one anchor on the SP/training split. A second anchor comes from `num_epochs_run` in `metrics.jsonl`: at steady state, `epoch_early_stop_rel = 0.005` fires aggressively and **only 2 of the configured 10 epochs run** (see `checkpoints/run-2026-05-13/metrics.jsonl`, iters 5–13 all log `num_epochs_run = 2`).

The third anchor is the per-step training time. The old `run-2026-05-06` numbers (30 min for ~7 effective epochs × 977 batches) imply **~263 ms/step** wall-clock. Per-step time is invariant across stacks — same 1.6M-param model, same batch=512, same RTX 4060, same 1-thread CPU re-encode (which is the actual bottleneck — pure GPU forward/backward is ~30–50 ms, the encode loop pads it out). Applying that rate:

| Phase | V1 sync wall (estimated, current stack) | Hardware engaged |
|---|---|---|
| Self-play | **~245 s** (post-shortcut); ~365 s pre-shortcut | 32 CPU threads at ~100%, GPU idle |
| Training (2 × 977 × 263 ms) | **~510 s** (8.5 min) | GPU at **90–97%** (sampled), 1 CPU thread for re-encode |
| ONNX export | ~30 s | mixed CPU/GPU, Python subprocess |
| **Total** | **~785 s** (13.1 min) | each half idles the other half's hardware |

A GPU-utilization sample during iter 13 of `run-2026-05-13` confirmed the order: GPU stayed at 90–97% right up to the iter-complete log line, then dropped to ~2% — so training is the **last** phase. The capture caught the last ~25 s of training; that's consistent with ~510 s of training ending at iter end (starting ~509 s into a 785 s iter).

> **SP duty cycle: ~31%** under the current stack (vs 34% on the old `run-2026-05-06` mixed-player stack). The proportions held because both halves got faster by similar factors during the 2026-05 fixes:
>
> - Iter wall: 47 min → 13 min (~3.6× drop)
> - Epochs at early-stop: ~7 → 2 (~3.5× drop) — accounts for most of the training-half shrink
> - SP per iter: ~16 min → ~4 min (~4× drop) — faster games (fixed [5,7] vs mixed) + forced-move shortcut + Cause-3 changes
> - **Per-step training time: unchanged at ~263 ms**
>
> The "reclaim idle CPU during training" framing the original async case was built on still applies. The 30%-SP-duty-cycle headline is intact; the absolute walls just shrank ~3.6× in both halves.
>
> *Caveat — the SP/training split is still derived, not directly measured. Adding a `tracing::info_span!("self_play")` / `("training_step")` pair around the two halves is a ~10-line change and would replace the derivation with a single number. Recommended before V2 commits to engineering effort.*

## 2. Throughput model — refreshed numbers, same shape

| Metric | V1 sync (post-shortcut, current stack) | V2 async (continuous SP + continuous trainer) |
|---|---|---|
| SP duty cycle | **~31%** (245 / 785) | ~100% |
| GPU duty cycle | ~65% (510 / 785) | ~85–95% |
| SP rate during SP phase | 44,840 examples / 245 s ≈ **11k/min** at 32 threads | same per-thread rate |
| Examples produced per 13 min | 44,840 | **~120k–140k** (**~2.7–3.2×**) |
| Sample-uses per 13 min | 1.0M (2 epochs × 977 × 512) | up to ~3.0M (continuous trainer, gated by per-step time) |
| Buffer life (500k cap) | 11.2 iters ≈ 2.4 hr | **~55 min** (3× faster fill rate) |
| **Reuse per example over buffer life** | **~22 uses** (2 epochs × 11.2 iters) | depends on V2 trainer rate — see §2.1 |

Headline: the **~3× SP-throughput multiplier from V1 → V2 is intact** because the new stack's SP duty cycle (~31%) is essentially the same as the old (~34%). The user-intuition "self-play will nearly triple in speed because it runs all the time and not just 30% of the time" reads against the post-shortcut numbers exactly the same way it read against the original.

Two specific changes from the original §2:

1. **Reuse-per-example is lower than the old number** (~22 vs the old ~66) because early-stop now fires at 2 epochs instead of running ~7 effective epochs. That's a structural win the early-stop heuristic delivered for free — *V1 already trains on each example one-third as often as the old async.md modeled*.
2. **Whether the V2 trainer should run "continuously" or also early-stop is now an explicit design question** (see §2.1). On the old stack the V2 trainer would clearly do more work than V1; on the new stack, V1 already self-throttles to 2 epochs and the V2-trainer-vs-V1-trainer step delta depends on what gates the early-stop heuristic in V2.

## 2.1. Reuse-per-example is the quality axis

The headline `~22 uses → unknown` in §2's table is the more important half of the case. Two effects:

1. **Less overfit-on-stale-data.** At 22 reuses per example, each example's gradient signal has been blended into the optimizer ~22 times before it leaves the buffer. Modern AlphaZero implementations target single-digit reuses; we're well above that. The 7→2 epoch early-stop is itself partial evidence — by the third epoch the loss isn't moving on most of the buffer, because most of the buffer has already been trained on many times.
2. **Younger average example age.** With buffer life dropping from 2.4 hr → ~55 min under V2, the median example the trainer sees is generated against a model that is **less stale** by wall clock — typically 2–3 model-versions old in V2 vs 5–6 in V1. This both raises the quality of the visit-distribution target (newer model has stronger priors → better MCTS targets) and shrinks the off-policy gap between the data source and the trainer's current network.

V1 trainer hitting `epoch_early_stop_rel < 0.005` after 2 epochs is not "the trainer has nothing left to learn"; it's "the trainer has nothing left to learn *from this buffer*, which is 91% data it already trained on 1–22 times." Two ways to read that:

1. **Optimistic read** — the model has converged on what the buffer offers. V2's extra training capacity would just over-fit faster.
2. **Pessimistic read** — the buffer is too stale to provide useful gradient signal beyond 2 epochs. V2's continuously-replenished buffer is the fix; the trainer would have real work to do every step.

The pessimistic read is consistent with the reuse math and matches the precedent from public AlphaZero implementations (all of which run async-style data generation, none of which hit our reuse number). The optimistic read can't be falsified inside V1, because V1 cannot present the trainer with fresh data on a step cadence.

**Two corollaries that matter for engineering priorities:**

- **SP speedups have different leverage in V1 vs V2.** V1's reuse-per-example is fixed at `num_epochs_run × (buffer_capacity / examples_per_iter) ≈ 22`, independent of how fast SP produces those examples. The forced-move shortcut cut iter wall by 14% but did **not** change V1's reuse number, because `num_games = 118` caps examples-per-iter regardless of how fast the games run. In V2 there is no per-iter cap — workers play continuously, so SP throughput translates directly into more examples-per-trainer-step, which translates directly into lower reuse. The current forced-move shortcut helps V2 even though it didn't help V1's reuse.
- **Do all SP speedups before V2, not after.** Each %-speedup of SP in V1 saves a little wall-clock but doesn't move reuse. The same speedup in V2 drops reuse proportionally and improves training quality. INT8 self-play (currently parked) would compound on top of forced-move in V2 in a way it cannot in V1.

## 3. Model-version freshness — the other quality axis

AlphaZero-style targets (MCTS visit distributions + final z-scored outcomes) are computed by a tree search using a past model — they are effectively supervised labels, not bootstrapped from the current network's predictions. So the relevant question for V2 isn't "is off-policy lag tolerable?" (defensive framing) but "how stale is the model that generated each example's MCTS target?"

V1 sync publishes a new ONNX once per iter (~13 min cadence). Across the ~11 iters an example sits in the buffer, its MCTS target was generated against a model that is now **0–11 publishes** out of date — and the trainer treats all those labels as if they were equally valid current-policy targets.

V2 async decouples publish cadence from iter cadence. The trainer can publish every K gradient steps; at K = 1k steps × ~263 ms/step, that's **~4–5 min cadence** instead of 13 min — about a **3× faster publish rate**. (Aggressive K = 200 would push cadence to ~50 s but at the cost of ONNX export overhead at every publish; tune K against measured publish wall.) Self-play workers swap on game boundaries (a few seconds), so each MCTS target is generated against a model that is typically **1–2 publishes** out of date in V2 vs ~5–6 in V1.

Buffer residency *also* shrinks under V2 from §2's table: 2.4 hr → ~55 min, a ~3× drop, because SP runs continuously instead of 31% of the time. So both freshness axes move in the same direction simultaneously:
- **Example-age axis**: median example is ~1/3 as old (55 min vs 2.4 hr).
- **Model-cadence axis**: median publish-staleness is ~1/3 as deep (1–2 publishes vs 5–6).

The publish-to-consume lag in V2 (between an example landing in the buffer and the trainer next sampling it) is bounded by 1–2 trainer steps — negligible. The matching precedent from public AlphaZero implementations is uniform: all of them run async-style data generation, for exactly the freshness reason above.

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
2. **Backpressure.** What happens if the trainer falls behind self-play and the buffer is being overwritten faster than it can be sampled? Unlike V1 (where production stops during training), in V2 production never stops. Either accept (buffer is FIFO, oldest wins) or rate-limit producers. Recommend accept — at our throughput ratio (~3× SP speedup, trainer rate unchanged) we still get ~7 reuses/example minimum in V2, so we're nowhere near losing examples before they're trained on. Re-check the ratio if SP gets materially faster again (INT8, GPU-batched eval) — that's the regime where backpressure becomes a live concern.
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
- **CUDA kernel launch latency** is currently invisible (~1–4 ms / step at our shape, vs ~263 ms step time on the current 1.6M-param model at batch=512 — wall-dominated by the 1-thread CPU re-encode, not the GPU forward/backward). It would become a bottleneck only if we drove batch size much smaller. Not a concern for V2 at batch=512.
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

- [ ] **Wall-clock to a fixed strength target is ≤ 60% of V1's time** on the same hardware (the ~2× target). Under the post-2026-05 stack, this should be achievable from the ~3× SP-throughput multiplier alone (V1 SP duty cycle ~31% → V2 ~100%), provided the continuous-fresh-data training (§2.1) breaks V1's 2-epoch early-stop ceiling rather than over-fitting to fresh-but-still-stale data. The bracket is 1.5× (compute-only view) to 2.7× (data-only view).
- [ ] GPU utilization sustained ≥ 90% over a multi-hour run (`nvidia-smi` sampling).
- [ ] CPU utilization sustained ≥ 28 of 32 threads at >80% over the same window.
- [ ] STOP file → clean exit with serialized buffer + checkpoint in ≤ 30 sec.
- [ ] Resume from STOP produces no measurable loss spike (no cold-buffer regression).
- [ ] Eval-vs-anchor pipeline produces results comparable to V1's iter-keyed eval (same Wilson methodology, just step-keyed cadence).
- [ ] **Reuse-per-example metric is logged and tracked.** V2 only justifies itself fully if this drops materially below V1's ~22, so make it observable. Emit `reuse_estimate = (training_sample_uses_so_far) / (examples_pushed_so_far)` in `metrics.jsonl` and watch it stabilize during the warmup phase.
