# Async Training — V2 Design Notes

Input for a future implementation plan. The goal is a **standalone V2 training pipeline** built around an actor–learner async architecture from day one. V1 (`blob-train` + `blob-nn::training_loop`) remains untouched as a reference; V2 may copy state/encoder/buffer modules but does not depend on V1's driver, LR schedule, or eval orchestration.

## 1. Why async — measured baseline (post-2026-05 stack)

Hardware: AM5 7950X (16C/32T) + RTX 4060 8GB. V1 baseline comes from two consecutive runs on the current stack (Cause 1–3 MCTS fixes, fixed [5,7] player distribution, `num_games=118`, `target_batch=5`, batch=512, buffer=500k, `epoch_early_stop_rel=0.005`, 32 self-play threads):

- **`run-2026-05-12`** — Cause 1–3 fixes, no forced-move shortcut. Steady-state (iters 11–15, mtime-derived): **~960 s/iter**.
- **`run-2026-05-13`** — same config + forced-move shortcut. Steady-state (iters 12–13, excluding the eval-overhead spike on iter 11): **~785 s/iter**. Mean of iters 11–13 incl. eval = ~825 s.

The forced-move shortcut accounts for the gap: 960 − 785 ≈ **~175 s/iter saved**, all on the self-play half (the shortcut never touches the training step). A second anchor comes from `num_epochs_run` in `metrics.jsonl`: at steady state, `epoch_early_stop_rel = 0.005` fires aggressively and **only 2 of the configured 10 epochs run** (see `checkpoints/run-2026-05-13/metrics.jsonl`, iters 5–13 all log `num_epochs_run = 2`).

The third anchor is the per-step training time. Old `run-2026-05-06` actually ran **10 epochs almost every iter** (147 rows in `metrics.jsonl`; only iters 2/13/139 dipped to 4/7/8). 47 min/iter wall with 10 × 977 × 263 ms = ~42.8 min of training implies **~263 ms/step** and only ~4 min of SP+ONNX per iter on the old stack. Per-step time is invariant across stacks — same 1.6M-param model, same batch=512, same RTX 4060. **The 263 ms/step is GPU-bound forward/backward**: `nvidia-smi` shows 95–100% utilisation sustained through the training phase, which is incompatible with the "30–50 ms GPU + CPU-encode tail" framing that earlier drafts of this doc carried. The 1-thread CPU re-encode is fine because the GPU saturates anyway. Applying that rate:

| Phase | V1 sync wall (measured, current stack) | Hardware engaged |
|---|---|---|
| Self-play | **~270 s** (post-shortcut); ~445 s pre-shortcut | 32 CPU threads at ~100%, GPU idle |
| Training (2 × 977 × 263 ms) | **~514 s** (8.6 min) | GPU at **95–100%** sustained, 1 CPU thread for re-encode (not the bottleneck) |
| ONNX export | ~25 s | mixed CPU/GPU, Python subprocess |
| **Total** | **~785 s** (13.1 min) | each half idles the other half's hardware |

A GPU-utilization sample during iter 13 of `run-2026-05-13` confirmed the order: GPU stayed at 95–100% right up to the iter-complete log line, then dropped to ~2% — so training is the **last** phase, and the GPU is saturated during it.

> **Current-stack SP duty cycle: ~34%** (271 / 785). On the old `run-2026-05-06` stack the SP duty cycle was actually **~9%**, not 34% — old iters were dominated by the 10-epoch training half (~43 min of 47 min). Earlier drafts misread this as a "both halves shrunk proportionally" story; the real decomposition is:
>
> - Iter wall: 47 min → 13 min (~3.6× drop)
> - Epochs at early-stop: **10 → 2 (~5× drop)** — accounts for essentially all of the iter-wall shrink. Cause 1–3 MCTS fixes give the trainer a much sharper signal, so loss bottoms out within 2 passes of the buffer instead of needing the full 10. (LR-schedule compression from the short 16-iter `total_iterations` is a secondary contributor at iter 8+ but doesn't explain the early iters: iter 1 of `run-2026-05-13` already runs only 3 epochs at LR=2.97e-4, while the old run at the same LR ran 10.)
> - SP per iter: **~4 min → ~4.5 min (~unchanged)**. The forced-move shortcut saved ~175 s on top, but the underlying SP wall didn't move much — fixed [5,7] vs mixed-player only trimmed examples-per-iter ~17% (54k → 44.8k), not the SP wall.
> - **Per-step training time: unchanged at ~263 ms** — GPU-bound on the 4060.
>
> The "reclaim idle CPU during training" framing for V2 is **stronger** under the current stack than the old: SP duty cycle is now 34% (not 9%), so freeing the trainer-phase CPU yields a real ~3× SP throughput multiplier. The old stack had nothing to gain from continuous SP because training, not SP, was the bottleneck.
>
> *Caveat — the SP/training split is still derived from `num_epochs_run × 977 × 263 ms`, not directly measured. Adding a `tracing::info_span!("self_play")` / `("training_step")` pair around the two halves is a ~10-line change and would replace the derivation with a single number. Recommended before V2 commits to engineering effort.*

## 2. Throughput model — refreshed numbers, same shape

| Metric | V1 sync (post-shortcut, current stack) | V2 async (continuous SP + continuous trainer) |
|---|---|---|
| SP duty cycle | **~34%** (271 / 785) | ~100% |
| GPU duty cycle | ~65% (514 / 785) | ~85–95% (trainer GPU-bound at 263 ms/step) |
| SP rate during SP phase | 44,840 examples / 271 s ≈ **9.9k/min** at 32 threads (~5.7 ex/s/thread) | same per-thread rate, scaled to 30T |
| Examples produced per 13 min | 44,840 | **~135k** at 30T (~3.0×) |
| Sample-uses per 13 min | 1.0M (2 epochs × 977 × 512) | ~1.5M (3.80 steps/s × 512 × 780 s; trainer GPU-bound) |
| Buffer life (500k cap) | 11.2 iters ≈ 2.4 hr | **~48 min** (~3× faster fill rate) |
| **Reuse per example over buffer life** | **~22 uses** (2 epochs × 11.2 iters) | **~11 uses** (trainer step rate fixed at 263 ms; see §2.1) |

Headline: the **~3× SP-throughput multiplier from V1 → V2 is intact** against the current stack — V1's ~34% SP duty cycle goes to ~100% in V2. (The earlier "essentially the same as the old (~34%)" framing was based on a wrong old-stack number — the old run was actually ~9% SP-duty, training-bound at 10 epochs/iter.)

Two specific changes from earlier drafts of §2:

1. **Reuse-per-example dropped from ~93 (old stack, 10 × 9.3 iters) to ~22 (current stack, 2 × 11.2 iters)** because early-stop now fires at 2 epochs instead of running the full 10. That's a structural win the early-stop heuristic delivered for free — *V1 already trains on each example roughly one-quarter as often as the old run did*. V2 brings this further to **~11 reuses**: trainer rate is GPU-bound at ~3.80 steps/s, so over one ~48 min buffer-life it consumes ~5.66M sample-uses against a 500k buffer.
2. **The V2 trainer is GPU-bound, not "continuously runnable as fast as you want."** The 263 ms/step is what the 4060 delivers at batch=512 on the current 1.6M-param model with 95–100% GPU utilisation. V2 doesn't speed the trainer up; it just keeps it busy 100% of the time instead of the 66% it manages in V1's training phase. The case for V2 is therefore SP-side (3× throughput) and freshness-side (§3), not trainer-side.
ö
## 2.1. Reuse-per-example is the quality axis

The headline `~22 uses → ~11 uses` in §2's table is the more important half of the case. Two effects:

1. **Less overfit-on-stale-data.** At 22 reuses per example (V1), each example's gradient signal has been blended into the optimizer ~22 times before it leaves the buffer. V2 cuts that to ~11. Modern AlphaZero implementations target single-digit reuses; V2 lands at the upper edge of that range, V1 is still well above. The 10→2 epoch early-stop is itself partial evidence — by the third epoch the loss isn't moving on most of the buffer, because most of the buffer has already been trained on many times.
2. **Younger average example age.** With buffer life dropping from 2.4 hr → ~48 min under V2, the median example the trainer sees is generated against a model that is **less stale** by wall clock — typically 2–3 model-versions old in V2 vs 5–6 in V1. This both raises the quality of the visit-distribution target (newer model has stronger priors → better MCTS targets) and shrinks the off-policy gap between the data source and the trainer's current network.

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
- **CUDA kernel launch latency** is currently invisible (~1–4 ms / step at our shape, vs ~263 ms step time on the current 1.6M-param model at batch=512). The 263 ms is GPU-bound forward/backward on the 4060 (sustained 95–100% utilisation), not CPU-encode-bound — V2 can't reclaim time here by parallelising encode. Kernel-launch latency would become visible only if we drove batch size much smaller. Not a concern for V2 at batch=512.
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

- [ ] **Wall-clock to a fixed strength target is ≤ 60% of V1's time** on the same hardware (the ~2× target). Under the post-2026-05 stack, this should be achievable from the ~3× SP-throughput multiplier alone (V1 SP duty cycle ~34% → V2 ~100%), provided the continuous-fresh-data training (§2.1) breaks V1's 2-epoch early-stop ceiling rather than over-fitting to fresh-but-still-stale data. The bracket is 1.5× (compute-only view) to 2.7× (data-only view). Reference V1 wall for a 200-iter target run on the current stack: **~43.5 h** (~8000 s for the partial-buffer ramp on iters 0–10, plus 189 × 785 s steady state).
- [ ] GPU utilization sustained ≥ 90% over a multi-hour run (`nvidia-smi` sampling).
- [ ] CPU utilization sustained ≥ 28 of 32 threads at >80% over the same window.
- [ ] STOP file → clean exit with serialized buffer + checkpoint in ≤ 30 sec.
- [ ] Resume from STOP produces no measurable loss spike (no cold-buffer regression).
- [ ] Eval-vs-anchor pipeline produces results comparable to V1's iter-keyed eval (same Wilson methodology, just step-keyed cadence).
- [ ] **Reuse-per-example metric is logged and tracked.** V2 only justifies itself fully if this drops materially below V1's ~22, so make it observable. Emit `reuse_estimate = (training_sample_uses_so_far) / (examples_pushed_so_far)` in `metrics.jsonl` and watch it stabilize during the warmup phase.
