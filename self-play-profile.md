# Self-play profiling

## Current optimum (2026-04-27, post Session 7.4c)

| knob | value | source |
|---|---|---|
| `self_play.num_threads` | **32** | Stage-1 B=5 thread sweep (T=4..32 → T=32 wins at 4.770 s/game) |
| `mcts.num_determinizations` | **5** | Section 4 baseline (unchanged) |
| `mcts.sims_per_determinization` | **100** | 7.3c-validated |
| `mcts.target_batch` | **5** = `num_determinizations` | Stage-2 sweep (B∈{5,8,12,16}) — tb=5 is the bowl bottom |
| `self_play.use_int8` | **false** | 7.4b INT8 quality gate failed; deferred levers parked |

This config gives **4.696 s/game** on the iter_000014 model (5P7C, flat 5×100 MCTS) — **1.544× faster** than the B=1 T=16 7.3c reference (7.250 s/game) and the post-Session-7.4 ship target. Headline wall-clock at 118 games/iter ≈ **9.2 min/iter**.

The remaining 7.5-headroom levers, in expected-ROI order: (1) raise `num_determinizations` (cross-det batching free, needs a quality study), (2) revisit 7.4b INT8 with the deferred levers (S8S8 / entropy calibration / sensitivity-driven body exclusions; ~1.4× on top), (3) `target_batch > num_dets` only if the model widens to d_model ≥ 256.

Reproduce the current optimum:

```bash
LIBTORCH_DIR="$(find target/release/build -maxdepth 6 -type d -name lib -path '*/libtorch/libtorch/lib' | head -n1)"
LD_LIBRARY_PATH="$LIBTORCH_DIR:${LD_LIBRARY_PATH:-}" \
RUST_LOG=info \
./target/release/blobmaster-train profile \
  --model checkpoints/7.3c-run/iter_000014/model.onnx \
  --config blob-train/config.sample.toml \
  --games-per-thread 5 \
  --num-threads 32 \
  --num-players 5 \
  --cards-dealt 7
```

Detailed run history follows.

---

## Initial profiling — 2026-04-24

## Setup

Rust-side profiler added behind `blob_engine::profiling` (global `AtomicU64` buckets, toggled on only for the `blobmaster-train profile` subcommand — inert in normal training). Instrumented hot paths: `encode`, `determinize`, `mcts_search`, `expand`, `backprop`, `OnnxEvaluator::run_encoded` (split: tensor build / `sess.run` / output extract), `OnnxEvaluator::from_file`, `play_one_game_with_stats`.

All runs shared:

| | |
|---|---|
| Model | `checkpoints/7.3c-run/iter_000014/model.onnx` (1.63M-param Structured Entity Transformer) |
| MCTS | 5 determinizations × 100 sims, `c_puct=1.5`, floor=60 |
| Game | 5 players, 7 start cards, full game to completion |
| Decisions/game | 380 |

Configs run — all **5 games per thread** for comparable startup-cost amortization:

| label | threads | games/thread | total games | invocation |
|---|---|---|---|---|
| **1T** | 1 | 5 | 5 | `profile --num-threads 1 --games-per-thread 5` |
| **16T** | 16 | 5 | 80 | `profile --num-threads 16 --games-per-thread 5` |
| **32T** | 32 | 5 | 160 | `profile --num-threads 32 --games-per-thread 5` |

Reproduce (run from repo root after `cargo build --release -p blob-train`; substitute `<N>` with 1/16/32):

```bash
LIBTORCH_DIR="$(find target/release/build -maxdepth 6 -type d -name lib -path '*/libtorch/libtorch/lib' | head -n1)" \
LD_LIBRARY_PATH="$LIBTORCH_DIR:${LD_LIBRARY_PATH:-}" \
RUST_LOG=info \
./target/release/blobmaster-train profile \
  --model checkpoints/7.3c-run/iter_000014/model.onnx \
  --games-per-thread 5 \
  --num-threads <N> \
  --num-players 5 \
  --cards-dealt 7
```

## Results

| metric | 1T | 16T | 32T |
|---|---:|---:|---:|
| wall clock (s) | 342.1 | 579.0 | 1,550.1 |
| per-game wall (s) | 68.4 | **7.24** | 9.69 |
| per-decision wall (ms) | 180 | **305** | 816 |
| throughput (games / wall-sec) | 0.0146 | **0.138** | 0.103 |
| speedup vs 1T | 1.00× | **9.45×** | 7.05× |
| scaling efficiency (vs 1T) | 100% | **59%** | 22% |
| per-ONNX-call (µs) | 780 | **1,305** | 3,476 |
| ONNX slowdown vs 1T | 1.00× | **1.67×** | 4.46× |
| ONNX calls | 435,710 | 6,898,086 | 13,863,067 |
| ONNX share of thread time | 99.3% | 97.2% | 97.2% |
| `encode` avg (µs) | 0.95 | 2.36 | 8.28 |
| `expand` avg (µs) | 1.46 | 3.86 | 13.71 |
| `onnx_tensor_build` avg (µs) | 1.27 | 2.66 | 8.66 |
| `determinize` avg (µs) | 1.02 | 2.91 | 5.63 |
| `session_construction` avg (ms) | 44 | 163 | 385 |

Buckets are nested — ONNX_* is a slice of MCTS_SEARCH, which is ~100% of GAME_TOTAL. Non-inference work is <3% of thread time in every config.

## Conclusion

**97%+ of self-play thread time is in `OnnxEvaluator::run_encoded`'s `sess.run()` call.** No Rust-side hot path (encode, determinize, MCTS bookkeeping, backprop) costs enough to matter.

**Scaling collapses from 1→32 threads because each individual ONNX call gets 4.46× slower under SMT load.** The 7950X is 16C/32T; 32 rayon workers put two threads per physical core, contending for AVX/FP units in the transformer forward pass (despite each worker owning its own `ort::Session` with `intra_op_num_threads=1`). At 16 threads the per-call slowdown drops to 1.67×, and scaling efficiency roughly triples (22% → 59%).

## Recommendation

Use **16 threads, not 32**, for self-play on this box. It is +34% faster per wall-second at half the CPU budget. Extrapolated to 118-game iterations: self-play drops from ~1,150 s → ~855 s (~5 min saved per iteration).

The `GATES.md` "32-thread self-play >80% scaling efficiency" target is unachievable on this hardware/model combination; 16-thread >60% is the realistic bar.

Secondary wins still available, independent of thread count:
- Reuse `OnnxEvaluator` per thread instead of per game (saves ~20 s/iter — session construction is currently called 118× instead of 16×).
- Try `.with_inter_threads(1)` on the Session builder to stop ORT's default inter-op pool from oversubscribing.

## Follow-ups (2026-04-24)

**Applied — reuse `OnnxEvaluator` per worker, not per game.** `blob-nn/src/engine.rs` now uses `rayon::iter::ParallelIterator::map_init` so each worker constructs one `OnnxEvaluator` when it picks up a chunk and reuses it across every game in that chunk. `session_construction` count per iteration drops from ~118 to a small multiple of `num_threads`, and the allocator no longer churns ~10 MB of weight buffers every game. End-to-end `runs_iteration_if_model_available` test passes against `checkpoints/7.3c-run/iter_000014/model.onnx`.

**Not applied — `.with_inter_threads(1)`.** Investigated and dropped. ORT's default execution mode is `Sequential` (the `ort` 2.0 crate defaults `with_parallel_execution` to `false`), and the crate's own docstring on `with_inter_threads` states *"This has no effect when the session execution mode is set to `Sequential`."* The inter-op pool is never used by our sessions — there is nothing to constrain. This would only matter if we ever called `with_parallel_execution(true)`, which has no payoff for a batch-1 transformer with a mostly-linear graph. The 16T per-call slowdown of 1.67× confirms intra-op is already correctly constrained by `with_intra_threads(1)` at [blob-engine/src/onnx.rs:63](blob-engine/src/onnx.rs#L63); no hidden inter-op pool is spawning work.

**Open — SMT-related.** Disabling SMT in BIOS was considered: not expected to change the 16T number measurably (Linux's CFS already places 16 runnable workers one-per-physical-core), and it would penalize other workloads on the box. Equivalent effect available without a BIOS change via `RAYON_NUM_THREADS=16` and/or `taskset`. No action taken.

## Follow-ups (2026-04-24) — thread-count sweep after evaluator-reuse

After landing the per-thread `OnnxEvaluator` reuse change ([blob-nn/src/engine.rs](blob-nn/src/engine.rs)),
re-baselined T=16 and swept neighbouring thread counts to find the per-game-wall optimum.
Each row is 5 games per thread (so total games = 5 × T). Same model, MCTS, and game shape as the original profile.

Algorithm: from T=16, walk one step in each direction; on a regression vs best-so-far, take one validation
step further; stop that direction if the validation step also regresses. Script: [scripts/thread-sweep.sh](scripts/thread-sweep.sh),
raw logs: [logs/thread-sweep-2026-04-24/](logs/thread-sweep-2026-04-24/).

| threads | total games | wall (s) | per-game wall (s) | per-decision (ms) | onnx_inference avg (µs) | vs 16T |
|--------:|------------:|---------:|------------------:|------------------:|------------------------:|-------:|
| 14 | 70 | 529.2 | 7.560 | 278.5 | 1186.6 | 1.043× |
| 15 | 75 | 548.6 | 7.315 | 288.7 | 1229.7 | 1.009× |
| 16 | 80 | 580.0 | 7.250 | 305.3 | 1290.2 | 1.000× |
| 17 | 85 | 646.2 | 7.602 | 340.1 | 1355.1 | 1.049× |
| 18 | 90 | 686.6 | 7.629 | 361.4 | 1441.9 | 1.052× |

Run order (sequential, one config at a time):

```
16 (per_game=7.249700s)
15 (per_game=7.314957s)
14 (per_game=7.560087s)
17 (per_game=7.602072s)
18 (per_game=7.629394s)
```

**Direction bests:** DOWN best at T= (7.249700s/game), UP best at T=16 (7.249700s/game), baseline T=16 (7.249700s/game).

## Follow-ups (2026-04-26) — INT8 quantization (Session 7.4b)

Quantized [checkpoints/7.3c-run/iter_000014/model.onnx](checkpoints/7.3c-run/iter_000014/model.onnx) (1.63M-param FP32, 6.6 MB) to a QDQ-INT8 sibling (2.1 MB on disk, 3.1× smaller) using `scripts/export_onnx.py --int8-out ... --calibration ...`. Calibration: 500 real `EncodedState`s captured during a 1T self-play run via the new `blobmaster-train profile --dump-calibration` path. Quantization spec: QDQ format, INT8 weights (`per_channel=True`), UINT8 activations, MinMax calibration, LayerNorm/Softmax + the three output heads (play/bid/value MLPs) excluded; the heads are <1% of FLOPs but their CLS-slot logits are most-quantization-sensitive. `quant_pre_process` is run before `quantize_static` to silence NaN-scale warnings on activations whose range collapses on some branches.

Re-ran the same 16T / 5-games-per-thread / 5P7C profile against the INT8 model. Reproduce:

```bash
LIBTORCH_DIR="$(find target/release/build -maxdepth 6 -type d -name lib -path '*/libtorch/libtorch/lib' | head -n1)" \
LD_LIBRARY_PATH="$LIBTORCH_DIR:${LD_LIBRARY_PATH:-}" \
RUST_LOG=info \
./target/release/blobmaster-train profile \
  --model checkpoints/7.3c-run/iter_000014/model.onnx \
  --games-per-thread 5 --num-threads 16 \
  --num-players 5 --cards-dealt 7 \
  --use-int8
```

Raw log: [logs/int8-2026-04-26/profile-16T-int8.log](logs/int8-2026-04-26/profile-16T-int8.log).

| metric | 16T FP32 (2026-04-24) | 16T INT8 (2026-04-26) | INT8 vs FP32 |
|---|---:|---:|---:|
| wall clock (s) | 579.98 | **414.55** | **0.715× (1.40× faster)** |
| per-game wall (s) | 7.250 | **5.182** | **0.715×** |
| per-decision wall (ms) | 305.3 | **218.2** | **0.715×** |
| onnx_inference avg (µs) | 1290.2 | **908.5** | **0.704× (1.42× faster)** |
| onnx_inference share | 97.2% | 95.0% | -2.2pp |
| ONNX calls | 6,898,086 | 6,937,725 | +0.6% |
| `encode` avg (µs) | 2.36 | 4.45 | +89% |
| `expand` avg (µs) | 3.86 | 7.42 | +92% |

**Speed result:** **1.40× per-iteration speedup at 16T**, exactly at the dev-plan pass threshold (≥1.4×). The per-ONNX-call cost drops from 1.30 ms to 0.91 ms, right at the 0.9 ms ship-line and a touch above the 0.7 ms "bandwidth-bound theory confirmed" mark — consistent with the d_model=128 / 1.63M-param model being weight-bandwidth-bound but with a modest compute share that VNNI can only partly amortize. The 1.40× wall-clock speedup matching the 1.42× per-call speedup confirms self-play is still essentially a sequence of single-batch forwards, not bottlenecked by Rust-side work. The Rust-side bucket inflation (`encode`, `expand` avg µs ~doubled) is most plausibly cache pressure: with each `sess.run` returning faster, the surrounding code now runs more often per wall-second and shares L1/L2 with concurrently-running workers' decoder work.

**Quality gate FAIL:** `scripts/validate_int8.py` over the same 500 calibration states gives:

| metric | result | gate | pass |
|---|---:|---:|:---:|
| bid argmax agreement (INT8 vs FP32) | 0.848 | ≥0.95 | ❌ |
| play argmax agreement | 0.960 | (informational) | — |
| value sign agreement | 0.942 | =1.00 | ❌ |

Excluding the three output heads from quantization barely moved the needle (84.2 → 84.8% / 93.8 → 94.2%), so the corruption is upstream — error compounding through 8 transformer layers at d_model=128 is hammering the CLS representation before the heads see it. The dev plan flagged exactly this risk ("our `d_model = 128` shrinks the compute share of that gain"). Static-gate failure does **not** automatically mean an eval-win-rate regression, but the dev plan's hold-back rule (>5pp eval drop at iter 10 → revert) almost certainly trips at this static disagreement level.

**Recommendation: hold INT8 back from 7.5.** The speed prize is real (1.4× / saves ~3 min/iter at the 100-iter budget) but the static gate is well below the 95% / 100% bars and tuning didn't close it cheaply. Three independent levers remain if we want to revisit:

1. **Symmetric INT8 activations (S8S8 instead of U8S8).** The ORT transformer-quantization guide recommends this for transformers; we currently use U8S8 because VNNI prefers it. Worth a one-flag try (`activation_type=QuantType.QInt8`) before bigger surgery.
2. **Entropy calibration.** MinMax saturates on extremes; entropy minimizes the FP32-vs-INT8 KL on activations and typically helps deep transformers.
3. **Body exclusions by sensitivity.** Re-quantize layer-by-layer and find which transformer blocks contribute most agreement loss; exclude only those (likely the last 2–3, where CLS reads occur).

7.4c (batched MCTS) is independent of this and doesn't need the INT8 path to land first. Recommend pursuing 7.4c next, then revisiting 7.4b only if the combined 7.4a + 7.4c speedup falls short of the 100-iter budget.

Artifacts on disk:
- [checkpoints/7.3c-run/iter_000014/model.int8.onnx](checkpoints/7.3c-run/iter_000014/model.int8.onnx) — quantized model (kept for future tuning experiments).
- [checkpoints/7.3c-run/calibration.bin](checkpoints/7.3c-run/calibration.bin) — 500 real EncodedStates in BCAL format.
- [logs/int8-2026-04-26/](logs/int8-2026-04-26/) — quantize, validate, and 16T INT8 profile logs.

## Follow-ups (2026-04-26) — thread-count sweep at B=5 (Session 7.4c stage-1)

After landing the cross-determinization batching driver in
[blob-engine/src/mcts.rs](blob-engine/src/mcts.rs), re-swept thread counts at B=5 lockstep batched ONNX
inference. Same workload as [logs/thread-sweep-2026-04-24/](logs/thread-sweep-2026-04-24/): 5 games per thread,
fixed 5P7C, `iter_000014/model.onnx`, MCTS at flat 5×100. Speedup column compares
per-game wall against the B=1 T=16 baseline (7.2497 s/game from
[logs/thread-sweep-2026-04-24/results.csv](logs/thread-sweep-2026-04-24/results.csv)).

Script: [scripts/thread-sweep-b5.sh](scripts/thread-sweep-b5.sh) (T=4..16) and the T=20/24/32 extension run; raw logs in
[logs/thread-sweep-b5-2026-04-26/](logs/thread-sweep-b5-2026-04-26/).

| threads | total games | wall (s) | per-game wall (s) | per-decision (ms) | onnx_inference avg (µs) | speedup vs B=1 16T |
|--------:|------------:|---------:|------------------:|------------------:|------------------------:|-------------------:|
| 4 | 20 | 333.7 | 16.685 | 175.6 | 3375.1 | 0.435× |
| 6 | 30 | 360.4 | 12.012 | 189.7 | 3664.8 | 0.604× |
| 8 | 40 | 373.3 | 9.332 | 196.5 | 3820.4 | 0.777× |
| 10 | 50 | 384.2 | 7.684 | 202.2 | 3855.3 | 0.943× |
| 12 | 60 | 392.8 | 6.547 | 206.7 | 3964.3 | 1.107× |
| 14 | 70 | 415.1 | 5.930 | 218.5 | 4115.7 | 1.223× |
| 16 | 80 | 424.0 | 5.300 | 223.2 | 4289.4 | 1.368× |
| 20 | 100 | 523.8 | 5.238 | 275.7 | 5109.0 | 1.384× |
| 24 | 120 | 623.4 | 5.195 | 328.1 | 6061.0 | 1.395× |
| **32** | **160** | **763.1** | **4.770** | **401.7** | **7763.6** | **1.520×** |

**Optimum: T=32 at 1.52× per-game wall over the B=1 T=16 baseline.** Notable:

- The plan's prediction ("expect optimum to drop to ~8T") did not hold. At B=5 each ONNX call does 5× the work as a fatter GEMM (vs GEMV at B=1), so per-thread compute density is higher and SMT contention reverses sign — the second hyperthread now fills memory-stall slots inside long GEMM bursts rather than fighting for AVX units. Measured per-call ONNX cost rises sub-linearly: 4.29 ms (T=16) → 5.11 ms (T=20) → 6.06 ms (T=24) → 7.76 ms (T=32), so doubling threads only ~1.8× the per-call cost while feeding 2× the calls in flight.
- The curve from T=20 → T=24 is essentially flat (1.384× → 1.395×, +0.8%); T=32 picks up a real but modest +9% on top via SMT. Throughput is converging, not still climbing.
- **The 1.52× ceiling falls short of the stage-1 pass condition (≥1.7×).** Per the development-plan branch, stage-2 (virtual loss within one tree) is the next lever — `target_batch ∈ {5, 8, 12, 16}` raises B further per call without adding more dets. With INT8 also held back ([Session 7.4b](#follow-ups-2026-04-26--int8-quantization-session-74b) above), getting 7.5's wall-clock budget under control likely needs both stage-2 and a revisit of INT8 with the deferred levers (S8S8 / entropy calibration / sensitivity-driven exclusions).
- Visit-count parity vs the serial driver is verified by `mcts::tests::lockstep_search_matches_serial_per_det`; ORT batched-vs-serial parity by `onnx::tests::evaluate_batch_matches_serial`. Numbers above are pure throughput, not a quality regression.

## Follow-ups (2026-04-27) — `target_batch` sweep (Session 7.4c stage-2)

After landing the within-tree virtual-loss driver in [blob-engine/src/mcts.rs](blob-engine/src/mcts.rs) (the `target_batch` knob raises in-flight leaves per `evaluate_batch` past `num_determinizations` by queueing concurrent descents inside one det's tree, with `in_flight: u16` decorating each path's UCB1 selection), swept `target_batch ∈ {5, 8, 12, 16}` at **T=32** (the Stage-1 B=5 optimum) plus a **T=16 / target_batch=8** row to re-confirm SMT direction at Stage 2. Same workload as the prior Stage-1 sweep: 5 games per thread, fixed 5P7C, `iter_000014/model.onnx`, MCTS at flat 5×100. Speedup column compares per-game wall against the B=1 T=16 baseline (7.250 s/game from [logs/thread-sweep-2026-04-24/results.csv](logs/thread-sweep-2026-04-24/results.csv)).

Script: [scripts/target-batch-sweep.sh](scripts/target-batch-sweep.sh); per-config TOMLs and raw logs in [logs/target-batch-sweep-2026-04-27/](logs/target-batch-sweep-2026-04-27/).

| threads | target_batch | total games | wall (s) | per-game wall (s) | per-decision (ms) | onnx_inference avg (µs) | speedup vs B=1 16T |
|--------:|-------------:|------------:|---------:|------------------:|------------------:|------------------------:|-------------------:|
| **32** | **5** | **160** | **751.3** | **4.696** | **395.4** | **8,378** | **1.544×** |
| 32 | 8 | 160 | 766.8 | 4.793 | 403.6 | 13,488 | 1.513× |
| 32 | 12 | 160 | 886.8 | 5.542 | 466.7 | 22,200 | 1.308× |
| 32 | 16 | 160 | 1,109.1 | 6.932 | 583.8 | 36,312 | 1.046× |
| 16 | 8 | 80 | 423.1 | 5.289 | 222.7 | 7,328 | 1.371× |

**Stage-2 pass condition (≥2.5× per-iter speedup) FAILS — by a wide margin.** The best config is T=32 / target_batch=5 at 1.544×, only +1.6% over Stage-1's T=32 row (4.770 s/game, 1.520×) and below typical run-to-run noise. Notably, **`target_batch = 5 = num_determinizations` is the *degenerate* case for virtual loss**: round-robin already lands one descent on each det per outer step, so `in_flight` along any path stays at 1 and the VL bias term has no neighbouring sibling to redirect to. The "best" stage-2 config is functionally Stage 1, re-measured.

Going past `num_determinizations` actively hurts throughput on this hardware:

- **Per-call ONNX cost rises super-linearly with `target_batch`.** From tb=5 → 16 the batch grows 3.2×, but `onnx_inference avg` grows 4.3× (8.38 → 36.31 ms). At T=32 the CPU is already saturated by 32 concurrent batched forwards; padding more in-flight leaves into each call stretches per-call latency without raising aggregate throughput. The fatter-GEMM/SMT-friendly story from Stage 1's T=20→32 curve does not generalize when the GEMM grows along the *batch* dimension instead of the *thread* dimension — at tb>5 each thread's working set spills more L1/L2 weight cache between phases.
- **Virtual loss bias is a real-but-tiny tax.** tb=8 (3 in-flight leaves redirected by VL out of 8) regresses 2.0% vs tb=5 (1.513× vs 1.544×). VL pushes concurrent descents to lower-prior siblings; those leaves expand a slightly less-on-policy slice of the tree, the search visits the same nodes at a slightly worse MCTS-quality-per-sim ratio, and you pay both the GEMM-batch tax *and* the search-quality tax. tb=12 / tb=16 amplify both.
- **SMT direction unchanged at Stage 2.** T=16 vs T=32 at tb=8: 5.29 → 4.79 s/game, T=32 is +10% faster. Stage 1's "second hyperthread fills GEMM memory-stall slots" finding holds — Stage-2 in-flight pressure does not flip the sign.

**Recommendation: ship Stage 1 only. Lower the default `target_batch` from 8 → `num_determinizations` (5) and treat `target_batch > num_determinizations` as a future revisit, not a 7.5 input.** The within-tree VL plumbing is correct (visit-count budget invariant pinned by `lockstep_search_root_visit_count_matches_sim_budget`, in-flight clears pinned by `lockstep_search_clears_in_flight_at_target_batch_above_num_dets`, serial parity at tb=1 by `target_batch_one_matches_serial_per_det`), but at d_model=128 / 5 dets / batch ≤16 the cost model doesn't favour any value past `num_determinizations`. The eval-win-rate gate (5-iter parity) wasn't run because the speed gate failed — no need to spend training compute on a config we won't ship.

For 7.5 wall-clock headroom the remaining levers, in order of expected ROI:
- **Bigger `num_determinizations`** (e.g. 8). Raises the "free" cross-tree batch ceiling, doesn't add VL bias, costs more per-decision wall but more sims-per-decision is a quality win — needs a separate quality study.
- **INT8 revisit** (Session 7.4b deferred levers: S8S8 activations, entropy calibration, sensitivity-driven body exclusions). 1.40× on top of Stage 1's 1.52× → 2.13× combined, a real path to the original 7.4 throughput target.
- **`target_batch > num_dets`** stays parked unless the model grows (d_model ≥ 256 would tip the GEMM regime to actually-batch-bound, where the math finally works).

