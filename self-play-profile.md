# Self-play profiling — 2026-04-24

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
