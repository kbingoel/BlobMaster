# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

- **Gen 1 concluded on 2026-05-17.** The final run is `checkpoints/run-2026-05-14`, 168 iterations.
  - Since 2026-10-02 only reference files remain: models for iters 0/25/125/167, the iter-167 buffer and tch weights, metrics, `strength.csv` and `signal_ratio_by_iter.csv`.
  - All other gen-1 checkpoints were deleted (gen-2.md §9).
- **Never commit per-iteration weights.** `*.onnx` is git-ignored; add a deliberate reference model with `git add -f`. `.git` is still 4.9 GB of old model blobs; the rewrite recipe is in gen-2.md §9 (it must include the `gui` branch).
- **On 2026-10-02 it was measured against a fixed rule bot and found weak.** With 5×100 search it scores −9.2 ± 2.6 points per game against 4 rule bots.
  - Root causes: a final-game value target that the value head memorized, values credited only to the seat to move at each leaf, and clipped end-of-round values.
  - The remake is planned in **[gen-2.md](gen-2.md)**. Phase 0 (yardsticks) is done except a human playtest and the deferred `.git` rewrite; next is Phase 1 (cheap correctness fixes).
- **The gen-1 planning documents were retired on 2026-10-02.** Code comments still cite them (`fix-mcts-plan.md`, `development-plan.md`, `self-play-profile.md`, `7.3b-analysis.md`); read them with `git show c6f0c2a:<file>` (gen-2.md §10).

## Documents

- **[gen-2.md](gen-2.md)** — authoritative findings, design and roadmap. Read it before any architectural decision.
- **[README.md](README.md)** — overview, game rules, repo layout.
- **[legacy/](legacy/)** — archived gen-0 Python source. Read-only; never modify.

## Judging strength — rule

**Never report progress from training losses or checkpoint-vs-checkpoint win rates alone.** Gen 1 looked healthy on both while losing to an 80-line bot. Strength claims need:
- the absolute benchmark: one model seat vs 4 rule bots (`blob-engine/src/rule_bot.rs`), with the 95% CI (command below);
- held-out (not training-buffer) losses: `val_*` vs `train_eval_*` in `metrics.jsonl` (see the gen-1 driver section).

```bash
cargo build --release -p blob-bin
./target/release/blobmaster bench <model.onnx> --mode network   # 128 deals × 5 seats, ~10 s
./target/release/blobmaster bench <model.onnx> --mode search    # 64 deals × 5 seats at 5×100, ~4.5 min
```

`bench` plays duplicate deals: every deal seed once from each seat. Its CI is over deals. The default seed is fixed, so models are compared on the same cards. Opponents default to the rule bot; `--opponent <model.onnx>` uses a checkpoint's raw policy (bots never search).

The reference is gen-1 final (`run-2026-05-14/iter_000167`): with search **−10.2 ± 2.9**, network only −12.1 ± 2.0 (`bench`, 2026-10-02). The older non-duplicate `diagnostics match` numbers were −9.2 ± 2.6 and −13.5 ± 1.9. The tool uses every core, so don't run it next to a training run.

## Architecture (current code = gen 1; gen-2 changes are in gen-2.md §5)

```
BlobState (stack, ~410 B, Copy)
  → entity encoder (perspective = acting seat) → 14–58 typed tokens
  → transformer (d=128, 8 layers, 1.63M params)
  → bid head (CLS → 14), play head (per hand-card token), value head (CLS → 1, tanh)
```

**Crate boundaries** (keep these in gen 2):
- `blob-engine` is pure Rust plus `ort` and must **never** depend on `tch`.
- `tch` lives only in `blob-nn`.
- `blob-bin` is inference-only (no `blob-nn`, `tch` or `rayon`), so it builds on the Windows deployment target.
- The encoder, MCTS, belief tracking, ONNX evaluator, replay buffer and rule bot all live in `blob-engine`.

**Card encoding:** `suit × 13 + rank`, with ♠=0 ♥=1 ♣=2 ♦=3 and 2=0 … A=12. Hands are `u64` bitmasks.

**Scoring:** `tricks_won == bid ? 10 + bid : 0`.

**Encoder contracts:**
- Hand-card tokens are emitted in `Hand::iter()` order (ascending card index). That order is the play-policy action order used by MCTS, the replay buffer and the ONNX postprocessing. Don't reorder it.
- Rank, suit and player are raw one-hots inside per-token feature vectors, with one input projection per token type. Played-card tokens also get a learned chronological embedding.

**MCTS:**
- Determinization: N sampled deals per decision, one arena-allocated tree each, root visits summed across trees.
- Lockstep batching across deals: one `evaluate_batch` of size 5 per step.
- Forced moves are skipped without a network call.
- Root Dirichlet noise is on in self-play.
- `policy_target` is always τ=1; `policy_sampling` follows the τ schedule.
- **Known flaw (gen-2.md §2.3):** `backprop` credits a network leaf only to the seat about to move there.

**Training (gen-1 driver):** synchronous iterations. Self-play runs in a rayon pool with one ONNX session per thread, examples go into a FIFO replay buffer (raw `BlobState` + sparse policy, re-encoded at sample time), then the GPU trains on them via tch. Value target: final-game z-score (to be replaced; gen-2.md §5.1).

## Crate choices

| Crate | Used for |
|---|---|
| `tch` 0.20 (libtorch, pinned) | training |
| `ort` 2.0-rc | inference |
| `rayon` | self-play |
| `serde` + `bincode` | checkpoints and buffer |
| `smallvec` | MCTS children |
| `rand` + `rand_xoshiro` | RNG |
| `clap` | CLI |
| `tracing` | logging |

## Tests, benchmarks, diagnostics

- `cargo test -p blob-engine` — run in the **debug** profile. Several tests expect debug-assertion panics and fail under `--release`.
- `cargo bench -p blob-engine --bench core` — engine micro-benchmarks. Gen-1 numbers are in gen-2.md §3.1.
- `BLOB_ONNX_MODEL=<model.onnx> cargo bench -p blob-engine --bench onnx_mcts` — ONNX and search benches; they skip without the env var.
- ONNX ↔ tch parity: `BLOB_ONNX_MODEL=… BLOB_TCH_CHECKPOINT=<dir with model.ot> cargo test -p blob-nn onnx_tch_value_parity`. Tolerance 1e-4: an 8-layer fp32 transformer drifts ~2e-5 between kernels.
- `cargo test -p blob-bin` — `play` and CLI tests, including whole games through the terminal UI.
- Known failure: `blob-nn` `self_play::tests::five_games_produce_valid_examples` (pre-existing; gen-2.md §9).
- `blob-engine/examples/diagnostics.rs` — `match`, `value` and `tokens` commands; reproduces gen-2.md §2. See its header and gen-2.md Appendix A.
- `blobmaster play [--model <onnx>] [--show]` — play in the terminal. Without a model the bots are rule bots; with one they search at 5×100. Type `help` in the game.

## Hardware target

- **Training:** Ubuntu 24.04, Ryzen 9 7950X (16C/32T), RTX 4060 8 GB, 128 GB DDR5.
- **Future inference:** Windows + Intel iGPU via ONNX Runtime.
- **Self-play:** run at 32 threads (with batch-5 lockstep, 32T beats 16T).
- **GPU:** training only; GPU-batched inference was measured slower (gen-2.md §3.2).

## Runtime environment (training runs on this machine)

All long training runs go through `./target/release/blobmaster-train train ...` and need three things lined up. Skip any of them and you get one of:
- a "missing shared library" abort at startup;
- a silent CPU fallback;
- `scripts/export_onnx.py` failing with `ModuleNotFoundError: torch`.

The three things:

- **Pinned Python venv.** `.venv/` at the repo root: Python 3.12.3 with `torch==2.5.1+cu124`, `onnxruntime==1.24.4`, `onnx==1.21.0`, `numpy==2.4.4`.
  - System Python (`/usr/bin/python3`, also 3.12.3) has none of these.
  - [blob-train/src/main.rs](blob-train/src/main.rs) invokes `scripts/export_onnx.py` as `python3` every iteration, so the venv must be first on `PATH`.
- **Downloaded libtorch.** `tch` with `download-libtorch` puts a libtorch tree under `target/<profile>/build/torch-sys-*/out/libtorch/libtorch/lib`.
  - The `torch-sys-*` hash changes whenever `tch` rebuilds, so find the directory with `find` rather than hard-coding it.
  - `tch = 0.20.0` is pinned in Cargo.lock; it ships a libtorch 2.4-class build with a CUDA 12.x runtime.
- **CUDA preload.**
  - Without `LD_PRELOAD=libtorch_cuda.so`, libtorch loads CPU-only and the run silently falls back.
  - Without `LD_LIBRARY_PATH=$LIBTORCH_DIR`, the binary fails to load at all.
  - The CUDA driver on the box is 580.x; the runtime is carried by libtorch. `nvcc` is **not installed system-wide**, so don't reach for it.
- **Do NOT let `LD_PRELOAD` reach Python subshells.**
  - Tch's vendored libtorch (~2.4) has a different C++ ABI from the venv's `torch==2.5.1+cu124`. Preloading it crashes `import torch` with `undefined symbol: ...torch::jit::Graph::toString...`.
  - Wrap Python calls in `( unset LD_PRELOAD; python3 ... )`.
  - The training driver itself is fine: its export call uses `Command::new`, which builds the environment explicitly.

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

`scripts/README.md` has the same incantation; keep the two in sync when re-rooting. The GPU is a single RTX 4060 (`cuda:0`). Run `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` before launching if another run might be resident.

## Gen-1 driver behavior (valid while `blob-train` is unchanged)

### `total_iterations` is the absolute target iteration (not a count)

- The loop runs `while tl.iteration < total_iterations`.
- Eval runs when `iter > anchor_iter && iter % eval_interval == 0`. **To get an eval row at iter K, set `total_iterations = K + 1`.**
- Resumes: `total_iterations` is still absolute. To run `add` more iterations, set `target = latest_iter + 1 + add`. The deleted gen-1 resume script did this: `git show 36bba81:scripts/sweep-2026-04-28-resume.sh`.
- `LrSchedule` keys its cosine on the same field. When it used to be a count, a resume pinned the LR at `MIN_LR` (1e-5) for the whole resume window ("Bug #2", 2026-04-28).
  - **Symptom:** `learning_rate` flat at 1e-5 across consecutive iterations in `metrics.jsonl`. The `iteration complete` log line prints `learning_rate=`.
- `--resume` sets the eval anchor to the resume baseline, not to `iter_000000`. To compare against the start, run `blobmaster-train evaluate iter_K/model.onnx iter_000000/model.onnx`.

### MCTS sim budget is config-driven (since 2026-05-17)

- `adaptive_budget` reads `num_determinizations` and `sims_per_determinization` from `[mcts]`, whether in the training TOML or the `--config` passed to `evaluate`.
  - Forced moves short-circuit to `(1, 0)`.
  - `min_sims_floor` can raise the simulation count.
  - Before 2026-05-17 the budget was hardcoded to 5×100 and the TOML fields were silently ignored.
- `evaluate` echoes `num_determinizations=` and `sims_per_determinization=` in its startup line. Check it to confirm an override landed.
- The `adaptive_budget_reads_cfg` test pins this behavior.
- The gen-1 "never below 5×100" rule was measured with gen-1's uninformative values. Re-measure it in gen 2 instead of treating it as a law.

### Validation split (since 2026-10-02)

- `[training] validation_fraction` (default 0.03) holds out whole self-play games by a hash of their seed. Their examples go to `val_buffer` and are never trained on.
  - The split is by game, not round, because the gen-1 value target is the final game score.
  - `val_buffer` holds `buffer_capacity × fraction` examples.
  - It is saved as `iter_*/val_buffer.bin` and restored by `--resume`. A pre-split checkpoint resumes with an empty one and logs a warning.
- `metrics.jsonl` gets `val_*` (whole validation buffer) and `train_eval_*` (an equal-size replay sample). Both are measured after training, with dropout off. Also `val_examples_added` and `val_buffer_len`.
  - Compare `val_*` with `train_eval_*`, never with the logged training losses: those are averaged over the iteration with dropout on.
  - `*_value_loss_predict0` is the always-predict-0 baseline.

### Graceful exit (`STOP` file)

- `touch checkpoints/<run-name>/STOP`: the loop finishes the current iteration (checkpoint + ONNX export), deletes the file and exits.
- There is no signal handler, so Ctrl-C loses the iteration in flight.

### Visualizing a run

Both scripts use the venv interpreter.

- **Training dashboard:**
  ```bash
  .venv/bin/python scripts/visualize_strength.py --csv checkpoints/<run>/strength.csv \
    --metrics checkpoints/<run>/metrics.jsonl --stderr logs/<run>.stderr --out-dir logs/<run>-progress
  ```
  Produces win rate vs anchor, score differential, losses, convergence diagnostics, per-iteration wall-clock and bid success, plus held-out vs training-data losses (`07_generalization.png`) for runs with a validation split. Its win rates are relative to past checkpoints only; see the rule above.
- **Weight evolution:**
  ```bash
  .venv/bin/python scripts/visualize_weight_evolution.py --checkpoint-dir checkpoints/<run> \
    --out-dir logs/<run>-weight-evolution
  ```
  Loads every `iter_*/model.onnx` into memory (~11 MB per iteration), so a long run needs a few GB of RAM.
