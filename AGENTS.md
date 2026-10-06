# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

- **Gen 1 concluded on 2026-05-17** (final run `run-2026-05-14`, 168 iterations). On 2026-10-02 it was measured against a fixed rule bot and found weak: **−10.2 ± 2.9** points per game with 5×100 search (gen-2.md §2).
  - Root causes: a final-game value target that the value head memorized, values credited only to the seat to move at each leaf, and clipped end-of-round values.
- **Gen 2 is the remake, planned in [gen-2.md](gen-2.md).** Phases 0 (yardsticks), 1 (encoder layout, determinization, tie-break), 2 (clean break) and 3 (the gen-2 engine: per-round utility, all-seat backup, policy and value evaluators, layout id) are done (2026-10-05).
- **Phase 4 (two networks, supervised warm start from rule bot 2) is done (2026-10-06):** G1 and G2 pass with layout 4, which gives V's hand cards, mine and every opponent's, "beats the current winner" and their standing in suit, computed on the deal. Search defaults changed with it: c_puct 0.2, and 1-card bids from P alone. Next: **Phase 5, async self-play RL**.
- **Gen 1 is gone from `master`.**
  - Code: tag `gen-1-final` (as trained) and tag `gen-1-compat` (the last commit whose tooling runs gen-1 models). Retired documents: `git show gen-1-final:<file>` (gen-2.md §10).
  - Reference models, `buffer.bin`, `model.ot` and metrics: archived outside the repo in `~/blobmaster-archive/run-2026-05-14/` (this machine only). gen-2.md Appendix A reproduces the gen-1 measurements from it.
- **The warm-start model is `checkpoints/pretrain-2026-10-06/model`** (layout 4: its V, with the P of `checkpoints/pretrain-2026-10-05`, layout 3's run; this machine only, weights are git-ignored). Each run directory also holds the config, `metrics.jsonl`, `held_out.json` and the bench reports (`bench/`).
- **Never commit per-iteration weights.** `*.onnx` is git-ignored; add a deliberate reference model with `git add -f`. History was rewritten on 2026-10-05 to drop the gen-1 model blobs (`.git` 4.9 GB → 31 MB, gen-2.md §9); a clone from before then must re-clone.

## Documents

- **[gen-2.md](gen-2.md)** — authoritative findings, design and roadmap. Read it before any architectural decision.
- **[README.md](README.md)** — overview, game rules, repo layout.
- **[legacy/](legacy/)** — archived gen-0 Python source. Read-only; never modify.

## Judging strength — rule

**Never report progress from training losses or checkpoint-vs-checkpoint win rates alone.** Gen 1 looked healthy on both while losing to an 80-line bot. Strength claims need:
- the absolute benchmark: one model seat vs 4 rule bots (`blob-engine/src/rule_bot.rs`), with the 95% CI (command below);
- held-out losses: losses on validation rounds next to the same losses on an equal-size training sample, both with dropout off (`Learner::policy_held_out` / `value_held_out`; `pretrain` logs them in `metrics.jsonl` and `held_out.json`).

```bash
cargo build --release -p blob-bin
./target/release/blobmaster bench <model dir> --mode network   # 128 deals × 5 seats, ~10 s
./target/release/blobmaster bench <model dir> --mode search    # 64 deals × 5 seats; bids 20×25, plays 5×100, ~9 min
```

A model is a directory: `policy.onnx`, `value.onnx`, `meta.json` (gen-2.md §5.3). Network mode loads only the policy net. Search budgets: `--bid-dets/--bid-sims` (default 20×25) and `--dets/--sims` (plays, default 5×100). `--c-puct` (default 0.2) and `--search-one-card-bids` (default `false`: P's bid, no tree) apply to `bench` and `play`; a search report's first line names all four settings.

`bench` plays duplicate deals: every deal seed once from each seat. Its CI is over deals. The default seed is fixed, so models are compared on the same cards. Opponents default to the rule bot; `--opponent <model dir>` uses that model's policy net, greedy (bots never search).

`rulebot2` (`blob-engine/src/rule_bot_2.rs`: card counting, bid-aware, +14.5 ± 0.3 vs the rule bot with `--deals 4000`) works as the focal player or as `--opponent`. It is a harder second yardstick; the rule bot stays the reference, so never retune `rule_bot.rs`. `rulebot2r` is rule bot 2 with rollouts (`--samples`, default 128; `--depth`, default full; `--play-only`); its bench label carries the settings. Its gains vs `rulebot2` are in the `rule_bot_2.rs` header.

References (default deals, against the rule bot):
- gen-1 final: with 5×100 search on every decision **−10.2 ± 2.9**, network only −12.1 ± 2.0 (2026-10-02). Gen-1 models no longer load on `master`, so compare against these numbers or reproduce them at `gen-1-compat` (gen-2.md Appendix A).
- the Phase-4 warm start (layout 4): network only **+15.5 ± 1.5** (rule bot 2 on the same deals: +15.2 ± 1.5); search at the defaults **+20.0 ± 2.4**. Against four rule bot 2s on 128 fresh deals (`--seed 7`): search **+4.7 ± 1.0**, network only +0.1 ± 0.5 (gen-2.md §6 Phase 4).
- in games of 1-card rounds only (`--cards 1`), search lost to P alone (−0.8 ± 0.1): the sampled deals ignore the bids already made. Hence 1-card bids from P (gen-2.md §8, exact 1-card bids).

The tool uses every core, so don't run it next to a training run.

## Architecture (gen 2, gen-2.md §5)

```
BlobState (stack, ~410 B, Copy)
  ├─ P mode encoder (the seat to move's view) → 14–58 typed tokens
  │    → policy net P (d=128, 8 layers) → bid policy (CLS → 14), play scores (per hand-card token)
  └─ V mode encoder (every hand, on a sampled deal) → + opponents' card tokens
       → value net V (d=128, 4 layers) → expected ŝ per seat (per player token)
search: u_s = ŝ_s − λ·mean_{j≠s} ŝ_j at every leaf, exact at the round's end; backed up to every seat
```

**Crate boundaries** (keep these in gen 2):
- `blob-engine` is pure Rust plus `ort` and must **never** depend on `tch`.
- `tch` lives only in `blob-nn`.
- `blob-bin` is inference-only (no `blob-nn` or `tch`), so it builds on the Windows deployment target.
- `blob-train` is the training CLI (`pretrain`, `export`; `train` in Phase 5). It links `tch`, so run it through `scripts/blobmaster-train.sh`, which sets the library path (Runtime environment).
- The encoder, MCTS, belief tracking, ONNX evaluator, replay buffer and rule bots all live in `blob-engine`.

**Card encoding:** `suit × 13 + rank`, with ♠=0 ♥=1 ♣=2 ♦=3 and 2=0 … A=12. Hands are `u64` bitmasks.

**Scoring:** `tricks_won == bid ? 10 + bid : 0`. Search and the value target use the per-round utility (`scoring.rs`, gen-2.md §5.1): ŝ = round points / (10 + cards dealt), `u_s = ŝ_s − λ·mean_{j≠s} ŝ_j`, λ = 1 by default (`MctsConfig::lambda`).

**Encoder contracts:**
- Hand-card tokens are emitted in `Hand::iter()` order (ascending card index). That order is the play-policy action order used by MCTS, the replay buffer and the ONNX postprocessing. Don't reorder it. `encoder::hand_card_indices` gives it without encoding.
- Rank, suit and seat are raw one-hots inside per-token feature vectors, with one input projection per token type. Played-card tokens also get a learned chronological embedding.
- Seats are relative to the perspective (gen-2.md §6 Phase 1): "me" is seat 0, and player tokens come in that order.
- **Two modes, one code path.** `encode` (P mode) shows only the seat to move's own cards; `encode_value` (V mode) adds every opponent's hand as `TOKEN_TYPE_OPP_HAND` tokens tagged with the owner's relative seat. Nothing game-level is encoded (no cumulative scores, no round number).
- **V mode describes the deal** (layout 4). Every hand card, mine and each opponent's, carries the same features (`card_features`), computed for its owner against the other hands: suit standing, legal, beats the current winner. Opponents' cards also carry "owner still to play". P mode counts against the cards I haven't seen.
- **One layout, guarded** (padded width `FEAT_DIM` = 49). `encoder::LAYOUT_ID` is stamped into every exported ONNX file (`blob_layout_id`, plus `blob_network` = `policy`/`value`), and `OnnxPolicy` / `OnnxValue` refuse any other. The golden-hash test `golden_layout_hash` hashes P mode and V mode separately and fails on any encoding change: bump `LAYOUT_ID` and record the new hashes. A layout change means retraining, never a compatibility path. The exception is a layout that leaves the P hash unchanged: P's input is then the same, so `[learner] policy_from` copies P's weights and only V trains (layout 4 did this).
- `scripts/export_onnx.py` mirrors the token widths and `LAYOUT_ID`; `export_script_mirrors_feature_widths` checks them.

**MCTS:**
- Determinization: N sampled deals per decision, one arena-allocated tree each, root visits summed across trees.
- Budgets per phase (`MctsConfig::bid_budget`, `play_budget`): bids 20 deals × 25 sims, plays 5 × 100 by default.
- Leaves: one `PolicyEvaluator::policy_batch` (priors, own view) and one `ValueEvaluator::values_batch` (ŝ for every seat, sampled deal) per lockstep step, batch up to 5. At the round's end the exact utilities are used instead.
- Backup adds `u_s` to every seat's sum at every node on the path; UCB reads the acting seat's mean over all visits. No per-seat counts.
- Forced moves are skipped without a network call.
- 1-card bids come from P without a tree (`MctsConfig::search_one_card_bids`, default off).
- c_puct defaults to 0.2 (`DEFAULT_C_PUCT`). The warm start's priors are sharp; at gen 1's 1.5 the visit counts barely depart from them.
- Root Dirichlet noise is on in self-play.
- `policy_target` is always τ=1; `policy_sampling` follows the τ schedule.

**Training:** `blobmaster-train pretrain` (Phase 4, the supervised warm start); the RL driver `train` comes in Phase 5.
- `blob-nn`:
  - `model.rs`: `PolicyNet` (P, 8 layers, bid and play heads) and `ValueNet` (V, 4 layers, a projection for opponents' cards, a per-token ŝ head read at the player tokens). Parameter names match `scripts/export_onnx.py`.
  - `train.rs`: losses, AdamW, the LR schedule keyed to learner steps, checkpoints (a directory: `policy.ot`, `value.ot`, `meta.json` with `learner_step`; written beside and renamed in).
  - `learner.rs`: replay batches → tensors, `Learner` (one P and one V update per learner step), held-out measurements, `is_validation_round`, `is_forced`.
- **V trains with sigmoid cross-entropy against ŝ** (a soft target), not MSE: same minimizer, but under MSE a high LR pinned the saturated sigmoid at 0 for good. Held-out V is reported as MSE, the targets' variance, correlation, the last-trick error at the seats that trick still decides (G1: every remaining play is forced, so ≈ exact) and the 1-card-round error.
- **P skips forced decisions** (one legal move: its loss is 0 whatever it outputs); V trains on every state.
- **Teacher data** (`blob-engine/src/teacher.rs`): rounds played by rule bot 2, with rule-bot seats (`rule_bot_share`) and exploration (`explore`) mixed in. Every decision is labelled with rule bot 2's policy, whoever played it: `argmax_weight` on its own move, the rest a softmax over each move's expected points. `fill_buffer` is deterministic in the seed on any thread count.
- **`pretrain`** (`blob-train/src/pretrain.rs`): teacher rounds → split by round → loader threads build batches while the GPU trains → `metrics.jsonl` (a data row; training rows; held-out rows: validation vs an equal training sample) → checkpoint → held-out on every validation example (`held_out.json`, with G1) → `<run>/model`. A fresh run starts a new `metrics.jsonl`. A `STOP` file in the run directory saves and exits (after that step's held-out row); `--resume` continues with the run's own `config.toml` (the data replays from the seed; the optimizers restart, tch can't save their state). `[learner] policy_from = "<checkpoint dir>"` copies P and trains V only (~37 min instead of ~80).
- Config: `blob-train/pretrain.sample.toml` lists every key at its default (a test keeps them equal).
- The replay buffer (`blob-engine/src/replay.rs`) stores raw states, sparse policies and each seat's round points; `push_round` writes a finished round's decisions with a round id. `SharedReplay` is the concurrent wrapper; `sample_batch(.., augment)` relabels suits at random (`augment.rs`); `sample_batch_from(slots, ..)` samples from a subset, e.g. the training rounds.
- Single rounds start with `dealing::new_round(RoundParams)`; `round::RoundMix` draws their parameters from real games' round mix.
- Config structs reject unknown keys (`#[serde(deny_unknown_fields)]`), so a stale config fails loudly. Keep that for every new config type.

## Crate choices

| Crate | Used for |
|---|---|
| `tch` 0.20 (libtorch, pinned) | training |
| `ort` 2.0-rc | inference |
| `serde` + `bincode` | buffer |
| `toml`, `serde_json` | configs; metrics and checkpoint meta |
| `smallvec` | MCTS children |
| `rand` + `rand_xoshiro` | RNG |
| `clap` | CLI |

## Tests, benchmarks, diagnostics

- `cargo test -p blob-engine` — run in the **debug** profile. Several tests expect debug-assertion panics and fail under `--release`.
- `cargo test -p blob-bin` — `play` and CLI tests, including whole games through the terminal UI.
- `blob-nn` tests link libtorch: run them in **release** with the library path set (see Runtime environment). `tch` downloads libtorch per profile, and only `target/release` has it; a debug build would download it again.
  ```bash
  LIBTORCH_DIR="$(find target/release/build -maxdepth 6 -type d -name lib -path '*/libtorch/libtorch/lib' | head -n1)"
  LD_LIBRARY_PATH="$LIBTORCH_DIR" cargo test --release -p blob-nn
  ```
- `cargo bench -p blob-engine --bench core` — engine micro-benchmarks. Gen-1 numbers are in gen-2.md §3.1.
- `BLOB_MODEL_DIR=<model dir> cargo bench -p blob-engine --bench onnx_mcts` — ONNX and P + V search benches; they skip without the env var. With the same variable, `cargo test -p blob-engine` also runs the model-loading tests in `onnx.rs`.
- `cargo test --release -p blob-train` (with the library path) — the config schema and sample.
- **Random-init model directory**: `scripts/blobmaster-train.sh export --output <dir>` writes a random-init P and V.
- **Pretrain smoke test** (~35 s): a config with `[data] rounds = 2000`, `[learner] steps = 200, warmup_steps = 20`, `[log] every = 50, eval_every = 100, eval_examples = 1000`, then `scripts/blobmaster-train.sh pretrain --config <toml> --output <dir>`.
- **ONNX ↔ tch parity** (P and V), on any learner checkpoint, e.g. `<run>/checkpoint` with `<run>/model`, or a random one:
  ```bash
  BLOB_SAVE_CKPT_DIR=<ckpt> LD_LIBRARY_PATH="$LIBTORCH_DIR" \
    cargo test --release -p blob-nn --test save_random_checkpoint -- --ignored save_random_init
  scripts/blobmaster-train.sh export --checkpoint <ckpt> --output <dir>
  BLOB_MODEL_DIR=<dir> BLOB_TCH_CHECKPOINT=<ckpt> LD_LIBRARY_PATH="$LIBTORCH_DIR" \
    cargo test --release -p blob-nn --test onnx_tch_parity
  ```
  - Use absolute paths: tests run in the crate directory.
  - Tolerance is 1e-5 on the legal bid and play policies and on every seat's ŝ.
  - `export --check` (the Python-side check on random in-range inputs, gate 1e-5, relative above 1) passes on trained weights but not on a random *tch* init (~2e-5 for P: tch initializes with ~2.5× torch's weight scale). The Rust parity above is the authoritative check.
- `cargo test --release -p blob-nn -- --ignored numerical_stability` — 50 learner steps at a high LR on teacher data; losses, outputs and weights stay finite and in range.
- `blobmaster play [--model <dir>] [--show]` — play in the terminal. Without a model the bots are rule bots; with one they search (bids 20×25, plays 5×100). Type `help` in the game.
- The gen-1 diagnostics (`examples/diagnostics.rs`: `match`, `value`, `tokens`) are at tag `gen-1-compat`; see gen-2.md Appendix A.

## Hardware target

- **Training:** Ubuntu 24.04, Ryzen 9 7950X (16C/32T), RTX 4060 8 GB, 128 GB DDR5.
- **Future inference:** Windows + Intel iGPU via ONNX Runtime.
- **Self-play:** run at 32 threads (gen 1: with batch-5 lockstep, 32T beats 16T).
- **GPU:** training only; GPU-batched inference was measured slower (gen-2.md §3.2).

## Runtime environment (anything that links `tch` or runs the export script)

That is the `blob-nn` and `blob-train` tests and `blobmaster-train` (every subcommand: the binary links libtorch). `scripts/blobmaster-train.sh` sets the library path and the CUDA preload for it. Skip one of the three things below and you get one of:
- a "missing shared library" abort at startup;
- a CPU-only libtorch (`pretrain` then refuses `cuda`; other code that links `tch` silently runs on the CPU);
- `scripts/export_onnx.py` failing with `ModuleNotFoundError: torch`.

The three things:

- **Pinned Python venv.** `.venv/` at the repo root: Python 3.12.3 with `torch==2.5.1+cu124`, `onnxruntime==1.24.4`, `onnx==1.21.0`, `numpy==2.4.4`.
  - System Python (`/usr/bin/python3`, also 3.12.3) has none of these.
  - `blobmaster-train export` runs the script with `.venv/bin/python` when it exists, else `python3` from `PATH`.
- **Downloaded libtorch.** `tch` with `download-libtorch` puts a libtorch tree under `target/<profile>/build/torch-sys-*/out/libtorch/libtorch/lib`.
  - The `torch-sys-*` hash changes whenever `tch` rebuilds, so find the directory with `find` rather than hard-coding it.
  - `tch = 0.20.0` is pinned in Cargo.lock; it ships a libtorch 2.4-class build with a CUDA 12.x runtime.
- **Library path and CUDA preload.**
  - Without `LD_LIBRARY_PATH=$LIBTORCH_DIR`, a binary that links `tch` fails to load at all.
  - Without `LD_PRELOAD=$LIBTORCH_DIR/libtorch_cuda.so`, libtorch loads CPU-only: `pretrain` refuses `cuda`, anything else silently runs on the CPU.
  - The CUDA driver on the box is 580.x; the runtime is carried by libtorch. `nvcc` is **not installed system-wide**, so don't reach for it.
- **Do NOT let `LD_PRELOAD` reach Python subshells.**
  - Tch's vendored libtorch (~2.4) has a different C++ ABI from the venv's `torch==2.5.1+cu124`. Preloading it crashes `import torch` with `undefined symbol: ...torch::jit::Graph::toString...`.
  - Wrap Python calls in `( unset LD_PRELOAD; .venv/bin/python ... )`. `blobmaster-train export` removes `LD_PRELOAD` itself.

`scripts/README.md` has the same notes; keep the two in sync. Launch a run with `scripts/blobmaster-train.sh pretrain --config <toml> --output checkpoints/<run>` (the learner refuses `cuda` when libtorch has no CUDA, instead of silently using the CPU). The GPU is a single RTX 4060 (`cuda:0`). Run `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` before launching if another run might be resident.

`scripts/visualize_strength.py` and `scripts/visualize_weight_evolution.py` still read gen-1 outputs (`strength.csv`, per-iteration `metrics.jsonl`, `iter_*` directories). They are re-pointed at gen-2 outputs with the Phase-5 driver, whose evaluator produces the strength series they plot (gen-2.md §4).
