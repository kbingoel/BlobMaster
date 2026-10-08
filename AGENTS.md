# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

- **Gen 1 concluded on 2026-05-17** (final run `run-2026-05-14`, 168 iterations). On 2026-10-02 it was measured against a fixed rule bot and found weak: **−10.2 ± 2.9** points per game with 5×100 search (gen-2.md §2).
  - Root causes: a final-game value target that the value head memorized, values credited only to the seat to move at each leaf, and clipped end-of-round values.
- **Gen 2 is the remake, planned in [gen-2.md](gen-2.md).** Phases 0 (yardsticks), 1 (encoder layout, determinization, tie-break), 2 (clean break) and 3 (the gen-2 engine: per-round utility, all-seat backup, policy and value evaluators, layout id) are done (2026-10-05).
- **Phase 4 (two networks, supervised warm start from rule bot 2) is done (2026-10-06):** G1 and G2 pass with layout 4, which gives V's hand cards, mine and every opponent's, "beats the current winner" and their standing in suit, computed on the deal. Search defaults changed with it: c_puct 0.2.
- **Phase 4b (bid-aware sampling) is done (2026-10-06):** 1-card bids are computed exactly, and every search's sampled deals are weighted by the bids already made, under P. Search vs rule bot 2: +6.4, paired +1.7 over Phase 4, at +43% time (gen-2.md §6 Phase 4b). Phase 5 followed (below).
- **Gen 1 is gone from `master`.**
  - Code: tag `gen-1-final` (as trained) and tag `gen-1-compat` (the last commit whose tooling runs gen-1 models). Retired documents: `git show gen-1-final:<file>` (gen-2.md §10).
  - Reference models, `buffer.bin`, `model.ot` and metrics: archived outside the repo in `~/blobmaster-archive/run-2026-05-14/` (this machine only). gen-2.md Appendix A reproduces the gen-1 measurements from it.
- **Phase 5 (async self-play RL): the driver is built; three runs and the rollout validations done (2026-10-08).** `checkpoints/rl-2026-10-06` (stopped at step 7696, resumable): P alone went from +0.2 to +11.0 vs rule bot 2 (paired +10.8; +10 within 1.4 h of learning) and search from +6.4 to +9.1, then both plateaued, in self-play too (P at step 7600 vs four P's of step 2000: +1.0 ± 0.6).
- **Day 2 (2026-10-07, gen-2.md §6 Phase 5):** against rule bot 2, search models the opponents as P, which stopped fitting as P left rule bot 2 behind, so the improvement step is now judged by search vs four copies of P (P alone = 0): +1.0 ± 1.0 at step 7600, whatever the root rule. One hour of V trained on rounds of P alone (the V stream) raised it to +1.9 ± 1.0 (paired +0.9 ± 0.8). Built: the Q root rule (`root_rule = "q"`), the V stream (`[value_stream]`), search-vs-P benches in the driver, the fixed V check. Night runs `checkpoints/rl-2026-10-07a` (Q rule + V stream) and `…b` (Q rule, no V stream: the control), each from run 1's step 7600 (`checkpoints/night-2026-10-07.sh`).
- **Day 3 (2026-10-08, gen-2.md §6 Phase 5):** the night runs both gained ~+1.2 vs rule bot 2, mostly in their first 25 min, then sat flat; the V stream improved V's MSE but not search's margin or P. Built: policy iteration by rollouts (`[rollout]`, `blob_engine::rollout`): P trains on decisions of its own rounds with every legal move played out on the real deal, no search. Validations (1 h each from 2b's final): with greedy rounds P gains ~+0.8 against four copies of its start P within ~10 min and nothing against rule bot 2 (the search loop: +0.9 at 25 min, +1.2–1.6 at 3.5 h, same measure); rule bot 2 at 40% of the seats buys +0.5 against rule bot 2 but not against the rule bot. Night runs `checkpoints/pi-2026-10-08a` (6 h, validation 2's settings) and `…b` (5.5 h, LR 3e-5, 8 × 512 per step): noise or a fixed point (`checkpoints/night-2026-10-08.sh`).
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

A model is a directory: `policy.onnx`, `value.onnx`, `meta.json` (gen-2.md §5.3). Network mode loads only the policy net. Search budgets: `--bid-dets/--bid-sims` (default 20×25) and `--dets/--sims` (plays, default 5×100). `--c-puct` (default 0.2), `--one-card-bids` (`exact` by default; `policy` is P's bid, `search` a searched one) the bid weighting of sampled deals (`--bid-candidates`, default 8 per deal, 0 = off; `--bid-noise`, default 0.1) and the root rule (`--root visits|q`, `--q-temp`, below) apply to `bench` and `play`; a search report's first line names every setting. `--dets 1 --sims 1 --bid-dets 1 --bid-sims 1 --one-card-bids policy` plays exactly as P alone (checked: paired 0.0 ± 0.0), so `--dets 1 --sims 1` searches bids only and `--bid-dets 1 --bid-sims 1 --one-card-bids policy` plays only.

`bench` plays duplicate deals: every deal seed once from each seat. Its CI is over deals. The default seed is fixed, so models are compared on the same cards. `--per-deal-out <file>` saves each deal's result and `--compare <file>` prints this run minus that one with a paired CI (same seed, deals and table); use it to compare settings or models, since separate CIs overlap long after a difference is real. Opponents default to the rule bot; `--opponent <model dir>` uses that model's policy net, greedy (bots never search).

`rulebot2` (`blob-engine/src/rule_bot_2.rs`: card counting, bid-aware, +14.5 ± 0.3 vs the rule bot with `--deals 4000`) works as the focal player or as `--opponent`. It is a harder second yardstick; the rule bot stays the reference, so never retune `rule_bot.rs`. `rulebot2r` is rule bot 2 with rollouts (`--samples`, default 128; `--depth`, default full; `--play-only`); its bench label carries the settings. Its gains vs `rulebot2` are in the `rule_bot_2.rs` header.

References (default deals, against the rule bot):
- gen-1 final: with 5×100 search on every decision **−10.2 ± 2.9**, network only −12.1 ± 2.0 (2026-10-02). Gen-1 models no longer load on `master`, so compare against these numbers or reproduce them at `gen-1-compat` (gen-2.md Appendix A).
- the Phase-4 warm start (layout 4): network only **+15.5 ± 1.5** (rule bot 2 on the same deals: +15.2 ± 1.5); search at the Phase-4b defaults **+20.1 ± 2.4** (Phase 4's settings: +20.0). Against four rule bot 2s on 128 fresh deals (`--seed 7`): search **+6.4 ± 1.1** (Phase 4's settings: +4.7 ± 1.0), network only +0.1 ± 0.5 (gen-2.md §6 Phase 4, 4b).
- the first RL run, step 7600 (`checkpoints/rl-2026-10-06/models/step-007600/model`): network only **+11.0 ± 1.0** vs rule bot 2 (default deals, 256) and +24.1 ± 1.6 vs the rule bot; against four rule bot 2s on `--seed 7` (128 deals), search +9.1 ± 1.5 at c_puct 0.2 and **+11.2 ± 1.5 at c_puct 1.0**, network only +10.8 ± 1.3 (gen-2.md §6 Phase 5).
- in games of 1-card rounds only (`--cards 1`, against rule bot 2), search lost to P alone (−0.9 ± 0.1): the sampled deals ignored the bids already made. Exact 1-card bids (Phase 4b) score +0.2 ± 0.1 there.

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
- `blob-train` is the training CLI (`pretrain`, `train`, `export`). It links `tch`, so run it through `scripts/blobmaster-train.sh`, which sets the library path (Runtime environment).
- **ONNX Runtime and libtorch never run in one process.** A search bench inside the `train` process crashed in ONNX Runtime (`BiasGelu … GetElementType is not implemented`) while the learner ran; the same bench in `blobmaster` didn't. So `train`'s actors are a separate `blobmaster selfplay` process and its benches are `blobmaster bench` subprocesses, all started without `LD_PRELOAD` / `LD_LIBRARY_PATH`.
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
- Bid-weighted deals (`belief::sample_deals`, `MctsConfig::bid_weighting`, gen-2.md §6 Phase 4b): draw N × `candidates` deals consistent with the voids, weight each by the likelihood of every earlier bid under P, read from that bidder's bid-time view of the candidate (`belief::rewind_to_bid`), with a noise floor (`noise`: a share of uniform bids); keep N by systematic resampling. Costs one P call per candidate per bidder.
- Budgets per phase (`MctsConfig::bid_budget`, `play_budget`): bids 20 deals × 25 sims, plays 5 × 100 by default.
- Leaves: one `PolicyEvaluator::policy_batch` (priors, own view) and one `ValueEvaluator::values_batch` (ŝ for every seat, sampled deal) per lockstep step, batch up to 5. At the round's end the exact utilities are used instead.
- Backup adds `u_s` to every seat's sum at every node on the path; UCB reads the acting seat's mean over all visits. No per-seat counts.
- Forced moves are skipped without a network call.
- 1-card bids are computed, not searched (`one_card.rs`, `MctsConfig::one_card_bids` = `Exact`): earlier bidders' cards drawn by their bids' likelihood under P, later bids from P's policy (every line weighted), the forced cards played out, the best expected `u` chosen. No tree, no V. `Policy` (P's bid) and `Search` remain as settings.
- c_puct defaults to 0.2 (`DEFAULT_C_PUCT`). The warm start's priors are sharp; at gen 1's 1.5 the visit counts barely depart from them.
- Root Dirichlet noise is on in self-play.
- `policy_target` is always τ=1; `policy_sampling` follows the τ schedule.
- Root rule (`MctsConfig::root_rule`, gen-2.md §6 Phase 5): `visits` (default) sums the root visits over the trees, a vote of each deal's best move. `q` averages each move's per-tree mean utility over the trees with equal weight (Q̄) and returns π' ∝ P · exp(Q̄ / T) (`q_temperature`, utility units; 0 = the best Q̄) as target, sampling distribution and greedy move. Give each tree more simulations than legal moves, so every tree values every move. `rollouts` takes Q̄ from playing each move out with P's top move at every seat (each from its own view) on the budget's deals: an exact critic of P's play, for diagnostics (~1 ms per P state; minutes per decision at 32 deals).
- **Against rule bot 2, search models the opponents as P** (in its trees and in reading their bids). That fit while P copied rule bot 2 and stopped fitting as P left it behind, so the improvement step is judged by search vs four copies of P (`bench <model> --opponent <same model>`, P alone = 0), and P alone vs rule bot 2 stays the strength yardstick (gen-2.md §6 Phase 5, day 2).

**Training:** `blobmaster-train pretrain` (Phase 4, the supervised warm start) and `blobmaster-train train` (Phase 5, self-play RL; first run `checkpoints/rl-2026-10-06`).
- `blob-nn`:
  - `model.rs`: `PolicyNet` (P, 8 layers, bid and play heads) and `ValueNet` (V, 4 layers, a projection for opponents' cards, a per-token ŝ head read at the player tokens). Parameter names match `scripts/export_onnx.py`.
  - `train.rs`: losses, AdamW, the LR schedule keyed to learner steps, checkpoints (a directory: `policy.ot`, `value.ot`, `meta.json` with `learner_step`; written beside and renamed in).
  - `learner.rs`: replay batches → tensors, `Learner` (one P and one V update per learner step; `train_value_on` is a V update that isn't a step), held-out measurements, `is_validation_round`, `is_forced`.
- **V trains with sigmoid cross-entropy against ŝ** (a soft target), not MSE: same minimizer, but under MSE a high LR pinned the saturated sigmoid at 0 for good. Held-out V is reported as MSE, the targets' variance, correlation, the last-trick error at the seats that trick still decides (G1: every remaining play is forced, so ≈ exact) and the 1-card-round error.
- **P skips forced decisions** (one legal move: its loss is 0 whatever it outputs); V trains on every state.
- **Teacher data** (`blob-engine/src/teacher.rs`): rounds played by rule bot 2, with rule-bot seats (`rule_bot_share`) and exploration (`explore`) mixed in. Every decision is labelled with rule bot 2's policy, whoever played it: `argmax_weight` on its own move, the rest a softmax over each move's expected points. `fill_buffer` is deterministic in the seed on any thread count.
- **`pretrain`** (`blob-train/src/pretrain.rs`): teacher rounds → split by round → loader threads build batches while the GPU trains → `metrics.jsonl` (a data row; training rows; held-out rows: validation vs an equal training sample) → checkpoint → held-out on every validation example (`held_out.json`, with G1) → `<run>/model`. A fresh run starts a new `metrics.jsonl`. A `STOP` file in the run directory saves and exits (after that step's held-out row); `--resume` continues with the run's own `config.toml` (the data replays from the seed; the optimizers restart, tch can't save their state). `[learner] policy_from = "<checkpoint dir>"` copies P and trains V only (~37 min instead of ~80).
- **`train`** (`blob-train/src/rl.rs`; actors in `blob-engine/src/selfplay.rs`):
  - Two processes: `blobmaster selfplay` (ONNX; `run.actors` threads play single rounds with P + V search, every seat; reads `<run>/selfplay.json` and the model named in `<run>/model.json`; writes `<run>/replay/chunk-*.bin`; stops when its stdin closes) and the driver (libtorch: tails `replay/` into a training and a validation buffer split by round id; learner; publisher; evaluator; monitor). The driver restarts the actor process if it dies (up to 20 times).
  - Rollouts (`[rollout]`, off by default; gen-2.md §6 Phase 5, day 3): `rollout.actors` threads of the actor process play rounds of P (`rollout::rollout_round`, P's top move by default) and value `samples_per_round` decisions with a choice by playing out every legal move on the real deal, P's top move at every seat (`mcts::play_out_greedy`), into `replay-pi/` (read and deleted). P then trains on these instead of search targets (`run.actors` may be 0): `train::pi_loss` = `T · KL(π ‖ π_ref) − Σ π · u`, linear in the per-deal utilities so it averages them, over `micro_batches` batches per step, under `rollout.ratio`. The state after each valued move trains V. Options: `bid_deals` (bids also played out on bid-weighted drawn deals), `rule_bot_2_share` (rule bot 2 at some seats). Held out (`Learner::pi_held_out`, on unseen deals): P's top move's gain over the playing P's and over the start P's, V's pick's gain, by phase. `net-vs0` benches every publish against four copies of the start P. ~1.5M valued decisions/h on 28–30 threads (`examples/rollout_profile.rs`).
  - V stream (`[value_stream]`, off by default): `value_stream.actors` more threads of the actor process play rounds of P alone (`selfplay::policy_round`: P sampled before a random decision U, one uniformly random move there, P's top move after; the states after the random move are kept) into `replay-v/`, which the driver reads into its own pair of buffers and deletes. About 100× a searched round's rate per thread. The learner trains V on them in V-only updates while P waits on the governor, up to `value_stream.ratio` samples per state. A resume starts these buffers empty.
  - Targets: root rule `visits`: root visits at τ = 1 less the one visit per tree every legal move gets (`prune_forced_visits`; without it a 20 × 25 bid puts ≥ 4% on every legal bid); root rule `q`: π', unpruned. Moves drawn from the same distribution: bids at τ = 1, plays greedy; root Dirichlet noise ε = 0.25 by default.
  - Learner: constant LR after a warm-up (1e-4), replay-ratio governor (V samples per training example produced, default 6), a held-out row every 200 steps (self-play validation vs an equal training sample; fixed teacher "probe" states; the V stream's validation vs training sample; the fixed V check: V on the last validation rounds of `eval.value_rounds_from`, by phase, the same states every row and every run), a publish every 400 steps (export, `model.json`, the actors switch; P's KL and V's change on the probe states vs the previous publish and the start).
  - Evaluator: network-only benches at every publish on 4 threads: vs rule bot 2 (256 deals) and the rule bot (128), and vs rule bot 2 on the search bench's deals (`net-rb2-s7`), each paired vs the run's step 0 and the previous publish. A search bench vs rule bot 2 (`--seed 7`, 128 deals) every `eval.search_every_hours` with the actor process frozen (SIGSTOP), with the self-play search settings, paired vs P alone at the same step (the improvement margin), `eval.search_baseline` and the previous search bench; with it, search vs four copies of the same model's P (`search-vsP`, `eval.search_vs_p_deals`), where P alone scores exactly 0: the improvement margin in self-play's own setting. At `run.hours`, the final publish and search benches vs rule bot 2, the same P and the rule bot.
  - Run directory: `status.md` (the run at a glance, rewritten every minute), `metrics.jsonl` (`train`, `held_out`, `probe`, `selfplay`, `publish`, `bench`, `event` rows), `train.log`, `selfplay.log`, `checkpoint/`, `models/step-NNNNNN/{checkpoint,model}`, `replay/`, `replay-v/`, `bench/`. `scripts/plot_rl_run.py <run>` draws `<run>/plots/` (overview, strength, bids, learning, held out, self-play, weight evolution, search's margin and V; venv Python, `LD_PRELOAD` unset).
  - Control files: `STOP` (actors finish their rounds, the learner saves, exit; `--resume` reloads `replay/` and continues; AdamW restarts, so the LR ramps up again over `resume_warmup_steps`), `PAUSE` (actors frozen, learner idle, clock stopped, until removed), `FINISH` (final publish and benches now). A kill loses at most the steps since the last checkpoint (10 min) and the chunk being filled (≤ 1 min).
  - Launch detached: `setsid nohup scripts/blobmaster-train.sh train --config <toml> --output checkpoints/<run> > checkpoints/<run>/train.log 2>&1 < /dev/null &` (create the directory first). Needs `cargo build --release -p blob-bin -p blob-train`.
- Config: `blob-train/pretrain.sample.toml` and `blob-train/train.sample.toml` list every key at its default (tests keep them equal).
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
- `cargo run --release -p blob-engine --example rollout_profile -- <model dir> <threads> <secs> <samples per round>,...` — throughput of rollout rounds (rounds, valued decisions, P calls per round) and how often P's top move is not the best on the deal.
- `cargo run --release -p blob-engine --example rl_value_check -- <run> 0.05 <rounds> <model dir>...` — V of several models (e.g. the warm start and each publish) on the same last `<rounds>` validation rounds of a `train` run, by phase. ONNX only, so it runs beside the run. The held-out rows compare windows that move with the policy; this compares models on fixed positions.

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
