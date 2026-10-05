# BlobMaster — Generation 2

The single source of truth for the remake. It replaces every gen-1 planning document (`development-plan.md`, `fix-mcts-plan.md`, `async.md`, `self-play-profile.md` and others), which were retired on 2026-10-02. §10 maps each one to where its surviving content went and how to read the original from git.

**Gen 2 is a clean break.** The code is rewritten for the gen-2 design only. Nothing is kept to run, train or compare against gen-1 models. Gen 1 survives as evidence (§2–§3) and as two git tags (§10).

Status, 2026-10-05: gen 1 is concluded; Phases 0, 1 and 2 are done; next is Phase 3, the gen-2 engine (§6).

---

## 1. Summary

**What gen 1 built.** Gen 1 (Rust, March–May 2026) produced a fast, well-tested game engine and an AlphaZero-style pipeline. Its final run trained for 168 iterations over about four days (`checkpoints/run-2026-05-14`).

**How strong it is.** The final network **loses to an 80-line rule bot**. With full 5×100 search it scores **−9.2 ± 2.6 points per game** against four copies of the bot and wins 9.8% of games, where a fair share is 20%.

**What went wrong.** Not compute, model size or hidden-card leakage. The training signal never told the network which bids and early-round plays were good:

1. **Wrong value target.** It was the final 17-round score. That number barely depends on any single decision, so the value head memorized games instead of learning positions.
2. **Values reached only one seat.** The search gave each leaf's value to the seat about to move at that leaf. The player making the decision rarely received feedback on its own options, so bid search ended in near-uniform visits.
3. **End-of-round values were clipped and mismatched.** Inside the search they were z-scores of the game score so far, clipped to ±1. That is a different quantity from what the network was trained to predict.

None of this showed up. Every evaluation compared the model with its own earlier checkpoints, and every loss was measured on the data the model was trained on.

**What gen 2 does.**
- **Keeps:** the engine, the encoder skeleton, the transformer, ONNX inference, replay storage, the rule bots, `bench`, `play` and the run tooling.
- **Drops:** everything that exists only to run, train or compare gen-1 models (§4).
- **Changes what the value means:** per round and per seat.
- **Changes how the search uses it:** every seat gets a value at every leaf, from a value network that sees the sampled deal.
- **Changes how strength is measured:** a fixed external opponent plus held-out data.
- **Replaces the training driver:** a supervised warm start from rule bot 2, then async actor–learner self-play over single rounds instead of whole games.

---

## 2. Gen-1 post-mortem (evidence)

All numbers come from `checkpoints/run-2026-05-14` and were measured on 2026-10-02 with `blob-engine/examples/diagnostics.rs` (commands in Appendix A). The table is 5 players, starting at 7 cards. One player sits in a seat that rotates between games; the other four seats are rule bots (`blob-engine/src/rule_bot.rs`), which never run search.

### 2.1 Absolute strength

| Player | Points per game vs the four bots | Win rate (fair 0.20) | Bids made |
|---|---|---|---|
| iter 0, network only | −55.9 ± 1.7 | 0.000 | 32% |
| iter 25, network only | −24.0 ± 1.6 | 0.010 | 51% |
| iter 125, network only | −14.0 ± 1.9 | 0.091 | 58% |
| iter 167, network only | −13.5 ± 1.9 | 0.085 | 59% |
| iter 167, 5×100 search | **−9.2 ± 2.6** | 0.098 | 60% |
| *rule bot, for reference* | — | — | 64% |

Reference points:
- Rule bot vs 4× `HeuristicEvaluator`: +19.2 ± 1.0.
- Rule bot vs 4× random: +70.4 ± 1.0.
- iter 167 with search vs 4× `HeuristicEvaluator`: only +9.8 ± 2.1.

The model kept improving slowly all run (−56 → −24 → −13.5), but levelled off below the rule bot. Search adds only about 4 points over the raw policy. The in-loop 96-game checkpoint-vs-checkpoint evals were too coarse to see any of this.

### 2.2 Root cause 1 — the value target was the final game score

Every decision a seat made in a game got the same label: the z-scored final 17-round score, clipped to ±1 (`blob-nn/src/self_play.rs::backfill_values`).
- **The label is mostly unrelated to the decision.** In rounds 1–6, the score of the round being played correlates only 0.20 with that label. So about 4% of the label's variance is about the decision at hand; the rest is noise from cards nobody has seen yet and from the other 16 rounds.
- **Ideal conditions for memorization.** Each example was re-read about 80 times: a median of 7 epochs per iteration, over a buffer that held about 11 iterations. A 7-card hand nearly identifies its game, and there was no held-out data.

| Value-head error (MSE) | Model | Always predict 0 |
|---|---|---|
| Replay buffer (training data) | **0.031** | 0.563 |
| Fresh self-play games, all decisions | 0.486 | 0.575 |
| Fresh games, rounds 1–6 | **0.796** | 0.575 |
| Fresh games, rounds 13–17 | 0.158 | 0.575 |

- **It memorized.** The value head is near-perfect on its training data. On new games it is barely better than guessing 0, and early in a game it is worse than guessing 0.
- **It knows almost nothing about the round being played.** Its correlation with the current round's score is 0.05. At bid time it is also 0.05.
- **The reported loss hid this.** The value loss falling from 0.64 to 0.05 was memorization, not learning.

### 2.3 Root cause 2 — values went to one seat only

Each search node keeps a separate tally per seat, and a network leaf adds its value only to the tally of the seat about to move there (`blob-engine/src/mcts.rs::backprop`). When choosing for a seat, UCB reads only that seat's tally and treats an empty tally as 0 ("average").

```
seat 0 picks "bid 2"
 └─ seat 1 to bid    ← leaf: "+0.3 for seat 1" → stored only in seat 1's tally
     └─ seat 2 to bid
         └─ …
             └─ seat 0 to play   ← the first point where seat 0's tally gets a number
```

The decider only hears about its own options when a line gets a full lap around the table deeper (5+ moves), or reaches the end of the round. With 100 simulations per sampled deal, most lines never do.

Measured inside the real search:

| | Bid decisions | Play decisions |
|---|---|---|
| Explored root options that never got a value for the deciding player | **70%** | 33% |
| Root simulations that carried a value for the deciding player | 37% | 50% |

### 2.4 Root cause 3 — end-of-round values were clipped and mismatched

At the end of a round, the search scored the position as the z-score of (cumulative score + this round), clipped to ±1 (`blob-engine/src/scoring.rs::terminal_z_scores`).
- **Clipping removed the incentive.** A seat that is clearly ahead or behind sits at the clip, so making or missing the bid changes nothing for it.
- **The numbers didn't match.** The network was trained to predict the *final* game z-score, while these values were the *score-so-far* z-score. The search averaged the two together.

### 2.5 Result: bidding collapsed

The signal ratio is 1 − H(visits)/ln(number of legal moves); 0 means visits spread uniformly. Median values:

| Decision | iter 0 | iter 25 | iter 167 |
|---|---|---|---|
| Bids, 8 legal | 0.79 | 0.10 | **0.004** |
| Bids, 6 legal | 0.73 | 0.08 | 0.007 |
| Plays, 4 legal | 0.16 | 0.12 | 0.10 |

- **Search and network copied each other.** The network learned flat bid priors from flat bid visits. KL from network to search was about 0.03, so the two were nearly identical.
- **Bid loss got worse.** The bid policy loss rose over the run, from 0.82 to 1.07.
- **The bot bid badly.** In self-play it bids 0 on **63%** of all bids. Bids of 3 or more succeed less than 10% of the time: 3 → 9.8%, 4 → 2.8%, 5 → 0.5%, 6–7 → 0%.
- **The high signal at iter 0 was noise.** At iter 0 the random network's search was confidently decisive. Decisive is not the same as right.

### 2.6 Why it went unnoticed — process lessons

1. **Only relative evals.** Checkpoint-vs-checkpoint win rates can't detect "weak in absolute terms". At 96 games they also can't resolve differences under about 10 points, which is why everything from iter 70 onwards was "inconclusive".
2. **Only training-set losses.** A falling value loss was read as learning.
3. **Decisiveness metrics read as quality.** Signal ratio, visit entropy and top-1 share measure whether the search is *decisive*, not whether it is *right*.
4. **Bid success was only compared to itself.** It was averaged over five 1-card rounds per game and never compared with a fixed opponent.
5. **Fixes came from reading code, not from measurements that could falsify them.** `fix-mcts-plan.md` (Dirichlet noise, terminal values, separate temperature for targets and sampling) was correct but treated secondary causes.
6. **Original design decisions were never re-tested** (§2.7).

**Rule for gen 2:** every hypothesis gets a measurement that could prove it wrong, against a fixed external yardstick (§5.7).

### 2.7 Design decisions gen 2 reverses

- **Per-round targets.** `development-plan.md` §5.2 said: *"Training unit is the full game … Round-scoped z-scoring would produce bimodal targets (0 or 10+bid) with poor gradient signal."*
  - The measurement above shows the opposite. The game-level target is the one without usable signal.
  - "Bimodal" is fine: predicting the *expected* round score of a bimodal outcome is ordinary regression.
- **Every seat gets a value at every leaf.** `development-plan.md` §4.1–4.2 credited each leaf only to the seat to move, explicitly to avoid "diluting" Q. That avoided dilution by starving the root (§2.3).

### 2.8 Other confirmed defects

| Defect | Where | Effect | Status |
|---|---|---|---|
| "Not yet bid" encodes like "bid 0, already made" (no `has_bid`) | `encoder.rs` player tokens; `dealing.rs` resets bids to 0 | the network misreads earlier bidders' bids | fixed, Phase 1 |
| No bid-sum / bids-to-come features | `encoder.rs` context token | the network must add up bids through attention | fixed, Phase 1 |
| Seats encoded as absolute one-hots | `encoder.rs` player and played-card tokens | no symmetry across seats; mixed table sizes are hard | fixed, Phase 1 |
| Count features not scaled (up to 13) | `encoder.rs` hand-card features | minor | fixed, Phase 1 |
| `is_highest_in_suit` counts my own higher cards as unseen | `encoder.rs` | minor | fixed, Phase 1 |
| Greedy pick breaks ties to the **last** index | `mcts.rs::visits_to_policy` (`max_by_key`) | with flat bid visits, ties go to the highest bid | fixed, Phase 1 |
| `void_suits` ignores the current trick | `belief.rs` | sampled deals contradict the encoder's void flags | fixed, Phase 1 |
| After 32 failed attempts, sampling drops **all** void constraints | `belief.rs` | about 3–7% of sampled deals ignore known voids | fixed, Phase 1 |
| Eval "heuristic" seats actually run 5×100 search | `blob-nn/src/eval.rs` | the eval opponent is not what it claims | deleted with `eval.rs`, Phase 2 |
| `HeuristicEvaluator` ignores its own bid when playing | `evaluator.rs` | weak, incoherent baseline | deleted, Phase 2 |
| `DynEval` doesn't forward `evaluate_batch` | `eval.rs` | speed only | fixed, Phase 1 (wrapper removed) |
| `blobmaster play` / `analyze` are stubs | `blob-bin` | no way to play the bot properly | `play` built in Phase 0; `analyze` stub deleted, Phase 2 |

---

## 3. What gen 1 established (keep these facts)

### 3.1 Performance on this machine

These hold for a model with d_model = 128, 8 layers and 1.63M parameters, on a Ryzen 9 7950X with an RTX 4060.

**Engine** (criterion):

| Operation | Time |
|---|---|
| `BlobState` copy | 43.6 ns |
| `legal_plays` | 1.0 ns |
| `legal_bids` | 1.2 ns |
| `encode` (5p7c, mid-trick) | 297 ns (272 ns with the Phase-1 layout) |

**ONNX inference:**
- **One sample, one thread:** 0.6–0.8 ms per call.
- **Under self-play load (32 threads, batch of 5):** cost grows about linearly with sequence length:

  | Tokens | Cost per batch-of-5 call |
  |---|---|
  | 12 | 4.5 ms |
  | 20 | 7.1 ms |
  | 26 | 9.4 ms |
  | 32 | 11.6 ms |
  | 37 | 15.2 ms |

  The mean is 17.3 tokens per non-forced decision today, and would be 30.7 with every hand visible.

**Self-play:**
- 97–99% of thread time is spent inside the ONNX call; nothing on the Rust side matters.
- The best setup is 32 threads with lockstep batching of 5 across sampled deals: about **4.7 s per 5p7c game in aggregate**, 1.54× faster than batch 1 at 16 threads.
- The forced-move short-cut saved about 175 s per iteration.

**One iteration of `run-2026-05-14`** (medians):

| Phase | Time | Notes |
|---|---|---|
| Self-play | 523 s | |
| Training | 1,022 s | ~149 s per pass over the 500k buffer, 7 passes |
| ONNX export | ~25 s | |

Training was **two thirds** of the iteration, mostly re-reading the same examples.

**Training step:** about 150 ms per 512-sample step (bid and play sub-batches). That is far slower than the arithmetic needs. `nvidia-smi` showed 95–100% "utilization", but that only says a kernel was running, not that the GPU was compute-bound. Profile before optimizing (§8).

### 3.2 Ruled out — don't retry unless the stated condition changes

| Idea | Result | Retry only if |
|---|---|---|
| GPU-batched self-play inference | 5.3% slower than CPU ONNX ×32 (2026-04-21) | the model grows more than 10× (~15M+ params) |
| INT8 self-play (11 variants) | 1.40× faster, but bid-argmax agreement only 0.84–0.85 (gate 0.95) | quantization-aware training, or a wider model |
| Muon optimizer | identical strength to AdamW at iter 9; bid success −2.2pp | the model is 100M+ params |
| Virtual loss beyond 5 per tree | slower: batch 8 → 1.51×, 16 → 1.05×, vs 1.54× at 5 | the model widens (d_model ≥ 256) |
| More sampled deals for throughput | 6 ≈ 5; 8 → −2%; 10 → −5% | — (more deals for *quality* is a separate question, §5.4) |
| Python for the core loop | gen 0: ~1000× per-operation overhead | — |

### 3.3 Process lessons from gen-1 runs

- **Change one thing at a time.** Run 7.3b bundled four changes and regressed, and nobody could tell which change did it. Gen 2 keeps this without keeping gen-1 code: each training phase starts from the previous phase's benchmarked model (§6).
- **Key the LR schedule to the real progress counter.**
  - Run 7.3b: more epochs per iteration silently compressed the cosine schedule.
  - "Bug #2" (2026-04-28): on resume, the iteration counter and the schedule span disagreed, so the learning rate stayed pinned at its minimum for 14 iterations.
  - In gen 2 the schedule is keyed to learner steps, and the learning rate is logged every metrics row.
- **"Never go below 5×100 simulations" was measured under gen-1's broken values.** Re-measure it once the values carry signal.
- **Carry into the gen-2 driver:** the STOP file; resume with the replay buffer; search budgets driven by config; a per-decision log that is summarized when a run ends (gen 1's raw `decision_stats.jsonl` reached 0.73 GB for one run); and the two visualization scripts, re-pointed at gen-2 outputs.

---

## 4. Component inventory (gen 2 only)

**Clean-break rules.**
- **One of everything.** One encoder layout, one model format (P + V, §5.3), one training driver, one config schema. No version switches, compatibility shims or legacy code paths.
- **A model belongs to the code that trained it.** A layout or head change means retraining. Exported models carry a layout id, and the evaluator refuses a mismatch instead of adapting (§5.5).
- **Gen 1 is evidence, not a dependency.** Its `bench` numbers were measured on the default deals, so gen-2 results compare with them without running a gen-1 model (§5.7). Its code and documents stay reachable through tags (§10).
- **Stale config fails loudly.** The gen-2 config rejects unknown keys, so a gen-1 TOML can't half-load.
- **Code goes when its last caller goes.** Don't park it behind a flag; git history is the archive.

**Keep** (as is, or with small additions)

| Component | Notes |
|---|---|
| Game rules: `card`, `hand`, `state`, `dealing`, `bidding`, `playing`, `round`, `game` + 143 ported tests | Correct and fast. Add a helper that starts one round with given parameters (§5.2). `game.rs` and `cumulative_scores` stay because `bench` and `play` play whole games; no network input reads them |
| `belief.rs` determinization | Fixed in Phase 1. Later: weight sampled deals by the observed bids (§8) |
| `rule_bot.rs`, `rule_bot_2.rs` | Fixed yardsticks (never retune `rule_bot.rs`) and warm-start teachers. Rule bot 2: +14.5 ± 0.3 points/game vs the rule bot (5p/7c); `bid_chances` / `play_chances` give per-action scores for soft targets |
| `bench.rs`, `blobmaster bench` | The §5.7 yardstick. Keeps playing whole games, so results stay comparable with §2 and with rule bot 2. Loads a gen-2 model directory (§5.3) |
| `blobmaster play` | Human vs bots in the terminal; loads a gen-2 model directory |
| `mcts.rs` skeleton: arena, UCB, lockstep batching, forced-move fast path, Dirichlet noise, τ split, tie-break, `signal_ratio` | Backup and leaf evaluation change (next table) |
| `profiling.rs` buckets | Re-attached to the actors (Phase 5) |
| `blob-nn` building blocks: `input.rs`, `transformer.rs`, `heads.rs`; `train.rs` losses, AdamW, grad clip, checkpoint I/O | Shared by both networks |
| `blob-engine/benches/core.rs`; `blob-nn` tests `numerical_stability`, `onnx_tch_parity`, `save_random_checkpoint` | The parity and random-checkpoint tests extend to both networks |

**Rewrite for gen 2**

| Component | Becomes |
|---|---|
| `encoder.rs` | The only layout (the "v2" naming went in Phase 2). Add §5.5 items 7, 8 and 10 |
| `evaluator.rs` | Two batched traits: a policy evaluator (own view → priors) and a value evaluator (sampled deal → ŝ for every seat). `DummyEvaluator` stays for tests |
| `onnx.rs` `OnnxEvaluator` | One session per network; checks the layout id. No layout detection by feature width |
| `scoring.rs` z-score helpers | The per-round utility `u_s` (§5.1) |
| `mcts.rs` backup and leaves | All-seat backup, exact `u_s` at round end, V at leaves, per-phase budgets (§5.4). Per-seat counts, `backprop_terminal`'s z-scores and the "Q = 0 when empty" fallback go |
| `replay.rs` | Same raw-state layout. Per-seat round scores instead of one value, a round id for the validation split, a concurrent wrapper, delta persistence (§5.6) |
| `blob-nn` `model.rs` | P: today's net with the policy heads only. V: a new 4-layer net with an input projection for opponents' hand cards and a per-seat ŝ head |
| `blob-nn` `train.rs` | LR schedule keyed to learner steps; per-seat value MSE on ŝ. `z_score_clip` and the value-head LR group go |
| `blob-nn` `training_loop.rs`, `engine.rs`, `self_play.rs` | A learner module (seeded in Phase 2 as `blob_nn::learner` with the batch construction and held-out-loss code from `training_loop.rs`) and an actor module that plays single rounds (§5.2). Whole-game self-play, `backfill_values` and the synchronous iteration loop were deleted in Phase 2 |
| `blob-train`: `main.rs`, `config.rs`, `config.sample.toml` | Subcommands `pretrain` (Phase 4) and `train` (Phase 5) plus `export` (working since Phase 2), on a new config schema. `evaluate`, `self-play`, `profile`, gen-1 `train` and the gen-1 config were deleted in Phase 2 |
| `scripts/export_onnx.py` | Exports P and V and writes the layout id into the ONNX metadata; `export_script_mirrors_feature_widths` covers both |
| `scripts/visualize_strength.py`, `visualize_weight_evolution.py` | Read gen-2 metrics (keyed by learner step) and model directories; gen-1 formats dropped |
| `blob-engine/benches/onnx_mcts.rs` | P + V search bench. Absorbs the cost-by-sequence-length measurement (`tokens`) from `diagnostics.rs` |
| `AGENTS.md`, `README.md`, `scripts/README.md` | Gen-1 driver, reference-model and `encoder::v1` notes removed (Phase 2); the gen-2 launch template once the driver exists |

**Delete in Phase 2** (done 2026-10-05)

| Component | Why it can go |
|---|---|
| `encoder::v1` and its golden-hash test; `EncoderVersion` and width-based layout detection in `onnx.rs` | Existed only to run gen-1 models |
| `evaluator.rs` `HeuristicEvaluator` | Incoherent baseline (§2.8); the rule bots replace it |
| `blob-nn` `eval.rs`: checkpoint-vs-checkpoint harness, anchor promotion, `strength.csv` | Replaced by `bench`; its "heuristic" seats secretly searched (§2.8). Checkpoint-vs-checkpoint stays possible as `bench --opponent <model>` |
| `blob-nn` `training_loop.rs`, `engine.rs`, `self_play.rs`; `blob-train` `evaluate`, `self-play`, `profile` | The gen-1 driver. Salvage first: see Phase 2 |
| `blob-engine/examples/diagnostics.rs` | Reproduces gen-1 measurements only; kept at tag `gen-1-compat` (Appendix A) |
| `blobmaster analyze` stub | Never implemented; `play --show` and `hint` cover position analysis |
| `blob-train/run-2026-05-*.toml`, `scripts/run-2026-05-*.sh`, `scripts/run-train.sh` | Gen-1 run configs and launchers |
| `checkpoints/run-2026-05-14/` | Gen-1 models stop loading once `encoder::v1` is gone; their numbers are in §2 |
| `logs/` (every tracked gen-1 log, including 10 INT8 ONNX files) | Gen-1 run output; reachable at tag `gen-1-final` |
| Code comments citing retired documents (~50, half in `mcts.rs`) | Most sit in files deleted or rewritten here; remove the rest in the same pass |

**Not in `master`: the `gui` branch** (reviewed 2026-10-02). It is a Tauri + Svelte copilot for a *real* table: the user enters their hand and every played card, and the app shows policy / MCTS visits per card. It adds only `blob-gui/` and a workspace entry, and runs its own per-deal MCTS loop on gen-1 engine APIs, so it won't build against gen 2. Port it in Phase 8. It shares this history, so the `.git` rewrite must include it (§9).

---

## 5. Gen-2 design

### 5.1 Objective and value target

Each round is a fresh deal and round scores add up, so for the decisions inside a round, the goal is that round's points. Gen 2 scores seat `s` in a round as

```
u_s = ŝ_s − λ · mean_{j≠s} ŝ_j          ŝ = round score / (10 + cards_dealt),  score = 10 + bid if exact, else 0
```

- **λ = 1 (default)** is "my points minus the table's", so spoiling an opponent's bid has value.
- **λ = 0** is "my points only".
- **Fixed scale:** no per-game standard deviation, nothing clipped, and nothing in the target that identifies the game.
- **The value net predicts each seat's expected ŝ,** and the search turns that into `u_s`. So λ can change without retraining.
- **Game-level tactics are ignored in gen 2,** such as taking more risk when far behind late in the game (§8).

### 5.2 Self-play in rounds, not games

Per-round targets leave nothing linking rounds, so self-play plays **single rounds**:
- **Sampling:** draw (player count, cards dealt, trump, dealer) from the mix found in real games, or oversample the larger rounds, where bidding matters most.
- **Targets are written as soon as the round ends** (10–40 decisions later). Gen 1 made an example wait for a whole 17-round game.
- **No game-level input:** cumulative scores are gone (Phase 1) and `round_number` goes in Phase 3.
- **Engine change:** add a helper that starts one round with given parameters.

### 5.3 Two networks

| | Policy net P | Value net V |
|---|---|---|
| Sees | the acting player's own view | the whole deal: every hand plus everything public |
| Outputs | bid distribution / per-card scores; no value head | expected round score ŝ for every seat |
| Used for | move priors at every expanded node; the fast no-search player | the value at every search leaf, on the *sampled* deal |
| Trained on | visit distributions at real decisions (τ = 1); bot policies in the warm start | the true full state at each decision → the actual per-seat round scores |
| Size | d = 128, 8 layers (gen 1's) | start at d = 128, 4 layers; grow only if validation loss says so |

**Why V may see every hand.** Inside a sampled deal the search already treats all cards as known. V never sees the *real* hidden cards at play time, only deals sampled from what the player knows.

**Why V doesn't over-promise.** V is trained on rounds played by players who did *not* see each other's hands. So it predicts realistic outcomes, not "everyone plays perfectly with open cards" ones.

**Why V is easier to learn.** With all hands known, a round's outcome is nearly decided. In 1-card rounds, every play is forced, so after bidding the outcome is fully determined. That gives a free exactness test (§7).

**Packaging.** A model is a directory: `policy.onnx`, `value.onnx` and `meta.json` (layout id, learner step, config). `bench`, `play` and the actors take the directory; network-only mode reads only `policy.onnx`.

### 5.4 Search

**Same skeleton as gen 1:**
- several sampled deals per decision, one tree per deal;
- lockstep batching across deals;
- the forced-move fast path;
- root Dirichlet noise in self-play only;
- τ = 1 training targets with τ-scheduled sampling.

**What changes:**
- **Leaves:**
  - Each leaf costs one P call (priors for the seat to move, own view) and one V call (all seats, sampled deal), batched across the deals.
  - At the end of the round, the exact `u_s` is used instead.
- **Backup:** add `u_s` to every seat's sum at every node on the path. UCB reads the acting seat's mean. Per-seat counts and the "Q = 0 when empty" fallback go away.
- **Budget per phase:**
  - A bid's value depends mostly on the hidden cards, so bids use more deals and fewer simulations: start at 20 × 25.
  - Plays keep 5 × 100 until measurements say otherwise.
  - Both are config values per phase.
- **Greedy play** = most visits; ties go to the higher prior (done, Phase 1).
- **Determinization:** voids from the current trick; on fallback, relax only the seat that can't be satisfied (done, Phase 1).

### 5.5 Encoder

P and V share the encoder code. Items 1–6 were done in Phase 1 (as built: §6 Phase 1).

1. **`has_bid` per player.** Derived from the dealer and the current player; no state change needed.
2. **Bid context:** sum of bids so far, bids still to come, (sum − cards)/cards, and my position in bidding order. The dealer-constraint bit already existed.
3. **Seat-relative encoding:** rotate so "me" is seat 0, with a one-hot relative seat on player and played-card tokens. Required for mixed table sizes.
4. **Trick features:** a "winning so far" flag on cards in the current trick; "legal" and "beats current winner" flags on hand cards.
5. **Small fixes:** counts scaled to [0, 1]; `is_highest_in_suit` ignores my own cards.
6. **Cumulative-score features removed.**
7. **Remove `round_number`** from the context token. With single rounds (§5.2) it describes nothing, and no trained model depends on it.
8. **V mode:** opponents' hand cards become a new token type, tagged with the owner's relative seat.
9. **Suit-permutation augmentation** when sampling training batches: relabel suits consistently, including trump; 24 permutations. Cheap, and it multiplies data variety against memorization.
10. **One layout, guarded.** `encoder.rs` holds the only layout. Its `LAYOUT_ID` goes into every exported ONNX file, and `OnnxEvaluator` refuses a model whose id differs. A golden-hash test over fixed states fails on any encoding change, so a change can't land without bumping the id. No old layout is kept: a bump means retraining.

### 5.6 Training: async actor–learner

```
 actors (≈26–28 threads)          shared replay buffer              learner (1 thread + GPU)
 each owns P+V ONNX sessions ──▶  raw BlobState + sparse policy ◀── samples, alternates P / V steps
 plays rounds forever,            + per-seat round scores            LR keyed to learner steps
 swaps models at round ends       ~3% of rounds → validation set     replay-ratio governor
        ▲                                                            every K steps: publish
        └──────────── publisher: export P+V to ONNX, atomic rename, bump version ◀──┘
 evaluator (2–4 reserved cores): network-only bench at every publish, search bench ~hourly
```

- **One learner, two feeds.** The warm start (Phase 4) runs the same learner on a fixed buffer of bot rounds; the async driver (Phase 5) adds the actors, publisher and evaluator around it.
- **Replay-ratio governor.** The learner may use at most R samples per sample produced, starting at R ≈ 4–8; it sleeps when it gets ahead. This replaces "epochs" and directly limits memorization. Log the actual ratio.
- **Validation set by round, not by position.** Positions from one round share a label, so a position-level split would leak. Gen 1 split by whole game, because its label was the game score.
- **Warm-up gate.** The learner starts once the buffer holds at least 50k examples. LR warm-up applies on top.
- **Publishing.** Export both networks every K learner steps (the Python bridge takes ~25 s; run it off the learner thread). Actors swap networks between rounds, never mid-round.
- **STOP file.** Drain the actors, save, exit. Resume continues with the same buffer, so no cold-buffer special case.
- **Buffer persistence as delta chunks.** Save one file per ~N new examples instead of a full 200 MB snapshot every iteration. Resume reloads the newest chunks up to capacity. Tolerate a missing or corrupt chunk by skipping it.
- **Metrics:** one row per minute or so, keyed by learner step. Validation losses next to the same losses on an equal-size training sample, both with dropout off; the logged training losses are not comparable.
- **Not bit-reproducible;** accepted.

### 5.7 Evaluation and diagnostics

**Primary yardstick:** one model seat against four rule bots.
- **`bench` command:**
  - **Modes:** search, or network-only.
  - **Opponents:** the rule bot (the reference), rule bot 2, or any gen-2 model's raw policy. Bots never run search.
  - **Duplicate deals:** a fixed list of deal seeds, each played once from every seat position, so card luck cancels out.
  - **Reports:**
    - points per game ± 95% CI;
    - win share;
    - bids made, split by cards dealt (1 / 2–4 / 5–8);
    - share of 0-bids, and a histogram of bid errors.
- **Gen-1 reference, as recorded:** −10.2 ± 2.9 with 5×100 search, −12.1 ± 2.0 network only, on `bench`'s default deals. Gen-2 results compare with these directly.
- **Cadence:**
  - **Network-only bench at every publish:** about 10 s for 640 games. Gen-1 data shows it tracks strength well (§2.1).
  - **Search bench about hourly:** ~4–5 min for 320 games at 5 × 100 with gen 1's single network; expect about twice that with P + V (§5.8).
- **Held-out checks:**
  - validation losses for P and V;
  - V's correlation with the actual round outcome;
  - V's error on 1-card rounds after bidding, which should approach 0.
- **Search health:**
  - share of root options with a value for the deciding seat, which should be 100% by construction;
  - signal ratio per phase and branching factor.

  Read these only alongside the bench: decisive ≠ right.
- **Human playtests** via `blobmaster play`.
- **Checkpoint-vs-checkpoint** (`bench --opponent <model>`) only as a secondary signal.

### 5.8 Compute budget

**Per-leaf cost** (from §3.1): a gen-1 leaf is one call at ~17 tokens. In gen 2 it's a P call (~17 tokens) plus a V call (~31 tokens):
- **≈ 2.8×** gen 1's cost with a V the same size as P;
- **≈ 1.9×** with a 4-layer V.

**Throughput:**

| | Gen 1 (measured) | Gen 2 (estimated) |
|---|---|---|
| Self-play, all 32 threads | ~310k examples/h | ~110–165k examples/h, before the levers below |
| Learner | ~23k steps/h available on the GPU | ~1.7–2.6k steps/h needed at replay ratio 8 |
| GPU load | — | ~10–25% (P and V both trained) |
| Training share of an iteration | ~2/3 (~80 re-reads per example) | small |

**Self-play on the CPU is the bottleneck in gen 2, not the GPU.** Because inference runs on the CPU, the self-play budget also caps how big the networks can be.

**Levers, cheapest first:**
1. Fewer simulations once V is accurate (re-measure 5 × 50).
2. A smaller V.
3. One call per leaf: V also supplies the in-tree move priors, and P is only used at the root. That's ~1.8× instead of 2.8×, at the cost of a second policy head trained on per-deal visits.

Playing a human stays fast: two networks per move is still a fraction of a second.

---

## 6. Roadmap

Each phase ends with a measurable exit criterion. The order follows build dependencies:
1. Delete gen 1 first, so no later change has to keep it compiling.
2. Build the gen-2 engine, then the networks.
3. Run a supervised warm start: the first measurement of the new value design and search, with no RL.
4. Build the async driver for RL, then do the long run.

Each training phase starts from the previous phase's benchmarked model, so every step's gain is measured on its own (§3.3).

**Phase 0 — Yardsticks first** (done 2026-10-02)
- [x] Rule bots: `rule_bot.rs` (2026-10-02, the reference) and `rule_bot_2.rs` (2026-10-05, +14.5 ± 0.3 vs the rule bot).
- [x] `blobmaster bench` (logic in `bench.rs`): duplicate deals, both modes, bid stats by hand size, 0-bid share, bid-error histogram.
- [x] `blobmaster play`: human vs bots in the terminal. `--show` prints every bot decision's search visits, network policy and value; `hint` does the same for your seat.
- [x] Gen-1 diagnostics (`examples/diagnostics.rs`) and a validation split in the gen-1 driver. Both go in Phase 2; the split's lesson is kept in §5.6.
- [x] Repo hygiene (§9).
- *Exit, met:*
  - `bench` on gen-1 final: **−10.2 ± 2.9** with 5×100 search (64 deals × 5 seats = 320 games, 262 s), −12.1 ± 2.0 network only (128 deals, 7 s). Both agree with §2.1.
  - Full 17-round games ran through `play`'s terminal UI by script. The human playtest moves to gen-2 models (Phase 5).
- *Findings:*
  - **Duplicate deals barely narrow the CI here:** ±2.0 over 128 deals vs ±1.9 for 640 independent games. In Blob, outcome variance comes mostly from play, not from card quality. Keep them anyway: they cost nothing and models are compared on identical cards.
  - **Gen-1 bids 0 in 84% of rounds,** 84% even in 5–8-card rounds, against the rule bot's 25%. It makes only 41% of 5–8-card bids, against the rule bot's 55%.

**Phase 1 — Cheap correctness fixes** (done 2026-10-05)
- [x] Encoder items 1–6 from §5.5 (layout v2, below).
- [x] Determinization fixes.
- [x] Greedy tie-break; `DynEval` removed.
- *Exit, met:* `blob-engine` 272 unit + 44 integration tests, `blob-nn` 48, `blob-bin` 6. ONNX↔tch parity passes on a freshly exported random v2 model.

*As built:*
- **Encoder layout v2** (`encoder.rs`). Token widths: hand 32, played 49, player 28, context 17; padded width `FEAT_DIM` = 49.
  - **Player tokens** come in relative-seat order (me first), with a relative-seat one-hot, `has_bid` and `is_to_move`. Bid, tricks needed and bid status stay 0 until the seat has bid. `has_bid` and `bid_order_position` are in `bidding.rs`.
  - **Played cards:** relative-seat one-hot; `winning_so_far` on the trick in progress.
  - **Hand cards:** counts / 13; `is_legal`; `beats_current_winner`, from `playing::beats` and `current_trick_winner`, which `apply_play` also uses. `is_highest` and `is_lowest` both ignore my own cards.
  - **Context:** bid sum / 13, seats still to bid / players, (sum − cards) / cards, my bidding position. Cumulative scores removed. Void flags read `belief::void_suits`.
  - **Tests** check `is_legal` against `legal_plays`, `beats_current_winner` against actually playing the card, and that rotating every seat leaves the encoding unchanged.
  - The gen-1 layout was frozen as `encoder::v1` so gen-1 models kept running; Phase 2 deletes it.
- **Determinization** (`belief.rs`): `void_suits` includes the trick in progress. After 32 failed rejection attempts, `constrained_deal` deals seat by seat and only draws cards that keep the rest of the deal feasible (Hall's condition over suit sets). It relaxes only seats whose voids can't all be met together, which for a real game state is none.
- **Greedy play:** most visits, ties to the higher root prior, then the lower index. This applies to τ→0 sampling, `root_action_probs` and `bench::search_action` (used by `bench` and `play`). `MctsResult.root_prior` holds the root priors averaged over sampled deals.
- **`DynEval` removed:** `mcts_search` takes `&dyn Evaluator` directly, so batches reach `OnnxEvaluator::evaluate_batch`. MCTS gets the hand order from `encoder::hand_card_indices` instead of encoding a whole state.
- **`scripts/export_onnx.py`** uses the v2 widths; the Rust test `export_script_mirrors_feature_widths` keeps them in sync.

*Open point carried forward:* `export_onnx.py --check` reports 1.9e-5 on a random tch init, over its 1e-5 gate. tch's random init has ~2.5× the weight scale of torch's; with torch's init the script gives 4.8e-7, so the layout isn't the cause. Trained gen-1 weights gave 4.5e-6. Set the gate when the export is rewritten for P and V (Phase 4).

**Phase 2 — Clean break** (done 2026-10-05, no training)

Delete gen-1 support in one pass, before any gen-2 code is written on top of it.
- [x] Tag `c6f0c2a` as `gen-1-final` and the last commit before the deletions as `gen-1-compat` (§10).
- [x] Decide whether to archive `checkpoints/run-2026-05-14/` outside the repo: archived (below), then deleted.
- [x] Salvage from the gen-1 driver: the batch construction (`bid_train_batch`, `play_train_batch`) and the held-out-loss code, with their tests, are in a `learner` module for Phase 4. `blob-train` keeps only `export` until Phase 4.
- [x] Delete everything in §4 "Delete in Phase 2".
- [x] Make the config schema reject unknown keys.
- [x] Rewrite `AGENTS.md`, `README.md` and `scripts/README.md` for gen 2. Drop the gen-1 driver section, the gen-1 reference model and its parity recipe, and the `encoder::v1` notes.
- [ ] Optional: the `.git` rewrite (§9).
- Not yet: `scoring.rs`, the per-seat counts in `mcts.rs` and the single-value `Evaluator` stay until Phase 3 replaces them, because the search needs a value until then.
- *Exit, met:*
  - `cargo build --release` has no warnings. Tests pass: `blob-engine` 279 unit + 44 integration (debug), `blob-nn` 27 (release), `blob-bin` 6.
  - `bench` and `play` run with rule bots and with a random-init model from `save_random_checkpoint`, exported by `blobmaster-train export`:
    - network-only vs the rule bot: −74.9 ± 2.1 (128 deals);
    - 5×100 search: 4 deals in 20 s;
    - full scripted `play` games with rule bots and with search bots (`--show`).
  - ONNX↔tch parity passes on that pair.
  - `bench rulebot2 --deals 4000` reproduces **+14.5 ± 0.3** (bids made 0.725 vs the opponents' 0.639).
  - No code refers to `encoder::v1`, `HeuristicEvaluator`, `backfill_values`, `blob_nn::eval` or a retired document.

*As built:*
- **Tags** (annotated): `gen-1-final` and `gen-1-compat`; see §10.
- **Archive:** `checkpoints/run-2026-05-14/` (223 MB: the four ONNX models, iter-167 `model.ot` and `buffer.bin`, metrics, CSVs) was copied to `~/blobmaster-archive/run-2026-05-14/` on the training machine, hash-checked and deleted from the repo. Appendix A uses it.
- **`blob_nn::learner`** holds `bid_train_batch`, `play_train_batch`, `HeldOutLosses` and `held_out_losses`, now a free function over a `BlobNet`.
  - The validation-split hash is there too, as `is_validation_round`: it takes the round id that the replay buffer gets in Phase 3.
  - Tests: the split, the bid mask, the play-policy scatter onto hand tokens (new) and held-out losses.
  - `blob-nn` now depends only on `blob-engine` and `tch`.
- **`blobmaster-train export`** was a stub; now it runs `scripts/export_onnx.py` with the repo's `.venv` and without `LD_PRELOAD`. It takes a checkpoint directory or `model.ot`, plus `--check`. `blob-train` no longer links `tch`.
- **Layout guard:** `OnnxEvaluator` refuses a model whose `features` width isn't `FEAT_DIM`. `bench` and `play` exit with that message for a gen-1 model. Phase 3 replaces the width check with the layout id (§5.5 item 10).
- **`HeuristicEvaluator`:** its trick helper moved unchanged into `rule_bot.rs`, and the rule-bot test against it was dropped. The test against random opponents stays; `bench rulebot2` covers a stronger opponent.
- **Config:** `MctsConfig` and `TemperatureSchedule` reject unknown keys, tested on TOML. The blob-train config schema went with its last caller; Phase 4's new schema follows the same rule.
- **Comments:** about 100 comments citing retired documents or deleted code were rewritten or dropped. A few comments that had become wrong were fixed on the way: the terminal-leaf backup, the parity test's scope and the `train.rs` LR schedule.
- **Not moved:** the `tokens` measurement from `diagnostics.rs` joins the P + V `onnx_mcts` bench when that is rewritten. Until then it runs at `gen-1-compat` (Appendix A).
- **Left alone:** 62 git-ignored gen-1 `*.log` files in `logs/`, which were never in git. `logs/` otherwise keeps only the rule bot 2 rollout measurements.

**Phase 3 — Gen-2 engine** (no training)
- [ ] Start-one-round helper and the round sampler (§5.2).
- [ ] Per-round utility `u_s` (§5.1), replacing `scoring.rs`.
- [ ] Policy and value evaluator traits; `OnnxEvaluator` with one session per network and the layout-id check (§4, §5.5).
- [ ] Search: all-seat backup, exact `u_s` at round end, V at leaves, per-phase budgets (§5.4). Delete the per-seat counts and the "Q = 0 when empty" fallback.
- [ ] Encoder: V mode, `round_number` removed, layout id with its golden test (§5.5 items 7, 8, 10).
- [ ] Replay: per-seat round scores, round id, concurrent wrapper; suit-permutation augmentation when sampling (§5.5 item 9).
- *Exit:* unit tests pass, including these:
  - every explored root option carries a value for the deciding seat (100%, by construction);
  - for the last bidder in a fully known 1-card round, each bid's search value equals its exact `u_s`;
  - a suit permutation leaves legal moves, trick winners and scores unchanged;
  - V mode encodes every hand; P mode encodes none of the opponents' cards;
  - `bench` and `play` run on the new search with a random-init P + V pair.

**Phase 4 — Two networks + supervised warm start**
- [ ] P and V models (§5.3); `export_onnx.py` for both, writing the layout id; ONNX↔tch parity for both; the model directory.
- [ ] Learner: alternating P / V steps from a buffer, LR keyed to learner steps, validation split by round, a metrics row every N steps, checkpoints. CLI: `blobmaster-train pretrain`.
- [ ] Teacher data: rounds played by rule bot 2, with some rule-bot seats mixed in for variety, stored in the replay format. P imitates rule bot 2's `bid_chances` / `play_chances`; V learns the actual per-seat round scores.
- [ ] G1 on held-out teacher rounds.
- [ ] Bench P network-only and P + V with search, against the rule bot and against rule bot 2.
- *Exit:* G1 and G2 pass.
- *Record:*
  - how much search adds over network-only (gen 1: about 4 points, §2.1);
  - the 0-bid share;
  - bids made by hand size, against both bots.
- No RL yet. This phase measures the new value design and search on their own, which is what the dropped "per-round values in the gen-1 driver" step was meant to isolate.

**Phase 5 — Async self-play RL**
- [ ] Actor–learner per §5.6: actors play single rounds with P + V search; publisher; replay-ratio governor; delta persistence; STOP / resume; an evaluator running `bench` at every publish; metrics. CLI: `blobmaster-train train`.
- [ ] End-to-end smoke test on a tiny config (few actors, small buffer and budgets) before any real run.
- [ ] Short run (a few hours, 5p7c) from the Phase-4 checkpoint.
- [ ] First human playtest (`blobmaster play`).
- *Exit:*
  - G3 and G4 pass;
  - STOP / resume is clean, with no loss spike;
  - the validation–training gap stays flat.

**Phase 6 — Long run, 5 players / 7 cards**
- *Exit:* G5. That means ≥ +20 points per game vs the rule bot, ahead of rule bot 2, bids made in 5–8-card rounds clearly above the rule bot's, and human playtests that feel strong.

**Phase 7 — Mixed table sizes**
- n = 4–7 players, C = 7–8 cards (seat-relative encoding is the prerequisite).
- Fine-tune per table size only if one model lags.

**Phase 8 — Deployment**
- Play UI: port the `gui` branch (§4) to the gen-2 model directory and search API.
- Windows + Intel iGPU via ONNX Runtime (consider the OpenVINO execution provider).
- Per-table-size models if Phase 7 needed them.

---

## 7. Gates

| Gate | Measure | Pass |
|---|---|---|
| G0 yardstick | `bench`, gen-1 final, search | −9 ± 3 reproduced — **passed 2026-10-02: −10.2 ± 2.9**. Historic: gen-1 models are deleted in Phase 2 |
| G1 value learnable | V on held-out teacher rounds (Phase 4) | correlation with the actual round outcome > 0.7; 1-card rounds after bidding ≈ exact |
| G2 search helps | Phase-4 warm start: P + V with search vs P network-only, same deals | search clearly ahead (95% CIs separate) |
| G3 beats the rule bot | search bench | ≥ +10 points/game |
| G4 RL adds strength | Phase-5 run, search bench | clearly above the Phase-4 warm start (CIs separate) and still rising; replay ratio on target |
| G5 strong | long run, search bench | ≥ +20 points/game vs the rule bot; > 0 vs four rule-bot-2 opponents; human playtests |

G3 may already pass at the warm start: rule bot 2 itself scores +14.5. That is fine. G4 is the gate that shows RL adding strength.

**Always also check:**
- the share of root options with a deciding-seat value is 100%;
- validation losses track training losses (a growing gap means memorization);
- the 0-bid share in 5–8-card rounds is not exploding.

---

## 8. Open questions and later work

- **Bid inference in sampling.** Weight sampled deals by how likely the observed bids are under P ("they bid 3, so they hold strength"). Probably the biggest remaining gain for both bidding and play.
- **Strategy fusion.** Inside a sampled deal, opponents act as if they see it. Mitigate with more deals and shallower search; information-set MCTS variants later.
- **Game-aware objective.** Use the standings in the final rounds, e.g. through a standings-conditioned fine-tune.
- **Opponent diversity.** Mix rule bots and past checkpoints into self-play, so the bot doesn't only learn to beat itself. Humans play differently.
- **Training-step efficiency.** ~150 ms per 512 samples for 1.6M params is far above the arithmetic cost. Profile (kernel count, mixed precision, batch size) before buying hardware time.
- **Model size.** The GPU is mostly idle and CPU inference sets the limit (§5.8).
- **Plan B.** DouZero-style "Deep Monte-Carlo": no search; learn Q(state, action) directly from round scores. Worth running as a comparison if search-based training stalls.
- **Opponent modelling across rounds.** Learn each player's style during a game and use it in sampling, rollouts and (later) the networks. See §8.1.

### 8.1 Opponent modelling across rounds

Rounds stay independent for training (§5.2), but people carry habits from round to round. A few per-player style numbers, estimated during the game, carry that information across rounds without coupling them.

**Why it's tractable in Blob.** Every card in a hand gets played, so when a round ends every opponent's full hand is known. Each of their decisions in that round can be replayed from their own view: what they could see, what rule bot 2 would have done there, what they did. Every past decision becomes a labelled example.

**Style parameters**, each measured against rule bot 2 on the reconstructed view:

| Parameter | Measured as | Captures |
|---|---|---|
| Bid bias | bid − rule bot 2's bid on the same hand, seat and earlier bids | systematic over- or under-bidding |
| Bid spread | spread of that residual | erratic vs consistent bidding |
| Control | made (0/1) − rule bot 2's P(make) for that hand and bid | play skill |
| Duck will | share of avoidable tricks taken while at the bid | how reliably they shed tricks (`WILL_DONE` per player) |
| Spite | share of plays that hurt a hungry seat or the leader at a cost to themselves | sabotage, possibly aimed at the leader |
| Temperature | noise of their choices around rule bot 2's ranking of the options | randomness |

Optional covariates: standing (behind/ahead), stage of the game, missed the last round.

**Estimation.**
- Online, with a prior centred on the population: every estimate starts at "average player" and moves only with evidence. Slow forgetting, since people adapt too.
- **Skewed vs random:** a skew is a bias whose credible interval excludes 0; randomness is a high temperature with no bias. The deciding test is prediction: the per-player model must predict that player's *later* decisions better than the population model on held-out rounds. Otherwise shrink back to the population.
- **Data budget (rough):** an opponent gives ~17 bids and ~35 unforced plays per game. If the bid residual has a spread of ~0.7 tricks, one game gives a standard error of ~0.17 tricks on the bias. Strong skews show within a game; subtle ones need several games against the same person.

**Use in the rule bot (first).**
- **Sampling:** weight sampled deals by the likelihood of each opponent's actual bids and plays under *their* model. This is the per-player form of "bid inference in sampling" above.
- **Rollouts:** in v2r, play each opponent as a rule bot 2 with that player's fitted settings. This needs rule bot 2's constants as per-seat parameters (a `Style` input) instead of constants.
- **Strategy:** dump tricks on poor duckers, expect more contested tricks at a table of overbidders, and prefer bids that can be steered either way when leading the game against a known saboteur.
- **Safety:** blend the fitted model with the baseline, so a wrong model costs little.

**Use in the networks (later).** A network trained only to be strong can't adapt mid-game; style has to be an input.
- **Training population:** rule bot 2s with randomly drawn styles (bias, duck will, spite, temperature), mixed with past checkpoints.
- **Inputs:** add per-seat style features (estimate plus uncertainty or observation count) to the player tokens of P and V.
- **Train on estimates, not true styles:** during self-play, run the same online estimator on each opponent's observed history and feed its noisy early-game guesses, so the networks learn how far to trust them.
- **Rounds stay independent:** a training round just carries a style vector per seat. A learned history encoder over past rounds is the stronger but data-hungrier alternative; it would re-couple rounds.

**Order and gates.**
1. **Synthetic opponents with known styles.** The estimator recovers each style within N rounds and doesn't flag random opponents. A style-aware v2r beats a style-blind v2r on `bench`, opponent type by opponent type.
2. **Human games via `play`.** Log decisions; estimates are stable across games for the same person and predict held-out rounds.
3. **Only then** condition P and V on styles, measured against the same synthetic population.

---

## 9. Repo hygiene

**Done 2026-10-02.**
- **`checkpoints/` pruned from 41 GB to 0.22 GB.** All that remains is `run-2026-05-14/`:
  - `iter_000000`, `iter_000025`, `iter_000125`, `iter_000167` (`model.onnx` + `meta.json`; the four rows of §2.1);
  - `iter_000167/model.ot` and `iter_000167/buffer.bin` (never committed);
  - `metrics.jsonl`, `strength.csv`;
  - `signal_ratio_by_iter.csv`, which replaces the 0.73 GB `decision_stats.jsonl`.

  Phase 2 archived it outside the repo and deleted it (§6 Phase 2).
- **`*.onnx` is ignored by git.** Per-iteration weights are never committed; a deliberate reference model goes in with `git add -f`.
- **Deleted:** the INT8 path (calibration capture, `--int8-out`, `use_int8`, `validate_int8.py`, `int8_levers.py`), Muon (`muon.rs`, its param group, `enable_muon`), and the gen-1 sweep, overnight and diagnostic scripts with their configs.
- **Machine-level clean-up:** `target/debug` (13 GB) and the pip download cache (12 GB) were removed. Free disk went from 22 GB to 86 GB.

**Phase 2:** the deletions listed in §4.

**Shrinking `.git` (4.9 GB).**
- **What's in it:** about 4.6 GB of model blobs in history:
  - `sweep-2026-04-28-anchor` 1.4 GB, `run-2026-05-14` 1.1 GB, `run-2026-05-06` 1.0 GB;
  - smaller runs, plus 0.24 GB of gen-0 `.pth` files.
- **When:** any time after Phase 2's deletions are committed. Nothing needs re-adding afterwards, because no gen-1 model is kept.
- **Why it needs care:** it is a history rewrite and a force-push. It must include the `gui` branch on GitHub, which shares this history (§4). Rewriting `master` alone would leave `gui` with diverged history and wouldn't shrink GitHub.
- **Recipe:**
  1. Check that the tags `gen-1-final` and `gen-1-compat` exist (Phase 2).
  2. If §2 should stay reproducible, archive the four gen-1 ONNX models outside the repo. The rewrite removes them from history for good.
  3. Check out `gui` locally.
  4. Run `git filter-repo --force --prune-empty never --invert-paths --path-glob '*.onnx' --path-glob '*.pth' --path-glob '*calibration.bin' --path-glob '*decision_stats.jsonl'` on all branches.
  5. Verify: commit count, HEAD tree, `git show gen-1-final:fix-mcts-plan.md`.
  6. Force-push `master`, `gui` and both tags. Other clones must re-clone.

---

## 10. Retired documents and code

All retired documents are recoverable with `git show gen-1-final:<file>`.

The retired code lives at two tags:
- **`gen-1-final`** (`c6f0c2a`): the gen-1 pipeline as it was trained.
- **`gen-1-compat`** (`3f5a83f`): the last commit whose tooling still runs gen-1 models (`encoder::v1`, `examples/diagnostics.rs`). Appendix A uses it.

| Document | What it was | Where its live content went |
|---|---|---|
| `development-plan.md` | Gen-1 session-by-session plan and specs | Reversed decisions §2.7; perf facts §3.1; ruled-out list §3.2; deployment and fine-tuning ideas §6 Phases 7–8; crate boundaries in `AGENTS.md` |
| `fix-mcts-plan.md` | 2026-05-12 diagnosis (Dirichlet, terminal values, τ split, warm start, forced moves) | Implemented parts kept (§4 `mcts.rs` row); warm start → Phase 4; superseded diagnosis → §2 |
| `async.md` | V2 actor–learner design notes | §5.6 (numbers corrected to `run-2026-05-14` measurements) |
| `self-play-profile.md` | Thread, batch, INT8, determinization-count sweeps | §3.1, §3.2 |
| `7.3b-analysis.md` | Why run 7.3b regressed | §3.3 |
| `gpu-inference-summary.md` | GPU-batched inference experiment | §3.2 |
| `GATES.md` | Gen-1 gate checklist | Replaced by §7; test and bench commands in `AGENTS.md` |
| `conclusion.md` | Gen-0 (Python) post-mortem | Python ruled out (§3.2); "simplest config first" → 1-card exactness test (§5.3, §7) |
| `prepare-migration.md` | Python→Rust migration plan | Done; rules in `README.md` |
| `personal-notes.md` | Delta-buffer and async notes | §5.6 (delta persistence, async), §8 (training-step efficiency) |

---

## Appendix A — Reproducing the gen-1 measurements

After Phase 2 the gen-1 models no longer run on `master`. Use a worktree at `gen-1-compat`. It contains the four reference ONNX models until the `.git` rewrite (§9). The whole directory, including `iter_000167/buffer.bin` (never committed; `diagnostics value` needs it), is archived at `~/blobmaster-archive/run-2026-05-14/` on the training machine; copy it into the worktree's `checkpoints/`.

Run from the worktree root with nothing else busy (the tools use every core).

```bash
git worktree add ../blob-gen1 gen-1-compat && cd ../blob-gen1
mkdir -p checkpoints && cp -r ~/blobmaster-archive/run-2026-05-14 checkpoints/
cargo build --release -p blob-bin -p blob-engine --example diagnostics
B=./target/release/blobmaster
D=./target/release/examples/diagnostics
M=checkpoints/run-2026-05-14/iter_000167/model.onnx

$B bench $M --mode search         # G0 yardstick, duplicate deals (−10.2 ± 2.9, ~4.5 min)
$B bench $M --mode network        # network only (−12.1 ± 2.0, ~10 s)

$D match  $M mcts rulebot 320     # §2.1 headline (~4.5 min)
$D match  $M raw  rulebot 640     # network only (~10 s)
$D match  $M rulebot heuristic 2000
$D value  $M checkpoints/run-2026-05-14/iter_000167/buffer.bin 96   # §2.2–2.5 (~8 min)
$D tokens $M                      # §3.1 cost by sequence length
```

The signal-ratio table (§2.5) is read from `checkpoints/run-2026-05-14/signal_ratio_by_iter.csv` (column `signal_median`). That file summarizes the per-decision log (7.84M decisions, deleted 2026-10-02) by iteration, phase and legal-move count. The other §2.1 rows use `iter_000000`, `iter_000025` and `iter_000125` in place of `iter_000167`.
