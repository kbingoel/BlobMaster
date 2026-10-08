# BlobMaster — Generation 2

The single source of truth for the remake. It replaces every gen-1 planning document (`development-plan.md`, `fix-mcts-plan.md`, `async.md`, `self-play-profile.md` and others), which were retired on 2026-10-02. §10 maps each one to where its surviving content went and how to read the original from git.

**Gen 2 is a clean break.** The code is rewritten for the gen-2 design only. Nothing is kept to run, train or compare against gen-1 models. Gen 1 survives as evidence (§2–§3) and as two git tags (§10).

Status, 2026-10-07: gen 1 is concluded; Phases 0–4b are done; Phase 5's driver is built and its first run done (§6 Phase 5: P alone +10.8 over the warm start vs rule bot 2, search +2.7, then a plateau with search no better than P alone). The supervised warm start (`checkpoints/pretrain-2026-10-06`): P alone matches rule bot 2 (+15.5 vs the rule bot); with V's trick features (layout 4), V passes G1 and search adds 4–5 points (G2), scoring +20.0 against the rule bot and +4.7 against rule bot 2 (§6 Phase 4). Phase 4b (bid-aware sampling: exact 1-card bids and bid-weighted deals) lifts search to +6.4 against rule bot 2, paired +1.7 over Phase 4 (§6 Phase 4b). Day 2 (§6 Phase 5): against rule bot 2 search models its opponents as P, which stopped fitting, so the improvement step is now judged against P: search beats P by ~1 point there, ~2 with an hour of V trained on rounds of P alone (the V stream). Next: the night runs from run 1's step 7600, Q rule with and without the V stream.

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

**Training step:** about 150 ms per 512-sample step (bid and play sub-batches). That is far slower than the arithmetic needs. `nvidia-smi` showed 95–100% "utilization", but that only says a kernel was running, not that the GPU was compute-bound. Profiled in Phase 4 (as built, "Step cost"): the forward and backward passes are bound by activation memory traffic; the optimizer, clipping and batch building cost almost nothing.

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
| `rule_bot.rs`, `rule_bot_2.rs` | Fixed yardsticks (never retune `rule_bot.rs`) and warm-start teachers. Rule bot 2: +14.5 ± 0.3 points/game vs the rule bot (5p/7c); `bid_chances` / `play_chances` give per-action scores for soft targets; `teacher.rs` plays and labels the warm start's rounds (Phase 4) |
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
| `blob-nn` `train.rs` | LR schedule keyed to learner steps; per-seat sigmoid cross-entropy against ŝ (Phase 4: MSE could pin a saturated V at 0). `z_score_clip` and the value-head LR group go |
| `blob-nn` `training_loop.rs`, `engine.rs`, `self_play.rs` | A learner module (seeded in Phase 2 as `blob_nn::learner` with the batch construction and held-out-loss code from `training_loop.rs`) and an actor module that plays single rounds (§5.2). Whole-game self-play, `backfill_values` and the synchronous iteration loop were deleted in Phase 2 |
| `blob-train`: `main.rs`, `config.rs`, `pretrain.sample.toml` | Subcommands `pretrain` (Phase 4) and `train` (Phase 5) plus `export` (working since Phase 2), on a new config schema. `evaluate`, `self-play`, `profile`, gen-1 `train` and the gen-1 config were deleted in Phase 2 |
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

**Why V is easier to learn.** With all hands known, a round's outcome is nearly decided. On the last trick of any round every play is forced, so the deal fully determines the outcome. That gives a free exactness test (§7). It needs V to compare cards across hands, which V learned only once its input said which card beats which (layout 4, Phase 4).

**Packaging.** A model is a directory: `policy.onnx`, `value.onnx` and `meta.json` (layout id, learner step, the checkpoint it was exported from, layer counts). `bench`, `play` and the actors take the directory; network-only mode reads only `policy.onnx`.

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
- **Determinization:** voids from the current trick; on fallback, relax only the seat that can't be satisfied (done, Phase 1). Deals are weighted by the bids already made, under P (done, Phase 4b).
- **1-card bids** are computed, not searched (done, Phase 4b).

### 5.5 Encoder

P and V share the encoder code. Items 1–6 were done in Phase 1 and items 7–10 in Phase 3 (as built: §6).

1. **`has_bid` per player.** Derived from the dealer and the current player; no state change needed.
2. **Bid context:** sum of bids so far, bids still to come, (sum − cards)/cards, and my position in bidding order. The dealer-constraint bit already existed.
3. **Seat-relative encoding:** rotate so "me" is seat 0, with a one-hot relative seat on player and played-card tokens. Required for mixed table sizes.
4. **Trick features:** a "winning so far" flag on cards in the current trick; "legal" and "beats current winner" flags on hand cards.
5. **Small fixes:** counts scaled to [0, 1]; `is_highest_in_suit` ignores my own cards.
6. **Cumulative-score features removed.**
7. **Remove `round_number`** from the context token. With single rounds (§5.2) it describes nothing, and no trained model depends on it.
8. **V mode:** opponents' hand cards become a new token type, tagged with the owner's relative seat.
9. **Suit-permutation augmentation** when sampling training batches: relabel suits consistently, including trump; 24 permutations. Cheap, and it multiplies data variety against memorization.
10. **One layout, guarded.** `encoder.rs` holds the only layout. Its `LAYOUT_ID` goes into every exported ONNX file, and `OnnxEvaluator` refuses a model whose id differs. A golden-hash test over fixed states fails on any encoding change, so a change can't land without bumping the id. No old layout is kept: a bump means retraining. The test hashes P mode and V mode separately: a layout that leaves P's hash unchanged leaves P's input as it was, so P's weights may carry over (`learner.policy_from`) while V retrains.
11. **V mode describes the deal** (layout 4, Phase 4). Every hand card, mine and each opponent's, carries the same features, computed for its owner against the other hands: trump, suit length, highest / lowest in suit and the cards above / below it, legal into the trick in progress, beats the current winner. Opponents' cards also say whether their owner is still to play to the trick. P mode keeps counting against the cards I haven't seen.

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
- [x] Tag the last gen-1 commit as `gen-1-final` and the last commit before the deletions as `gen-1-compat` (§10).
- [x] Decide whether to archive `checkpoints/run-2026-05-14/` outside the repo: archived (below), then deleted.
- [x] Salvage from the gen-1 driver: the batch construction (`bid_train_batch`, `play_train_batch`) and the held-out-loss code, with their tests, are in a `learner` module for Phase 4. `blob-train` keeps only `export` until Phase 4.
- [x] Delete everything in §4 "Delete in Phase 2".
- [x] Make the config schema reject unknown keys.
- [x] Rewrite `AGENTS.md`, `README.md` and `scripts/README.md` for gen 2. Drop the gen-1 driver section, the gen-1 reference model and its parity recipe, and the `encoder::v1` notes.
- [x] Optional: the `.git` rewrite (§9), done 2026-10-05: 4.9 GB → 31 MB.
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

**Phase 3 — Gen-2 engine** (done 2026-10-05, no training)
- [x] Start-one-round helper and the round sampler (§5.2).
- [x] Per-round utility `u_s` (§5.1), replacing `scoring.rs`.
- [x] Policy and value evaluator traits; `OnnxEvaluator` with one session per network and the layout-id check (§4, §5.5).
- [x] Search: all-seat backup, exact `u_s` at round end, V at leaves, per-phase budgets (§5.4). The per-seat counts and the "Q = 0 when empty" fallback are deleted.
- [x] Encoder: V mode, `round_number` removed, layout id with its golden test (§5.5 items 7, 8, 10).
- [x] Replay: per-seat round scores, round id, concurrent wrapper; suit-permutation augmentation when sampling (§5.5 item 9).
- *Exit, met:* unit tests pass, including these:
  - every explored root option carries a value for the deciding seat (100%, by construction): `every_explored_root_option_carries_the_deciders_value` (`mcts.rs`);
  - for the last bidder in a fully known 1-card round, each bid's search value equals its exact `u_s`, at λ = 1 and λ = 0, for every seat: `last_bidder_in_known_one_card_round_values_each_bid_exactly`;
  - a suit permutation leaves legal moves, trick winners and scores unchanged: `relabelling_keeps_legal_moves_trick_winners_and_scores` (`augment.rs`, 96 rounds played move for move on both labellings);
  - V mode encodes every hand, `value_mode_encodes_every_hand`; P mode encodes none of the opponents' cards, `policy_mode_hides_opponents_cards` (re-dealing the hidden cards leaves P's encoding unchanged);
  - `bench` and `play` run on the new search with a random-init P + V pair from `blobmaster-train export --output <dir>`:
    - network-only vs the rule bot: −57.1 ± 1.4 (128 deals, 7 s; gen 1's random init: −55.9);
    - search (bids 20×25, plays 5×100): −47.3 ± 2.4 (64 deals, 516 s; gen 1 took 262 s with one network). Search adds 10 points even to random networks, presumably through the exact values near each round's end (not measured);
    - a full scripted 17-round `play` game against search bots, with `--show`.
  - Totals: `blob-engine` 292 unit + 44 integration (debug), `blob-nn` 27 (release), `blob-bin` 6. `cargo build --release` has no warnings. ONNX↔tch parity passes on P, exported from `save_random_checkpoint`.

*As built:*
- **Single rounds** (`dealing.rs`, `round.rs`):
  - `new_round(RoundParams)` deals one round from player count, cards dealt, trump and dealer.
  - `RoundMix` draws the table size uniformly from a list and cards dealt from the rounds of a game at `start_cards`, so the game's mix (1-card rounds once per player, the others twice). Each round is weighted by `cards ^ large_round_exponent` to oversample large rounds. Trump is uniform over the five, the dealer uniform. Unknown keys are rejected.
- **Utility** (`scoring.rs`): `round_points`, `normalized_scores`, `utilities(ŝ, n, λ)` and `terminal_utilities`. At λ = 1 the utilities sum to zero. `score_round` reads `round_points`. The z-score helpers are gone.
- **Evaluators** (`evaluator.rs`): `PolicyEvaluator` (priors, own view) and `ValueEvaluator` (ŝ per absolute seat, full deal), each with a batch method. `DummyEvaluator` implements both: uniform priors, ŝ = 0.
- **ONNX** (`onnx.rs`):
  - `OnnxPolicy` reads `policy.onnx` (outputs `bid_policy`, `play_scores`, masked to legal moves as before).
  - `OnnxValue` reads `value.onnx` (output `seat_values [B, S]`, read at the player tokens and mapped back from relative to absolute seats).
  - `OnnxEvaluator` holds both, for search. Each file must carry `blob_layout_id` = `LAYOUT_ID` and the right `blob_network` in its metadata, or loading fails with the reason. The feature-width check is gone.
  - Gated tests run with `BLOB_MODEL_DIR=<dir>` (was `BLOB_ONNX_MODEL`).
- **Search** (`mcts.rs`):
  - Nodes keep `visit_count` and per-seat `value_sums`; `backup` adds `u_s` for every seat. `q(seat)` is `None` before the first visit. An unvisited child still scores +∞ in UCB, so every option is tried once before priors and values rank them.
  - Each lockstep step makes one `policy_batch` and one `values_batch` call over its leaves. Terminal leaves use `terminal_utilities`.
  - `MctsConfig` has `lambda` (default 1) and `bid_budget` / `play_budget` (`SearchBudget`, default 20×25 and 5×100). `num_determinizations`, `sims_per_determinization`, `min_sims_floor` and `adaptive_budget` are gone; a config naming them fails to load.
  - `MctsResult.action_values`: the deciding seat's mean utility per action. `play --show` prints it after each move's visit share.
  - `bench` and `play` take `--bid-dets/--bid-sims` next to `--dets/--sims` (plays).
- **Encoder** (`encoder.rs`, `LAYOUT_ID = "layout-3"`; layouts 1 and 2 never carried an id):
  - Context token 16 wide, without the round number.
  - V mode (`encode_value`) inserts every opponent's hand cards between my hand and the played cards: token type 5, width 41 (rank, suit, owner's relative seat, trump flag). Apart from those tokens it equals P mode.
  - `golden_layout_hash` hashes both modes over every decision state of three random games; it is the same in debug and release.
- **Augmentation** (`augment.rs`): the 24 suit relabellings, `permute_suits` for a state, and `hand_position_map` for play policies.
- **Replay** (`replay.rs`):
  - Stores raw states, sparse policies, each seat's round points (absolute seats) and a round id; `phases` is derived from the state.
  - `push_round(decisions, end)` writes a finished round's decisions. Round ids are saved with the buffer, so they continue after a resume.
  - Batches carry `seat_scores`: ŝ relative to the seat to move, V's output order.
  - `sample_batch(n, rng, augment)` gives each example its own relabelling. `SharedReplay` wraps the buffer in an `RwLock`.
  - Delta persistence is Phase 5's.
- **Export** (`scripts/export_onnx.py`; `blobmaster-train export --output <dir> [--checkpoint …]`):
  - Writes the model directory (§5.3). `meta.json` holds the layout id, the learner step (none yet) and each net's depth and weights.
  - P loads a tch checkpoint strictly: a missing or unexpected parameter is an error, except the tch value head, which is skipped. V is random-init (seed 1) until Phase 4 has a tch V. Without `--checkpoint`, P is random-init too.
  - `--check` compares both nets on random inputs that use every token type.
- **blob-nn**: the learner building blocks read `seat_scores`; the tch net's scalar value head trains on the seat to move's ŝ until Phase 4. The parity test compares the tch and ONNX legal policies on bidding and playing states (gate 1e-5), instead of the value.
- **Cost** (random-init P + V; `onnx_mcts` bench, one thread, idle machine):
  - P call 0.55 ms and V call 0.77 ms (batch 1, 5p7c at the first trick);
  - a 5×100 play decision 0.66 s, a 20×25 bid 0.59 s;
  - `bench --mode search`: 4 deals in 38 s on 20 threads, against 20 s with the single gen-1-shaped network at 5×100 in Phase 2: ≈1.9×, as §5.8 estimated for a 4-layer V.
- **Not moved:** the `tokens` measurement (cost by sequence length under 32-thread load) still runs at `gen-1-compat`. It needs a multi-threaded harness rather than criterion, so it waits for the actors (Phase 5).

*Open points carried forward:*
- `export --check` on a random tch init: P 3.0e-5, over the 1e-5 gate (its inputs now use every token type; with the old all-CLS inputs it was 1.9e-5). V gives 1.2e-7 and a torch-initialized P 5e-7. The Rust policy parity on game states passes. Set the gate in Phase 4.
- A random V predicts ŝ ≈ 0.5 for every seat, so search values stay near 0 except close to a round's end. The random-init bench numbers say nothing about the design; Phase 4 is its first measurement.

**Phase 4 — Two networks + supervised warm start** (done 2026-10-06: G1 and G2 pass)
- [x] P and V models in tch (§5.3): P drops the gen-1 value head; V matches `export_onnx.py`'s `ValueNet`. The export loads V's weights too (it writes both nets, the layout id and the model directory since Phase 3); ONNX↔tch parity for V (P's runs since Phase 3).
- [x] Learner: alternating P / V steps from a buffer, LR keyed to learner steps, validation split by round, a metrics row every N steps, checkpoints. CLI: `blobmaster-train pretrain`.
- [x] Teacher data: rounds played by rule bot 2, with some rule-bot seats mixed in for variety, stored in the replay format. P imitates rule bot 2's `bid_chances` / `play_chances`; V learns the actual per-seat round scores.
- [x] G1 on held-out teacher rounds.
- [x] Bench P network-only and P + V with search, against the rule bot and against rule bot 2.
- [x] Conclusion (2026-10-06): layout 4 (V-mode trick features) with V retrained, G1's exactness check on the last trick of every round, 1-card bids from P, c_puct 0.2.
- *Exit:* G1 and G2 pass.
- *Record:*
  - how much search adds over network-only (gen 1: about 4 points, §2.1);
  - the 0-bid share;
  - bids made by hand size, against both bots.
- No RL yet. This phase measures the new value design and search on their own, which is what the dropped "per-round values in the gen-1 driver" step was meant to isolate.

*First run, layout 3 (2026-10-05): G1 half met, G2 met with a lower c_puct.* The conclusion (layout 4) follows the as-built notes.
- **G1 — correlation passes, exactness fails.** V's correlation with the actual ŝ on every validation position is 0.769 (> 0.7). On 1-card rounds after bidding, where the deal decides the outcome, its RMSE is 0.129, not ≈ 0 (`pretrain` checks < 0.05). Diagnosis below.
- **G2 — passes at c_puct 0.2, fails at the default 1.5.** On 128 fresh deals (`--seed 7`) against four rule bot 2s, search scores **+2.6 ± 1.0** and P alone **+0.1 ± 0.5**. At c_puct 1.5 search adds only +0.7 (+0.9 ± 0.8 vs +0.2 ± 0.7): P's priors outvote V. c_puct was picked on the default deals, so the fresh-deal run is the out-of-sample check.
- **G3 already passes** at the warm start: search scores +17.0 ± 2.4 against the rule bot at c_puct 1.5, +19.1 ± 2.6 at 0.2.

*Results* (run `checkpoints/pretrain-2026-10-05`, default config: 1M teacher rounds, 30k learner steps, 80 min; weights git-ignored, on this machine only):

| Held out (every validation position; training sample of equal size) | validation | training sample |
|---|---|---|
| P bid cross-entropy / play cross-entropy | 0.187 / 0.666 | 0.187 / 0.667 |
| P's top move = rule bot 2's, bids / plays | 0.992 / 0.967 | 0.992 / 0.967 |
| V MSE (the targets' variance: 0.137) | 0.0560 | 0.0557 |
| V correlation | 0.769 | 0.771 |
| V RMSE, 1-card rounds after bidding | 0.129 | 0.131 |

- **No memorization:** validation and training sample stayed within 0.005 of each other on every measurement, at every held-out row, and the gap didn't grow (V's correlation: training 0.003–0.005 higher throughout). P saw each of its 13.1M training positions about 1.2 times, V each of its 21.7M about 0.7 times (15.4M samples per net).
- **The curves had flattened:** from step 20000 to 28000 (the last periodic row), V's MSE went 0.0571 → 0.0566 and its correlation 0.764 → 0.767. V's 1-card MSE had stopped falling by step 6000 (0.0168, then 0.015–0.016).

`bench`, 5 players / 7 cards, default seed; search at bids 20×25, plays 5×100:

| Focal player | Opponents | Deals | Points/game | Bids made: 1 / 2–4 / 5–8 cards | 0-bids, 5–8 cards |
|---|---|---|---|---|---|
| rule bot 2 (the teacher) | rule bot | 128 | +15.2 ± 1.5 | 0.820 / 0.720 / 0.648 | 0.423 |
| P, network only | rule bot | 128 | **+15.5 ± 1.5** | 0.820 / 0.720 / 0.649 | 0.423 |
| P, network only | rule bot | 64 | +16.1 ± 2.4 | 0.810 / 0.727 / 0.657 | 0.433 |
| P + V search, c_puct 1.5 | rule bot | 64 | +17.0 ± 2.4 | 0.807 / 0.733 / 0.664 | 0.432 |
| P + V search, c_puct 0.2 | rule bot | 64 | **+19.1 ± 2.6** | 0.801 / 0.737 / 0.693 | 0.412 |
| P, network only | rule bot 2 | 128 | −0.2 ± 0.5 | 0.785 / 0.688 / 0.576 | 0.382 |
| P, network only | rule bot 2 | 64 | +0.2 ± 0.7 | 0.782 / 0.686 / 0.578 | 0.390 |
| P + V search, c_puct 1.5 | rule bot 2 | 64 | +0.9 ± 0.8 | 0.779 / 0.690 / 0.588 | 0.390 |
| same, plays 32×16, bids 64×8 | rule bot 2 | 64 | +0.3 ± 0.8 | 0.781 / 0.686 / 0.582 | 0.390 |
| P + V search, c_puct 0.5 | rule bot 2 | 64 | +1.5 ± 1.4 | 0.770 / 0.694 / 0.602 | 0.389 |
| P + V search, c_puct 0.2 | rule bot 2 | 64 | +2.7 ± 1.7 | 0.771 / 0.694 / 0.616 | 0.375 |
| P + V search, c_puct 0.1 | rule bot 2 | 64 | +2.8 ± 1.6 | 0.769 / 0.687 / 0.623 | 0.354 |
| P, network only, `--seed 7` | rule bot 2 | 128 | +0.1 ± 0.5 | 0.784 / 0.696 / 0.578 | 0.379 |
| P + V search, c_puct 0.2, `--seed 7` | rule bot 2 | 128 | **+2.6 ± 1.0** | 0.768 / 0.705 / 0.615 | 0.373 |

For scale, gen 1 final scored −12.1 (network) and −10.2 (search) against the rule bot (§2.1).

*Findings:*
- **P is a faithful copy of rule bot 2.** Against the rule bot it scores what its teacher does on the same deals (+15.5 vs +15.2), with the same bid statistics to within 0.001. Against rule bot 2 it is even. At step 10000 it was already +14.9 ± 1.5.
- **What search adds over network-only:** +3.0 against the rule bot and +2.5 against rule bot 2 at c_puct 0.2, from bidding and playing larger hands better (5–8-card bids made: 0.616 vs 0.578 against rule bot 2). Gen 1's search added about 4 points, over a much weaker network. At c_puct 1.5 search adds under a point.
- **Why c_puct matters this much:** the teacher target gives rule bot 2's move 0.5 plus its softmax share, and P learned it, so P's priors are sharp. With a top prior of 0.8 against 0.1, 100 simulations and c_puct 1.5, the other move only wins the visit count if V rates it ~0.2 higher in u: about 20 points more likely to make its bid. Spreading the same budget over more deals doesn't help (32×16 / 64×8: +0.3), because the visit counts still follow the priors. Lowering c_puct lets V's values decide.
- **The 0-bid share is not a problem:** in 5–8-card rounds P bids 0 42% of the time against the rule bot, exactly as rule bot 2 does; search lowers it to 41%. Gen 1 bid 0 in 84% of those rounds.
- **For Phase 5:** self-play turns visit counts into policy targets. At c_puct 1.5 they would be P's own priors, so RL would have nothing to learn. Use a lower c_puct in self-play (start at 0.2–0.5 and measure), or softer priors.
- **V's 1-card errors** are not a data-share problem:
  - By cards already in the trick, its 1-card MSE is 0.031, 0.025, 0.018, 0.009 and 0.000 for 0–4; no-trump rounds 0.044, trump rounds 0.010.
  - A V trained only on 1-card positions (712k positions, 8000 steps of 256) plateaus at MSE 0.0105.
  - Typical miss: no trump, I lead 8♥, an opponent holds 10♥, and V predicts that my 8♥ wins. V doesn't learn to compare an opponent's hand card with mine or with the led card.
  - Opponents' hand-card tokens carry only rank, suit, owner and trump, while played cards carry led-suit and winning flags, and my hand cards legality and "beats the winner".
  - Candidate fix (a layout change, so a retrain): give V-mode hand cards the full-information trick features — follows the led suit, beats the current winner, and its rank among the cards of its suit still held by anyone.

*As built:*
- **Teacher** (`blob-engine/src/teacher.rs`, `TeacherConfig`):
  - Rounds come from a `RoundMix` (default 5p/7c, real games' mix). Each seat is rule bot 2, or the rule bot with chance `rule_bot_share` = 0.2. With chance `explore` = 0.1 an unforced move is drawn from the teacher policy instead.
  - **Every decision is labelled with rule bot 2's policy, whoever played it**, so P also gets targets off rule bot 2's own path. The target puts `argmax_weight` = 0.5 on rule bot 2's move; the rest is a softmax at `temperature` = 1 point over each legal move's expected points: `(10 + b) · P(make b)` for a bid (`bid_chances`), `(10 + bid) · P(make)` after a card (`play_chances`). Its top move is always rule bot 2's.
  - **Forced moves are stored.** V needs them: every play of a 1-card round is forced, and G1 measures exactly those. P's learner skips them (`learner::is_forced`); their loss is 0 whatever P outputs.
  - `fill_buffer` plays on every core; round `i` is seeded from `(seed, i)`, so the buffer is the same on any thread count. 1M rounds (22.3M decisions) take 9.6 s.
- **Networks** (`blob-nn` `model.rs`): `PolicyNet` is the input projection, 8 layers and the bid and play heads (1.62M parameters). `ValueNet` is the input projection plus `opp_hand_proj`, 4 layers and `SeatValueHead`, a per-token MLP with a sigmoid read at the player tokens (0.83M). `BlobNet`, gen 1's scalar value head, `z_score_clip`, the value-head LR group and the iteration-keyed LR schedule are gone.
- **V's loss is sigmoid cross-entropy against ŝ**, a soft target in [0, 1], not MSE. Both are minimized by the expected ŝ, but under MSE the stability test's high learning rate pinned V's sigmoid at 0 for good (held-out MSE 0.41: predicting 0 everywhere). Held-out V is still reported as MSE.
- **Learner** (`blob-nn` `learner.rs`, `train.rs`):
  - `Learner` holds both nets, an AdamW each (weight decay 1e-4). One learner step is one P update (bid and play sub-batches, weighted by their sizes) and one V update, 512 examples each.
  - LR: warm-up over 1000 steps, then cosine from 3e-4 to 1e-5 at the last step; a function of the step alone.
  - Grad-norm clip 1.0 that keeps the norm on the GPU (tch's `clip_grad_norm` reads it back every step).
  - Checkpoint: a directory with `policy.ot`, `value.ot` and `meta.json` (`learner_step`), written beside the old one and renamed in. Optimizer state isn't saved (tch can't); a resume restarts AdamW.
  - Held-out: `policy_held_out` (cross-entropy and agreement with the target's top move, per phase) and `value_held_out` (MSE, the targets' variance, Pearson correlation over (state, seat) pairs, and MSE and max error in 1-card rounds after bidding).
- **`blobmaster-train pretrain`** (`blob-train/src/pretrain.rs`; config sections `[data]`, `[teacher]`, `[learner]`, `[log]`, unknown keys rejected; `pretrain.sample.toml` is the defaults):
  - teacher buffer → split by round (3% validation) → 4 loader threads build CPU batches while the GPU trains;
  - `metrics.jsonl`: a training row every 100 steps (LR, mean training losses, steps/s, share of time waiting for batches) and a held-out row every 2000 steps (20k validation examples per net vs an equally large training sample);
  - checkpoint every 5000 steps; at the end, the held-out measurement on every validation example (`held_out.json`, with G1) and the export to `<run>/model`;
  - a `STOP` file saves and exits; `--resume` continues with the run's own `config.toml`, replaying the teacher data from the seed.
- **`blob-train` links tch now.** `scripts/blobmaster-train.sh` runs it with the library path and the CUDA preload; the learner refuses `cuda` when libtorch has none, instead of falling back to the CPU.
- **Export** (`scripts/export_onnx.py`): `--checkpoint <dir>` loads both nets strictly, and `meta.json` gets the learner step. `--check` now draws its random inputs uniformly from [0, 1], the encoder's range (they were N(0, 1)), and gates at 1e-5, relative above 1. A random *tch* init still fails it (P 2.1e-5: tch initializes with ~2.5× torch's weight scale); the Rust parity on game states is the authoritative gate (random init: P 3.3e-6, V 2.1e-6).
- **Replay:** `sample_batch_from(slots, …)` samples from a subset (the training rounds). `sample_batch` draws with `rand::seq::index::sample` instead of scanning every slot, which cost O(buffer) per batch.
- **Step cost** (profiled 2026-10-05, batch 512, RTX 4060): P forward + backward 84 ms, V 67 ms; AdamW and the clip ≈ 0; building a batch 4 ms on a loader thread; host → device 1.6 ms. So 6.4 learner steps/s with the GPU at 100%. Both passes are bound by memory traffic on activations (the 4060 has 272 GB/s), not by arithmetic or launches. fp16 autocast halves both (P 40 ms, V 36 ms) but tch has no gradient scaler, so it isn't used: the warm start fits in ~80 min, and Phase 5 is bound by CPU self-play (§5.8).

*Conclusion, layout 4 (2026-10-06): G1 and G2 pass.* Run `checkpoints/pretrain-2026-10-06`: V retrained on layout 4, P copied from the first run (`learner.policy_from`), every other setting the first run's; 37 min (V alone trains at 15 steps/s against 6.4 for both nets).

**Why V missed: the last trick of every round, not 1-card rounds.** V's RMSE at the seats the last trick still decides (every remaining play is forced, so the deal decides the outcome), on the same 16k fresh teacher rounds for both layouts:

| Cards already in the last trick | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| layout 3, 1-card rounds | 0.178 | 0.158 | 0.134 | 0.098 | 0.003 |
| layout 3, 2–7-card rounds | 0.134 | 0.120 | 0.101 | 0.075 | 0.010 |
| layout 4, 1-card rounds | 0.083 | 0.003 | 0.003 | 0.002 | 0.002 |
| layout 4, 2–7-card rounds | 0.068 | 0.009 | 0.009 | 0.008 | 0.008 |

- The error was the same in every round size, so it wasn't a 1-card quirk, and dropping 1-card rounds from training or from the gate wouldn't have fixed it.
- With only the mover's card left to play, V was exact: that card carried "beats the current winner". With one opponent's card still to come, the error jumped to 0.07–0.10. Opponents' cards carried no such feature.
- Layout 4 is exact once a card is on the table. What's left is the lead (0 cards), where no card is winning yet to compare with.
- The gain reaches earlier positions too: in 7-card rounds V's RMSE with 2 cards left went 0.136 → 0.076, with 3 left 0.196 → 0.156 (there the outcome still depends on play, so it isn't 0).
- So G1's exactness check now covers the last trick of every round, at the seats it still decides (§7). Search never asks V about these positions (it plays forced moves out to the exact result); they are a probe of V's card comparisons, which it does need at earlier leaves.

**The change: V mode describes the deal** (§5.5 item 11, `LAYOUT_ID = "layout-4"`). Every hand card in V mode, mine and each opponent's, carries the same features, computed for its owner against the other hands (suit standing, legal, beats the current winner), plus "owner still to play" on opponents' cards. My cards' standings no longer count undealt cards (in a 1-card round, 47 of the 52). P mode is unchanged: the golden test's P hash is layout 3's, so P's weights carry over.

| Held out (every validation position) | layout 3 | layout 4 |
|---|---|---|
| V MSE (the targets' variance: 0.137) | 0.0560 | **0.0427** |
| V correlation | 0.769 | **0.830** |
| V RMSE, last trick (G1: < 0.05) | 0.110 (the fresh rounds above) | **0.033** (also on the fresh rounds) |
| V RMSE, 1-card rounds after bidding | 0.129 | **0.036** |
| P bid / play cross-entropy | 0.187 / 0.666 | the same (P is the first run's) |

- G1 passed by step 4000 (last-trick RMSE 0.042), where V's MSE was already below the first run's final value. Validation and training sample stayed within 0.005 throughout.

**1-card bids: search loses to P alone, whatever V knows.** In games of 1-card rounds only (`bench --cards 1`, 1000 deals, against rule bot 2), every focal decision is a 1-card bid:

| Focal player | points per game (5 rounds) | bids made |
|---|---|---|
| P alone (identical to rule bot 2: 4000 deals, ± 0.0) | 0.0 | 0.786 |
| layout 3, search, c_puct 1.5 / 0.2 | −0.1 ± 0.1 / **−0.8 ± 0.1** | 0.782 / 0.771 |
| layout 4, search, c_puct 0.5 / 0.2 | −0.8 ± 0.1 / **−0.9 ± 0.1** | 0.772 / 0.770 |

- A near-exact V didn't help, so V's error wasn't the cause. The sampled deals ignore the bids already made, and in a 1-card round those bids are almost the only clue to the opponents' cards. P read them from rule bot 2; search throws them away. The lower the c_puct, the more V's bid-blind values decide, and the more search loses. Rule bot 2r's rollouts lose there the same way (bids made 0.773 vs 0.783, `rule_bot_2.rs` header).
- 1-card rounds are 5 of a game's 17 and about a third of its points, so this cost ~0.8 points per full game.
- **Decision:** 1-card bids come from P (then `MctsConfig::search_one_card_bids`, default off; replaced in Phase 4b by exact 1-card bids, `MctsConfig::one_card_bids`). Self-play inherits it, so RL doesn't train P toward the worse bids. The real fix is bid-weighted sampling, cheapest to build exactly for 1-card bids first (§8).

**c_puct 0.2 is the new default** (`DEFAULT_C_PUCT`, gen 1's 1.5 before): against rule bot 2 on the default deals, 0.2 and 0.1 scored the same (+5.4 ± 1.8, +5.3 ± 1.7). 0.2 is the less extreme of the two, and RL will soften P's priors.

`bench`, 5 players / 7 cards, layout 4, search at bids 20×25, plays 5×100, c_puct 0.2, 1-card bids from P (P alone is the first run's P, so its rows above stand):

| Focal player | Opponents | Deals | Points/game | Bids made: 1 / 2–4 / 5–8 cards | 0-bids, 5–8 cards |
|---|---|---|---|---|---|
| P + V search, c_puct 0.2 | rule bot 2 | 64 | **+5.4 ± 1.8** | 0.782 / 0.695 / 0.644 | 0.373 |
| P + V search, c_puct 0.1 | rule bot 2 | 64 | +5.3 ± 1.7 | 0.782 / 0.692 / 0.645 | 0.356 |
| P + V search, `--seed 7` | rule bot 2 | 128 | **+4.7 ± 1.0** | 0.784 / 0.701 / 0.633 | 0.371 |
| P + V search | rule bot | 64 | **+20.0 ± 2.4** | 0.810 / 0.736 / 0.693 | 0.410 |

- **G2 passes clearly:** on 128 fresh deals (`--seed 7`) against rule bot 2, search scores **+4.7 ± 1.0** and P alone +0.1 ± 0.5 (first run: +2.6 ± 1.0). The network-only reports of the new model directory are identical to the first run's.
- **G3:** +20.0 ± 2.4 against the rule bot (first run +19.1 ± 2.6, P alone +16.1 ± 2.4 on the same deals). That touches G5's numeric bars (≥ +20 against the rule bot, > 0 against rule bot 2), with the CI reaching below +20; G5 also needs the long run and human playtests.
- **What search adds over network-only:** +4.6 against rule bot 2 (fresh deals) and +3.9 against the rule bot (first run: +2.5 and +3.0; gen 1: about 4, over a much weaker network). Against rule bot 2 the gain over the first run's search is +2.1, about 0.8 of it from leaving 1-card bids to P. It comes from the larger rounds: 5–8-card bids made 0.633, against 0.615 for the first run's search and 0.578 for P alone.

**Phase 4b — Bid-aware sampling** (added and done 2026-10-06: both gates pass)

The sampled deals ignored the bids already made (§6 Phase 4, 1-card bids), and the Phase-5 run trains P on search's choices, so the fix comes before that run. The Phase-5 driver doesn't depend on it: it calls search as a black box.
- [x] **Exact 1-card bids** (§8; `one_card.rs`, `MctsConfig::one_card_bids = Exact`, the new default): no tree and no V. The earlier bidders' cards are drawn weighted by how likely each one's bid was under P, the later bidders' bids follow P's policy, the cards play out, and the bid with the best expected `u` is chosen. Replaces "1-card bids from P" (still available as `Policy`).
- [x] **Bid-weighted sampling for every decision** (`belief::sample_deals`, `MctsConfig::bid_weighting`, default on): draw more candidate deals than trees, weight each by the likelihood of every earlier bid under P (from that bidder's bid-time view of the candidate, `belief::rewind_to_bid`), and keep the trees' deals by systematic resampling. A noise floor in the bid model keeps an unlikely bid from ruling a deal out.
- [x] `bench --per-deal-out <file>` and `--compare <file>`: a paired comparison of two runs on the same deals (same seed, deals and table).
- *Gates:*
  - exact 1-card bids beat P alone in 1-card-only games against rule bot 2 (`--cards 1`; Phase 4: P 0.0, search −0.9) — **passed: +0.2 ± 0.1**;
  - search with both beats Phase 4's +4.7 ± 1.0 against rule bot 2 (128 fresh deals, `--seed 7`), with the cost per decision measured — **passed: +6.4 ± 1.1, paired +1.7 ± 1.1 over Phase 4, at +43% wall time**.

*As built:*
- **Exact 1-card bids** (`one_card::one_card_bid`):
  - Each earlier bidder's likelihood is tabled over the 51 cards it might hold (one P call each, from a view that holds that card and placeholders elsewhere: P reads only its own hand and the hand sizes).
  - 2048 deals per bid. The earlier bidders draw their cards in turn, in proportion to their likelihood over the cards left; each deal's weight is the product of the normalizers, so the deals are exact draws from "uniform, weighted by the bids". The later bidders' cards are uniform over the rest.
  - The later bidders' bids are enumerated as lines, each weighted by P's probability given the card and the bids before it, mine included (one P call per distinct line and card, cached). Lines below 1e-4 within a deal are dropped; raising that to 1e-2 changed neither the values nor the cost.
  - The dealer rule enters through P's legal mask. At λ = 1 my bid moves the later seats' expected scores (for example, forcing the dealer), and the values include that.
  - Both policies of the result are one-hot on the chosen bid, as visits would be after unlimited search; `action_values` are each bid's expected `u`.
  - Cost with the warm start's P, one thread: 50 ms (the dealer, 4 earlier bidders, 204 P states) to 340 ms (first to bid, ~1400 states), against ~590 ms for a 20×25 searched bid. P costs ~0.25 ms per state even batched.
- **Bid-weighted deals** (`belief::sample_deals`):
  - Defaults `candidates = 8` per kept deal and `noise = 0.1`: each bidder bids from P's policy with probability 0.9 and uniformly over its legal bids otherwise (`BidWeighting::likelihood`). The noise floor also serves the exact 1-card bids.
  - Cost: one P call per candidate per earlier bidder: 320 per bid on average at 20 deals, 160 per play at 5 deals (4 bidders).
  - Effective sample size (the warm start's P, rule-bot-2 rounds, 80 candidates): about a quarter of the candidates on average, 6% in the worst tenth of decisions (noise 0.1; 0.05 is lower, 0.2 similar). At 5 × 8 candidates a play's 5 trees see 2–3 distinct deals in the hardest cases.
- **Tests:** the rewind recovers every bidder's real bid-time state; systematic resampling keeps each index in proportion; a bid that reveals a card puts that card in every kept deal; the exact 1-card values match hand-computed cases (a sure winner; P(win) at λ = 0; the later bidders' lines and the dealer rule at λ = 1, exact to 1e-5); weighting costs `deals × candidates × bidders` P calls.

*Results* (`checkpoints/pretrain-2026-10-06/bench4b/`; c_puct 0.2, bids 20×25, plays 5×100):

| Focal player | Opponents | Deals | Points/game | Paired vs Phase 4 | Bids made: 1 / 2–4 / 5–8 cards | Wall time |
|---|---|---|---|---|---|---|
| 1-card games: P alone (Phase 4) | rule bot 2 | 1000 | 0.0 | | 0.786 | |
| 1-card games: search (Phase 4) | rule bot 2 | 1000 | −0.9 ± 0.1 | | 0.770 | 86 s |
| 1-card games: **exact** | rule bot 2 | 1000 | **+0.2 ± 0.1** | | 0.784 | 660 s |
| Phase 4 search: 1-card bids from P, uniform deals | rule bot 2, `--seed 7` | 128 | +4.7 ± 1.0 | | 0.784 / 0.701 / 0.633 | 940 s |
| exact 1-card bids, uniform deals | rule bot 2, `--seed 7` | 128 | +5.1 ± 1.0 | +0.5 ± 0.8 | 0.787 / 0.700 / 0.636 | 989 s |
| **exact 1-card bids, bid-weighted deals** | rule bot 2, `--seed 7` | 128 | **+6.4 ± 1.1** | **+1.7 ± 1.1** | 0.787 / 0.715 / 0.636 | 1340 s |
| Phase 4 search (Phase 4 report) | rule bot | 64 | +20.0 ± 2.4 | | 0.810 / 0.736 / 0.693 | 481 s |
| exact 1-card bids, bid-weighted deals | rule bot | 64 | +20.1 ± 2.4 | | 0.812 / 0.743 / 0.688 | 670 s |

- **Exact 1-card bids gain a little:** +0.2 per 5-round 1-card game, so ~+0.2 per full game (5 of its 17 rounds are 1-card rounds). They make about as many bids as P (0.784 vs 0.786) and score more: at λ = 1 they also weigh what the bid does to the later seats. P is rule bot 2's 1-card formula almost exactly, which is already bid-aware, so little was left to gain over P; the fix is that search no longer loses 0.9 there.
- **Bid-weighted deals carry most of the gain:** +1.3 ± 1.1 over exact 1-card bids alone (paired), almost all of it in 2–4-card rounds, where bids made rose from 0.700 to 0.715. In 5–8-card rounds my own 7 cards say more than the bids, and the trees see play as it happens.
- **Against the rule bot nothing changes** (+20.1 against +20.0 on the same deals). The weighting reads bids through P, which bids like rule bot 2; the rule bot bids by a cruder rule, so its bids are misread as often as read. The gain depends on the bid model matching the opponents: in self-play it does by construction, against humans it won't, which is what the noise floor and per-player models (§8.1) are for.
- **The cost is +36% wall time** for the weighting (989 s → 1340 s), more than the P-call count suggests at one thread: under a full 32-thread load each P state costs more. Exact 1-card bids cost +5%.
- **The bench's default deals are reproducible:** rerunning Phase 4's settings with the new code gave the same +4.7 ± 1.0 and the same points, so the old reports stand.

*Open points carried forward:*
- The noise floor and the candidate count were set once, not tuned. Larger counts raise the effective sample size at more P calls; noise 0.1 against 0.05 and 0.2 was compared only by effective sample size.
- Plays carry information too (a seat at its bid ducks): the same weighting could score candidates by the plays made so far. Not built.
- Self-play: the exact 1-card bids' targets are one-hot, so 1-card bids get no exploration from τ; and in self-play the opponents are P, the model the weighting assumes, the best case for it. Against humans the noise floor, then per-player models (§8.1), stand in.

**Phase 5 — Async self-play RL**
- [x] Actor–learner per §5.6: actors play single rounds with P + V search; publisher; replay-ratio governor; delta persistence; STOP / resume; an evaluator running `bench` at every publish; metrics. CLI: `blobmaster-train train` (2026-10-06, as built below).
- [x] End-to-end smoke test on a tiny config (few actors, small buffer and budgets) before any real run: start, PAUSE, a killed actor process, a mid-run search bench, STOP, `--resume`, the final evaluation (2026-10-06).
- [x] Self-play search settings: the warm start's priors are sharp; at c_puct 1.5 visit counts ≈ P's priors (Phase 4), hence the default of 0.2. Check it in self-play (the visit targets must depart from P's priors) before the short run. 1-card bids are exact (Phase 4b): their targets are one-hot on the computed bid. *Checked 2026-10-06:* at c_puct 0.2 search's top move differs from P's in ~12% of bids and ~15% of plays at the start; and every legal move gets one visit per tree, so the raw visits put ≥ 4% on every legal bid at 20 × 25 — targets and moves drawn now leave that visit out (below).
- [x] Short run (a few hours, 5p7c) from the Phase-4 checkpoint: `checkpoints/rl-2026-10-06` (2026-10-06/07, 5.6 h of learning; results below). P alone gains +10.8, search +2.7 (G4's first half); then both plateau and search falls behind P alone, because c_puct 0.2 no longer fits a strong P and V barely learns.
- [x] The improvement margin measured, at every search bench: vs rule bot 2 paired against P alone, and vs four copies of P (day 2, below). The planned c_puct-1.0 run was replaced: at step 7600 neither root rule beats P by more than ~1 point, and V turned out to be the lever.
- [x] Root rule `q`, the V stream, the fixed V check (2026-10-07, day 2 below).
- [ ] Night runs 2a / 2b from step 7600: Q rule with and without the V stream (day 2, below).
- [ ] First human playtest (`blobmaster play`).
- *Exit:*
  - G3 and G4 pass;
  - STOP / resume is clean, with no loss spike;
  - the validation–training gap stays flat.

*As built (2026-10-06):*
- **Two processes.** A search bench run inside the libtorch process crashed in ONNX Runtime while the learner trained (`BiasGelu … GetElementType is not implemented`); the same bench in `blobmaster` ran clean. So the actors are `blobmaster selfplay` (ONNX only, `blob-engine/src/selfplay.rs`), the benches `blobmaster bench` subprocesses, and the driver (`blob-train/src/rl.rs`) holds libtorch alone. They talk through files: `selfplay.json` (settings), `model.json` (the model to play), `replay/chunk-*.bin` (rounds, written beside and renamed in). The driver restarts a dead actor process, freezes it (SIGSTOP) for PAUSE and search benches, and stops it by closing its stdin. A chunk is also the delta persistence: a resume reloads `replay/`.
- **Targets:** root visits at τ = 1 less one visit per tree per legal move (`prune_forced_visits`): an unvisited child scores +∞, so every legal move gets one visit in every tree, ≥ 4% of a 20 × 25 bid's visits each, however bad (KataGo's policy-target pruning). In the smoke test the unpruned bid targets had entropy 1.2 nats against P's 0.17, and 53% of the bids drawn at τ = 1 were not search's top. Moves are drawn from the pruned visits: bids at τ = 1, plays greedy; root Dirichlet noise ε = 0.25, α = 10 / legal moves.
- **Learner:** constant LR 1e-4 after a 300-step warm-up (an open-ended run has no cosine end); replay-ratio governor at 6 V samples per training example produced; validation by round id (5%) in its own buffer over the same window; P samples decisions with a choice by rejection from the live buffer.
- **Measurements:** a held-out row every 200 steps (self-play validation vs an equal training sample; fixed teacher "probe" states: P's agreement with rule bot 2, V's error); at every publish (400 steps) P's KL and V's mean change on the probe states vs the previous publish and the start — the signal an adaptive-LR controller would read, logged only; network-only benches vs rule bot 2 (256 deals) and the rule bot (128) paired vs the run's step 0; search benches vs rule bot 2 every 2.75 h and at the end, paired vs the Phase-4b per-deal file. `status.md` holds the run at a glance; `scripts/plot_rl_run.py` draws `<run>/plots/`.
- **Throughput** (profiled before the run, 30 threads, warm start): 3.8k rounds/h, 92k decisions/h, 37% of them forced; P is 37% of thread time, V 20%, bid weighting 10%.

*First run* (`checkpoints/rl-2026-10-06`, 2026-10-06 23:11 → 2026-10-07 05:11; the defaults above; stopped at step 7696 to test search settings, resumable). 31.1k rounds, 693k examples (39% forced), 19 publishes. The actor process played 5.9k rounds/h (130k examples/h) without a restart; the learner ran ~1,500 steps/h at the replay ratio, its GPU waiting 92% of the time; an export took 5 s. Charts: `checkpoints/rl-2026-10-06/plots/`.

| Learner step (hours of learning) | 0 | 400 (0.3) | 2000 (1.4) | 4000 (2.7) | 5200 (3.9) | 7600 (5.6) |
|---|---|---|---|---|---|---|
| P alone vs rule bot 2 (256 deals) | +0.2 ± 0.4 | +5.9 ± 0.8 | +9.9 ± 0.9 | +10.1 ± 0.9 | +11.3 ± 0.9 | +11.0 ± 1.0 (paired +10.8 ± 1.0) |
| P alone vs the rule bot (128 deals) | +15.5 ± 1.5 | +20.9 ± 1.6 | +22.9 ± 1.7 | +23.6 ± 1.7 | +23.8 ± 1.7 | +24.1 ± 1.6 (paired +8.6 ± 1.4) |
| Search (c_puct 0.2) vs rule bot 2, `--seed 7` | +6.4 ± 1.1 (Phase 4b) | | | +9.2 ± 1.2 (paired +2.8 ± 1.3) | | +9.1 ± 1.5 (paired +2.7 ± 1.4) |

- **P learned search's play in minutes, then plateaued.** After 400 steps (5 min of learning) P alone was +5.7 over the warm start (paired), about what search scored before; +10 by step 2000, ~+11 from step 5200 on. The gain is in the bids: made in 2–4-card rounds 0.694 → 0.731, in 5–8-card rounds 0.579 → 0.687 (search at the warm start: 0.636); 0-bids in 5–8-card rounds 0.379 → 0.314. P's change per publish (probe KL, bids / plays) fell from 0.18 / 0.08 to ~0.005 / 0.003 nats by step 2000; P left rule bot 2 behind (same top move on the probe states 0.99 / 0.97 → 0.80 / 0.78).
- **G4, first half: search is clearly above the warm start** (+9.2 ± 1.2 against +6.4 ± 1.1, paired +2.8 ± 1.3). Second half not met: from step 4000 to 7600 search moved −0.1 ± 1.0 (paired).
- **Search fell behind P alone.** On the same deals, P alone minus search: +0.7 ± 1.1 at step 4000, +1.8 ± 1.0 at step 7600 (at the warm start search was 6.3 ahead). Self-play then trains P on targets no better than itself, and the loop stalls.
- **c_puct 0.2 is part of the cause.** Search settings on the step-7600 networks (`--seed 7`, 128 deals, vs rule bot 2; `rl-2026-10-06/bench/experiments/`, chart `plots/07_search_experiments.png`):

  | Search | Points/game | Paired vs c_puct 0.2 | Paired vs P alone |
  |---|---|---|---|
  | P alone (network only) | +10.8 ± 1.3 | +1.8 ± 1.0 | |
  | c_puct 0.2 | +9.1 ± 1.5 | | −1.8 ± 1.0 |
  | c_puct 0.5 | +10.3 ± 1.5 | +1.2 ± 1.0 | −0.6 ± 0.9 |
  | c_puct 1.0 | +11.2 ± 1.5 | +2.2 ± 1.1 | +0.4 ± 0.8 |
  | c_puct 2.0 | +11.4 ± 1.4 | +2.3 ± 1.1 | +0.5 ± 0.7 |
  | c_puct 1.0, plays 5 × 200 | +10.2 ± 1.5 | +1.2 ± 1.1 | −0.6 ± 0.9 |
  | c_puct 0.2, warm-start V with step-7600 P | +8.9 ± 1.4 | −0.2 ± 1.1 | −1.9 ± 0.9 |

  c_puct 0.2 was chosen when P was a copy of rule bot 2 and V the better judge (§6 Phase 4). Now P is the stronger, and a low c_puct lets V's noisy values overrule it. From 1.0 up, search is back level with P alone (c_puct 1.0 against the warm start's search: +4.8 ± 1.4), but not ahead of it: the improvement margin is ~0 either way.
- **More simulations per tree hurt:** plays at 5 × 200 instead of 5 × 100 (c_puct 1.0) scored −1.0 ± 0.6 (paired), at 1.4× the time. Inside a sampled deal every card is known, and a deeper tree leans harder on that (strategy fusion) and on V's errors. If the budget is a lever, it is more deals with fewer simulations each, not deeper trees.
- **V barely learned.** With the warm start's V in place of the trained one, search scored the same (−0.2 ± 1.1). On the same 600 recent validation rounds (`blob-engine/examples/rl_value_check.rs`), V's MSE was 0.0494 (warm start) → 0.0475 (step 5600), −4%, mostly at bids (0.0829 → 0.0768; plays 0.0399 → 0.0392). The warm start's V got worse as self-play moved away from it (0.0454 on the rounds of steps 0–1200, 0.0494 on those of steps 2800–5600); the trained V kept up but learned little more. Search can't outgrow P while V adds nothing P doesn't already know.
- **P's play targets look noisy.** Search's top card differed from P's in ~16% of plays all run long, P's held-out agreement with search's top play went 0.857 → 0.832, and P's play entropy rose above the targets' (0.67 → 0.79 nats, targets ~0.70): P spreads over moves the 5 × 100 searches don't agree on. The bids converged: P picks search's top bid 95% of the time, KL(target ‖ P) 0.36 → 0.05.
- **No memorization:** P's held-out losses equal the training sample's throughout (step 7600: bids 0.400 / 0.413, plays 0.803 / 0.803); V's gap stayed at 0.0015–0.003 MSE (0.045 / 0.043), not growing.
- **Exploration:** 15% of the bids drawn at τ = 1 were not search's top; P's bid entropy rose 0.17 → 0.41 nats.

*For the next run:*
- Self-play c_puct 1.0, and re-measure it as P improves.
- Re-measure the play budget the other way: more deals, fewer simulations (e.g. 10 × 50, 20 × 25) at c_puct 1.0, paired against 5 × 100.
- Measure the improvement margin at every search bench: a network-only bench on the search bench's deals, paired (seconds). Training on while it is ≤ 0 teaches P nothing; change the search instead.
- V is the bottleneck: it needs to learn more from self-play than its warm start knew. Candidates: lower-variance value targets (search's root value per seat, which needs ŝ backed up beside `u`), more self-play per V update, a larger V. Track V on recent validation rounds against the warm start's V at every publish (`rl_value_check`).
- The run can continue from step 7696 with a new search setting: edit `config.toml`, `train --resume`.

*Day 2 (2026-10-07): reading the first run, and what changed.*

**Reading.** The loop works mechanically (P's gain is real, held-out gaps are flat, optimization is not the limit: P's KL per publish was ~0.005 nats from step 2000 and its held-out loss equals its training loss). What stalled is the improvement step, and the evidence points at how search turns its trees into a move, not only at V:
- **Sampled deals are the lever of a one-step improvement.** Rule bot 2r (rollouts of rule bot 2 on sampled deals, `rule_bot_2.rs` header) gains +0.0 over rule bot 2 at 8 deals, +3.8 at 16, +6.7 at 32, +9.6 at 128: below ~16 deals the noise of comparing moves on few deals eats the gain. Search plays used 5 deals per decision, bids 20.
- **Summed visits are a vote.** Each tree piles its visits onto its own deal's best move, so the sum counts in how many deals a move came out best, not its mean value over the deals. Deeper trees sharpen the vote (5 × 200 lost to 5 × 100), and a higher c_puct only pulls it back to P's prior (c_puct ≥ 1 ≈ P).
- **Distillation averages out the vote's noise.** P trains on the searches of many similar positions, so it learns their mean vote, in effect far more than 5 deals; that is how P passed search. P's play entropy above its targets' (0.79 against 0.70) is the entropy of a mean of inconsistent targets, not a defect of P.
- **V is short of data, not of capacity or optimization:** 31k self-play rounds against the warm start's 1M teacher rounds, the GPU idle 88–92% of the time, a small and flat held-out gap. The held-out V MSE rose (0.040 → 0.045) because the validation window moves with the policy; only fixed rounds compare V over time.
- **Search's root value is not a valid V target.** V sees the real deal; the root value averages over deals sampled from the mover's view. Training V(real deal) toward it would teach V to ignore the hidden cards. The valid low-variance alternative is TD(λ) along the real trajectory.

**Decomposition** (`checkpoints/rl-2026-10-06/bench/day2/`; step-7600 networks, `--seed 7`, 128 deals vs rule bot 2, paired against P alone on the same deals). `--dets 1 --sims 1` expands only the root, so visits tie and the move is P's top; with `--bid-dets 1 --bid-sims 1 --one-card-bids policy` too, the search path reproduces P alone exactly (paired 0.0 ± 0.0).

| Search (c_puct 1.0, visits) | Paired vs P alone |
|---|---|
| bids only (20 × 25, exact 1-card bids), plays from P | +0.3 ± 0.5 |
| plays only (5 × 100), bids from P | +0.0 ± 0.7 |
| bids 20 × 25, plays 20 × 25 (4× the deals at the same budget) | +0.3 ± 0.7 |
| both, default budgets (run 1's experiment) | +0.4 ± 0.8 |

At c_puct 1.0 neither half of the search adds anything over P, and more deals don't help while the visit counts decide.

**The Q rule** (same deals and pairing; built below). T in utility units (a point of a 7-card round ≈ 0.06); `--bid-candidates 2` with 32–64 deals:

| Search | Paired vs P alone | Wall time (shared CPU) |
|---|---|---|
| plays only, 32 × 16, T 0.05 | +0.5 ± 0.8 | 28 min |
| plays only, 64 × 8 (each move valued once per deal: depth 1), T 0.05 | −0.0 ± 0.8 | 44 min |
| bids and plays 32 × 16, T 0.05, c_puct 1.0 | +0.3 ± 0.9 | 69 min |

Neither averaging the values over the deals nor 6–13× the deals beats P against rule bot 2. At depth 1 the move is only as good as V's ranking of the children, so this points back at V, or at a P that one step of improvement can't beat.

**Against rule bot 2, search models the opponents wrongly.** Search plays the other seats as P inside its trees, and reads their bids through P (bid-weighted deals, exact 1-card bids). That fit when P was rule bot 2's copy; in run 1, P left it behind (same top move 0.99 → 0.80), and search's margin over P against rule bot 2 fell from +6.3 to −1.8 meanwhile. The sharpest case: the round played out by P's top move at every seat from each move (root rule `rollouts`: an exact critic of P's own play, no V, no tree; rule bot 2r with P in its place, which gained +9.6 over rule bot 2 at 128 deals) scored **−4.0** against P alone on 16 deals against rule bot 2 (+10.1 against +14.1 on the same deals; 60 min on every core). So against rule bot 2 the improvement step is mostly measured against a wrong opponent model, while self-play, where the targets are made, has the right one. The step is judged in self-play's own setting from here: search against four copies of P, where P alone scores exactly 0 (checked: 0.0 ± 0.0). Rule bot 2 stays the yardstick for P alone, which models nobody.

| Against four copies of step 7600's P (`--seed 7`, 128 deals) | Points/game |
|---|---|
| P alone | 0.0 ± 0.0 |
| search, root rule `q`, bids and plays 32 × 16, T 0.05, c_puct 1.0 | **+1.0 ± 1.0** |
| the same with the V-stream test's V (below) | **+1.9 ± 1.0**; paired vs step 7600's V **+0.9 ± 0.8** |
| search, root rule `visits`, default budgets (bids 20 × 25, plays 5 × 100), c_puct 1.0 | +0.9 ± 0.8 (the `q` row minus this, paired: +0.1 ± 1.0) |

| root rule `rollouts` (each move played out by P's top move at every seat), 32 deals, T 0 (32 deals of the list: 72 min on every core) | −1.5 ± 3.2 (5–8-card bids made 0.614 against the opponents' 0.656) |

In self-play's setting both root rules improve on P by about a point, and equally: at step 7600 the root rule doesn't matter, V does. The Q rule stays for the next runs: it reads V's values directly rather than through a vote, so it should gain more as V improves; it costs ~20% more time per bench.

The rollout critic is exact per deal but not over 32 deals: a round played out by greedy P is one deterministic outcome, so 32 deals put the chance of making a bid within about ±0.09, too coarse to beat a bid policy distilled from millions of positions (its large-round bids were made less often). V's expectations vary less across deals than single outcomes do, which is why V-based search does better here than the exact critic. More deals per decision cost P calls the self-play budget doesn't have (~1 ms per P state).

**The V-stream test** (`checkpoints/vfit-2026-10-07`, 1 h): step 7600's networks, 6 threads playing rounds of P alone, 2 search actors (P took no step), so only V changed: ~14k V-only updates of 512 at LR 1e-4, ratio 3. V's MSE on the stream's held-out rounds 0.0225 → 0.0204 (validation = training sample); on run 1's fixed rounds 0.0450 → 0.0460 (bid states 0.0741 → 0.0790, play states 0.0365 → 0.0364): V moved from the value of run 1's search play toward that of P's play. With it, the Q-rule search gains +0.9 ± 0.8 against P (paired, same P): **one hour of V on P's own play nearly doubled the improvement margin.** The first change of the day that improves the improvement step, and the evidence that V, fed the right data, is the lever. Against rule bot 2 the same V changed nothing (paired −0.2 ± 0.8 against step 7600's V; +0.1 ± 1.0 against P alone): V learned P's play, which rule bot 2 doesn't play. Whether gains in self-play reach rule bot 2 is P alone's to show (run 1's did: +10.8).

Run 1's P against its own earlier P's (network only, 256 deals): step 2000 vs 400 **+4.4 ± 0.7**; 7600 vs 2000 +1.0 ± 0.6; 4000 vs 2000 +0.3 ± 0.5; 7600 vs 5600 +0.4 ± 0.5. The self-play plateau is real too: the loop's gains shrank with its improvement margin.

**Changes (as built, 2026-10-07):**
- **Root rule `q`** (`MctsConfig::root_rule`, `q_temperature`; `bench --root q --q-temp T`; `[selfplay.search] root_rule, q_temperature`). Per tree, each root move's mean utility for the deciding seat; Q̄ = their mean over the trees, each tree weighted equally (a move no tree visited gets the roots' mean value); π' ∝ P · exp(Q̄ / T) over the legal moves, from the noise-free prior; T = 0 takes the best Q̄. π' is the target, the sampling distribution (bids τ = 1) and the greedy move; nothing is pruned. Regularized policy improvement (Grill et al. 2020; Gumbel MuZero): T, not c_puct, sets how far the values may move P, whatever P's sharpness. With more simulations per tree than legal moves, every tree values every move (an unvisited child scores +∞), so the trees compare the moves on the same deals. Arenas now reserve `1 + 16 · sims` nodes, not 4096, so 32–64 small trees are cheap.
- **The V stream** (`[value_stream]`; `selfplay::policy_round`). AlphaGo trained its value net on games of its policy net, not of search; this does the same. `value_stream.actors` threads of the actor process play rounds of P alone: a decision U drawn uniformly, P sampled at τ = 1 before it, one uniformly random legal move at the first decision with a choice from U on, P's top move after it. The states after the random move are kept, so V learns the value of greedy P after one move off P's policy: what a depth-1 improvement over P needs from its critic, and what search asks at its root's children. They go to `replay-v/`, which the driver reads into its own pair of buffers (split by round id) and deletes. V-only updates (`Learner::train_value_on`, not a learner step; V's LR follows its own update count) run while P waits on the governor, up to `value_stream.ratio` samples per state. Throughput: ~60k rounds/h per thread under a full bench load (~110× a searched round per thread), ~11 states kept per round.
- **The driver:** a training row at least every two minutes while anything trains; the starting held-out row comes before any update (V-only updates can start before P's first step); a final held-out row; a final publish whenever V changed since the last one.
- **Measurements:** a network bench on the search bench's deals at every publish (`net-rb2-s7`), so every search bench is paired against P alone at the same step (the improvement margin); search benches run with the self-play search settings (they ran with `bench`'s defaults before); the fixed V check (`eval.value_rounds_from`: the last 600 validation rounds of run 1, by phase, the same states at every held-out row and in every run); the V stream's validation vs training sample; `scripts/plot_rl_run.py` draws `08_value_margin.png` (search's margin, the fixed V check, the V stream, the learner's pace). The fixed V check scores V against outcomes of run 1's search play: as V moves toward the value of P's play, its MSE there can rise while it becomes the better critic.
- **Tests:** the Q rule follows the mean values against a P that prefers another move, T = 0 / large / closed form, the noise-free prior under root noise; self-play rounds under the Q rule; P-alone rounds record greedy play after the random move; STOP / `--resume` with the V stream; end-to-end smoke runs.
- **Search vs P in the driver** (`eval.search_vs_p_deals`, default 64): with every search bench, search against four copies of the same model's P (`search-vsP`), the improvement margin where the targets are made.

**The night of 2026-10-07: two runs from run 1's step 7600** (`checkpoints/night-2026-10-07.sh` runs them back to back; configs `checkpoints/rl-2026-10-07a.toml`, `…b.toml`). Both: root rule `q`, T 0.05, c_puct 1.0, bids and plays 32 × 16, 2 candidates per deal, no root noise (bids explore by sampling π'), 24 search actors, 3.5 h, search benches at 1.75 h and at the end (vs rule bot 2 paired against P alone, and vs P).
- **2a:** with the V stream (4 threads, ratio 3).
- **2b:** without it: the control for what the V stream adds in the loop (the Q rule alone adds ~+1 vs P).
- *Pass marks, set before the runs:* P alone vs rule bot 2 (256 deals) paired vs step 0 (= run 1's step 7600) above 0 by the end; `search-vsP` above 0 at both benches and not falling; for 2a, V's MSE on the stream's held-out rounds falling and 2a's gains above 2b's. Guardrails as before (held-out gaps flat, 0-bids in 5–8-card rounds not drifting).

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
| G1 value learnable | V on held-out teacher rounds (Phase 4) | correlation with the actual round outcome > 0.7; the last trick ≈ exact at the seats it still decides, every round size (`pretrain` checks RMSE < 0.05; until 2026-10-06 the check was 1-card rounds only) — **passed 2026-10-06 with layout 4:** correlation 0.830, last-trick RMSE 0.033 (layout 3: 0.769, 1-card RMSE 0.129; §6 Phase 4) |
| G2 search helps | Phase-4 warm start: P + V with search vs P network-only, same deals | search clearly ahead (95% CIs separate) — **passed 2026-10-05 at c_puct 0.2:** +2.6 ± 1.0 vs +0.1 ± 0.5 against rule bot 2, fresh deals; not at c_puct 1.5. Layout 4 (2026-10-06): **+4.7 ± 1.0** |
| G3 beats the rule bot | search bench | ≥ +10 points/game — met by the warm start: +17.0 ± 2.4 (c_puct 1.5), +19.1 ± 2.6 (0.2); layout 4: +20.0 ± 2.4 |
| G4 RL adds strength | Phase-5 run, search bench | clearly above the Phase-4 warm start (CIs separate) and still rising; replay ratio on target |
| G5 strong | long run, search bench | ≥ +20 points/game vs the rule bot; > 0 vs four rule-bot-2 opponents; human playtests |

G3 may already pass at the warm start: rule bot 2 itself scores +14.5. That is fine. G4 is the gate that shows RL adding strength.

**Always also check:**
- the share of root options with a deciding-seat value is 100%;
- validation losses track training losses (a growing gap means memorization);
- the 0-bid share in 5–8-card rounds is not exploding.

---

## 8. Open questions and later work

- **Bid inference in sampling** (done, Phase 4b). Weight sampled deals by how likely the observed bids are under P ("they bid 3, so they hold strength"). Next: weight by the plays made too, and per player (§8.1).
- **Strategy fusion.** Inside a sampled deal, opponents act as if they see it. Mitigate with more deals and shallower search; information-set MCTS variants later.
- **Game-aware objective.** Use the standings in the final rounds, e.g. through a standings-conditioned fine-tune.
- **Opponent diversity.** Mix rule bots and past checkpoints into self-play, so the bot doesn't only learn to beat itself. Humans play differently.
- **Exact 1-card bids** (done, Phase 4b). A 1-card round has one real decision per seat (every play is forced), worth as much as any other bid, and 5 of a 17-round game's rounds. It can be computed instead of searched: sample the hidden cards weighted by how likely each earlier bid was under P, take the later bids from P's policy, play out the forced cards, and pick the bid with the best expected `u`. No V at all. It is bid inference in sampling (above) in its smallest form, so it is also the place to test that idea first. Rule bot 2's 1-card formula already weights cards by their holder's bid (`BID_NOISE`), a starting point for the weights. Done in Phase 4b (`one_card.rs`).
- **Training-step efficiency.** Profiled in Phase 4: P's forward + backward is 84 ms per 512 examples and V's 67 ms, bound by activation memory traffic. fp16 autocast halves both, but tch has no gradient scaler; worth adding (a manual loss scale, or bf16) only if the learner ever limits a run.
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

**Shrinking `.git` (4.9 GB → 31 MB, done 2026-10-05).**
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
- **Result (2026-10-05):**
  - **Size:** `.git` went from 4.9 GB to 31 MB; `fsck` is clean.
  - **History:** commit counts are unchanged on every ref (`--prune-empty never`), and `master`'s tree is byte-identical.
  - **Tips:** `gui`, `gen-1-final` and `gen-1-compat` lost exactly their 272, 265 and 14 model files.
  - **Local-only branch:** `feature/gpu-server-experiment` was rewritten too; its history held no matching files.
  - **Remote:** `master`, `gui` and both tags were force-pushed.
  - Every commit hash before the deletions changed; the tags keep pointing at the right commits. A clone from before 2026-10-05 must re-clone.

---

## 10. Retired documents and code

All retired documents are recoverable with `git show gen-1-final:<file>`.

The retired code lives at two tags:
- **`gen-1-final`** (`3cecbb3`; `c6f0c2a` before the §9 rewrite): the gen-1 pipeline as it was trained.
- **`gen-1-compat`** (`302b8f8`; `3f5a83f` before the §9 rewrite): the last commit whose tooling still runs gen-1 models (`encoder::v1`, `examples/diagnostics.rs`). Appendix A uses it.

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

After Phase 2 the gen-1 models no longer run on `master`. Use a worktree at `gen-1-compat`. Since the `.git` rewrite (§9) it holds no models. The whole reference directory, including `iter_000167/buffer.bin` (never committed; `diagnostics value` needs it), is archived at `~/blobmaster-archive/run-2026-05-14/` on the training machine; copy it into the worktree's `checkpoints/`.

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
