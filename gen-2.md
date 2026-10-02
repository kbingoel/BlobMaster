# BlobMaster — Generation 2

The single source of truth for the remake. It replaces every gen-1 planning document (`development-plan.md`, `fix-mcts-plan.md`, `async.md`, `self-play-profile.md` and others), which were retired on 2026-10-02. §10 maps each one to where its surviving content went and how to read the original from git.

Status, 2026-10-02: gen 1 is concluded; gen 2 is at Phase 0 (§6).

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
- **Keeps:** the engine, the encoder skeleton, the transformer, ONNX inference, replay storage and the run tooling.
- **Changes what the value means:** per round and per seat.
- **Changes how the search uses it:** every seat gets a value at every leaf, from a value network that sees the sampled deal.
- **Changes how strength is measured:** a fixed external opponent plus held-out data.
- **Replaces the training driver:** async actor–learner, playing rounds instead of whole games.

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

### 2.8 Other confirmed defects (fix during gen 2)

| Defect | Where | Effect |
|---|---|---|
| "Not yet bid" encodes like "bid 0, already made" (no `has_bid`) | `encoder.rs` player tokens; `dealing.rs` resets bids to 0 | the network misreads earlier bidders' bids |
| No bid-sum / bids-to-come features | `encoder.rs` context token | the network must add up bids through attention |
| Seats encoded as absolute one-hots | `encoder.rs` player and played-card tokens | no symmetry across seats; mixed table sizes are hard |
| Count features not scaled (up to 13) | `encoder.rs` hand-card features | minor |
| `is_highest_in_suit` counts my own higher cards as unseen | `encoder.rs` | minor |
| Greedy pick breaks ties to the **last** index | `mcts.rs::visits_to_policy` (`max_by_key`) | with flat bid visits, ties go to the highest bid |
| `void_suits` ignores the current trick | `belief.rs` | sampled deals contradict the encoder's void flags |
| After 32 failed attempts, sampling drops **all** void constraints | `belief.rs` | about 3–7% of sampled deals ignore known voids |
| Eval "heuristic" seats actually run 5×100 search | `blob-nn/src/eval.rs` | the eval opponent is not what it claims |
| `HeuristicEvaluator` ignores its own bid when playing | `evaluator.rs` | weak, incoherent baseline |
| `DynEval` doesn't forward `evaluate_batch` | `eval.rs` | speed only |
| `blobmaster play` / `analyze` are stubs | `blob-bin` | no way to play the bot properly |

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
| `encode` (5p7c, mid-trick) | 297 ns |

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

- **Change one thing at a time.** Run 7.3b bundled four changes and regressed, and nobody could tell which change did it.
- **Key the LR schedule to the real progress counter.**
  - Run 7.3b: more epochs per iteration silently compressed the cosine schedule.
  - "Bug #2" (2026-04-28): on resume, the iteration counter and the schedule span disagreed, so the learning rate stayed pinned at its minimum for 14 iterations.
  - In gen 2 the schedule is keyed to learner steps, and the learning rate is logged every metrics row.
- **"Never go below 5×100 simulations" was measured under gen-1's broken values.** Re-measure it once the values carry signal.
- **Keep:** the STOP file, resume with the replay buffer, the `[mcts]` budget being driven by config, `decision_stats.jsonl` (summarize it when a run ends; the raw file reached 0.73 GB for one run), and the two visualization scripts.

---

## 4. Component inventory

| Component | Verdict | Notes |
|---|---|---|
| Game rules: `card`, `hand`, `state`, `dealing`, `bidding`, `playing`, `round`, `game` + 143 ported tests | **Keep** | Correct and fast. Add a helper to start a single round directly (§5.2) |
| `scoring.rs` z-score helpers | **Replace** | Use the per-round utility (§5.1) |
| `belief.rs` determinization | **Keep + fix** | Current-trick voids; partial fallback; later, weight sampled deals by the observed bids (§8) |
| `encoder.rs` | **Keep structure, change features** | §5.5, plus a full-deal mode for the value net |
| `mcts.rs` arena, UCB, lockstep batching, forced-move fast path, Dirichlet noise, separate τ for targets and sampling | **Keep** | Change the backup to all seats, the end-of-round value and the tie-break (§5.4). Delete per-seat counts |
| `onnx.rs` `OnnxEvaluator` | **Keep + extend** | Two sessions (policy and value), per-seat value output |
| `rule_bot.rs` (new, 2026-10-02) | **Keep** | Fixed benchmark opponent and warm-start teacher |
| `evaluator.rs` `HeuristicEvaluator` | **Drop** once eval uses the rule bot | Incoherent baseline |
| `replay.rs` storage (raw `BlobState` + sparse policy) | **Keep layout** | Per-seat round scores instead of one value; concurrent wrapper; round-level validation split; delta persistence |
| `blob-nn` transformer, input projections, heads | **Keep** | Per-seat value head; second (value) model |
| `blob-nn` `self_play.rs` | **Rewrite** | Per-round targets; play rounds, not games |
| `blob-nn` `training_loop.rs`, `blob-train` driver | **Replace** with an async driver | Reuse the batch construction, train step, metrics, STOP, export call |
| `blob-nn` `eval.rs` | **Replace** with `bench` | Keep the Wilson CI helper |
| `muon.rs`, INT8 path (`use_int8`, `validate_int8.py`, `int8_levers.py`, `--int8-out`) | **Delete** | Ruled out (§3.2) |
| `scripts/export_onnx.py` | **Keep + extend** | Second model; per-seat value; the Python mirror must match the Rust model |
| `scripts/visualize_*.py` | **Keep** | Re-key to learner steps |
| Gen-1 sweep, overnight and diagnostic scripts | **Delete** | Tied to gen-1 runs |
| `blob-bin` | **Build** | `play` (human vs bot) early (Phase 0) |
| `gui` branch on GitHub: `blob-gui/` app + `gui-development-plan.md`, 9 commits not in `master` | **Review** before building `play` | Built on gen 1 and not examined in the 2026-10-02 diagnosis; holds the history that blocks the `.git` rewrite (§9) |
| `blob-engine/examples/diagnostics.rs` | **Keep** | Becomes `bench` |

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
- **Targets are written as soon as the round ends** (10–40 decisions later). Today an example waits for a whole 17-round game.
- **Cumulative scores are not an input in gen 2.**
- **Engine change:** add a helper that starts one round with given parameters.

### 5.3 Two networks

| | Policy net P | Value net V |
|---|---|---|
| Sees | the acting player's own view (as today) | the whole deal: every hand plus everything public |
| Outputs | bid distribution / per-card scores | expected round score ŝ for every seat |
| Used for | move priors at every expanded node; the fast no-search player | the value at every search leaf, on the *sampled* deal |
| Trained on | visit distributions at real decisions (τ = 1) | the true full state at each decision → the actual per-seat round scores |
| Size | today's (d = 128, 8 layers) | start at d = 128, 4 layers; grow only if validation loss says so |

**Why V may see every hand.** Inside a sampled deal the search already treats all cards as known. V never sees the *real* hidden cards at play time, only deals sampled from what the player knows.

**Why V doesn't over-promise.** V is trained on rounds played by players who did *not* see each other's hands. So it predicts realistic outcomes, not "everyone plays perfectly with open cards" ones.

**Why V is easier to learn.** With all hands known, a round's outcome is nearly decided. In 1-card rounds, every play is forced, so after bidding the outcome is fully determined. That gives a free exactness test (§7).

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
- **Greedy play** = most visits; ties go to the higher prior.
- **Determinization fixes:** voids from the current trick; on fallback, relax only the seat that can't be satisfied.

### 5.5 Encoder changes

P and V share the encoder code.

1. **`has_bid` per player.** Derive it from the dealer and the current player; no state change needed.
2. **Bid context:** sum of bids so far, bids still to come, (sum − cards)/cards, and my position in bidding order. The dealer-constraint bit already exists.
3. **Seat-relative encoding:** rotate so "me" is seat 0, with a one-hot relative seat on player and played-card tokens. Required for mixed table sizes.
4. **Trick features:** a "winning so far" flag on cards in the current trick; "legal" and "beats current winner" flags on hand cards.
5. **Small fixes:** scale counts to [0, 1]; `is_highest_in_suit` ignores my own cards.
6. **Remove cumulative-score features.**
7. **V mode:** opponents' hand cards become a new token type, tagged with the owner's relative seat.
8. **Suit-permutation augmentation** when sampling training batches: relabel suits consistently, including trump; 24 permutations. Cheap, and it multiplies data variety against memorization.

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

- **Replay-ratio governor.** The learner may use at most R samples per sample produced, starting at R ≈ 4–8; it sleeps when it gets ahead. This replaces "epochs" and directly limits memorization. Log the actual ratio.
- **Validation set by round, not by position.** Positions from one round share a label, so a position-level split would leak.
- **Warm-up gate.** The learner starts once the buffer holds at least 50k examples. LR warm-up applies on top.
- **Publishing.** Export both networks every K learner steps (the Python bridge takes ~25 s; run it off the learner thread). Actors swap networks between rounds, never mid-round.
- **STOP file.** Drain the actors, save, exit. Resume continues with the same buffer, so no cold-buffer special case.
- **Buffer persistence as delta chunks.** Save one file per ~N new examples instead of a full 200 MB snapshot every iteration. Resume reloads the newest chunks up to capacity. Tolerate a missing or corrupt chunk by skipping it.
- **Metrics:** one row per minute or so, keyed by learner step.
- **Not bit-reproducible;** accepted.

### 5.7 Evaluation and diagnostics

**Primary yardstick:** one model seat against four rule bots.
- **`bench` command** (grown from `examples/diagnostics.rs`):
  - **Modes:** search, or network-only.
  - **Opponents:** rule bot, the gen-1 final checkpoint, or any checkpoint. Bots never run search.
  - **Duplicate deals:** a fixed list of deal seeds, each played once from every seat position, so card luck cancels out.
  - **Reports:**
    - points per game ± 95% CI;
    - win share;
    - bids made, split by cards dealt (1 / 2–4 / 5–8);
    - share of 0-bids, and a histogram of bid errors.
- **Cadence:**
  - **Network-only bench at every publish:** about 10 s for 640 games. Gen-1 data shows it tracks strength well (§2.1).
  - **Search bench about hourly:** ~4–5 min for 300 games at 5 × 100 today.
- **Held-out checks:**
  - validation losses for P and V;
  - V's correlation with the actual round outcome;
  - V's error on 1-card rounds after bidding, which should approach 0.
- **Search health:**
  - share of root options with a value for the deciding seat, which should be 100% by construction;
  - signal ratio per phase and branching factor.

  Read these only alongside the bench: decisive ≠ right.
- **Human playtests** via `blobmaster play`.
- **Checkpoint-vs-checkpoint** stays only as a secondary signal.

### 5.8 Compute budget

**Per-leaf cost** (from §3.1): today a leaf is one call at ~17 tokens. In gen 2 it's a P call (~17 tokens) plus a V call (~31 tokens):
- **≈ 2.8×** today's cost with a V the same size as P;
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

Each phase ends with a measurable exit criterion. Short validation runs (10–20 iterations, a few hours) between phases; the long run only after Phase 4.

**Phase 0 — Yardsticks first (no training)**
- [x] Rule bot `blob-engine/src/rule_bot.rs` (2026-10-02).
- [x] Gen-1 diagnostics `blob-engine/examples/diagnostics.rs` (2026-10-02).
- [ ] `bench` subcommand: duplicate deals, both modes, by-hand-size bid stats.
- [ ] `blobmaster play`: human vs bot in the terminal, with an option to show the bot's policy and values. First check what the `gui` branch already provides (§4).
- [ ] Round-level validation split and validation losses in the gen-1 driver.
- [ ] Repo hygiene (§9).
- *Exit:* `bench` reproduces −9 ± 3 for the gen-1 final checkpoint; you can play a full game against it.

**Phase 1 — Cheap correctness fixes**
- [ ] Encoder items 1–6 from §5.5.
- [ ] Determinization fixes.
- [ ] Greedy tie-break; `DynEval` batch forwarding.
- *Exit:* unit tests pass. No training run needed.

**Phase 2 — Per-round, per-seat values (cheap version, gen-1 driver)**
- [ ] Per-round `u_s` targets; per-seat value head on P (own view); all-seat backup; exact end-of-round `u_s`.
- [ ] Training passes cut to a replay ratio of ~8.
- [ ] 20-iteration run at 5p7c.
- *Exit:* the search bench clearly beats gen-1 final's −9 by iter 20 and is still rising; V's held-out correlation with round outcome above 0.5.
- This phase isolates the effect of the target change. Expect plays to improve more than bids: an opponent's view can't judge my bid, because it doesn't see my hand.

**Phase 3 — Full-deal value net + warm start**
- [ ] V model + encoder V-mode + export + two-session evaluator; search leaves use V for all seats.
- [ ] Supervised pre-training on rule-bot rounds: V on outcomes, P imitating the bot. This is also V's first test: held-out error, plus exactness on 1-card rounds.
- [ ] Bench the pre-trained P+V with search before any RL.
- [ ] Short RL run (gen-1 driver).
- *Exit:* search bench ≥ +10 points per game vs the rule bot.

**Phase 4 — Async driver**
- [ ] Actor–learner per §5.6, rounds-not-games self-play, replay-ratio governor, delta persistence, continuous bench, suit augmentation.
- *Exit:*
  - it reproduces the Phase-3 result in less wall time;
  - STOP/resume is clean, with no loss spike;
  - the replay ratio holds at target.

**Phase 5 — Long run, 5 players / 7 cards**
- *Exit:* ≥ +20 points per game vs the rule bot; far ahead of gen-1 final; bids made in 5–8-card rounds clearly above the rule bot's; human playtests feel strong.

**Phase 6 — Mixed table sizes**
- n = 4–7 players, C = 7–8 cards (seat-relative encoding is the prerequisite).
- Fine-tune per table size only if one model lags.

**Phase 7 — Deployment**
- Play UI; Windows + Intel iGPU via ONNX Runtime (consider the OpenVINO execution provider); per-table-size models if Phase 6 needed them.

---

## 7. Gates

| Gate | Measure | Pass |
|---|---|---|
| G0 yardstick | `bench`, gen-1 final, search | −9 ± 3 reproduced |
| G1 value learnable | V on held-out rule-bot rounds | correlation with outcome > 0.7; 1-card rounds after bidding ≈ exact |
| G2 per-round values help | Phase-2 run, search bench | > −9 by iter 20, rising |
| G3 beats the rule bot | search bench | ≥ +10 points/game |
| G4 async parity | Phase-4 run | G3 result in less wall time; replay ratio on target |
| G5 strong | long run | ≥ +20 points/game vs rule bot; human playtests |

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

---

## 9. Repo hygiene

**Done 2026-10-02.**
- **`checkpoints/` pruned from 41 GB to 0.22 GB.** All that remains is `run-2026-05-14/`:
  - `iter_000000`, `iter_000025`, `iter_000125`, `iter_000167` (`model.onnx` + `meta.json`; the four rows of §2.1);
  - `iter_000167/model.ot` (for the ONNX↔tch parity test) and `iter_000167/buffer.bin` (for `diagnostics value`);
  - `metrics.jsonl`, `strength.csv`;
  - `signal_ratio_by_iter.csv`, which replaces the 0.73 GB `decision_stats.jsonl`.

  All other runs and iterations were deleted. Their small text files (metrics, strength.csv, meta.json) are still in git history.
- **`*.onnx` is ignored by git.** Reference models go in with `git add -f`; per-iteration weights are never committed.
- **Machine-level clean-up:** `target/debug` (13 GB) and the pip download cache (12 GB) were removed. Free disk went from 22 GB to 86 GB.

**Deferred: shrinking `.git` (4.9 GB).**
- **What's in it:** about 4.6 GB of model blobs in history:
  - `sweep-2026-04-28-anchor` 1.4 GB, `run-2026-05-14` 1.1 GB, `run-2026-05-06` 1.0 GB;
  - smaller runs, plus 0.24 GB of gen-0 `.pth` files.
- **Why it waits:** reclaiming it needs a history rewrite and force-push. That must include the `gui` branch on GitHub, which shares this history (see §4). Rewriting `master` alone would leave `gui` with diverged history and wouldn't shrink GitHub.
- **Recipe when ready:**
  1. Tag `c6f0c2a` as `gen-1-final` and point the §10 / `AGENTS.md` references at the tag.
  2. Check out `gui` locally.
  3. Run `git filter-repo --force --prune-empty never --invert-paths --path-glob '*.onnx' --path-glob '*.pth' --path-glob '*calibration.bin' --path-glob '*decision_stats.jsonl'` on all branches.
  4. Back up the reference models before running it: the rewrite removes them from the working tree. Then re-add them with `git add -f`.
  5. Verify: commit count, HEAD tree, `git show gen-1-final:fix-mcts-plan.md`.
  6. Force-push `master`, `gui` and the tag. Other clones must re-clone.

**Still to do:**
- **Delete dead code as its replacement lands:** the INT8 path, Muon, gen-1 sweep and overnight scripts, and `HeuristicEvaluator` (§4).
- **Code comments still cite the retired documents,** about 50 references, most to `fix-mcts-plan.md`. Rewrite them as the code changes; until then use §10.

---

## 10. Retired documents (2026-10-02)

All are recoverable with `git show c6f0c2a:<file>`.

| Document | What it was | Where its live content went |
|---|---|---|
| `development-plan.md` | Gen-1 session-by-session plan and specs | Reversed decisions §2.7; perf facts §3.1; ruled-out list §3.2; deployment and fine-tuning ideas §6 Phase 6–7; crate boundaries in `AGENTS.md` |
| `fix-mcts-plan.md` | 2026-05-12 diagnosis (Dirichlet, terminal values, τ split, warm start, forced moves) | Implemented parts kept (§4 `mcts.rs` row); warm start → Phase 3; superseded diagnosis → §2 |
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

Run from the repo root with nothing else busy (the tool uses every core).

```bash
cargo build --release -p blob-engine --example diagnostics
D=./target/release/examples/diagnostics
M=checkpoints/run-2026-05-14/iter_000167/model.onnx

$D match  $M mcts rulebot 320     # §2.1 headline (~4.5 min)
$D match  $M raw  rulebot 640     # network only (~10 s)
$D match  $M rulebot heuristic 2000
$D value  $M checkpoints/run-2026-05-14/iter_000167/buffer.bin 96   # §2.2–2.5 (~8 min)
$D tokens $M                      # §3.1 cost by sequence length
```

The signal-ratio table (§2.5) is read from `checkpoints/run-2026-05-14/signal_ratio_by_iter.csv` (column `signal_median`). That file summarizes the per-decision log (7.84M decisions, deleted 2026-10-02) by iteration, phase and legal-move count. The other §2.1 rows use `iter_000000`, `iter_000025` and `iter_000125` in place of `iter_000167`.
