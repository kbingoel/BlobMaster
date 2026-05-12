# MCTS / training fix plan — 2026-05-12

Concise diagnosis and action list for the "model trains but doesn't gain
strength" failure pattern documented in
[conclusion.md](conclusion.md), [7.3b-analysis.md](7.3b-analysis.md), and
the [checkpoints/run-2026-05-06/](checkpoints/run-2026-05-06/) metrics.

The four correctness items below address the *learning signal*. The fifth
is a pure perf item — kept in this plan because cutting iter wall-clock
lets us validate the correctness fixes on a useful cadence.

---

## 1. Observed symptoms

- Loss curves and top-1 accuracy "look healthy": `bid_policy_loss` falls
  from 1.04 → 0.25, `play_top1_acc` reaches 0.96
  ([metrics.jsonl iter 146](checkpoints/run-2026-05-06/metrics.jsonl)).
- Strength does not follow. `run-2026-05-06` reaches **0.66 vs iter_0 at
  iter 80** and then **plateaus through iter 146** (≈ 0.526 vs the new
  anchor iter_80) — see
  [strength.csv](checkpoints/run-2026-05-06/strength.csv).
- The 7.2 run reached 0.77 vs iter_0 by iter 10 with the same network
  but a different MCTS regime — see
  [checkpoints/7.2-run/strength.csv](checkpoints/7.2-run/strength.csv).
  So the architecture has been shown capable of *some* learning; the
  pipeline regressed since then.
- At iteration 0 of `run-2026-05-06` MCTS already produces near-one-hot
  policy targets: `visit_entropy_mean = 0.074`,
  `top1_visit_share_mean = 0.97`. With a randomly-initialised network
  those targets *are* noise — the model is being trained to confidently
  imitate random preferences.

## 2. Root causes (in priority order)

### C1. No Dirichlet noise at the root prior

AlphaZero's standard exploration mechanism is absent. Grep `dirichlet`
returns zero hits across the codebase. Without it, the first K sims
after root expansion each evaluate one previously-unvisited child
(`f32::INFINITY` at [mcts.rs:195](blob-engine/src/mcts.rs#L195) for
`visit_count == 0`), so by sim K every legal child has exactly 1 visit.
From sim K+1 onward UCB1's exploration term
(`c_puct * P * sqrt(N) / (1+N) ≈ 0.34` for the typical
5-legal / `c_puct=1.5` / uniform-prior case) cannot overcome the spread
of random network Q-values in `[-1, 1]`. The result: ~all remaining sims
land on whichever child got the highest random Q on its single visit.
The metrics above are the smoking gun.

Already flagged as a candidate in
[7.3b-analysis.md §8.4](7.3b-analysis.md), never implemented.

### C2. Terminal states backprop `value = 0.0` instead of the real outcome

Hardcoded in two places:
[mcts.rs:378-389](blob-engine/src/mcts.rs#L378-L389) (`run_search`) and
[mcts.rs:526-533](blob-engine/src/mcts.rs#L526-L533) (`run_lockstep_search`):

```rust
let (policy, value) = if is_terminal(&state) {
    (Vec::new(), 0.0)        // ← this throws away ground truth
} else { eval.evaluate(&state) };
```

`is_terminal` returns true for `Scoring` and `Complete`
([mcts.rs:276](blob-engine/src/mcts.rs#L276)). Both phases have a real
outcome available — Scoring has the just-completed round's
`score = (tricks_won == bid) ? 10 + bid : 0`, Complete has cumulative
scores. MCTS therefore *never* sees a ground-truth signal in-tree; it
only re-runs network value estimates through the search. The value head
itself is trained from root-backfilled z-scored final scores
([self_play.rs:222-249](blob-nn/src/self_play.rs#L222-L249)), but that
signal only enters at root level and cannot correct mid-tree drift.

**Two related-but-distinct issues bundled with C2 (both addressed by
Step 2 below):**

- **C2a. Terminal credit is leaf-seat-only.** [`backprop`](blob-engine/src/mcts.rs#L333-L342)
  writes only to `value_sums[leaf_seat]`. That is the only correct
  choice for *network* leaves (the net knows one perspective). At a
  *terminal*, ground truth exists for **every** seat, but the current
  code still credits only one slot. Combined with the per-seat Q
  storage at [mcts.rs:78-87](blob-engine/src/mcts.rs#L78-L87), this
  means at any interior node where seat `j ≠ leaf_seat` acts,
  `value_counts[j]` stays at 0 and UCB1 falls back to `Q = 0`
  ([mcts.rs:199-202](blob-engine/src/mcts.rs#L199-L202)) — so seat-`j`
  selection is driven almost entirely by the exploration term. This
  per-seat Q sparsity is structural; multi-seat terminal credit is the
  cheapest mitigation and folds naturally into Step 2.

- **C2b. MCTS cannot search past round boundaries.**
  [`apply_action`](blob-engine/src/mcts.rs#L266-L272) no-ops on
  `Scoring`/`Complete`, so a 5P7C game with 7 rounds gives MCTS at
  round-1 zero visibility into rounds 2–7 — where most of the
  cumulative-score variance lives. Step 2 patches the *value* at the
  boundary; it does **not** restore search depth. Either accept the
  truncation as a known limitation (Step 2's round-z is a useful proxy
  but bounded by "this round in isolation") or extend MCTS to roll past
  `Scoring` by sampling a fresh deal — non-trivial; not in this plan,
  but called out so the next debugging round does not rediscover it.

- **C2c. Value-scale mismatch between in-tree terminal and training
  target.** [`backfill_values`](blob-nn/src/self_play.rs#L224-L250)
  z-scores **full-game `cumulative_scores`**. A naive Step 2 would
  z-score only the just-finished round score. Along a single path,
  UCB1 then mixes Q from network leaves (predicting full-game z) and
  terminal leaves (round-z) — a Q-scale discontinuity that biases
  search at round boundaries. Step 2 below resolves this by deriving
  the in-tree terminal value from the **same** statistic
  `backfill_values` uses, applied to
  `cumulative_scores + just_finished_round_score`.

### C3. Fused training-target and sampling distribution; τ-schedule collapses both

[mcts.rs:993-1018](blob-engine/src/mcts.rs#L993-L1018) applies the
temperature schedule to a single `policy` vector that is used as both
the training label *and* the action sampler
([self_play.rs:163](blob-nn/src/self_play.rs#L163),
[self_play.rs:192](blob-nn/src/self_play.rs#L192)).
The current run sets `late = 0.1`, `switch_at = 15`
([run-2026-05-06.toml](blob-train/run-2026-05-06.toml)), so ~95% of
decisions in a 225-decision game emit a near-one-hot **target**.

The intent is acknowledged at
[mcts.rs:665-674](blob-engine/src/mcts.rs#L665-L674):

> "At τ → 0 the late-game training target collapses to one-hot on the
> argmax-visit action, which is intentional… If you need a τ=1 training
> target with τ→0 sampling (canonical AlphaZero), split `MctsResult`
> into separate `policy_target` and `policy_sampling` fields — out of
> scope for 7.4d."

Canonical AlphaZero uses τ=1 for the *target* and τ→0 only for *action
sampling* after the opening. The 7.2 run (constant τ=1,
[7.2-run metrics.jsonl](checkpoints/7.2-run/metrics.jsonl) iter 0:
`visit_entropy_mean=0.34`) reached 0.77 vs iter_0 by iter 10. The
τ-scheduled run collapses entropy to 0.07 and reaches 0.66 by iter 80.
The schedule eliminated the exploration that made the network learn.

### C4. (Adjacent) No supervised warm-start despite `HeuristicEvaluator` existing

[evaluator.rs:113-250](blob-engine/src/evaluator.rs#L113-L250) implements
a sensible heuristic (bid = high-card count; play = cheapest card that
takes the trick, else cheapest). It is used as the eval *opponent* but
never to bootstrap policy. With C1–C3 fixed this may not be needed, but
a few iterations of supervised pretraining would short-circuit any
remaining random-confidence cycle at iter 0 cheaply.

## 3. What has *not* been tried in any checkpointed run

| Hypothesis | Tested? | Evidence |
|---|---|---|
| Dirichlet root noise | No | no code; only mentioned in 7.3b-analysis.md §8.4 |
| Real terminal value backprop | No | `0.0` hardcoded since the Rust port |
| Split τ_train vs τ_sample | No | every run uses the fused policy |
| Heuristic supervised warm-start | No | `HeuristicEvaluator` only wired into eval |
| Cross-buffer value normalisation | No | per-game z-score throughout |

Levers that *have* been swept (none touch C1–C4): epoch-stop thresholds
(7.3b vs 7.3c), Muon vs AdamW (7.4d overnight), int8/fp32 self-play,
`target_batch`, thread counts, temperature on/off.

---

## 4. Step-by-step fix instructions

Each step is independently mergeable and individually testable. Land in
order: 1 → 2 → 3, validate against 7.2's iter-10 benchmark, then 4 if
needed, then 5.

### Step 1 — Add Dirichlet noise at the root prior

**Where:** [mcts.rs `mcts_search`](blob-engine/src/mcts.rs#L868),
after the per-determinization root expansion runs inside
`run_lockstep_search` and before the first child is selected. Easiest
implementation: decorate priors after `expand` for the root only, inside
the search driver. Alternative: do it in the `mcts_search` orchestrator
by replacing the standard `expand` call on root with a custom
`expand_root_with_noise`.

**What:**
- Sample `η ~ Dir(α)` over the legal-action count.
- `P'(a) = (1 − ε) · P(a) + ε · η(a)` for each root child.
- Defaults: `ε = 0.25`, `α = 0.3` (chess) or `α = 10 / num_legal`
  (DeepMind scaling heuristic, more robust across Blob's variable
  branching).
- Apply *per determinization tree* (each det has its own root).
- Drive from `MctsConfig` with new fields
  `root_dirichlet_alpha: f32`, `root_dirichlet_epsilon: f32`. Default
  to noise-on for self-play, noise-off for eval — gate via a flag on
  `MctsConfig` or via a `mcts_search_eval` variant so eval comparisons
  stay deterministic.

**Tests:**
- Add a unit test asserting that root priors after expansion differ from
  the raw evaluator output by the expected mixing weight, and that they
  still sum to 1.
- Existing `target_batch_one_matches_serial_per_det` and
  `lockstep_search_matches_serial_per_det` parity tests will need a
  "noise disabled" branch; keep them with `ε=0`.

**Validation signal:** at iter 0 of a fresh run, `visit_entropy_mean`
should land in the 0.3–0.5 band (matching
[7.2-run metrics.jsonl iter 0](checkpoints/7.2-run/metrics.jsonl)) and
`top1_visit_share_mean` should fall below 0.9.

### Step 2 — Backprop real terminal outcomes (per-seat, scale-aligned)

Addresses C2 and the bundled C2a/C2b/C2c concerns above.

**Where:** the two terminal short-circuits in MCTS:
- [mcts.rs:378-389](blob-engine/src/mcts.rs#L378-L389) in `run_search`
- [mcts.rs:526-533](blob-engine/src/mcts.rs#L526-L533) in
  `run_lockstep_search`

Plus a new sibling to [`backprop`](blob-engine/src/mcts.rs#L333-L342)
that takes a per-seat value vector instead of a single
`(leaf_seat, v)` pair.

**What:**

1. Add a shared helper `terminal_z_scores(state) -> [f32; MAX_PLAYERS]`
   in a module reachable from both `mcts.rs` and `self_play.rs` (e.g.
   `blob_engine::scoring`). Both phases use the **same statistic** as
   [`backfill_values`](blob-nn/src/self_play.rs#L224-L250) — full-game
   z-score over active seats, clipped to `[-1, 1]` — so the in-tree
   value and the training target live on a single coherent scale (C2c):
   - `Complete` phase: z-score `state.cumulative_scores[..n]` directly.
   - `Scoring` phase: z-score
     `cumulative_scores[..n] + this_round_score[..n]`, where
     `this_round_score[i] = (tricks_won[i] == bid[i]) ? 10 + bid[i] : 0`.
     Rationale: at `Scoring`, `cumulative_scores` has *not* yet been
     updated with the just-finished round
     ([self_play.rs:209-211](blob-nn/src/self_play.rs#L209-L211) →
     `advance_round` is what folds it in), so we pre-add it locally
     before z-scoring. This is the closest single-statistic
     approximation of "what `backfill_values` would emit if the game
     ended now" available without rolling rounds 2..N forward.

2. Add `backprop_terminal(arena, path, z_scores: &[f32; MAX_PLAYERS])`
   that increments `visit_count` once per node and adds `z_scores[s]`
   to `value_sums[s]` for **every** active seat `s` (with matching
   `value_counts[s] += 1`). This fixes C2a: at terminal leaves, ground
   truth exists for all seats and should be credited to all of them.
   Network leaves continue to use the existing single-slot
   [`backprop`](blob-engine/src/mcts.rs#L333-L342) since the net only
   knows one perspective.

3. In both terminal short-circuits, replace
   `(Vec::new(), 0.0)` + `backprop(... leaf_seat, value)` with
   `backprop_terminal(... &terminal_z_scores(state))`. The lockstep
   driver already runs the terminal path without `in_flight`
   decoration ([mcts.rs:526-533](blob-engine/src/mcts.rs#L526-L533));
   the only thing changing is the backprop call.

4. **Out of scope but document in code**: rolling MCTS past `Scoring`
   into the next round (C2b). Add a `// TODO(C2b)` comment at
   [`apply_action`](blob-engine/src/mcts.rs#L266-L272) noting that
   `Scoring`/`Complete` no-op truncates the search horizon, and the
   round-z proxy used here is bounded by single-round signal. A future
   pass can extend `apply_action` to call `advance_round` with a
   freshly sampled deal when a determinization tree wants to look
   ahead. Not blocking on it for this plan.

**Tests:**
- New unit test: construct a `Complete` state where seat 0 has a clearly
  highest cumulative score, run `mcts_search` on a state one apply away
  from `Complete`, assert root `value_counts[s] > 0` for **every** seat
  `s` (proves C2a multi-seat credit landed) and that
  `q(perspective)` matches the closed-form z-score.
- New unit test for `terminal_z_scores` parity with `backfill_values`:
  construct an artificial final state, call both, assert the per-seat
  vector matches.
- Adjust `run_search_visits_all_legal_actions_and_sums_correctly`
  and any other test that asserts on terminal `v=0`.

**Validation signal:** `value_loss` should fall faster than in
[run-2026-05-06](checkpoints/run-2026-05-06/metrics.jsonl) over the
first 5 iters, and `value_mean` should stay closer to 0 (no drift from
a head trained on inconsistent in-tree-vs-root targets). Additionally,
search Q at intermediate non-perspective seats should no longer
default to 0 along most paths — sample a few `mcts_search` runs in a
dev build and inspect `value_counts[seat]` distributions if you want a
direct read.

### Step 3 — Decouple training target τ from action-sampling τ

**Where:** [mcts.rs `MctsResult`](blob-engine/src/mcts.rs#L782-L789)
and downstream consumers in self-play
([self_play.rs:154-208](blob-nn/src/self_play.rs#L154-L208)).

**What:**
- Split `MctsResult.policy` into:
  - `policy_target: Vec<f32>` — always computed at τ = 1.0 from the
    aggregated root visit counts.
  - `policy_sampling: Vec<f32>` — computed at
    `cfg.temperature_at(decision_index)`. Used only for action sampling
    in self-play.
- In [self_play.rs](blob-nn/src/self_play.rs#L154-L208):
  - Push `policy_target` into `TrainingExample.policy` (training label).
  - Pass `policy_sampling` to `sample_from_policy`.
- For eval, **explicitly select actions by `argmax(visit_counts)`**
  (or equivalently τ→0 on `policy_sampling`). The current eval path at
  [eval.rs:147,158](blob-nn/src/eval.rs#L147) inherits the *training*
  τ-schedule, which means the τ=1 opening still injects sampling noise
  into the strength signal you're using to validate. With Step 1 also
  disabling Dirichlet for eval, this gives a fully greedy, noiseless
  evaluation regime. Add an `eval_greedy: bool` (or a separate eval
  `MctsConfig`) and route accordingly.

**Tests:**
- New unit test asserting that with `temperature = 0.1`, `policy_target`
  has higher entropy than `policy_sampling` for the same visit counts.
- Keep an integration test that one full game's training examples have
  non-degenerate policy distributions even when `late = 0.1`.

**Validation signal:** with this and Step 1 in place,
`visit_entropy_mean` in metrics.jsonl should track `policy_target`
entropy (≈ 0.3+ on diverse decisions) rather than collapsing to the τ-0.1
sampling distribution.

**Caveat — `decision_index` excludes forced moves.**
[self_play.rs:158,188](blob-nn/src/self_play.rs#L158) only increments
`decision_index` around `mcts_search` calls; forced moves short-circuit
inside `mcts_search` and never reach the increment. So `switch_at = 15`
in [run-2026-05-06.toml:62](blob-train/run-2026-05-06.toml#L62) really
means "15 *non-forced* decisions," not "decision 15 of ~225." In late
tricks with many forced plays this lands the τ-step earlier in
real-game-time than the comment in the TOML implies. Decide whether to
(a) keep the current semantic and update the comment, or (b) move the
counter increment to *every* decision (forced or not) so the schedule
maps to game-time, then rerun the smoke. Either is fine; just pick one
intentionally as part of Step 3.

### Step 4 — (Optional, ship only if 1+2+3 isn't enough) Supervised warm-start

**Where:** a new pre-iter-0 phase invoked by
[blob-train/src/main.rs run_train](blob-train/src/main.rs#L272) before
`bootstrap_initial_onnx`.

**What:**
- Generate ~10k decisions with `HeuristicEvaluator` self-play (no MCTS,
  one-hot heuristic policy as the target, value = z-scored final score
  as today).
- Run ~3 epochs of supervised training on the policy head only (freeze
  transformer? — simpler: train end-to-end at peak_lr/3 for fewer
  steps).
- Then proceed into the AlphaZero loop. The first iter's MCTS will run
  on a non-random prior, breaking any residual confidence cycle.

**Defer until:** Steps 1–3 have been validated against 7.2's iter-10
benchmark. If the post-fix run crosses ≥ 0.85 win rate vs iter_0 by
iter 10, Step 4 is unnecessary.

### Step 5 — In-tree forced-move short-circuit (perf, not correctness)

**Why include in this plan:** it does not change what the model learns,
but it cuts ~20–40% of NN inference per iter — and the next two months
of debugging will depend on running short experiments quickly. The 7.4c
work already established that interior forced nodes are common in
trick-taking games (root-forced rate ~37%, see
[personal-notes.md](personal-notes.md)). Verbatim from that note:

> During leaf descent, after applying the selected action, peek at
> legal_plays(state). If exactly 1 legal action: auto-apply it, push a
> placeholder child (synthesized prior=1.0), keep descending — don't
> break out for an NN evaluation yet. Continue this fast-path until you
> hit a multi-legal node or terminal. Then either: (a) evaluate that
> real branching node with the NN, or (b) if terminal, backprop the
> terminal value (per Step 2) and exit.

**Where:**
- [mcts.rs `select_leaf_state`](blob-engine/src/mcts.rs#L399) — extend
  the descent loop with a "while-leaf-is-forced" inner fast-path that
  auto-allocates the forced child, applies the action, and continues.
- [mcts.rs `expand`](blob-engine/src/mcts.rs#L292) — unchanged; the
  fast-path allocates forced children inline rather than calling
  `expand` (no policy needed, prior is 1.0 by construction).
- The pending-batch path in
  [mcts.rs `run_lockstep_search`](blob-engine/src/mcts.rs#L470) — make
  sure the in-flight bookkeeping covers every node in the forced chain
  too (each gets `in_flight += 1` on descent, `-= 1` before backprop).

**What to check:**
- `legal_bids` / `legal_plays` are popcount on a `u64` bitmask — cheap.
- Backprop value comes from the *first non-forced* leaf the chain hits
  (or the terminal-state value from Step 2). Variance is identical in
  expectation to the current implementation; the forced-chain nodes
  share that downstream value.
- Visit counts on the forced chain still increment, so root visit
  distributions are unchanged — except that more sims now reach
  branching decisions per unit wall-clock.

**Caveat from the original note:** this changes the tree-build trace,
so the lockstep-vs-serial parity goldens
(`target_batch_one_matches_serial_per_det`,
`lockstep_search_matches_serial_per_det` —
see [mcts.rs:451-464](blob-engine/src/mcts.rs#L451-L464)) will need to
be updated to compare *root visit distributions* rather than node-by-node
traces. The semantics they guard (no-VL degeneracy at
`target_batch ∈ {1, num_dets}`) are preserved.

**Order it: after Steps 1–3 have landed and been validated.** Doing it
earlier risks invalidating the parity tests during the same window we're
trying to read learning signal from.

---

## 5. Validation plan

1. After Step 1: run a 5-iter smoke ("micro-run") using the 7.2 recipe
   (`fixed_player_count = [5, 7]`, constant τ=1, 118 games, default MCTS
   budget). Eval path runs with **Dirichlet off and greedy action
   selection** (Step 1 + Step 3 eval-greedy gate), even though Step 3
   hasn't fully landed — the eval-greedy switch is a one-line config
   change and worth doing now so the strength signal isn't polluted.
   Compare iter-0 `visit_entropy_mean` and iter-5 `strength.csv` against
   [7.2-run](checkpoints/7.2-run/).
2. After Step 2: rerun the smoke. Expect lower `value_loss` and faster
   strength climb. Spot-check that interior nodes have non-zero
   `value_counts` for non-perspective seats (C2a landed) and that the
   in-tree terminal Q values look like z-scores in `[-1, 1]` rather than
   the round-score raw scale (C2c landed).
3. After Step 3: switch the run config to the τ-schedule
   (`late = 0.1, switch_at = 15`) — the schedule should now help, not
   hurt, because training targets stay τ=1. Verify the
   `decision_index`-vs-forced-moves choice from Step 3's caveat lines up
   with the TOML comment (or update one of them).
4. Full 15-iter rerun once the smoke is green. Acceptance gate: ≥ 0.85
   vs iter_0 by iter 10 (vs 7.2's 0.77, vs run-2026-05-06's 0.50–0.55 at
   iter 5–10). If hit, drop Step 4 and move to long-run; if missed, add
   Step 4 and rerun.
5. Step 5 lands separately, validated by `decision_stats.jsonl`
   sim-budget reduction and iter wall-clock drop — not by strength
   metrics (it cannot change strength).

## 6. Out of scope for this plan

- **C2b — rolling MCTS past round boundaries.** Acknowledged limitation:
  in a 5P7C game with 7 rounds, a round-1 search sees zero of rounds
  2–7. Step 2's round-z proxy is bounded by single-round signal.
  Extending [`apply_action`](blob-engine/src/mcts.rs#L266-L272) to call
  `advance_round` with a freshly sampled deal is the real fix and is
  parked behind Steps 1–3 landing. If post-fix strength still plateaus
  with healthy losses, this is the next thing to try.
- Re-encoding parallelism (`personal-notes.md` item): GPU runs at 100%
  during training in the current setup; CPU encode keeps pace. No win.
- Model width / depth bumps: capacity is not the binding constraint —
  losses converge low (0.25/0.17) and top-1 hits 0.96, indicating the
  network fits its targets fine. Revisit only if `play_top1_acc`
  plateaus < 0.8 *after* Steps 1–3.
- Cross-buffer value z-scoring: minor; defer until C1–C4 have moved the
  strength needle.
- Anchor promotion logic and LR-schedule edge cases: already iterated on
  in [7.3b-analysis.md](7.3b-analysis.md) and main.rs; orthogonal to the
  learning-signal failure.
