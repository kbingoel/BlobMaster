# Overnight battery results — 2026-04-27

Started: 2026-04-27T23:05:00+02:00
Host: AM5-ubuntu
Binary: ./target/release/blobmaster-train
Anchor FP32 model: checkpoints/7.3c-run/iter_000014/model.onnx
Calibration BCAL: checkpoints/7.3c-run/calibration.bin


## Block A — Muon 10-iter validation

- elapsed: 8427s (140.4 min)
- iter 5 : 0.604 (lower95 0.534)  vs 7.3c 0.660 (lower95 0.563)  Δwr=-0.056
- iter 10: missing
- verdict (per dev plan §7.4d): **no-data**

Full strength.csv: `checkpoints/overnight-2026-04-27/A-muon/strength.csv`

## Block B — Muon-OFF control (10-iter, paired with A)

- elapsed: 8251s (137.5 min)
- iter 5 : 0.542 (lower95 0.471)  vs 7.3c 0.660 (lower95 0.563)  Δwr=-0.118
- iter 10: missing

This is the AdamW-only baseline on the current stack — used as the
paired control for Block A. Cross-block comparison rendered below.

## Block A vs B — Muon vs no-Muon, iter 5 / 10

| iter | A (Muon-on) | B (Muon-off) | Δ (A − B) |
|---:|---:|---:|---:|
| 5 | 0.604 | 0.542 | +0.062 |
| 10 | None | None | — |

**Caveat:** with `eval_games = 192`, the Wilson 95% half-width is ≈ ±7pp around 0.5, so a Δwr of ±0.05 at iter 10 is inside noise. 
Treat differences smaller than that as 'inconclusive after 10 iters'; 
the dev plan's actual decision criterion is the 7.5 100-iter trajectory.

## Block C — INT8 deferred-lever sweep (initial run — superseded)

The original 2026-04-27 run failed all 10 levers instantly because the
script's `LD_PRELOAD=libtorch_cuda.so` (set for CUDA training) bled into
the Python subshells; tch's vendored libtorch has a different C++ ABI
than the venv's `torch==2.5.1+cu124` wheel, crashing `import torch` with
`undefined symbol: torch::jit::Graph::toString`. Fix: wrap python
invocations in `( unset LD_PRELOAD; python3 ... )` —
[overnight-2026-04-27.sh:436-441](../../scripts/overnight-2026-04-27.sh#L436-L441).
**See "Block C re-run" section near the bottom of this file for the
real data.**

## Block D — num_determinizations profile sweep (T=32, target_batch = num_dets)

| num_dets | per-game (s) | per-decision (ms) | ONNX-avg (µs) | speedup vs 7.3c B=1 16T (7.250) |
|---:|---:|---:|---:|---:|
| 5 | 4.750 | 399.965 | 8468.10 | 1.526× |
| 6 | 4.738 | 398.958 | 10112.67 | 1.530× |
| 8 | 4.843 | 407.791 | 13572.01 | 1.497× |
| 10 | 4.998 | 420.922 | 17247.59 | 1.450× |

**Fastest:** num_dets=6 at 4.738 s/game.
Note: this block is **speed-only**. Quality (policy-KL, top-1 visit share) must be checked from a paired 1-iter training run before raising the default `num_determinizations` past 5.

## Block E — 5-iter mixed-player smoke

- elapsed: 5438s (90.6 min)
- iters logged: 5

| iter | combined_loss | bid_top1 | policy_kl | top1_visit_share | examples |
|---:|---:|---:|---:|---:|---:|
| 0 | 2.5741 | 0.902 | 0.2373 | 0.943 | 53708 |
| 1 | 1.2508 | 0.942 | 0.0847 | 0.840 | 53826 |
| 2 | 1.2068 | 0.948 | 0.0760 | 0.839 | 52400 |
| 3 | 1.2504 | 0.945 | 0.0713 | 0.832 | 52936 |
| 4 | 1.2891 | 0.940 | 0.0675 | 0.832 | 55046 |

Smoke pass: no panics, all 5 iters logged, variable-arena code exercised on n∈{4,5,6,7}.
**Note:** n=5 bid-success-rate vs 7.3c baseline must be read from the eval run (no eval triggers here since eval_interval=5 and total_iterations=5 → only one eval at iter 5).

---

Total wall: 25239s (420 min)
Finished: 2026-04-28T06:05:39+02:00

---

## Addendum (2026-04-28) — fills the iter-10 hole and re-runs Block C

Two follow-up runs after the main battery surfaced issues, kicked off
2026-04-28 ~07:48 and finished by 08:25 (~37 min total).

### Off-by-one fixed retroactively via direct eval

`total_iterations = 10` runs the train loop over iters 0..9, so the eval
trigger (`iter > anchor_iter && iter % 5 == 0`) only fired at iter 5 —
the iter-10 row required by dev-plan §7.4d was never produced, and the
saved-on-disk model is `iter_000009/`. To recover a comparable signal
without re-training, ran `blobmaster-train evaluate iter_000009 vs
iter_000000` on both A and B with the standard 5×100 MCTS / 192-game
config. (`--resume` for one extra iter would *not* fix this cleanly —
`anchor_iter` resets to the resume point, so the next eval boundary
becomes iter 15. Captured the gotcha in [AGENTS.md:106-108](../../AGENTS.md#L106-L108).)

### Block A vs B — paired iter-9 eval against iter-0 anchor

| | win-rate | 95% CI | wins / N | bid_success (current vs opp) | score Δ | inconclusive |
|---|---:|---|---:|---:|---:|:---:|
| **A — Muon-on**  | **0.609** | [0.539, 0.676] | 117 / 192 | 0.358 vs 0.315 | +6.99 | true |
| **B — Muon-off** | **0.609** | [0.539, 0.676] | 117 / 192 | 0.380 vs 0.339 | +6.96 | true |
| **Δ (A − B)**    | **0.000** | (CIs identical) | 0 | −0.022 | +0.03 | |

**Identical to 4 sig figs.** Both runs landed on 117/192 wins. The
+6pp Muon advantage at iter 5 was within Wilson noise (CIs already
overlapped); by iter 9 it has fully converged. Score differentials and
CIs are also indistinguishable — the only real persistent gap is in
**bid-success-rate, where Muon-off is +2.2pp ahead** (iter 5 showed
+2.7pp in the same direction, so the signal is consistent across both
checkpoints).

**Conclusion (per dev-plan §7.4d):** the hold-back trigger ("> 5pp
drop at iter 10") does **not** fire — Muon may technically be retained.
But the validation criterion was "Muon helps", and the empirical answer
is "Muon does nothing measurable at 1.63M / d_model=128 / 10 iters".
The published-Muon-gains caveat in the dev plan is confirmed.
**Recommendation: ship 7.5 with `enable_muon = false`.** Trades a tiny
training-step CPU saving (no Newton-Schulz iterations) for a simpler
optimizer surface; the bid-success +2.2pp is small but in favour of off.

### Block C re-run — INT8 deferred levers (real data)

10 levers × 500 calibration states; LD_PRELOAD unset for the
python subshell.

| lever | bid-argmax | play-argmax | value-sign | static gate |
|---|---:|---:|---:|:---:|
| s8s8 (symmetric activations) | 0.842 | 0.960 | 0.938 | ❌ |
| entropy (KL calibration) | 0.848 | 0.960 | 0.942 | ❌ |
| exclude-block-0 | 0.852 | 0.950 | 0.932 | ❌ |
| exclude-block-1 | 0.846 | 0.964 | 0.938 | ❌ |
| exclude-block-2 | 0.848 | 0.960 | 0.938 | ❌ |
| exclude-block-3 | 0.850 | 0.954 | 0.940 | ❌ |
| exclude-block-4 | 0.852 | 0.966 | 0.940 | ❌ |
| exclude-block-5 | 0.846 | 0.956 | 0.940 | ❌ |
| exclude-block-6 | 0.848 | 0.962 | 0.944 | ❌ |
| exclude-block-7 | 0.844 | 0.962 | 0.942 | ❌ |
| (7.4b U8S8 baseline, for reference) | 0.848 | 0.960 | 0.942 | ❌ |

Static-gate bars: bid-argmax ≥ 0.95, value-sign = 1.00. Spread across
all 10 variants is **< 1pp on bid-argmax and < 1.2pp on value-sign**;
nothing comes near the gates. Per-block exclusion at any depth — even
the input or output blocks — moves the needle by 0–0.4pp; the heads are
already FP32 in all variants, so the failure is the cumulative
quantization error through 8 transformer layers at d_model=128, exactly
as the 7.4b dev-plan note flagged.

**Conclusion: INT8 is ruled out for this architecture, not parked.**
The deferred levers list ("S8S8 / entropy / sensitivity-driven body
exclusions") is exhausted as of this run; the only remaining paths
would be quantization-aware training (out of scope) or a wider model
(d_model ≥ 256 — out of 7.5 scope). FP32 is the only ship config.

### Combined verdict for 7.5

| 7.4 lever | overnight outcome | 7.5 default |
|---|---|---|
| Muon optimizer | converges to no-Muon by iter 9 | **`enable_muon = false`** |
| INT8 (10 calibration variants) | all fail static gate by 5–11pp | FP32 |
| `num_determinizations > 5` | flat (dets=6) → regressive (dets ≥ 8) | 5 |
| Mixed-player self-play | 5-iter smoke clean, no panics on n∈{4,5,6,7} | `fixed_player_count = None` |
| target_batch | (Stage-2 sweep 2026-04-27 already settled) | 5 (= num_determinizations) |
| self-play threads | (already settled) | 32 |

Throughput stays at the post-7.4c-Stage-1 number: **~4.7 s/game × 118
games ≈ ~9 min/iter** ⇒ 100-iter 7.5 run ≈ **~15 h** wall-clock.
