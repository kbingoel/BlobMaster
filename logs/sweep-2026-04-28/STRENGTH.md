# Sweep 2026-04-28 anchor — strength tracker

Persistent append-only log of head-to-head evals run via
`scripts/sweep-2026-04-28-strength-eval.sh`. All rows are 5P7C, 192 games
cap, Wilson-95 early-stop active (so `n` may be 32/64/96/.../192 depending
on how decisive the matchup was). `inconc=true` means the CI didn't clear
the [0.45, 0.55] decision band — treat the win rate as "approximately tied".

The first 5 rows (timestamps before 20260429T184639) were back-filled from the
in-loop `eval vs anchor` driver in `checkpoints/.../strength.csv` — same Wilson
early-stop logic and same 5P7C 2-models-vs-3-heuristics seat layout, so they
are directly comparable to the offline strength-eval rows below.

| ts | current | opponent | wins/n | wr | lo95 | hi95 | score_diff | bid_a | bid_b | inconc |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 20260428T094119 | iter_5  | iter_0  | 47/64 | 0.734 | 0.615 | 0.827 | 15.48 | 0.329 | 0.238 | false |
| 20260428T120940 | iter_10 | iter_0  | 31/32 | 0.969 | 0.843 | 0.994 | 19.78 | 0.353 | 0.235 | false |
| 20260428T141137 | iter_15 | iter_0  | 28/32 | 0.875 | 0.719 | 0.950 | 22.72 | 0.368 | 0.230 | false |
| 20260429T102753 | iter_20 | iter_16 | 97/192 | 0.505 | 0.435 | 0.575 | 1.95 | 0.378 | 0.368 | true |
| 20260429T134647 | iter_25 | iter_16 | 92/192 | 0.479 | 0.410 | 0.550 | 0.35 | 0.373 | 0.371 | true |
| 20260429T184639 | iter_29 | iter_0 | 27/32 | 0.844 | 0.682 | 0.931 | 23.28 | 0.384 | 0.246 | false |
| 20260429T184639 | iter_25 | iter_0 | 50/64 | 0.781 | 0.666 | 0.865 | 18.30 | 0.367 | 0.254 | false |
| 20260429T184639 | iter_20 | iter_0 | 25/32 | 0.781 | 0.612 | 0.890 | 18.25 | 0.358 | 0.248 | false |
| 20260429T184639 | iter_29 | iter_15 | 87/192 | 0.453 | 0.384 | 0.524 | 0.17 | 0.370 | 0.368 | true |
| 20260430T054449 | iter_30 (Run-4) | iter_15 | 85/192 | 0.443 | 0.374 | 0.513 | -0.72 | 0.383 | 0.385 | true |
| 20260430T054449 | iter_30 (Run-4) | iter_29-run3 | 92/192 | 0.479 | 0.410 | 0.550 | +2.03 | 0.390 | 0.378 | true |
