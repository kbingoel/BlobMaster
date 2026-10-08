# RL run `checkpoints/rl-2026-10-07a`

Updated 22:04 · running 4.23 h of 3.5 (wall 4.23 h) · **finished**

Control: `touch STOP` (save and exit; `--resume` continues), `touch PAUSE` (idle until removed), `touch FINISH` (final benches now).

## Strength

Points per game vs the opponents' mean, 95% CI over deals. Network benches: default seed; paired = same deals as this run's step 0 / the previous publish.

| step | h | bench | deals | diff | paired vs | made 1 / 2–4 / 5–8 | 0-bids 5–8 |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | net-rb2 | 256 | +11.01 ± 0.99 |  | 0.781 / 0.731 / 0.687 | 0.314 |
| 0 | 0.0 | net-rb | 128 | +24.05 ± 1.61 |  | 0.823 / 0.753 / 0.722 | 0.351 |
| 0 | 0.1 | net-rb2-s7 | 128 | +10.82 ± 1.38 |  | 0.787 / 0.721 / 0.691 | 0.312 |
| 400 | 0.4 | net-rb2 | 256 | +11.91 ± 0.96 | start +0.90 ± 0.52 | 0.781 / 0.738 / 0.689 | 0.305 |
| 400 | 0.5 | net-rb | 128 | +24.23 ± 1.67 | start +0.18 ± 0.67 | 0.823 / 0.754 / 0.720 | 0.341 |
| 400 | 0.5 | net-rb2-s7 | 128 | +11.33 ± 1.43 | start +0.51 ± 0.80 | 0.787 / 0.730 / 0.686 | 0.303 |
| 800 | 0.8 | net-rb2 | 256 | +11.73 ± 0.97 | start +0.72 ± 0.56; previous -0.18 ± 0.44 | 0.781 / 0.734 / 0.691 | 0.316 |
| 800 | 0.9 | net-rb | 128 | +24.28 ± 1.57 | start +0.22 ± 0.69; previous +0.04 ± 0.56 | 0.823 / 0.749 / 0.725 | 0.353 |
| 800 | 0.9 | net-rb2-s7 | 128 | +11.23 ± 1.38 | start +0.41 ± 0.76; previous -0.10 ± 0.66 | 0.787 / 0.727 / 0.689 | 0.317 |
| 1200 | 1.3 | net-rb2 | 256 | +11.81 ± 0.94 | start +0.80 ± 0.59; previous +0.08 ± 0.42 | 0.781 / 0.736 / 0.688 | 0.304 |
| 1200 | 1.3 | net-rb | 128 | +24.24 ± 1.60 | start +0.19 ± 0.77; previous -0.03 ± 0.56 | 0.823 / 0.752 / 0.718 | 0.341 |
| 1200 | 1.3 | net-rb2-s7 | 128 | +11.43 ± 1.44 | start +0.61 ± 0.78; previous +0.20 ± 0.70 | 0.787 / 0.727 / 0.690 | 0.305 |
| 1600 | 1.7 | net-rb2 | 256 | +12.18 ± 0.93 | start +1.17 ± 0.58; previous +0.36 ± 0.39 | 0.781 / 0.736 / 0.692 | 0.303 |
| 1600 | 1.7 | net-rb | 128 | +24.51 ± 1.60 | start +0.46 ± 0.76; previous +0.26 ± 0.54 | 0.823 / 0.753 / 0.721 | 0.337 |
| 1600 | 1.7 | net-rb2-s7 | 128 | +11.37 ± 1.39 | start +0.55 ± 0.76; previous -0.06 ± 0.53 | 0.787 / 0.729 / 0.687 | 0.301 |
| 2000 | 2.1 | net-rb2 | 256 | +12.40 ± 0.96 | start +1.39 ± 0.63; previous +0.22 ± 0.45 | 0.781 / 0.738 / 0.693 | 0.290 |
| 2000 | 2.1 | net-rb | 128 | +23.87 ± 1.65 | start -0.18 ± 0.82; previous -0.63 ± 0.55 | 0.823 / 0.750 / 0.711 | 0.327 |
| 2000 | 2.1 | net-rb2-s7 | 128 | +11.10 ± 1.47 | start +0.28 ± 0.88; previous -0.27 ± 0.65 | 0.787 / 0.728 / 0.684 | 0.290 |
| 2000 | 2.5 | search-rb2 | 128 | +11.15 ± 1.41 | P alone +0.05 ± 0.81; baseline +2.10 ± 1.22 | 0.786 / 0.727 / 0.682 | 0.292 |
| 2000 | 2.8 | search-vsP | 64 | +0.64 ± 1.02 |  | 0.766 / 0.715 / 0.657 | 0.331 |
| 2400 | 3.2 | net-rb2 | 256 | +12.35 ± 0.97 | start +1.34 ± 0.59; previous -0.05 ± 0.46 | 0.781 / 0.737 / 0.695 | 0.305 |
| 2400 | 3.2 | net-rb | 128 | +24.57 ± 1.62 | start +0.52 ± 0.72; previous +0.70 ± 0.59 | 0.823 / 0.751 / 0.724 | 0.345 |
| 2400 | 3.2 | net-rb2-s7 | 128 | +11.34 ± 1.38 | start +0.51 ± 0.80; previous +0.24 ± 0.62 | 0.787 / 0.728 / 0.689 | 0.308 |
| 2760 | 3.5 | net-rb2 | 256 | +12.08 ± 0.94 | start +1.07 ± 0.59; previous -0.27 ± 0.39 | 0.781 / 0.735 / 0.691 | 0.297 |
| 2760 | 3.5 | net-rb | 128 | +24.57 ± 1.60 | start +0.51 ± 0.84; previous -0.01 ± 0.57 | 0.823 / 0.751 / 0.724 | 0.336 |
| 2760 | 3.5 | net-rb2-s7 | 128 | +11.39 ± 1.45 | start +0.56 ± 0.81; previous +0.05 ± 0.59 | 0.787 / 0.729 / 0.686 | 0.300 |
| 2760 | 4.0 | search-rb2 | 128 | +10.97 ± 1.38 | P alone -0.42 ± 0.81; baseline +1.92 ± 1.24; previous -0.19 ± 0.80 | 0.787 / 0.724 / 0.681 | 0.301 |
| 2760 | 4.2 | search-vsP | 64 | +0.42 ± 1.10 | previous -0.23 ± 1.31 | 0.764 / 0.714 / 0.656 | 0.336 |

## Learner

step 2750 · LR 1.00e-4 · P loss 0.3800 · V loss 0.4664 · replay ratio 5.98 (target 6) · 1062 steps/h · waiting on the governor 56.3% · actors' model: step 2760

V stream (rounds of P alone): 19819 V-only updates/h (GPU 38.2% of the time), V loss 0.4023 · ratio 2.99 (cap 3) · buffer 3000000 states

Held out at step 2760 (self-play rounds; dropout off). Compare the columns: a growing gap means memorization.

| | validation | training sample |
|---|---|---|
| P bid cross-entropy | 0.1257 | 0.1117 |
| P play cross-entropy | 0.5058 | 0.4924 |
| P top = search's top, bids | 0.9608 | 0.9715 |
| P top = search's top, plays | 0.9192 | 0.9257 |
| V MSE | 0.0371 | 0.0378 |
| targets' variance | 0.1346 | 0.1321 |
| V correlation | 0.8513 | 0.8451 |
| V last-trick MSE | 0.0000 | 0.0000 |
| V stream: V MSE | 0.0196 | 0.0188 |
| V stream: targets' variance | 0.1350 | 0.1352 |

Fixed V check (validation rounds of `checkpoints/rl-2026-10-06`, the same states every row): MSE 0.0425 (bids 0.0729, plays 0.0336), correlation 0.836.

Teacher probe (fixed rule-bot-2 rounds): P top = rule bot 2's, bids 0.784 / plays 0.763; V MSE 0.0435 (variance 0.1372), correlation 0.827, last-trick RMSE 0.005.

P's and V's change per publish, on the probe states (KL in nats; V in ŝ):

| step | KL bids / plays vs previous | vs start | V mean abs change vs previous / start |
|---|---|---|---|
| 400 | 0.2035 / 0.0897 | 0.2035 / 0.0897 | 0.0250 / 0.0250 |
| 800 | 0.0140 / 0.0124 | 0.2633 / 0.1550 | 0.0179 / 0.0280 |
| 1200 | 0.0171 / 0.0058 | 0.3171 / 0.1723 | 0.0182 / 0.0301 |
| 1600 | 0.0132 / 0.0045 | 0.3575 / 0.1797 | 0.0176 / 0.0325 |
| 2000 | 0.0258 / 0.0057 | 0.3762 / 0.2036 | 0.0177 / 0.0338 |
| 2400 | 0.0177 / 0.0071 | 0.3777 / 0.2166 | 0.0167 / 0.0345 |

## Self-play

Last 60 s: 3000 rounds/h, 81893 examples/h (35.5% forced); 11092 rounds, 249580 examples in all; buffers: training 236900 / 600000, validation 12680; 24 actors, process restarts 0.

| search health | bids | plays |
|---|---|---|
| decisions with a choice | 243 | 637 |
| target's top ≠ P's top | 3.7% | 6.8% |
| KL(target ‖ P) | 0.038 | 0.075 |
| entropy: target / P | 0.081 / 0.146 | 0.374 / 0.495 |
| move played ≠ target's top | 1.6% | 0.0% |

V stream: 37677 rounds/h, 442703 states/h; 917269 rounds in all; buffers: training 3000000 / 3000000, validation 157895; 4 actors.

Bids made by cards dealt 1 / 2-4 / 5+: 0.743 / 0.700 / 0.648; 0-bids 0.657 / 0.567 / 0.336.

## Events

- 22:04 finished: final model checkpoints/rl-2026-10-07a/models/step-002760/model
- 22:04 bench search-vsP step 2760: +0.42 ± 1.10, paired vs previous -0.23 ± 1.31 (849 s)
- 21:50 bench search-rb2 step 2760: +10.97 ± 1.38, paired vs P alone -0.42 ± 0.81 (1679 s)
- 21:22 bench net-rb2-s7 step 2760: +11.39 ± 1.45, paired vs start +0.56 ± 0.81 (11 s)
- 21:22 bench net-rb step 2760: +24.57 ± 1.60, paired vs start +0.51 ± 0.84 (11 s)
- 21:21 bench net-rb2 step 2760: +12.08 ± 0.94, paired vs start +1.07 ± 0.59 (23 s)
- 21:21 published step 2760 (1 s export); the actors switch within seconds
- 21:21 11092 rounds, 249580 examples in all
- 21:21 saved the checkpoint at step 2760; stopping the actors
- 21:20 3.5 h reached: winding down for the final evaluation
- 21:02 bench net-rb2-s7 step 2400: +11.34 ± 1.38, paired vs start +0.51 ± 0.80 (49 s)
- 21:01 bench net-rb step 2400: +24.57 ± 1.62, paired vs start +0.52 ± 0.72 (48 s)
- 21:00 bench net-rb2 step 2400: +12.35 ± 0.97, paired vs start +1.34 ± 0.59 (96 s)
- 20:59 published step 2400 (5 s export); the actors switch within seconds
- 20:37 bench search-vsP step 2000: +0.64 ± 1.02 (833 s)
