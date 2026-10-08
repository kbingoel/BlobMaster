# RL run `checkpoints/rl-2026-10-07b`

Updated 02:18 · running 4.22 h of 3.5 (wall 4.22 h) · **finished**

Control: `touch STOP` (save and exit; `--resume` continues), `touch PAUSE` (idle until removed), `touch FINISH` (final benches now).

## Strength

Points per game vs the opponents' mean, 95% CI over deals. Network benches: default seed; paired = same deals as this run's step 0 / the previous publish.

| step | h | bench | deals | diff | paired vs | made 1 / 2–4 / 5–8 | 0-bids 5–8 |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | net-rb2 | 256 | +11.01 ± 0.99 |  | 0.781 / 0.731 / 0.687 | 0.314 |
| 0 | 0.0 | net-rb | 128 | +24.05 ± 1.61 |  | 0.823 / 0.753 / 0.722 | 0.351 |
| 0 | 0.0 | net-rb2-s7 | 128 | +10.82 ± 1.38 |  | 0.787 / 0.721 / 0.691 | 0.312 |
| 400 | 0.4 | net-rb2 | 256 | +11.76 ± 0.97 | start +0.75 ± 0.55 | 0.781 / 0.734 / 0.691 | 0.310 |
| 400 | 0.4 | net-rb | 128 | +24.29 ± 1.56 | start +0.24 ± 0.70 | 0.823 / 0.748 / 0.723 | 0.347 |
| 400 | 0.4 | net-rb2-s7 | 128 | +11.27 ± 1.41 | start +0.45 ± 0.73 | 0.787 / 0.725 / 0.691 | 0.311 |
| 800 | 0.7 | net-rb2 | 256 | +11.70 ± 0.93 | start +0.69 ± 0.60; previous -0.06 ± 0.49 | 0.781 / 0.735 / 0.686 | 0.286 |
| 800 | 0.7 | net-rb | 128 | +24.10 ± 1.64 | start +0.04 ± 0.75; previous -0.19 ± 0.58 | 0.823 / 0.749 / 0.717 | 0.326 |
| 800 | 0.7 | net-rb2-s7 | 128 | +11.06 ± 1.39 | start +0.23 ± 0.77; previous -0.21 ± 0.72 | 0.787 / 0.726 / 0.683 | 0.284 |
| 1200 | 1.0 | net-rb2 | 256 | +11.67 ± 0.94 | start +0.66 ± 0.56; previous -0.03 ± 0.49 | 0.781 / 0.734 / 0.691 | 0.307 |
| 1200 | 1.1 | net-rb | 128 | +24.64 ± 1.65 | start +0.58 ± 0.72; previous +0.54 ± 0.62 | 0.822 / 0.753 / 0.725 | 0.350 |
| 1200 | 1.1 | net-rb2-s7 | 128 | +11.44 ± 1.32 | start +0.62 ± 0.68; previous +0.38 ± 0.63 | 0.787 / 0.728 / 0.691 | 0.307 |
| 1600 | 1.4 | net-rb2 | 256 | +11.73 ± 0.92 | start +0.72 ± 0.59; previous +0.06 ± 0.43 | 0.781 / 0.735 / 0.688 | 0.298 |
| 1600 | 1.4 | net-rb | 128 | +24.29 ± 1.66 | start +0.23 ± 0.77; previous -0.35 ± 0.50 | 0.822 / 0.752 / 0.719 | 0.340 |
| 1600 | 1.4 | net-rb2-s7 | 128 | +11.42 ± 1.40 | start +0.60 ± 0.85; previous -0.02 ± 0.67 | 0.787 / 0.728 / 0.687 | 0.300 |
| 2000 | 1.7 | net-rb2 | 256 | +11.99 ± 0.96 | start +0.98 ± 0.54; previous +0.26 ± 0.39 | 0.781 / 0.737 / 0.692 | 0.304 |
| 2000 | 1.7 | net-rb | 128 | +24.53 ± 1.65 | start +0.48 ± 0.79; previous +0.24 ± 0.46 | 0.823 / 0.752 / 0.722 | 0.345 |
| 2000 | 1.8 | net-rb2-s7 | 128 | +11.63 ± 1.35 | start +0.81 ± 0.75; previous +0.21 ± 0.56 | 0.787 / 0.729 / 0.690 | 0.304 |
| 2000 | 2.2 | search-rb2 | 128 | +11.39 ± 1.38 | P alone -0.24 ± 0.75; baseline +2.34 ± 1.19 | 0.787 / 0.728 / 0.682 | 0.301 |
| 2000 | 2.5 | search-vsP | 64 | +0.94 ± 1.05 |  | 0.766 / 0.717 / 0.657 | 0.336 |
| 2400 | 2.8 | net-rb2 | 256 | +12.07 ± 0.96 | start +1.06 ± 0.62; previous +0.08 ± 0.41 | 0.781 / 0.736 / 0.692 | 0.303 |
| 2400 | 2.8 | net-rb | 128 | +24.23 ± 1.65 | start +0.18 ± 0.78; previous -0.30 ± 0.50 | 0.823 / 0.750 / 0.721 | 0.345 |
| 2400 | 2.8 | net-rb2-s7 | 128 | +11.20 ± 1.42 | start +0.37 ± 0.79; previous -0.43 ± 0.62 | 0.787 / 0.726 / 0.687 | 0.304 |
| 2800 | 3.1 | net-rb2 | 256 | +11.81 ± 0.91 | start +0.80 ± 0.57; previous -0.27 ± 0.42 | 0.781 / 0.735 / 0.690 | 0.310 |
| 2800 | 3.1 | net-rb | 128 | +24.43 ± 1.66 | start +0.38 ± 0.78; previous +0.20 ± 0.48 | 0.823 / 0.751 / 0.722 | 0.351 |
| 2800 | 3.1 | net-rb2-s7 | 128 | +11.30 ± 1.39 | start +0.47 ± 0.77; previous +0.10 ± 0.53 | 0.787 / 0.728 / 0.686 | 0.312 |
| 3200 | 3.4 | net-rb2 | 256 | +12.06 ± 0.93 | start +1.05 ± 0.60; previous +0.25 ± 0.40 | 0.781 / 0.736 / 0.693 | 0.302 |
| 3200 | 3.5 | net-rb | 128 | +24.30 ± 1.59 | start +0.25 ± 0.78; previous -0.13 ± 0.42 | 0.823 / 0.751 / 0.720 | 0.347 |
| 3200 | 3.5 | net-rb2-s7 | 128 | +11.33 ± 1.39 | start +0.50 ± 0.73; previous +0.03 ± 0.56 | 0.787 / 0.728 / 0.686 | 0.306 |
| 3285 | 3.5 | net-rb2 | 256 | +12.30 ± 0.91 | start +1.29 ± 0.58; previous +0.24 ± 0.38 | 0.781 / 0.736 / 0.697 | 0.310 |
| 3285 | 3.5 | net-rb | 128 | +24.66 ± 1.66 | start +0.60 ± 0.83; previous +0.36 ± 0.54 | 0.823 / 0.752 / 0.726 | 0.355 |
| 3285 | 3.5 | net-rb2-s7 | 128 | +11.64 ± 1.37 | start +0.82 ± 0.71; previous +0.32 ± 0.55 | 0.787 / 0.729 / 0.690 | 0.313 |
| 3285 | 4.0 | search-rb2 | 128 | +11.86 ± 1.31 | P alone +0.22 ± 0.76; baseline +2.81 ± 1.09; previous +0.47 ± 0.72 | 0.788 / 0.730 / 0.688 | 0.314 |
| 3285 | 4.2 | search-vsP | 64 | +0.65 ± 1.23 | previous -0.29 ± 1.38 | 0.765 / 0.719 / 0.653 | 0.352 |

## Learner

step 3250 · LR 1.00e-4 · P loss 0.3694 · V loss 0.4724 · replay ratio 5.98 (target 6) · 1125 steps/h · waiting on the governor 94.9% · actors' model: step 3285

Held out at step 3285 (self-play rounds; dropout off). Compare the columns: a growing gap means memorization.

| | validation | training sample |
|---|---|---|
| P bid cross-entropy | 0.1211 | 0.1269 |
| P play cross-entropy | 0.5049 | 0.4755 |
| P top = search's top, bids | 0.9644 | 0.9658 |
| P top = search's top, plays | 0.9107 | 0.9237 |
| V MSE | 0.0424 | 0.0395 |
| targets' variance | 0.1316 | 0.1329 |
| V correlation | 0.8237 | 0.8382 |
| V last-trick MSE | 0.0010 | 0.0010 |

Fixed V check (validation rounds of `checkpoints/rl-2026-10-06`, the same states every row): MSE 0.0464 (bids 0.0771, plays 0.0375), correlation 0.819.

Teacher probe (fixed rule-bot-2 rounds): P top = rule bot 2's, bids 0.798 / plays 0.765; V MSE 0.0464 (variance 0.1372), correlation 0.814, last-trick RMSE 0.029.

P's and V's change per publish, on the probe states (KL in nats; V in ŝ):

| step | KL bids / plays vs previous | vs start | V mean abs change vs previous / start |
|---|---|---|---|
| 400 | 0.1973 / 0.1031 | 0.1973 / 0.1031 | 0.0242 / 0.0242 |
| 800 | 0.0151 / 0.0096 | 0.2537 / 0.1512 | 0.0180 / 0.0253 |
| 1200 | 0.0201 / 0.0059 | 0.3232 / 0.1657 | 0.0163 / 0.0277 |
| 1600 | 0.0107 / 0.0058 | 0.3357 / 0.1869 | 0.0142 / 0.0264 |
| 2000 | 0.0089 / 0.0058 | 0.3888 / 0.2081 | 0.0150 / 0.0276 |
| 2400 | 0.0106 / 0.0060 | 0.3679 / 0.2204 | 0.0132 / 0.0258 |
| 2800 | 0.0090 / 0.0072 | 0.4037 / 0.2250 | 0.0125 / 0.0256 |
| 3200 | 0.0062 / 0.0062 | 0.3883 / 0.2376 | 0.0120 / 0.0249 |

## Self-play

Last 60 s: 3720 rounds/h, 100789 examples/h (34.9% forced); 13213 rounds, 296770 examples in all; buffers: training 281820 / 600000, validation 14950; 24 actors, process restarts 0.

| search health | bids | plays |
|---|---|---|
| decisions with a choice | 301 | 792 |
| target's top ≠ P's top | 3.0% | 7.6% |
| KL(target ‖ P) | 0.041 | 0.067 |
| entropy: target / P | 0.079 / 0.134 | 0.399 / 0.508 |
| move played ≠ target's top | 3.7% | 0.0% |

Bids made by cards dealt 1 / 2-4 / 5+: 0.800 / 0.716 / 0.694; 0-bids 0.733 / 0.695 / 0.329.

## Events

- 02:18 finished: final model checkpoints/rl-2026-10-07b/models/step-003285/model
- 02:18 bench search-vsP step 3285: +0.65 ± 1.23, paired vs previous -0.29 ± 1.38 (847 s)
- 02:03 bench search-rb2 step 3285: +11.86 ± 1.31, paired vs P alone +0.22 ± 0.76 (1677 s)
- 01:35 bench net-rb2-s7 step 3285: +11.64 ± 1.37, paired vs start +0.82 ± 0.71 (11 s)
- 01:35 bench net-rb step 3285: +24.66 ± 1.66, paired vs start +0.60 ± 0.83 (11 s)
- 01:35 bench net-rb2 step 3285: +12.30 ± 0.91, paired vs start +1.29 ± 0.58 (23 s)
- 01:35 published step 3285 (1 s export); the actors switch within seconds
- 01:35 13213 rounds, 296770 examples in all
- 01:34 saved the checkpoint at step 3285; stopping the actors
- 01:34 3.5 h reached: winding down for the final evaluation
- 01:32 bench net-rb2-s7 step 3200: +11.33 ± 1.39, paired vs start +0.50 ± 0.73 (36 s)
- 01:31 bench net-rb step 3200: +24.30 ± 1.59, paired vs start +0.25 ± 0.78 (40 s)
- 01:31 bench net-rb2 step 3200: +12.06 ± 0.93, paired vs start +1.05 ± 0.60 (74 s)
- 01:29 published step 3200 (4 s export); the actors switch within seconds
- 01:12 bench net-rb2-s7 step 2800: +11.30 ± 1.39, paired vs start +0.47 ± 0.77 (40 s)
