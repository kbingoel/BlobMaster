# RL run `checkpoints/rl-2026-10-06`

Updated 05:11 · running 6.01 h of 8.0 (wall 6.01 h) · **stopped (continue with --resume)**

Control: `touch STOP` (save and exit; `--resume` continues), `touch PAUSE` (idle until removed), `touch FINISH` (final benches now).

## Strength

Points per game vs the opponents' mean, 95% CI over deals. Network benches: default seed; paired = same deals as this run's step 0 / the previous publish.

| step | h | bench | deals | diff | paired vs | made 1 / 2–4 / 5–8 | 0-bids 5–8 |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | net-rb2 | 256 | +0.23 ± 0.36 |  | 0.782 / 0.694 / 0.579 | 0.379 |
| 0 | 0.0 | net-rb | 128 | +15.47 ± 1.49 |  | 0.820 / 0.720 / 0.649 | 0.423 |
| 400 | 0.3 | net-rb2 | 256 | +5.94 ± 0.77 | start +5.71 ± 0.78 | 0.781 / 0.719 / 0.636 | 0.380 |
| 400 | 0.3 | net-rb | 128 | +20.87 ± 1.57 | start +5.40 ± 0.97 | 0.822 / 0.745 / 0.703 | 0.419 |
| 800 | 0.6 | net-rb2 | 256 | +8.03 ± 0.78 | start +7.80 ± 0.80; previous +2.09 ± 0.52 | 0.781 / 0.726 / 0.657 | 0.340 |
| 800 | 0.6 | net-rb | 128 | +21.25 ± 1.64 | start +5.78 ± 1.15; previous +0.37 ± 0.76 | 0.822 / 0.753 / 0.694 | 0.378 |
| 1200 | 0.9 | net-rb2 | 256 | +8.46 ± 0.84 | start +8.22 ± 0.89; previous +0.42 ± 0.48 | 0.781 / 0.728 / 0.660 | 0.331 |
| 1200 | 0.9 | net-rb | 128 | +22.37 ± 1.65 | start +6.90 ± 1.23; previous +1.12 ± 0.60 | 0.822 / 0.750 / 0.706 | 0.369 |
| 1600 | 1.1 | net-rb2 | 256 | +9.29 ± 0.86 | start +9.06 ± 0.90; previous +0.84 ± 0.48 | 0.781 / 0.731 / 0.669 | 0.339 |
| 1600 | 1.1 | net-rb | 128 | +22.55 ± 1.71 | start +7.08 ± 1.27; previous +0.18 ± 0.68 | 0.823 / 0.752 / 0.705 | 0.373 |
| 2000 | 1.4 | net-rb2 | 256 | +9.90 ± 0.90 | start +9.67 ± 0.95; previous +0.61 ± 0.47 | 0.781 / 0.731 / 0.674 | 0.316 |
| 2000 | 1.4 | net-rb | 128 | +22.94 ± 1.70 | start +7.47 ± 1.35; previous +0.38 ± 0.62 | 0.822 / 0.749 / 0.713 | 0.353 |
| 2400 | 1.7 | net-rb2 | 256 | +10.43 ± 0.87 | start +10.19 ± 0.92; previous +0.52 ± 0.41 | 0.781 / 0.733 / 0.679 | 0.325 |
| 2400 | 1.7 | net-rb | 128 | +22.52 ± 1.80 | start +7.05 ± 1.40; previous -0.42 ± 0.59 | 0.822 / 0.749 / 0.708 | 0.361 |
| 2800 | 1.9 | net-rb2 | 256 | +9.91 ± 0.91 | start +9.68 ± 0.96; previous -0.51 ± 0.42 | 0.781 / 0.730 / 0.676 | 0.323 |
| 2800 | 2.0 | net-rb | 128 | +23.29 ± 1.72 | start +7.82 ± 1.40; previous +0.77 ± 0.64 | 0.822 / 0.751 / 0.720 | 0.358 |
| 3200 | 2.2 | net-rb2 | 256 | +10.50 ± 0.90 | start +10.26 ± 0.95; previous +0.59 ± 0.47 | 0.781 / 0.730 / 0.683 | 0.322 |
| 3200 | 2.2 | net-rb | 128 | +23.33 ± 1.72 | start +7.86 ± 1.40; previous +0.04 ± 0.59 | 0.823 / 0.751 / 0.717 | 0.356 |
| 3600 | 2.5 | net-rb2 | 256 | +10.53 ± 0.93 | start +10.30 ± 1.00; previous +0.03 ± 0.45 | 0.781 / 0.731 / 0.682 | 0.302 |
| 3600 | 2.5 | net-rb | 128 | +23.33 ± 1.68 | start +7.86 ± 1.38; previous +0.01 ± 0.55 | 0.823 / 0.750 / 0.714 | 0.337 |
| 4000 | 2.8 | net-rb2 | 256 | +10.10 ± 0.89 | start +9.87 ± 0.93; previous -0.43 ± 0.52 | 0.781 / 0.729 / 0.681 | 0.338 |
| 4000 | 2.8 | net-rb | 128 | +23.59 ± 1.68 | start +8.11 ± 1.29; previous +0.25 ± 0.65 | 0.823 / 0.754 / 0.717 | 0.375 |
| 4000 | 3.1 | search-rb2 | 128 | +9.18 ± 1.20 | warm start +2.78 ± 1.26 | 0.787 / 0.713 / 0.669 | 0.320 |
| 4400 | 3.4 | net-rb2 | 256 | +10.71 ± 0.92 | start +10.48 ± 0.99; previous +0.61 ± 0.47 | 0.781 / 0.732 / 0.682 | 0.316 |
| 4400 | 3.4 | net-rb | 128 | +24.08 ± 1.63 | start +8.61 ± 1.39; previous +0.49 ± 0.61 | 0.823 / 0.753 / 0.722 | 0.351 |
| 4800 | 3.7 | net-rb2 | 256 | +10.74 ± 0.88 | start +10.51 ± 0.93; previous +0.04 ± 0.44 | 0.781 / 0.732 / 0.683 | 0.337 |
| 4800 | 3.7 | net-rb | 128 | +23.99 ± 1.71 | start +8.52 ± 1.40; previous -0.09 ± 0.62 | 0.823 / 0.749 / 0.726 | 0.372 |
| 5200 | 4.0 | net-rb2 | 256 | +11.25 ± 0.91 | start +11.02 ± 0.98; previous +0.51 ± 0.41 | 0.781 / 0.731 / 0.691 | 0.317 |
| 5200 | 4.0 | net-rb | 128 | +23.76 ± 1.72 | start +8.29 ± 1.48; previous -0.23 ± 0.51 | 0.823 / 0.752 / 0.719 | 0.351 |
| 5600 | 4.2 | net-rb2 | 256 | +11.13 ± 0.92 | start +10.90 ± 0.97; previous -0.12 ± 0.44 | 0.781 / 0.731 / 0.690 | 0.324 |
| 5600 | 4.2 | net-rb | 128 | +24.18 ± 1.70 | start +8.71 ± 1.37; previous +0.42 ± 0.57 | 0.823 / 0.752 / 0.724 | 0.359 |
| 6000 | 4.5 | net-rb2 | 256 | +11.06 ± 0.92 | start +10.83 ± 0.99; previous -0.07 ± 0.41 | 0.781 / 0.732 / 0.687 | 0.314 |
| 6000 | 4.5 | net-rb | 128 | +24.36 ± 1.64 | start +8.89 ± 1.38; previous +0.18 ± 0.52 | 0.822 / 0.752 / 0.729 | 0.353 |
| 6400 | 4.8 | net-rb2 | 256 | +10.94 ± 0.92 | start +10.71 ± 0.98; previous -0.13 ± 0.41 | 0.781 / 0.732 / 0.686 | 0.317 |
| 6400 | 4.8 | net-rb | 128 | +24.45 ± 1.58 | start +8.98 ± 1.34; previous +0.09 ± 0.59 | 0.823 / 0.754 / 0.725 | 0.351 |
| 6800 | 5.0 | net-rb2 | 256 | +11.04 ± 0.93 | start +10.81 ± 0.99; previous +0.11 ± 0.41 | 0.781 / 0.732 / 0.687 | 0.321 |
| 6800 | 5.1 | net-rb | 128 | +24.13 ± 1.66 | start +8.66 ± 1.42; previous -0.32 ± 0.57 | 0.823 / 0.753 / 0.726 | 0.358 |
| 7200 | 5.3 | net-rb2 | 256 | +10.98 ± 0.94 | start +10.75 ± 1.00; previous -0.07 ± 0.40 | 0.781 / 0.730 / 0.689 | 0.324 |
| 7200 | 5.3 | net-rb | 128 | +24.17 ± 1.62 | start +8.70 ± 1.33; previous +0.04 ± 0.49 | 0.823 / 0.752 / 0.725 | 0.361 |
| 7600 | 5.6 | net-rb2 | 256 | +11.01 ± 0.99 | start +10.78 ± 1.04; previous +0.03 ± 0.44 | 0.781 / 0.731 / 0.687 | 0.314 |
| 7600 | 5.6 | net-rb | 128 | +24.05 ± 1.61 | start +8.58 ± 1.40; previous -0.12 ± 0.61 | 0.823 / 0.753 / 0.722 | 0.351 |
| 7600 | 6.0 | search-rb2 | 128 | +9.05 ± 1.47 | warm start +2.65 ± 1.41; previous -0.13 ± 1.04 | 0.787 / 0.711 / 0.666 | 0.317 |

## Learner

step 7650 · LR 1.00e-4 · P loss 0.6644 · V loss 0.4582 · replay ratio 6.00 (target 6) · 1312 steps/h · waiting on the governor 88.2% · actors' model: step 7600

Held out at step 7600 (self-play rounds; dropout off). Compare the columns: a growing gap means memorization.

| | validation | training sample |
|---|---|---|
| P bid cross-entropy | 0.3999 | 0.4132 |
| P play cross-entropy | 0.8030 | 0.8026 |
| P top = search's top, bids | 0.9512 | 0.9474 |
| P top = search's top, plays | 0.8319 | 0.8421 |
| V MSE | 0.0449 | 0.0432 |
| targets' variance | 0.1386 | 0.1388 |
| V correlation | 0.8224 | 0.8298 |
| V last-trick MSE | 0.0011 | 0.0010 |

Teacher probe (fixed rule-bot-2 rounds): P top = rule bot 2's, bids 0.796 / plays 0.780; V MSE 0.0454 (variance 0.1372), correlation 0.818, last-trick RMSE 0.030.

P's and V's change per publish, on the probe states (KL in nats; V in ŝ):

| step | KL bids / plays vs previous | vs start | V mean abs change vs previous / start |
|---|---|---|---|
| 3200 | 0.0057 / 0.0031 | 0.3369 / 0.1551 | 0.0151 / 0.0294 |
| 3600 | 0.0069 / 0.0029 | 0.3928 / 0.1687 | 0.0159 / 0.0269 |
| 4000 | 0.0133 / 0.0041 | 0.3355 / 0.1657 | 0.0129 / 0.0283 |
| 4400 | 0.0051 / 0.0053 | 0.3412 / 0.1722 | 0.0124 / 0.0278 |
| 4800 | 0.0066 / 0.0035 | 0.3805 / 0.1730 | 0.0136 / 0.0288 |
| 5200 | 0.0047 / 0.0029 | 0.3877 / 0.1733 | 0.0134 / 0.0275 |
| 5600 | 0.0034 / 0.0037 | 0.3913 / 0.1912 | 0.0125 / 0.0272 |
| 6000 | 0.0052 / 0.0032 | 0.3807 / 0.1926 | 0.0135 / 0.0281 |
| 6400 | 0.0062 / 0.0024 | 0.4266 / 0.1930 | 0.0117 / 0.0286 |
| 6800 | 0.0053 / 0.0027 | 0.4073 / 0.1927 | 0.0146 / 0.0283 |
| 7200 | 0.0038 / 0.0029 | 0.3976 / 0.1865 | 0.0114 / 0.0286 |
| 7600 | 0.0057 / 0.0034 | 0.4431 / 0.2002 | 0.0134 / 0.0291 |

## Self-play

Last 60 s: 5999 rounds/h, 143375 examples/h (38.3% forced); 31056 rounds, 691045 examples in all; buffers: training 600000 / 600000, validation 31579; 28 actors, process restarts 0.

| search health | bids | plays |
|---|---|---|
| decisions with a choice | 481 | 994 |
| target's top ≠ P's top | 5.0% | 20.4% |
| KL(target ‖ P) | 0.057 | 0.093 |
| entropy: target / P | 0.426 / 0.457 | 0.718 / 0.801 |
| move played ≠ target's top | 16.4% | 0.0% |

Bids made by cards dealt 1 / 2-4 / 5+: 0.705 / 0.610 / 0.573; 0-bids 0.695 / 0.552 / 0.368.

## Events

- 05:11 stopped
- 05:11 31142 rounds, 693355 examples in all
- 05:11 saved the checkpoint at step 7696; stopping the actors
- 05:11 STOP file: winding down; continue with --resume
- 05:09 bench search-rb2 step 7600: +9.05 ± 1.47, paired vs warm start +2.65 ± 1.41 (1339 s)
- 04:47 bench net-rb step 7600: +24.05 ± 1.61, paired vs start +8.58 ± 1.40 (46 s)
- 04:46 bench net-rb2 step 7600: +11.01 ± 0.99, paired vs start +10.78 ± 1.04 (92 s)
- 04:45 published step 7600 (5 s export); the actors switch within seconds
- 04:30 bench net-rb step 7200: +24.17 ± 1.62, paired vs start +8.70 ± 1.33 (44 s)
- 04:29 bench net-rb2 step 7200: +10.98 ± 0.94, paired vs start +10.75 ± 1.00 (90 s)
- 04:28 published step 7200 (5 s export); the actors switch within seconds
- 04:14 bench net-rb step 6800: +24.13 ± 1.66, paired vs start +8.66 ± 1.42 (47 s)
- 04:13 bench net-rb2 step 6800: +11.04 ± 0.93, paired vs start +10.81 ± 0.99 (94 s)
- 04:12 published step 6800 (5 s export); the actors switch within seconds
- 03:58 bench net-rb step 6400: +24.45 ± 1.58, paired vs start +8.98 ± 1.34 (47 s)
