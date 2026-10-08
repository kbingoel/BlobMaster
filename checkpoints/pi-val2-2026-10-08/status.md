# RL run `checkpoints/pi-val2-2026-10-08`

Updated 12:49 · running 1.05 h of 1.0 (wall 1.05 h) · **finished**

Control: `touch STOP` (save and exit; `--resume` continues), `touch PAUSE` (idle until removed), `touch FINISH` (final benches now).

## Strength

Points per game vs the opponents' mean, 95% CI over deals. Network benches: default seed; paired = same deals as this run's step 0 / the previous publish.

| step | h | bench | deals | diff | paired vs | made 1 / 2–4 / 5–8 | 0-bids 5–8 |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | net-rb2 | 512 | +11.90 ± 0.64 |  | 0.782 / 0.736 / 0.692 | 0.310 |
| 0 | 0.0 | net-rb | 128 | +24.66 ± 1.66 |  | 0.823 / 0.752 / 0.726 | 0.355 |
| 0 | 0.0 | net-rb2-s7 | 128 | +11.64 ± 1.37 |  | 0.787 / 0.729 / 0.690 | 0.313 |
| 125 | 0.1 | net-rb2 | 512 | +11.88 ± 0.69 | start -0.02 ± 0.34 | 0.782 / 0.736 / 0.689 | 0.300 |
| 125 | 0.1 | net-rb | 128 | +24.71 ± 1.61 | start +0.06 ± 0.60 | 0.823 / 0.753 / 0.722 | 0.347 |
| 125 | 0.1 | net-rb2-s7 | 128 | +11.57 ± 1.44 | start -0.07 ± 0.65 | 0.787 / 0.729 / 0.687 | 0.303 |
| 125 | 0.2 | net-vs0 | 256 | +0.95 ± 0.43 |  | 0.774 / 0.714 / 0.675 | 0.332 |
| 250 | 0.2 | net-rb2 | 512 | +11.49 ± 0.66 | start -0.41 ± 0.37; previous -0.39 ± 0.36 | 0.782 / 0.733 / 0.691 | 0.334 |
| 250 | 0.3 | net-rb | 128 | +24.98 ± 1.67 | start +0.32 ± 0.77; previous +0.27 ± 0.76 | 0.822 / 0.747 / 0.734 | 0.378 |
| 250 | 0.3 | net-rb2-s7 | 128 | +11.38 ± 1.34 | start -0.26 ± 0.74; previous -0.19 ± 0.77 | 0.787 / 0.729 / 0.687 | 0.335 |
| 250 | 0.3 | net-vs0 | 256 | +0.30 ± 0.50 | previous -0.65 ± 0.50 | 0.773 / 0.713 / 0.671 | 0.364 |
| 375 | 0.4 | net-rb2 | 512 | +11.66 ± 0.67 | start -0.24 ± 0.40; previous +0.17 ± 0.35 | 0.781 / 0.732 / 0.693 | 0.328 |
| 375 | 0.4 | net-rb | 128 | +24.75 ± 1.63 | start +0.09 ± 0.78; previous -0.23 ± 0.57 | 0.822 / 0.750 / 0.725 | 0.373 |
| 375 | 0.4 | net-rb2-s7 | 128 | +11.17 ± 1.37 | start -0.47 ± 0.78; previous -0.22 ± 0.67 | 0.787 / 0.725 / 0.689 | 0.325 |
| 375 | 0.4 | net-vs0 | 256 | +0.59 ± 0.54 | previous +0.30 ± 0.49 | 0.773 / 0.712 / 0.675 | 0.358 |
| 500 | 0.5 | net-rb2 | 512 | +11.40 ± 0.70 | start -0.50 ± 0.43; previous -0.25 ± 0.37 | 0.782 / 0.732 / 0.685 | 0.307 |
| 500 | 0.5 | net-rb | 128 | +24.18 ± 1.69 | start -0.48 ± 0.85; previous -0.57 ± 0.74 | 0.822 / 0.748 / 0.722 | 0.354 |
| 500 | 0.5 | net-rb2-s7 | 128 | +10.68 ± 1.36 | start -0.97 ± 0.82; previous -0.49 ± 0.75 | 0.787 / 0.724 / 0.679 | 0.307 |
| 500 | 0.6 | net-vs0 | 256 | +0.98 ± 0.56 | previous +0.39 ± 0.52 | 0.774 / 0.710 / 0.679 | 0.337 |
| 625 | 0.6 | net-rb2 | 512 | +11.30 ± 0.69 | start -0.60 ± 0.41; previous -0.11 ± 0.33 | 0.782 / 0.733 / 0.686 | 0.330 |
| 625 | 0.6 | net-rb | 128 | +24.62 ± 1.65 | start -0.04 ± 0.78; previous +0.44 ± 0.64 | 0.822 / 0.751 / 0.726 | 0.377 |
| 625 | 0.6 | net-rb2-s7 | 128 | +11.36 ± 1.41 | start -0.28 ± 0.80; previous +0.68 ± 0.64 | 0.787 / 0.727 / 0.686 | 0.333 |
| 625 | 0.7 | net-vs0 | 256 | +0.80 ± 0.54 | previous -0.18 ± 0.49 | 0.773 / 0.714 / 0.676 | 0.363 |
| 750 | 0.7 | net-rb2 | 512 | +11.36 ± 0.64 | start -0.54 ± 0.41; previous +0.07 ± 0.35 | 0.782 / 0.731 / 0.691 | 0.349 |
| 750 | 0.7 | net-rb | 128 | +25.00 ± 1.64 | start +0.35 ± 0.79; previous +0.38 ± 0.65 | 0.822 / 0.755 / 0.729 | 0.391 |
| 750 | 0.7 | net-rb2-s7 | 128 | +10.87 ± 1.38 | start -0.77 ± 0.83; previous -0.49 ± 0.68 | 0.787 / 0.727 / 0.683 | 0.353 |
| 750 | 0.8 | net-vs0 | 256 | +0.78 ± 0.55 | previous -0.02 ± 0.46 | 0.774 / 0.715 / 0.677 | 0.379 |
| 875 | 0.8 | net-rb2 | 512 | +11.66 ± 0.66 | start -0.24 ± 0.42; previous +0.29 ± 0.35 | 0.781 / 0.733 / 0.689 | 0.340 |
| 875 | 0.8 | net-rb | 128 | +25.28 ± 1.70 | start +0.62 ± 0.80; previous +0.28 ± 0.64 | 0.822 / 0.752 / 0.736 | 0.385 |
| 875 | 0.8 | net-rb2-s7 | 128 | +11.38 ± 1.39 | start -0.26 ± 0.82; previous +0.51 ± 0.63 | 0.787 / 0.727 / 0.687 | 0.342 |
| 875 | 0.9 | net-vs0 | 256 | +0.57 ± 0.56 | previous -0.21 ± 0.51 | 0.773 / 0.713 / 0.673 | 0.370 |
| 1000 | 0.9 | net-rb2 | 512 | +12.04 ± 0.68 | start +0.14 ± 0.42; previous +0.38 ± 0.35 | 0.782 / 0.734 / 0.696 | 0.339 |
| 1000 | 0.9 | net-rb | 128 | +24.60 ± 1.62 | start -0.06 ± 0.85; previous -0.68 ± 0.65 | 0.822 / 0.749 / 0.726 | 0.380 |
| 1000 | 1.0 | net-rb2-s7 | 128 | +11.68 ± 1.34 | start +0.04 ± 0.77; previous +0.30 ± 0.64 | 0.788 / 0.728 / 0.693 | 0.340 |
| 1000 | 1.0 | net-vs0 | 256 | +0.58 ± 0.53 | previous +0.01 ± 0.48 | 0.773 / 0.714 / 0.674 | 0.368 |
| 1095 | 1.0 | net-rb2 | 512 | +11.87 ± 0.66 | start -0.03 ± 0.42; previous -0.17 ± 0.37 | 0.782 / 0.733 / 0.693 | 0.315 |
| 1095 | 1.0 | net-rb | 128 | +24.71 ± 1.66 | start +0.05 ± 0.84; previous +0.11 ± 0.65 | 0.823 / 0.746 / 0.727 | 0.359 |
| 1095 | 1.0 | net-rb2-s7 | 128 | +11.39 ± 1.43 | start -0.25 ± 0.85; previous -0.29 ± 0.70 | 0.787 / 0.726 / 0.686 | 0.312 |
| 1095 | 1.0 | net-vs0 | 256 | +0.84 ± 0.59 | previous +0.26 ± 0.50 | 0.774 / 0.713 / 0.676 | 0.345 |

## Learner

step 1086 · LR 1.00e-4 · P loss -0.2075 · V loss 0.4610 · replay ratio 2.00 (target 2) · 1079 steps/h · waiting on the governor 60.5% · actors' model: step 1095

P trains on rollouts (T 0.05, ε 0.03): 1113146 samples produced, buffer 1000000 · replay ratio cap 2.

V's own buffer (V stream, rollout next states): 12500 V-only updates/h (GPU 24.5% of the time), V loss 0.4636 · ratio 1.98 (cap 2) · buffer 3000000 states

Held out at step 1095 (self-play rounds; dropout off). Compare the columns: a growing gap means memorization.

| | validation | training sample |
|---|---|---|
| P bid cross-entropy | - | - |
| P play cross-entropy | - | - |
| P top = search's top, bids | - | - |
| P top = search's top, plays | - | - |
| V MSE | - | - |
| targets' variance | - | - |
| V correlation | - | - |
| V last-trick MSE | - | - |
| V stream: V MSE | 0.0480 | 0.0481 |
| V stream: targets' variance | 0.1425 | 0.1432 |

Rollout samples (utility units on each sample's own deal; gain = P's top move minus the playing P's, the rest of the round by that P):

| | validation | training sample | newest validation |
|---|---|---|---|
| PI loss, bids | -0.3431 | -0.3437 | -0.3491 |
| P's gain, bids | 0.0001 ± 0.0012 | 0.0023 ± 0.0013 | 0.0002 ± 0.0021 |
| P's gain over the start's top move, bids | | | -0.0004 ± 0.0026 (2.6% changed) |
| top move changed, bids | 2.1% | 2.0% | 1.6% |
| V's pick gain, bids | 0.1000 ± 0.0047 | 0.0958 ± 0.0047 | 0.0953 ± 0.0092 |
| hindsight gap, bids | 0.1594 | 0.1572 | 0.1494 |
| PI loss, plays | -0.0606 | -0.0589 | -0.0565 |
| P's gain, plays | 0.0005 ± 0.0010 | 0.0015 ± 0.0011 | -0.0010 ± 0.0019 |
| P's gain over the start's top move, plays | | | -0.0007 ± 0.0025 (5.1% changed) |
| top move changed, plays | 4.0% | 3.9% | 3.2% |
| V's pick gain, plays | 0.0234 ± 0.0026 | 0.0240 ± 0.0026 | 0.0233 ± 0.0051 |
| hindsight gap, plays | 0.0572 | 0.0561 | 0.0572 |

Fixed V check (validation rounds of `checkpoints/rl-2026-10-06`, the same states every row): MSE 0.0455 (bids 0.0779, plays 0.0361), correlation 0.823.

Teacher probe (fixed rule-bot-2 rounds): P top = rule bot 2's, bids 0.783 / plays 0.758; V MSE 0.0458 (variance 0.1372), correlation 0.817, last-trick RMSE 0.025.

P's and V's change per publish, on the probe states (KL in nats; V in ŝ):

| step | KL bids / plays vs previous | vs start | V mean abs change vs previous / start |
|---|---|---|---|
| 125 | 0.0442 / 0.0146 | 0.0442 / 0.0146 | 0.0206 / 0.0206 |
| 250 | 0.0756 / 0.0187 | 0.0756 / 0.0422 | 0.0133 / 0.0225 |
| 375 | 0.0617 / 0.0127 | 0.1108 / 0.0561 | 0.0139 / 0.0242 |
| 500 | 0.0577 / 0.0397 | 0.1009 / 0.0904 | 0.0148 / 0.0250 |
| 625 | 0.0487 / 0.0181 | 0.1187 / 0.0934 | 0.0136 / 0.0254 |
| 750 | 0.0626 / 0.0281 | 0.1839 / 0.0937 | 0.0126 / 0.0267 |
| 875 | 0.0781 / 0.0229 | 0.1247 / 0.1052 | 0.0152 / 0.0278 |
| 1000 | 0.0473 / 0.0269 | 0.1322 / 0.1034 | 0.0147 / 0.0276 |

## Self-play

Last 60 s: 0 rounds/h, 0 examples/h (- forced); 0 rounds, 0 examples in all; buffers: training 0 / 600000, validation 0; 0 actors, process restarts 0.

Rollouts: 593089 rounds/h, 1186178 valued decisions/h (10304 bids, 9472 plays; moves per decision 3.16 / 2.86); P's top move not the best on the deal 24.5% / 13.0%, hindsight gap 0.1608 / 0.0575; 600241 rounds in all; buffers: training 1000000 / 1000000, validation 52632; 28 actors.

Bids made by cards dealt 1 / 2-4 / 5+: - / - / -; 0-bids - / - / -.

## Events

- 12:49 finished: final model checkpoints/pi-val2-2026-10-08/models/step-001095/model
- 12:49 bench net-vs0 step 1095: +0.84 ± 0.59, paired vs previous +0.26 ± 0.50 (76 s)
- 12:48 bench net-rb2-s7 step 1095: +11.39 ± 1.43, paired vs start -0.25 ± 0.85 (7 s)
- 12:47 bench net-rb step 1095: +24.71 ± 1.66, paired vs start +0.05 ± 0.84 (7 s)
- 12:47 bench net-rb2 step 1095: +11.87 ± 0.66, paired vs start -0.03 ± 0.42 (29 s)
- 12:47 published step 1095 (2 s export); the actors switch within seconds
- 12:47 bench net-vs0 step 1000: +0.58 ± 0.53, paired vs previous +0.01 ± 0.48 (211 s)
- 12:47 0 rounds, 0 examples in all
- 12:47 saved the checkpoint at step 1095; stopping the actors
- 12:46 1.0 h reached: winding down for the final evaluation
- 12:43 bench net-rb2-s7 step 1000: +11.68 ± 1.34, paired vs start +0.04 ± 0.77 (21 s)
- 12:43 bench net-rb step 1000: +24.60 ± 1.62, paired vs start -0.06 ± 0.85 (21 s)
- 12:43 bench net-rb2 step 1000: +12.04 ± 0.68, paired vs start +0.14 ± 0.42 (82 s)
- 12:41 published step 1000 (4 s export); the actors switch within seconds
- 12:39 bench net-vs0 step 875: +0.57 ± 0.56, paired vs previous -0.21 ± 0.51 (216 s)
