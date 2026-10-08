# RL run `checkpoints/pi-val3-2026-10-08`

Updated 14:03 · running 1.05 h of 1.0 (wall 1.05 h) · **finished**

Control: `touch STOP` (save and exit; `--resume` continues), `touch PAUSE` (idle until removed), `touch FINISH` (final benches now).

## Strength

Points per game vs the opponents' mean, 95% CI over deals. Network benches: default seed; paired = same deals as this run's step 0 / the previous publish.

| step | h | bench | deals | diff | paired vs | made 1 / 2–4 / 5–8 | 0-bids 5–8 |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | net-rb2 | 512 | +11.90 ± 0.64 |  | 0.782 / 0.736 / 0.692 | 0.310 |
| 0 | 0.0 | net-rb | 128 | +24.66 ± 1.66 |  | 0.823 / 0.752 / 0.726 | 0.355 |
| 0 | 0.0 | net-rb2-s7 | 128 | +11.64 ± 1.37 |  | 0.787 / 0.729 / 0.690 | 0.313 |
| 125 | 0.1 | net-rb2 | 512 | +12.21 ± 0.64 | start +0.31 ± 0.27 | 0.782 / 0.737 / 0.694 | 0.296 |
| 125 | 0.1 | net-rb | 128 | +24.49 ± 1.69 | start -0.17 ± 0.55 | 0.822 / 0.754 / 0.722 | 0.341 |
| 125 | 0.1 | net-rb2-s7 | 128 | +11.95 ± 1.40 | start +0.30 ± 0.53 | 0.787 / 0.731 / 0.691 | 0.296 |
| 125 | 0.2 | net-vs0 | 256 | +0.31 ± 0.41 |  | 0.774 / 0.712 / 0.670 | 0.327 |
| 250 | 0.2 | net-rb2 | 512 | +12.05 ± 0.65 | start +0.15 ± 0.35; previous -0.16 ± 0.32 | 0.782 / 0.737 / 0.694 | 0.311 |
| 250 | 0.2 | net-rb | 128 | +24.55 ± 1.74 | start -0.11 ± 0.72; previous +0.06 ± 0.65 | 0.822 / 0.752 / 0.728 | 0.357 |
| 250 | 0.2 | net-rb2-s7 | 128 | +11.71 ± 1.30 | start +0.07 ± 0.68; previous -0.24 ± 0.64 | 0.787 / 0.731 / 0.689 | 0.313 |
| 250 | 0.3 | net-vs0 | 256 | -0.11 ± 0.48 | previous -0.42 ± 0.46 | 0.774 / 0.713 / 0.668 | 0.344 |
| 375 | 0.3 | net-rb2 | 512 | +12.05 ± 0.64 | start +0.15 ± 0.32; previous +0.01 ± 0.30 | 0.782 / 0.736 / 0.694 | 0.302 |
| 375 | 0.3 | net-rb | 128 | +24.38 ± 1.74 | start -0.27 ± 0.72; previous -0.17 ± 0.62 | 0.823 / 0.750 / 0.726 | 0.351 |
| 375 | 0.3 | net-rb2-s7 | 128 | +11.62 ± 1.38 | start -0.02 ± 0.68; previous -0.09 ± 0.64 | 0.787 / 0.730 / 0.688 | 0.302 |
| 375 | 0.4 | net-vs0 | 256 | -0.09 ± 0.50 | previous +0.02 ± 0.44 | 0.774 / 0.711 / 0.669 | 0.337 |
| 500 | 0.4 | net-rb2 | 512 | +12.43 ± 0.64 | start +0.53 ± 0.37; previous +0.37 ± 0.34 | 0.781 / 0.736 / 0.698 | 0.277 |
| 500 | 0.4 | net-rb | 128 | +24.27 ± 1.71 | start -0.39 ± 0.80; previous -0.12 ± 0.75 | 0.823 / 0.749 / 0.718 | 0.325 |
| 500 | 0.4 | net-rb2-s7 | 128 | +12.16 ± 1.36 | start +0.51 ± 0.79; previous +0.54 ± 0.78 | 0.787 / 0.732 / 0.695 | 0.277 |
| 500 | 0.5 | net-vs0 | 256 | +0.57 ± 0.52 | previous +0.66 ± 0.53 | 0.774 / 0.711 / 0.673 | 0.308 |
| 625 | 0.5 | net-rb2 | 512 | +11.82 ± 0.67 | start -0.08 ± 0.39; previous -0.61 ± 0.36 | 0.782 / 0.736 / 0.690 | 0.301 |
| 625 | 0.5 | net-rb | 128 | +24.27 ± 1.74 | start -0.38 ± 0.79; previous +0.01 ± 0.71 | 0.823 / 0.750 / 0.721 | 0.349 |
| 625 | 0.5 | net-rb2-s7 | 128 | +11.30 ± 1.39 | start -0.34 ± 0.83; previous -0.85 ± 0.70 | 0.787 / 0.730 / 0.683 | 0.302 |
| 625 | 0.5 | net-vs0 | 256 | +0.25 ± 0.49 | previous -0.33 ± 0.49 | 0.774 / 0.712 / 0.670 | 0.334 |
| 875 | 0.6 | net-rb2 | 512 | +12.53 ± 0.64 | start +0.63 ± 0.39; previous +0.71 ± 0.36 | 0.782 / 0.736 / 0.696 | 0.286 |
| 875 | 0.6 | net-rb | 128 | +24.52 ± 1.70 | start -0.14 ± 0.91; previous +0.24 ± 0.73 | 0.822 / 0.754 / 0.717 | 0.339 |
| 875 | 0.6 | net-rb2-s7 | 128 | +11.75 ± 1.40 | start +0.10 ± 0.83; previous +0.44 ± 0.71 | 0.787 / 0.728 / 0.692 | 0.286 |
| 875 | 0.6 | net-vs0 | 256 | +0.23 ± 0.50 | previous -0.02 ± 0.48 | 0.773 / 0.710 / 0.671 | 0.323 |
| 1000 | 0.7 | net-rb2 | 512 | +12.06 ± 0.69 | start +0.15 ± 0.42; previous -0.47 ± 0.36 | 0.782 / 0.736 / 0.687 | 0.261 |
| 1000 | 0.7 | net-rb | 128 | +23.73 ± 1.71 | start -0.93 ± 0.85; previous -0.79 ± 0.81 | 0.823 / 0.748 / 0.709 | 0.305 |
| 1000 | 0.7 | net-rb2-s7 | 128 | +11.65 ± 1.41 | start +0.01 ± 0.83; previous -0.09 ± 0.74 | 0.787 / 0.728 / 0.685 | 0.258 |
| 1000 | 0.7 | net-vs0 | 256 | +0.92 ± 0.56 | previous +0.68 ± 0.53 | 0.774 / 0.710 / 0.675 | 0.292 |
| 1125 | 0.8 | net-rb2 | 512 | +12.79 ± 0.65 | start +0.89 ± 0.38; previous +0.73 ± 0.37 | 0.781 / 0.739 / 0.701 | 0.275 |
| 1125 | 0.8 | net-rb | 128 | +23.24 ± 1.71 | start -1.41 ± 0.84; previous -0.48 ± 0.70 | 0.822 / 0.749 / 0.704 | 0.324 |
| 1125 | 0.8 | net-rb2-s7 | 128 | +12.40 ± 1.33 | start +0.76 ± 0.82; previous +0.75 ± 0.75 | 0.788 / 0.733 / 0.695 | 0.273 |
| 1125 | 0.8 | net-vs0 | 256 | +0.14 ± 0.56 | previous -0.78 ± 0.56 | 0.773 / 0.711 / 0.670 | 0.308 |
| 1250 | 0.9 | net-rb2 | 512 | +12.73 ± 0.66 | start +0.83 ± 0.38; previous -0.06 ± 0.33 | 0.782 / 0.736 / 0.701 | 0.284 |
| 1250 | 0.9 | net-rb | 128 | +23.93 ± 1.64 | start -0.73 ± 0.85; previous +0.68 ± 0.66 | 0.823 / 0.747 / 0.716 | 0.330 |
| 1250 | 0.9 | net-rb2-s7 | 128 | +11.78 ± 1.42 | start +0.14 ± 0.91; previous -0.62 ± 0.72 | 0.787 / 0.728 / 0.693 | 0.283 |
| 1250 | 0.9 | net-vs0 | 256 | +0.69 ± 0.52 | previous +0.55 ± 0.49 | 0.774 / 0.711 / 0.673 | 0.315 |
| 1500 | 1.0 | net-rb2 | 512 | +12.58 ± 0.65 | start +0.67 ± 0.40; previous -0.15 ± 0.31 | 0.781 / 0.736 / 0.700 | 0.296 |
| 1500 | 1.0 | net-rb | 128 | +23.86 ± 1.71 | start -0.80 ± 0.87; previous -0.07 ± 0.65 | 0.822 / 0.751 / 0.714 | 0.345 |
| 1500 | 1.0 | net-rb2-s7 | 128 | +12.47 ± 1.27 | start +0.82 ± 0.90; previous +0.68 ± 0.64 | 0.788 / 0.732 / 0.698 | 0.299 |
| 1500 | 1.0 | net-vs0 | 256 | +0.62 ± 0.53 | previous -0.07 ± 0.49 | 0.773 / 0.712 / 0.674 | 0.330 |
| 1644 | 1.0 | net-rb2 | 512 | +12.44 ± 0.63 | start +0.54 ± 0.39; previous -0.13 ± 0.36 | 0.781 / 0.738 / 0.699 | 0.304 |
| 1644 | 1.0 | net-rb | 128 | +24.56 ± 1.69 | start -0.10 ± 0.84; previous +0.70 ± 0.63 | 0.822 / 0.757 / 0.720 | 0.347 |
| 1644 | 1.0 | net-rb2-s7 | 128 | +11.13 ± 1.40 | start -0.51 ± 0.81; previous -1.33 ± 0.70 | 0.787 / 0.726 / 0.688 | 0.302 |
| 1644 | 1.1 | net-vs0 | 256 | +0.66 ± 0.53 | previous +0.04 ± 0.49 | 0.773 / 0.716 / 0.672 | 0.335 |

## Learner

step 1600 · LR 1.00e-4 · P loss -0.2070 · V loss 0.4629 · replay ratio 1.99 (target 2) · 1877 steps/h · waiting on the governor 35.1% · actors' model: step 1644

P trains on rollouts (T 0.05, ε 0.03): 1642986 samples produced, buffer 1000000 · replay ratio cap 2.

V's own buffer (V stream, rollout next states): 18774 V-only updates/h (GPU 36.6% of the time), V loss 0.4620 · ratio 1.98 (cap 2) · buffer 3000000 states

Held out at step 1644 (self-play rounds; dropout off). Compare the columns: a growing gap means memorization.

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
| V stream: V MSE | 0.0490 | 0.0488 |
| V stream: targets' variance | 0.1450 | 0.1448 |

Rollout samples (utility units on each sample's own deal; gain = P's top move minus the playing P's, the rest of the round by that P):

| | validation | training sample | newest validation |
|---|---|---|---|
| PI loss, bids | -0.3449 | -0.3430 | -0.3422 |
| P's gain, bids | 0.0003 ± 0.0013 | 0.0001 ± 0.0014 | 0.0013 ± 0.0024 |
| P's gain over the start's top move, bids | | | 0.0017 ± 0.0026 (2.1% changed) |
| top move changed, bids | 2.2% | 2.4% | 2.2% |
| V's pick gain, bids | 0.0981 ± 0.0047 | 0.1020 ± 0.0047 | 0.1014 ± 0.0095 |
| hindsight gap, bids | 0.1587 | 0.1613 | 0.1613 |
| PI loss, plays | -0.0564 | -0.0584 | -0.0550 |
| P's gain, plays | -0.0000 ± 0.0011 | 0.0004 ± 0.0010 | 0.0015 ± 0.0022 |
| P's gain over the start's top move, plays | | | 0.0024 ± 0.0027 (6.3% changed) |
| top move changed, plays | 4.0% | 4.0% | 3.5% |
| V's pick gain, plays | 0.0259 ± 0.0027 | 0.0256 ± 0.0026 | 0.0261 ± 0.0054 |
| hindsight gap, plays | 0.0616 | 0.0605 | 0.0611 |

Fixed V check (validation rounds of `checkpoints/rl-2026-10-06`, the same states every row): MSE 0.0451 (bids 0.0789, plays 0.0353), correlation 0.824.

Teacher probe (fixed rule-bot-2 rounds): P top = rule bot 2's, bids 0.792 / plays 0.758; V MSE 0.0414 (variance 0.1372), correlation 0.836, last-trick RMSE 0.015.

P's and V's change per publish, on the probe states (KL in nats; V in ŝ):

| step | KL bids / plays vs previous | vs start | V mean abs change vs previous / start |
|---|---|---|---|
| 250 | 0.0398 / 0.0109 | 0.0647 / 0.0400 | 0.0141 / 0.0255 |
| 375 | 0.0273 / 0.0126 | 0.0540 / 0.0600 | 0.0124 / 0.0271 |
| 500 | 0.0551 / 0.0309 | 0.1074 / 0.0895 | 0.0136 / 0.0262 |
| 625 | 0.0464 / 0.0239 | 0.0717 / 0.1049 | 0.0125 / 0.0276 |
| 750 | 0.0305 / 0.0222 | 0.0823 / 0.0956 | 0.0139 / 0.0292 |
| 875 | 0.0355 / 0.0161 | 0.0992 / 0.0967 | 0.0173 / 0.0292 |
| 1000 | 0.0702 / 0.0212 | 0.1340 / 0.1164 | 0.0134 / 0.0308 |
| 1125 | 0.0515 / 0.0374 | 0.1255 / 0.1023 | 0.0138 / 0.0307 |
| 1250 | 0.0653 / 0.0156 | 0.1255 / 0.1255 | 0.0151 / 0.0313 |
| 1375 | 0.0401 / 0.0143 | 0.1633 / 0.1124 | 0.0135 / 0.0315 |
| 1500 | 0.0431 / 0.0217 | 0.1646 / 0.1359 | 0.0141 / 0.0324 |
| 1625 | 0.0868 / 0.0236 | 0.2049 / 0.1473 | 0.0126 / 0.0317 |

## Self-play

Last 60 s: 0 rounds/h, 0 examples/h (- forced); 0 rounds, 0 examples in all; buffers: training 0 / 600000, validation 0; 0 actors, process restarts 0.

Rollouts: 888477 rounds/h, 1725845 valued decisions/h (14951 bids, 13819 plays; moves per decision 3.22 / 2.87); P's top move not the best on the deal 24.5% / 13.7%, hindsight gap 0.1549 / 0.0596; 924365 rounds in all; buffers: training 1000000 / 1000000, validation 52632; 28 actors.

Bids made by cards dealt 1 / 2-4 / 5+: - / - / -; 0-bids - / - / -.

## Events

- 14:03 finished: final model checkpoints/pi-val3-2026-10-08/models/step-001644/model
- 14:03 bench net-vs0 step 1644: +0.66 ± 0.53, paired vs previous +0.04 ± 0.49 (76 s)
- 14:01 bench net-rb2-s7 step 1644: +11.13 ± 1.40, paired vs start -0.51 ± 0.81 (7 s)
- 14:01 bench net-rb step 1644: +24.56 ± 1.69, paired vs start -0.10 ± 0.84 (7 s)
- 14:01 bench net-rb2 step 1644: +12.44 ± 0.63, paired vs start +0.54 ± 0.39 (29 s)
- 14:01 bench net-vs0 step 1500: +0.62 ± 0.53, paired vs previous -0.07 ± 0.49 (177 s)
- 14:00 published step 1644 (2 s export); the actors switch within seconds
- 14:00 0 rounds, 0 examples in all
- 14:00 saved the checkpoint at step 1644; stopping the actors
- 14:00 1.0 h reached: winding down for the final evaluation
- 13:59 published step 1625 (7 s export); the actors switch within seconds
- 13:58 bench net-rb2-s7 step 1500: +12.47 ± 1.27, paired vs start +0.82 ± 0.90 (22 s)
- 13:57 bench net-rb step 1500: +23.86 ± 1.71, paired vs start -0.80 ± 0.87 (22 s)
- 13:57 bench net-rb2 step 1500: +12.58 ± 0.65, paired vs start +0.67 ± 0.40 (85 s)
- 13:56 bench net-vs0 step 1250: +0.69 ± 0.52, paired vs previous +0.55 ± 0.49 (220 s)
