# RL run `checkpoints/pi-val-2026-10-08`

Updated 11:46 · running 0.32 h of 1.0 (wall 0.32 h) · **stopped (continue with --resume)**

Control: `touch STOP` (save and exit; `--resume` continues), `touch PAUSE` (idle until removed), `touch FINISH` (final benches now).

## Strength

Points per game vs the opponents' mean, 95% CI over deals. Network benches: default seed; paired = same deals as this run's step 0 / the previous publish.

| step | h | bench | deals | diff | paired vs | made 1 / 2–4 / 5–8 | 0-bids 5–8 |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | net-rb2 | 512 | +11.90 ± 0.64 |  | 0.782 / 0.736 / 0.692 | 0.310 |
| 0 | 0.0 | net-rb | 128 | +24.66 ± 1.66 |  | 0.823 / 0.752 / 0.726 | 0.355 |
| 0 | 0.1 | net-rb2-s7 | 128 | +11.64 ± 1.37 |  | 0.787 / 0.729 / 0.690 | 0.313 |
| 500 | 0.1 | net-rb2 | 512 | +11.13 ± 0.64 | start -0.78 ± 0.44 | 0.781 / 0.732 / 0.689 | 0.326 |
| 500 | 0.1 | net-rb | 128 | +24.39 ± 1.71 | start -0.26 ± 0.76 | 0.823 / 0.754 / 0.722 | 0.367 |
| 500 | 0.1 | net-rb2-s7 | 128 | +11.01 ± 1.32 | start -0.63 ± 0.84 | 0.787 / 0.732 / 0.683 | 0.323 |

## Learner

step 2100 · LR 1.00e-4 · P loss -0.2078 · V loss 0.4628 · replay ratio 2.98 (target 3) · 11893 steps/h · waiting on the governor 0.0% · actors' model: step 2000

P trains on rollouts (T 0.05, ε 0.03): 360736 samples produced, buffer 360736 · replay ratio cap 3.

V's own buffer (V stream, rollout next states): 26165 V-only updates/h (GPU 49.2% of the time), V loss 0.4649 · ratio 1.94 (cap 2) · buffer 1090918 states

Held out at step 2000 (self-play rounds; dropout off). Compare the columns: a growing gap means memorization.

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
| V stream: V MSE | 0.0484 | 0.0493 |
| V stream: targets' variance | 0.1445 | 0.1436 |

Rollout samples (utility units on each sample's own deal; gain = P's top move minus the playing P's, the rest of the round by that P):

| | validation | training sample | newest validation |
|---|---|---|---|
| PI loss, bids | -0.3418 | -0.3436 | -0.3378 |
| P's gain, bids | -0.0021 ± 0.0030 | 0.0022 ± 0.0032 | -0.0024 ± 0.0057 |
| P's gain over the start's top move, bids | | | -0.0063 ± 0.0065 (3.5% changed) |
| top move changed, bids | 2.9% | 2.8% | 2.8% |
| V's pick gain, bids | 0.0985 ± 0.0092 | 0.0961 ± 0.0095 | 0.1095 ± 0.0193 |
| hindsight gap, bids | 0.1586 | 0.1619 | 0.1696 |
| PI loss, plays | -0.0608 | -0.0625 | -0.0646 |
| P's gain, plays | 0.0019 ± 0.0025 | 0.0015 ± 0.0024 | 0.0017 ± 0.0039 |
| P's gain over the start's top move, plays | | | 0.0001 ± 0.0050 (5.9% changed) |
| top move changed, plays | 5.0% | 5.3% | 4.1% |
| V's pick gain, plays | 0.0250 ± 0.0053 | 0.0212 ± 0.0052 | 0.0260 ± 0.0099 |
| hindsight gap, plays | 0.0613 | 0.0564 | 0.0551 |

Fixed V check (validation rounds of `checkpoints/rl-2026-10-06`, the same states every row): MSE 0.0455 (bids 0.0764, plays 0.0365), correlation 0.823.

Teacher probe (fixed rule-bot-2 rounds): P top = rule bot 2's, bids 0.807 / plays 0.764; V MSE 0.0450 (variance 0.1372), correlation 0.820, last-trick RMSE 0.026.

P's and V's change per publish, on the probe states (KL in nats; V in ŝ):

| step | KL bids / plays vs previous | vs start | V mean abs change vs previous / start |
|---|---|---|---|
| 500 | 0.1450 / 0.0662 | 0.1450 / 0.0662 | 0.0219 / 0.0219 |
| 1000 | 0.1248 / 0.0298 | 0.2003 / 0.0952 | 0.0149 / 0.0244 |
| 1500 | 0.1160 / 0.0567 | 0.1638 / 0.1219 | 0.0153 / 0.0249 |
| 2000 | 0.1318 / 0.0349 | 0.2554 / 0.1406 | 0.0151 / 0.0248 |

## Self-play

Last 60 s: 0 rounds/h, 0 examples/h (- forced); 0 rounds, 0 examples in all; buffers: training 0 / 600000, validation 0; 0 actors, process restarts 0.

Rollouts: 633424 rounds/h, 1266847 valued decisions/h (11109 bids, 10009 plays; moves per decision 3.18 / 2.86); P's top move not the best on the deal 26.0% / 13.2%, hindsight gap 0.1708 / 0.0548; 189795 rounds in all; buffers: training 360736 / 1000000, validation 18854; 28 actors.

Bids made by cards dealt 1 / 2-4 / 5+: - / - / -; 0-bids - / - / -.

## Events

- 11:46 stopped
- 11:46 0 rounds, 0 examples in all
- 11:46 saved the checkpoint at step 2138; stopping the actors
- 11:46 STOP file: winding down; continue with --resume
- 11:45 published step 2000 (6 s export); the actors switch within seconds
- 11:41 published step 1500 (5 s export); the actors switch within seconds
- 11:37 published step 1000 (7 s export); the actors switch within seconds
- 11:36 bench net-rb2-s7 step 500: +11.01 ± 1.32, paired vs start -0.63 ± 0.84 (36 s)
- 11:35 bench net-rb step 500: +24.39 ± 1.71, paired vs start -0.26 ± 0.76 (35 s)
- 11:34 bench net-rb2 step 500: +11.13 ± 0.64, paired vs start -0.78 ± 0.44 (142 s)
- 11:32 published step 500 (5 s export); the actors switch within seconds
- 11:30 bench net-rb2-s7 step 0: +11.64 ± 1.37 (39 s)
- 11:30 learner starts at step 0 on 55426 rollout samples
- 11:30 bench net-rb step 0: +24.66 ± 1.66 (37 s)
- 11:29 bench net-rb2 step 0: +11.90 ± 0.64 (139 s)
