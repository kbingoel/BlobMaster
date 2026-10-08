# RL run `checkpoints/vfit-2026-10-07`

Updated 13:25 · running 1.09 h of 1.0 (wall 1.09 h) · **finished**

Control: `touch STOP` (save and exit; `--resume` continues), `touch PAUSE` (idle until removed), `touch FINISH` (final benches now).

## Strength

Points per game vs the opponents' mean, 95% CI over deals. Network benches: default seed; paired = same deals as this run's step 0 / the previous publish.

| step | h | bench | deals | diff | paired vs | made 1 / 2–4 / 5–8 | 0-bids 5–8 |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | net-rb2 | 256 | +11.01 ± 0.99 |  | 0.781 / 0.731 / 0.687 | 0.314 |
| 0 | 0.1 | net-rb | 128 | +24.05 ± 1.61 |  | 0.823 / 0.753 / 0.722 | 0.351 |
| 0 | 0.1 | net-rb2-s7 | 2 | +13.00 ± 5.49 |  | 0.820 / 0.767 / 0.700 | 0.317 |
| 0 | 1.0 | net-rb2 | 256 | +11.01 ± 0.99 |  | 0.781 / 0.731 / 0.687 | 0.314 |
| 0 | 1.1 | net-rb | 128 | +24.05 ± 1.61 |  | 0.823 / 0.753 / 0.722 | 0.351 |
| 0 | 1.1 | net-rb2-s7 | 2 | +13.00 ± 5.49 |  | 0.820 / 0.767 / 0.700 | 0.317 |
| 0 | 1.1 | search-rb2 | 2 | +10.35 ± 7.45 | P alone -2.65 ± 1.96 | 0.820 / 0.750 / 0.667 | 0.350 |

## Learner

step 0 · LR 3.33e-7 · P loss - · V loss - · replay ratio 0.00 (target 6) · 0 steps/h · waiting on the governor 56.7% · actors' model: step 0

V stream (rounds of P alone): 20217 V-only updates/h (GPU 42.4% of the time), V loss 0.3920 · ratio 2.97 (cap 3) · buffer 2445349 states

Held out at step 0 (self-play rounds; dropout off). Compare the columns: a growing gap means memorization.

| | validation | training sample |
|---|---|---|
| P bid cross-entropy | 0.3973 | 0.3731 |
| P play cross-entropy | 0.8148 | 0.8710 |
| P top = search's top, bids | 0.9792 | 0.9815 |
| P top = search's top, plays | 0.8879 | 0.8515 |
| V MSE | 0.0677 | 0.0357 |
| targets' variance | 0.1260 | 0.1267 |
| V correlation | 0.6946 | 0.8479 |
| V last-trick MSE | 0.0000 | 0.0010 |
| V stream: V MSE | 0.0204 | 0.0204 |
| V stream: targets' variance | 0.1390 | 0.1390 |

Fixed V check (validation rounds of `checkpoints/rl-2026-10-06`, the same states every row): MSE 0.0460 (bids 0.0790, plays 0.0364), correlation 0.821.

Teacher probe (fixed rule-bot-2 rounds): P top = rule bot 2's, bids 0.796 / plays 0.780; V MSE 0.0446 (variance 0.1372), correlation 0.822, last-trick RMSE 0.024.

## Self-play

Last 60 s: 180 rounds/h, 5699 examples/h (25.3% forced); 207 rounds, 4840 examples in all; buffers: training 4550 / 600000, validation 290; 2 actors, process restarts 0.

| search health | bids | plays |
|---|---|---|
| decisions with a choice | 15 | 56 |
| target's top ≠ P's top | 0.0% | 10.7% |
| KL(target ‖ P) | 0.021 | 0.024 |
| entropy: target / P | 0.372 / 0.467 | 0.809 / 0.786 |
| move played ≠ target's top | 33.3% | 0.0% |

V stream: 91303 rounds/h, 965336 states/h; 241816 rounds in all; buffers: training 2565712 / 3000000, validation 134698; 6 actors.

Bids made by cards dealt 1 / 2-4 / 5+: - / 0.800 / 0.500; 0-bids - / 0.800 / 0.100.

## Events

- 13:25 finished: final model checkpoints/vfit-2026-10-07/models/step-000000/model
- 13:25 bench search-rb2 step 0: +10.35 ± 7.45, paired vs P alone -2.65 ± 1.96 (112 s)
- 13:23 bench net-rb2-s7 step 0: +13.00 ± 5.49 (2 s)
- 13:23 bench net-rb step 0: +24.05 ± 1.61 (57 s)
- 13:22 bench net-rb2 step 0: +11.01 ± 0.99 (116 s)
- 13:20 published step 0 (8 s export); the actors switch within seconds
- 13:20 207 rounds, 4840 examples in all
- 13:19 saved the checkpoint at step 0; stopping the actors
- 13:19 1.0 h reached: winding down for the final evaluation
- 12:23 bench net-rb2-s7 step 0: +13.00 ± 5.49 (3 s)
- 12:23 bench net-rb step 0: +24.05 ± 1.61 (82 s)
- 12:22 bench net-rb2 step 0: +11.01 ± 0.99 (161 s)
- 12:19 started from checkpoints/rl-2026-10-06/models/step-007600/checkpoint (actors: /home/kbuntu/Documents/Github/BlobMaster/checkpoints/rl-2026-10-06/models/step-007600/model)
