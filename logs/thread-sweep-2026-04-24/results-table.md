
## Follow-ups (2026-04-24) — thread-count sweep after evaluator-reuse

After landing the per-thread `OnnxEvaluator` reuse change ([blob-nn/src/engine.rs](blob-nn/src/engine.rs)),
re-baselined T=16 and swept neighbouring thread counts to find the per-game-wall optimum.
Each row is 5 games per thread (so total games = 5 × T). Same model, MCTS, and game shape as the original profile.

Algorithm: from T=16, walk one step in each direction; on a regression vs best-so-far, take one validation
step further; stop that direction if the validation step also regresses. Script: [scripts/thread-sweep.sh](scripts/thread-sweep.sh),
raw logs: [logs/thread-sweep-2026-04-24/](logs/thread-sweep-2026-04-24/).

| threads | total games | wall (s) | per-game wall (s) | per-decision (ms) | onnx_inference avg (µs) | vs 16T |
|--------:|------------:|---------:|------------------:|------------------:|------------------------:|-------:|
| 14 | 70 | 529.2 | 7.560 | 278.5 | 1186.6 | 1.043× |
| 15 | 75 | 548.6 | 7.315 | 288.7 | 1229.7 | 1.009× |
| 16 | 80 | 580.0 | 7.250 | 305.3 | 1290.2 | 1.000× |
| 17 | 85 | 646.2 | 7.602 | 340.1 | 1355.1 | 1.049× |
| 18 | 90 | 686.6 | 7.629 | 361.4 | 1441.9 | 1.052× |

Run order (sequential, one config at a time):

```
16 (per_game=7.249700s)
15 (per_game=7.314957s)
14 (per_game=7.560087s)
17 (per_game=7.602072s)
18 (per_game=7.629394s)
```

**Direction bests:** DOWN best at T= (7.249700s/game), UP best at T=16 (7.249700s/game), baseline T=16 (7.249700s/game).
