
## Follow-ups (2026-04-26) — thread-count sweep at B=5 (Session 7.4c stage-1)

After landing the cross-determinization batching driver in
[blob-engine/src/mcts.rs](blob-engine/src/mcts.rs), re-swept thread counts at B=5 lockstep batched ONNX
inference. Same workload as [logs/thread-sweep-2026-04-24/](logs/thread-sweep-2026-04-24/): 5 games per thread,
fixed 5P7C, `iter_000014/model.onnx`, MCTS at flat 5×100. Speedup column compares
per-game wall against the B=1 T=16 baseline (7.249700 s/game).

Script: [scripts/thread-sweep-b5.sh](scripts/thread-sweep-b5.sh), raw logs: [logs/thread-sweep-b5-2026-04-26/](logs/thread-sweep-b5-2026-04-26/).

| threads | total games | wall (s) | per-game wall (s) | per-decision (ms) | onnx_inference avg (µs) | speedup vs B=1 16T |
|--------:|------------:|---------:|------------------:|------------------:|------------------------:|-------------------:|
| 4 | 20 | 333.7 | 16.685 | 175.6 | 3375.1 | 0.435× |
| 6 | 30 | 360.4 | 12.012 | 189.7 | 3664.8 | 0.604× |
| 8 | 40 | 373.3 | 9.332 | 196.5 | 3820.4 | 0.777× |
| 10 | 50 | 384.2 | 7.684 | 202.2 | 3855.3 | 0.943× |
| 12 | 60 | 392.8 | 6.547 | 206.7 | 3964.3 | 1.107× |
| 14 | 70 | 415.1 | 5.930 | 218.5 | 4115.7 | 1.223× |
| 16 | 80 | 424.0 | 5.300 | 223.2 | 4289.4 | 1.368× |

