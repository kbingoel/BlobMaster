Profiling Artifacts Overview

This document lists the files to read for a complete view of the self‑play profiling setup and outputs. It describes what each file contains and where profiling measures are collected. Use this as a navigation guide; it does not include analysis of results.

How To Run
- `ml/profile_selfplay.py`: Orchestrates profiling phases and output files.
  - Default phases: Sessions 1+2 validation, cProfile run, manual timing, evaluator overhead.
  - Flags:
    - `--instrument`: enable lightweight instrumentation; writes per‑run aggregates.
    - `--compare-overhead`: runs with/without worker cProfile to measure overhead.

Primary Outputs (keep/read)
- Aggregate metrics (instrumentation): `profile_<run_id>_aggregate.json`
  - Aggregated determinization, node, and batch evaluator metrics for the run.
- cProfile dump: `profile_w{workers}_g{games}_c{cards}.prof`
  - Function‑level profile for inspection with `python -m pstats`.
- Helper: `inspect_profile.sh`
  - Opens the most recent `*.prof` in `pstats` for interactive inspection.

Instrumentation: What and Where It Measures
- Determinization sampling (imperfect‑info MCTS)
  - File: `ml/mcts/determinization.py`
  - APIs: `enable_metrics()`, `reset_metrics()`, `get_metrics()`
  - Measures: calls, attempts, successes, total/avg sample time, validation counts/time.

- MCTS node expansion (copy + apply action)
  - File: `ml/mcts/node.py`
  - APIs: `enable_metrics()`, `reset_metrics()`, `get_metrics()`
  - Measures: `simulate_action` calls and total/avg time.

- Batched neural evaluator (GPU batching)
  - File: `ml/mcts/batch_evaluator.py`
  - Config: `enable_profiling_metrics` (constructor)
  - `get_stats()` includes (when enabled): total/avg batch size, min/max batch, queue timings, total/avg inference time, average inference per item.

Self‑Play Engine Integration
- File: `ml/training/selfplay.py`
  - Flags (constructor):
    - `enable_worker_profiling`: enables cProfile on worker 0.
    - `enable_worker_metrics`: enables instrumentation in workers.
    - `run_id`: tag used to name output files for a run.
  - Modes covered: multiprocessing workers, threaded workers, and GPU‑server path.

Where To Start Reading
- Runner and options: `ml/profile_selfplay.py`
- Determinization metrics code: `ml/mcts/determinization.py`
- Node expansion metrics code: `ml/mcts/node.py`
- Evaluator metrics/stats: `ml/mcts/batch_evaluator.py`
- Engine wiring & worker saves: `ml/training/selfplay.py`

Notes
- Per‑worker metric files (`profile_<run_id>_worker*_metrics.json` / `profile_<run_id>_thread*_metrics.json`) may be deleted after aggregation. Re‑run with `--instrument` to regenerate if needed.
