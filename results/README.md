# Benchmark Results

This directory contains CSV result files from performance benchmarks and tests.

## Files

### Self-Play Results

**[benchmark_selfplay_results.csv](benchmark_selfplay_results.csv)**
- Self-play game generation performance measurements
- Columns: workers, games_per_min, avg_game_time, etc.

**[baseline_results.csv](baseline_results.csv)**
- Baseline (Phase 2) multiprocessing performance
- Different MCTS configurations (light, medium, heavy)

**[gpu_server_results.csv](gpu_server_results.csv)**
- GPU inference server performance
- Comparison with baseline approach

**[gpu_server_test.csv](gpu_server_test.csv)**
- GPU server correctness and performance validation

### Configuration Testing

**[action_plan_windows_results.csv](action_plan_windows_results.csv)**
- Windows-specific configuration test results
- Multiple worker counts and parallelism approaches

## Key Findings

**Best Configuration**: 32 workers, multiprocessing, no batching
- Light MCTS: 80.8 games/min
- Medium MCTS: 43.3 games/min
- Heavy MCTS: 25.0 games/min

See [../PERFORMANCE-FINDINGS.md](../PERFORMANCE-FINDINGS.md) for detailed analysis.

## Generating New Results

Use the benchmark scripts in [../benchmarks/](../benchmarks/) to generate new result files:

```bash
# Example: benchmark self-play performance
python benchmarks/performance/benchmark_selfplay.py --output results/my_test.csv
```

Results are automatically timestamped and include system information for reproducibility.
