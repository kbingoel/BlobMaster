# Benchmarks

This directory contains performance benchmarking and testing scripts.

## Directory Structure

### [performance/](performance/)
Reusable benchmark scripts for measuring self-play, training, and iteration performance.

**Scripts**:
- `benchmark_selfplay.py` - Self-play game generation benchmarks
- `benchmark_training.py` - Neural network training benchmarks
- `benchmark_iteration.py` - Full training iteration benchmarks
- `benchmark_phase2.py` - Phase 2 (multiprocessing) benchmarks
- `benchmark_phase3.py` - Phase 3 (threading + batching) benchmarks
- `benchmark_diagnostic.py` - Comprehensive diagnostic benchmark suite
- `benchmark_report.py` - Benchmark result analysis and reporting

### [tests/](tests/)
Ad-hoc test scripts for configuration validation.

**Scripts**:
- `test_action_plan.py` - Action plan configuration tests
- `test_action_plan_windows.py` - Windows-specific configuration tests
- `test_batched_phase1.py` - Initial batching tests
- `test_diagnostic_quick.py` - Quick diagnostic tests
- `test_gpu_server.py` - GPU inference server tests

## Usage

### Running Performance Benchmarks

**Self-play benchmark**:
```bash
python benchmarks/performance/benchmark_selfplay.py --workers 32 --games 50
```

**Training benchmark**:
```bash
python benchmarks/performance/benchmark_training.py --batch-size 2048 --device cuda
```

**Full iteration benchmark**:
```bash
python benchmarks/performance/benchmark_iteration.py --workers 32
```

**Comprehensive diagnostics**:
```bash
python benchmarks/performance/benchmark_diagnostic.py --full
```

### Running Tests

**Quick diagnostic**:
```bash
python benchmarks/tests/test_diagnostic_quick.py
```

**GPU server test**:
```bash
python benchmarks/tests/test_gpu_server.py
```

## Results

Benchmark results are saved to [../results/](../results/) as CSV files.

## Performance Findings

See [../PERFORMANCE-FINDINGS.md](../PERFORMANCE-FINDINGS.md) for a summary of benchmark findings and recommendations.

Detailed analysis: [../docs/performance/](../docs/performance/)
