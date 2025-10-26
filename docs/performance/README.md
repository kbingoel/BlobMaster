# Performance Investigation Documentation

This directory contains detailed reports from the self-play performance optimization investigation (October 2025).

## Executive Summary

See [../../PERFORMANCE-FINDINGS.md](../../PERFORMANCE-FINDINGS.md) for a concise summary of findings and recommendations.

## Documents Overview

### Primary Findings

**[GPU-SERVER-TEST-RESULTS.md](GPU-SERVER-TEST-RESULTS.md)**
- Investigation of centralized GPU inference server architecture (Phase 3.5)
- Result: 3-5x slower than baseline due to small batch sizes (10-13 avg vs 128+ target)
- Conclusion: Use baseline multiprocessing approach

**[TRAINING-PERFORMANCE-MASTER.md](TRAINING-PERFORMANCE-MASTER.md)**
- Comprehensive measurement history across all configurations
- Complete root cause analysis of GPU underutilization
- Performance predictions and decision criteria

**[WINDOWS-TEST-RESULTS-ANALYSIS.md](WINDOWS-TEST-RESULTS-ANALYSIS.md)**
- Results from testing 5 different configurations
- Shocking finding: 60 workers multiprocessing = 52x slower (GPU thrashing)
- Best configuration: 32 workers, multiprocessing, no batching

### Implementation Details

**[GPU-SERVER-IMPLEMENTATION.md](GPU-SERVER-IMPLEMENTATION.md)**
- Technical details of GPU inference server implementation
- Architecture: Single GPU process + multiprocessing queues
- Files created: `ml/mcts/gpu_server.py`, `ml/test_gpu_server.py`

### Analysis Documents

**[FINDINGS-GPU-IDLE.md](FINDINGS-GPU-IDLE.md)**
- Phase 2 vs Phase 3 comparison
- Why batching overhead exceeded benefits at low worker counts

**[BENCHMARK-Performance.md](BENCHMARK-Performance.md)**
- Early benchmark results before bug fixes
- Training performance: 74k examples/sec with FP16

**[BENCHMARK-Summary.md](BENCHMARK-Summary.md)**
- Summary of initial training and self-play benchmarks
- Pre- and post-bug-fix measurements

**[DIAGNOSTIC-README.md](DIAGNOSTIC-README.md)**
- Guide to using diagnostic benchmark scripts
- How to run comprehensive performance tests

### Future Plans

**[PLAN-GPUBatchedMCTS.md](PLAN-GPUBatchedMCTS.md)**
- Design for GPU-batched MCTS with virtual loss
- Expected performance: 300-600 games/min (10x improvement)
- Implementation time: 4-6 hours

**[NEW-PLAN.md](NEW-PLAN.md)**
- Detailed implementation plan for GPU inference server
- Written before testing revealed performance issues

**[NEXT-STEPS.md](NEXT-STEPS.md)**
- Action items and decision points from investigation

## Key Findings Summary

### What Works ✅
- **32 workers, multiprocessing, no batching**: 43-80 games/min (baseline)
- **GPU for training**: 74k examples/sec with mixed precision
- **Per-worker networks**: Simple and effective

### What Doesn't Work ❌
- **GPU inference server**: Too much queue overhead for small batches
- **Threading + batching**: GIL contention worse than no parallelism
- **60+ workers**: GPU becomes bottleneck, thrashing occurs
- **Small batch sizes**: Overhead > benefit

### Lessons Learned 💡
- MCTS is fundamentally sequential (hard to batch)
- Worker count has a sweet spot (32 optimal for this hardware)
- Python GIL is a real limitation for threading
- Measure, don't assume - intuitions were often wrong

## Related Files

**Benchmark Scripts**: [../../benchmarks/performance/](../../benchmarks/performance/)
**Test Results**: [../../results/](../../results/)
**Phase Reports**: [../phases/](../phases/)
