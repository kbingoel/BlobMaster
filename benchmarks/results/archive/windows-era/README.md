# Windows Era Benchmark Results

**Platform:** Windows 10, Python 3.12.x
**Period:** Project inception → 2025-10-27
**Status:** Historical - archived for reference

---

## Overview

This directory contains all benchmark results from the Windows development era (Phase 1-4). The project has since migrated to Ubuntu 24.04 with Python 3.14, requiring fresh baseline measurements.

**Migration Date:** 2025-10-27
**Reason for Migration:** Better Linux tooling, GIL-disabled option in Python 3.14, improved profiling capabilities

---

## File Organization

Files are organized chronologically by creation date to show the natural progression of benchmark testing.

### Key Result Files

**Baseline Performance:**
- `baseline_results.csv` - Phase 2 baseline (32 workers, no batching)
- `baseline_reproduction.csv` - Baseline validation run
- `baseline_check.csv` - Quick baseline verification

**Self-Play Benchmarks:**
- `benchmark_selfplay_results.csv` - Comprehensive self-play performance measurements
- `selfplay_quick_screen.csv` - Quick 5-game screening tests

**Phase Validation:**
- `phase1_validation.csv` - Full Phase 1 intra-game batching validation
- `phase1_validation_quick.csv` - Quick Phase 1 batch size sweep
- `phase1_quick_baseline.csv` - Phase 1 baseline measurement
- `phase1_quick_sweep.csv` - Phase 1 batch sizes 30, 60, 90

**GPU Testing:**
- `gpu_batch_mcts_benchmark.csv` - GPU-batched MCTS results (most recent test)
- `gpu_server_results.csv` - GPU server architecture testing (Phase 3.5)
- `gpu_server_test.csv` - GPU server validation

**Configuration Testing:**
- `action_plan_windows_results.csv` - Windows configuration validation (32 workers optimal)

---

## Best Known Performance (Windows)

### Configuration

```python
config = TrainingConfig(
    num_workers=32,              # Optimal for RTX 4060 + Ryzen 9
    device="cuda",
    num_determinizations=3,      # Medium MCTS
    simulations_per_determinization=30,
    batch_size=512,
    use_batched_evaluator=False, # No batching
    use_thread_pool=False,       # Multiprocessing only
)
```

### Performance

| MCTS Config | Games/Min | Training Time (500 iter) |
|-------------|-----------|--------------------------|
| Light (2×20) | 80.8 | 4.2 days |
| **Medium (3×30)** | **43.3** | **7.1 days** |
| Heavy (5×50) | 25.0 | 12.1 days |

---

## Test Progression

### Phase 1: Baseline Establishment
- Validated network architecture (4.9M parameters)
- Measured baseline performance: 43.3 games/min (Medium MCTS, 32 workers)
- Identified 32 workers as optimal (60+ workers: 52x slower due to GPU thrashing)

### Phase 2: Intra-Game Batching
- Implemented GPU-batched MCTS with virtual loss
- Results: 1.76x speedup (below 2x target)
- Decision: Insufficient for production use

### Phase 3: Multi-Game Batching
- Tested per-worker BatchedEvaluator
- Results: Avg batch size 3.5 → overhead > benefit
- Decision: Failed - games out of phase

### Phase 4: Threading Experiments
- Tested threading + shared BatchedEvaluator
- Results: 8.5x slower than multiprocessing (GIL contention)
- Decision: Failed catastrophically

### Phase 5: GPU Server Architecture
- Tested centralized GPU server with queue
- Results: 3-5x slower than baseline (IPC overhead)
- Decision: Failed - queue overhead too high

---

## Key Findings

### Successful Strategies
✅ **32 workers, multiprocessing** - Optimal for RTX 4060 + Ryzen 9
✅ **Medium MCTS (3×30)** - Best balance of quality and speed
✅ **No batching** - Overhead exceeds benefits for current architecture
✅ **CUDA for training** - 90%+ GPU utilization, 15k+ examples/sec

### Failed Strategies
❌ **Phase 1 (Intra-game batching)** - 1.76x speedup insufficient
❌ **Phase 2 (Multi-game batching)** - Small batch sizes (3.5 avg)
❌ **Phase 3 (Threading)** - GIL destroyed performance
❌ **Phase 3.5 (GPU server)** - Queue overhead too high
❌ **60+ workers** - GPU thrashing (52x slower)

### Root Causes
- **GIL contention:** Python threading unsuitable for CPU-bound MCTS
- **Small batch sizes:** Games out of phase → can't accumulate large batches
- **Queue overhead:** IPC serialization costs exceed benefits
- **GPU thrashing:** Too many processes competing for GPU resources

---

## Ubuntu Comparison (Current)

**Windows Baseline (Historical):**
- Medium MCTS: 43.3 games/min
- Training time: 7.1 days (500 iterations)

**Ubuntu Current (2025-10-28):**
- Medium MCTS: ~20 games/min (preliminary)
- Training time: ~120 days (projected)
- **Status:** ⚠️ Performance regression detected (2.2x slower)

**Root Cause:** Unknown - systematic benchmarking needed

---

## Documentation References

**Primary Analysis:**
- [../../BENCHMARK-PLAN.md](../../BENCHMARK-PLAN.md) - Comprehensive test plan (Phase 1 executed)
- [../../docs/findings/PERFORMANCE-FINDINGS.md](../../docs/findings/PERFORMANCE-FINDINGS.md) - Detailed findings
- [../../docs/findings/PHASE1-EXECUTION-SUMMARY.md](../../docs/findings/PHASE1-EXECUTION-SUMMARY.md) - Phase 1 results

**Current Tracking:**
- [../RESULTS-TRACKER.md](../RESULTS-TRACKER.md) - Ongoing performance tracking (Ubuntu era)
- [../../README.md](../../README.md) - Benchmark suite navigation

---

## Notes

These results are preserved for historical reference and comparison. All future benchmark results will be stored in `../ubuntu-2025-10/` with fresh baseline measurements on Ubuntu 24.04 + Python 3.14.

The performance regression observed on Ubuntu requires investigation before proceeding with further optimization or training.

---

*Last updated: 2025-10-28*
