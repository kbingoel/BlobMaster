# BlobMaster Performance Results Tracker

**Last Updated:** 2025-10-28
**Platform:** Ubuntu 24.04, Python 3.14.0 (GIL enabled), RTX 4060 8GB, Ryzen 9 7950X 16-core
**Status:** Fresh testing environment - systematic benchmarking needed

---

## Executive Summary

### Current Performance Status

| Metric | Windows Baseline | Ubuntu Current | Status |
|--------|-----------------|----------------|---------|
| **Platform** | Windows 10, Python 3.12 | Ubuntu 24.04, Python 3.14 | Platform migrated |
| **Games/Min (Medium MCTS)** | 43.3 | ~20 | 🔴 **2.2x slower** |
| **Training Time (500 iter)** | 7.1 days | ~120 days | 🔴 **17x longer** |
| **Root Cause** | N/A | Unknown | 🔍 **Investigation needed** |

### Critical Issue

**Performance regression detected:** Ubuntu/Python 3.14 is running 2.2x slower than Windows/Python 3.12 baseline for self-play game generation. This makes the 500-iteration training run infeasible (~120 days vs target 7-12 days).

**Next Steps:**
1. Run systematic baseline benchmarks on Ubuntu
2. Compare with archived Windows results
3. Identify root cause of performance regression
4. Test optimization strategies once baseline is established

---

## Platform Context

### Migration History

**Windows Era (Phase 1-4 Development)**
- **Platform:** Windows 10
- **Python:** 3.12.x
- **Hardware:** Same (RTX 4060 + Ryzen 9 7950X)
- **Period:** Project inception → 2025-10-27
- **Status:** All Phase 1-4 testing completed on Windows

**Ubuntu Migration (Current)**
- **Platform:** Ubuntu 24.04 LTS
- **Python:** 3.14.0 (GIL enabled, option for future GIL-disabled testing)
- **Hardware:** Same (RTX 4060 + Ryzen 9 7950X)
- **Migration Date:** 2025-10-27
- **Rationale:** Better performance tooling, GIL-disabled option, Linux ecosystem

### Why Ubuntu?

1. **GIL-disabled option:** Python 3.14+ supports `--disable-gil` builds for potential threading speedups
2. **Better profiling:** Linux tools (perf, flamegraph) superior to Windows
3. **Native environment:** Production deployment likely on Linux
4. **Optimization potential:** Access to different pipeline techniques

### Why Python 3.14?

1. **GIL-disabled experiments:** Can test threading optimizations without GIL
2. **Latest features:** Performance improvements in CPython 3.14
3. **Future-proofing:** Stay current with Python ecosystem

---

## Best Known Configuration

### Windows Baseline (Validated)

```python
config = TrainingConfig(
    # Self-play configuration
    num_workers=32,              # Optimal for RTX 4060 + Ryzen 9 7950X
    device="cuda",               # GPU for neural network inference

    # MCTS configuration (Medium)
    num_determinizations=3,      # Sample 3 possible opponent hand distributions
    simulations_per_determinization=30,  # 30 MCTS simulations per world

    # Training configuration
    batch_size=512,              # GPU training batch size
    learning_rate=0.001,         # Adam optimizer learning rate

    # Optimization flags
    use_batched_evaluator=False, # No multi-game batching (Phase 2 failed)
    use_thread_pool=False,       # Multiprocessing only (Phase 3 failed)
    use_gpu_batched_mcts=False,  # Not yet implemented
)
```

### Performance Baselines (Windows)

| MCTS Config | Det | Sims | Games/Min | Sec/Game | Training Time (500 iter) |
|-------------|-----|------|-----------|----------|--------------------------|
| Light | 2 | 20 | 80.8 | 0.74 | 4.2 days |
| **Medium** | **3** | **30** | **43.3** | **1.39** | **7.1 days** |
| Heavy | 5 | 50 | 25.0 | 2.40 | 12.1 days |

**Recommended:** Medium MCTS (3×30) for balance of quality and speed

### Ubuntu Current (Needs Validation)

**Status:** Only preliminary measurements available
- **Medium MCTS:** ~20 games/min (vs 43.3 expected)
- **Training projection:** ~120 days
- **Confidence:** Low - needs systematic benchmarking

---

## Performance Testing History

### Phase 1-4: Windows Development Era

#### Phase 1: Core Optimizations (Windows)
**Date:** 2025-10-26
**Platform:** Windows 10, Python 3.12

**Tests Conducted:**
1. ✅ Baseline multiprocessing (32 workers)
2. ✅ Intra-game GPU batching (Phase 1)
3. ✅ Multi-game batching (Phase 2)
4. ✅ Threading experiments (Phase 3)
5. ✅ GPU server architecture (Phase 3.5)

**Key Results:**
- **Best:** 32 workers, multiprocessing, no batching = 43.3 games/min (Medium MCTS)
- **Phase 1 (Intra-game batching):** 1.76x speedup (below 2x target)
- **Phase 2 (Multi-game batching):** Failed - avg batch size 3.5, overhead > benefit
- **Phase 3 (Threading):** Failed - GIL contention, 8.5x slower
- **Phase 3.5 (GPU server):** Failed - queue overhead, 3-5x slower
- **60+ workers:** Failed - GPU thrashing, 52x slower

**Decision:** Use 32 workers, multiprocessing baseline for training

**Documentation:** See [../BENCHMARK-PLAN.md](../BENCHMARK-PLAN.md) for detailed execution results

---

#### Phase 2-4: Training Infrastructure (Windows)
**Period:** 2025-10-15 → 2025-10-27

**Focus:** Build training pipeline (not performance optimization)
- Implemented self-play engine
- Built replay buffer
- Created training loop
- Added evaluation arena with ELO tracking
- Validated all 426+ tests passing

**Performance Testing:**
- Minimal - focused on correctness
- Confirmed 32 workers baseline still optimal
- No new optimization attempts

---

### Ubuntu Era: Fresh Start

#### Platform Migration
**Date:** 2025-10-27
**Status:** In progress

**Completed:**
1. ✅ Platform migrated to Ubuntu 24.04
2. ✅ Python 3.14.0 installed (GIL enabled)
3. ✅ Virtual environment recreated
4. ✅ All dependencies installed (PyTorch 2.x + CUDA 12.4)
5. ✅ All 426+ tests passing

**Pending:**
1. 🔲 Systematic baseline benchmarks
2. 🔲 Performance regression investigation
3. 🔲 GIL-disabled testing (future)
4. 🔲 New optimization strategies

---

## Test Results Log

### 2025-10-28: Initial Ubuntu Observations

**Test:** Quick self-play check (informal)
**Config:** 32 workers, Medium MCTS (3×30)
**Results:** ~20 games/min
**Expected:** 43.3 games/min (Windows baseline)
**Status:** 🔴 **Performance regression detected**

**Observations:**
- 2.2x slower than Windows baseline
- GPU utilization: 15-20% (similar to Windows)
- CPU utilization: Unknown (needs measurement)

**Next Steps:**
- Run systematic benchmarks via `benchmark_selfplay.py`
- Compare GPU%, CPU%, memory usage vs Windows
- Profile code to identify bottlenecks

---

### Archived Windows Results

All Windows-era results moved to [archive/windows-era/](archive/windows-era/)

**Key Files:**
- `baseline_results.csv` - Phase 2 baseline (32 workers, no batching)
- `phase1_validation.csv` - Intra-game batching sweep
- `gpu_batch_mcts_benchmark.csv` - Most recent GPU-batched MCTS test
- `gpu_server_results.csv` - GPU server failure analysis

**File Organization:** Sorted chronologically by creation date to show natural progression of testing

---

## Optimization Attempts Summary

### Phase 1: Intra-Game GPU Batching ⚠️ Partial Success

**Concept:** Batch neural network calls within a single MCTS search
**Implementation:** Virtual loss + pending evaluations queue
**Results:** 1.76x speedup (below 2x target)
**Status:** Insufficient for production use

**Pros:**
- Easy to implement
- No multiprocessing complexity
- Measurable speedup

**Cons:**
- Small batch sizes (avg 5-10 nodes)
- Virtual loss reduces MCTS quality
- Below target speedup

**Conclusion:** Not worth trade-offs, but validates batching concept

---

### Phase 2: Multi-Game Batching ❌ Failed

**Concept:** Batch neural network calls across multiple concurrent games
**Implementation:** Per-worker `BatchedEvaluator` with queue
**Results:** Avg batch size 3.5 → overhead > benefit
**Status:** Failed

**Why it failed:**
- Games out of phase (different MCTS depths)
- Small batch sizes (3-4 nodes average)
- Queue overhead
- Serialization costs

**Conclusion:** Multiprocessing games inherently out of sync, can't achieve good batch sizes

---

### Phase 3: Threading + Shared Batching ❌ Failed

**Concept:** Use threads (not processes) to share BatchedEvaluator
**Implementation:** ThreadPoolExecutor with shared queue
**Results:** 8.5x slower than multiprocessing
**Status:** Failed catastrophically

**Why it failed:**
- **GIL contention:** Python GIL destroys parallelism
- Threads competing for single GIL
- MCTS CPU-bound → terrible fit for threading
- Queue overhead still present

**Conclusion:** Python threading unsuitable for CPU-bound MCTS (unless GIL disabled)

---

### Phase 3.5: GPU Server Architecture ❌ Failed

**Concept:** Centralized GPU server process, workers send requests via queue
**Implementation:** Manager process + multiprocessing queue
**Results:** 3-5x slower than baseline
**Status:** Failed

**Why it failed:**
- Queue serialization overhead
- IPC costs (inter-process communication)
- Network tensor overhead
- Server becomes bottleneck

**Conclusion:** Queue-based architectures too slow for high-frequency inference

---

### Phase 4: 60+ Workers ❌ Failed

**Concept:** More workers → more parallelism
**Implementation:** 60, 128, 256 workers tested
**Results:** 52x slower at 60 workers
**Status:** Failed catastrophically

**Why it failed:**
- **GPU thrashing:** Too many processes competing for GPU
- Context switching overhead
- Memory contention
- Diminishing returns past 32 workers

**Conclusion:** 32 workers optimal for RTX 4060, more workers counterproductive

---

### Future: GPU-Batched MCTS 🔜 Not Yet Implemented

**Concept:** Parallel MCTS expansion with GPU batching
**Implementation:** Expand multiple nodes simultaneously, batch GPU calls
**Expected:** 5-10x speedup (based on AlphaZero literature)
**Status:** Planned (not implemented)

**Requirements:**
1. Synchronous tree expansion
2. Batch all leaf evaluations in single GPU call
3. Parallel backpropagation
4. Virtual loss for exploration

**Challenges:**
- Complex implementation
- Needs synchronization across workers
- May require tree-level parallelism (not game-level)

**Priority:** High - only remaining promising approach

---

## Performance Investigation Plan

### Ubuntu Baseline Validation (PRIORITY 1)

**Goal:** Establish Ubuntu/Python 3.14 baseline performance

**Tests:**
1. ✅ Network architecture validation
   - Verify 4.9M parameters (not 361K)
   - Confirm correct Transformer config

2. 🔲 Self-play baseline
   - Run `benchmark_selfplay.py` (full suite)
   - Test: 1, 4, 8, 16, 32 workers
   - Test: Light, Medium, Heavy MCTS
   - Compare with Windows baselines

3. 🔲 GPU training baseline
   - Run `benchmark_training.py`
   - Test batch sizes: 256, 512, 1024, 2048
   - Measure: examples/sec, GPU%, VRAM

4. 🔲 Full iteration baseline
   - Run `benchmark_iteration.py`
   - Measure complete iteration timing
   - Compare phase breakdown with Windows

**Expected Duration:** 2-4 hours
**Output:** CSV files in `ubuntu-2025-10/`

---

### Root Cause Analysis (PRIORITY 2)

**Goal:** Identify why Ubuntu is 2.2x slower

**Hypotheses:**
1. **Python 3.14 regression:** New version slower for some operations?
2. **PyTorch/CUDA differences:** Different library versions?
3. **OS scheduling:** Linux process scheduling different from Windows?
4. **Memory/cache:** Different memory management on Linux?
5. **Thermal throttling:** Different power management?

**Investigation Steps:**
1. 🔲 Profile code with `cProfile` + `snakeviz`
2. 🔲 Compare PyTorch versions (Windows vs Ubuntu)
3. 🔲 Monitor GPU clocks with `nvidia-smi`
4. 🔲 Check CPU frequency scaling
5. 🔲 Run diagnostic benchmark suite
6. 🔲 Compare system resource usage

**Expected Duration:** 3-6 hours
**Output:** Performance profile analysis

---

### Optimization Strategy (PRIORITY 3)

**After baseline established:**

1. 🔲 **GIL-disabled testing** (if root cause is GIL)
   - Rebuild Python 3.14 with `--disable-gil`
   - Re-test threading approaches (Phase 3)
   - Expected: 2-4x speedup if GIL is bottleneck

2. 🔲 **GPU-batched MCTS** (if baseline acceptable)
   - Implement parallel tree expansion
   - Batch all leaf evaluations
   - Expected: 5-10x speedup

3. 🔲 **Profiling-driven optimization**
   - Identify hot spots from profiler
   - Optimize critical paths
   - Expected: 1.2-1.5x speedup

**Goal:** Achieve <10 day training time for 500 iterations

---

## Metrics Tracking

### Key Performance Indicators (KPIs)

**Primary Metric:** Games per minute (Medium MCTS)
- **Target:** 40+ games/min
- **Windows Baseline:** 43.3 games/min
- **Ubuntu Current:** ~20 games/min
- **Status:** 🔴 Below target

**Secondary Metrics:**
- **Training Time (500 iter):** Target <10 days
- **GPU Utilization:** Target >80% during training, 15-20% during self-play
- **CPU Utilization:** Target 70-90%
- **Memory Usage:** <6GB GPU VRAM, <64GB system RAM

### Benchmark Result Format

**Self-Play Results** (`benchmark_selfplay.py` output):
```csv
workers,determinizations,simulations,games_per_min,avg_game_duration_sec,cpu_percent,training_examples_per_min
32,3,30,43.3,1.39,85.2,12450
```

**Training Results** (`benchmark_training.py` output):
```json
{
  "batch_size": 512,
  "device": "cuda",
  "examples_per_sec": 15234,
  "gpu_percent": 92.4,
  "vram_gb": 4.2
}
```

**Iteration Results** (`benchmark_iteration.py` output):
```json
{
  "self_play_time_sec": 7842,
  "training_time_sec": 1235,
  "eval_time_sec": 342,
  "total_time_sec": 9419,
  "projected_days": 7.2
}
```

---

## Documentation References

### Primary Documentation
- **[../README.md](../README.md)** - Complete benchmark suite guide
- **[../BENCHMARK-PLAN.md](../BENCHMARK-PLAN.md)** - Historical test plan (Windows era)
- **[docs/findings/PERFORMANCE-FINDINGS.md](../docs/findings/PERFORMANCE-FINDINGS.md)** - Comprehensive analysis

### Archived Documentation
- **[archive/windows-era/](archive/windows-era/)** - All Windows-era results
- **[docs/archive/](../docs/archive/)** - Superseded planning docs

### Root Project Documentation
- **[../../CLAUDE.md](../../CLAUDE.md)** - Project overview & development guide
- **[../../README.md](../../README.md)** - BlobMaster main README
- **[../../docs/performance/](../../docs/performance/)** - Original performance findings

---

## Quick Reference Commands

### Run Baseline Benchmarks
```bash
# Activate environment
source venv/bin/activate

# Quick check (5 min)
python benchmarks/performance/benchmark_selfplay.py --quick

# Full self-play suite (30-60 min)
python benchmarks/performance/benchmark_selfplay.py

# GPU training benchmark (15-30 min)
python benchmarks/performance/benchmark_training.py

# Full iteration test (1-2 hours)
python benchmarks/performance/benchmark_iteration.py
```

### Run Diagnostics
```bash
# Quick diagnostic (30 min)
python benchmarks/performance/benchmark_diagnostic.py --quick

# Full diagnostic (2-3 hours)
python benchmarks/performance/benchmark_diagnostic.py

# Generate report
python benchmarks/performance/benchmark_report.py
```

### Validation Tests
```bash
# Verify network size
python benchmarks/tests/test_network_size.py

# Reproduce baseline
python benchmarks/tests/test_reproduce_baseline.py
```

---

## Action Items

### Immediate (This Week)
1. ✅ Organize benchmark directory structure
2. ✅ Archive Windows-era results
3. ✅ Create centralized tracking documentation
4. 🔲 Run full baseline benchmark suite on Ubuntu
5. 🔲 Identify root cause of 2.2x performance regression

### Short-term (Next 2 Weeks)
1. 🔲 Resolve performance regression
2. 🔲 Validate all benchmarks reproduce expected results
3. 🔲 Document Ubuntu baseline performance
4. 🔲 Decision: Proceed with training or optimize further?

### Long-term (Next Month)
1. 🔲 Implement GPU-batched MCTS (if needed)
2. 🔲 Test GIL-disabled Python builds (if helpful)
3. 🔲 Optimize critical paths based on profiling
4. 🔲 Achieve <10 day training time target

---

## Notes & Observations

### 2025-10-28: Migration Complete

- Platform successfully migrated to Ubuntu 24.04
- Python 3.14.0 installed with GIL enabled
- All 426+ tests passing
- Virtual environment recreated
- Dependencies installed (PyTorch + CUDA 12.4)
- Benchmark directory reorganized for clarity
- Ready to begin systematic testing

**Next:** Run baseline benchmarks to establish Ubuntu performance characteristics

---

## Changelog

### 2025-10-28
- Created centralized RESULTS-TRACKER.md
- Reorganized benchmark directory structure
- Archived all Windows-era results
- Created comprehensive README.md for benchmark suite
- Documented performance regression issue
- Established systematic testing plan

---

*For detailed benchmark usage instructions, see [../README.md](../README.md)*
