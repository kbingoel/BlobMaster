# BlobNet Diagnostic Session Summary

**Date**: 2025-10-26
**Duration**: ~2-3 hours
**Status**: Diagnostic infrastructure complete, validation in progress

---

## Problem Addressed

You halted BlobNet development after Phase 4 Session 5 due to **suspicious performance metrics**:

### The Concern
- Training benchmarks showed **196 days** estimated training time
- Both **CPU and GPU were barely utilized** (mostly idle)
- **5M parameter model** seemed tiny for the available hardware (RTX 4060)
- Numbers "didn't add up" - something felt fundamentally wrong

### Your Hypothesis
The original plan to develop a parameter sweep benchmark wasn't working out because:
1. No actual data on what parameters to sweep
2. Training speeds were ridiculously slow
3. Both CPU and GPU idle suggested **architectural problem**, not parameter tuning issue

---

## Root Cause Analysis

Through reviewing your findings documents, we identified:

### The Core Issue: **GPU Starvation**

**Math**:
- RTX 4060 has 3,072 CUDA cores
- Needs batch sizes of **128-512** for 70%+ utilization
- Current setup: **4 workers** generating ~8-16 concurrent requests
- Actual batch sizes achieved: **3.5** (way too small!)

**Why Both CPU and GPU Were Idle**:
- **Not enough parallel work** to saturate GPU (too few workers)
- **Synchronization overhead** in BatchedEvaluator exceeded benefit at low worker counts
- **Sequential MCTS** within each game limits concurrency
- Result: Pipeline is **bottlenecked on coordination**, not computation

### Key Findings from Previous Benchmarks

**Phase 2** (multiprocessing, 4 workers):
- 14.8 games/min
- BatchedEvaluator showed 0 requests (not working!)
- Low but stable performance

**Phase 3** (threading, 4 workers):
- 8.9 games/min (**40% SLOWER**)
- Batch size: 3.5
- **Synchronization overhead** exceeded batching benefit

**Conclusion**: Need **32-128 workers** to saturate GPU, not 4!

---

## Solution Approach

### Our Strategy: **Diagnose, Don't Guess**

Instead of blindly trying different configurations, we built **comprehensive diagnostic infrastructure**:

1. **Systematic benchmarking** across worker scales (4-128)
2. **GPU utilization monitoring** (nvidia-smi integration)
3. **Code profiling** to find actual bottlenecks
4. **Visualization** for easy analysis
5. **Decision framework** based on data, not assumptions

---

## What We Built

### 1. Core Diagnostic Scripts

#### [`ml/benchmark_diagnostic.py`](ml/benchmark_diagnostic.py) (~500 lines)
**Comprehensive benchmark suite**

**Features**:
- Tests worker counts: 4, 8, 16, 32, 64, 128
- Compares batching strategies (BatchedEvaluator vs direct)
- Tests parallelism types (ThreadPoolExecutor vs multiprocessing)
- Varies game complexity (3, 5, 8 cards)
- GPU monitoring via nvidia-smi
- Saves results to CSV + JSON

**Metrics tracked**:
- Games/min throughput
- GPU utilization (avg & max)
- GPU memory usage
- Batch sizes
- Examples per game
- CPU usage

**Output**:
- `benchmark_diagnostic_results.csv`
- `benchmark_diagnostic_results.json`
- Console summary with recommendations

**Duration**: ~30-40 minutes

#### [`ml/visualize_diagnostic.py`](ml/visualize_diagnostic.py) (~300 lines)
**Result visualization and analysis**

**Creates**:
- Worker scaling plots (games/min, GPU%, batch size, efficiency)
- Batching comparison charts
- Game complexity analysis
- Summary statistics table

**Output**:
- `benchmark_worker_scaling.png`
- `benchmark_batching_comparison.png`
- `benchmark_game_complexity.png`
- Console analysis with actionable recommendations

**Requires**: matplotlib, seaborn, pandas

#### [`ml/profile_selfplay.py`](ml/profile_selfplay.py) (~350 lines)
**Code-level profiling**

**Approaches**:
1. **cProfile**: Function-level profiling
2. **Manual timing**: Component breakdown
3. **Overhead analysis**: BatchedEvaluator synchronization cost

**Output**:
- `.prof` files (viewable with pstats)
- Top functions by cumulative time
- Top functions by total time
- Overhead measurements

**Duration**: ~10 minutes

#### [`ml/run_full_diagnostics.py`](ml/run_full_diagnostics.py) (~150 lines)
**Master orchestration script**

**Workflow**:
1. Runs profiling (~10 min)
2. Asks user for confirmation
3. Runs comprehensive benchmark (~30 min)
4. Generates visualizations
5. Provides summary and next steps

**Duration**: ~40-50 minutes total

#### [`ml/test_diagnostic_quick.py`](ml/test_diagnostic_quick.py) (~150 lines)
**Quick validation test**

**Purpose**: Verify diagnostic tools work before committing to full run

**Tests**:
- GPU monitoring functionality
- 4 workers configuration
- 16 workers configuration
- Scaling validation

**Duration**: ~5-10 minutes

### 2. Supporting Infrastructure

#### [`GPUMonitor` class](ml/benchmark_diagnostic.py)
- Background thread monitoring GPU stats
- Samples every 0.5 seconds using nvidia-smi
- Tracks utilization % and memory usage
- Returns avg/max statistics

#### Data structures
- `BenchmarkResult` dataclass: All metrics for a config
- CSV/JSON serialization for easy analysis
- Compatible with pandas for visualization

### 3. Documentation

#### [DIAGNOSTIC-README.md](DIAGNOSTIC-README.md)
**Complete usage guide**
- Tool descriptions
- Usage instructions
- Expected findings
- Decision framework
- Troubleshooting

#### [NEXT-STEPS.md](NEXT-STEPS.md)
**Action plan**
- Step-by-step guide
- Decision tree
- Expected scenarios
- Integration with Phase 4

#### [SESSION-SUMMARY.md](SESSION-SUMMARY.md)
**This document**
- What we built
- Why we built it
- How to use it
- What to expect

---

## Dependencies Installed

```bash
pip install matplotlib seaborn pandas
```

All installed in your venv successfully.

---

## Current Status

### ✅ Completed
- [x] Comprehensive benchmark script
- [x] GPU monitoring integration
- [x] Visualization tools
- [x] Profiling script
- [x] Master runner script
- [x] Complete documentation
- [x] Dependencies installed
- [x] Unicode encoding issues fixed

### ⏳ In Progress
- [ ] Validation test running
- [ ] Waiting for initial results

### 📋 Next Steps
1. Review validation test results
2. Run full diagnostic suite
3. Analyze benchmark data
4. Identify optimal configuration
5. Implement architectural fix
6. Validate improvements
7. Continue Phase 4 with confidence

---

## How to Use the Diagnostic Suite

### Quick Start (Recommended First)

```bash
# Run validation test (5-10 min)
venv/Scripts/python.exe ml/test_diagnostic_quick.py
```

**What it does**:
- Tests 2 configurations (4 and 16 workers)
- Verifies tools work correctly
- Shows if scaling helps
- Quick sanity check

### Full Diagnostic Suite

```bash
# Run everything (40-50 min)
venv/Scripts/python.exe ml/run_full_diagnostics.py
```

**What it does**:
1. Profiling analysis
2. Comprehensive benchmark
3. Visualization
4. Recommendations

### Individual Tools

```bash
# Just benchmarking (30-40 min)
venv/Scripts/python.exe ml/benchmark_diagnostic.py

# Just visualization (requires benchmark data)
venv/Scripts/python.exe ml/visualize_diagnostic.py

# Just profiling (10 min)
venv/Scripts/python.exe ml/profile_selfplay.py
```

---

## Expected Findings

Based on analysis in [FINDINGS-GPU-IDLE.md](FINDINGS-GPU-IDLE.md):

### Most Likely Scenario

**Finding**: Need 32-128 workers for GPU saturation

**Evidence**:
- 4 workers → 15-20 games/min, <10% GPU, batch 3-5
- 16 workers → ~50-100 games/min, ~20-30% GPU, batch ~10-20
- 32 workers → ~150-300 games/min, ~40-60% GPU, batch ~50-100
- **64 workers** → ~300-600 games/min, ~70-90% GPU, batch ~128-256
- 128 workers → ~500-1000 games/min, >90% GPU, batch ~256-512

**Solution**: Scale to 64-128 workers with ThreadPoolExecutor

**Expected result**:
- Training time: **15-30 days** (vs 196 days)
- GPU utilization: **70-90%** (vs 5-10%)
- Games/min: **500-1000** (vs 15-19)

### Alternative Scenarios

**Scenario A**: Batching overhead dominates
- Remove BatchedEvaluator at low worker counts
- Use direct inference
- Scale workers with multiprocessing

**Scenario B**: Unexpected bottleneck
- Profiling shows non-network code dominating
- Optimize specific component
- Then scale workers

---

## Success Criteria

After diagnostics complete:

### Understanding
- [ ] Know exact bottleneck (with data, not guesses)
- [ ] Understand worker scaling behavior
- [ ] Know if batching helps or hurts at different scales
- [ ] Have realistic training timeline

### Performance
- [ ] >500 games/min throughput
- [ ] >70% GPU utilization
- [ ] Avg batch size >128 (if batching)
- [ ] <30 day training estimate

### Implementation
- [ ] Optimal configuration identified
- [ ] Fix implemented
- [ ] Improvements validated
- [ ] Ready for Phase 4 continuation

---

## Key Insights

### Why This Approach Works

1. **Data-driven**: No more guessing, measure everything
2. **Systematic**: Test hypotheses methodically
3. **Comprehensive**: Cover all variables (workers, batching, complexity)
4. **Visual**: Easy to understand with charts
5. **Actionable**: Clear recommendations based on findings

### Why Previous Approaches Struggled

1. **Too few data points**: Only tested 4 and 16 workers
2. **No GPU monitoring**: Assumed GPU was being used
3. **No visualization**: Hard to see trends in raw numbers
4. **No profiling**: Guessed where time was spent
5. **Incomplete testing**: Didn't vary enough parameters

### What Makes This Different

- **10+ configurations tested** (not just 2-3)
- **Real GPU monitoring** (nvidia-smi integration)
- **Visual analysis** (charts show trends clearly)
- **Code profiling** (actual bottleneck identification)
- **Full parameter sweep** (workers, batching, complexity)

---

## Files Created This Session

### Scripts
- `ml/benchmark_diagnostic.py` - Main benchmark suite
- `ml/visualize_diagnostic.py` - Visualization tool
- `ml/profile_selfplay.py` - Profiling tool
- `ml/run_full_diagnostics.py` - Master script
- `ml/test_diagnostic_quick.py` - Quick validation

### Documentation
- `DIAGNOSTIC-README.md` - Detailed usage guide
- `NEXT-STEPS.md` - Action plan
- `SESSION-SUMMARY.md` - This file

### Output Files (Generated When Run)
- `benchmark_diagnostic_results.csv` - Raw data
- `benchmark_diagnostic_results.json` - Detailed data
- `benchmark_worker_scaling.png` - Scaling plots
- `benchmark_batching_comparison.png` - Batching analysis
- `benchmark_game_complexity.png` - Complexity analysis
- `profile_*.prof` - Profiling data

---

## Timeline Estimate

| Phase | Task | Duration |
|-------|------|----------|
| ✅ Done | Build diagnostic infrastructure | 2-3 hours |
| ⏳ Now | Validation test | 5-10 min |
| 📋 Next | Full diagnostic suite | 40-50 min |
| 📋 Next | Analysis and decision | 15-30 min |
| 📋 Next | Implement fix | 2-4 hours |
| 📋 Next | Validate improvement | 10-20 min |
| **Total** | **Diagnosis → Fix → Validate** | **4-6 hours** |

---

## Integration with BlobNet Roadmap

### Current Phase: Phase 4 Session 5 (Halted)

You stopped here because performance was suspicious.

### After Diagnostics

**Phase 4 Sessions Remaining**:
- Session 6: Evaluation & ELO tracking
- Session 7: Main training script & configuration

**Updates Based on Findings**:
- Training config with optimal worker count
- Realistic timeline estimates
- Hardware utilization targets
- Hyperparameter ranges based on measured performance

**Phase 5**: ONNX Export (unchanged)

**Phase 6**: Backend API (unchanged)

**Phase 7**: Frontend UI (unchanged)

---

## Lessons Learned

### What Went Right

1. **Stopped to question assumptions** - Didn't blindly continue
2. **Systematic diagnosis** - Built tools instead of guessing
3. **Data-driven approach** - Measure, don't assume
4. **Comprehensive testing** - Test many configurations
5. **Good documentation** - Can reproduce and understand later

### What This Fixes

1. **No more guessing** about optimal parameters
2. **No more assuming** GPU is being used
3. **No more wondering** why it's slow
4. **Clear path forward** based on data
5. **Confidence** in training timeline estimates

### Why This Matters

- **196 days → 15-30 days**: Difference between practical and impractical
- **5% GPU → 70-90% GPU**: Actually using available hardware
- **Understanding → Confidence**: Know what's happening, not hoping

---

## What to Expect Next

### Validation Test Results (Any Minute)

Should show:
- 4 workers: ~15-20 games/min, <10% GPU
- 16 workers: ~30-60 games/min, ~15-25% GPU
- Speedup: ~2-3x
- Conclusion: Need more workers

### Full Benchmark Results (~40 min)

Will show:
- Complete worker scaling curve
- Optimal worker count for your hardware
- Whether batching helps
- Realistic training time
- Specific bottlenecks

### Decision (After Benchmark)

Based on data, will recommend:
- Exact worker count to use
- Whether to use batching
- Threading vs multiprocessing
- Any code optimizations needed
- Realistic training timeline

---

## Questions Answered

### Before This Session

❓ Why is training so slow?
❓ Why are both CPU and GPU idle?
❓ Is something fundamentally broken?
❓ What parameters should I use?
❓ How long will training actually take?

### After This Session

✅ Will know exact bottleneck (data-driven)
✅ Will understand hardware utilization
✅ Will have optimal configuration
✅ Will have realistic timeline
✅ Will have tools to verify future changes

---

## Next Action

**Immediate**: Wait for validation test to complete (~5 min)

**Then**:
1. Review validation results
2. Decide: Run full suite now or later?
3. If now: `venv/Scripts/python.exe ml/run_full_diagnostics.py`
4. If later: Schedule 40-50 min block, then run

**After Diagnostics**:
1. Analyze charts and recommendations
2. Implement suggested fix
3. Validate improvements
4. Continue Phase 4 with confidence

---

**Status**: Diagnostic infrastructure complete ✅
**Next**: Validation test results → Full diagnostic suite → Data-driven fix

**Estimated time to fix**: 4-6 hours total (diagnosis + implementation + validation)

**Expected outcome**: 15-30 day training time, >70% GPU utilization, >500 games/min
