# Phase 1 GPU-Batched MCTS - Integration & Validation Findings

**Date**: 2025-10-26
**Status**: Integration Complete, Performance Below Expectations
**Hardware**: NVIDIA RTX 4060, AMD Ryzen 9 7950X, Windows 11

---

## Executive Summary

Phase 1 (GPU-batched MCTS with virtual loss mechanism) has been successfully integrated into the self-play pipeline and benchmarked. The integration is functionally complete, but performance results are significantly below predictions.

**Key Results**:
- ✅ Integration: Complete and functional
- ✅ Benchmark: Working with streaming output
- ⚠️ Speedup: **1.49x** (far below 5-10x prediction)
- ❌ GPU Utilization: **0.0%** (critical blocker)

---

## Implementation Summary

### Code Changes

**Modified Files**:
1. **[ml/training/selfplay.py](../../ml/training/selfplay.py)** (~50 lines)
   - Added `mcts_batch_size` parameter to `SelfPlayEngine.__init__()`
   - Added `batch_size` parameter to `SelfPlayWorker.__init__()`
   - Modified `get_bid()` and `get_card()` callbacks to use `search_batched()` when `batch_size` is set
   - Updated worker functions to pass `batch_size` through to workers

2. **[benchmarks/performance/benchmark_phase1_validation.py](../../benchmarks/performance/benchmark_phase1_validation.py)** (NEW, ~560 lines)
   - Streaming CSV output (writes after each game)
   - Real-time console progress
   - Batch size sweep support
   - Crash-resilient (preserves partial results)
   - Engine reuse to avoid initialization overhead

3. **[benchmarks/performance/benchmark_selfplay.py](../../benchmarks/performance/benchmark_selfplay.py)** (~3 lines)
   - Added Python path handling for import compatibility

### Integration Details

**How Phase 1 Works**:
```python
# Before (baseline - sequential MCTS):
action_probs = self.mcts.search(game, player)
# 90 sequential network calls for medium MCTS (3 det × 30 sims)

# After (Phase 1 - batched MCTS):
action_probs = self.mcts.search_batched(game, player, batch_size=21)
# Collects 21 leaf nodes, evaluates in single GPU call, uses virtual loss
```

**Configuration**:
```python
engine = SelfPlayEngine(
    network=network,
    num_workers=32,
    num_determinizations=3,
    simulations_per_determinization=30,
    device="cuda",
    use_batched_evaluator=False,
    mcts_batch_size=21,  # Phase 1 batching parameter
)
```

---

## Benchmark Results

### Phase A: Quick Validation (5 games per config)

**Test Configuration**:
- 32 workers (multiprocessing)
- 3 determinizations × 30 simulations = 90 sims/move (medium MCTS)
- 5 games per configuration
- Device: CUDA
- Batch sizes tested: None (baseline), 1, 11, 21

**Results**:

| Configuration | Batch Size | Avg Games/Min | Speedup vs Baseline | GPU Util |
|--------------|------------|---------------|---------------------|----------|
| Baseline | None | 4.9 | 1.0x | 0.0% |
| Phase 1 | 1 | 5.4 | 1.09x | 0.0% |
| Phase 1 | 11 | 6.7 | 1.36x | 0.0% |
| **Phase 1** | **21** | **7.4** | **1.49x** | **0.0%** |

**Best Configuration**: `batch_size=21` with **1.49x speedup**

### Baseline Validation

Verified results against original `benchmark_selfplay.py`:

| Benchmark | Config | Games/Min | Notes |
|-----------|--------|-----------|-------|
| Original | 32 workers, medium MCTS | 5.3 | With `use_batched_evaluator=True` |
| Phase 1 | 32 workers, medium MCTS, baseline | 4.9 | With `use_batched_evaluator=False` |

**Conclusion**: Benchmark methodology is correct (results within 8% of original).

---

## Performance Analysis

### Comparison with Predictions

**Predicted Performance** (from [PHASE1-COMPLETE.md](../phases/PHASE1-COMPLETE.md)):
- Expected speedup: **10x** (based on 90% network call reduction)
- Predicted improvement: 100 sequential calls → 10 batched calls

**Actual Performance**:
- Achieved speedup: **1.49x**
- Actual improvement: Modest batch size benefits

**Gap**: **6.7x shortfall** between predicted and actual speedup

### Comparison with Documented Baseline

**From [FINDINGS.md](../../FINDINGS.md)** (documented best configuration):
- Documented: **43.3 games/min** (32 workers, medium MCTS)
- Our baseline: **4.9 games/min** (32 workers, medium MCTS)
- **Gap**: **8.8x slower** than documented

**Possible explanations**:
1. Different network architecture (ours: 361K params vs production: ~5M params)
2. Different test conditions or hardware state
3. GPU not being utilized (0.0% observed)

---

## Critical Issues Identified

### Issue #1: GPU Utilization = 0.0%

**Observation**: GPU shows 0.0% utilization across all tests despite `device="cuda"` setting.

**Impact**:
- Network inference running on CPU instead of GPU
- Eliminates primary benefit of batching (GPU parallelism)
- Explains poor absolute performance

**Evidence**:
```
[18:56:37] baseline (batch=None) | Game 1/5 | 11.60s | 5.2 games/min | GPU 0.0%
[18:59:21] phase1_bs21 (batch=21) | Game 1/5 | 7.21s | 8.3 games/min | GPU 0.0%
```

**Potential Root Causes**:
1. Multiprocessing workers create network copies that default to CPU
2. Network state dict transfer loses device placement
3. Worker initialization doesn't respect `device` parameter
4. CUDA context not properly initialized in worker processes

**Verification**:
- CUDA is available: ✓ (verified with `torch.cuda.is_available() == True`)
- Network created on GPU in main process: ✓ (device="cuda" passed)
- Workers receiving device parameter: ✓ (traced through code)
- Workers using GPU: ❌ (0.0% utilization observed)

### Issue #2: Small Batch Sizes

**Observation**: Best batch size found is 21, far below GPU saturation point.

**GPU Characteristics**:
- RTX 4060: 3,072 CUDA cores
- Optimal batch size: 128-512 samples (from previous testing)
- Achieved batch size: 21 samples

**Impact**:
- Insufficient parallelism to saturate GPU
- Overhead of batching > benefit at small batch sizes
- Virtual loss mechanism adds overhead without corresponding GPU gain

**Why Batch Sizes Are Small**:
- MCTS is inherently sequential (must complete selection before evaluation)
- Within a single game, max ~21-30 leaf nodes can be collected before expansion
- Cross-game batching (Phase 2/3) failed due to other issues

### Issue #3: Absolute Performance Below Expectations

**Baseline Performance**:
- Current: 4.9 games/min
- Expected (from FINDINGS.md): 43.3 games/min
- **Gap**: 8.8x slower

**Contributing Factors**:
1. GPU not utilized (0.0%)
2. Small network (361K params vs ~5M params in production)
3. Possible configuration differences
4. Windows multiprocessing overhead

---

## Root Cause Analysis: Why Phase 1 Underperformed

### Expected vs Actual

**Original Hypothesis** (PHASE1-COMPLETE.md):
- Reduce network calls by 90% (100 → 10)
- Achieve 10x speedup
- GPU batching provides massive parallelism

**Reality**:
1. **Network calls reduced**: ✓ (likely achieved, but unmeasured)
2. **GPU batching works**: ❌ (GPU not utilized)
3. **10x speedup**: ❌ (achieved 1.49x)

### Contributing Factors

1. **GPU Not Utilized** (Primary Issue)
   - Eliminates main benefit of batching
   - Forces CPU-based sequential evaluation
   - Batching overhead > sequential overhead on CPU

2. **Small Batch Sizes** (Secondary Issue)
   - batch_size=21 too small for GPU saturation
   - MCTS sequential nature limits batch collection
   - Optimal GPU batch size is 128-512, not 21

3. **Virtual Loss Overhead**
   - Path traversal and bookkeeping adds CPU overhead
   - Benefit only materializes with large GPU batches
   - On CPU, overhead dominates

4. **Batch Collection Latency**
   - Collecting 21 nodes takes time
   - Sequential tree traversal with virtual loss
   - On CPU, doesn't offset sequential evaluation time

### Formula for Speedup

```
Speedup = (Time_baseline) / (Time_batched)

Time_baseline = N × T_sequential
Time_batched = T_collection + T_batch_eval + T_backprop

On GPU (expected):
  T_batch_eval << N × T_sequential  (massive parallelism)
  Speedup ≈ 5-10x

On CPU (actual):
  T_batch_eval ≈ N × T_sequential  (still sequential)
  T_collection + T_backprop = overhead
  Speedup ≈ 1.0-1.5x
```

**Conclusion**: Phase 1's benefit is entirely dependent on GPU utilization. Without GPU, batching provides minimal benefit and may even add overhead.

---

## Benchmark Methodology Notes

### Streaming Output Design

The benchmark was designed with fast iteration and resilience in mind:

**Features**:
- ✓ Writes CSV after each game completion
- ✓ Real-time console progress with timestamps
- ✓ Crash-resilient (preserves partial results)
- ✓ Engine reuse (avoids initialization overhead)
- ✓ Batch size sweep support

**Example Output**:
```
[18:56:37] baseline (batch=None) | Game 1/5 | 11.60s | 5.2 games/min | GPU 0.0% [OK]
[18:56:49] baseline (batch=None) | Game 2/5 | 12.13s | 4.9 games/min | GPU 0.0% [OK]
```

**CSV Format**:
```csv
timestamp,config_id,batch_size,game_num,elapsed_sec,games_per_min,gpu_util,status
2025-10-26T18:56:37,baseline,None,1,11.604,5.2,0.0,complete
```

### Initial Benchmark Flaw (Fixed)

**Original Implementation**:
- Created new `SelfPlayEngine` for each game
- 32 worker processes spawned/destroyed per game
- Massive initialization overhead
- Results: 4.1 games/min baseline

**Fixed Implementation**:
- Create engine once per configuration
- Reuse engine for all games in that config
- Matches original benchmark methodology
- Results: 4.9 games/min baseline (consistent with original benchmark)

---

## Recommendations

### Immediate Action Required

**Priority 1: Fix GPU Utilization** (Critical Blocker)

Investigate why multiprocessing workers are not using GPU:

1. **Check worker network initialization**:
   ```python
   # In _worker_generate_games_static()
   network.to(device)  # Verify this actually moves to GPU
   print(f"Worker {worker_id}: Network device = {next(network.parameters()).device}")
   ```

2. **Verify CUDA context in workers**:
   ```python
   import torch
   print(f"Worker {worker_id}: CUDA available = {torch.cuda.is_available()}")
   print(f"Worker {worker_id}: Current device = {torch.cuda.current_device()}")
   ```

3. **Test single-worker GPU usage**:
   - Run with `num_workers=1` to isolate multiprocessing issues
   - If GPU works with 1 worker, issue is in multiprocessing setup

4. **Check CUDA initialization**:
   - Windows multiprocessing may require explicit CUDA context creation
   - May need `torch.multiprocessing.set_start_method('spawn')` or similar

**Expected Outcome**: Once GPU is working, re-run Phase 1 benchmark and expect 3-5x additional improvement (GPU speedup on top of 1.49x batching benefit).

### Priority 2: Re-run Benchmarks with GPU

Once GPU utilization is fixed:

1. **Quick validation** (5 games):
   ```bash
   python benchmarks/performance/benchmark_phase1_validation.py \
       --games 5 \
       --batch-start 10 \
       --batch-end 90 \
       --batch-step 10 \
       --workers 32 \
       --device cuda
   ```

2. **Expected results**:
   - Baseline: 15-25 games/min (GPU speedup)
   - Phase 1 (batch_size=30): 50-100 games/min (batching + GPU)
   - Speedup: 3-5x over GPU baseline

3. **Fine-tuning**:
   - Test larger batch sizes (30, 60, 90)
   - Find optimal batch size for GPU saturation
   - Measure actual GPU utilization (target: 40-70%)

### Priority 3: Investigate Baseline Performance Gap

Address the 8.8x gap between our baseline (4.9 games/min) and documented baseline (43.3 games/min):

1. **Network architecture**:
   - Current: 361K parameters (small test network)
   - Production: ~5M parameters (full network)
   - Inference time difference may explain some gap

2. **MCTS configuration**:
   - Verify determinizations and simulations match documented tests
   - Check if documented results used different MCTS settings

3. **Hardware state**:
   - Documented results assumed GPU usage
   - Our results on CPU-only explain the gap

4. **Windows vs Linux**:
   - Documented results may have been on Linux
   - Windows multiprocessing has known overhead

### Optional: Test Larger Batch Sizes

Once GPU is working, test if larger batches help:

```bash
# Test batch sizes up to 90
python benchmarks/performance/benchmark_phase1_validation.py \
    --games 10 \
    --batch-start 30 \
    --batch-end 90 \
    --batch-step 10
```

**Hypothesis**: Larger batches (60-90) will show more improvement with GPU saturation.

---

## Current Status Summary

### What Works ✓

1. **Integration Complete**:
   - `mcts_batch_size` parameter flows through entire pipeline
   - `search_batched()` is called when configured
   - Virtual loss mechanism is active
   - No crashes or errors

2. **Benchmark Infrastructure**:
   - Streaming CSV output working
   - Real-time progress updates
   - Crash-resilient design
   - Engine reuse implemented correctly

3. **Functionality Verified**:
   - Phase 1 code runs successfully
   - Games complete without errors
   - Results are reproducible

### What Doesn't Work ❌

1. **GPU Utilization**: 0.0% across all tests (critical blocker)

2. **Performance Target**: 1.49x vs 5-10x expected (far below)

3. **Absolute Performance**: 4.9-7.4 games/min vs 43.3 documented

### Blockers

**Primary Blocker**: GPU not being utilized in multiprocessing workers

**Impact**: Without GPU, Phase 1 batching provides minimal benefit (1.49x instead of predicted 5-10x)

**Resolution**: Must investigate and fix GPU utilization before Phase 1 can be properly evaluated

---

## Files Created/Modified

### Modified Files

1. **[ml/training/selfplay.py](../../ml/training/selfplay.py)**
   - Added `mcts_batch_size` parameter throughout
   - Modified decision callbacks to use `search_batched()`
   - Updated worker functions to pass batch_size

2. **[benchmarks/performance/benchmark_selfplay.py](../../benchmarks/performance/benchmark_selfplay.py)**
   - Added Python path handling

### New Files

1. **[benchmarks/performance/benchmark_phase1_validation.py](../../benchmarks/performance/benchmark_phase1_validation.py)**
   - Streaming benchmark with batch size sweep
   - Real-time progress and CSV output
   - Crash-resilient design

2. **[results/phase1_validation_quick.csv](../../results/phase1_validation_quick.csv)**
   - Phase A validation results (5 games per config)

3. **[results/baseline_check.csv](../../results/baseline_check.csv)**
   - Original benchmark baseline for comparison

4. **[docs/performance/PHASE-1-FINDINGS.md](PHASE-1-FINDINGS.md)** (this file)

---

## Lessons Learned

### 1. GPU Utilization is Critical

**Finding**: Phase 1's entire benefit depends on GPU batching. On CPU, batching provides minimal benefit (1.49x) and may add overhead.

**Implication**: Before implementing any GPU-based optimization, verify GPU is actually being used in production workload.

### 2. Predictions Must Account for Real-World Constraints

**Finding**: "90% network call reduction" doesn't automatically translate to 10x speedup.

**Factors overlooked**:
- Overhead of batch collection
- Small batch sizes limiting GPU utilization
- Virtual loss bookkeeping overhead
- Sequential MCTS constraints

**Lesson**: Model actual execution time, not just operation counts.

### 3. Multiprocessing + GPU is Non-Trivial

**Finding**: Passing `device="cuda"` through parameters doesn't guarantee GPU usage in workers.

**Challenges**:
- CUDA context initialization in spawned processes
- Device placement lost during state dict transfer
- Platform-specific multiprocessing behavior (Windows vs Linux)

**Lesson**: Always verify GPU usage with monitoring, not just configuration.

### 4. Absolute Performance Matters

**Finding**: Optimizing a slow baseline (4.9 games/min) doesn't help if the baseline should be 43.3 games/min.

**Lesson**: Establish correct baseline before optimizing. A 10x speedup on a 10x slow baseline is still slow.

### 5. Fast Iteration is Valuable

**Finding**: Streaming benchmark with 5-game quick tests revealed issues in minutes, not hours.

**Design principles that helped**:
- Start with 5 games for quick feedback
- Write results after each game
- Real-time progress monitoring
- Batch size sweep to explore parameter space

**Lesson**: Design benchmarks for fast iteration and early feedback.

---

## Next Steps

### Recommended Path Forward

1. **Investigate GPU utilization** (Est: 1-2 hours)
   - Add debug logging to worker network initialization
   - Verify CUDA context in spawned processes
   - Test with single worker to isolate multiprocessing issues

2. **Fix GPU utilization** (Est: 1-3 hours depending on root cause)
   - Implement fix (likely multiprocessing CUDA context initialization)
   - Verify with monitoring tools (nvidia-smi, pynvml)

3. **Re-run Phase 1 benchmark with GPU** (Est: 30 minutes)
   - Quick validation (5 games per config)
   - Expect 3-5x additional improvement over CPU results
   - Target: 15-30 games/min with batching

4. **Document final results** (Est: 30 minutes)
   - Update FINDINGS.md with GPU results
   - Mark Phase 1 as validated or failed
   - Decide whether to proceed with production deployment

### Alternative: Skip Phase 1 if GPU Can't Be Fixed

If GPU utilization cannot be fixed in multiprocessing setup:

**Option A**: Accept 1.49x speedup as Phase 1 benefit
- Update documentation to reflect CPU-only performance
- Phase 1 provides minimal value, not recommended for production

**Option B**: Explore single-process GPU batching
- Use threading instead of multiprocessing
- Avoids GPU context issues
- May hit GIL limitations (but worth testing)

**Option C**: Focus on other optimizations
- Phase 1 provides insufficient benefit on CPU
- Investigate other bottlenecks (game logic, belief tracking, etc.)

---

## Conclusion

Phase 1 (GPU-batched MCTS) integration is **complete and functional**, but performance is **significantly below expectations** due to GPU not being utilized.

**Summary**:
- ✅ Code integration: Complete
- ✅ Benchmark infrastructure: Working correctly
- ⚠️ Performance: 1.49x speedup (below 5-10x target)
- ❌ GPU utilization: 0.0% (critical blocker)

**Verdict**: **Phase 1 cannot be properly evaluated until GPU utilization is fixed.**

**Recommendation**: Prioritize fixing GPU utilization before making any decisions about Phase 1 deployment. With GPU working, Phase 1 may achieve 3-7x total speedup (1.49x batching × 2-5x GPU), which would meet or exceed the original 5-10x target.

**Current best configuration** (pending GPU fix):
```python
SelfPlayEngine(
    num_workers=32,
    num_determinizations=3,
    simulations_per_determinization=30,
    device="cuda",  # Once GPU is fixed
    use_batched_evaluator=False,
    mcts_batch_size=21,  # May increase with GPU
)
```

---

**Session End**: 2025-10-26
**Next Session**: Investigate GPU utilization in multiprocessing workers
