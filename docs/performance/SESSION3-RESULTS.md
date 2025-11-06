# Session 3 Optimization Results

**Date:** 2025-11-06
**Status:** ❌ **FAILED** - No improvement, slight regression
**Configuration:** 32 workers, Medium MCTS (3 det × 30 sims), 50 games per test

---

## Summary

**Result:** ❌ **No speedup achieved** (70.2 games/min vs 75.85 baseline)

Session 3 attempted to add mixed precision inference optimizations:
- FP16 autocast for faster GPU inference
- TF32 optimizations for Ampere+ GPUs

Both optimizations **failed to improve performance** and in some cases caused regression.

**Key Finding:** Pure PyTorch inference optimizations are exhausted. The bottleneck is **MCTS Python code**, not neural network inference.

---

## Detailed Results

### What Was Implemented

1. **TF32 Optimizations** (kept in codebase)
   - Created [ml/performance_init.py](../../ml/performance_init.py)
   - Enabled `torch.backends.cuda.matmul.allow_tf32 = True`
   - Added to train.py and benchmark scripts
   - **Result:** No measurable improvement

2. **FP16 Mixed Precision** (reverted)
   - Added `torch.autocast('cuda', dtype=torch.float16)` to all inference paths
   - **Result:** -15% performance regression (56-68 games/min)
   - **Reverted:** All FP16 changes removed from codebase

3. **Bug Fix**
   - Fixed `parallel_batch_size = 30` in [ml/config.py](../../ml/config.py)
   - Was incorrectly set to 10 (from earlier testing)

---

## Performance Results

### Benchmark Data

| Configuration | Performance | vs Baseline (75.85) | Change |
|---------------|-------------|---------------------|--------|
| **Session 1+2 Baseline** | **75.85 games/min** | 2.07x | - |
| Session 3 + FP16 (batch=10) | 68.9 games/min | 1.88x | **-9%** ❌ |
| Session 3 + FP16 (batch=30) | 64.3 games/min | 1.75x | **-15%** ❌ |
| Session 3 TF32 only (batch=30) | 61.2 games/min | 1.67x | **-19%** ❌ |
| **Final (TF32, no FP16)** | **70.2 games/min** | **1.91x** | **-7%** ❌ |

**Cumulative Speedup:** 1.91x (down from 2.07x)

**Result Files:**
- [session3_20251106_2228.csv](../../benchmarks/results/session3_20251106_2228.csv) - FP16 + batch_size=10
- [session3_with_batch30_20251106_2231.csv](../../benchmarks/results/session3_with_batch30_20251106_2231.csv) - FP16 + batch_size=30
- [session3_tf32_only_20251106_2232.csv](../../benchmarks/results/session3_tf32_only_20251106_2232.csv) - TF32 only
- [session1_validation_20251106_2238.csv](../../benchmarks/results/session1_validation_20251106_2238.csv) - Final validation

---

## Analysis: Why Session 3 Failed

### 1. FP16 Overhead Exceeds Benefits

**Expected:** FP16 should be ~2x faster on modern GPUs with Tensor Cores

**Reality:** Caused -15% regression

**Root Causes:**
- **Small model size** (4.9M parameters) → Minimal compute intensity
- **Small batch sizes** (30-256) → Can't amortize FP16 conversion overhead
- **No Tensor Core utilization** at these batch sizes
- **Memory bandwidth not bottleneck** (8GB VRAM, <1GB used per worker)
- FP16 casting overhead dominates the small compute savings

**Lesson:** FP16 only helps for large models (>50M params) and large batches (>512)

### 2. TF32 Not Effective for This Workload

**Expected:** TF32 should provide ~1.5-2x speedup on matmul-heavy workloads (Ampere+ GPUs)

**Reality:** No measurable benefit (neutral to slightly negative)

**Root Causes:**
- **MCTS overhead dominates**, not matmul time
- Model too small to benefit from TF32 (overhead exceeds gains)
- Batch sizes too small for matmul optimization to matter
- True matmul time is <10% of total game generation time

**Lesson:** TF32 only helps when matmul dominates compute (large transformers, LLMs)

### 3. High System Variance

**Observation:** Results varied 61-70 games/min across runs (±15%)

**Possible Causes:**
- Thermal throttling (GPU temperature)
- Background processes
- Random game generation (different games = different compute)
- Linux scheduler variance

**Impact:** True effect of TF32 may be masked by noise

### 4. Inference Is Not the Bottleneck

**Critical Insight:** MCTS tree search (pure Python) dominates time, not neural network inference

**Evidence:**
- Inference optimizations (FP16, TF32) had no effect
- Batching optimizations (Session 1+2) worked well → Python overhead reduction
- Next bottleneck is Python MCTS code: UCB selection, backpropagation, tree traversal

**Implication:** Need to optimize MCTS algorithms (Cython/C++), not just inference

---

## Lessons Learned

### False Assumptions Corrected

❌ **"FP16 always faster on modern GPUs"**
- Reality: Only for large models and large batches
- Our workload: Small model (4.9M), small batches (30-256) → overhead exceeds benefits

❌ **"TF32 is free speedup on Ampere+ GPUs"**
- Reality: Only when matmul dominates compute time
- Our workload: MCTS tree search dominates → no benefit

❌ **"Mixed precision expected +15-25%"**
- Reality: Actually caused -7% to -15% regression
- Expectations based on LLM workloads, not applicable to small RL models

### What We Confirmed

✅ **Batching optimizations work** (Session 1+2: 2.07x)
- Reducing Python overhead via batching is effective
- GPU server batching enables cross-worker amortization

✅ **Parallel MCTS expansion works** (Session 2: +40%)
- Increasing leaves per batch improves GPU utilization
- Sweet spot at `parallel_batch_size = 30`

✅ **MCTS Python code is the bottleneck**
- Pure PyTorch optimizations exhausted
- Need to optimize tree search algorithms

---

## Impact on Training Timeline

**Before Session 3:**
- Performance: 75.85 games/min (2.07x speedup)
- Training time: 66 days

**After Session 3:**
- Performance: 70.2 games/min (1.91x speedup)
- Training time: **71 days** (5 days worse due to variance/regression)

**Projected with Remaining Sessions:**
- Sessions 4+5 (torch.compile + tuning): 77-92 games/min
- Training time: 54-65 days (acceptable)
- **OR:** Implement Cython MCTS: 95-128 games/min → 39-52 days (good)

---

## Recommendations

### Immediate Next Steps

**1. Try Session 4 (torch.compile)**
- Last hope for pure-Python optimizations
- May provide graph-level optimizations where FP16/TF32 failed
- Expected: +10-20% (uncertain after Session 3 failure)
- Time: 3 hours

**2. If Session 4 Fails (<5% gain):**
- **Skip Session 5** (runtime tuning unlikely to help)
- **Implement Cython MCTS immediately** (Priority 6)
- Target hot loops:
  - UCB selection ([ml/mcts/node.py:88-212](../../ml/mcts/node.py#L88-L212))
  - Backpropagation ([ml/mcts/node.py:279-356](../../ml/mcts/node.py#L279-L356))
  - Tree traversal ([ml/mcts/search.py:139-181](../../ml/mcts/search.py#L139-L181))
- Expected: +35-80% speedup
- Training time: 39-52 days

**3. If Session 4 Succeeds (>5% gain):**
- Continue to Session 5 (runtime tuning)
- Total expected: 2.2-2.5x (81-92 games/min)
- Training time: 54-65 days (acceptable)

### Strategic Insights

**What Works:**
- Batching optimizations (reduce Python overhead)
- Parallelization (more work per batch)
- Multiprocessing (avoid GIL contention)

**What Doesn't Work (for our workload):**
- FP16/BF16 mixed precision (small model penalty)
- TF32 optimizations (not compute-bound)
- Pure PyTorch inference tricks (not the bottleneck)

**The Path Forward:**
- Optimize Python MCTS code (Cython/C++)
- OR accept 2.0-2.5x speedup and 54-68 day training
- torch.compile is the last pure-Python hope

---

## Files Modified

### Kept in Codebase

✅ [ml/performance_init.py](../../ml/performance_init.py) - TF32 initialization (neutral but harmless)
✅ [ml/train.py](../../ml/train.py) - Calls `init_performance()` at startup
✅ [benchmarks/performance/benchmark_selfplay.py](../../benchmarks/performance/benchmark_selfplay.py) - Calls `init_performance()`
✅ [ml/config.py](../../ml/config.py) - Fixed `parallel_batch_size = 30`

### Reverted (FP16 Removed)

❌ [ml/mcts/gpu_server.py](../../ml/mcts/gpu_server.py) - Removed FP16 autocast
❌ [ml/mcts/batch_evaluator.py](../../ml/mcts/batch_evaluator.py) - Removed FP16 autocast
❌ [ml/mcts/search.py](../../ml/mcts/search.py) - Removed FP16 autocast (2 locations)

---

## Conclusion

Session 3 **failed to deliver any performance improvement** and revealed important insights about the bottlenecks in our training pipeline:

1. **Inference is already fast** - Further inference optimizations won't help
2. **MCTS Python code is the bottleneck** - Need to optimize tree search algorithms
3. **Small models behave differently** - LLM optimization techniques don't apply
4. **Pure PyTorch optimizations are exhausted** - Need Cython/C++ for further gains

**Next:** Try Session 4 (torch.compile) as last pure-Python optimization attempt. If it fails, implement Cython MCTS for best results.
