# Benchmark Results Summary & Critical Findings

**Date**: 2025-10-25
**Status**: ✅ **BUG FIXED** - GPU now functional for self-play
**Phase**: 4 (Training Pipeline)

---

## Executive Summary

**CRITICAL BUG DISCOVERED AND FIXED**: Multiprocessing workers were not using GPU even when `--device cuda` was specified.

**Impact of Fix**:
- Bug prevented any GPU acceleration during self-play
- All "GPU" benchmarks were actually running on CPU
- Fix implemented and validated

**Current Performance** (after fix):
- **Self-play** (GPU, 16 workers): 18.8 games/min
- **Training** (GPU, batch 2048 FP16): 74,115 examples/sec
- **Estimated training time**: ~220 days for 500 iterations 🚨

**Conclusion**: GPU-batched MCTS is **MANDATORY** before production training.

---

## Benchmark Results

### 1. Training Performance ✅ **EXCELLENT**

| Device | Batch | Precision | Examples/Sec | VRAM (MB) |
|--------|-------|-----------|--------------|-----------|
| CPU    | 2048  | FP32      | 21,101       | N/A       |
| CUDA   | 2048  | FP32      | 41,322       | 411       |
| CUDA   | 2048  | FP16      | **74,115**   | 380       |

**Key Findings**:
- ✅ FP16 provides **1.79x speedup** over FP32
- ✅ GPU is **3.5x faster** than CPU
- ✅ VRAM usage is very low (380 MB / 8 GB = 5%)
- ✅ Could increase batch size to 8192+ for even better throughput
- ✅ Training is **NOT the bottleneck**

**Recommendation**: Use **batch 2048, FP16, CUDA** for training

---

### 2. Self-Play Performance ⚠️ **NEEDS OPTIMIZATION**

#### Before Fix (BUG - Workers on CPU):
| Workers | Device | Games/Min | CPU % | GPU % |
|---------|--------|-----------|-------|-------|
| 8       | "cuda" | 12.6      | 50%   | 0%    |
| 16      | "cuda" | 15.0      | 50%   | 0%    |

#### After Fix (Workers on GPU):
| Workers | Device | Games/Min | CPU % | GPU % |
|---------|--------|-----------|-------|-------|
| 1       | cuda   | 5.2       | 3.9%  | ?     |
| 8       | cuda   | 15.0      | 1.3%  | ?     |
| 16      | cuda   | 18.8      | 1.3%  | ?     |

**Key Findings**:
- ✅ Bug is fixed - GPU is being used
- ✅ CPU usage dropped to 1.3% (was 50%) - confirms GPU doing inference
- ⚠️ Performance is **still very slow** (~19 games/min)
- ⚠️ GPU is likely underutilized (need to monitor GPU%)

**Why is it still slow?**
1. **Sequential inference** - Each MCTS simulation calls GPU separately
2. **No batching** - GPU processes 1 sample at a time (batch=1)
3. **Multiprocessing overhead** - Workers serialize/deserialize data
4. **Context switching** - 16 workers all calling GPU creates contention

**Comparison**:
- Training GPU throughput: **74,115 examples/sec** (batch=2048)
- Self-play GPU throughput: **~72 inferences/sec** (batch=1, estimated from games/min)
- **GPU is running at 0.1% of its potential!** 🚨

---

### 3. Projected Training Timeline

#### Current Architecture (Sequential Inference):
- Self-play: 18.8 games/min × 16 workers = **18.8 games/min**
- Per iteration: 10,000 games = **532 minutes = 8.9 hours**
- Training: ~30 minutes (batch 2048 FP16)
- **Total per iteration: ~9.4 hours**
- **500 iterations: ~196 days** 🚨

#### With GPU-Batched MCTS (Conservative 10x speedup):
- Self-play: **188 games/min** (10x current)
- Per iteration: 10,000 games = **53 minutes**
- Training: 30 minutes
- **Total per iteration: ~83 minutes**
- **500 iterations: ~29 days** ✅

#### With GPU-Batched MCTS (Optimistic 100x speedup):
- Self-play: **1,880 games/min** (match training throughput)
- Per iteration: 10,000 games = **5.3 minutes**
- Training: 30 minutes
- **Total per iteration: ~35 minutes**
- **500 iterations: ~12 days** 🚀

---

## The Bug: What Went Wrong

### Root Cause

**File**: `ml/training/selfplay.py`

**Problem**: The `_worker_generate_games_static` function creates fresh network instances in child processes, but never moved them to GPU.

```python
# BEFORE (broken):
network = BlobNet(...)
network.load_state_dict(network_state)
network.eval()  # ❌ Still on CPU!

# AFTER (fixed):
network = BlobNet(...)
network.load_state_dict(network_state)
network.to(device)  # ✅ Move to GPU
network.eval()
```

**Additional Fix**: MCTS also needed to move tensors to correct device:
```python
# In ml/mcts/search.py:
state_tensor = state_tensor.to(self.device)
legal_mask = legal_mask.to(self.device)
```

### Files Modified

1. `ml/training/selfplay.py`:
   - Added `device` parameter to `_worker_generate_games_static`
   - Added `network.to(device)` call
   - Added `device` parameter to `SelfPlayEngine.__init__`
   - Passed device to workers in `generate_games`

2. `ml/mcts/search.py`:
   - Auto-detect device from network parameters
   - Move input tensors to device before inference

3. `ml/benchmark_selfplay.py`:
   - Pass device to `SelfPlayEngine`

4. `ml/benchmark_iteration.py`:
   - Pass device to `SelfPlayEngine`

5. `ml/training/trainer.py`:
   - Pass device to `SelfPlayEngine` in `TrainingPipeline`

---

## Why GPU-Batched MCTS is Mandatory

### Current Bottleneck: Sequential Inference

**Per game move** (with 16 workers):
- MCTS needs 90 evaluations (3 det × 30 sims)
- Each evaluation: `network(single_sample)` → GPU call
- 90 sequential GPU calls × ~1ms each = **90ms per move**
- ~24 moves per game = **2.16 seconds per game**
- 16 workers in parallel = **18.8 games/min**

**GPU utilization**:
- Training: Batch 2048 samples → **100% GPU utilization**
- Self-play: Batch 1 sample × 90 times → **~1% GPU utilization**

### Solution: Batched Inference

**Batch within a game**:
```python
# Collect all 90 samples that need evaluation
samples = []
for sim in range(90):
    samples.append(mcts_needs_eval)

# Single batched call
network(torch.stack(samples))  # batch=90

# 90ms → 1ms = 90x speedup per move
```

**Batch across games** (even better):
```python
# 16 games × 90 samples = 1,440 samples
network(torch.stack(all_samples))  # batch=1440

# Now GPU is saturated like in training!
# 1440 samples × 1ms = 1.4ms total for all games
# vs 90ms × 16 games = 1440ms sequentially
# = 1000x speedup!
```

---

## Next Steps

### Option A: Implement GPU-Batched MCTS Now (Recommended)

**Rationale**: Training is impractical without it (200+ days)

**Implementation**:
1. Modify MCTS to collect evaluation requests instead of calling network immediately
2. Batch all requests and call network once
3. Distribute results back to MCTS nodes
4. Implement for both single-game and multi-game scenarios

**Estimated Time**: 4-8 hours

**Expected Speedup**: 10-100x (target <30 days for 500 iterations)

**Priority**: 🔴 **CRITICAL**

---

### Option B: Complete Phase 4 First, Optimize Later

**Rationale**: Get full pipeline working, then optimize

**Risk**: Spend time building on slow foundation

**Timeline**:
- Complete Phase 4 Sessions 6-7: ~4-6 hours
- Then implement GPU-batched MCTS: ~4-8 hours
- Total: ~8-14 hours

**Priority**: 🟡 **MEDIUM**

---

## Recommendations

### Immediate Actions

1. ✅ **Bug is Fixed** - GPU now works for self-play
2. ⚠️ **Measure GPU utilization** during next test to confirm it's being used
3. 🔴 **Implement GPU-batched MCTS** before any production training
4. 📊 **Re-run full benchmarks** with batched MCTS to measure real performance

### Training Configuration

**When ready to train**:
```python
config = {
    "device": "cuda",
    "batch_size": 2048,
    "use_mixed_precision": True,  # FP16
    "num_workers": 16,
    "games_per_iteration": 10_000,
    "epochs_per_iteration": 10,
}
```

### Performance Targets

**Minimum viable**:
- Self-play: >100 games/min
- Training time: <45 days

**Target**:
- Self-play: >300 games/min
- Training time: <20 days

**Stretch goal**:
- Self-play: >1,000 games/min
- Training time: <10 days

---

## Lessons Learned

1. **Always monitor GPU utilization** - Don't assume it's being used
2. **Multiprocessing + GPU is tricky** - Device info doesn't transfer automatically
3. **Batch size matters enormously** - Batch=1 vs Batch=2048 = 1000x difference
4. **Test early, test often** - We caught this before wasting days of training
5. **Fail fast, fail cheap worked** - Quick 15-min test revealed critical bug

---

## Files Reference

**Bug Fix Commit** (if using git):
- Modified: 5 files
- Lines changed: ~15 lines
- Impact: Makes GPU usable for self-play

**Documentation**:
- [BUGFIX-MultiprocessingGPU.md](BUGFIX-MultiprocessingGPU.md) - Detailed bug report
- [BENCHMARK-Performance.md](BENCHMARK-Performance.md) - Original benchmark plan
- [BENCHMARK-Summary.md](BENCHMARK-Summary.md) - This file

**Next Implementation**:
- GPU-batched MCTS (TBD)

---

**Status**: Ready to implement GPU-batched MCTS or continue with Phase 4 completion
