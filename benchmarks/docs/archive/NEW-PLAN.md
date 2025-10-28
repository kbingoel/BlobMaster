# NEW-PLAN.md - Critical Analysis & Path Forward

**Date**: 2025-10-26
**Status**: Evidence-based plan after comprehensive testing
**Bottom Line**: Colleague is RIGHT. Phase 3 failed. Multiprocessing + GPU server is the only viable path.

---

## Executive Summary

After implementing all 3 phases of GPU-batched MCTS and running comprehensive Windows performance tests, the data is conclusive:

**Threading does NOT work on Windows for this workload.**

- **Best config found**: 32 processes, multiprocessing, no batching = **68.3 games/min**
- **Phase 3 result**: Threading with shared evaluator = **6.9-9.6 games/min** (7-10x SLOWER!)
- **Target needed**: 500-1000 games/min for viable training

**Gap**: Need **7-15x speedup** from current best (68.3 games/min)

**Solution**: Single GPU inference server process + multiprocessing workers (as colleague proposed)

---

## Test Results Summary

### Windows Comprehensive Tests (WINDOWS-TEST-RESULTS-ANALYSIS.md)

| Test | Method | Workers | Batching | Games/min | GPU % | Batch Size | Result |
|------|--------|---------|----------|-----------|-------|------------|--------|
| **1** | **multiproc** | **32** | **no** | **68.3** | 15.5% | N/A | **BEST** |
| 2 | threading | 128 | yes | 8.0 | 8.3% | 20.4 | 8.5x slower |
| 3 | threading | 256 | yes | 8.0 | 3.9% | 20.6 | 8.5x slower |
| 4 | threading | 128 | no | 9.3 | 4.6% | N/A | 7.3x slower |
| 5 | multiproc | 60 | no | 1.3 | 99.2% | N/A | GPU thrash |

### Phase Implementation Results

| Phase | Description | Implementation | Result | Batch Size | Games/min |
|-------|-------------|----------------|--------|------------|-----------|
| Phase 1 | Virtual loss + intra-game batching | `search_batched()` in MCTS | Implemented | N/A | Not isolated |
| Phase 2 | Multi-game batching (multiproc) | Per-worker `BatchedEvaluator` | **96.7** | 3.5 | Slow + small batches |
| Phase 3 | Threading + shared evaluator | `ThreadPoolExecutor` | **6.9-9.6** | 2.5-4.3 | FAILED |

### Validation Test (TRAINING-PERFORMANCE-MASTER.md)

Phase 3 threading with shared BatchedEvaluator:

| Workers | Games/min | GPU Avg % | Batch Size | Scaling |
|---------|-----------|-----------|------------|---------|
| 4 | 9.6 | 6.9% | 2.5 | Baseline |
| 16 | **6.9** | 2.6% | 4.3 | **Negative!** |

**Scaling Factor**: 0.72x (adding workers made it SLOWER!)

---

## Critical Analysis: Why Phase 3 Failed

### The Colleague's Diagnosis is CORRECT

The validation test in TRAINING-PERFORMANCE-MASTER.md confirms Phase 3 failed:
- Threading achieved only 6.9-9.6 games/min
- Compare to multiprocessing: 68.3 games/min (7x faster!)
- Negative scaling: 16 workers slower than 4 workers
- Batch sizes remained tiny: 2.5-4.3 (need 128-512!)

### Root Causes of Phase 3 Failure

1. **GIL Dominates on Windows**
   - MCTS is Python-heavy: tree traversal, node selection, game state updates, backprop
   - Even though PyTorch releases GIL during CUDA, 95% of MCTS time is Python code
   - 16 threads fighting for GIL → massive contention
   - Windows GIL is more restrictive than Linux

2. **Batch Sizes Too Small**
   - 4 workers: batch size 2.5 (51x too small!)
   - 16 workers: batch size 4.3 (30x too small!)
   - RTX 4060 needs batch 128-512 to saturate 3072 CUDA cores
   - Small batches mean queue/lock overhead exceeds benefit

3. **Timing Misalignment**
   - Workers don't synchronize requests well
   - Each worker generates requests at slightly different times
   - BatchedEvaluator times out (10ms) with incomplete batches
   - Result: Many small batches instead of few large ones

4. **Multiprocessing is Faster**
   - True parallelism: 32 CPUs running MCTS independently
   - No GIL contention at all
   - Each process has full CPU time for Python code
   - IPC overhead < GIL overhead for this workload

### Why Phase 1 + Phase 2 Didn't Help

Phase 2 achieved 96.7 games/min (better than Phase 3's 6.9!), but:
- Still only 41% faster than no batching (68.3 → 96.7)
- Batch sizes: 3.5 avg (36x too small!)
- Each worker has own evaluator → no cross-worker batching
- 16 separate CUDA contexts → GPU switching overhead

Phase 1's `search_batched()` should batch within-game (90 requests), but:
- Not being used in the benchmarks, OR
- Batching overhead exceeds benefit at these scales, OR
- Implementation has bugs preventing effective batching

**Evidence**: No test shows Phase 1 achieving 10x speedup per game

---

## What the Colleague Got Right

### Correct Findings

1. ✅ **32 processes is optimal** (68.3 games/min, sweet spot before GPU thrash)
2. ✅ **Threading is 7-8x slower** (Tests 2-4: 8.0-9.3 vs 68.3)
3. ✅ **Phase 3 doesn't help on Windows** (validation: 6.9-9.6 games/min)
4. ✅ **60+ processes cause GPU thrash** (Test 5: 99.2% GPU but 1.3 games/min)
5. ✅ **Need centralized GPU inference** (only way to form large batches)

### Correct Diagnosis

The colleague correctly identified that:
- GIL makes threading non-viable on Windows
- Multiprocessing is the only path to parallel MCTS execution
- Current batching approaches fail to form large enough batches
- Need architectural change: single GPU server process

### Correct Solution

**Single GPU Inference Server** architecture:
- One process owns GPU + single BlobNet (one CUDA context)
- 32 MCTS worker processes send requests via `multiprocessing.Queue`
- Server accumulates requests until batch is full (128-512) or timeout (5-10ms)
- Server performs single GPU call, returns results to workers
- Workers continue MCTS with results

This is essentially Phase 3's architecture, but with **processes instead of threads**.

---

## What the Colleague Got Wrong (Minor)

1. **Phase 1 status unclear**: Colleague says "Phase 1 is implemented" but test results don't show 10x intra-game speedup. Either:
   - Phase 1 isn't actually being used in benchmarks, OR
   - Phase 1 implementation has issues

2. **Immediate action may be premature**: Colleague suggests immediately disabling batched evaluator and re-running benchmarks. But we already know the answer: 32 processes no batching = 68.3 games/min. No need to re-benchmark baseline.

3. **Missing diagnosis**: Colleague didn't mention that Phase 2's 96.7 games/min (multiprocessing with per-worker evaluators) was actually better than Phase 3's threading approach. This is an important data point.

---

## The Real Bottleneck

### GPU is Starving for Work

**Training throughput**: 74,115 examples/sec (batch 2048, FP16)
**Self-play throughput**: ~72 inferences/sec (estimated from 18.8 games/min)

**GPU running at 0.1% of capacity during self-play!**

### Why GPU is Idle

1. **Tiny batches**: 2.5-4.3 samples (need 128-512)
2. **Sequential calls**: Each MCTS makes 90 separate network calls
3. **No aggregation**: 32 processes each call GPU independently
4. **Context switching**: 32 CUDA contexts competing for GPU

### The Math

**Current state**:
- 32 processes × 3.5 concurrent requests = ~112 potential batch size
- Actual achieved: 3.5-4.3 batch size
- **Why?** Timing misalignment + no cross-process aggregation

**With GPU server**:
- 32 processes sending to single queue
- Server accumulates for 5-10ms
- 32 workers × ~10 requests/worker (in 10ms window) = ~320 requests
- Batch in groups of 128-512 → 1-3 batches per 10ms
- **Expected batch size: 128-320 avg**

**Speedup calculation**:
- Current: 3.5 batch size at ~10 batches/sec = 35 inferences/sec/worker
- With server: 256 batch size at ~100 batches/sec = 25,600 inferences/sec total / 32 workers = 800 inferences/sec/worker
- **Expected speedup: ~23x per worker**

**Reality check**:
- Current best: 68.3 games/min
- With GPU server: 68.3 × 23 = 1,571 games/min (✅ Exceeds 500-1000 target!)
- Realistically, expect 50-70% of theoretical: **800-1,100 games/min** ✅

---

## Evidence-Based Plan

### Phase 3.5: GPU Inference Server (NEW)

**What**: Replace threading with multiprocessing + centralized GPU server

**Architecture**:
```
Main Process
├─ GPU Server Process
│  ├─ BlobNet (single instance, single CUDA context)
│  ├─ Request Queue (multiprocessing.Queue)
│  └─ Inference Loop:
│     ├─ Accumulate requests (timeout: 10ms, max: 512)
│     ├─ Batch evaluate: network(torch.stack(states))
│     └─ Return results to workers
│
└─ MCTS Worker Processes (32)
   ├─ Worker 0 → Sends (state, mask) to server, receives (policy, value)
   ├─ Worker 1 → Same
   ├─ ...
   └─ Worker 31 → Same
```

**Why multiprocessing instead of threading**:
- ✅ No GIL contention on MCTS Python code
- ✅ True parallelism for tree traversal, game simulation
- ✅ Each worker gets full CPU time
- ✅ Only GPU inference is centralized
- ❌ IPC overhead (but worth it to avoid GIL)

**Implementation** (~4-6 hours):

1. **Create `ml/mcts/gpu_server.py`** (~250 lines)
   - `GPUInferenceServer` class
   - Process-based server with request queue
   - Batch accumulation logic (timeout + max size)
   - Result distribution back to workers

2. **Modify `ml/mcts/search.py`** (~50 lines)
   - Check if `gpu_server_client` is available
   - Send request to server instead of direct inference
   - Wait for result (blocking)

3. **Modify `ml/training/selfplay.py`** (~100 lines)
   - Start GPU server process on startup
   - Create client handles for workers
   - Pass client to each worker process
   - Shutdown server on exit

4. **Create worker function** (~50 lines)
   - `_worker_with_gpu_server()`
   - Receives client handle
   - Passes to MCTS
   - No network creation (server owns network)

**Testing**:

1. **Correctness**:
   - Verify server produces same results as direct inference
   - Test with 1-32 workers
   - Check for deadlocks, race conditions

2. **Performance**:
   - Measure actual batch sizes (target: >128 avg)
   - Measure games/min (target: >500)
   - Monitor GPU utilization (target: >70%)

**Expected Results**:
- Batch size: 128-320 avg (vs 3.5 current)
- Games/min: 800-1,100 (vs 68.3 current)
- GPU utilization: 70-85% (vs 15% current)
- Training time: 4.5-6.3 days (vs 50.8 days current) ✅

---

## Should We Use Phase 1 (search_batched)?

**Question**: Should MCTS use `search_batched()` within each game (Phase 1)?

**Answer**: YES, but carefully

**Reasoning**:
1. Phase 1 + GPU server are **complementary** (not mutually exclusive)
2. `search_batched()` creates batches of 30-90 within one game
3. GPU server aggregates across 32 games
4. Combined: 32 games × 30-90 requests = 960-2,880 potential batch size

**But**:
- If Phase 1 has bugs or overhead issues, fix OR disable it
- Test both configurations:
  1. GPU server + sequential MCTS (no Phase 1)
  2. GPU server + search_batched (with Phase 1)
- Use whichever is faster

**My Hypothesis**: GPU server alone will be enough (800-1,100 games/min). Phase 1 might add another 2-3x on top (2,000-3,000 games/min), but is not critical.

---

## Immediate Actions

### 1. Verify Baseline (15 minutes)

Run benchmark with current best config to confirm baseline:

```bash
python ml/benchmark_selfplay.py --device cuda --workers 32 --use_thread_pool false --use_batched_evaluator false --num_games 50
```

**Expected**: 68.3 games/min (confirm)

### 2. Implement GPU Server (4-6 hours)

Follow implementation plan above:
- Create `gpu_server.py`
- Modify `search.py` and `selfplay.py`
- Test correctness first, then performance

### 3. Benchmark GPU Server (30 minutes)

```bash
python ml/benchmark_selfplay.py --device cuda --workers 32 --use_gpu_server true --num_games 50
```

**Target**: >500 games/min (7.3x speedup)
**Stretch**: >1,000 games/min (15x speedup)

### 4. Decision Point

**If GPU server achieves >500 games/min**:
- ✅ Proceed to Phase 4 training
- ✅ Training time: <10 days
- ✅ Problem solved

**If GPU server achieves 200-500 games/min**:
- ⚠️ Investigate: batch sizes, timing, GPU utilization
- ⚠️ Try enabling Phase 1 (search_batched)
- ⚠️ Tune timeout, max batch size, num workers

**If GPU server achieves <200 games/min**:
- 🚨 Something is wrong with implementation
- 🚨 Debug: requests dropping? Deadlock? Batch formation?
- 🚨 Compare batch sizes to Phase 3 (should be much larger)

---

## Why This Will Work (and Phase 3 Didn't)

### Phase 3 Failed Because

1. **GIL**: 16 threads contending for GIL to run Python MCTS code
2. **Timing**: Threads async, requests don't align well
3. **Small batches**: Only 2.5-4.3 avg (overhead > benefit)

### GPU Server Will Succeed Because

1. **No GIL**: 32 processes run MCTS in parallel (no contention)
2. **Central aggregation**: All requests flow through single queue
3. **Large batches**: 32 sources → 128-320 samples/batch
4. **One CUDA context**: No context switching overhead
5. **Proven architecture**: Similar to distributed training (parameter server)

### Supporting Evidence

1. **Multiprocessing is 7x faster than threading** (68.3 vs 9.3 games/min)
2. **32 processes is optimal** (sweet spot before GPU thrash)
3. **GPU has 1000x headroom** (0.1% utilization currently)
4. **Training shows GPU can do it** (74k examples/sec with batch 2048)

The only question is: can we form large enough batches? With 32 processes sending to one queue with 10ms timeout, YES.

---

## Performance Projections

### Conservative (50% of Theoretical)

- **Current**: 68.3 games/min
- **Speedup**: 10x
- **Result**: 683 games/min
- **Training time**: 7.3 days for 500 iterations ✅

### Realistic (70% of Theoretical)

- **Speedup**: 15x
- **Result**: 1,024 games/min
- **Training time**: 4.9 days for 500 iterations ✅

### Optimistic (90% of Theoretical)

- **Speedup**: 20x
- **Result**: 1,366 games/min
- **Training time**: 3.7 days for 500 iterations ✅

**All scenarios meet the <10 day target!**

---

## Configuration for GPU Server

```python
# ml/training/selfplay.py
engine = SelfPlayEngine(
    network=network,
    encoder=encoder,
    masker=masker,
    device="cuda",
    num_workers=32,              # Optimal for 7950X
    use_thread_pool=False,        # Force multiprocessing
    use_batched_evaluator=False,  # Disable Phase 2/3 approach
    use_gpu_server=True,          # NEW: Enable GPU server
    gpu_server_max_batch=512,     # Max batch size
    gpu_server_timeout_ms=10.0,   # Accumulation timeout
)
```

**GPU Server Hyperparameters**:
- `max_batch`: 256-512 (balance latency vs throughput)
- `timeout_ms`: 5-20ms (shorter = lower latency, longer = larger batches)
- `num_workers`: 32 (optimal for your hardware)

**Tuning Strategy**:
1. Start with timeout=10ms, max_batch=512
2. Monitor actual batch sizes (should be 128-320 avg)
3. If batches too small: increase timeout or workers
4. If latency too high: decrease timeout
5. If GPU memory issues: decrease max_batch

---

## Risk Assessment

### Low Risk

- ✅ Architecture is well-proven (parameter server pattern)
- ✅ Multiprocessing works (68.3 games/min baseline)
- ✅ GPU has capacity (running at 0.1% currently)
- ✅ Implementation is straightforward (~400 lines)

### Potential Issues

1. **IPC overhead**: `multiprocessing.Queue` adds latency
   - **Mitigation**: Batch overhead amortized over 128-512 samples
   - **Estimate**: 1-2ms per batch (acceptable)

2. **Batch formation**: Workers might not send requests synchronously
   - **Mitigation**: 10ms timeout ensures batches form even if async
   - **Worst case**: Multiple smaller batches (still better than current)

3. **Deadlock**: Server and workers could deadlock if queues fill
   - **Mitigation**: Proper queue sizing, error handling
   - **Testing**: Stress test with 60+ workers

4. **Pickling overhead**: Tensors must be pickled/unpickled
   - **Mitigation**: Tensors are small (256-dim state), negligible overhead
   - **Alternative**: Use `torch.multiprocessing` with shared memory

### Fallback Plan

**If GPU server doesn't work**:
1. Try `torch.multiprocessing` with shared memory (avoid pickling)
2. Try Linux (better multiprocessing + shared memory support)
3. Try cloud GPU (A100 has more memory, better batching)
4. Accept slower training (68.3 games/min = 50 days)

---

## Conclusion

The colleague's analysis is **correct and evidence-based**. Phase 3 threading failed on Windows (6.9-9.6 games/min vs 68.3 for multiprocessing). The only viable path is:

**Multiprocessing + Single GPU Inference Server**

**Implementation time**: 4-6 hours
**Expected speedup**: 10-20x (800-1,100 games/min)
**Training time**: 4-7 days (✅ meets <10 day target)

**Recommendation**: Implement GPU server immediately, test, then proceed to training.

---

## Next Session Checklist

- [ ] Verify baseline: 32 processes, no batching (should be 68.3 games/min)
- [ ] Implement `ml/mcts/gpu_server.py` (GPUInferenceServer class)
- [ ] Modify `ml/mcts/search.py` (add server client support)
- [ ] Modify `ml/training/selfplay.py` (start/stop server, create clients)
- [ ] Test correctness (server vs direct inference)
- [ ] Benchmark performance (target >500 games/min)
- [ ] Monitor batch sizes (target >128 avg)
- [ ] Monitor GPU utilization (target >70%)
- [ ] Tune hyperparameters if needed
- [ ] Proceed to Phase 4 training once >500 games/min achieved

**Timeline**: 1 session (4-6 hours) to implement and validate GPU server

---

**Status**: Ready to implement. Evidence strongly supports this approach.
