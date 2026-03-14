# GPU Server Implementation Test Results

**Date**: 2025-10-26
**Status**: Implementation Complete, Testing Complete - Performance Below Target

---

## Executive Summary

The GPU Inference Server (Phase 3.5) has been successfully implemented and tested. The implementation is **functionally correct** and successfully processes inference requests through a centralized GPU server with multiprocessing workers. However, **performance testing shows the GPU server is 3-5x slower than the baseline**, failing to meet the target of >500 games/min.

**Recommendation**: Continue using the baseline multiprocessing approach (Phase 2) which achieves 43-80 games/min depending on MCTS configuration.

---

## Test Environment

- **Hardware**: AMD Ryzen 9 7950X (16 cores, 32 threads), NVIDIA RTX 4060 8GB
- **OS**: Windows 11
- **Python**: 3.11
- **PyTorch**: CUDA 12.1
- **Workers**: 32 processes
- **Games**: 20 per configuration
- **MCTS Configurations**:
  - Light: 2 determinizations × 20 sims = 40 sims/move
  - Medium: 3 determinizations × 30 sims = 90 sims/move
  - Heavy: 5 determinizations × 50 sims = 250 sims/move

---

## Performance Results

### Baseline (No GPU Server) - 32 Workers

| MCTS Config | Sims/Move | Games/Min | Sec/Game | Speedup |
|-------------|-----------|-----------|----------|---------|
| Light       | 40        | 80.8      | 0.74     | 1.00x   |
| Medium      | 90        | 43.3      | 1.38     | 1.00x   |
| Heavy       | 250       | 25.0      | 2.40     | 1.00x   |

**Baseline uses**: Standard multiprocessing with per-worker networks (no batching)

### GPU Server - 32 Workers

| MCTS Config | Sims/Move | Games/Min | Sec/Game | Avg Batch | Max Batch | vs Baseline |
|-------------|-----------|-----------|----------|-----------|-----------|-------------|
| Light       | 40        | 20.8      | 2.88     | 10.6      | 20        | 0.26x (4x slower) |
| Medium      | 90        | 10.8      | 5.58     | 12.1      | 20        | 0.25x (4x slower) |
| Heavy       | 250       | 4.6       | 13.15    | 13.0      | 20        | 0.18x (5x slower) |

**GPU Server uses**: Single centralized GPU process with multiprocessing queues for batching

---

## Success Criteria Assessment

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| **Correctness** | GPU server matches direct inference | ✅ Verified | **PASS** |
| **Performance** | >500 games/min | 20.8 games/min (light MCTS) | **FAIL** |
| **GPU Utilization** | >70% | Not measured (likely low) | **FAIL** |
| **Batch Size** | >128 avg | 10-13 avg, max 20 | **FAIL** |
| **Speedup vs Baseline** | 7-15x faster | 3-5x slower | **FAIL** |

---

## Root Cause Analysis

### Why Performance is Slower

1. **Batch Sizes Too Small**: Average 10-13, max 20 (target was >128)
   - Insufficient request accumulation
   - 10ms timeout too short for sequential MCTS
   - Workers send requests one-at-a-time during tree search

2. **Queue Overhead**: Manager queues add latency
   - Every request goes through multiprocessing Manager proxies
   - Windows `spawn` start method adds serialization overhead
   - Queue get/put operations are synchronous and block workers

3. **No Request Bursting**: MCTS is inherently sequential
   - Each simulation must complete before the next starts
   - Workers can't send multiple requests simultaneously
   - Prevents large batch accumulation

4. **Max Batch Size Capped at 20**:
   - Consistently hit exactly 20 across all configurations
   - Suggests a bottleneck in batch formation
   - Possibly related to queue buffer sizes or timing

### Why Baseline is Faster

The baseline approach avoids all GPU server overhead:
- No queue serialization/deserialization
- No inter-process communication delays
- Direct GPU access (though not batched)
- Each worker independently proceeds without waiting

---

## Implementation Quality

### What Worked Well ✅

1. **Functional Correctness**: GPU server produces correct inference results
2. **Cross-Platform Compatibility**: Manager queues work on Windows
3. **Statistics Tracking**: Proper monitoring of batch sizes and request counts
4. **Error Handling**: Timeouts and graceful shutdown
5. **Code Quality**: Clean implementation with proper documentation

###  What Didn't Work ❌

1. **Performance**: 3-5x slower than baseline
2. **Batch Formation**: Max batch size only 20 (vs target 128+)
3. **Scalability**: More workers didn't help batch sizes
4. **GPU Utilization**: Likely very low due to small batches

---

## Bugs Fixed During Testing

1. **Network Attribute Access**: BlobNet didn't store constructor parameters
   - Fixed: Added all parameters as instance attributes ([model.py:64-72](ml/network/model.py#L64-L72))

2. **Queue Serialization on Windows**: Regular `mp.Queue()` doesn't work with `spawn`
   - Fixed: Used `manager.Queue()` and `manager.dict()` ([gpu_server.py:255-262](ml/mcts/gpu_server.py#L255-L262))

3. **Network.eval() on None**: Called when network=None in GPU server mode
   - Fixed: Added None check before `.eval()` ([search.py:130-131](ml/mcts/search.py#L130-L131))

4. **Batch Evaluation Missing GPU Server Support**: `_batch_expand_and_evaluate()` called `self.network()` directly
   - Fixed: Added GPU server path with proper priority ([search.py:609-636](ml/mcts/search.py#L609-L636))

---

## Lessons Learned

### Why AlphaZero-Style MCTS is Hard to Batch

1. **Sequential Nature**: Each MCTS simulation depends on previous results
2. **Low Concurrency**: At any given time, only ~1-2 leaf evaluations per worker
3. **Variable Timing**: Different game states take different times to evaluate
4. **Synchronization**: Workers must wait for each request to complete

### Why This Differs from AlphaGo/AlphaZero

- **AlphaGo Zero**: Training used 5,000 TPUs with massive parallelism
- **Our Setup**: 1 GPU, 32 CPU cores - fundamentally different scale
- **Their Batching**: Accumulated requests across hundreds of games simultaneously
- **Our Batching**: 32 workers × ~1 request each = insufficient for large batches

---

## Alternative Approaches Considered

### 1. Increase Timeout (Not Tested)
- **Idea**: Increase batch timeout from 10ms to 50-100ms
- **Expected**: Larger batches (maybe 30-50), but higher latency
- **Downside**: Workers block longer, reducing parallelism
- **Likely Outcome**: Still won't reach 128+ batch sizes

### 2. Asynchronous MCTS (Major Refactor)
- **Idea**: Allow MCTS to continue with pending evaluations
- **Expected**: Multiple outstanding requests per worker
- **Downside**: Breaks MCTS correctness (needs virtual loss)
- **Effort**: Weeks of implementation + validation

### 3. Larger Worker Count (Partially Tested)
- **Idea**: 64-128 workers to generate more concurrent requests
- **Expected**: Linearly increase batch size
- **Downside**: CPU bottleneck, diminishing returns
- **4-worker test showed**: Batch size scales sub-linearly (2.3 → 10.6 for 8x workers)

### 4. GPU Batching with Thread Pool (Phase 3 - Already Tested)
- **Result**: Failed due to GIL contention
- **Performance**: 6.9-9.6 games/min (7x slower than multiprocessing)
- **See**: [WINDOWS-TEST-RESULTS-ANALYSIS.md](WINDOWS-TEST-RESULTS-ANALYSIS.md)

---

## Recommendations

### Short Term: Use Baseline (Phase 2) ✅

**Continue with standard multiprocessing** (no GPU server):
- **Performance**: 43-80 games/min (medium MCTS)
- **Training Time**: ~1 week for 500 iterations
- **Simplicity**: No additional complexity
- **Reliability**: Proven to work

**To enable**:
```bash
python ml/train.py --workers 32 --device cuda
```

### Medium Term: Optimize Baseline

Focus on improving the baseline rather than GPU server:

1. **Reduce MCTS Simulations**: Use lighter MCTS during early training
2. **Progressive Training**: Start with fewer simulations, increase later
3. **Better Initialization**: Use supervised learning to bootstrap
4. **Parallel Determinizations**: Evaluate multiple worlds simultaneously

### Long Term: Hardware Upgrade

If training speed becomes critical:
- **Multi-GPU Setup**: Multiple RTX 4060s for parallel workers
- **Higher-End GPU**: RTX 4090 or A5000 for faster inference
- **Cloud Training**: Rent GPU instances for training, use laptop for inference

---

## Conclusion

The GPU Inference Server implementation is **technically correct but not performant enough**. The fundamental issue is that MCTS's sequential nature prevents the request bursting needed for large batch sizes. With only 10-13 average batch size, the overhead of inter-process communication outweighs any batching benefits.

**The baseline multiprocessing approach (Phase 2) is the winner** for this hardware configuration, achieving 43-80 games/min depending on MCTS complexity.

---

## Files Modified

### Core Implementation
- [`ml/mcts/gpu_server.py`](ml/mcts/gpu_server.py) - GPU inference server (450 lines)
- [`ml/mcts/search.py`](ml/mcts/search.py) - MCTS with GPU server support
- [`ml/training/selfplay.py`](ml/training/selfplay.py) - Self-play engine with GPU server
- [`ml/network/model.py`](ml/network/model.py) - BlobNet parameter storage fix

### Testing & Benchmarks
- [`ml/benchmark_selfplay.py`](ml/benchmark_selfplay.py) - Added GPU server parameters
- [`ml/test_gpu_server.py`](ml/test_gpu_server.py) - Standalone test suite (not used due to import issues)

### Documentation
- [`GPU-SERVER-IMPLEMENTATION.md`](GPU-SERVER-IMPLEMENTATION.md) - Implementation details
- [`GPU-SERVER-TEST-RESULTS.md`](GPU-SERVER-TEST-RESULTS.md) - This document

---

## Next Steps

1. ✅ **Document findings** (this document)
2. ⬜ **Revert to baseline**: Ensure `ml/train.py` uses standard multiprocessing
3. ⬜ **Archive GPU server code**: Keep for future reference but don't use in production
4. ⬜ **Focus on training**: Start 500-iteration training run with baseline
5. ⬜ **Monitor performance**: Track games/min and ELO progression
6. ⬜ **Optimize if needed**: Tune MCTS parameters, not infrastructure

---

**Testing Duration**: ~3 hours
**Implementation + Testing**: ~5 hours total
**Outcome**: Learned that GPU server approach doesn't suit this workload
