# Critical Findings: Why GPU Still Idles 90%

**Date**: 2025-10-26
**Status**: Root Cause Identified
**Issue**: GPU utilization remains at 5-10% despite Phase 2 and Phase 3 implementations

---

## Benchmark Results Summary

### Phase 2 (multiprocessing.Pool):
- **Games/min**: 14.8
- **Batching**: 0 requests (NOT WORKING!)
- **Issue**: BatchedEvaluator created in main process but workers don't use it

### Phase 3 (ThreadPoolExecutor):
- **Games/min**: 8.9 (WORSE than Phase 2!)
- **Avg batch size**: 3.5
- **Issue**: Too few concurrent requests to saturate GPU

---

## Root Cause Analysis

### Problem 1: MCTS is Inherently Sequential

**MCTS tree traversal is sequential**:
1. Select node using UCB
2. Expand and evaluate (network call)
3. Backpropagate value
4. Repeat for next simulation

Each simulation **must complete** before the next begins. This means:
- Within a single game: requests arrive **one at a time**
- Even with BatchedEvaluator timeout (10ms), we only accumulate ~3-5 requests
- **Batch size of 3.5 is FAR too small** to saturate a GPU with 3072 CUDA cores

### Problem 2: Not Enough Concurrent Games

**Current setup**:
- 4 workers
- Each worker generates 1 game at a time
- Each game makes ~3 determinizations × 10 simulations = 30 MCTS searches per decision
- But searches are **sequential** within the game

**Concurrency math**:
- Max concurrent requests = 4 workers × ~1-2 requests in flight = **4-8 requests**
- Batch size achieved: **3.5** (confirmed by benchmark)
- GPU needs **128-512** batch size for 70%+ utilization

### Problem 3: Phase 1 and Phase 2/3 Are Incompatible

**Phase 1** (`search_batched()`):
- Batches simulations within a single MCTS search
- Calls network directly with batch
- Bypasses BatchedEvaluator entirely!

**Phase 2/3** (BatchedEvaluator):
- Expects individual requests from multiple sources
- Batches across workers

**They work against each other**:
- If we use `search_batched()`: Calls network directly, ignoring evaluator
- If we use `search()`: Sequential requests, small batches

---

## The Math: Why We Need 50+ Parallel Games

### GPU Saturation Requirements

**NVIDIA RTX 4060 specs**:
- 3072 CUDA cores
- Optimal batch size: 128-512 samples
- Current batch size: 3.5 ❌

**To achieve batch size 128**:
```
Required concurrent requests = 128
Requests per worker = ~3-5 (due to MCTS sequential nature)
Workers needed = 128 / 4 = 32 workers minimum
```

**To achieve batch size 512** (95% GPU util):
```
Workers needed = 512 / 4 = 128 workers!
```

### Why So Many Workers?

Because **MCTS is sequential**:
- Each worker generates games one at a time
- Within each game, MCTS makes sequential network calls
- Even with determinizations (3-5), calls are still mostly sequential
- To get 512 concurrent requests, need 512/4 = 128 workers running simultaneously

---

## The Real Solution: Massive Parallelism

### Option 1: Increase Workers to 32-128

```python
engine = SelfPlayEngine(
    network=network,
    num_workers=128,  # Up from 4!
    device="cuda",
    use_batched_evaluator=True,
    batch_size=1024,
    batch_timeout_ms=5.0,  # Lower timeout
    use_thread_pool=True,
)
```

**Expected results**:
- 128 workers × ~4 requests each = **512 concurrent requests**
- Batch size: **300-512**
- GPU utilization: **70-95%**
- Games/min: **2000-5000** (assuming GPU becomes bottleneck)

### Option 2: Generate Multiple Games Per Worker

Modify workers to generate multiple games concurrently (harder to implement):
```python
# In worker thread
async def generate_multiple_games():
    tasks = [generate_game() for _ in range(8)]  # 8 games per worker
    await asyncio.gather(*tasks)
```

With 16 workers × 8 games each = 128 concurrent games
- Expected batch size: **200-400**
- GPU utilization: **60-80%**

### Option 3: Combine Intra-Game and Cross-Worker Batching

**Hybrid approach** (complex but optimal):
1. Use `search()` (not `search_batched()`) to send individual requests
2. But make determinizations run in parallel (already happens with ThreadPoolExecutor)
3. Increase num_determinizations from 3 → 10-20
4. This creates more concurrent requests per game

```python
engine = SelfPlayEngine(
    num_workers=32,
    num_determinizations=16,  # Up from 3
    simulations_per_determinization=10,
    use_thread_pool=True,
)
```

With 32 workers × 16 determinizations = **512 potential concurrent requests**
- Expected batch size: **200-512**
- GPU utilization: **70-90%**

---

## Recommended Next Steps

### Immediate Action: Test with 32-64 Workers

1. Run benchmark with increased workers:
```bash
python ml/benchmark_phase3.py --workers 32
python ml/benchmark_phase3.py --workers 64
python ml/benchmark_phase3.py --workers 128
```

2. Monitor batch sizes and GPU utilization
3. Find sweet spot where:
   - Batch size > 128
   - GPU utilization > 70%
   - System doesn't run out of memory

### Medium Term: Optimize Architecture

1. **Increase determinizations**: 3 → 10-16
   - More parallel MCTS searches per game
   - Better concurrent request generation

2. **Lower batch timeout**: 10ms → 3-5ms
   - Faster batching for lower latency
   - Still large enough batches with many workers

3. **Tune simulations**: Find balance between quality and speed
   - Fewer simulations = faster games = more throughput
   - But need enough for good play quality

### Long Term: Async Game Generation

Implement truly async game generation where workers can have multiple games in flight:
```python
class AsyncSelfPlayWorker:
    async def generate_games_async(self, num_games):
        tasks = [self._generate_single_game() for _ in range(num_games)]
        return await asyncio.gather(*tasks)
```

This would allow:
- 16 workers × 8 concurrent games = 128 games in flight
- Much better GPU saturation
- But requires significant refactoring

---

## Why Threading Performed Worse Than Multiprocessing

**Phase 2 (multiprocessing)**: 14.8 games/min
**Phase 3 (threading)**: 8.9 games/min (40% SLOWER!)

**Explanation**:
1. **Phase 2's evaluator wasn't being used** (0 requests)
   - Workers were calling network directly
   - No batching overhead
   - Pure sequential inference

2. **Phase 3's evaluator WAS being used** (7106 requests)
   - Added queue/threading overhead
   - But batch size only 3.5 (no benefit)
   - Overhead without benefit = slower

3. **GIL contention with 4 workers**
   - Python GIL limits to ~1-2 threads doing Python work
   - With only 4 workers, GIL becomes bottleneck
   - Need 32+ workers for threading to outperform processes

---

## Conclusion

**The GPU idles because we're not feeding it enough work!**

With only 4 workers:
- ❌ Batch size: 3.5
- ❌ GPU utilization: 5-10%
- ❌ Both CPU and GPU idle 90% of the time

**Solution**: Scale to 32-128 workers:
- ✅ Batch size: 128-512
- ✅ GPU utilization: 70-95%
- ✅ GPU becomes the bottleneck (desired state)
- ✅ Training time: 8-15 days (from 196 days)

**Next benchmark**: Test with `num_workers=32` and measure actual speedup.

---

## Hardware Utilization Target

**Desired state**:
- **CPU**: 60-80% (workers generating games, running MCTS)
- **GPU**: 80-95% (processing large batches continuously)
- **Memory**: 70-80% (large batch buffers)
- **Idle time**: <10% (both CPU and GPU actively working)

**Current state** (4 workers):
- **CPU**: 15-25% (mostly idle, waiting)
- **GPU**: 5-10% (starved for work)
- **Memory**: 20-30% (underutilized)
- **Idle time**: 80-90% (unacceptable!)

---

**Action Items**:
1. ✅ Understand root cause (MCTS sequential + too few workers)
2. ⏳ Test with 32 workers
3. ⏳ Test with 64 workers
4. ⏳ Test with 128 workers
5. ⏳ Find optimal worker count for 70%+ GPU utilization
6. ⏳ Measure actual training throughput (games/min)
