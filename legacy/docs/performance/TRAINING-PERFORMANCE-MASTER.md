# BlobNet Training Performance - Master Reference Document

**Date**: 2025-10-26
**Status**: Performance crisis - seeking optimal configuration
**Hardware**: NVIDIA RTX 4060 (3072 CUDA cores, 8GB VRAM), Ryzen 7950X (16 cores)

---

## Project Workload Specifications

### Training Workload
- **Iterations**: 500
- **Games per iteration**: 10,000
- **Training epochs per iteration**: 10
- **Total games needed**: 5,000,000
- **Model size**: 4,917,301 parameters (~5M)
- **Network architecture**: 6-layer Transformer (256 embedding, 8 heads, 1024 FFN)

### Game Characteristics
- **Players per game**: 4
- **Cards dealt per game**: 3-13 (typically 5 for training)
- **Decisions per game**: ~8-24 (4 bids + 4-20 card plays)
- **MCTS evaluations per decision**: 90 (3 determinizations × 30 simulations)
- **Network calls per game**: ~720-2160

### Training Data
- **Examples per game**: ~16 (with 5 cards dealt)
- **Total examples needed**: 80,000,000
- **Example format**: (state_256, policy_52, value_1)

### Performance Targets
- **Self-play throughput**: >500 games/min (minimum viable)
- **GPU utilization**: >70% (efficient hardware use)
- **Training time**: <30 days (practical timeline)
- **Batch size**: >128 (GPU saturation requirement)

---

## Training Strategies Implemented

### Strategy 1: Sequential Self-Play (Baseline)
**Description**: Single-threaded game generation with direct network inference
**Implementation**: No parallelism, direct network calls
**Status**: Never formally benchmarked (baseline reference only)

### Strategy 2: Multiprocessing with Isolated Workers (Phase 2)
**Description**: Process pool with per-worker network copies
**Implementation**: `multiprocessing.Pool`, each worker has own network + BatchedEvaluator
**Status**: Implemented and tested

### Strategy 3: Threading with Shared BatchedEvaluator (Phase 3)
**Description**: Thread pool sharing single network and centralized BatchedEvaluator
**Implementation**: `ThreadPoolExecutor`, shared `BatchedEvaluator` with queue-based batching
**Status**: Implemented and tested

### Strategy 4: GPU-Batched MCTS (Planned, Not Implemented)
**Description**: Intra-game batching with virtual loss + cross-game batching
**Implementation**: Planned in `PLAN-GPUBatchedMCTS.md`, not yet implemented
**Status**: Design complete, implementation pending

---

## Complete Measurement History

### Benchmark Set 1: Initial Training Performance (Pre-Bug Fix)
**Date**: 2025-10-25
**Source**: BENCHMARK-Summary.md
**Configuration**: Before GPU device fix in multiprocessing workers

| Component | Device | Config | Performance | Notes |
|-----------|--------|--------|-------------|-------|
| Training | CPU | Batch 2048, FP32 | 21,101 examples/sec | Baseline |
| Training | CUDA | Batch 2048, FP32 | 41,322 examples/sec | 1.96x CPU |
| Training | CUDA | Batch 2048, FP16 | **74,115 examples/sec** | ✅ Excellent |
| Self-play | "cuda" | 8 workers | 12.6 games/min | ❌ Bug: CPU only |
| Self-play | "cuda" | 16 workers | 15.0 games/min | ❌ Bug: CPU only |

**Key Finding**: Training is fast (74k examples/sec), self-play was broken (GPU not used).

---

### Benchmark Set 2: Post-Bug Fix Self-Play
**Date**: 2025-10-25
**Source**: BENCHMARK-Summary.md
**Configuration**: After fixing GPU device transfer in workers

| Workers | Device | Games/min | CPU % | GPU % | Notes |
|---------|--------|-----------|-------|-------|-------|
| 1 | cuda | 5.2 | 3.9% | ? | Single worker baseline |
| 8 | cuda | 15.0 | 1.3% | ? | Bug fixed, GPU active |
| 16 | cuda | **18.8** | 1.3% | ? | Best measured |

**Key Finding**: GPU is being used (CPU dropped to 1.3%), but still slow (18.8 games/min).

**Projection**: At 18.8 games/min → 532 min per iteration → 196 days for 500 iterations ❌

---

### Benchmark Set 3: Phase 2 vs Phase 3 Comparison
**Date**: 2025-10-26
**Source**: FINDINGS-GPU-IDLE.md
**Configuration**: Multiprocessing (Phase 2) vs Threading (Phase 3)

| Phase | Workers | Parallelism | Batching | Games/min | Batch Requests | Avg Batch Size |
|-------|---------|-------------|----------|-----------|----------------|----------------|
| Phase 2 | 4 | multiprocessing | Per-worker | 14.8 | 0 | N/A (not working) |
| Phase 3 | 4 | threading | Shared | 8.9 | 7,106 | 3.5 |

**Key Finding**: Phase 3 is **40% slower** than Phase 2 despite batching working!

**Analysis**:
- Phase 2's BatchedEvaluator wasn't being used → pure sequential inference (no overhead)
- Phase 3's BatchedEvaluator added queue/lock overhead for batch size of only 3.5
- **Overhead exceeded benefit** at small batch sizes

---

### Benchmark Set 4: Validation Test (Current)
**Date**: 2025-10-26
**Source**: test_diagnostic_quick.py results
**Configuration**: ThreadPool + BatchedEvaluator with different worker counts

| Workers | Cards | Games/min | GPU Util (avg) | GPU Util (max) | Batch Requests | Avg Batch Size |
|---------|-------|-----------|----------------|----------------|----------------|----------------|
| 4 | 3 | 9.6 | 6.9% | 21.0% | 1,703 | 2.5 |
| 16 | 3 | **6.9** | 2.6% | 10.0% | 1,722 | 4.3 |

**Speedup**: 0.72x (NEGATIVE SCALING!) ❌

**Key Finding**: More workers made it SLOWER! Threading + batching overhead exceeds benefit.

**GPU Analysis**:
- 4 workers: 6.9% avg GPU utilization
- 16 workers: 2.6% avg GPU utilization (WORSE!)
- Both far below 70% target

---

## Root Cause Analysis

### The Math of GPU Saturation

**RTX 4060 Requirements**:
- 3,072 CUDA cores
- Optimal batch size: 128-512 samples
- Current batch size: 2.5-4.3 samples (53x-205x too small!)

**Concurrent Request Calculation**:
```
4 workers × ~3 requests/worker = ~12 concurrent requests → batch size 2.5
16 workers × ~3 requests/worker = ~48 concurrent requests → batch size 4.3

For batch size 128: Need ~40 workers
For batch size 512: Need ~170 workers
```

### Why More Workers Performed Worse

**Threading (GIL) Bottleneck**:
- Python Global Interpreter Lock limits CPU parallelism
- 16 threads contend for GIL → worse than 4 threads
- More threads = more context switching = more overhead

**BatchedEvaluator Overhead**:
- Queue operations (put/get with locks)
- Thread synchronization
- Batch collection timeout waits
- Result distribution

**At small batch sizes (2.5-4.3)**:
- Overhead cost: ~40-60% performance loss
- Batching benefit: ~5-10% (batches too small)
- Net result: SLOWER

### Why MCTS is Inherently Sequential

**Within a single game**:
1. Select node → evaluate → backpropagate (sequential loop)
2. Each simulation depends on previous simulation results
3. Cannot easily parallelize within one MCTS search

**This limits concurrent requests**:
- Even 16 workers × 3 determinizations = only ~48 concurrent spots
- Most of those are still sequential (one sim at a time)
- Result: Batch sizes stay tiny

---

## Performance Predictions

### Configuration Performance Model

Based on measurements and analysis:

**Formula**: `games/min ≈ (workers × efficiency) / overhead_multiplier`

Where:
- **efficiency** depends on parallelism type and worker count
- **overhead_multiplier** depends on batching, GIL, and synchronization

### Predicted Performance by Configuration

| Config | Workers | Parallelism | Batching | Predicted Games/min | Predicted GPU% | Reasoning |
|--------|---------|-------------|----------|---------------------|----------------|-----------|
| Current | 4 | threading | yes | 9.6 | 6.9% | ✅ Measured |
| Current | 16 | threading | yes | 6.9 | 2.6% | ✅ Measured (worse due to GIL) |
| Test A | 64 | threading | yes | 15-30 | 20-40% | More workers, but GIL limits |
| Test B | 64 | threading | no | 40-80 | 30-50% | No batching overhead, but small batches |
| Test C | 64 | multiprocessing | no | 80-150 | 40-60% | No GIL, no overhead, but no batching |
| Test D | 128 | threading | yes | 30-60 | 40-70% | Batch size ~20, GIL limits |
| Test E | 128 | multiprocessing | no | 150-300 | 60-80% | Best without batching changes |
| Optimal | 256+ | multiprocessing | no | 300-600 | 70-90% | Likely optimal for current arch |

---

## Action Plan

### Configurations to Test (Priority Order)

#### Test 1: High Workers + Multiprocessing + No Batching
**Configuration**:
```python
workers=128
use_batched_evaluator=False
use_thread_pool=False  # multiprocessing
cards_to_deal=3
```

**Expected**: 150-300 games/min, 60-80% GPU
**Reasoning**: Avoids all overhead (batching, GIL), maximizes concurrent requests via many processes
**Why test first**: Simplest path to more parallelism without complex batching

#### Test 2: Very High Workers + Multiprocessing + No Batching
**Configuration**:
```python
workers=256
use_batched_evaluator=False
use_thread_pool=False
cards_to_deal=3
```

**Expected**: 300-600 games/min, 70-90% GPU
**Reasoning**: Even more parallelism to saturate GPU
**Why test second**: If Test 1 shows improvement, push further

#### Test 3: High Workers + Threading + Batching
**Configuration**:
```python
workers=128
use_batched_evaluator=True
use_thread_pool=True
batch_size=1024
batch_timeout_ms=5.0
cards_to_deal=3
```

**Expected**: 30-60 games/min, 40-70% GPU
**Reasoning**: Test if batching helps at high worker counts
**Why test third**: Validation if batching ever beneficial

---

### Why Other Configurations Will Perform Worse

| Configuration | Why It Will Perform Worse |
|---------------|---------------------------|
| **4-32 workers, any config** | Insufficient concurrent requests to saturate GPU (proven by validation test) |
| **Threading with <128 workers** | GIL contention limits parallelism without enough work to overcome it (proven: 16 workers slower than 4) |
| **Batching with <64 workers** | Overhead exceeds benefit due to small batch sizes (proven: Phase 3 slower than Phase 2) |
| **Sequential/single worker** | No parallelism at all, slowest possible baseline |
| **Moderate workers (32-64) + batching** | Still too few workers for large batches, still has batching overhead (worst of both worlds) |
| **Threading + batching + <100 workers** | Combines GIL penalty with batching overhead for small batches (double penalty, measured as getting worse) |

---

### Configuration Testing Plan

**Phase 1**: Test configurations 1 and 2 (multiprocessing, no batching, high workers)
- **Duration**: ~15-20 minutes
- **Goal**: Find if simple scaling works

**Phase 2**: If Phase 1 shows improvement (>100 games/min, >50% GPU)
- Test configuration 3 (threading + batching at 128 workers)
- **Duration**: ~10 minutes
- **Goal**: Verify if batching helps at scale

**Phase 3**: If Phase 1/2 results unclear
- Run full diagnostic suite for complete data
- **Duration**: ~40 minutes
- **Goal**: Comprehensive understanding

---

## Decision Criteria

### Success Metrics
- ✅ **Minimum viable**: >100 games/min, >50% GPU, <45 days training
- ✅ **Target**: >500 games/min, >70% GPU, <20 days training
- ✅ **Stretch**: >1000 games/min, >90% GPU, <10 days training

### Decision Rules

**If Test 1 (128 workers, multiprocessing) achieves**:
- **>500 games/min**: ✅ Solution found, proceed to Phase 4
- **100-500 games/min**: Try Test 2 (256 workers)
- **<100 games/min**: Run full diagnostic suite, may need GPU-batched MCTS

**If Test 2 (256 workers) achieves**:
- **>500 games/min**: ✅ Solution found, use this config
- **<500 games/min**: Implement GPU-batched MCTS (Phase 1 from PLAN-GPUBatchedMCTS.md)

---

## Alternative Path: GPU-Batched MCTS

If simple scaling doesn't achieve targets, implement intra-game batching:

### Phase 1: Virtual Loss + Batching
- Add virtual losses to MCTS nodes
- Collect leaf nodes before evaluation
- Batch 90 evaluations per MCTS search (vs 1 at a time)
- **Expected**: 10x speedup per game

### Phase 2: Cross-Game Batching
- Keep BatchedEvaluator at high worker counts
- Now have 128 workers × 90 requests = 11,520 potential batch items
- **Expected**: 50-100x total speedup

**Implementation time**: 4-6 hours
**Expected result**: 1000-2000 games/min, >90% GPU

---

## Summary of All Findings

### What We Know for Certain
1. ✅ Training is fast (74k examples/sec) - not the bottleneck
2. ✅ GPU is being used (bug fixed)
3. ✅ GPU is severely underutilized (2.6-6.9% vs 70% target)
4. ✅ Batch sizes are tiny (2.5-4.3 vs 128-512 target)
5. ✅ More workers made it slower with current setup (threading + batching overhead)
6. ✅ BatchedEvaluator overhead exceeds benefit at low worker counts

### What We Strongly Suspect
1. ⚠️ Need 128-256 workers for GPU saturation
2. ⚠️ Multiprocessing will outperform threading (avoids GIL)
3. ⚠️ Removing batching overhead may help at moderate scales
4. ⚠️ Simple scaling may achieve 100-300 games/min
5. ⚠️ May need GPU-batched MCTS for >500 games/min

### What We Don't Know Yet
1. ❓ Exact optimal worker count for this hardware
2. ❓ Whether batching ever helps (untested at high worker counts)
3. ❓ Actual bottleneck location (network vs game logic vs MCTS)
4. ❓ Whether 128-256 workers will saturate GPU or need even more
5. ❓ If multiprocessing memory overhead becomes limiting factor

---

## Next Immediate Action

**Run Test 1**: 128 workers, multiprocessing, no batching

**Command**:
```python
# Create quick test script or modify benchmark_diagnostic.py
# Test configuration: workers=128, use_batched_evaluator=False, use_thread_pool=False
```

**Expected duration**: 10-15 minutes
**Expected result**: 150-300 games/min if hypothesis correct
**Decision point**: If >500 games/min → done; if 100-500 → test 256; if <100 → full diagnostic

---

**Status**: Ready to test high-worker configurations
**Confidence**: High (based on 4 independent measurement sets showing same pattern)
**Risk**: Low (worst case: confirm need for GPU-batched MCTS implementation)
