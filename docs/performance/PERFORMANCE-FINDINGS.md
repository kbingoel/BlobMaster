# BlobMaster Performance Findings - Executive Summary

**Date**: 2025-10-26
**Status**: Performance investigation complete
**Hardware**: NVIDIA RTX 4060 8GB, AMD Ryzen 9 7950X (16 cores), 128GB DDR5

---

## TL;DR - Key Findings

**Best Configuration Found**: 32 workers, multiprocessing, no batching
**Performance**: 43-80 games/min (depending on MCTS complexity)
**Training Timeline**: ~50 days for 500 iterations (10,000 games each)
**Recommendation**: Use baseline for now; GPU-batched MCTS needed for >500 games/min target

---

## Investigation Overview

After extensive performance testing across multiple architectures (threading, multiprocessing, GPU batching, GPU inference server), we discovered that:

1. **GPU Server Architecture Failed**: 3-5x slower than baseline due to small batch sizes (10-13 avg vs 128+ target)
2. **Threading + Batching Failed**: GIL contention and batching overhead exceeded benefits
3. **Simple Multiprocessing Wins**: 32 workers with per-worker networks achieves best results
4. **GPU Severely Underutilized**: Only 15-20% utilization due to MCTS sequential nature

---

## Performance Results Summary

### Winning Configuration: Baseline Multiprocessing (Phase 2)

| MCTS Config | Sims/Move | Games/Min | Sec/Game | GPU Util | Training Time |
|-------------|-----------|-----------|----------|----------|---------------|
| Light       | 40        | 80.8      | 0.74     | ~20%     | 29 days       |
| Medium      | 90        | 43.3      | 1.38     | ~15%     | 54 days       |
| Heavy       | 250       | 25.0      | 2.40     | ~15%     | 93 days       |

**Configuration**: `workers=32, device='cuda', use_batching=False`

### Failed Approaches

| Approach | Workers | Performance | vs Baseline | Why It Failed |
|----------|---------|-------------|-------------|---------------|
| GPU Server | 32 | 10-20 games/min | 0.25x (4x slower) | Batch size only 10-13, queue overhead |
| Threading + Batching | 128 | 8-9 games/min | 0.19x (5x slower) | GIL contention + batching overhead |
| High Worker Multiproc | 60 | 1.3 games/min | 0.02x (52x slower!) | GPU thrashing, CUDA context switching |

---

## Root Cause Analysis

### Why GPU Utilization is Low

**The MCTS Problem**:
- MCTS is inherently sequential (each simulation depends on previous results)
- Workers can only issue ~1-3 concurrent inference requests at a time
- 32 workers × 3 requests = ~96 concurrent requests maximum
- This produces batch sizes of 2-13 (vs 128-512 needed for GPU saturation)

**GPU Requirements**:
- RTX 4060 has 3,072 CUDA cores
- Needs batches of 128-512 to saturate
- Current batch sizes: 53x-205x too small

### Why More Workers Made Things Worse

**Tested 60 workers, multiprocessing**:
- Result: 1.3 games/min (52x SLOWER!)
- GPU Utilization: 99.2% (fully saturated)
- Problem: GPU became serialization bottleneck
  - 60 processes × 20MB network = 1.2GB VRAM for duplicate models
  - CUDA context switching overhead between processes
  - Processes waiting in GPU queue instead of computing

**Sweet Spot**: 32 workers balances parallelism without GPU thrashing

---

## Recommendations

### Short Term: Use Baseline Configuration ✅

**For current training (Phase 4)**:
```bash
python ml/train.py --workers 32 --device cuda
```

**Expected performance**:
- Medium MCTS (3 det × 30 sims): 43.3 games/min
- Training time: ~54 days for 500 iterations
- GPU utilization: 15-20% (acceptable for this architecture)

**Why this is acceptable**:
- Proven stable and correct
- No additional complexity
- Sufficient for initial model training
- Can iterate on model improvements while training

### Medium Term: Optimize Within Baseline

**Without changing architecture**:
1. **Progressive MCTS**: Start with light MCTS (2 det × 20 sims) for early iterations
2. **Adaptive complexity**: Increase MCTS simulations as model improves
3. **Early stopping**: Stop training if ELO plateaus

**Expected improvement**: 30-50% faster effective training

### Long Term: GPU-Batched MCTS (If Needed)

**If 54 days is unacceptable**:

Implement intra-game MCTS batching with virtual loss:
- Batch all 90 leaf evaluations per MCTS search (vs 1 at a time)
- Expected: 10x speedup per game
- Implementation time: 4-6 hours
- See [docs/performance/PLAN-GPUBatchedMCTS.md](docs/performance/PLAN-GPUBatchedMCTS.md)

**Expected result**: 300-600 games/min, 70-90% GPU utilization, <15 day training

---

## Lessons Learned

### What Worked
- ✅ Systematic benchmarking across configurations
- ✅ Measuring actual performance vs predictions
- ✅ Fixing bugs discovered during testing (GPU device transfer, network attributes)
- ✅ Documenting findings for future reference

### What Didn't Work
- ❌ Assuming more workers always helps (60 workers catastrophically failed)
- ❌ Batching at low worker counts (overhead > benefit)
- ❌ Threading for CPU-bound Python workloads (GIL limits parallelism)
- ❌ GPU server with small batches (queue overhead dominated)

### Key Insights
1. **MCTS is fundamentally sequential** - hard to parallelize within a single search
2. **Small batches worse than no batching** - overhead exceeds GPU efficiency gains
3. **Worker count sweet spot exists** - too few = underutilized, too many = thrashing
4. **Python GIL is real** - threads perform worse than processes for CPU work
5. **Measure, don't assume** - many intuitions were wrong

---

## Detailed Documentation

For in-depth analysis, see:

- **[GPU Server Investigation](docs/performance/GPU-SERVER-TEST-RESULTS.md)** - Why centralized GPU server failed
- **[Windows Performance Tests](docs/performance/WINDOWS-TEST-RESULTS-ANALYSIS.md)** - Comprehensive configuration testing
- **[Performance Master Doc](docs/performance/TRAINING-PERFORMANCE-MASTER.md)** - Complete measurement history
- **[GPU Idle Findings](docs/performance/FINDINGS-GPU-IDLE.md)** - Phase 2 vs Phase 3 comparison
- **[GPU-Batched MCTS Plan](docs/performance/PLAN-GPUBatchedMCTS.md)** - Future optimization strategy

**Benchmark Scripts**: [benchmarks/performance/](benchmarks/performance/)
**Test Results**: [results/](results/)
**Phase Completion Reports**: [docs/phases/](docs/phases/)

---

## Success Criteria Status

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| **Games/min** | >500 | 43-80 | ⚠️ Below target but usable |
| **GPU Utilization** | >70% | 15-20% | ❌ Low (architectural limit) |
| **Training Time** | <30 days | 54 days | ⚠️ Longer but acceptable |
| **Correctness** | 100% | 100% | ✅ All tests passing |
| **Stability** | No crashes | Stable | ✅ Production ready |

**Overall Assessment**: **Viable for training**, but performance optimization opportunities exist if needed.

---

## Next Steps

1. ✅ **Document findings** (this file)
2. ✅ **Clean up repository** (organize docs, scripts, results)
3. ⬜ **Proceed to Phase 4** - Self-Play Training Pipeline ([PLAN-Phase-4.md](PLAN-Phase-4.md))
4. ⬜ **Start 500-iteration training** with baseline configuration
5. ⬜ **Monitor progress** - Track games/min, ELO curves, training loss
6. ⬜ **Optimize if needed** - Implement GPU-batched MCTS if timeline becomes critical

---

**Investigation Duration**: ~8 hours across multiple sessions
**Scripts Created**: 12 benchmark/test scripts (~180 KB)
**Documentation**: 11 detailed analysis documents (~150 KB)
**Configurations Tested**: 15+ combinations of workers, threading, batching
**Outcome**: Clear path forward with realistic expectations
