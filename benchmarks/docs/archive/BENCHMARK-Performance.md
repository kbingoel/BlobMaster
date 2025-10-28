# BENCHMARK-Performance.md
# Performance Benchmarking for BlobMaster Training Pipeline

**Status**: Ready to Execute
**Hardware**: AMD 7950X (16 cores/32 threads) + RTX 4060 (8GB VRAM)
**Phase**: 4 Sessions 1-4 Complete (Self-play, Replay Buffer, Trainer implemented)

---

## Executive Summary

**Goal**: Measure real-world performance of the training pipeline on your hardware to:
1. Validate theoretical estimates (45 min/iteration for 500 iterations = 15 days training)
2. Identify actual bottlenecks (is self-play really 3x slower than training?)
3. Make evidence-based decision: implement GPU-batched MCTS now or later?

**Why now?** All core components exist (self-play, training), only evaluation/orchestration missing (not needed for perf benchmarks).

---

## Current Implementation Status

### ✅ Ready for Benchmarking:
- **SelfPlayWorker**: Single-threaded game generation with MCTS
- **SelfPlayEngine**: Parallel generation with multiprocessing (16 workers)
- **ReplayBuffer**: Experience storage and batch sampling
- **NetworkTrainer**: GPU training with loss computation
- **BlobNet**: Neural network with CPU/GPU support

### ❌ Not Yet Implemented:
- **Evaluation System**: Model vs model tournaments (Session 6)
- **Training Pipeline**: Orchestration of full loop (Session 5)
- **Main Script**: `ml/train.py` entry point (Session 7)

**Verdict**: Can benchmark individual phases, but not full automated training loop yet.

---

## What to Benchmark

### Benchmark 1: Self-Play Speed
**Script**: `ml/benchmark_selfplay.py`

**Test Configurations**:
- **Workers**: [1, 4, 8, 16, 32] (test hyperthreading benefit)
- **MCTS settings**:
  - Light: (2 determinizations, 20 sims) = 40 tree searches/move
  - Medium: (3 determinizations, 30 sims) = 90 tree searches/move
  - Heavy: (5 determinizations, 50 sims) = 250 tree searches/move
- **Games**: 50 per configuration (statistical significance)
- **Players/Cards**: 4 players, 5 cards (standard game)

**Metrics**:
- Games per minute
- Seconds per game
- Training examples per minute
- CPU utilization (%)

**Expected Results**:
- 16 workers: 200-400 games/min (theoretical: 320 games/min)
- 32 workers: Likely 10-15% faster than 16 (hyperthreading limited benefit)
- MCTS settings: Heavy 6x slower than Light

**Output**: CSV + pretty table

---

### Benchmark 2: GPU Training Speed
**Script**: `ml/benchmark_training.py`

**Test Configurations**:
- **Device**: ['cpu', 'cuda']
- **Precision**: ['fp32', 'fp16'] (test mixed precision speedup)
- **Batch size**: [256, 512, 1024, 2048] (find optimal for 8GB VRAM)
- **Data**: 100,000 synthetic training examples
- **Epochs**: 5 (enough to measure steady-state throughput)

**Metrics**:
- Batches per second
- Examples per second
- GPU utilization (%)
- VRAM usage (GB)
- Loss convergence (sanity check)

**Expected Results**:
- FP16 should be 1.5-2x faster than FP32
- Optimal batch size: 512-1024 (VRAM limited)
- GPU training: 10-50x faster than CPU

**Output**: JSON + performance table

---

### Benchmark 3: End-to-End Iteration Simulation
**Script**: `ml/benchmark_iteration.py`

**What it does**:
1. **Self-Play Phase**: Generate 1,000 games (1/10th of full iteration)
2. **Training Phase**: Train for 10 epochs on generated examples
3. *Skip Evaluation* (not implemented yet)

**Metrics**:
- Self-play time (minutes)
- Training time (minutes)
- Total iteration time (minutes)
- **Projected full-scale time**: (1,000 games → 10,000 games)

**Expected Results**:
- Self-play: 3-5 minutes (scaled to 30-50 min for 10k games)
- Training: 1-2 minutes (scaled to 10-20 min for 10 epochs)
- Confirms: Self-play is 2-3x slower than training (bottleneck!)

**Output**: Phase breakdown + projection

---

### Benchmark 4: Report Aggregation
**Script**: `ml/benchmark_report.py`

**What it does**:
- Reads results from benchmarks 1-3
- Generates markdown report with:
  - Performance baseline summary
  - Estimated full training run time (500 iterations)
  - Bottleneck identification
  - Optimization recommendations

**Output**: `BENCHMARK-Results.md`

---

## Benchmark Scripts Implementation

### File 1: `ml/benchmark_selfplay.py` (~150 lines)

```python
"""
Benchmark self-play game generation performance.

Tests different worker counts and MCTS configurations to find optimal settings.
"""

import argparse
import time
from ml.training import SelfPlayEngine
from ml.network.model import BlobNet
from ml.network.encode import StateEncoder, ActionMasker

# Test configurations
WORKER_COUNTS = [1, 4, 8, 16, 32]
MCTS_CONFIGS = [
    ("light", 2, 20),
    ("medium", 3, 30),
    ("heavy", 5, 50),
]

def benchmark_config(workers, num_det, num_sims, num_games=50):
    """Run benchmark for specific configuration."""
    # Implementation details...
    pass

def main():
    # Run all configurations
    # Output results table
    pass
```

### File 2: `ml/benchmark_training.py` (~150 lines)

```python
"""
Benchmark GPU training performance.

Tests different batch sizes and precision modes.
"""

import torch
from ml.training import NetworkTrainer, ReplayBuffer
from ml.network.model import BlobNet

# Test configurations
BATCH_SIZES = [256, 512, 1024, 2048]
PRECISIONS = ['fp32', 'fp16']

def benchmark_training(batch_size, precision, device='cuda'):
    """Benchmark training throughput."""
    # Implementation details...
    pass
```

### File 3: `ml/benchmark_iteration.py` (~100 lines)

```python
"""
Benchmark full training iteration (self-play + training).

Simulates one iteration at 1/10th scale.
"""

def benchmark_iteration():
    """Run self-play + training, measure time."""
    # 1. Self-play: 1,000 games
    # 2. Training: 10 epochs
    # 3. Report breakdown
    pass
```

### File 4: `ml/benchmark_report.py` (~50 lines)

```python
"""
Aggregate benchmark results and generate report.
"""

def generate_report():
    """Create BENCHMARK-Results.md with findings."""
    pass
```

---

## Expected Insights

### Question 1: Is 16 or 32 workers optimal?
- **If 32 workers is <15% faster**: Use 16 (cleaner, less CPU overhead)
- **If 32 workers is >20% faster**: Use 32 (hyperthreading helps)

### Question 2: How fast is self-play really?
- **Target**: 300+ games/minute (10k games in 30 min)
- **If slower**: GPU-batched MCTS becomes HIGH priority
- **If faster**: Can proceed with Phase 4 completion first

### Question 3: Is GPU training fast enough?
- **Target**: <15 min per epoch (10 epochs = 2.5 hours training)
- **If slower**: Optimize training before self-play
- **If faster**: Confirms self-play is bottleneck

### Question 4: Total training time realistic?
- **Estimate**: 500 iterations × 45 min = 375 hours = **15.6 days**
- **If >30 days**: Need to optimize before starting
- **If <10 days**: Can start training confidently

---

## Execution Plan

### Session 1: Implement Benchmarks (~2 hours)
1. Create `ml/benchmark_selfplay.py` with all worker/MCTS configs
2. Create `ml/benchmark_training.py` with batch/precision tests
3. Create `ml/benchmark_iteration.py` for end-to-end simulation
4. Create `ml/benchmark_report.py` for aggregation
5. Add CLI arguments for flexibility

### Session 2: Run Benchmarks (~1 hour)
1. Run `benchmark_selfplay.py` (30 min - many configurations)
2. Run `benchmark_training.py` (15 min - GPU tests)
3. Run `benchmark_iteration.py` (10 min - scaled iteration)
4. Monitor: CPU%, GPU%, VRAM, temperatures

### Session 3: Generate Report (~30 min)
1. Run `benchmark_report.py` to create `BENCHMARK-Results.md`
2. Review findings
3. Make decision: implement GPU-batched MCTS now or later?

---

## Decision Framework

After benchmarking, decide on GPU-batched MCTS implementation:

### Scenario A: Self-Play is SLOW (<200 games/min)
- **Action**: Implement GPU-batched MCTS **before** Phase 4 Sessions 5-7
- **Reason**: Without it, training will take >30 days (impractical)
- **Estimated speedup**: 3-6x (200 games/min → 600-1200 games/min)
- **Time investment**: ~4 hours implementation + testing

### Scenario B: Self-Play is FAST (>400 games/min)
- **Action**: Complete Phase 4 Sessions 5-7 first, GPU-batching later
- **Reason**: Training time acceptable (10-12 days), can optimize after
- **Estimated speedup**: Nice-to-have (400 → 1200+ games/min)
- **Time investment**: Defer to Phase 4.5 or Phase 5

### Scenario C: Self-Play is MEDIUM (200-400 games/min)
- **Action**: Judgment call based on patience vs optimization desire
- **Reason**: Training feasible but slow (15-25 days)
- **Recommendation**: Implement GPU-batching if you want <10 day training

---

## Benchmark Output Format

### `BENCHMARK-Results.md` (generated)

```markdown
# BlobMaster Training Pipeline Performance Benchmark

**Hardware**: AMD 7950X + RTX 4060
**Date**: 2025-10-25
**Phase**: 4 Sessions 1-4 Complete

## Self-Play Performance
| Workers | MCTS | Games/Min | Sec/Game | CPU% |
|---------|------|-----------|----------|------|
| 16      | Med  | 287       | 3.3      | 94%  |
| 32      | Med  | 312       | 3.1      | 98%  |

**Recommendation**: Use 16 workers (32 only 9% faster)

## GPU Training Performance
| Batch | Precision | Batches/Sec | GPU% | VRAM |
|-------|-----------|-------------|------|------|
| 512   | FP32      | 24          | 87%  | 4.2G |
| 512   | FP16      | 38          | 92%  | 3.1G |

**Recommendation**: Use FP16 (1.58x speedup)

## Iteration Timing (1/10th scale)
- Self-play: 3.5 min (projected: 35 min for 10k games)
- Training: 1.2 min (projected: 12 min for full training)
- **Total**: 4.7 min (projected: 47 min/iteration)

## Full Training Estimate
- 500 iterations × 47 min = **391 hours = 16.3 days**
- Bottleneck: Self-play (74% of iteration time)

## Recommendations
1. ⚠️ **GPU-Batched MCTS**: HIGH PRIORITY (would reduce to 6 days)
2. ✅ **FP16 Training**: Already fast, use mixed precision
3. ✅ **16 Workers**: Optimal for 7950X
```

---

## Success Criteria

After running benchmarks, you should have:

- ✅ Real performance numbers from your hardware
- ✅ Confidence in training time estimate
- ✅ Evidence-based decision on GPU-batched MCTS
- ✅ Optimal hyperparameter settings (workers, batch size, precision)
- ✅ Identified bottlenecks with quantified impact

---

## Next Steps

### If GPU-Batched MCTS Needed:
1. Implement batched inference in MCTS (Session 4.5)
2. Re-run benchmarks to confirm speedup
3. Continue Phase 4 Sessions 5-7

### If Current Performance Acceptable:
1. Continue Phase 4 Session 5 (Training Pipeline)
2. Continue Phase 4 Session 6 (Evaluation & ELO)
3. Continue Phase 4 Session 7 (Main Script)
4. Defer GPU-batching to Phase 4.5 (optional optimization)

---

**Ready to Execute**: Start by creating `ml/benchmark_selfplay.py` in a new session.
