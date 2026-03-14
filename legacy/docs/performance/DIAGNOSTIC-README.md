# BlobNet Training Performance Diagnostic Suite

**Created**: 2025-10-26
**Purpose**: Systematically diagnose and fix slow training performance and low GPU/CPU utilization

---

## Problem Statement

Initial benchmarks showed:
- **Training speed**: 15-19 games/min (expected: 500-2000 games/min)
- **GPU utilization**: 5-10% (target: >70%)
- **CPU utilization**: Low (both CPU and GPU mostly idle)
- **Estimated training time**: 196 days (unacceptable)

**Root cause hypothesis**: Insufficient parallelism - only 4 workers generating ~8-16 concurrent requests, but GPU needs 128-512 batch sizes for saturation.

---

## Diagnostic Tools

### 1. `benchmark_diagnostic.py`

**Comprehensive benchmark suite** that systematically tests:

- **Worker scaling**: 4, 8, 16, 32, 64, 128 workers
- **Batching strategies**: BatchedEvaluator vs direct inference
- **Parallelism**: ThreadPoolExecutor vs multiprocessing
- **Game complexity**: 3, 5, 8 cards dealt

**Metrics tracked**:
- Games/min throughput
- GPU utilization (avg and max)
- GPU memory usage
- Batch sizes
- Examples per game

**Usage**:
```bash
cd ml
python benchmark_diagnostic.py
```

**Output**:
- `benchmark_diagnostic_results.csv` - Raw data
- `benchmark_diagnostic_results.json` - Detailed data
- Console summary with recommendations

**Duration**: ~30-40 minutes for full suite

---

### 2. `visualize_diagnostic.py`

**Visualization and analysis** of benchmark results.

Creates plots:
- Worker scaling (games/min, GPU%, batch size)
- Batching comparison (batched vs direct)
- Game complexity impact
- Worker efficiency

**Usage**:
```bash
python visualize_diagnostic.py
```

**Output**:
- `benchmark_worker_scaling.png`
- `benchmark_batching_comparison.png`
- `benchmark_game_complexity.png`
- Console analysis with recommendations

**Requirements**:
- matplotlib
- seaborn
- pandas

Install with:
```bash
pip install matplotlib seaborn pandas
```

---

### 3. `profile_selfplay.py`

**Code profiling** to identify bottlenecks.

Three profiling approaches:
1. **cProfile**: Function-level profiling
2. **Manual timing**: Component breakdown
3. **Overhead analysis**: BatchedEvaluator overhead

**Usage**:
```bash
python profile_selfplay.py
```

**Output**:
- `profile_w16_g10_c3.prof` - Profile data
- Console output with top functions by time
- Overhead analysis

**Duration**: ~10 minutes

View profile later with:
```bash
python -m pstats profile_w16_g10_c3.prof
```

---

### 4. `run_full_diagnostics.py`

**Master script** that runs all diagnostics in sequence.

**Usage**:
```bash
python run_full_diagnostics.py
```

**What it does**:
1. Runs profiling analysis
2. Asks for confirmation to run full benchmark (takes time)
3. Runs comprehensive benchmark suite
4. Generates visualizations
5. Provides summary and next steps

**Duration**: ~30-50 minutes total

---

## Expected Findings

Based on analysis in `FINDINGS-GPU-IDLE.md`, we expect:

### Worker Scaling
- **4 workers**: ~15-20 games/min, <10% GPU, batch size ~3-5
- **16 workers**: ~50-100 games/min, ~20-30% GPU, batch size ~10-20
- **32 workers**: ~150-300 games/min, ~40-60% GPU, batch size ~50-100
- **64 workers**: ~300-600 games/min, ~70-90% GPU, batch size ~128-256
- **128 workers**: ~500-1000 games/min, >90% GPU, batch size ~256-512

### Batching Impact
- **At 4 workers**: Batching may be SLOWER due to synchronization overhead
- **At 16+ workers**: Batching should provide 2-5x speedup
- **At 64+ workers**: Batching should provide 10-20x speedup

### Bottlenecks
Profiling should reveal time spent in:
- MCTS search
- Network inference
- Queue operations (if batching)
- Game logic
- Determinization sampling

---

## Decision Framework

After running diagnostics, answer these questions:

### 1. What worker count maximizes GPU utilization?
- Target: >70% GPU utilization
- If not achieved at 128 workers, consider hybrid approach

### 2. Does batching help at the optimal worker count?
- Compare batched vs direct at optimal worker count
- If batching overhead exceeds benefit, consider removing it

### 3. What's the realistic training timeline?
- Use optimal config's games/min to calculate
- Training target: <30 days for 500 iterations

### 4. Where are the actual bottlenecks?
- Profiling should show top time consumers
- If network calls dominate: batching helps
- If queue operations dominate: reduce synchronization
- If game logic dominates: optimize game code

---

## Recommended Actions

Based on expected findings:

### If GPU utilization is low (<40%)
- [ ] Increase workers to 32, 64, or 128
- [ ] Increase determinizations from 3 → 10-16
- [ ] Increase simulations if GPU still underutilized

### If batch sizes are small (<128)
- [ ] More workers (primary solution)
- [ ] More concurrent work per worker (determinizations)
- [ ] Lower batch timeout (collect batches faster)

### If batching adds overhead
- [ ] Disable BatchedEvaluator at low worker counts
- [ ] Use direct inference if workers < 16
- [ ] Re-enable batching only at high worker counts

### If training time still >30 days
- [ ] Consider reducing games per iteration (10k → 5k)
- [ ] Consider fewer iterations (500 → 300)
- [ ] Consider hybrid approach (GPU batching + intra-game batching)

---

## Quick Start

**Fastest path to answers:**

```bash
# 1. Quick benchmark (4, 16, 32 workers only)
cd ml
python benchmark_diagnostic.py

# 2. Visualize results
python visualize_diagnostic.py

# 3. Check PNG files and recommendations
# Look at: benchmark_worker_scaling.png

# 4. If needed, profile for bottlenecks
python profile_selfplay.py
```

**Full diagnostic (comprehensive):**

```bash
cd ml
python run_full_diagnostics.py
```

---

## Success Criteria

After implementing fixes based on diagnostics:

- [x] Understand exact bottleneck (with data)
- [ ] Achieve >500 games/min throughput
- [ ] Achieve >70% GPU utilization
- [ ] Achieve avg batch size >128
- [ ] Training time estimate <30 days
- [ ] Both CPU and GPU actively utilized

---

## Files Created

### Diagnostic Tools
- `ml/benchmark_diagnostic.py` - Main benchmark suite
- `ml/visualize_diagnostic.py` - Visualization and analysis
- `ml/profile_selfplay.py` - Code profiling
- `ml/run_full_diagnostics.py` - Master runner script

### Documentation
- `DIAGNOSTIC-README.md` - This file

### Output Files (generated when run)
- `benchmark_diagnostic_results.csv` - Benchmark data
- `benchmark_diagnostic_results.json` - Detailed data
- `benchmark_worker_scaling.png` - Worker scaling plots
- `benchmark_batching_comparison.png` - Batching comparison
- `benchmark_game_complexity.png` - Game complexity analysis
- `profile_w*_g*_c*.prof` - Profiling data

---

## Integration with Phase 4

Once diagnostics are complete and optimal configuration is found:

1. **Update training config** with optimal parameters:
   - `num_workers`: Optimal worker count from benchmarks
   - `batch_size`: Optimal batch size
   - `use_batched_evaluator`: True/False based on findings
   - `use_thread_pool`: True/False based on findings

2. **Continue Phase 4 development**:
   - Session 6: Evaluation & ELO tracking
   - Session 7: Main training script & configuration

3. **Use measured performance** for planning:
   - Realistic training time estimates
   - Hardware utilization targets
   - Hyperparameter sweep feasibility

---

## Troubleshooting

### nvidia-smi not found
- Ensure NVIDIA drivers are installed
- Add NVIDIA to PATH: `C:\Program Files\NVIDIA Corporation\NVSMI`

### CUDA not available
- Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### Benchmark takes too long
- Reduce worker counts tested: edit `benchmark_diagnostic.py`, line ~470
- Reduce num_games: Change `num_games = 20` to `num_games = 10`

### Visualization fails
- Install dependencies: `pip install matplotlib seaborn pandas`
- Check CSV file exists: `benchmark_diagnostic_results.csv`

---

## Next Steps After Diagnostics

1. **Analyze results** - Review plots and recommendations
2. **Choose optimal config** - Based on data, not guessing
3. **Implement architectural fix** - Adjust workers/batching as needed
4. **Validate fix** - Re-run benchmark to confirm improvement
5. **Update Phase 4 plan** - With realistic timelines
6. **Continue development** - Complete remaining Phase 4 sessions

---

**Status**: Ready to run diagnostics
**Estimated time**: 30-50 minutes for full suite
**Expected outcome**: Data-driven decision on optimal training configuration
