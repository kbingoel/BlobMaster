# Session 1+2 Validation Benchmark Suite

Comprehensive parameter exploration for batch submission API and parallel MCTS expansion optimizations.

## Quick Start

### 1. Quick Validation (~30 minutes)
Test core hypotheses with reduced parameter space:
```bash
cd /home/kbuntu/Documents/Github/BlobMaster
source venv/bin/activate
python benchmarks/performance/benchmark_session1_validation.py \
  --sweep quick \
  --games 20 \
  --device cuda
```

**Output:** `benchmarks/results/session1_validation_YYYYMMDD_HHMM.csv`

### 2. Visualize Results
Generate plots and summary report:
```bash
python benchmarks/performance/visualize_session1_results.py \
  benchmarks/results/session1_validation_YYYYMMDD_HHMM.csv
```

**Output:**
- `*_batch_size.png` - Parallel batch size vs performance
- `*_worker_scaling.png` - Worker scaling efficiency
- `*_heatmap.png` - 2D interaction matrix
- `*_mcts_comparison.png` - MCTS config comparison
- `*_summary.txt` - Text summary with recommendations

---

## Full Validation (~4-6 hours)

Comprehensive sweep of all parameters:
```bash
python benchmarks/performance/benchmark_session1_validation.py \
  --sweep all \
  --games 50 \
  --device cuda
```

**Total benchmarks:**
- ~1,750 games tested
- ~5-6 hours on RTX 4060
- All 5 sweeps included

---

## Individual Sweeps

### Sweep 1: Find Optimal Batch Size (Most Important!)
```bash
python benchmarks/performance/benchmark_session1_validation.py \
  --sweep batch_size \
  --games 50
```

Tests `parallel_batch_size` = [5, 10, 15, 20, 25, 30] at 32 workers.
**Duration:** ~30 minutes (300 games)

### Sweep 2: Worker Scaling
```bash
python benchmarks/performance/benchmark_session1_validation.py \
  --sweep workers \
  --games 50
```

Tests workers = [1, 4, 8, 16, 24, 32, 40, 48] with optimal batch size.
**Duration:** ~40 minutes (400 games)

### Sweep 3: 2D Interaction Matrix
```bash
python benchmarks/performance/benchmark_session1_validation.py \
  --sweep 2d \
  --games 30
```

Tests workers × batch_size combinations for heatmap.
**Duration:** ~90 minutes (750 games)

### Sweep 4: MCTS Configuration
```bash
python benchmarks/performance/benchmark_session1_validation.py \
  --sweep mcts \
  --games 50
```

Tests Light/Medium/Heavy MCTS with optimal settings.
**Duration:** ~15 minutes (150 games)

### Sweep 5: Batch Timeout
```bash
python benchmarks/performance/benchmark_session1_validation.py \
  --sweep timeout \
  --games 50
```

Tests batch timeout = [3, 5, 8, 10, 15]ms.
**Duration:** ~25 minutes (250 games)

---

## Expected Results

### Baseline (Before Optimizations)
- **36.7 games/min** @ 32 workers, Medium MCTS
- Worker scaling efficiency: 64% @ 32 workers
- GPU utilization: 15-20%

### Session 1+2 Initial Result
- **63.6 games/min** @ 32 workers, Medium MCTS, `parallel_batch_size=10`
- **1.73x speedup**
- Room for improvement with optimal parameters

### Target After Validation
- **110+ games/min** (3x speedup minimum)
- Optimal `parallel_batch_size` likely 15-25
- Worker scaling efficiency ≥70% @ 32 workers
- Training time: <50 days (down from 136 days)

---

## What to Look For

### 1. Optimal Parallel Batch Size
- **Expected:** 15-25 for 32 workers
- **Why:** Balances GPU batch size vs MCTS overhead
- **Too low (5):** Underutilizes batch submission API
- **Too high (30+):** MCTS tree bottleneck, virtual loss conflicts

### 2. Improved Worker Scaling
- **Baseline:** 64% efficiency @ 32 workers
- **Target:** 75%+ efficiency @ 32 workers
- **Stretch:** Viable at 40-48 workers (was CUDA OOM before)

### 3. 2D Interaction Pattern
- **Expected:** Diagonal "ridge" of optimal configs
- **Pattern:** Higher workers → larger optimal batch size
  - 8 workers: optimal ~10-15
  - 32 workers: optimal ~20-25
  - 40 workers: optimal ~25-30

### 4. Consistent MCTS Speedup
- **Expected:** All configs benefit ~2x
- **Light MCTS:** Most benefit (more leaves expanded)
- **Heavy MCTS:** Still compute-bound

### 5. Batch Timeout Insensitivity
- **Expected:** 5-10ms all similar
- **Optimal:** Likely 5-8ms (slightly better than 10ms default)

---

## Interpreting Results

### Success Criteria

**Minimum (Acceptable):**
- ✅ 2x speedup (73+ games/min)
- ✅ Optimal batch size identified
- ✅ Worker scaling ≥70% @ 32 workers

**Target (Expected):**
- ✅ 2.5x speedup (92+ games/min)
- ✅ Viable at 40 workers
- ✅ Training time <60 days

**Stretch (Optimistic):**
- ✅ 3x speedup (110+ games/min)
- ✅ 48 workers functional
- ✅ Training time <45 days

### Next Steps Based on Results

**If ≥3x speedup achieved:**
- 🎉 Excellent! Proceed to Sessions 3-5 for additional gains
- 🎉 Could achieve 4-5x total speedup
- 🎉 Update config defaults and documentation

**If 2-3x speedup achieved:**
- ✅ Good progress! Proceed to Sessions 3-5
- ✅ Mixed precision + torch.compile for final boost
- ✅ Expected final speedup: 3-4x

**If <2x speedup achieved:**
- ⚠️ Debug: Check batch sizes in logs
- ⚠️ Profile: Identify bottleneck
- ⚠️ Consider GPU server architecture (Phase 3.5)

---

## Comparison to PERFORMANCE-FINDINGS.md

### What's Different
| Aspect | Before (Sequential) | After (Parallel) |
|--------|-------------------|------------------|
| MCTS expansion | Sequential (1 at a time) | Parallel (10-30 leaves) |
| Batch submission | Per-request | Batch API (`evaluate_many()`) |
| GPU batch sizes | 2-13 (too small) | 128-320+ (much better) |
| Worker scaling | 64% @ 32 workers | 75%+ expected |
| Max workers | 32 (48 failed OOM) | 40-48 possibly viable |

### What's Similar
- Hardware: RTX 4060 8GB, Ryzen 9 7950X
- MCTS configs: Light/Medium/Heavy
- Methodology: games/min, worker scaling

### New Opportunities
- `parallel_batch_size` parameter (new)
- Better GPU utilization (30-50%+ expected)
- Higher worker counts viable

### New Risks
- Virtual loss conflicts at very high batch sizes
- Memory pressure at high workers × batch sizes
- Diminishing returns if GPU becomes bottleneck

---

## Files Created

### Benchmark Scripts
- `benchmark_session1_validation.py` - Main benchmark suite
- `visualize_session1_results.py` - Plot generator

### Documentation
- `SESSION1-VALIDATION-PLAN.md` - Detailed sweep design
- `README_SESSION1.md` - This file

### Expected Outputs
- `session1_validation_*.csv` - Raw benchmark data
- `*_batch_size.png` - Batch size analysis
- `*_worker_scaling.png` - Scaling efficiency
- `*_heatmap.png` - 2D parameter interaction
- `*_mcts_comparison.png` - Config comparison
- `*_summary.txt` - Text report with recommendations

---

## Troubleshooting

### CUDA Out of Memory
If you get CUDA OOM at high worker counts:
```bash
# Reduce workers or batch size
python benchmarks/performance/benchmark_session1_validation.py \
  --sweep quick --games 10  # Smaller test
```

### Slow Benchmark
For faster iteration:
```bash
# Use --games 20 instead of 50
python benchmarks/performance/benchmark_session1_validation.py \
  --sweep batch_size --games 20
```

### Missing Matplotlib
If visualization fails:
```bash
pip install matplotlib numpy
```

---

## After Validation

### 1. Update Config Defaults
Edit `ml/config.py` with optimal values:
```python
# MCTS parallelization
use_parallel_expansion: bool = True
parallel_batch_size: int = <optimal_value>  # From Sweep 1
```

### 2. Document Results
Create `docs/performance/OPTIMIZATION-RESULTS.md`:
- Performance comparison table
- Updated training time estimates
- Parameter recommendations

### 3. Update CLAUDE.md
Add optimal configuration to training commands:
```python
engine = SelfPlayEngine(
    num_workers=32,
    use_parallel_expansion=True,
    parallel_batch_size=<optimal>,  # From validation
    ...
)
```

### 4. Proceed to Session 3
If target not yet met, continue with:
- **Session 3:** Mixed Precision Inference (+10-30%)
- **Session 4:** torch.compile Integration (+10-30%)
- **Session 5:** Linux Runtime Tuning (+5-15%)

---

## Questions?

See detailed sweep design: `docs/performance/SESSION1-VALIDATION-PLAN.md`

For implementation details: `docs/performance/OPTIMIZATION-PLAN.md`
