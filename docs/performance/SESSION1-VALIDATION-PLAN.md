# Session 1+2 Validation Benchmark Plan

**Created:** 2025-11-05
**Purpose:** Systematically explore parameter space opened by batch submission API and parallel MCTS expansion
**Expected Duration:** 4-8 hours for full sweep

---

## Motivation

The baseline performance findings (PERFORMANCE-FINDINGS.md) tested worker scaling and MCTS configurations with **sequential MCTS** and **per-request batching**. With the new optimizations:

- **Session 1**: Batch submission API (`evaluate_many()`)
- **Session 2**: Parallel MCTS expansion (`use_parallel_expansion=True`)

We now have **new parameters to optimize** and **improved scaling potential**. This validation benchmark systematically explores this new parameter space.

---

## Baseline Context

From PERFORMANCE-FINDINGS.md (Ubuntu, before optimizations):
- **36.7 games/min** @ 32 workers, Medium MCTS (sequential)
- Worker scaling efficiency degraded beyond 16 workers (64% efficiency @ 32 workers)
- GPU severely underutilized (15-20% usage)

**Initial Session 1+2 result:**
- **63.6 games/min** @ 32 workers, Medium MCTS, `parallel_batch_size=10`
- **1.73x speedup** over baseline
- Target: **3x speedup (110+ games/min)**

---

## Proposed Sweeps

### Sweep 1: Parallel Batch Size Optimization ⭐ (HIGHEST PRIORITY)

**Purpose:** Find the optimal `parallel_batch_size` parameter
**Hypothesis:** Current default (10) is too conservative; 15-25 likely better for 32 workers

**Configuration:**
```python
workers = 32  # Fixed
mcts_config = "Medium" (3 det × 30 sims)  # Fixed
parallel_batch_size = [5, 10, 15, 20, 25, 30]  # SWEEP
games_per_config = 50
```

**Expected findings:**
- Sweet spot likely 15-25 (balances batch size vs MCTS overhead)
- Too small (5): Underutilizes batch submission API
- Too large (30): MCTS tree becomes bottleneck, more virtual loss conflicts
- Performance curve: increases then plateaus or decreases

**Why this matters:**
- Most impactful parameter for GPU utilization
- Directly affects batch sizes seen by GPU (10 leaves × 32 workers = 320 concurrent)
- Informs all subsequent sweeps

---

### Sweep 2: Worker Scaling with New Optimizations

**Purpose:** Re-test worker scaling with optimal `parallel_batch_size`
**Hypothesis:** Better batching improves scaling efficiency at high worker counts

**Configuration:**
```python
workers = [1, 4, 8, 16, 24, 32, 40, 48]  # SWEEP
parallel_batch_size = <optimal from Sweep 1>  # Fixed
mcts_config = "Medium" (3 det × 30 sims)  # Fixed
games_per_config = 50
```

**Expected findings:**
- Improved scaling efficiency vs baseline (PERFORMANCE-FINDINGS.md showed 64% @ 32 workers)
- Possibly viable to use 40-48 workers now (baseline failed due to GPU thrashing)
- Diminishing returns still expected beyond ~32 workers (hardware limit: 8GB VRAM)
- Efficiency analysis: speedup vs ideal linear speedup

**Comparison to baseline:**
| Workers | Baseline (seq) | New (parallel) | Expected Improvement |
|---------|---------------|----------------|---------------------|
| 1       | 2.6 g/min     | ~4 g/min       | 1.5x (better batching) |
| 8       | 17.1 g/min    | ~30 g/min      | 1.75x |
| 16      | 26.6 g/min    | ~55 g/min      | 2.1x |
| 32      | 36.7 g/min    | ~80 g/min      | 2.2x |
| 48      | FAILED (OOM)  | ~90 g/min?     | May work now! |

**Why this matters:**
- Determines optimal worker count for production
- Validates if hardware limit (8GB VRAM) still applies
- Shows if new optimizations enable >32 worker configurations

---

### Sweep 3: 2D Interaction Matrix (Workers × Batch Size)

**Purpose:** Visualize interaction between workers and `parallel_batch_size`
**Hypothesis:** Optimal batch size increases with worker count

**Configuration:**
```python
workers = [8, 16, 24, 32, 40]  # SWEEP
parallel_batch_size = [10, 15, 20, 25, 30]  # SWEEP
mcts_config = "Medium" (3 det × 30 sims)  # Fixed
games_per_config = 30  # Lower for 25 configs
```

**Expected findings:**
- **Diagonal pattern**: Higher workers benefit from larger batch sizes
  - 8 workers: optimal ~10-15
  - 16 workers: optimal ~15-20
  - 32 workers: optimal ~20-25
  - 40 workers: optimal ~25-30
- Performance heatmap shows "ridge" of optimal configurations
- Diminishing returns beyond certain batch size for each worker count

**Example heatmap (hypothetical):**
```
Workers    bs=10    bs=15    bs=20    bs=25    bs=30
   8       28.5     31.2     29.8     27.5     25.1
  16       51.3     58.7     62.1     59.4     55.2
  24       68.2     77.5     85.3     88.1     84.7
  32       79.1     88.4     97.2    101.5     98.3  ← peak
  40       82.5     93.1    103.7    110.2    108.9  ← possible new peak
```

**Why this matters:**
- Shows whether "one size fits all" or need worker-dependent tuning
- Identifies best (worker, batch_size) combination
- Validates if 40+ workers viable with correct batch size

---

### Sweep 4: MCTS Configuration Impact

**Purpose:** Test Light/Medium/Heavy MCTS with optimal settings
**Hypothesis:** All configs benefit from new optimizations, update training time estimates

**Configuration:**
```python
workers = 32  # Fixed
parallel_batch_size = <optimal from Sweep 1>  # Fixed
mcts_configs = [
    ("Light", 2 det × 20 sims = 40 sims/move),
    ("Medium", 3 det × 30 sims = 90 sims/move),
    ("Heavy", 5 det × 50 sims = 250 sims/move)
]
games_per_config = 50
```

**Expected findings:**
- All configs see ~2x speedup vs baseline
- Updated training time estimates:

| Config | Baseline | Optimized | Training Time (500 iter) |
|--------|----------|-----------|-------------------------|
| Light  | 69.1 g/min | ~138 g/min | 36 days → 18 days |
| Medium | 36.7 g/min | ~80 g/min  | 136 days → 62 days |
| Heavy  | 16.0 g/min | ~35 g/min  | 312 days → 142 days |

**Why this matters:**
- Updates project timeline estimates
- Shows if optimization benefits consistent across MCTS complexities
- Helps choose MCTS config for production (quality vs speed tradeoff)

---

### Sweep 5: Batch Timeout Sensitivity (OPTIONAL)

**Purpose:** Test if faster batch timeout improves responsiveness
**Hypothesis:** Lower timeout (5ms) may work better with batch submission API

**Configuration:**
```python
workers = 32  # Fixed
parallel_batch_size = <optimal from Sweep 1>  # Fixed
mcts_config = "Medium"  # Fixed
batch_timeout_ms = [3.0, 5.0, 8.0, 10.0, 15.0]  # SWEEP
games_per_config = 30
```

**Expected findings:**
- Sweet spot likely 5-8ms (current default: 10ms)
- Too low (3ms): Smaller batches, lower GPU utilization
- Too high (15ms): Added latency, workers wait longer
- May be relatively insensitive (5-10ms all similar)

**Why this matters:**
- Fine-tuning parameter (lower priority than batch size)
- May unlock additional 5-10% performance
- Informs default config for production

---

## Execution Plan

### Quick Validation (~1-2 hours)
Test hypotheses quickly with reduced parameter space:
```bash
python benchmarks/performance/benchmark_session1_validation.py \
  --sweep quick \
  --games 20 \
  --output benchmarks/results/session1_validation_quick.csv
```

**Quick sweep covers:**
- Sweep 1: batch_size = [10, 20, 30] @ 32 workers (60 games)
- Sweep 3: 2D matrix [8,16,32] × [10,20,30] (180 games)
- **Total: ~240 games, ~30 minutes**

### Full Validation (~4-6 hours)
Comprehensive parameter exploration:
```bash
python benchmarks/performance/benchmark_session1_validation.py \
  --sweep all \
  --games 50 \
  --output benchmarks/results/session1_validation_full.csv
```

**Full sweep covers:**
- Sweep 1: batch_size × 6 values (300 games)
- Sweep 2: workers × 8 values (400 games)
- Sweep 3: 2D matrix 5×5 (750 games)
- Sweep 4: MCTS configs × 3 (150 games)
- Sweep 5: timeout × 5 values (150 games)
- **Total: ~1,750 games, ~4-6 hours**

### Individual Sweeps
Run specific sweeps independently:
```bash
# Just find optimal batch size (most important!)
python benchmarks/performance/benchmark_session1_validation.py \
  --sweep batch_size --games 50

# Test worker scaling with default batch_size=20
python benchmarks/performance/benchmark_session1_validation.py \
  --sweep workers --games 50

# 2D interaction heatmap
python benchmarks/performance/benchmark_session1_validation.py \
  --sweep 2d --games 30
```

---

## Success Criteria

### Minimum (Acceptable)
- **2x speedup**: 73+ games/min with Medium MCTS
- Optimal `parallel_batch_size` identified (likely 15-25)
- Worker scaling efficiency ≥70% @ 32 workers (vs 64% baseline)

### Target (Expected)
- **2.5x speedup**: 92+ games/min with Medium MCTS
- Worker scaling viable up to 40 workers
- Training time for 500 iterations: <60 days (down from 136 days)

### Stretch (Optimistic)
- **3x speedup**: 110+ games/min with Medium MCTS
- 48 workers functional (no CUDA OOM)
- Training time for 500 iterations: <45 days

---

## Expected Insights

1. **Optimal batch size scales with worker count**
   - 8 workers → batch_size ~10-15
   - 32 workers → batch_size ~20-25
   - 48+ workers → batch_size ~25-30

2. **Improved worker scaling efficiency**
   - Baseline: 64% efficiency @ 32 workers
   - Optimized: 75%+ efficiency @ 32 workers
   - Possibly viable beyond 32 workers now

3. **Consistent speedup across MCTS configs**
   - All configs benefit ~2x from optimizations
   - Light MCTS gets most benefit (more leaves expanded)
   - Heavy MCTS still bottlenecked by computation

4. **Batch timeout relatively insensitive**
   - 5-10ms all perform similarly
   - 5ms slightly better for responsiveness

5. **Hardware limit still exists but pushed higher**
   - Baseline: Failed @ 48 workers (CUDA OOM)
   - Optimized: May work @ 48-56 workers before hitting limit

---

## Post-Sweep Actions

After running the benchmark:

1. **Update CLAUDE.md with new optimal config:**
   ```python
   # ml/config.py
   use_parallel_expansion: bool = True
   parallel_batch_size: int = <optimal_value>  # From Sweep 1
   ```

2. **Document findings in OPTIMIZATION-RESULTS.md:**
   - Performance comparison table
   - Updated training time estimates
   - Worker scaling efficiency analysis
   - Parameter sensitivity analysis

3. **Update README.md training timeline:**
   - Old: ~136 days @ 36.7 games/min
   - New: ~XX days @ YY games/min

4. **Proceed to Session 3 if target not met:**
   - If <110 games/min, continue with mixed precision inference
   - Sessions 3+4+5 should provide additional 30-50% speedup

---

## Comparison to PERFORMANCE-FINDINGS.md

**What's different:**
- **Then**: Sequential MCTS, per-request batching, no parallel expansion
- **Now**: Parallel MCTS expansion, batch submission API, optimized batching

**What's similar:**
- Same hardware (RTX 4060 8GB, Ryzen 9 7950X)
- Same MCTS configurations (Light/Medium/Heavy)
- Same benchmarking methodology (games/min, worker scaling)

**New opportunities:**
- `parallel_batch_size` parameter (didn't exist before)
- Potential for >32 workers (failed before due to poor batching)
- Better GPU utilization (was 15-20%, should improve to 30-50%+)

**New risks:**
- Virtual loss conflicts at very high batch sizes
- Memory pressure at high workers × high batch sizes
- Diminishing returns if GPU becomes bottleneck

---

## Next Steps After Validation

If Session 1+2 achieves **≥2x speedup**:
- ✅ Proceed to Session 3 (Mixed Precision Inference)
- ✅ Sessions 3+4+5 for final 30-50% boost to reach 3x total

If Session 1+2 achieves **<1.5x speedup**:
- ⚠️ Debug: Check batch sizes in logs
- ⚠️ Profile: Where is time spent?
- ⚠️ Consider Session 6 (Cython MCTS) earlier

If Session 1+2 achieves **≥3x speedup already**:
- 🎉 Excellent! Sessions 3-5 are gravy
- 🎉 Could achieve 4-5x total speedup
- 🎉 Training time: potentially 30-40 days
