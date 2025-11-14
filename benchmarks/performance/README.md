# Performance Benchmarking Suite

This directory contains benchmarking scripts for measuring and optimizing the BlobMaster training pipeline performance.

## Current Baseline Performance (2025-11-14)

**Hardware:** RTX 4060 8GB, Ryzen 9 7950X 16-core, 128GB DDR5, Ubuntu 24.04
**Phase:** Phase 1 - Independent Rounds Training
**Configuration:** 32 workers, multiprocessing (use_thread_pool=False), parallel expansion enabled, parallel_batch_size=30, batch_size=512

| MCTS Config | Determinizations × Simulations | Total Sims/Move | Rounds/Min | Training Time (5M rounds) |
|-------------|-------------------------------|-----------------|------------|---------------------------|
| **Light**   | 2 × 20                        | 40              | ~668       | ~5.2 days                 |
| **Medium**  | 3 × 30                        | 90              | ~540-980*  | ~3.5-6.4 days             |
| **Heavy**   | 5 × 50                        | 250             | ~250       | ~13.9 days                |

\* **Note:** Performance varies with test duration. Short tests (~200 rounds) achieve ~540 r/min, while longer tests (500+ rounds) achieve ~980 r/min due to warmup effects and batch efficiency. Auto-tune validation uses ~980 r/min baseline.

**Recommended:** Medium MCTS (~540-980 rounds/min) for best quality/speed balance.

---

## Terminology

**IMPORTANT:** This project distinguishes between two training phases:

- **Round (Phase 1):** Single bidding + trick-taking cycle with fixed cards (e.g., one 5-card deal). Phase 1 trains on independently sampled rounds. Metric: **rounds/min**

- **Game (Phase 2):** Complete Blob game = full sequence of rounds (e.g., 5 players, C=7 → 17 rounds: 7→6→5→4→3→2→1→1→1→1→1→2→3→4→5→6→7). Phase 2 will train on complete game sequences (not yet implemented). Metric: **games/min**

**All current benchmarks measure Phase 1 (rounds/min).**

---

## Benchmark Scripts

### 1. benchmark_optimal_config.py ⭐ PRIMARY BASELINE

**Purpose:** Measure baseline performance with validated optimal configuration.

**Usage:**
```bash
# Test single MCTS config (default: Medium)
python benchmarks/performance/benchmark_optimal_config.py --games 500

# Test all three MCTS configs (Light/Medium/Heavy)
python benchmarks/performance/benchmark_optimal_config.py --games 500 --test-mcts-variants

# Quick validation (fewer rounds)
python benchmarks/performance/benchmark_optimal_config.py --games 100
```

**Parameters:**
- `--games INT`: Number of rounds to benchmark (default: 1000)
- `--device`: cuda or cpu (default: cuda)
- `--test-mcts-variants`: Test Light/Medium/Heavy MCTS configs
- `--output PATH`: Output JSON path (default: benchmark_optimal_results.json)

**Output:**
- JSON with detailed metrics
- CSV summary
- Console: Rounds/min, training time estimates

**Expected Results (varies by test duration):**
- Light: ~668 rounds/min
- Medium: ~540-980 rounds/min (short vs long tests)
- Heavy: ~250 rounds/min

---

### 2. benchmark_selfplay.py

**Purpose:** Worker scaling and MCTS parameter exploration.

**Usage:**
```bash
# Test specific worker counts
python benchmarks/performance/benchmark_selfplay.py --workers 1 4 8 16 32 --games 100

# Quick screening test (5 rounds per config)
python benchmarks/performance/benchmark_selfplay.py --workers 32 --games 5

# Test multiple MCTS configs
python benchmarks/performance/benchmark_selfplay.py --workers 32 --games 100 --quick
```

**Parameters:**
- `--workers INT [INT ...]`: Worker counts to test (default: [32])
- `--games INT`: Rounds per configuration (default: 5)
- `--device`: cuda or cpu (default: cpu)
- `--output PATH`: Output CSV path
- `--quick`: Quick test with fewer configurations

**Output:**
- CSV with worker scaling results
- Console: Performance table, recommendations

**Use Cases:**
- Worker scaling analysis
- Hardware limit identification (max: 32 workers for RTX 4060 8GB)
- MCTS configuration comparison

---

### 3. benchmark_iteration.py

**Purpose:** Full training iteration timing (self-play + network training).

**Usage:**
```bash
# Quick iteration test (reduced scale)
python benchmarks/performance/benchmark_iteration.py --games 500 --epochs 5

# Test with specific worker count
python benchmarks/performance/benchmark_iteration.py --games 1000 --epochs 10 --workers 32

# Test specific device
python benchmarks/performance/benchmark_iteration.py --games 500 --epochs 5 --device cuda
```

**Parameters:**
- `--games INT`: Rounds for self-play (default: 500)
- `--epochs INT`: Training epochs (default: 5)
- `--batch-size INT`: Training batch size (default: 512)
- `--workers INT`: Self-play workers (default: 16)
- `--device`: cuda or cpu (default: cuda)
- `--output PATH`: Output JSON path

**Output:**
- JSON with phase breakdown
- Self-play time vs training time percentage
- Projected full training time

**Use Cases:**
- Identify bottlenecks (self-play vs training)
- Project full 500-iteration training time
- Test adaptive curriculum impact

---

### 4. benchmark_training.py

**Purpose:** Neural network training performance (batch sizes, precision modes).

**Usage:**
```bash
# Test specific batch size
python benchmarks/performance/benchmark_training.py --batch-sizes 512 --device cuda

# Test multiple batch sizes
python benchmarks/performance/benchmark_training.py --batch-sizes 256 512 1024 2048 --device cuda

# Quick test
python benchmarks/performance/benchmark_training.py --quick
```

**Parameters:**
- `--devices`: cpu, cuda (default: cpu cuda)
- `--batch-sizes INT [INT ...]`: Batch sizes to test (default: [2048])
- `--epochs INT`: Epochs per config (default: 3)
- `--examples INT`: Synthetic training examples (default: 50000)
- `--quick`: Quick test mode
- `--output PATH`: Output JSON path

**Output:**
- JSON with batches/sec, examples/sec
- VRAM usage (if CUDA)
- FP16 vs FP32 comparison

**Use Cases:**
- Find optimal batch size for hardware
- Measure mixed precision speedup
- VRAM usage profiling

---

### 5. visualize_session1_results.py

**Purpose:** Generate plots and summary from benchmark CSV results.

**Usage:**
```bash
# Visualize benchmark results
python benchmarks/performance/visualize_session1_results.py benchmark_selfplay_results.csv

# Specify output directory
python benchmarks/performance/visualize_session1_results.py results.csv --output-dir plots/
```

**Parameters:**
- `input_csv`: Path to benchmark results CSV
- `--output-dir PATH`: Output directory for plots (default: same as CSV)

**Output:**
- PNG plots: batch size sweep, worker scaling, 2D heatmap, MCTS comparison
- TXT summary: Best configuration, recommendations, scaling analysis

**Features:**
- Backward compatible (accepts both "games_per_minute" and "rounds_per_minute" columns)
- Automatic baseline comparison (980 rounds/min for Medium MCTS)
- Scaling efficiency analysis

---

### 6. auto_tune.py ⭐ INTELLIGENT PARAMETER SWEEP

**Purpose:** Automatically find optimal training configuration through systematic parameter exploration.

**Usage:**
```bash
# Full sweep (5-6 hours)
python benchmarks/performance/auto_tune.py

# Quick mode (reduced parameter space, ~75 minutes)
python benchmarks/performance/auto_tune.py --quick

# With time budget (6 hours max)
python benchmarks/performance/auto_tune.py --time-budget 360

# Resume from checkpoint
python benchmarks/performance/auto_tune.py --resume results/auto_tune_20251114_153000/auto_tune_results.db

# Run specific phases only
python benchmarks/performance/auto_tune.py --phases baseline,individual
```

**Parameters:**
- `--time-budget INT`: Maximum runtime in minutes (default: unlimited)
- `--resume PATH`: Resume from checkpoint database
- `--output DIR`: Output directory (default: results/auto_tune_YYYYMMDD_HHMMSS/)
- `--quick`: Quick mode with reduced parameter space
- `--phases STR`: Comma-separated list (baseline,individual,interaction,final)
- `--verbose`: Verbose logging
- `--baseline-tolerance FLOAT`: Baseline validation tolerance (default: 0.10 = ±10%)
- `--early-stop-threshold FLOAT`: Performance threshold for early termination (default: 0.30)
- `--device STR`: cuda or cpu (default: cuda)

**What It Does:**

The auto-tune script performs intelligent parameter exploration in four phases:

1. **Phase 1: Baseline Validation** (5-10 min)
   - Validates current optimal config (32w × 30batch × 3×30 MCTS)
   - Confirms system performance (expected: ~980 rounds/min ±10%)
   - Fails fast if baseline is significantly degraded

2. **Phase 2: Individual Parameter Sweeps** (1-3 hours)
   - One-at-a-time exploration of each parameter
   - Order: workers → parallel_batch_size → num_determinizations → simulations_per_det
   - Quick screening (100 rounds) with early termination for poor performers
   - Extended validation (500 rounds) for promising configs (within 5% of best)

3. **Phase 3: Interaction Exploration** (1-2 hours)
   - Tests promising combinations from Phase 2
   - Focuses on top 3 values per parameter
   - Medium validation (500 rounds per config)

4. **Phase 4: Final Validation** (30-60 min)
   - Top 5 configs, extended runs (1000 rounds × 3 runs)
   - Statistical validation with confidence intervals
   - Variance analysis for stability testing

**Parameter Space:**

**Full mode:**
- `workers`: [8, 16, 24, 32]
- `parallel_batch_size`: [10, 20, 30, 40, 50]
- `num_determinizations`: [2, 3, 4, 5]
- `simulations_per_det`: [15, 20, 25, 30, 35, 40, 50]
- `batch_timeout_ms`: [5, 10, 15, 20]

**Quick mode:**
- `workers`: [16, 24, 32]
- `parallel_batch_size`: [20, 30, 40]
- `num_determinizations`: [2, 3, 4]
- `simulations_per_det`: [20, 30, 40]
- `batch_timeout_ms`: [10]

**Smart Features:**

- **Early termination:** Skips configs >30% slower than current best
- **Hardware-aware:** Respects known limits (max 32 workers for RTX 4060 8GB)
- **Adaptive sampling:** More validation for promising configs, less for poor performers
- **Checkpoint/resume:** Can resume after interruption (Ctrl+C saves progress)
- **Error handling:** Handles CUDA OOM gracefully, continues sweep
- **Progress tracking:** Real-time ETA based on actual runtime
- **Statistical rigor:** Multiple runs for top candidates, variance analysis

**Output:**

- **SQLite database:** `auto_tune_results.db` with all results
- **Checkpoint data:** Resume capability
- **Console:** Real-time progress with ETA, current best, phase summary

**Example Output:**
```
Auto-Tune Parameter Sweep
=========================
Output: results/auto_tune_20251114_153000
Device: cuda
Phases: baseline,individual,interaction,final

Phase 1/4: Baseline Validation
Config: 32w × 30batch × 3×30 MCTS
Expected: 980.0 r/min ±10%
✓ Result: 970 r/min (-1.0% vs expected)

Phase 2/4: Individual Parameter Sweeps (parallel_batch_size)
  parallel_batch_size=40:
    ✓ 750 r/min (promising, validating with 500 rounds)
    ★ NEW BEST: 812 r/min
Current best: 32w × 40batch × 3×30 = 812 r/min

...

SWEEP COMPLETE
Best configuration: 32w × 40batch × 3×30 MCTS
Performance: 812 r/min
Improvement: +9.6%
```

**Success Criteria:**
- ✓ Runs unattended for 6+ hours without crashes
- ✓ Finds config ≥10% better than baseline (>815 r/min)
- ✓ Produces actionable report with clear recommendations
- ✓ Results reproducible (same config → same perf ±5%)

---

### 7. auto_tune_report.py

**Purpose:** Generate comprehensive report from auto-tune sweep results.

**Usage:**
```bash
# Generate report from database
python benchmarks/performance/auto_tune_report.py results/auto_tune_20251114_153000/auto_tune_results.db

# Custom output path
python benchmarks/performance/auto_tune_report.py auto_tune_results.db --output custom_report.md

# Skip plots (faster)
python benchmarks/performance/auto_tune_report.py auto_tune_results.db --no-plots

# Custom baseline for comparison
python benchmarks/performance/auto_tune_report.py auto_tune_results.db --baseline 750.0
```

**Parameters:**
- `db_path`: Path to auto_tune_results.db (required)
- `--output PATH`: Output markdown file (default: auto_tune_report.md in same dir)
- `--baseline FLOAT`: Baseline performance for comparison (default: 980.0)
- `--no-plots`: Skip plot generation

**Output:**

**Markdown report** (`auto_tune_report.md`):
- Summary statistics (configs tested, best found, improvement)
- Top 10 configurations table
- Parameter analysis (best values for each parameter)
- Failed configurations with error types
- Key findings and recommendations
- Exportable config for ml/config.py

**Plots** (saved to `plots/` subdirectory):
- `worker_scaling.png` - Worker count vs performance
- `batch_size_sweep.png` - Parallel batch size impact
- `mcts_heatmap.png` - Determinizations × Simulations heatmap
- `top10_comparison.png` - Bar chart of top 10 configs
- `variance_analysis.png` - Performance vs stability scatter plot

**Example Report Snippet:**
```markdown
## Top 10 Configurations

| Rank | Workers | Batch Size | Det × Sims | Total Sims | Rounds/Min | vs Baseline | Variance |
|------|---------|------------|------------|------------|------------|-------------|----------|
| 1    | 32      | 40         | 3×30       | 90         | 812        | +9.6%       | 2.1%     |
| 2    | 32      | 50         | 3×25       | 75         | 798        | +7.7%       | 3.8%     |
| 3    | 32      | 30         | 3×30       | 90         | 745        | baseline    | 1.9%     |

## Key Findings
- **parallel_batch_size**: Sweet spot at 40 (+9.6% over baseline)
- **workers**: 32 remains optimal (24 shows 14% degradation)

## Recommendations
1. **Immediate**: Use 32w × 40batch × 3×30 (+9.6% speedup)
2. **Investigate**: Test intermediate batch sizes (32-38) for fine-tuning
```

**Use Cases:**
- Review auto-tune sweep results
- Share findings with team
- Export optimal config for production
- Visualize parameter relationships

---

## Archive

The `archive/` directory contains retired benchmark scripts from earlier optimization sessions:

- `benchmark_session1_validation.py` - Session 1+2 parameter exploration (completed)
- `benchmark_phase1_validation.py` - Phase 1 validation (completed)
- `benchmark_phase2.py` - Phase 2 batched MCTS (Phase 2 not active)
- `benchmark_phase3.py` - GPU server architecture (failed - 3-5x slower)
- `benchmark_gpu_batch_mcts.py` - Intra-game MCTS batching (future work)
- `benchmark_diagnostic.py` - Ad-hoc diagnostics
- `benchmark_report.py` - Report generation

These are kept for historical reference but are no longer maintained.

---

## Workflow Examples

### Validate Current Baseline

```bash
# Run primary baseline benchmark
python benchmarks/performance/benchmark_optimal_config.py \
  --games 500 \
  --test-mcts-variants \
  --output results/baseline_verification.json

# Expected: 668/540-980/250 rounds/min (Light/Medium/Heavy)
```

### Worker Scaling Analysis

```bash
# Test 1, 4, 8, 16, 32 workers
python benchmarks/performance/benchmark_selfplay.py \
  --workers 1 4 8 16 32 \
  --games 100 \
  --output results/worker_scaling.csv

# Visualize results
python benchmarks/performance/visualize_session1_results.py \
  results/worker_scaling.csv \
  --output-dir results/plots/
```

### Full Iteration Timing

```bash
# Measure one iteration at reduced scale
python benchmarks/performance/benchmark_iteration.py \
  --games 2000 \
  --workers 32 \
  --device cuda \
  --output results/iteration_timing.json

# Check self-play vs training bottleneck
cat results/iteration_timing.json | jq '.iteration_summary'
```

### Automated Parameter Sweep

Find optimal configuration automatically:

```bash
# Run auto-tune (5-6 hours, unattended)
python benchmarks/performance/auto_tune.py --output results/auto_tune/

# Quick mode for testing (~75 minutes)
python benchmarks/performance/auto_tune.py --quick

# Generate comprehensive report
python benchmarks/performance/auto_tune_report.py results/auto_tune/auto_tune_results.db

# View plots
ls results/auto_tune/plots/
```

**What you get:**
- Optimal configuration (tested across 50+ combinations)
- Performance improvement quantification
- Statistical validation with variance analysis
- Exportable config for ml/config.py
- Comprehensive plots and markdown report

---

## Hardware Limits

**RTX 4060 8GB Constraints:**
- **Max workers:** 32 (each worker uses ~150MB GPU memory, total ~5-6GB)
- **Beyond 32 workers:** CUDA out of memory errors
- **Batch sizes:** Up to 2048 for training (with mixed precision)
- **Scaling efficiency:** Diminishes beyond 16 workers (~44% at 32 workers)

---

## Adaptive Curriculum

The training pipeline uses a two-dimensional adaptive curriculum:

1. **MCTS Curriculum:** Progressive search depth increase
   - Iteration 50: 1×15 (30 sims/move)
   - Iteration 150: 2×25 (50 sims/move)
   - Iteration 300: 3×35 (105 sims/move)
   - Iteration 450: 4×45 (180 sims/move)
   - Iteration 500: 5×50 (250 sims/move)

2. **Training Units Curriculum:** Linear ramp from 2,000 → 10,000 rounds over 500 iterations

Benchmark scripts support testing at specific iterations with `--iteration` parameter (where applicable).

---

## Performance Investigation History

See `docs/performance/` for detailed historical analysis:

- **PERFORMANCE-FINDINGS.md** - Executive summary of optimization work
- **TRAINING-PERFORMANCE-MASTER.md** - Complete measurement history
- **SESSION1-VALIDATION-PLAN.md** - Parallel expansion optimization strategy
- **BASELINE-REPRODUCTION-FINDINGS.md** - Baseline verification methodology

---

## Troubleshooting

### Low Performance (~30-80 rounds/min instead of ~980)

**Symptom:** Benchmarks running 10-15x slower than expected baseline.

**Root Cause:** Threading instead of multiprocessing due to Python's GIL (Global Interpreter Lock).

**Solution:** Ensure `use_thread_pool=False` is set when creating `SelfPlayEngine`:

```python
engine = SelfPlayEngine(
    network=network,
    encoder=encoder,
    masker=masker,
    num_workers=32,
    num_determinizations=3,
    simulations_per_determinization=30,
    device="cuda",
    use_thread_pool=False,  # CRITICAL: Use multiprocessing, not threads!
    use_parallel_expansion=True,
    parallel_batch_size=30,
)
```

**Why this happens:**
- When `use_thread_pool` is not specified (None), the engine auto-selects based on device
- For CUDA, it defaults to `True` (threading), which causes GIL contention
- Threading with 32 workers results in ~75-80 r/min (GIL-limited)
- Multiprocessing with 32 workers achieves ~980 r/min (full parallelism)

**All active benchmark scripts have been updated to use multiprocessing by default.**

---

## Contributing New Benchmarks

When adding new benchmark scripts:

1. **Use correct terminology:** rounds/min (Phase 1) vs games/min (Phase 2)
2. **Always set `use_thread_pool=False`:** Ensure multiprocessing for proper performance
3. **Support current config:** Import from `ml.config import TrainingConfig`
4. **Standardize CLI:** Use argparse with --workers, --games, --device, --output
5. **Include progress:** Use tqdm or progress callbacks for long-running benchmarks
6. **Document expected results:** Add baseline numbers to this README
7. **Handle errors gracefully:** CUDA OOM, missing files, invalid parameters

---

## Quick Reference

| Script | Primary Use | Output | Duration |
|--------|-------------|--------|----------|
| benchmark_optimal_config.py | Baseline validation | JSON + CSV | ~40 sec (500 rounds) |
| benchmark_selfplay.py | Worker scaling | CSV | ~5-60 sec per config |
| benchmark_iteration.py | Full iteration timing | JSON | ~3-10 min |
| benchmark_training.py | Network training | JSON | ~2-5 min |
| visualize_session1_results.py | Plot generation | PNG + TXT | <10 sec |
| **auto_tune.py** | **Automated optimization** | **SQLite DB** | **5-6 hours (full)** |
| **auto_tune_report.py** | **Report from auto-tune** | **Markdown + Plots** | **<30 sec** |

---

**Last Updated:** 2025-11-14
**Baseline Validated:** 2025-11-14 (~980 rounds/min for 500-round tests, Medium MCTS, 32 workers, multiprocessing enabled)
