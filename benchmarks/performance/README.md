# Performance Benchmarking Suite

This directory contains benchmarking scripts for measuring and optimizing the BlobMaster training pipeline performance.

## Current Baseline Performance (2025-11-13)

**Hardware:** RTX 4060 8GB, Ryzen 9 7950X 16-core, 128GB DDR5, Ubuntu 24.04
**Phase:** Phase 1 - Independent Rounds Training
**Configuration:** 32 workers, parallel expansion enabled, parallel_batch_size=30

| MCTS Config | Determinizations × Simulations | Total Sims/Move | Rounds/Min | Training Time (5M rounds) |
|-------------|-------------------------------|-----------------|------------|---------------------------|
| **Light**   | 2 × 20                        | 40              | 1,049      | 3.3 days                  |
| **Medium**  | 3 × 30                        | 90              | 741        | 4.7 days                  |
| **Heavy**   | 5 × 50                        | 250             | 310        | 11.2 days                 |

**Recommended:** Medium MCTS (741 rounds/min) for best quality/speed balance.

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

**Expected Results (±5%):**
- Light: ~1,049 rounds/min
- Medium: ~741 rounds/min
- Heavy: ~310 rounds/min

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
- Automatic baseline comparison (741 rounds/min)
- Scaling efficiency analysis

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

# Expected: 1,049/741/310 rounds/min (Light/Medium/Heavy) ±5%
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

### Parameter Sweep (Future Auto-Tune)

The updated scripts are designed to support future automated parameter sweeping:

1. Run `benchmark_selfplay.py` with various configurations
2. Collect results in CSV
3. Use `visualize_session1_results.py` to analyze
4. Identify optimal settings

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

## Contributing New Benchmarks

When adding new benchmark scripts:

1. **Use correct terminology:** rounds/min (Phase 1) vs games/min (Phase 2)
2. **Support current config:** Import from `ml.config import TrainingConfig`
3. **Standardize CLI:** Use argparse with --workers, --games, --device, --output
4. **Include progress:** Use tqdm or progress callbacks for long-running benchmarks
5. **Document expected results:** Add baseline numbers to this README
6. **Handle errors gracefully:** CUDA OOM, missing files, invalid parameters

---

## Quick Reference

| Script | Primary Use | Output | Duration (500 rounds) |
|--------|-------------|--------|-----------------------|
| benchmark_optimal_config.py | Baseline validation | JSON + CSV | ~40 sec (Medium) |
| benchmark_selfplay.py | Worker scaling | CSV | ~5-60 sec per config |
| benchmark_iteration.py | Full iteration timing | JSON | ~3-10 min |
| benchmark_training.py | Network training | JSON | ~2-5 min |
| visualize_session1_results.py | Plot generation | PNG + TXT | <10 sec |

---

**Last Updated:** 2025-11-14
**Baseline Validated:** 2025-11-13 (741 rounds/min, Medium MCTS, 32 workers)
