# BlobMaster Benchmark Suite

**Platform:** Ubuntu 24.04 | Python 3.14.0 (GIL enabled) | RTX 4060 8GB | Ryzen 9 7950X
**Status:** Fresh testing environment - all previous results archived
**Last Updated:** 2025-10-28

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run baseline performance check (5 minutes)
python benchmarks/performance/benchmark_selfplay.py --quick

# Run full self-play benchmark suite (30-60 minutes)
python benchmarks/performance/benchmark_selfplay.py

# Run complete training iteration test (1-2 hours)
python benchmarks/performance/benchmark_iteration.py

# Generate comprehensive report from all results
python benchmarks/performance/benchmark_report.py
```

## Navigation

- **[RESULTS-TRACKER.md](results/RESULTS-TRACKER.md)** - Centralized findings & progress tracking
- **[BENCHMARK-PLAN.md](BENCHMARK-PLAN.md)** - Historical test plan (Windows era, Phase 1 executed)
- **[performance/](performance/)** - Core benchmark scripts
- **[tests/](tests/)** - Configuration validation & ad-hoc tests
- **[results/](results/)** - Test results (archived & current)
- **[docs/](docs/)** - Detailed findings & documentation

## Performance Testing Scripts

### Core Benchmarks (`performance/`)

#### 1. Self-Play Performance - [benchmark_selfplay.py](performance/benchmark_selfplay.py)
**Purpose:** Measure game generation throughput (games/min)
**Tests:** Worker counts (1, 4, 8, 16, 32) × MCTS configs (light, medium, heavy)
**Output:** CSV with games/min, CPU%, training examples/min

```bash
# Quick test (5 games per config, ~5 minutes)
python benchmarks/performance/benchmark_selfplay.py --quick

# Full test (100 games per config, ~30-60 minutes)
python benchmarks/performance/benchmark_selfplay.py

# Custom test
python benchmarks/performance/benchmark_selfplay.py --games 50 --workers 16,32
```

**MCTS Configurations:**
- **Light:** 2 determinizations × 20 simulations (fast, lower quality)
- **Medium:** 3 determinizations × 30 simulations (recommended for training)
- **Heavy:** 5 determinizations × 50 simulations (high quality, slow)

---

#### 2. GPU Training Performance - [benchmark_training.py](performance/benchmark_training.py)
**Purpose:** Measure neural network training throughput
**Tests:** Batch sizes (256, 512, 1024, 2048) × Precision (FP32, FP16) × Device (CPU, CUDA)
**Output:** JSON with examples/sec, GPU%, VRAM usage

```bash
# Full training benchmark
python benchmarks/performance/benchmark_training.py

# CUDA only
python benchmarks/performance/benchmark_training.py --device cuda

# Specific batch sizes
python benchmarks/performance/benchmark_training.py --batch-sizes 512,1024
```

---

#### 3. Full Iteration Timing - [benchmark_iteration.py](performance/benchmark_iteration.py)
**Purpose:** Measure complete training iteration (self-play + training + evaluation)
**Tests:** 1/10th scale (1,000 games) with timing breakdown
**Output:** Phase timing + 500-iteration projection

```bash
# Run full iteration test (1-2 hours)
python benchmarks/performance/benchmark_iteration.py

# Quick test with fewer games
python benchmarks/performance/benchmark_iteration.py --games 100
```

**Phases measured:**
1. Self-play game generation
2. Replay buffer updates
3. Network training
4. Model evaluation (ELO tournament)
5. Checkpointing

---

#### 4. Optimization Phase Benchmarks

**Phase 1: Intra-Game Batching** - [benchmark_phase1_validation.py](performance/benchmark_phase1_validation.py)
- Tests GPU-batched MCTS with virtual loss
- Batch sizes: 0 (baseline), 10-100
- Measures speedup vs baseline

```bash
python benchmarks/performance/benchmark_phase1_validation.py --quick
```

**Phase 2: Multi-Game Batching** - [benchmark_phase2.py](performance/benchmark_phase2.py)
- Tests per-worker BatchedEvaluator
- Measures average batch sizes achieved

```bash
python benchmarks/performance/benchmark_phase2.py
```

**Phase 3: Threading vs Multiprocessing** - [benchmark_phase3.py](performance/benchmark_phase3.py)
- Tests threading with shared BatchedEvaluator
- Worker counts: 32, 128, 256

```bash
python benchmarks/performance/benchmark_phase3.py
```

**GPU-Batched MCTS** - [benchmark_gpu_batch_mcts.py](performance/benchmark_gpu_batch_mcts.py)
- Tests parallel expansion MCTS
- Compares: Baseline vs Phase 1 vs GPU-Batched
- Expected: 5-10x speedup

```bash
python benchmarks/performance/benchmark_gpu_batch_mcts.py
```

---

#### 5. Diagnostic & Reporting

**Diagnostic Suite** - [benchmark_diagnostic.py](performance/benchmark_diagnostic.py)
- Comprehensive profiling and bottleneck detection
- Tests worker counts: 4, 8, 16, 32, 64, 128
- Tracks GPU%, CPU%, batch sizes

```bash
# Full diagnostic (2-3 hours)
python benchmarks/performance/benchmark_diagnostic.py

# Quick diagnostic (30 minutes)
python benchmarks/performance/benchmark_diagnostic.py --quick
```

**Report Generator** - [benchmark_report.py](performance/benchmark_report.py)
- Aggregates all benchmark results
- Generates markdown reports
- Identifies bottlenecks and optimization opportunities

```bash
python benchmarks/performance/benchmark_report.py
```

---

## Configuration Tests (`tests/`)

### Validation Scripts

1. **[test_network_size.py](tests/test_network_size.py)** - Verify 4.9M parameters (not 361K)
2. **[test_reproduce_baseline.py](tests/test_reproduce_baseline.py)** - Validate expected games/min
3. **[test_diagnostic_quick.py](tests/test_diagnostic_quick.py)** - Quick diagnostic validation

### Feature Tests

4. **[test_batched_phase1.py](tests/test_batched_phase1.py)** - Phase 1 batching validation
5. **[test_gpu_server.py](tests/test_gpu_server.py)** - GPU server architecture (failed: 3-5x slower)
6. **[test_gpu_worker.py](tests/test_gpu_worker.py)** - GPU worker process validation

### Historical Tests (Windows Era)

7. **[test_action_plan.py](tests/test_action_plan.py)** - Linux configuration tests (128/256 workers)
8. **[test_action_plan_windows.py](tests/test_action_plan_windows.py)** - Windows baseline validation

---

## Results Organization

```
results/
├── RESULTS-TRACKER.md         # Master findings & progress log
├── archive/
│   └── windows-era/           # Historical results (Python 3.12, Windows)
│       ├── *.csv              # Performance measurements
│       └── *.json             # Training benchmarks
└── ubuntu-2025-10/            # Current test results (Python 3.14, Ubuntu)
    └── (fresh results go here)
```

**Important:** All archived results are from Windows/Python 3.12 era. Platform switch to Ubuntu/Python 3.14 requires fresh baseline measurements.

---

## Expected Performance Baselines

### Windows Era (Historical - Python 3.12, Windows 10)

| MCTS Config | Games/Min | Training Time (500 iter × 10K games) |
|-------------|-----------|-------------------------------------|
| Light (2×20) | 80.8 | 4.2 days |
| **Medium (3×30)** | **43.3** | **7.1 days** |
| Heavy (5×50) | 25.0 | 12.1 days |

**Configuration:** 32 workers, multiprocessing, no batching

### Ubuntu Current (Python 3.14, Ubuntu 24.04) ✅ Validated 2025-10-28

**Status:** ✅ Baseline established - performs 5% faster than Windows

| MCTS Config | Games/Min | Sec/Game | Training Time (500 iter × 10K games) |
|-------------|-----------|----------|-------------------------------------|
| **Medium (3×30)** | **45.5** | **1.32** | **6.7 days** |

**Configuration:** 32 workers, CUDA device, `use_batched_evaluator=False`, `use_thread_pool=False`

**Hardware:** RTX 4060 @ 99% GPU, 6.7GB/8.2GB VRAM, 75.9% CPU

**Result:** Ubuntu/Python 3.14 is **1.05x faster** than Windows/Python 3.12 baseline!

---

## Recommended Testing Workflow

### Phase 1: Baseline Validation (Ubuntu/Python 3.14)

```bash
# 1. Verify network architecture
python benchmarks/tests/test_network_size.py

# 2. Quick self-play check (5 minutes)
python benchmarks/performance/benchmark_selfplay.py --quick

# 3. Full self-play baseline (30-60 minutes)
python benchmarks/performance/benchmark_selfplay.py

# 4. GPU training baseline (15-30 minutes)
python benchmarks/performance/benchmark_training.py

# 5. Full iteration test (1-2 hours)
python benchmarks/performance/benchmark_iteration.py
```

### Phase 2: Optimization Testing

```bash
# Test intra-game batching
python benchmarks/performance/benchmark_phase1_validation.py

# Test GPU-batched MCTS (if Phase 1 shows promise)
python benchmarks/performance/benchmark_gpu_batch_mcts.py
```

### Phase 3: Diagnostic Analysis

```bash
# Run comprehensive diagnostics
python benchmarks/performance/benchmark_diagnostic.py

# Generate report
python benchmarks/performance/benchmark_report.py
```

---

## Documentation

### Primary References

- **[results/RESULTS-TRACKER.md](results/RESULTS-TRACKER.md)** - Current findings & progress
- **[docs/findings/PERFORMANCE-FINDINGS.md](docs/findings/PERFORMANCE-FINDINGS.md)** - Comprehensive analysis

### Historical Documentation

- **[BENCHMARK-PLAN.md](BENCHMARK-PLAN.md)** - Original test plan (Phase 1 executed 2025-10-26)
- **[docs/archive/](docs/archive/)** - Superseded planning documents

---

## Key Findings Summary

### Best Configuration (Windows Era)

```python
config = TrainingConfig(
    num_workers=32,              # Optimal for RTX 4060 + Ryzen 9
    device="cuda",               # GPU for neural network
    num_determinizations=3,      # Medium MCTS
    simulations_per_determinization=30,
    batch_size=512,              # GPU training batch size
    use_batched_evaluator=False, # No multi-game batching
    use_thread_pool=False,       # Multiprocessing (not threading)
)
```

### Failed Optimization Attempts

1. **Phase 2 (Multi-game batching):** Small batch sizes (3.5 avg) → overhead > benefit
2. **Phase 3 (Threading):** GIL contention → 8.5x slower than multiprocessing
3. **Phase 3.5 (GPU Server):** Queue overhead → 3-5x slower than baseline
4. **Phase 1 (Intra-game batching):** 1.76x speedup (below 2x target)
5. **60+ workers:** GPU thrashing → 52x slower

### Current Challenges

- **Linux performance regression:** 2.2x slower than Windows baseline
- **Training time:** 120 days projected (vs 7-12 days target)
- **Root cause:** Unknown - systematic benchmarking needed

---

## Interpreting Results

### Self-Play Benchmarks

**Key metrics:**
- `games_per_min`: Primary throughput metric
- `avg_game_duration_sec`: Should be 30-60s for medium MCTS
- `cpu_percent`: Should be 70-90% (not 100% = bottleneck, not <50% = underutilized)
- `training_examples_per_min`: Includes all (state, policy, value) tuples

**Good result:** 40+ games/min with medium MCTS, 32 workers, 80-90% CPU

**Bad signs:**
- <20 games/min → investigate GPU/CPU bottleneck
- >95% CPU sustained → need fewer workers
- GPU% <10% → GPU underutilized

### Training Benchmarks

**Key metrics:**
- `examples_per_sec`: Should be >10,000 for batch_size=512, CUDA
- `gpu_percent`: Should be >80% during training
- `vram_gb`: Should be <6GB for RTX 4060 (8GB total)

**Good result:** 15,000+ examples/sec, 90%+ GPU, <6GB VRAM

### Iteration Benchmarks

**Key metrics:**
- `self_play_time_sec`: Should dominate (60-80% of total)
- `training_time_sec`: Should be 10-20% of total
- `total_iteration_time_hours`: Scale linearly with games

**Good result:** Self-play time = 2-3 hours per 1,000 games (medium MCTS)

---

## Troubleshooting

### Slow Self-Play (<20 games/min)

1. Check worker count: `nvidia-smi` - is GPU thrashing?
2. Check CPU utilization: Should be 70-90%
3. Run diagnostic: `python benchmarks/performance/benchmark_diagnostic.py --quick`
4. Profile code: Use `--profile` flag on benchmarks

### GPU Underutilization (<20%)

1. Check batch sizes: Increase to 1024 or 2048
2. Check for CPU bottleneck: Is self-play saturating CPU?
3. Verify CUDA available: `torch.cuda.is_available()`

### Out of Memory Errors

1. Reduce batch size: Try 256 instead of 512
2. Reduce workers: Try 16 instead of 32
3. Clear GPU cache: Add `torch.cuda.empty_cache()` calls

---

## Contact & Support

For detailed performance analysis, see [results/RESULTS-TRACKER.md](results/RESULTS-TRACKER.md)

For implementation details, see project root [CLAUDE.md](../CLAUDE.md)
