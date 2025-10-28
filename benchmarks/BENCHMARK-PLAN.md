# Comprehensive Benchmark Plan - Correct Network Configuration

> **⚠️ HISTORICAL DOCUMENT - Windows Era**
>
> This document contains the benchmark plan and execution results from the Windows/Python 3.12 development era (Phase 1-4).
> - **Platform:** Windows 10, Python 3.12
> - **Date:** 2025-10-26 (Phase 1 execution)
> - **Status:** Phase 1 executed, moderate results (1.76x speedup)
> - **Current Platform:** Ubuntu 24.04, Python 3.14 (migrated 2025-10-27)
>
> For current benchmark results and tracking, see [results/RESULTS-TRACKER.md](results/RESULTS-TRACKER.md)

**Date**: 2025-10-26
**Status**: ⚠️ PHASE 1 EXECUTED - MODERATE RESULTS (1.76x speedup)
**Hardware**: NVIDIA RTX 4060 (3,072 CUDA cores, 8GB VRAM), AMD Ryzen 9 7950X (16 cores, 32 threads)

**Execution Summary**: See [docs/findings/PHASE1-EXECUTION-SUMMARY.md](docs/findings/PHASE1-EXECUTION-SUMMARY.md)
**Next Steps**: See [docs/archive/NEXT-STEPS-GPU-MCTS.md](docs/archive/NEXT-STEPS-GPU-MCTS.md)

---

## Executive Summary

### Critical Discovery
Previous benchmarks used **WRONG network size** (~361K parameters instead of ~4.9M):

| Parameter | Wrong Config | Correct Config | Impact |
|-----------|--------------|----------------|--------|
| `embedding_dim` | 128 | **256** | 2x larger |
| `num_layers` | 2 | **6** | 3x more layers |
| `feedforward_dim` | 256 | **1024** | 4x larger |
| **Total Parameters** | **361,269** | **4,917,301** | **13.6x difference!** |

**Finding**: Larger network (4.9M) runs **5-7x FASTER** than small network (361K) in self-play!
- Small network: 5.9 games/min (361K params)
- Large network: 40.7 games/min (4.9M params)

**Reason**: GPU efficiency - larger network has better kernel occupancy and less overhead ratio.

---

## Baseline Configuration (CORRECT)

All benchmarks MUST use this exact configuration:

```python
network = BlobNet(
    state_dim=256,           # Fixed (state encoding dimension)
    embedding_dim=256,       # ← CORRECT (not 128!)
    num_layers=6,            # ← CORRECT (not 2!)
    num_heads=8,             # Standard
    feedforward_dim=1024,    # ← CORRECT (not 256!)
    dropout=0.0,             # Inference mode (or 0.1 for training)
)
```

**Expected Output**:
```
Network parameters: 4,917,301
```

### Baseline Self-Play Configuration

```python
SelfPlayEngine(
    network=network,
    num_workers=32,                 # Optimal for Windows/RTX 4060
    device="cuda",
    use_batched_evaluator=False,    # No Phase 2/3 batching
    use_thread_pool=False,          # Multiprocessing (not threading)
    num_determinizations=3,         # Medium MCTS
    simulations_per_determinization=30,
)
```

### Baseline Performance Targets

From [BASELINE-REPRODUCTION-FINDINGS.md](docs/performance/BASELINE-REPRODUCTION-FINDINGS.md):

| MCTS Config | Determinizations | Sims/Det | Total Sims | Expected Games/Min | Expected Sec/Game |
|-------------|------------------|----------|------------|-------------------|------------------|
| **Light** | 2 | 20 | 40 | **80.8** | **0.74** |
| **Medium** | 3 | 30 | 90 | **43.3** | **1.38** |
| **Heavy** | 5 | 50 | 250 | **25.0** | **2.40** |

---

## Fail-Fast-Fail-Cheap Philosophy

**Strategy**: Run minimal tests first to quickly identify promising configurations, then deep-dive only on winners.

**Defaults**:
- **5 games per test** (not 50) for initial screening
- **Single configuration** (not sweeps) for validation
- **Medium MCTS only** (3×30) for baseline comparison
- **Scale up only after validation** (promising tests get 20-50 games)

**Timeline**:
- Phase 1 (Quick Screen): 5 games × 3 critical configs = **~5-10 minutes**
- Phase 2 (Deep Dive): 50 games × 1-2 winners = **~20-40 minutes**
- Phase 3 (Production Run): 100+ games × 1 final config = **~60+ minutes**

---

## Benchmark Test Suite

### Test 1: Self-Play Performance (Worker Scaling)

**Script**: `benchmarks/performance/benchmark_selfplay.py`

**Purpose**: Measure self-play throughput with different worker counts and MCTS configurations

**Phase 1 - Quick Screen** (5 games each, Medium MCTS only):
```bash
python benchmarks/performance/benchmark_selfplay.py \
    --device cuda \
    --games 5 \
    --workers 32 \
    --output results/selfplay_quick_screen.csv
```

**Phase 2 - Deep Dive** (if 32 workers matches baseline):
```bash
python benchmarks/performance/benchmark_selfplay.py \
    --device cuda \
    --games 20 \
    --workers 1 4 8 16 32 \
    --output results/selfplay_worker_sweep.csv
```

**Phase 3 - Full Validation** (if scaling looks good):
```bash
python benchmarks/performance/benchmark_selfplay.py \
    --device cuda \
    --games 50 \
    --workers 32 \
    --output results/selfplay_baseline_correct.csv
```

**Tests**:
- **Quick Screen**: 32 workers, Medium MCTS, 5 games = **1 config (~1 min)**
- **Deep Dive**: 5 worker counts × 3 MCTS configs × 20 games = **15 configs (~30 min)**
- **Full**: 5 worker counts × 3 MCTS configs × 50 games = **15 configs (~60 min)**

**Expected Results** (from baseline):

| Workers | Light (games/min) | Medium (games/min) | Heavy (games/min) |
|---------|-------------------|-------------------|-------------------|
| 1 | ~5 | ~2.7 | ~1.6 |
| 4 | ~20 | ~10.8 | ~6.3 |
| 8 | ~40 | ~21.6 | ~12.5 |
| 16 | ~60 | ~32.5 | ~18.8 |
| **32** | **80.8** | **43.3** | **25.0** |

**Validation**: Performance should scale roughly linearly up to 32 workers, then plateau.

---

### Test 2: Training Performance (Batch Size & Precision)

**Script**: `benchmarks/performance/benchmark_training.py`

**Purpose**: Measure GPU training throughput with different batch sizes and FP16/FP32

**Phase 1 - Quick Screen** (validate training is not bottleneck):
```bash
python benchmarks/performance/benchmark_training.py \
    --devices cuda \
    --batch-sizes 2048 \
    --epochs 3 \
    --examples 50000 \
    --output results/training_quick_screen.json
```

**Phase 2 - Full Sweep** (if needed for optimization):
```bash
python benchmarks/performance/benchmark_training.py \
    --devices cuda \
    --batch-sizes 256 512 1024 2048 \
    --epochs 5 \
    --examples 100000 \
    --output results/training_baseline_correct.json
```

**Tests**:
- **Quick Screen**: 2048 batch, FP32+FP16, 3 epochs = **2 configs (~5 min)**
- **Full**: 4 batch sizes × 2 precisions × 5 epochs = **8 configs (~20 min)**

**Expected Results** (from [FINDINGS.md](FINDINGS.md)):

| Batch Size | FP32 (examples/sec) | FP16 (examples/sec) | Speedup |
|------------|---------------------|---------------------|---------|
| 256 | ~15,000 | ~27,000 | 1.8x |
| 512 | ~20,000 | ~36,000 | 1.8x |
| 1024 | ~25,000 | ~45,000 | 1.8x |
| **2048** | **41,322** | **74,115** | **1.79x** |

**Validation**: Training is NOT the bottleneck (can process 74K examples/sec vs 43 games/min × 20 examples/game = 14K examples/min needed).

---

### Test 3: Phase 1 Validation (Intra-Game Batching) ⭐ **MOST CRITICAL**

**Script**: `benchmarks/performance/benchmark_phase1_validation.py`

**Purpose**: Test if Phase 1 GPU-batched MCTS with virtual loss provides speedup

**Phase 1 - Quick Screen** (baseline + 2-3 batch sizes):
```bash
# Baseline (no batching)
python benchmarks/performance/benchmark_phase1_validation.py \
    --games 5 \
    --batch-size 0 \
    --no-baseline \
    --workers 32 \
    --device cuda \
    --output results/phase1_quick_baseline.csv

# Test batch sizes: 30, 60, 90
python benchmarks/performance/benchmark_phase1_validation.py \
    --games 5 \
    --batch-start 30 \
    --batch-end 90 \
    --batch-step 30 \
    --no-baseline \
    --workers 32 \
    --device cuda \
    --output results/phase1_quick_sweep.csv
```

**Phase 2 - Fine-Grained Sweep** (if Phase 1 shows 2x+ speedup):
```bash
python benchmarks/performance/benchmark_phase1_validation.py \
    --games 10 \
    --batch-start 20 \
    --batch-end 100 \
    --batch-step 10 \
    --workers 32 \
    --device cuda \
    --output results/phase1_fine_sweep.csv
```

**Phase 3 - Production Validation** (at discovered optimum):
```bash
python benchmarks/performance/benchmark_phase1_validation.py \
    --games 50 \
    --batch-size 60 \
    --workers 32 \
    --device cuda \
    --output results/phase1_production.csv
```

**Tests**:
- **Quick Screen**: Baseline + 3 batch sizes × 5 games = **4 configs (~5 min)** ⭐
- **Fine Sweep**: Baseline + 9 batch sizes × 10 games = **10 configs (~15 min)**
- **Production**: Baseline + 1 optimal × 50 games = **2 configs (~20 min)**

**Previous Results** (WRONG - with 361K network):
- Baseline: 5.9 games/min
- Best Phase 1 (batch=21): 11.1 games/min
- Speedup: **1.88x** (below 2-10x target)

**Expected Results** (CORRECT - with 4.9M network):
- Baseline: ~43 games/min (predicted from Test 1)
- Phase 1 (batch=30-60): **86-215 games/min** (2-5x speedup)
- **Hypothesis**: Larger network benefits more from batching due to better GPU utilization

**Validation Criteria**:
- ✅ **Good**: ≥2x speedup over baseline
- ✅ **Excellent**: ≥5x speedup over baseline
- ❌ **Poor**: <1.5x speedup (investigate overhead)

---

### Test 4: Phase 2 Multi-Game Batching (Multiprocessing)

**Script**: `benchmarks/performance/benchmark_phase2.py`

**Purpose**: Test if per-worker `BatchedEvaluator` improves performance

**Configuration**:
```bash
python benchmarks/performance/benchmark_phase2.py \
    --workers 4 8 16 32 \
    --games 50 \
    --device cuda \
    --output results/phase2_baseline_correct.csv
```

**Tests**:
- Workers: 4, 8, 16, 32
- With/without BatchedEvaluator
- Total: 8 configurations

**Previous Results** (from [PHASE2-COMPLETE.md](docs/phases/PHASE2-COMPLETE.md)):
- Small batches (3.5 avg) → overhead > benefit
- Status: **DISCREDITED** (doesn't achieve large enough batches)

**Expected Results** (with 4.9M network):
- Larger network may create more backpressure → larger batches
- If batches remain small (3-10), overhead will still dominate
- **Hypothesis**: Unlikely to help unless batches reach 32+

**Validation**: Compare batch_size achieved. If <16, Phase 2 provides no benefit.

---

### Test 5: Phase 3 Threading + Shared Evaluator

**Script**: `benchmarks/performance/benchmark_phase3.py`

**Purpose**: Test if threading with shared BatchedEvaluator beats multiprocessing

**Configuration**:
```bash
python benchmarks/performance/benchmark_phase3.py \
    --workers 32 128 256 \
    --games 50 \
    --device cuda \
    --output results/phase3_baseline_correct.csv
```

**Tests**:
- Threading workers: 32, 128, 256
- Multiprocessing baseline: 32
- Total: 4 configurations

**Previous Results** (from [PHASE3-COMPLETE.md](docs/phases/PHASE3-COMPLETE.md)):
- Threading (128 workers): 8.0 games/min
- Multiprocessing (32 workers): 68.3 games/min
- Verdict: **8.5x SLOWER** due to GIL

**Expected Results** (with 4.9M network):
- GIL still dominates (97% Python code in MCTS)
- Threading: **Still 5-10x slower** than multiprocessing
- **Hypothesis**: Network size doesn't fix GIL bottleneck

**Validation**: Confirm threading remains inferior on Windows.

---

### Test 6: Full Iteration Benchmark

**Script**: `benchmarks/performance/benchmark_iteration.py`

**Purpose**: Measure full training iteration (self-play + training) at 1/10th scale

**Configuration**:
```bash
python benchmarks/performance/benchmark_iteration.py \
    --games 1000 \
    --epochs 10 \
    --batch-size 2048 \
    --workers 32 \
    --device cuda \
    --output results/iteration_baseline_correct.json
```

**Tests**:
- 1,000 games (1/10th of full 10,000)
- 10 training epochs
- Projects to 500 iterations

**Previous Results** (WRONG - small network):
- Not available with correct network

**Expected Results** (with 4.9M network):
- Self-play (1,000 games): ~23 minutes (at 43 games/min)
- Training (10 epochs, 100K examples): ~2 minutes (at 74K examples/sec)
- **Total per iteration (10K games)**: ~230 minutes (3.8 hours)
- **500 iterations**: ~1,917 hours = **79.9 days**

**Validation**: Self-play should be 90%+ of iteration time (training is not bottleneck).

---

### Test 7: MCTS Configuration Sweep

**Script**: Custom test using `benchmark_selfplay.py`

**Purpose**: Find optimal MCTS configuration for training

**Configuration**:
```bash
python benchmarks/performance/benchmark_selfplay.py \
    --device cuda \
    --games 100 \
    --workers 32 \
    --output results/mcts_sweep_correct.csv
```

**Manual Config Edit**: Test these MCTS configurations in code:

| Name | Determinizations | Sims/Det | Total Sims | Purpose |
|------|------------------|----------|------------|---------|
| Ultra-Light | 2 | 10 | 20 | Fastest (quality check) |
| Light | 2 | 20 | 40 | Fast iteration |
| Medium | 3 | 30 | 90 | **Baseline** |
| Heavy | 5 | 50 | 250 | High quality |
| Ultra-Heavy | 5 | 100 | 500 | Research quality |

**Expected Results**:
- Games/min should decrease as MCTS sims increase
- Quality vs speed tradeoff

**Validation**: Choose config that balances training speed and model quality.

---

## Results Table Template

### Self-Play Performance

| Test ID | Network Params | Workers | MCTS Config | Games/Min | GPU Util % | Status | Notes |
|---------|---------------|---------|-------------|-----------|------------|--------|-------|
| T1.1 | 4.9M | 1 | Light | | | | |
| T1.2 | 4.9M | 4 | Light | | | | |
| T1.3 | 4.9M | 8 | Light | | | | |
| T1.4 | 4.9M | 16 | Light | | | | |
| T1.5 | 4.9M | 32 | Light | 80.8 | ~15% | Expected | Baseline target |
| T1.6 | 4.9M | 1 | Medium | | | | |
| T1.7 | 4.9M | 4 | Medium | | | | |
| T1.8 | 4.9M | 8 | Medium | | | | |
| T1.9 | 4.9M | 16 | Medium | | | | |
| T1.10 | 4.9M | 32 | Medium | 43.3 | ~15% | Expected | **PRIMARY BASELINE** |
| T1.11 | 4.9M | 1 | Heavy | | | | |
| T1.12 | 4.9M | 4 | Heavy | | | | |
| T1.13 | 4.9M | 8 | Heavy | | | | |
| T1.14 | 4.9M | 16 | Heavy | | | | |
| T1.15 | 4.9M | 32 | Heavy | 25.0 | ~15% | Expected | High quality |

### Training Performance

| Test ID | Network Params | Batch Size | Precision | Examples/Sec | VRAM (MB) | Status | Notes |
|---------|---------------|------------|-----------|--------------|-----------|--------|-------|
| T2.1 | 4.9M | 256 | FP32 | | | | |
| T2.2 | 4.9M | 512 | FP32 | | | | |
| T2.3 | 4.9M | 1024 | FP32 | | | | |
| T2.4 | 4.9M | 2048 | FP32 | 41,322 | | Expected | Baseline |
| T2.5 | 4.9M | 256 | FP16 | | | | |
| T2.6 | 4.9M | 512 | FP16 | | | | |
| T2.7 | 4.9M | 1024 | FP16 | | | | |
| T2.8 | 4.9M | 2048 | FP16 | 74,115 | | Expected | **BEST** |

### Phase 1 Validation (Intra-Game Batching)

| Test ID | Network Params | Batch Size | Games/Min | Speedup vs Baseline | Status | Notes |
|---------|---------------|------------|-----------|---------------------|--------|-------|
| T3.1 | 4.9M | None | 43.3 | 1.0x | Expected | Baseline |
| T3.2 | 4.9M | 10 | | | | |
| T3.3 | 4.9M | 20 | | | | |
| T3.4 | 4.9M | 30 | | | | Predicted optimum |
| T3.5 | 4.9M | 40 | | | | |
| T3.6 | 4.9M | 50 | | | | |
| T3.7 | 4.9M | 60 | | | | |
| T3.8 | 4.9M | 70 | | | | |
| T3.9 | 4.9M | 80 | | | | |
| T3.10 | 4.9M | 90 | | | | Max (3×30 MCTS) |
| T3.11 | 4.9M | 100 | | | | Overflow test |

**Validation Criteria**:
- ✅ **Excellent**: Speedup ≥5x (e.g., 215+ games/min)
- ✅ **Good**: Speedup ≥2x (e.g., 86+ games/min)
- ⚠️ **Moderate**: Speedup 1.5-2x (needs investigation)
- ❌ **Poor**: Speedup <1.5x (overhead too high)

### Phase 2 Multi-Game Batching

| Test ID | Network Params | Workers | Batched Eval | Avg Batch Size | Games/Min | Status | Notes |
|---------|---------------|---------|--------------|----------------|-----------|--------|-------|
| T4.1 | 4.9M | 4 | No | N/A | | | Baseline |
| T4.2 | 4.9M | 4 | Yes | | | | |
| T4.3 | 4.9M | 8 | No | N/A | | | Baseline |
| T4.4 | 4.9M | 8 | Yes | | | | |
| T4.5 | 4.9M | 16 | No | N/A | | | Baseline |
| T4.6 | 4.9M | 16 | Yes | | | | |
| T4.7 | 4.9M | 32 | No | N/A | 43.3 | Expected | **PRIMARY BASELINE** |
| T4.8 | 4.9M | 32 | Yes | | | | Check batch size |

**Validation**: If avg_batch_size < 16, expect no benefit (overhead > gain).

### Phase 3 Threading vs Multiprocessing

| Test ID | Network Params | Workers | Method | Batched Eval | Games/Min | Status | Notes |
|---------|---------------|---------|--------|--------------|-----------|--------|-------|
| T5.1 | 4.9M | 32 | Multiproc | No | 43.3 | Expected | **BASELINE** |
| T5.2 | 4.9M | 32 | Threading | Yes | | | Expect slower |
| T5.3 | 4.9M | 128 | Threading | Yes | | | Expect GIL bottleneck |
| T5.4 | 4.9M | 256 | Threading | Yes | | | Stress test |

**Validation**: Threading should be 5-10x slower due to GIL (Windows).

### Full Iteration Timing

| Test ID | Network Params | Games | Self-Play (min) | Training (min) | Total (min) | 500-Iter Projection (days) | Notes |
|---------|---------------|-------|-----------------|----------------|-------------|---------------------------|-------|
| T6.1 | 4.9M | 1,000 | | | | | 1/10th scale test |

**Projected for 10,000 games/iteration**:
- Self-play: Scale by 10x
- Training: Same (10 epochs on 100K vs 1M examples ≈ similar time)

### MCTS Configuration Comparison

| Test ID | Network Params | Det | Sims/Det | Total Sims | Games/Min | Quality Score | Notes |
|---------|---------------|-----|----------|------------|-----------|---------------|-------|
| T7.1 | 4.9M | 2 | 10 | 20 | | TBD | Ultra-fast |
| T7.2 | 4.9M | 2 | 20 | 40 | 80.8 | TBD | **Light** |
| T7.3 | 4.9M | 3 | 30 | 90 | 43.3 | TBD | **Medium (baseline)** |
| T7.4 | 4.9M | 5 | 50 | 250 | 25.0 | TBD | **Heavy** |
| T7.5 | 4.9M | 5 | 100 | 500 | | TBD | Research |

---

## Comparison to Previous Results

### Old Results (WRONG - 361K network)

| Test | Configuration | Result | Source |
|------|---------------|--------|--------|
| Self-play (Medium MCTS) | 361K, 32 workers | 5.9 games/min | Phase 1 benchmark |
| Phase 1 (batch=21) | 361K, 32 workers | 11.1 games/min | Phase 1 validation |
| Speedup | Phase 1 vs baseline | **1.88x** | Below 2x target |

### New Results (CORRECT - 4.9M network)

| Test | Configuration | Expected Result | Actual Result | Status |
|------|---------------|-----------------|---------------|--------|
| Self-play (Medium MCTS) | 4.9M, 32 workers | 43.3 games/min | | To measure |
| Phase 1 (batch=30-60) | 4.9M, 32 workers | 86-215 games/min | | To measure |
| Speedup | Phase 1 vs baseline | **2-5x** | | To measure |

**Key Insight**: Phase 1 may show MUCH BETTER speedup with correct network size!

---

## Execution Order (Fail-Fast-Fail-Cheap)

### Critical Path (Quick Screen - Run First)

**Time**: ~10-15 minutes total
**Goal**: Validate baseline and identify if Phase 1 is promising

```bash
# Step 1: Baseline Self-Play (32 workers, Medium MCTS, 5 games) - 1 min
python benchmarks/performance/benchmark_selfplay.py --device cuda --games 5 --workers 32

# Step 2: Phase 1 Baseline (no batching, 5 games) - 2 min
python benchmarks/performance/benchmark_phase1_validation.py --games 5 --batch-size 0 --no-baseline

# Step 3: Phase 1 Quick Sweep (3 batch sizes, 5 games each) - 6 min
python benchmarks/performance/benchmark_phase1_validation.py --games 5 --batch-start 30 --batch-end 90 --batch-step 30 --no-baseline

# Step 4: Training Quick Check (2048 batch, FP16+FP32, 3 epochs) - 5 min
python benchmarks/performance/benchmark_training.py --devices cuda --batch-sizes 2048 --epochs 3 --examples 50000
```

**Decision Point 1** (after 15 min):
- ✅ If Phase 1 shows ≥2x speedup → **Continue to Deep Dive**
- ⚠️ If Phase 1 shows 1.5-2x speedup → **Run fine-grained sweep**
- ❌ If Phase 1 shows <1.5x speedup → **STOP, implement GPU-batched MCTS instead**

### Deep Dive (If Phase 1 Promising)

**Time**: ~30-45 minutes
**Goal**: Find optimal Phase 1 configuration

```bash
# Step 5: Fine-Grained Batch Size Sweep (10 games per config) - 15 min
python benchmarks/performance/benchmark_phase1_validation.py --games 10 --batch-start 20 --batch-end 100 --batch-step 10

# Step 6: Worker Scaling Validation (20 games per config) - 30 min
python benchmarks/performance/benchmark_selfplay.py --device cuda --games 20 --workers 1 4 8 16 32

# Step 7: Full Iteration Timing (1000 games) - 30 min
python benchmarks/performance/benchmark_iteration.py --games 1000 --epochs 10 --workers 32
```

**Decision Point 2** (after 45 min):
- ✅ If Phase 1 optimum ≥5x speedup → **Production validation**
- ⚠️ If Phase 1 optimum 2-5x speedup → **Production validation, consider GPU-batched MCTS**
- ❌ If optimal <2x → **Skip production, implement GPU-batched MCTS**

### Production Validation (If Configuration Found)

**Time**: ~60 minutes
**Goal**: Confirm production-ready performance

```bash
# Step 8: Production Phase 1 Validation (50 games at optimal batch size) - 20 min
python benchmarks/performance/benchmark_phase1_validation.py --games 50 --batch-size <OPTIMAL>

# Step 9: Full Self-Play Baseline (50 games, all MCTS configs) - 60 min
python benchmarks/performance/benchmark_selfplay.py --device cuda --games 50 --workers 32
```

### Optional Tests (Completeness Only)

Run ONLY if you have extra time and want completeness:

```bash
# Test 4: Phase 2 Multi-Game Batching - likely fails (20 min)
python benchmarks/performance/benchmark_phase2.py --workers 4 8 16 32 --games 5

# Test 5: Phase 3 Threading - likely fails on Windows (20 min)
python benchmarks/performance/benchmark_phase3.py --workers 32 128 256 --games 5

# Test 7: MCTS Configuration Sweep (if optimizing quality vs speed) - 50 min
python benchmarks/performance/benchmark_selfplay.py --device cuda --games 50 --workers 32
```

---

## Quick Reference: Execution Timeline

| Phase | Tests | Time | Decision |
|-------|-------|------|----------|
| **Quick Screen** | T1, T3 baseline + 3 batch sizes, T2 | **15 min** | Continue or STOP? |
| **Deep Dive** | T3 fine sweep, T1 worker sweep, T6 | **45 min** | Production or reconsider? |
| **Production** | T3 optimal, T1 full baseline | **60 min** | Final validation |
| **Optional** | T4, T5, T7 | **90 min** | Completeness only |

**Total Critical Path**: 15 min (quick screen) → 45 min (deep dive) → 60 min (production) = **~2 hours**

**Priority Order**:
1. ⭐ **Test 3** (Phase 1 Validation) - Quick screen with 5 games
2. ⭐ **Test 1** (Self-Play Baseline) - 5 games to confirm baseline
3. ⭐ **Test 2** (Training) - 3 epochs to confirm not bottleneck
4. **Decision Point**: Continue only if Phase 1 shows promise
5. **Test 3** (Fine sweep) - If promising, find optimum
6. **Test 6** (Iteration) - Project total training time
7. **Test 1** (Full sweep) - Worker scaling validation
8. **Test 4, 5, 7** - Optional, for completeness

---

## Success Criteria

### Minimum Acceptable Results
- ✅ Test 1 (Self-play): Matches baseline ±10% (38-48 games/min for Medium MCTS)
- ✅ Test 2 (Training): Achieves >60K examples/sec with FP16
- ✅ Test 3 (Phase 1): Shows ≥1.5x speedup over baseline
- ✅ Test 6 (Iteration): Projects <100 days for 500 iterations

### Excellent Results
- ✅ Test 1: Matches or exceeds baseline (43+ games/min)
- ✅ Test 2: Matches baseline (74K examples/sec)
- ✅ Test 3: Shows ≥5x speedup (215+ games/min)
- ✅ Test 6: Projects <20 days for 500 iterations

### Decision Points

**If Phase 1 achieves ≥5x speedup**:
- ✅ Implement Phase 1 in production
- ✅ Expected training time: 8-16 days
- ✅ **DECISION**: Phase 1 is sufficient, skip GPU-batched MCTS

**If Phase 1 achieves 2-5x speedup**:
- ⚠️ Implement Phase 1 for moderate improvement
- ⚠️ Expected training time: 16-40 days
- ⚠️ **DECISION**: Consider GPU-batched MCTS for further 3-5x improvement

**If Phase 1 achieves <2x speedup**:
- ❌ Do NOT implement Phase 1 (overhead too high)
- ❌ Expected training time: >40 days
- ❌ **DECISION**: MUST implement GPU-batched MCTS (see [PLAN-GPUBatchedMCTS.md](docs/performance/PLAN-GPUBatchedMCTS.md))

---

## Files to Update

### Scripts Requiring Network Config Fix

| File | Current Config | Status | Priority |
|------|---------------|--------|----------|
| `benchmark_selfplay.py` | 128/2/256 | ❌ WRONG | **HIGH** |
| `benchmark_training.py` | 256/4/512 | ⚠️ Close | **HIGH** |
| `benchmark_phase1_validation.py` | 128/2/256 | ❌ WRONG | **CRITICAL** |
| `benchmark_phase2.py` | 128/2/256 | ❌ WRONG | MEDIUM |
| `benchmark_phase3.py` | 128/2/256 | ❌ WRONG | LOW |
| `benchmark_iteration.py` | 256/4/512 | ⚠️ Close | **HIGH** |
| `test_batched_phase1.py` | Default | ⚠️ Unknown | MEDIUM |

### Scripts Already Correct

| File | Config | Status |
|------|--------|--------|
| `test_action_plan_windows.py` | 256/6/1024 | ✅ CORRECT |
| `test_reproduce_baseline.py` | 256/6/1024 | ✅ CORRECT |
| `test_action_plan.py` | 256/6/1024 | ✅ CORRECT |
| `benchmark_diagnostic.py` | 256/6/1024 | ✅ CORRECT |

---

## Timeline Estimate

**Per Test** (approximate):
- Test 1 (15 configs × 50 games): ~60 minutes
- Test 2 (8 configs × 5 epochs): ~20 minutes
- Test 3 (11 configs × 20 games): ~30 minutes
- Test 4 (8 configs × 50 games): ~40 minutes
- Test 5 (4 configs × 50 games): ~20 minutes
- Test 6 (1 iteration): ~30 minutes
- Test 7 (5 configs × 100 games): ~50 minutes

**Total Runtime**: ~4-5 hours (all tests)
**Critical Tests Only** (T1, T2, T3, T6): ~2-3 hours

---

## Notes

- All tests use **cuda** device (RTX 4060)
- All CSV/JSON outputs saved to `results/` directory
- Scripts print parameter count for verification
- Monitor GPU temperature (thermal throttling possible on consecutive runs)
- Run with `--quick` flag for initial validation (fewer configs, fewer games)

---

## References

- [BASELINE-REPRODUCTION-FINDINGS.md](docs/performance/BASELINE-REPRODUCTION-FINDINGS.md) - Discovery of network size issue
- [FINDINGS.md](FINDINGS.md) - Performance investigation summary
- [PHASE1-COMPLETE.md](docs/phases/PHASE1-COMPLETE.md) - Phase 1 (intra-game batching) implementation
- [PHASE2-COMPLETE.md](docs/phases/PHASE2-COMPLETE.md) - Phase 2 (multi-game batching) failure analysis
- [PHASE3-COMPLETE.md](docs/phases/PHASE3-COMPLETE.md) - Phase 3 (threading) failure analysis
- [PLAN-GPUBatchedMCTS.md](docs/performance/PLAN-GPUBatchedMCTS.md) - Future GPU-batched MCTS plan

---

## Execution Results (2025-10-26)

**Status**: ⚠️ **PHASE 1 COMPLETED - MODERATE SPEEDUP**

### What Was Executed

Following the "Fail-Fast-Fail-Cheap" strategy, we executed the **Phase 1 Quick Screen**:

1. ✅ **Baseline Self-Play Performance** ([test_reproduce_baseline.py](benchmarks/tests/test_reproduce_baseline.py))
   - Medium MCTS: **31.8 games/min** (expected 43.3, -27%)
   - Actual baseline slower than expected but consistent

2. ✅ **Phase 1 GPU-Batched MCTS Quick Sweep** ([benchmark_phase1_validation.py](benchmarks/performance/benchmark_phase1_validation.py))
   - Tested batch sizes: 30, 60, 90
   - Best result: **1.76x speedup** at batch_size=60
   - Results: [results/phase1_quick_sweep.csv](results/phase1_quick_sweep.csv)

### Key Findings

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| **Phase 1 Speedup** | **1.76x** | 2.0x minimum | ⚠️ Below target |
| Baseline (Medium MCTS) | 31.8 games/min | 43.3 games/min | -27% variance |
| Phase 1 (batch=60) | 5.6 games/min* | N/A | *5-card games |
| GPU Utilization | 0% | 30%+ | ❌ GPU underutilized |

**Note**: Phase 1 benchmark used 5 cards/game (vs 3 cards in baseline), explaining the lower absolute games/min. The **speedup ratio (1.76x) is valid** for comparison.

### Decision: Stop Phase 1, Proceed to GPU-Batched MCTS

Per BENCHMARK-PLAN.md decision criteria (line 651-653):

> **If Phase 1 achieves <2x speedup**: ❌ Do NOT implement Phase 1, MUST implement GPU-batched MCTS

**Rationale**:
- 1.76x speedup insufficient (below 2x target)
- Training time: 80 days → 45 days (still too slow)
- GPU underutilization (0%) indicates MCTS bottleneck, not NN inference
- Phase 1 intra-game batching hit fundamental limits

**Next Steps**: Implement GPU-batched MCTS (target: 5-10x speedup)
- See detailed plan: [docs/performance/NEXT-STEPS-GPU-MCTS.md](docs/performance/NEXT-STEPS-GPU-MCTS.md)
- Expected performance: 160-320 games/min (vs 32 baseline)
- Expected training time: **8-16 days** (vs 80 days baseline)

### Time Investment

- **Phase 1 Quick Screen**: 30 minutes ✅
- **Avoided time**: 90+ minutes (skipped fine-grained sweep and production validation)
- **Fail-fast approach**: SUCCESS - quickly identified Phase 1 limitations

### Documentation Generated

1. ✅ [PHASE1-EXECUTION-SUMMARY.md](docs/performance/PHASE1-EXECUTION-SUMMARY.md) - Complete execution report
2. ✅ [NEXT-STEPS-GPU-MCTS.md](docs/performance/NEXT-STEPS-GPU-MCTS.md) - Implementation roadmap
3. ✅ [results/baseline_reproduction.csv](results/baseline_reproduction.csv) - Baseline measurements
4. ✅ [results/phase1_quick_sweep.csv](results/phase1_quick_sweep.csv) - Phase 1 results

---

**Status**: **PHASE 1 COMPLETE** ✓ - Recommend GPU-batched MCTS for transformative gains

All benchmark scripts verified to use the correct 4.9M parameter network configuration (256/6/1024).
