# BlobMaster MPPT Auto-Tune Report

**Generated**: 2025-11-14 13:21:31

This report shows MPPT (Maximum Power Point Tracking) style optimization results for each curriculum stage.
Each stage uses coarse-to-fine gradient ascent to find optimal parameters.

## Overview

- **Total experiments**: 82
- **Stages optimized**: 4
- **Successful tests**: 82
- **Failed tests**: 0

## Summary Across All Stages

| Stage | MCTS | Baseline (r/min) | Optimal (r/min) | Improvement | Optimal Config |
|-------|------|------------------|-----------------|-------------|----------------|
| 1 | 1×15 | 2698.4 | 3421.1 | +26.8% | 31w×30b×10ms |
| 2 | 2×25 | 1483.1 | 1646.9 | +11.0% | 32w×30b×10ms |
| 3 | 3×35 | 658.9 | 1029.6 | +56.2% | 32w×35b×9ms |

---

# Detailed Stage Reports


## Stage 1: 1×15 MCTS (15 total simulations)

### Baseline Configuration

- **Config**: 32w × 30batch × 1×15 × 10ms
- **Performance**: 2698.4 rounds/min

### Individual Parameter Optimization

#### Parameter: `workers`

**Coarse Search (±4 steps):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 28 | coarse_down | 1090.5 | -59.6% | ✓ |
| 36 | coarse_up | 890.7 | -67.0% | ✓ |

**Fine Search (±1 step):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 31 | fine_down | 2775.7 | +2.9% | ✓ |
| 32 | fine_center | 2718.5 | +0.7% | ✓ |
| 33 | fine_up | 2607.2 | -3.4% | ✓ |

**Optimal `workers`: 31** (2775.7 r/min, +2.9% vs baseline)

#### Parameter: `parallel_batch_size`

**Coarse Search (±4 steps):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 20 | coarse_down | 1029.1 | -61.9% | ✓ |
| 40 | coarse_up | 955.4 | -64.6% | ✓ |

**Fine Search (±1 step):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 25 | fine_down | 2666.1 | -1.2% | ✓ |
| 30 | fine_center | 2681.1 | -0.6% | ✓ |
| 35 | fine_up | 2644.9 | -2.0% | ✓ |

**Optimal `parallel_batch_size`: 30** (2681.1 r/min, -0.6% vs baseline)

#### Parameter: `batch_timeout_ms`

**Coarse Search (±4 steps):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 5 | coarse_down | 1003.1 | -62.8% | ✓ |
| 15 | coarse_up | 920.3 | -65.9% | ✓ |

**Fine Search (±1 step):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 9 | fine_down | 2706.9 | +0.3% | ✓ |
| 10 | fine_center | 2689.0 | -0.3% | ✓ |
| 11 | fine_up | 2668.3 | -1.1% | ✓ |

**Optimal `batch_timeout_ms`: 9** (2706.9 r/min, +0.3% vs baseline)

### Combination Testing

| Config | Search Type | Rounds/Min | vs Baseline | Status |
|--------|-------------|------------|-------------|--------|
| 30w×30b×9ms | fine_tune | 2757.6 | +2.2% | ✓ |
| 31w×30b×8ms | fine_tune | 2725.9 | +1.0% | ✓ |
| 31w×30b×10ms | fine_tune | 2720.2 | +0.8% | ✓ |
| 31w×30b×9ms | combined | 2709.9 | +0.4% | ✓ |
| 31w×35b×9ms | fine_tune | 2696.2 | -0.1% | ✓ |
| 32w×30b×9ms | fine_tune | 2647.5 | -1.9% | ✓ |
| 31w×25b×9ms | fine_tune | 2644.6 | -2.0% | ✓ |

### Final Validation (3 runs × 1000 rounds)

**Final Config**: 31w × 30batch × 1×15 × 10ms

| Run | Rounds/Min |
|-----|------------|
| 1 | 3440.9 |
| 2 | 3405.8 |
| 3 | 3416.4 |

- **Average**: 3421.1 rounds/min
- **Variance**: ±0.4%
- **Improvement vs Baseline**: +26.8%

### Stage Summary

| Metric | Baseline | Optimal | Change |
|--------|----------|---------|--------|
| Performance | 2698.4 r/min | 3421.1 r/min | +26.8% |
| Workers | 32 | 31 | -1 |
| Batch Size | 30 | 30 | +0 |
| Timeout (ms) | 10 | 10 | +0 |

**Recommendation**: ✅ **Strong improvement**: 26.8% speedup achieved

---


## Stage 2: 2×25 MCTS (50 total simulations)

### Baseline Configuration

- **Config**: 32w × 30batch × 2×25 × 10ms
- **Performance**: 1483.1 rounds/min

### Individual Parameter Optimization

#### Parameter: `workers`

**Coarse Search (±4 steps):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 28 | coarse_down | 828.3 | -44.2% | ✓ |
| 36 | coarse_up | 693.9 | -53.2% | ✓ |

**Fine Search (±1 step):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 31 | fine_down | 1482.5 | -0.0% | ✓ |
| 32 | fine_center | 1504.0 | +1.4% | ✓ |
| 33 | fine_up | 1438.1 | -3.0% | ✓ |

**Optimal `workers`: 32** (1504.0 r/min, +1.4% vs baseline)

#### Parameter: `parallel_batch_size`

**Coarse Search (±4 steps):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 20 | coarse_down | 551.1 | -62.8% | ✓ |
| 40 | coarse_up | 717.4 | -51.6% | ✓ |

**Fine Search (±1 step):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 25 | fine_down | 1477.6 | -0.4% | ✓ |
| 30 | fine_center | 1492.5 | +0.6% | ✓ |
| 35 | fine_up | 1477.4 | -0.4% | ✓ |

**Optimal `parallel_batch_size`: 30** (1492.5 r/min, +0.6% vs baseline)

#### Parameter: `batch_timeout_ms`

**Coarse Search (±4 steps):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 5 | coarse_down | 715.7 | -51.7% | ✓ |
| 15 | coarse_up | 693.6 | -53.2% | ✓ |

**Fine Search (±1 step):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 9 | fine_down | 1460.1 | -1.6% | ✓ |
| 10 | fine_center | 1473.7 | -0.6% | ✓ |
| 11 | fine_up | 1489.2 | +0.4% | ✓ |

**Optimal `batch_timeout_ms`: 11** (1489.2 r/min, +0.4% vs baseline)

### Combination Testing

| Config | Search Type | Rounds/Min | vs Baseline | Status |
|--------|-------------|------------|-------------|--------|
| 32w×35b×11ms | fine_tune | 1497.5 | +1.0% | ✓ |
| 32w×30b×12ms | fine_tune | 1490.4 | +0.5% | ✓ |
| 32w×30b×11ms | combined | 1486.1 | +0.2% | ✓ |
| 32w×25b×11ms | fine_tune | 1478.6 | -0.3% | ✓ |
| 31w×30b×11ms | fine_tune | 1475.4 | -0.5% | ✓ |
| 32w×30b×10ms | fine_tune | 1470.3 | -0.9% | ✓ |
| 33w×30b×11ms | fine_tune | 1447.0 | -2.4% | ✓ |

### Final Validation (3 runs × 1000 rounds)

**Final Config**: 32w × 30batch × 2×25 × 10ms

| Run | Rounds/Min |
|-----|------------|
| 1 | 1643.5 |
| 2 | 1651.0 |
| 3 | 1646.3 |

- **Average**: 1646.9 rounds/min
- **Variance**: ±0.2%
- **Improvement vs Baseline**: +11.0%

### Stage Summary

| Metric | Baseline | Optimal | Change |
|--------|----------|---------|--------|
| Performance | 1483.1 r/min | 1646.9 r/min | +11.0% |
| Workers | 32 | 32 | +0 |
| Batch Size | 30 | 30 | +0 |
| Timeout (ms) | 10 | 10 | +0 |

**Recommendation**: ✅ **Strong improvement**: 11.0% speedup achieved

---


## Stage 3: 3×35 MCTS (105 total simulations)

### Baseline Configuration

- **Config**: 32w × 30batch × 3×35 × 10ms
- **Performance**: 658.9 rounds/min

### Individual Parameter Optimization

#### Parameter: `workers`

**Coarse Search (±4 steps):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 28 | coarse_down | 467.4 | -29.1% | ✓ |
| 36 | coarse_up | 423.4 | -35.7% | ✓ |

**Fine Search (±1 step):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 31 | fine_down | 652.4 | -1.0% | ✓ |
| 32 | fine_center | 665.8 | +1.0% | ✓ |
| 33 | fine_up | 647.6 | -1.7% | ✓ |

**Optimal `workers`: 32** (665.8 r/min, +1.0% vs baseline)

#### Parameter: `parallel_batch_size`

**Coarse Search (±4 steps):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 20 | coarse_down | 417.5 | -36.6% | ✓ |
| 40 | coarse_up | 557.0 | -15.5% | ✓ |

**Fine Search (±1 step):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 25 | fine_down | 648.7 | -1.5% | ✓ |
| 30 | fine_center | 661.1 | +0.3% | ✓ |
| 35 | fine_up | 945.7 | +43.5% | ✓ |

**Optimal `parallel_batch_size`: 35** (945.7 r/min, +43.5% vs baseline)

#### Parameter: `batch_timeout_ms`

**Coarse Search (±4 steps):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 5 | coarse_down | 449.1 | -31.8% | ✓ |
| 15 | coarse_up | 393.9 | -40.2% | ✓ |

**Fine Search (±1 step):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 9 | fine_down | 666.1 | +1.1% | ✓ |
| 10 | fine_center | 667.0 | +1.2% | ✓ |
| 11 | fine_up | 660.1 | +0.2% | ✓ |

**Optimal `batch_timeout_ms`: 10** (667.0 r/min, +1.2% vs baseline)

### Combination Testing

| Config | Search Type | Rounds/Min | vs Baseline | Status |
|--------|-------------|------------|-------------|--------|
| 32w×35b×9ms | fine_tune | 954.6 | +44.9% | ✓ |
| 32w×35b×10ms | combined | 947.8 | +43.8% | ✓ |
| 32w×40b×10ms | fine_tune | 943.9 | +43.3% | ✓ |
| 32w×35b×11ms | fine_tune | 938.4 | +42.4% | ✓ |
| 31w×35b×10ms | fine_tune | 936.5 | +42.1% | ✓ |
| 33w×35b×10ms | fine_tune | 934.1 | +41.8% | ✓ |
| 32w×30b×10ms | fine_tune | 662.0 | +0.5% | ✓ |

### Final Validation (3 runs × 1000 rounds)

**Final Config**: 32w × 35batch × 3×35 × 9ms

| Run | Rounds/Min |
|-----|------------|
| 1 | 1031.3 |
| 2 | 1028.3 |
| 3 | 1029.1 |

- **Average**: 1029.6 rounds/min
- **Variance**: ±0.1%
- **Improvement vs Baseline**: +56.2%

### Stage Summary

| Metric | Baseline | Optimal | Change |
|--------|----------|---------|--------|
| Performance | 658.9 r/min | 1029.6 r/min | +56.2% |
| Workers | 32 | 32 | +0 |
| Batch Size | 30 | 35 | +5 |
| Timeout (ms) | 10 | 9 | -1 |

**Recommendation**: ✅ **Strong improvement**: 56.2% speedup achieved

---


## Stage 4: 4×45 MCTS (180 total simulations)

### Baseline Configuration

- **Config**: 32w × 30batch × 4×45 × 10ms
- **Performance**: 460.1 rounds/min

### Individual Parameter Optimization

#### Parameter: `workers`

**Coarse Search (±4 steps):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 28 | coarse_down | 357.2 | -22.4% | ✓ |
| 36 | coarse_up | 336.2 | -26.9% | ✓ |

**Fine Search (±1 step):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 31 | fine_down | 456.0 | -0.9% | ✓ |

**Optimal `workers`: 31** (456.0 r/min, -0.9% vs baseline)

### Stage Summary

---

---

## Overall Recommendations

### Key Findings

1. **Best improvement**: Stage 3 achieved +56.2% speedup
2. **Average improvement**: +31.4% across all stages
3. **Stability**: Average variance 0.2% across final validations

### Recommended Configurations

Add to `ml/config.py`:

```python
# MPPT Auto-tuned configurations (generated 2025-11-14)

# Stage 1: 1×15 MCTS
STAGE_1_WORKERS = 31
STAGE_1_BATCH_SIZE = 30
STAGE_1_TIMEOUT_MS = 10

# Stage 2: 2×25 MCTS
STAGE_2_WORKERS = 32
STAGE_2_BATCH_SIZE = 30
STAGE_2_TIMEOUT_MS = 10

# Stage 3: 3×35 MCTS
STAGE_3_WORKERS = 32
STAGE_3_BATCH_SIZE = 35
STAGE_3_TIMEOUT_MS = 9

```
