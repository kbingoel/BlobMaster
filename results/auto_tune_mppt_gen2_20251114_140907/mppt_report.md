# BlobMaster MPPT Auto-Tune Report

**Generated**: 2025-11-14 15:01:30

This report shows MPPT (Maximum Power Point Tracking) style optimization results for each curriculum stage.
Each stage uses coarse-to-fine gradient ascent to find optimal parameters.

## Overview

- **Total experiments**: 114
- **Stages optimized**: 5
- **Successful tests**: 114
- **Failed tests**: 0

## Summary Across All Stages

| Stage | MCTS | Baseline (r/min) | Optimal (r/min) | Improvement | Optimal Config |
|-------|------|------------------|-----------------|-------------|----------------|
| 1 | 1×15 | 1631.8 | 1614.8 | -1.0% | 32w×30b×10ms |
| 2 | 2×25 | 1079.0 | 1075.8 | -0.3% | 32w×25b×10ms |
| 3 | 3×35 | 554.7 | 786.6 | +41.8% | 32w×40b×6ms |
| 4 | 4×45 | 403.5 | 408.8 | +1.3% | 32w×35b×7ms |
| 5 | 5×50 | 319.8 | 318.9 | -0.3% | 32w×35b×11ms |

---

# Detailed Stage Reports


## Stage 1: 1×15 MCTS (15 total simulations)

### Baseline Configuration

- **Config**: 32w × 30batch × 1×15 × 10ms
- **Performance**: 1631.8 rounds/min

### Individual Parameter Optimization

#### Parameter: `parallel_batch_size`

**Coarse Search (±4 steps):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 25 | coarse_down | 944.8 | -42.1% | ✓ |
| 35 | coarse_up | 991.5 | -39.2% | ✓ |

**Fine Search (±1 step):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 25 | fine_initial_down | 1588.1 | -2.7% | ✓ |
| 30 | fine_initial_center | 1657.4 | +1.6% | ✓ |
| 35 | fine_initial_up | 1605.8 | -1.6% | ✓ |

**Optimal `parallel_batch_size`: 30** (1657.4 r/min, +1.6% vs baseline)

#### Parameter: `batch_timeout_ms`

**Coarse Search (±4 steps):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 5 | coarse_down | 998.4 | -38.8% | ✓ |
| 15 | coarse_up | 911.8 | -44.1% | ✓ |

**Fine Search (±1 step):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 9 | fine_initial_down | 1590.2 | -2.5% | ✓ |
| 10 | fine_initial_center | 1605.8 | -1.6% | ✓ |
| 11 | fine_initial_up | 1590.8 | -2.5% | ✓ |

**Optimal `batch_timeout_ms`: 10** (1605.8 r/min, -1.6% vs baseline)

### Combination Testing

| Config | Search Type | Rounds/Min | vs Baseline | Status |
|--------|-------------|------------|-------------|--------|
| 32w×25b×10ms | fine_tune | 1625.3 | -0.4% | ✓ |
| 32w×30b×9ms | fine_tune | 1605.7 | -1.6% | ✓ |
| 32w×30b×10ms | combined | 1601.9 | -1.8% | ✓ |
| 32w×35b×10ms | fine_tune | 1583.3 | -3.0% | ✓ |
| 32w×30b×11ms | fine_tune | 1548.7 | -5.1% | ✓ |

### Final Validation (3 runs × 1000 rounds)

**Final Config**: 32w × 30batch × 1×15 × 10ms

| Run | Rounds/Min |
|-----|------------|
| 1 | 1617.9 |
| 2 | 1598.1 |
| 3 | 1628.4 |

- **Average**: 1614.8 rounds/min
- **Variance**: ±0.8%
- **Improvement vs Baseline**: -1.0%

### Stage Summary

| Metric | Baseline | Optimal | Change |
|--------|----------|---------|--------|
| Performance | 1631.8 r/min | 1614.8 r/min | -1.0% |
| Workers | 32 | 32 | +0 |
| Batch Size | 30 | 30 | +0 |
| Timeout (ms) | 10 | 10 | +0 |

**Recommendation**: ⚠️ **No improvement**: Baseline configuration is optimal

---


## Stage 2: 2×25 MCTS (50 total simulations)

### Baseline Configuration

- **Config**: 32w × 30batch × 2×25 × 10ms
- **Performance**: 1079.0 rounds/min

### Individual Parameter Optimization

#### Parameter: `parallel_batch_size`

**Coarse Search (±4 steps):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 25 | coarse_down | 743.4 | -31.1% | ✓ |
| 35 | coarse_up | 722.2 | -33.1% | ✓ |

**Fine Search (±1 step):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 15 | fine_verify_down | 779.3 | -27.8% | ✓ |
| 20 | fine_climb_down | 788.2 | -27.0% | ✓ |
| 25 | fine_initial_down | 1101.8 | +2.1% | ✓ |
| 30 | fine_initial_center | 1083.5 | +0.4% | ✓ |
| 35 | fine_initial_up | 1068.0 | -1.0% | ✓ |

**Optimal `parallel_batch_size`: 25** (1101.8 r/min, +2.1% vs baseline)

#### Parameter: `batch_timeout_ms`

**Coarse Search (±4 steps):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 5 | coarse_down | 751.5 | -30.3% | ✓ |
| 15 | coarse_up | 690.9 | -36.0% | ✓ |

**Fine Search (±1 step):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 7 | fine_verify_down | 1071.4 | -0.7% | ✓ |
| 8 | fine_climb_down | 1077.2 | -0.2% | ✓ |
| 9 | fine_initial_down | 1086.3 | +0.7% | ✓ |
| 10 | fine_initial_center | 1066.5 | -1.2% | ✓ |
| 11 | fine_initial_up | 1085.6 | +0.6% | ✓ |

**Optimal `batch_timeout_ms`: 9** (1086.3 r/min, +0.7% vs baseline)

### Combination Testing

| Config | Search Type | Rounds/Min | vs Baseline | Status |
|--------|-------------|------------|-------------|--------|
| 32w×30b×9ms | fine_tune | 1091.9 | +1.2% | ✓ |
| 32w×25b×10ms | fine_tune | 1083.7 | +0.4% | ✓ |
| 32w×25b×9ms | combined | 1079.1 | +0.0% | ✓ |
| 32w×25b×8ms | fine_tune | 1065.8 | -1.2% | ✓ |
| 32w×20b×9ms | fine_tune | 803.0 | -25.6% | ✓ |

### Final Validation (3 runs × 1000 rounds)

**Final Config**: 32w × 25batch × 2×25 × 10ms

| Run | Rounds/Min |
|-----|------------|
| 1 | 1077.4 |
| 2 | 1082.4 |
| 3 | 1067.6 |

- **Average**: 1075.8 rounds/min
- **Variance**: ±0.6%
- **Improvement vs Baseline**: -0.3%

### Stage Summary

| Metric | Baseline | Optimal | Change |
|--------|----------|---------|--------|
| Performance | 1079.0 r/min | 1075.8 r/min | -0.3% |
| Workers | 32 | 32 | +0 |
| Batch Size | 30 | 25 | -5 |
| Timeout (ms) | 10 | 10 | +0 |

**Recommendation**: ⚠️ **No improvement**: Baseline configuration is optimal

---


## Stage 3: 3×35 MCTS (105 total simulations)

### Baseline Configuration

- **Config**: 32w × 30batch × 3×35 × 10ms
- **Performance**: 554.7 rounds/min

### Individual Parameter Optimization

#### Parameter: `parallel_batch_size`

**Coarse Search (±4 steps):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 25 | coarse_down | 422.5 | -23.8% | ✓ |
| 35 | coarse_up | 569.5 | +2.7% | ✓ |
| 40 | coarse_up | 574.0 | +3.5% | ✓ |
| 45 | coarse_up | 571.1 | +3.0% | ✓ |

**Fine Search (±1 step):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 25 | fine_verify_down | 542.6 | -2.2% | ✓ |
| 30 | fine_climb_down | 554.6 | -0.0% | ✓ |
| 35 | fine_initial_down | 777.2 | +40.1% | ✓ |
| 40 | fine_initial_center | 769.8 | +38.8% | ✓ |
| 45 | fine_initial_up | 774.8 | +39.7% | ✓ |

**Optimal `parallel_batch_size`: 35** (777.2 r/min, +40.1% vs baseline)

#### Parameter: `batch_timeout_ms`

**Coarse Search (±4 steps):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 5 | coarse_down | 446.5 | -19.5% | ✓ |
| 15 | coarse_up | 402.2 | -27.5% | ✓ |

**Fine Search (±1 step):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 4 | fine_verify_down | 562.4 | +1.4% | ✓ |
| 5 | fine_climb_down | 563.9 | +1.7% | ✓ |
| 6 | fine_climb_down | 570.2 | +2.8% | ✓ |
| 7 | fine_verify_down | 564.7 | +1.8% | ✓ |
| 8 | fine_climb_down | 552.7 | -0.4% | ✓ |
| 9 | fine_initial_down | 559.8 | +0.9% | ✓ |
| 10 | fine_initial_center | 558.9 | +0.8% | ✓ |
| 11 | fine_initial_up | 553.4 | -0.2% | ✓ |

**Optimal `batch_timeout_ms`: 6** (570.2 r/min, +2.8% vs baseline)

### Combination Testing

| Config | Search Type | Rounds/Min | vs Baseline | Status |
|--------|-------------|------------|-------------|--------|
| 32w×40b×6ms | fine_tune | 792.3 | +42.8% | ✓ |
| 32w×35b×6ms | combined | 790.4 | +42.5% | ✓ |
| 32w×35b×7ms | fine_tune | 777.6 | +40.2% | ✓ |
| 32w×35b×5ms | fine_tune | 766.8 | +38.2% | ✓ |
| 32w×30b×6ms | fine_tune | 565.8 | +2.0% | ✓ |

### Final Validation (3 runs × 1000 rounds)

**Final Config**: 32w × 40batch × 3×35 × 6ms

| Run | Rounds/Min |
|-----|------------|
| 1 | 785.4 |
| 2 | 782.4 |
| 3 | 792.0 |

- **Average**: 786.6 rounds/min
- **Variance**: ±0.5%
- **Improvement vs Baseline**: +41.8%

### Stage Summary

| Metric | Baseline | Optimal | Change |
|--------|----------|---------|--------|
| Performance | 554.7 r/min | 786.6 r/min | +41.8% |
| Workers | 32 | 32 | +0 |
| Batch Size | 30 | 40 | +10 |
| Timeout (ms) | 10 | 6 | -4 |

**Recommendation**: ✅ **Strong improvement**: 41.8% speedup achieved

---


## Stage 4: 4×45 MCTS (180 total simulations)

### Baseline Configuration

- **Config**: 32w × 30batch × 4×45 × 10ms
- **Performance**: 403.5 rounds/min

### Individual Parameter Optimization

#### Parameter: `parallel_batch_size`

**Coarse Search (±4 steps):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 25 | coarse_down | 326.7 | -19.0% | ✓ |
| 35 | coarse_up | 332.3 | -17.6% | ✓ |

**Fine Search (±1 step):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 25 | fine_initial_down | 401.3 | -0.5% | ✓ |
| 30 | fine_initial_center | 404.6 | +0.3% | ✓ |
| 35 | fine_initial_up | 403.6 | +0.0% | ✓ |

**Optimal `parallel_batch_size`: 30** (404.6 r/min, +0.3% vs baseline)

#### Parameter: `batch_timeout_ms`

**Coarse Search (±4 steps):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 5 | coarse_down | 337.4 | -16.4% | ✓ |
| 15 | coarse_up | 315.7 | -21.8% | ✓ |

**Fine Search (±1 step):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 5 | fine_verify_down | 402.0 | -0.4% | ✓ |
| 6 | fine_climb_down | 405.2 | +0.4% | ✓ |
| 7 | fine_climb_down | 410.0 | +1.6% | ✓ |
| 8 | fine_climb_down | 408.9 | +1.4% | ✓ |
| 9 | fine_initial_down | 408.8 | +1.3% | ✓ |
| 10 | fine_initial_center | 401.7 | -0.4% | ✓ |
| 11 | fine_initial_up | 403.6 | +0.0% | ✓ |

**Optimal `batch_timeout_ms`: 7** (410.0 r/min, +1.6% vs baseline)

### Combination Testing

| Config | Search Type | Rounds/Min | vs Baseline | Status |
|--------|-------------|------------|-------------|--------|
| 32w×35b×7ms | fine_tune | 410.8 | +1.8% | ✓ |
| 32w×30b×6ms | fine_tune | 406.8 | +0.8% | ✓ |
| 32w×30b×8ms | fine_tune | 405.6 | +0.5% | ✓ |
| 32w×30b×7ms | combined | 404.8 | +0.3% | ✓ |
| 32w×25b×7ms | fine_tune | 397.3 | -1.5% | ✓ |

### Final Validation (3 runs × 1000 rounds)

**Final Config**: 32w × 35batch × 4×45 × 7ms

| Run | Rounds/Min |
|-----|------------|
| 1 | 413.5 |
| 2 | 403.6 |
| 3 | 409.3 |

- **Average**: 408.8 rounds/min
- **Variance**: ±1.0%
- **Improvement vs Baseline**: +1.3%

### Stage Summary

| Metric | Baseline | Optimal | Change |
|--------|----------|---------|--------|
| Performance | 403.5 r/min | 408.8 r/min | +1.3% |
| Workers | 32 | 32 | +0 |
| Batch Size | 30 | 35 | +5 |
| Timeout (ms) | 10 | 7 | -3 |

**Recommendation**: ○ **Marginal improvement**: 1.3% speedup achieved

---


## Stage 5: 5×50 MCTS (250 total simulations)

### Baseline Configuration

- **Config**: 32w × 30batch × 5×50 × 10ms
- **Performance**: 319.8 rounds/min

### Individual Parameter Optimization

#### Parameter: `parallel_batch_size`

**Coarse Search (±4 steps):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 25 | coarse_down | 271.6 | -15.1% | ✓ |
| 35 | coarse_up | 273.7 | -14.4% | ✓ |

**Fine Search (±1 step):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 25 | fine_initial_down | 319.8 | -0.0% | ✓ |
| 30 | fine_initial_center | 320.2 | +0.1% | ✓ |
| 35 | fine_initial_up | 319.3 | -0.2% | ✓ |

**Optimal `parallel_batch_size`: 30** (320.2 r/min, +0.1% vs baseline)

#### Parameter: `batch_timeout_ms`

**Coarse Search (±4 steps):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 5 | coarse_down | 277.2 | -13.3% | ✓ |
| 15 | coarse_up | 257.3 | -19.5% | ✓ |

**Fine Search (±1 step):**

| Value | Search Type | Rounds/Min | vs Baseline | Status |
|-------|-------------|------------|-------------|--------|
| 9 | fine_initial_down | 321.1 | +0.4% | ✓ |
| 10 | fine_initial_center | 317.3 | -0.8% | ✓ |
| 11 | fine_initial_up | 322.5 | +0.8% | ✓ |
| 12 | fine_climb_up | 322.0 | +0.7% | ✓ |
| 13 | fine_verify_up | 317.6 | -0.7% | ✓ |

**Optimal `batch_timeout_ms`: 11** (322.5 r/min, +0.8% vs baseline)

### Combination Testing

| Config | Search Type | Rounds/Min | vs Baseline | Status |
|--------|-------------|------------|-------------|--------|
| 32w×35b×11ms | fine_tune | 324.0 | +1.3% | ✓ |
| 32w×30b×11ms | combined | 323.6 | +1.2% | ✓ |
| 32w×30b×10ms | fine_tune | 323.5 | +1.1% | ✓ |
| 32w×30b×12ms | fine_tune | 322.4 | +0.8% | ✓ |
| 32w×25b×11ms | fine_tune | 321.1 | +0.4% | ✓ |

### Final Validation (3 runs × 1000 rounds)

**Final Config**: 32w × 35batch × 5×50 × 11ms

| Run | Rounds/Min |
|-----|------------|
| 1 | 321.3 |
| 2 | 318.0 |
| 3 | 317.4 |

- **Average**: 318.9 rounds/min
- **Variance**: ±0.5%
- **Improvement vs Baseline**: -0.3%

### Stage Summary

| Metric | Baseline | Optimal | Change |
|--------|----------|---------|--------|
| Performance | 319.8 r/min | 318.9 r/min | -0.3% |
| Workers | 32 | 32 | +0 |
| Batch Size | 30 | 35 | +5 |
| Timeout (ms) | 10 | 11 | +1 |

**Recommendation**: ⚠️ **No improvement**: Baseline configuration is optimal

---

---

## Overall Recommendations

### Key Findings

1. **Best improvement**: Stage 3 achieved +41.8% speedup
2. **Average improvement**: +8.3% across all stages
3. **Stability**: Average variance 0.7% across final validations

### Recommended Configurations

Add to `ml/config.py`:

```python
# MPPT Auto-tuned configurations (generated 2025-11-14)

# Stage 1: 1×15 MCTS
STAGE_1_WORKERS = 32
STAGE_1_BATCH_SIZE = 30
STAGE_1_TIMEOUT_MS = 10

# Stage 2: 2×25 MCTS
STAGE_2_WORKERS = 32
STAGE_2_BATCH_SIZE = 25
STAGE_2_TIMEOUT_MS = 10

# Stage 3: 3×35 MCTS
STAGE_3_WORKERS = 32
STAGE_3_BATCH_SIZE = 40
STAGE_3_TIMEOUT_MS = 6

# Stage 4: 4×45 MCTS
STAGE_4_WORKERS = 32
STAGE_4_BATCH_SIZE = 35
STAGE_4_TIMEOUT_MS = 7

# Stage 5: 5×50 MCTS
STAGE_5_WORKERS = 32
STAGE_5_BATCH_SIZE = 35
STAGE_5_TIMEOUT_MS = 11

```
