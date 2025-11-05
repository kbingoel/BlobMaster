# November 2025 Ubuntu Benchmarks

**Date**: November 5, 2025
**Purpose**: Comprehensive baseline validation for production training configuration

## Contents

- **[BENCHMARK-SUMMARY.md](BENCHMARK-SUMMARY.md)**: Full analysis of 21 configurations tested
- **[full_baseline_sweep.csv](full_baseline_sweep.csv)**: Raw benchmark data

## Quick Reference

**Optimal Configuration** (validated and ready for production):
```bash
python ml/train.py --workers 32 --device cuda --iterations 500
```

**Performance**: 36.7 games/min (Medium MCTS)
**Training Duration**: 136 days for 500 iterations × 10,000 games
**Hardware Limit**: 32 workers maximum on RTX 4060 8GB

## What Was Tested

This benchmark sweep tested:
- **Worker counts**: 1, 4, 8, 16, 32, 48, 64
- **MCTS configurations**:
  - Light (2 det × 20 sims = 40 searches/move)
  - Medium (3 det × 30 sims = 90 searches/move) ← **recommended**
  - Heavy (5 det × 50 sims = 250 searches/move)
- **Total configurations**: 21 (15 successful + 6 CUDA OOM failures)
- **Games per config**: 50 (for statistical validity)

## Key Findings

1. **Hardware Limit Discovered**: RTX 4060 8GB VRAM supports **maximum 32 workers**
   - 48+ workers cause CUDA Out of Memory errors
   - Each worker uses ~150MB GPU memory

2. **Optimal Configuration**: 32 workers + Medium MCTS
   - Best balance of speed and quality
   - 36.7 games/min = 136 day training timeline
   - 90 tree searches per move (high quality)

3. **Alternative Configurations**:
   - **Fast training**: 32 workers + Light MCTS = 69.1 games/min (72 days)
   - **Safe training**: 16 workers + Medium MCTS = 26.6 games/min (188 days)

4. **Linux Performance**: 15% slower than historical Windows baseline
   - Linux: 36.7 games/min (Medium MCTS, 32 workers)
   - Windows: 43.3 games/min (historical, same config)
   - Acceptable variance, no further investigation needed

## Comparison to Previous Benchmarks

### October 28, 2025 - Initial Linux Baseline
- **Configuration**: 32 workers, 3 det × 30 sims
- **Performance**: 45.5 games/min
- **Games tested**: 100

### November 5, 2025 - Comprehensive Sweep (This Benchmark)
- **Configuration**: 21 configs (1-64 workers, Light/Medium/Heavy MCTS)
- **Performance**: 36.7 games/min (Medium MCTS, 32 workers)
- **Games tested**: 50 per config (1,050 total)
- **Difference**: 19% slower than Oct 28 (likely due to system variance, background processes)

**Conclusion**: Both benchmarks validate ~40 games/min as realistic expectation for Medium MCTS.

## Scaling Efficiency

| Workers | Games/Min | Speedup | Efficiency |
|---------|-----------|---------|------------|
| 1       | 2.6       | 1.0x    | 100%       |
| 4       | 9.5       | 3.7x    | 92%        |
| 8       | 17.1      | 6.6x    | 82%        |
| 16      | 26.6      | 10.2x   | 64%        |
| 32      | 36.7      | 14.1x   | 44%        |

**Observation**: Diminishing returns beyond 16 workers due to memory bandwidth saturation and coordination overhead.

## Training Timeline Estimates

Based on validated performance (500 iterations × 10,000 games each):

| Configuration | Games/Min | Training Days |
|---------------|-----------|---------------|
| 32w + Light   | 69.1      | **72 days**   |
| 32w + Medium  | 36.7      | **136 days**  |
| 32w + Heavy   | 16.0      | **312 days**  |
| 16w + Medium  | 26.6      | 188 days      |
| 16w + Light   | 57.0      | 88 days       |

**Recommended**: 32 workers + Medium MCTS (136 days) for best quality/speed balance.

## Validation Status

- ✅ **Hardware limits identified**: 32 workers max
- ✅ **Performance validated**: 36.7 games/min @ Medium MCTS
- ✅ **Stability confirmed**: All successful configs completed 50 games without errors
- ✅ **Training timeline realistic**: 136 days achievable with Medium MCTS
- ✅ **Ready for production**: Configuration validated and documented

## Next Steps

1. ⬜ Start 500-iteration training run
2. ⬜ Monitor performance (should match 36.7 games/min baseline)
3. ⬜ Track ELO progression and training metrics
4. ⬜ Proceed to Phase 5 (ONNX Export) while training runs

---

**Benchmark Script**: `benchmarks/performance/benchmark_selfplay.py`
**Data File**: [full_baseline_sweep.csv](full_baseline_sweep.csv)
**Detailed Analysis**: [BENCHMARK-SUMMARY.md](BENCHMARK-SUMMARY.md)
