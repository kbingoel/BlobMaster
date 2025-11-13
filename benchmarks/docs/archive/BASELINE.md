# Performance Baseline (2025-11-13)

**Official performance baseline** for BlobMaster Phase 1 training on Ubuntu Linux.

## Test Configuration

**Command:**
```bash
source venv/bin/activate
python benchmarks/performance/benchmark_optimal_config.py --games 500 --test-mcts-variants
```

**System:**
- **OS**: Ubuntu 24.04 LTS (Linux 6.14.0-35-generic)
- **CPU**: AMD Ryzen 9 7950X (16-core)
- **GPU**: NVIDIA GeForce RTX 4060 8GB
- **RAM**: 128GB DDR5
- **Python**: 3.14.0
- **PyTorch**: CUDA 12.4
- **Date**: 2025-11-13

**Test Parameters:**
- Workers: 32 (multiprocessing)
- Sample size: 500 rounds (main), 200 rounds (MCTS variants)
- Parallel expansion: ENABLED
- Parallel batch size: 30
- Batched evaluator: 512 max batch, 10ms timeout
- Cards per round: 5
- Training mode: Independent rounds (Phase 1)

## Results

**Training Mode**: Independent 5-card rounds

| MCTS Config | Det × Sims | Total Sims | Rounds/Min | Training Time* |
|-------------|------------|------------|------------|----------------|
| **Light**   | 2 × 20     | 40         | **1,049**  | **3.3 days**   |
| **Medium**  | 3 × 30     | 90         | **741**    | **4.7 days**   |
| **Heavy**   | 5 × 50     | 250        | **310**    | **11.2 days**  |

*Training time calculated for: 500 iterations × 10,000 rounds = 5M training rounds

**Examples per round**: 20 (zero-choice fast path working correctly)

## Files

- **JSON results**: [BASELINE_2025-11-13.json](BASELINE_2025-11-13.json)
- **CSV results**: [BASELINE_2025-11-13.csv](BASELINE_2025-11-13.csv)
- **Full log**: [BASELINE_2025-11-13.log](BASELINE_2025-11-13.log)

## Recommendations

**Use Medium MCTS (741 rounds/min)** for production training:
- Balanced quality vs speed
- Conservative estimate (accounts for training overhead)
- Tested across 500 rounds for statistical stability

## Historical Context

**Previous estimates:**
- ~360-380 rounds/min (conservative estimate from earlier code, Oct 2025)
- ~75 rounds/min (before optimization, Nov 6 2025)

**Recent improvements:**
- Zero-choice fast path (skips MCTS for forced last-card plays)
- Parallel expansion optimization
- Determinization sampling improvements
- Overall: **10-20x speedup** over October baseline

## Notes

- This benchmark measures **self-play generation only**
- Actual training includes additional overhead:
  - Network training epochs (~10 min per iteration)
  - Model evaluation (~5 min every 5 iterations)
  - Checkpointing and logging
- **Conservative recommendation**: Add 10-20% to estimates for full training pipeline
