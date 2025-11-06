# Sweep 1: Parallel Batch Size Optimization

**Status:** Running
**Duration:** ~15-20 minutes (200 games total)
**Strategy:** Exponential spacing to quickly identify trends

---

## What We're Testing

**Parameter:** `parallel_batch_size` - Number of MCTS leaves expanded in parallel per iteration

**Values:** [5, 10, 20, 40] - Exponential spacing (2x increments)

**Fixed parameters:**
- Workers: 32
- MCTS config: Medium (3 det × 30 sims = 90 sims/move)
- Games per config: 50
- Batch timeout: 10ms

---

## Why Exponential Spacing?

### Efficient Coverage
- **Linear** [5, 10, 15, 20, 25, 30]: Dense sampling, takes longer
- **Exponential** [5, 10, 20, 40]: Covers 8x range with fewer points

### Quick Trend Detection
With 4 points exponentially spaced, we can quickly identify:
1. **Linear growth** → Optimal likely at maximum tested (40+)
2. **Peak in middle** → Optimal found, can refine if needed
3. **Plateau** → Insensitive to parameter, any value works
4. **Decline at high values** → Hitting diminishing returns/bottleneck

---

## Expected Outcomes

### Hypothesis 1: Peak at 20-30 (MOST LIKELY)
```
Batch Size:  5     10     20     40
Games/min:  50    68     85     72  ← peaks then drops
```
**Interpretation:** Optimal ~20-25 (sweet spot)
**Next step:** Refine with [15, 20, 25, 30] if needed

### Hypothesis 2: Monotonic increase (OPTIMISTIC)
```
Batch Size:  5     10     20     40
Games/min:  50    68     88    108  ← keeps improving
```
**Interpretation:** Can push higher! Test [40, 60, 80]
**Next step:** Try larger batch sizes

### Hypothesis 3: Plateau (NEUTRAL)
```
Batch Size:  5     10     20     40
Games/min:  50    68     72     73  ← flat after 10
```
**Interpretation:** Relatively insensitive after 10
**Next step:** Use 20 for safety margin

### Hypothesis 4: Early peak (UNEXPECTED)
```
Batch Size:  5     10     20     40
Games/min:  50    75     68     52  ← peaks at 10!
```
**Interpretation:** High batch sizes hurt performance (virtual loss conflicts?)
**Next step:** Refine with [5, 8, 10, 12] to find precise peak

---

## What Comes Next

Based on Sweep 1 results, we'll design targeted follow-ups:

### If optimal found (peak detected):
✅ **Use optimal value for remaining sweeps**
- Sweep 2: Worker scaling @ optimal_batch_size
- Sweep 3: 2D matrix with focused range around optimal

### If monotonic increase (no peak yet):
📈 **Test larger batch sizes**
```bash
python benchmark_session1_validation.py \
  --sweep batch_size \
  --batch-sizes "40,60,80,100" \
  --games 50
```

### If need refinement (peak in tested range):
🔍 **Zoom in around peak**
```bash
# If peak at 20, test neighbors
python benchmark_session1_validation.py \
  --sweep batch_size \
  --batch-sizes "15,20,25,30" \
  --games 50
```

---

## Performance Targets

| Result | Speedup | Next Action |
|--------|---------|-------------|
| **<70 g/min** | <2x | Debug: Check batch sizes in logs |
| **70-90 g/min** | 2-2.5x | Good! Proceed with optimal to Sweep 2 |
| **90-110 g/min** | 2.5-3x | Excellent! Try slightly larger batch sizes |
| **>110 g/min** | >3x | 🎉 Target achieved! Document and proceed |

**Baseline reference:** 36.7 g/min (sequential MCTS, no parallel expansion)

---

## Real-Time Monitoring

Check progress:
```bash
# View running processes
ps aux | grep benchmark

# Monitor output (when ready)
# Results will appear in benchmarks/results/session1_validation_*.csv
```

---

## After Sweep 1 Completes

1. **Visualize results:**
   ```bash
   python benchmarks/performance/visualize_session1_results.py \
     benchmarks/results/session1_validation_*.csv
   ```

2. **Review plots:**
   - `*_batch_size.png` - Performance curve
   - `*_summary.txt` - Optimal batch size recommendation

3. **Decide next steps:**
   - If optimal found → Proceed to Sweep 2 (worker scaling)
   - If need refinement → Run targeted follow-up
   - If monotonic → Test larger batch sizes

4. **Update config:**
   ```python
   # ml/config.py
   parallel_batch_size: int = <optimal_from_sweep>
   ```

---

## Time Estimates

- **Config 1** (batch_size=5): ~3.5 min (50 games)
- **Config 2** (batch_size=10): ~3.5 min (50 games)
- **Config 3** (batch_size=20): ~3.5 min (50 games)
- **Config 4** (batch_size=40): ~3.5 min (50 games)

**Total:** ~15-20 minutes (including warmup/cooldown)

---

## Key Questions to Answer

1. **Where is the optimal batch size?**
   - Below 5? (need smaller values)
   - Around 10-20? (current range is good)
   - Above 40? (need larger values)

2. **How sensitive is performance?**
   - Highly sensitive (sharp peak)
   - Moderately sensitive (broad peak)
   - Insensitive (plateau)

3. **What's the limiting factor?**
   - GPU batch accumulation (favors larger)
   - Virtual loss conflicts (favors smaller)
   - Memory/compute (hard limit)

4. **Is 3x speedup achievable?**
   - 110+ g/min target
   - Current best: 63.6 g/min @ batch_size=10
   - Gap: Need 73% improvement from optimal batch size

---

## Exponential Strategy Benefits

✅ **Faster results** - 4 points vs 6 points (33% time savings)
✅ **Broader coverage** - 8x range vs 6x range
✅ **Clear trends** - Exponential spacing shows scaling behavior
✅ **Adaptive** - Results guide next sweep design

This approach mirrors performance engineering best practices: start broad, then zoom in where needed.
