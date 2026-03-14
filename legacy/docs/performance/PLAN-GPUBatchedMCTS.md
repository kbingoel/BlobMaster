# GPU-Batched MCTS Implementation Plan

**Goal**: Batch neural network inference calls during MCTS to achieve 10-100x speedup.

**Problem**: Current architecture calls network 90 times per move sequentially (batch=1), resulting in ~1% GPU utilization.

**Solution**: Collect evaluation requests and batch them into single GPU calls.

---

## Expected Results

- **Current**: 18.8 games/min → 196 days for 500 iterations
- **Phase 1 (Intra-game batching)**: ~200 games/min → 20 days training
- **Phase 2 (Multi-game batching)**: ~500-2000 games/min → 8-15 days training

---

## Three-Phase Implementation

### **Phase 1: Virtual Loss + Intra-Game Batching** (~2 hours)

**What**: Batch all 90 evaluations within a single MCTS search.

**How**:
1. Add `virtual_losses` field to `MCTSNode` to prevent duplicate evaluations
2. Modify `PerfectInfoMCTS` to collect leaf nodes before evaluation
3. Create `search_batched()` method that:
   - Runs simulations in batches (e.g., 90 at a time)
   - Collects all leaf nodes needing evaluation
   - Makes single batched network call for all nodes
   - Backpropagates results

**Files Modified**:
- `ml/mcts/node.py`: Add virtual_losses, update UCB calculation
- `ml/mcts/search.py`: Add search_batched(), _expand_and_evaluate_batch()

**Expected Speedup**: 10x (90 sequential calls → 1 batched call)

---

### **Phase 2: Multi-Game Batching** (~2 hours)

**What**: Batch evaluations across all 16 games running in parallel.

**How**:
1. Create `BatchedEvaluator` service (new file)
   - Background thread that collects requests from all MCTS instances
   - Accumulates requests until batch is full or timeout (10ms)
   - Performs single GPU call for entire batch (512-2048 samples)
   - Returns results to requesters

2. Integrate with `SelfPlayEngine`
   - Create shared `BatchedEvaluator` instance
   - Pass to all MCTS instances

3. Update MCTS to use evaluator
   - Check if batch_evaluator is available
   - Send request and wait for result
   - Fall back to direct inference if no evaluator

**Files Created**:
- `ml/mcts/batch_evaluator.py`: New centralized evaluation service

**Files Modified**:
- `ml/training/selfplay.py`: Create and manage BatchedEvaluator
- `ml/mcts/search.py`: Use BatchedEvaluator if available

**Expected Speedup**: 50-100x total (saturate GPU like training does)

---

### **Phase 3: Testing & Optimization** (~2 hours)

**Correctness Tests**:
- Compare batched vs sequential MCTS results
- Verify visit counts and value estimates match
- Ensure legal moves only

**Performance Benchmarks**:
- Measure actual speedup with different batch sizes
- Test: 128, 256, 512, 1024, 2048
- Monitor GPU utilization (target >70%)

**Hyperparameter Tuning**:
- Batch timeout: Test 5ms, 10ms, 20ms, 50ms
- Number of workers: Test 4, 8, 16, 32
- Virtual loss weight: Test 1, 3, 5, 10

**Files Created**:
- `ml/mcts/test_batched_mcts.py`: Correctness tests
- `ml/benchmark_batched_mcts.py`: Performance benchmarks

---

## Key Implementation Details

### Virtual Loss Mechanism

```python
# In MCTSNode
self.virtual_losses = 0  # Discourage concurrent selection

# In UCB calculation
adjusted_value = self.value_sum / (self.visit_count + self.virtual_losses)

# During selection
node.virtual_losses += 1  # Mark as "in progress"

# After backpropagation
node.virtual_losses -= 1  # Release
```

### Batched Evaluation Flow

```python
# Collect nodes
leaf_nodes = []
for sim in range(batch_size):
    leaf = select_with_virtual_loss(root)
    leaf_nodes.append(leaf)

# Batch evaluate
state_batch = torch.stack([encode(n) for n in leaf_nodes])
policies, values = network(state_batch)  # Single GPU call!

# Distribute results
for node, policy, value in zip(leaf_nodes, policies, values):
    node.expand(policy)
    backpropagate(node, value)
```

### BatchedEvaluator Pattern

```python
class BatchedEvaluator:
    def evaluate(self, state, mask, actions):
        # Put request in queue
        request_queue.put(Request(state, mask, actions, result_queue))

        # Wait for result
        return result_queue.get()  # Blocking

    def _evaluation_loop(self):  # Background thread
        while running:
            batch = collect_batch(timeout=10ms)
            results = network(batch)  # Single GPU call
            distribute_results(results)
```

---

## Implementation Checklist

### Phase 1 (~2 hours)
- [ ] Add virtual_losses to MCTSNode
- [ ] Update get_ucb_score() with virtual loss
- [ ] Create search_batched() method
- [ ] Implement _expand_and_evaluate_batch()
- [ ] Test with batch=90, verify 10x speedup

### Phase 2 (~2 hours)
- [ ] Create ml/mcts/batch_evaluator.py
- [ ] Implement BatchedEvaluator class
- [ ] Integrate with SelfPlayEngine
- [ ] Update MCTS to use BatchedEvaluator
- [ ] Test with batch=512, verify 50x+ speedup

### Phase 3 (~2 hours)
- [ ] Write correctness tests
- [ ] Run performance benchmarks
- [ ] Tune hyperparameters
- [ ] Measure GPU utilization
- [ ] Document results

---

## Success Criteria

**Minimum (Phase 1)**:
- 10x speedup (200 games/min)
- Training time <30 days
- Correctness verified

**Target (Phase 2)**:
- 50x speedup (500-1000 games/min)
- Training time <20 days
- GPU utilization >70%

**Stretch**:
- 100x speedup (2000 games/min)
- Training time <10 days
- GPU utilization >90%

---

## Quick Win Alternative

**If time is limited**: Implement only Phase 1 (~2 hours)
- Still get 10x speedup
- Training becomes practical (<30 days)
- Add Phase 2 later if needed

---

## Files Summary

**New Files** (Phase 2):
- `ml/mcts/batch_evaluator.py` (~200 lines)
- `ml/mcts/test_batched_mcts.py` (~100 lines)
- `ml/benchmark_batched_mcts.py` (~150 lines)

**Modified Files**:
- `ml/mcts/node.py` (~10 lines)
- `ml/mcts/search.py` (~200 lines)
- `ml/training/selfplay.py` (~20 lines)

**Total**: ~680 lines over 6 hours

---

**Next Session**: Start with Phase 1 implementation
