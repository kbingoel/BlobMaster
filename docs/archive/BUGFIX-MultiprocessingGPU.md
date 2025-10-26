# CRITICAL BUG: Multiprocessing Workers Not Using GPU

**Status**: 🚨 **BLOCKING** - Prevents GPU acceleration for self-play
**Severity**: Critical
**Discovery Date**: 2025-10-25
**Phase**: 4 (Training Pipeline)

---

## Executive Summary

**The self-play workers are NOT using the GPU even when `--device cuda` is specified.**

This causes:
- ❌ GPU sits idle during self-play (0% utilization)
- ❌ All inference happens on CPU (100% CPU usage)
- ❌ Performance is 24x slower than expected
- ❌ Benchmark results are completely invalid

**Impact**: Training would take **~280 days** instead of the potential **~12 days** with proper GPU usage.

**Root Cause**: Device information is not passed to multiprocessing workers, so worker networks default to CPU.

---

## Evidence

### Observed Behavior (from benchmarking)

1. **GPU Behavior**:
   - VRAM usage: 2.9 GB (constant) - only main process network loaded
   - GPU utilization: 0% during self-play
   - GPU only active during training phase

2. **CPU Behavior**:
   - CPU usage: 100% during self-play
   - Pattern: Sustained high usage (workers doing CPU inference)

3. **Performance**:
   - CPU self-play: 12.6 games/min
   - "GPU" self-play: 15.0 games/min (actually CPU + overhead!)
   - GPU training: 74,115 examples/sec (works correctly!)

4. **Comparison**:
   ```
   Training (GPU):     74k examples/sec  ✅ GPU utilized
   Self-play (GPU?):   15 games/min      ❌ GPU NOT utilized
   ```

---

## Root Cause Analysis

### Location: `ml/training/selfplay.py`

#### Problem Code (lines 546-621)

```python
def _worker_generate_games_static(
    worker_id: int,
    num_games: int,
    num_players: int,
    cards_to_deal: int,
    network_state: Dict[str, torch.Tensor],  # ❌ No device info!
    num_determinizations: int,
    simulations_per_determinization: int,
    temperature_schedule: Optional[Callable[[int], float]],
) -> List[Dict[str, Any]]:
    """Static worker function for parallel game generation."""

    # ... (lines 576-596: infer network architecture)

    # Create network for this worker
    network = BlobNet(
        state_dim=state_dim,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
    )
    network.load_state_dict(network_state)
    network.eval()

    # ❌ BUG: Network is never moved to GPU!
    # ❌ Defaults to CPU regardless of --device flag

    # Create encoder and masker
    encoder = StateEncoder()
    masker = ActionMasker()

    # Create SelfPlayWorker with CPU network
    worker = SelfPlayWorker(
        network=network,  # ❌ On CPU!
        encoder=encoder,
        masker=masker,
        ...
    )
```

### Why This Happens

1. **Main Process** (benchmark):
   - Creates network on specified device: `network.to(device)`
   - Extracts state dict: `network.state_dict()`
   - Passes state dict to workers via multiprocessing

2. **Worker Process**:
   - Receives state dict (tensors are serialized, device info lost)
   - Creates fresh network instance
   - Loads state dict: `network.load_state_dict(network_state)`
   - **Network defaults to CPU** (no `.to(device)` call)
   - All inference happens on CPU

3. **Result**:
   - Main process network sits on GPU (2.9 GB VRAM)
   - Worker networks run on CPU (100% CPU usage)
   - GPU is never utilized for self-play inference

---

## Impact Analysis

### Current Performance (CPU-only)

From benchmarks:
- Self-play: **15 games/min** (with 16 workers)
- Per iteration: 10,000 games = **667 minutes = 11.1 hours**
- Training: 30 minutes
- **Total per iteration: ~11.5 hours**
- **500 iterations: ~240 days** 🚨

### Expected Performance (GPU-fixed)

Conservative estimates based on training benchmark GPU speedup (3.5x):
- Self-play: **52 games/min** (15 × 3.5)
- Per iteration: 10,000 games = **192 minutes = 3.2 hours**
- Training: 30 minutes
- **Total per iteration: ~3.5 hours**
- **500 iterations: ~73 days** ⚠️

Optimistic estimates (GPU should be 10-50x faster for small batches):
- Self-play: **150-200 games/min**
- Per iteration: **50-67 minutes**
- Training: 30 minutes
- **Total per iteration: ~80-97 minutes**
- **500 iterations: ~28-34 days** ✅

### GPU-Batched MCTS (future optimization)

With proper GPU usage + batching:
- Self-play: **500-2,000 games/min** (saturate GPU like training does)
- Per iteration: **5-20 minutes**
- Training: 30 minutes
- **Total per iteration: ~35-50 minutes**
- **500 iterations: ~12-17 days** 🚀

---

## The Fix

### Required Changes

#### 1. Pass Device to Workers

**File**: `ml/training/selfplay.py`

**Change**: Add `device` parameter to worker function signature

```python
def _worker_generate_games_static(
    worker_id: int,
    num_games: int,
    num_players: int,
    cards_to_deal: int,
    network_state: Dict[str, torch.Tensor],
    num_determinizations: int,
    simulations_per_determinization: int,
    temperature_schedule: Optional[Callable[[int], float]],
    device: str = "cpu",  # ✅ NEW PARAMETER
) -> List[Dict[str, Any]]:
    """Static worker function for parallel game generation."""

    # ... (architecture inference code)

    # Create network
    network = BlobNet(...)
    network.load_state_dict(network_state)
    network.to(device)  # ✅ MOVE TO SPECIFIED DEVICE
    network.eval()

    # ... (rest of code)
```

#### 2. Update SelfPlayEngine to Pass Device

**File**: `ml/training/selfplay.py`

**Change**: Store device in `SelfPlayEngine.__init__` and pass to workers

```python
class SelfPlayEngine:
    def __init__(
        self,
        network: BlobNet,
        encoder: StateEncoder,
        masker: ActionMasker,
        num_workers: int = 16,
        num_determinizations: int = 3,
        simulations_per_determinization: int = 30,
        temperature_schedule: Optional[Callable[[int], float]] = None,
        device: str = "cpu",  # ✅ NEW PARAMETER
    ):
        # ... existing code ...
        self.device = device  # ✅ STORE DEVICE

        # Get network state dict
        self.network_state = network.state_dict()
```

**Change**: Update `generate_games` to pass device to workers

```python
def generate_games(self, ...):
    # ... existing code ...

    # Create tasks for each worker
    tasks = []
    for worker_id in range(self.num_workers):
        worker_games = games_per_worker + (1 if worker_id < remaining_games else 0)
        if worker_games > 0:
            tasks.append(
                (
                    worker_id,
                    worker_games,
                    num_players,
                    cards_to_deal,
                    self.network_state,
                    self.num_determinizations,
                    self.simulations_per_determinization,
                    self.temperature_schedule,
                    self.device,  # ✅ PASS DEVICE TO WORKER
                )
            )

    # Execute tasks in parallel
    results = self.pool.starmap(_worker_generate_games_static, tasks)
```

#### 3. Update All Callers

**Files to update**:
- `ml/benchmark_selfplay.py` - pass device to `SelfPlayEngine`
- `ml/benchmark_iteration.py` - pass device to `SelfPlayEngine`
- `ml/training/trainer.py` (`TrainingPipeline`) - pass device to `SelfPlayEngine`

**Example** (`benchmark_selfplay.py`):

```python
# Create self-play engine
engine = SelfPlayEngine(
    network=self.network,
    encoder=self.encoder,
    masker=self.masker,
    num_workers=num_workers,
    num_determinizations=num_determinizations,
    simulations_per_determinization=simulations_per_det,
    device=self.device,  # ✅ PASS DEVICE
)
```

---

## Validation Plan

After implementing the fix, validate with these tests:

### Test 1: GPU Utilization Check
```bash
# Run self-play benchmark with GPU
python -m ml.benchmark_selfplay --device cuda --quick --games 20

# Expected:
# - GPU utilization: 70-100% (in task manager)
# - VRAM usage: Stable at 3-4 GB
# - CPU usage: 20-40% (only MCTS tree search)
```

### Test 2: Performance Comparison
```bash
# Run CPU benchmark
python -m ml.benchmark_selfplay --device cpu --quick --games 20

# Run GPU benchmark
python -m ml.benchmark_selfplay --device cuda --quick --games 20

# Expected:
# - GPU should be 3-10x faster than CPU
# - GPU: 50-150 games/min
# - CPU: 10-20 games/min
```

### Test 3: Training Integration
```bash
# Run iteration benchmark with GPU
python -m ml.benchmark_iteration --games 500 --device cuda

# Expected:
# - Self-play phase: GPU active, 70-100% utilization
# - Training phase: GPU active, 90-100% utilization
# - No idle GPU periods
```

---

## Implementation Checklist

- [ ] Update `_worker_generate_games_static` signature to accept `device` parameter
- [ ] Add `network.to(device)` in worker function
- [ ] Update `SelfPlayEngine.__init__` to accept and store `device` parameter
- [ ] Update `SelfPlayEngine.generate_games` to pass device to workers
- [ ] Update `ml/benchmark_selfplay.py` to pass device to engine
- [ ] Update `ml/benchmark_iteration.py` to pass device to engine
- [ ] Update `ml/training/trainer.py` (`TrainingPipeline`) to pass device to engine
- [ ] Test GPU utilization during self-play
- [ ] Re-run benchmarks to get accurate GPU performance data
- [ ] Update performance estimates and recommendations

---

## Priority Assessment

**Priority**: 🔴 **CRITICAL - MUST FIX BEFORE ANY PRODUCTION TRAINING**

**Why**:
1. Current benchmarks are **completely invalid** (testing CPU, not GPU)
2. Training time estimates are **off by 10-20x**
3. Cannot make informed decision about GPU-batched MCTS without accurate GPU data
4. All subsequent Phase 4 work depends on this fix

**Estimated Fix Time**: 30-60 minutes
- Code changes: 20 minutes
- Testing: 10 minutes
- Re-run benchmarks: 20 minutes

---

## Next Steps

1. **Immediate**: Implement the fix (3 files, ~10 lines of code)
2. **Validate**: Run Test 1 (GPU utilization check)
3. **Benchmark**: Re-run full benchmark suite with correct GPU usage
4. **Decide**: Use real GPU data to decide on GPU-batched MCTS priority
5. **Continue**: Proceed with Phase 4 Sessions 6-7 (Evaluation + Main Script)

---

## Notes

### Why Training Works But Self-Play Doesn't

**Training** (works correctly):
```python
# trainer.py
trainer = NetworkTrainer(network=network, device="cuda")
# Network stays in main process, already on GPU
# No multiprocessing involved in training loop
```

**Self-Play** (broken):
```python
# selfplay.py
engine = SelfPlayEngine(network=network)  # Main process network on GPU
pool.starmap(worker_func, tasks)  # Workers get state dict, recreate on CPU!
```

The key difference: Training uses a single process (GPU stays active), while self-play uses multiprocessing (workers create fresh networks on CPU).

---

**Status**: Ready to implement fix
