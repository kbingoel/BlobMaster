# Performance Optimization Plan

**Created:** 2025-11-05
**Updated:** 2025-11-06 (Session 1+2 Complete)
**Baseline:** 36.7 games/min (Medium MCTS, 32 workers, Ubuntu + RTX 4060)
**Target:** 110-147 games/min (3-4x speedup, revised from 3-7x)
**Training Time:** 136 days → 34-45 days (revised from 19-44 days)

---

## Status Update (2025-11-06)

**Sessions 1+2 Complete:** ✅ **2.07x speedup achieved** (75.85 games/min)

**Key Findings:**
- Batch submission API + parallel expansion delivered 2.07x speedup (lower end of 2-5x estimate)
- Optimal `parallel_batch_size = 30` (tested 25, 30, 35, 40, 45, 50)
- Performance degrades with larger batches (>30) due to timeout overhead
- **Revised expectations:** Remaining optimizations likely at lower end of ranges

**Updated Projections:**
- Session 3 (Mixed Precision): +15-25% → **87-95 games/min**
- Session 4 (torch.compile): +10-20% → **96-114 games/min**
- Session 5 (Runtime Tuning): +5-10% → **101-125 games/min**
- **Expected Final: 101-125 games/min (2.75-3.4x total speedup)**

---

## Recommendations

**Priorities:**
1. ✅ Fix batching API (multi-submit/gather) - **DONE: 1.5x speedup**
2. ✅ Enable parallel MCTS expansion - **DONE: +40% additional (2.07x cumulative)**
3. 🔄 Mixed precision inference (FP16/BF16) - **Next: +15-25% expected**
4. ⏳ torch.compile integration - **+10-20% expected**
5. ⏳ Linux runtime tuning - **+5-10% expected**
6. ⚠️ Cython MCTS (optional if above insufficient) - **+10-50%**

Implement Priorities 3-5 next (expect 2.75-3.4x total speedup), then decide if Priority 6 needed.

---

## Existing Code Reuse Analysis

**Already Implemented (can reuse immediately):**
- ✅ GPU server multiprocessing architecture ([ml/mcts/gpu_server.py](../../ml/mcts/gpu_server.py))
- ✅ BatchedEvaluator threading architecture ([ml/mcts/batch_evaluator.py](../../ml/mcts/batch_evaluator.py))
- ✅ Parallel MCTS with virtual loss ([ml/mcts/search.py:531-613](../../ml/mcts/search.py#L531-L613))
- ✅ SelfPlayWorker parallel expansion support ([ml/training/selfplay.py:78-79](../../ml/training/selfplay.py#L78-L79))
- ✅ Mixed precision training ([ml/training/trainer.py:70](../../ml/training/trainer.py#L70))

**Needs Implementation:**
- ❌ `evaluate_many()` batch submission API
- ❌ Inference mixed precision (separate from training precision)
- ❌ torch.compile model compilation
- ❌ Config fields for parallel expansion parameters
- ❌ Environment variable tuning

---

## Validation Benchmark

**Baseline File:** [benchmarks/results/ubuntu-2025-11/full_baseline_sweep.csv](../../benchmarks/results/ubuntu-2025-11/full_baseline_sweep.csv)
**Baseline Performance:** 36.7 games/min (32 workers, Medium MCTS: 3 det × 30 sims)

**Standard Validation Command:**
```bash
source venv/bin/activate
python benchmarks/performance/benchmark_selfplay.py \
  --workers 32 \
  --device cuda \
  --games 50 \
  --output benchmarks/results/session_$(date +%Y%m%d_%H%M).csv
```

**Speedup Calculation:**
```
Speedup = (Your_games_per_min) / 36.7
```

**Success Criteria (Revised):**
- Session 1+2: ✅ **75.85 games/min (2.07x speedup)** - ACHIEVED
- Session 3: >87 games/min (2.4x cumulative speedup)
- Session 4: >96 games/min (2.6x cumulative speedup)
- Session 5: >101 games/min (2.75x cumulative speedup)
- Overall: >110 games/min (3x speedup minimum target)

---

## Session 1: Batch Submission API ✅ COMPLETE

**Goal:** Add `evaluate_many()` to enable true cross-worker batching
**Expected Speedup:** 1.5-2.5x (55-92 games/min)
**Actual Speedup:** 1.5x alone (estimated, combined with Session 2: 2.07x)
**Effort:** 2.5 hrs implementation + 0.5 hrs validation

### Implementation

#### 1.1 Add `evaluate_many()` to GPUServerClient

**File:** [ml/mcts/gpu_server.py](../../ml/mcts/gpu_server.py)
**Location:** After line 166 (after existing `evaluate()` method)

**Add:**
```python
def evaluate_many(
    self,
    states: List[torch.Tensor],
    masks: List[torch.Tensor],
    timeout: float = 30.0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Request neural network evaluation for multiple states (batch submission).

    Submits all requests at once, then waits for all results. This allows
    GPU server to batch across workers for maximum throughput.

    Args:
        states: List of encoded game state tensors (256-dim)
        masks: List of legal action mask tensors (52-dim)
        timeout: Maximum time to wait for all results (seconds)

    Returns:
        Tuple of (policies, values) lists
    """
    request_ids = []

    # Submit all requests
    for state, mask in zip(states, masks):
        request_id = f"{self.client_id}_{uuid.uuid4().hex[:8]}"
        request = InferenceRequest(
            request_id=request_id,
            state=state.cpu(),
            mask=mask.cpu(),
            response_queue_id=self.client_id,
        )
        self.request_queue.put(request)
        request_ids.append(request_id)

    # Gather all results
    policies = []
    values = []
    results_map = {}

    deadline = time.time() + timeout
    for _ in range(len(request_ids)):
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError(f"GPU server batch timeout after {timeout}s")

        try:
            result = self.response_queue.get(timeout=remaining)
        except queue.Empty:
            raise TimeoutError(f"GPU server batch timeout after {timeout}s")

        if result.error is not None:
            raise RuntimeError(f"GPU server error: {result.error}")

        results_map[result.request_id] = result

    # Reorder results to match input order
    for request_id in request_ids:
        result = results_map[request_id]
        policies.append(result.policy)
        values.append(result.value)

    return policies, values
```

**Imports needed:** Add `time` to imports at top of file if not present.

#### 1.2 Add `evaluate_many()` to BatchedEvaluator

**File:** [ml/mcts/batch_evaluator.py](../../ml/mcts/batch_evaluator.py)
**Location:** After line 260 (after existing `evaluate()` method)

**Add:**
```python
def evaluate_many(
    self,
    states: List[torch.Tensor],
    masks: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Evaluate multiple game states using batched neural network inference.

    Submits all requests at once for better cross-request batching.

    Args:
        states: List of encoded game state tensors (256-dim)
        masks: List of legal action mask tensors (52-dim)

    Returns:
        Tuple of (policies, values) lists
    """
    if not self.running:
        raise RuntimeError(
            "BatchedEvaluator is not running. Call start() first."
        )

    # Create result queues for all requests
    result_queues = [queue.Queue(maxsize=1) for _ in states]

    # Submit all requests
    for state, mask, result_queue in zip(states, masks, result_queues):
        with self.request_id_lock:
            request_id = self.next_request_id
            self.next_request_id += 1

        request = EvaluationRequest(
            request_id=request_id,
            state=state,
            mask=mask,
            result_queue=result_queue,
        )

        self.request_queue.put(request)

    # Gather all results
    policies = []
    values = []

    for result_queue in result_queues:
        result = result_queue.get()

        if result.error is not None:
            raise RuntimeError(f"Evaluation failed: {result.error}")

        policies.append(result.policy)
        values.append(result.value)

    return policies, values
```

#### 1.3 Update MCTS Search to Use Batch Submission

**File:** [ml/mcts/search.py](../../ml/mcts/search.py)
**Location:** Lines 739-759 (in `_batch_expand_and_evaluate()`)

**Replace GPU server path (lines 739-749):**
```python
        # OLD (submit one at a time):
        if self.gpu_server_client is not None:
            # Use GPU inference server - send each state individually
            # (GPU server handles batching internally)
            policy_list = []
            value_list = []
            for state, mask in zip(states, masks):
                policy, value = self.gpu_server_client.evaluate(state, mask)
                policy_list.append(policy)
                value_list.append(value)
            policy_batch = torch.stack(policy_list)
            value_batch = torch.stack(value_list)
```

**With NEW (batch submission):**
```python
        if self.gpu_server_client is not None:
            # Use GPU inference server - batch submit all states
            policy_list, value_list = self.gpu_server_client.evaluate_many(states, masks)
            policy_batch = torch.stack(policy_list)
            value_batch = torch.stack(value_list)
```

**Replace BatchedEvaluator path (lines 750-759):**
```python
        # OLD (submit one at a time):
        elif self.batch_evaluator is not None:
            # Use centralized batched evaluator - send each state individually
            policy_list = []
            value_list = []
            for state, mask in zip(states, masks):
                policy, value = self.batch_evaluator.evaluate(state, mask)
                policy_list.append(policy)
                value_list.append(value)
            policy_batch = torch.stack(policy_list)
            value_batch = torch.stack(value_list)
```

**With NEW (batch submission):**
```python
        elif self.batch_evaluator is not None:
            # Use centralized batched evaluator - batch submit all states
            policy_list, value_list = self.batch_evaluator.evaluate_many(states, masks)
            policy_batch = torch.stack(policy_list)
            value_batch = torch.stack(value_list)
```

### Validation Results ✅

**Implementation:** Combined with Session 2 (parallel expansion)
**Result File:** [benchmarks/results/session1_validation_20251106_1951.csv](../../benchmarks/results/session1_validation_20251106_1951.csv)
**Performance:** 75.85 games/min (2.07x speedup) with `parallel_batch_size=30`

**Tested batch sizes:** 25, 30, 35, 40, 45, 50
- **Best:** 30 (75.85 games/min)
- Larger batches (>30) showed performance degradation due to timeout overhead

**Success:** ✅ Exceeded 55 games/min minimum (achieved 75.85)

---

## Session 2: Enable Parallel Expansion ✅ COMPLETE

**Goal:** Activate existing parallel MCTS to increase leaves per batch
**Expected Speedup:** +30-100% on top of Session 1 (72-184 games/min cumulative)
**Actual Speedup:** +40% additional (2.07x cumulative with Session 1)
**Effort:** 2 hrs implementation + 1 hr tuning + 0.5 hrs validation

### Implementation

#### 2.1 Add Config Fields

**File:** [ml/config.py](../../ml/config.py)
**Location:** After line 20 (in self-play settings section)

**Add:**
```python
    # MCTS parallelization
    use_parallel_expansion: bool = False
    parallel_batch_size: int = 10
```

#### 2.2 Pass Config to SelfPlayWorker

**File:** [ml/training/selfplay.py](../../ml/training/selfplay.py)
**Location:** Find where SelfPlayWorker is instantiated (search for `SelfPlayWorker(`)

**Update instantiation to pass new parameters:**
```python
worker = SelfPlayWorker(
    network=network,
    encoder=encoder,
    masker=masker,
    num_determinizations=config.num_determinizations,
    simulations_per_determinization=config.simulations_per_determinization,
    batch_evaluator=batch_evaluator,  # or gpu_server_client
    use_parallel_expansion=config.use_parallel_expansion,  # NEW
    parallel_batch_size=config.parallel_batch_size,  # NEW
)
```

#### 2.3 Enable in Default Config

**File:** [ml/config.py](../../ml/config.py)
**Update new fields to:**
```python
    use_parallel_expansion: bool = True  # ENABLE
    parallel_batch_size: int = 10  # Start conservative
```

### Tuning (1 hour)

Test different `parallel_batch_size` values:

```bash
# Test parallel_batch_size = 10
python benchmarks/performance/benchmark_selfplay.py \
  --workers 32 --device cuda --games 20

# Edit config.py, set parallel_batch_size = 20, rerun
# Edit config.py, set parallel_batch_size = 30, rerun
# Find optimal value (likely 15-25 for 32 workers)
```

**Monitor:** GPU batch sizes in logs. Target: 128-256 batch sizes consistently.

### Validation Results ✅

**Implementation:** Combined with Session 1 (batch submission API)
**Result File:** [benchmarks/results/session1_validation_20251106_1951.csv](../../benchmarks/results/session1_validation_20251106_1951.csv)
**Performance:** 75.85 games/min (2.07x cumulative speedup)

**Tuning Results:**
- Tested `parallel_batch_size`: 25, 30, 35, 40, 45, 50
- **Optimal value: 30** (75.85 games/min)
- Performance profile:
  - 25: 71.65 games/min (-5.5%)
  - 30: 75.85 games/min (BEST)
  - 35: 73.69 games/min (-2.8%)
  - 40: 73.63 games/min (-2.9%)
  - 45: 68.89 games/min (-9.2%)
  - 50: 63.56 games/min (-16.2%)

**Key Insight:** Larger batches increase timeout overhead faster than batching benefits. Sweet spot at 30 for 32 workers.

**Success:** ✅ Exceeded 72 games/min minimum (achieved 75.85)

---

## Session 3: Mixed Precision Inference (3 hours)

**Goal:** Add FP16 autocast to all inference forward passes
**Expected Speedup:** +15-25% (87-95 games/min cumulative, revised from 79-239)
**Current Status:** Ready to implement
**Effort:** 2.5 hrs implementation + 0.5 hrs validation

### Implementation

#### 3.1 GPU Server Inference

**File:** [ml/mcts/gpu_server.py](../../ml/mcts/gpu_server.py)
**Location:** Lines 438-439 (forward pass)

**Replace:**
```python
            with torch.no_grad():
                policies, values = network(states, masks)
```

**With:**
```python
            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.float16):
                    policies, values = network(states, masks)
```

**Note:** RTX 4060 doesn't have native BF16, use FP16. Add `import torch` at top if needed.

#### 3.2 BatchedEvaluator Inference

**File:** [ml/mcts/batch_evaluator.py](../../ml/mcts/batch_evaluator.py)
**Location:** Find `_process_batch()` method, locate `with torch.no_grad():` line

**Replace forward pass:**
```python
            with torch.no_grad():
                policy_batch, value_batch = self.network(state_batch, mask_batch)
```

**With:**
```python
            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.float16):
                    policy_batch, value_batch = self.network(state_batch, mask_batch)
```

#### 3.3 Direct MCTS Inference

**File:** [ml/mcts/search.py](../../ml/mcts/search.py)
**Location:** Find all `with torch.no_grad():` lines followed by `network(` calls

**Update each (typically 2-3 locations):**
```python
            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.float16):
                    policy, value = self.network(state, mask)
```

#### 3.4 Enable TF32 (free speedup)

**File:** [ml/train.py](../../ml/train.py) or create new [ml/performance_init.py](../../ml/performance_init.py)
**Location:** At startup, before any model creation

**Add:**
```python
# Enable TF32 for faster matmul on Ampere+ GPUs
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
```

**Import this in train.py and benchmark scripts.**

### Validation

```bash
# Run 50-game benchmark
python benchmarks/performance/benchmark_selfplay.py \
  --workers 32 --device cuda --games 50 \
  --output benchmarks/results/session3_$(date +%Y%m%d_%H%M).csv

# Expected: 87-95 games/min (2.4-2.6x cumulative, revised)
```

**Success:** Games/min > 87 (2.4x minimum cumulative)

**Validation Check:** Compare 10 games' action probabilities before/after to ensure numerical equivalence (variance <0.01).

---

## Session 4: torch.compile Integration (3 hours)

**Goal:** Compile models at initialization for faster inference
**Expected Speedup:** +10-20% (96-114 games/min cumulative, revised from 87-311)
**Current Status:** Pending Session 3 completion
**Effort:** 2 hrs implementation + 1 hr compilation testing + 0.5 hrs validation

### Implementation

#### 4.1 GPU Server Model Compilation

**File:** [ml/mcts/gpu_server.py](../../ml/mcts/gpu_server.py)
**Location:** Lines 392-395 (after model initialization)

**Replace:**
```python
    # Initialize network in this process
    network = BlobNet(**network_config)
    network.load_state_dict(network_state)
    network.to(device)
    network.eval()
```

**With:**
```python
    # Initialize network in this process
    network = BlobNet(**network_config)
    network.load_state_dict(network_state)
    network.to(device)
    network.eval()

    # Compile for faster inference (first call will be slow)
    print("[GPU Server] Compiling model (this will take 30-60s)...")
    network = torch.compile(network, mode='reduce-overhead', fullgraph=True)
    print("[GPU Server] Model compiled successfully")
```

#### 4.2 BatchedEvaluator Model Compilation

**File:** [ml/mcts/batch_evaluator.py](../../ml/mcts/batch_evaluator.py)
**Location:** In `__init__()` method, after `self.network.eval()`

**Add:**
```python
        self.network.eval()

        # Compile for faster inference
        print("[BatchedEvaluator] Compiling model...")
        self.network = torch.compile(self.network, mode='reduce-overhead', fullgraph=True)
        print("[BatchedEvaluator] Model compiled successfully")
```

#### 4.3 Handle Compilation Warmup

**File:** [ml/mcts/gpu_server.py](../../ml/mcts/gpu_server.py)
**Location:** After model compilation (line ~398)

**Add warmup call:**
```python
    network = torch.compile(network, mode='reduce-overhead', fullgraph=True)

    # Warmup: run one dummy inference to trigger compilation
    dummy_state = torch.randn(1, 256, device=device)
    dummy_mask = torch.ones(1, 52, device=device)
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.float16):
            _ = network(dummy_state, dummy_mask)

    print("[GPU Server] Model compiled successfully")
```

**Repeat for BatchedEvaluator.**

### Testing (1 hour)

First run will be slow (30-60s compilation). Test that:
1. Compilation completes without errors
2. Subsequent inference is faster
3. Numerical outputs unchanged

```bash
# Quick test
python benchmarks/performance/benchmark_selfplay.py \
  --workers 4 --device cuda --games 5

# Check logs for compilation messages
# Verify no errors
```

### Validation

```bash
# Run 50-game benchmark
python benchmarks/performance/benchmark_selfplay.py \
  --workers 32 --device cuda --games 50 \
  --output benchmarks/results/session4_$(date +%Y%m%d_%H%M).csv

# Expected: 96-114 games/min (2.6-3.1x cumulative, revised)
```

**Success:** Games/min > 96 (2.6x minimum cumulative)

**Note:** First 1-2 games may be slower due to compilation. Measure games 10-50 for accurate rate.

---

## Session 5: Linux Runtime Tuning (3 hours)

**Goal:** Optimize environment variables and GPU server timeout
**Expected Speedup:** +5-10% (101-125 games/min cumulative, revised from 92-357)
**Current Status:** Pending Session 4 completion
**Effort:** 1.5 hrs tuning + 1.5 hrs testing

### Implementation

#### 5.1 Environment Variables

**File:** Create [ml/set_performance_env.sh](../../ml/set_performance_env.sh)

```bash
#!/bin/bash
# Performance environment variables for Linux training

# Disable OpenMP/MKL threading (we use multiprocessing, not threads)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Use jemalloc allocator if available (faster than glibc malloc)
if [ -f /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ]; then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
    echo "[Performance] Using jemalloc allocator"
else
    echo "[Performance] jemalloc not found, using default allocator"
fi

# Set CPU performance governor (requires sudo, optional)
# sudo cpupower frequency-set -g performance

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "[Performance] Environment configured"
```

**Make executable:**
```bash
chmod +x ml/set_performance_env.sh
```

**Use before training:**
```bash
source ml/set_performance_env.sh
python benchmarks/performance/benchmark_selfplay.py ...
```

#### 5.2 GPU Server Timeout Tuning

**File:** [ml/mcts/gpu_server.py](../../ml/mcts/gpu_server.py)
**Location:** Default timeout parameter (find `timeout_ms` default)

**Test different timeouts:**

```python
# Current default: 10ms
# Test: 5ms (more responsive, smaller batches)
# Test: 8ms (balance)
# Test: 3ms (very responsive)
```

**File:** [ml/config.py](../../ml/config.py)
**Add field:**
```python
    # GPU server settings
    gpu_server_timeout_ms: int = 5  # Reduced from 10ms
    gpu_server_max_batch_size: int = 512
```

**Pass to GPU server initialization.**

### Tuning (1.5 hours)

Test timeout values:

```bash
# Test 3ms
# Edit config, set gpu_server_timeout_ms = 3
python benchmarks/performance/benchmark_selfplay.py \
  --workers 32 --device cuda --games 20

# Test 5ms
# Edit config, set gpu_server_timeout_ms = 5
python benchmarks/performance/benchmark_selfplay.py \
  --workers 32 --device cuda --games 20

# Test 8ms
# Edit config, set gpu_server_timeout_ms = 8
python benchmarks/performance/benchmark_selfplay.py \
  --workers 32 --device cuda --games 20

# Find optimal (likely 5ms)
```

**Monitor:** Batch sizes, total throughput. Target: maximize games/min while keeping batch sizes >100.

### Validation

```bash
# Set optimal environment
source ml/set_performance_env.sh

# Run 50-game benchmark with optimal timeout
python benchmarks/performance/benchmark_selfplay.py \
  --workers 32 --device cuda --games 50 \
  --output benchmarks/results/session5_$(date +%Y%m%d_%H%M).csv

# Expected: 101-125 games/min (2.75-3.4x cumulative, revised)
```

**Success:** Games/min > 101 (2.75x minimum cumulative)

---

## Session 6: Full Validation (3 hours)

**Goal:** Comprehensive benchmarking and documentation
**Effort:** 2 hrs benchmarking + 1 hr documentation

### Full Benchmark Suite

```bash
# Ensure all optimizations enabled
source ml/set_performance_env.sh

# 1. Medium MCTS (production config) - 200 games
python benchmarks/performance/benchmark_selfplay.py \
  --workers 32 --device cuda --games 200 \
  --output benchmarks/results/final_medium_$(date +%Y%m%d).csv

# 2. Light MCTS (fast iteration)
python benchmarks/performance/benchmark_selfplay.py \
  --workers 32 --device cuda --games 100 \
  --output benchmarks/results/final_light_$(date +%Y%m%d).csv

# 3. Heavy MCTS (research quality)
python benchmarks/performance/benchmark_selfplay.py \
  --workers 32 --device cuda --games 50 \
  --output benchmarks/results/final_heavy_$(date +%Y%m%d).csv

# 4. Worker scaling test
python benchmarks/performance/benchmark_selfplay.py \
  --workers 1 4 8 16 32 \
  --device cuda --games 50 \
  --output benchmarks/results/final_scaling_$(date +%Y%m%d).csv
```

### Performance Report

**Create:** [docs/performance/OPTIMIZATION-RESULTS.md](OPTIMIZATION-RESULTS.md)

**Document:**
1. Baseline vs optimized performance (table)
2. Per-session speedup contributions
3. Final games/min for each MCTS config
4. Updated training time estimates
5. Lessons learned and future optimizations

**Compare:**
- Baseline: 36.7 games/min → Optimized: ??? games/min
- Training time: 136 days → ??? days

### Success Criteria (Revised)

**Minimum (2.75x speedup):**
- Medium MCTS: >101 games/min
- Training time: <50 days

**Target (3x speedup):**
- Medium MCTS: >110 games/min
- Training time: <45 days

**Stretch (3.4x speedup):**
- Medium MCTS: >125 games/min
- Training time: <40 days

**Note:** Original 5-7x targets were overly optimistic. Revised targets based on Session 1+2 actual results.

---

## Fallback: Priority 6 (Cython MCTS)

**Decision Point:** If Sessions 3-5 fail to reach 3x cumulative (110 games/min), implement Cython acceleration.

**Current Assessment:** With 2.07x achieved from Sessions 1+2, need +45% from Sessions 3-5 to reach 3x target. This is achievable but optimistic. Cython may be needed if Sessions 3-5 underperform.

**Effort:** 4-6 hours (separate session)

**Targets:**
1. [ml/mcts/node.py:88-212](../../ml/mcts/node.py#L88-L212) - UCB selection (hot loop)
2. [ml/mcts/node.py:279-356](../../ml/mcts/node.py#L279-L356) - Backpropagation
3. [ml/mcts/search.py:139-181](../../ml/mcts/search.py#L139-L181) - Tree traversal

**Tools:** Cython or pybind11 (C++ extension)

**Expected:** +10-50% additional speedup

---

## Summary Timeline (Revised)

| Session | Focus | Time | Cumulative Speedup | Games/Min | Status |
|---------|-------|------|-------------------|-----------|--------|
| Baseline | - | - | 1.0x | 36.7 | - |
| Session 1 | Batch API | 3h | ~1.5x | ~55 | ✅ (combined) |
| Session 2 | Parallel MCTS | 3h | 2.07x | **75.85** | ✅ DONE |
| Session 3 | Mixed Precision | 3h | 2.4-2.6x | 87-95 | 🔄 Next |
| Session 4 | torch.compile | 3h | 2.6-3.1x | 96-114 | ⏳ Pending |
| Session 5 | Runtime Tuning | 3h | 2.75-3.4x | 101-125 | ⏳ Pending |
| Session 6 | Validation | 3h | **2.75-3.4x** | **101-125** | ⏳ Pending |

**Total Time:** 18 hours (3 work days)
**Expected Outcome:** 2.75-3.4x speedup, reducing training from 136 days to 40-50 days (revised from 19-44 days)

**Key Insight:** Original 3-7x target was overly optimistic. Realistic target is 2.75-3.4x based on actual Session 1+2 results (2.07x vs predicted 2-5x).

---

## Next Steps & Recommendations

### Immediate Actions

**1. Proceed with Session 3 (Mixed Precision)**
- Expected: +15-25% boost (87-95 games/min)
- Low risk, well-tested optimization
- Implementation time: 3 hours

**2. Continue to Session 4 (torch.compile)**
- Expected: +10-20% boost (96-114 games/min)
- May require warmup/debugging
- Implementation time: 3 hours

**3. Complete Session 5 (Runtime Tuning)**
- Expected: +5-10% boost (101-125 games/min)
- Environment optimization, GPU timeout tuning
- Implementation time: 3 hours

**4. Decision Point After Session 5:**
- If ≥110 games/min (3x): **Begin 500-iteration training** (~45 days)
- If 101-109 games/min (2.75-3x): Decide whether Cython is worth it for 40-45 day training
- If <101 games/min (<2.75x): **Implement Priority 6 (Cython MCTS)** before training

### Analysis & Insights

**What Worked Well:**
- Batch submission API enabled better cross-worker batching
- Parallel expansion with `batch_size=30` hit sweet spot
- Combined optimizations delivered solid 2.07x speedup

**What Was Overly Optimistic:**
- Original 3-7x estimate too aggressive
- Diminishing returns from batching (timeout overhead at large batches)
- GPU memory constraints limit worker scaling (32 workers max)

**Recommendation on GPU Server:**
- Phase 3.5 GPU server was 3-5x slower than multiprocessing
- With new batch API, may be worth re-testing, but **NOT a priority**
- Current multiprocessing approach is working well
- Only revisit GPU server if hitting different bottleneck

### Training Decision Matrix

| Final Speed | Speedup | Training Time | Recommendation |
|-------------|---------|---------------|----------------|
| >125 games/min | >3.4x | <40 days | Excellent, begin training immediately |
| 110-125 games/min | 3.0-3.4x | 40-45 days | Good, begin training |
| 101-109 games/min | 2.75-3.0x | 45-50 days | Marginal, consider Cython first |
| <101 games/min | <2.75x | >50 days | Implement Cython before training |

**Current Trajectory:** Expecting 101-125 games/min after Session 5, which would make training feasible (40-50 days).
