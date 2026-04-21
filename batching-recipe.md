# Batching Recipe — GPU-Backed Self-Play Inference

Working doc for the next implementation session. Goal: cut per-iteration wall-clock by feeding the RTX 4060 batched forward passes during self-play instead of running 32 independent CPU inference streams.

## Current state (why we're doing this)

- `OnnxEvaluator` runs **one `ort::Session` per rayon thread**, batch=1, ~0.6 ms/eval on CPU ([blob-engine/src/onnx.rs:45-47](blob-engine/src/onnx.rs#L45-L47), [blob-nn/src/self_play.rs:98](blob-nn/src/self_play.rs#L98)).
- MCTS is serial inside a determinization — one `eval.evaluate()` per sim, no virtual loss ([blob-engine/src/mcts.rs:296-334](blob-engine/src/mcts.rs#L296-L334)).
- Self-play does ~40M evaluations per iteration; ~1.28M per thread. GPU is idle, CPU is saturated.
- `tch` is already a `blob-nn` dep with `Device` plumbed through training ([blob-nn/src/training_loop.rs:70](blob-nn/src/training_loop.rs#L70)); nothing in the `Evaluator` trait prevents batched inference ([blob-engine/src/evaluator.rs:30-32](blob-engine/src/evaluator.rs#L30-L32)).
- Crate rule to preserve: `blob-engine` must never depend on `tch`; `blob-bin` (deployment) must never depend on `tch` or `blob-nn`. The batched GPU evaluator lives in `blob-nn` and implements the `blob_engine::Evaluator` trait.

## Target architecture

```
rayon workers (N=concurrent_games)          inference thread (1)
┌─────────────────────────────┐             ┌──────────────────────┐
│ MCTS sim → encode() → send()│ ──mpsc──▶   │ drain ≤ max_batch    │
│             ▲               │             │ pad → forward(CUDA)  │
│             └── oneshot ◀───│ ──────────  │ scatter → oneshots   │
└─────────────────────────────┘             └──────────────────────┘
```

Evaluator trait stays **synchronous** — workers block on a oneshot recv, which naturally serves as the backpressure that feeds the GPU.

## Implementation sequence

### Step 1 — cargo feature + libtorch-CUDA wiring

- Add `cuda` feature to [blob-nn/Cargo.toml](blob-nn/Cargo.toml); default **off** so Windows/CPU builds stay unchanged.
- Under `cuda`, either switch `tch`'s `download-libtorch` to a CUDA variant, or document the `LIBTORCH` / `LD_LIBRARY_PATH` override (matches the existing runtime-env memory note).
- Propagate to `blob-train` as a passthrough feature so the training binary can enable it.
- Smoke test: `Device::cuda_if_available()` returns `Cuda(0)` and a 128×58×48 forward completes.

### Step 2 — `BatchedGpuEvaluator` skeleton (testable on CPU first)

New module `blob-nn/src/gpu_eval.rs`:

```rust
pub struct BatchedGpuEvaluator {
    tx: crossbeam_channel::Sender<Request>,
    _worker: std::thread::JoinHandle<()>, // holds the inference thread
}

struct Request {
    enc: EncodedState,        // encoded on the worker, off the critical path
    phase: GamePhase,         // dispatch to forward_bid vs forward_play
    reply: oneshot::Sender<(Vec<f32>, f32)>,
}

impl BatchedGpuEvaluator {
    pub fn new(model: Arc<BlobNet>, device: Device, cfg: BatchCfg) -> Self { ... }
}

impl blob_engine::Evaluator for BatchedGpuEvaluator {
    fn evaluate(&self, state: &BlobState) -> (Vec<f32>, f32) {
        let enc = encode(state, state.current_player);
        let (tx_reply, rx_reply) = oneshot::channel();
        self.tx.send(Request { enc, phase: state.phase(), reply: tx_reply }).unwrap();
        rx_reply.recv().unwrap() // worker blocks here — this is the coalescing point
    }
}
```

`BatchCfg`:

```rust
pub struct BatchCfg {
    pub max_batch: usize,      // cap; e.g. 128
    pub max_wait_us: u64,      // tail deadline; e.g. 200
    pub pad_to_max_seq: bool,  // true — fix seq=58 so CUDA kernel plans cache
}
```

**Inference thread main loop:**
1. Block on `rx.recv()` for the first request.
2. Opportunistically drain `rx.try_recv()` up to `max_batch - 1` more, OR until `max_wait_us` elapsed since first request — whichever first.
3. Separate requests by `phase` (bidding vs playing use different heads). Two sub-batches max.
4. Pad features to `[batch, MAX_SEQ=58, 48]`, move tensors to CUDA (can pre-allocate pinned host staging buffers).
5. `model.forward_bid` / `forward_play` under `tch::no_grad()`.
6. Scatter per-row output back through each `reply` oneshot.

**Parity test before going further:** with `Device::Cpu`, a batched forward on N encoded states must match N single-sample tch-CPU forwards within 1e-5. Same test with `Device::Cuda(0)` within 1e-4.

### Step 3 — swap self-play to shared evaluator

- Change `self_play_iteration` ([blob-nn/src/self_play.rs:66-70](blob-nn/src/self_play.rs#L66-L70)) signature from `model_path: &Path` to `eval: &(dyn Evaluator + Sync)`.
- Delete the per-thread `OnnxEvaluator::from_file` in the rayon map ([blob-nn/src/self_play.rs:98-99](blob-nn/src/self_play.rs#L98-L99)).
- `TrainingLoop::run_iteration` constructs one `BatchedGpuEvaluator` per iteration directly from the in-memory `VarStore` — **no ONNX round-trip on the hot path**. ONNX export becomes promotion-only (for `best_model.onnx` / deployment).

### Step 4 — oversubscribe games (Lever 2A)

- Add `concurrent_games: usize` to `SelfPlayConfig` ([blob-nn/src/engine.rs:33-46](blob-nn/src/engine.rs#L33-L46)), distinct from `num_threads`. Default ~128.
- Build the rayon pool with that many threads; keep CPU cores busy with the encode + apply_action work that runs between evaluator blocks.
- Rationale: workers spend ~95% of wall time blocked on oneshot `recv`; heavy oversubscription is cheap and grows the steady-state batch size without touching MCTS.

### Step 5 — measure, decide on virtual loss

Ship the instrumentation **with step 2**, not after. Minimum:

- Batch-size histogram per iteration (p10 / p50 / p90 / p99).
- Inference-thread duty cycle (fraction of wall-clock in `forward()`). Target > 90%.
- Per-request timings: `submit → picked_up → gpu_done → recv`.
- H2D / D2H bytes per iteration (sanity, not primary signal — ~3 MB/forward at bs=128).

**Decision after one real iteration with step 4 done:**
- Iteration wall-clock < 5 min **and** GPU duty > 70% → ship it. Skip step 6.
- Otherwise → step 6.

### Step 6 — virtual loss + leaf queue inside MCTS (only if needed)

Touch [blob-engine/src/mcts.rs:296-334](blob-engine/src/mcts.rs#L296-L334):

- Per tree, keep `K` leaves in flight (e.g. K=8).
- On selection, increment `visit_count` and a "virtual loss" on the path *before* submitting the eval, so concurrent selections from the same tree diverge to different leaves.
- On result arrival, undo the virtual-loss bias and apply the real backprop.
- Multiplies batch pressure by K per worker thread.

This is the cleaner long-term answer (standard AlphaZero) but adds real complexity to a currently-clean MCTS. Do it only if step 5 says we need it.

### Step 7 — cleanup

- Keep `OnnxEvaluator` for deployment (`blob-bin`) and eval-games vs baseline checkpoints.
- Prune ONNX from the self-play hot path.
- Update [development-plan.md](development-plan.md) Session 7.1 prerequisites to reflect the GPU inference path.

## Guardrails

- **Do not** change the `Evaluator` trait to async. The sync-blocking-on-oneshot pattern preserves all MCTS / self-play call sites unchanged.
- **Do not** put `tch` into `blob-engine`. The batched evaluator is a `blob-nn` type that implements the `blob-engine` trait.
- **Do not** skip the CPU-tch parity test in step 2 — catching a shape / head-dispatch bug there is cheap; catching it after 4 hours of self-play is not.
- **Do not** merge step 4 before step 2's parity test passes.

## Dials to expose in `TrainingConfig`

Add to [blob-train/src/config.rs](blob-train/src/config.rs):

```toml
[gpu_eval]
enabled = true
device = "cuda:0"
max_batch = 128
max_wait_us = 200
pad_to_max_seq = true

[self_play]
concurrent_games = 128        # separate from num_threads
```

Leaving `gpu_eval.enabled = false` must fall back cleanly to the current ONNX-per-thread path so we can A/B the two during rollout.

## Expected outcome

- Per-eval GPU-amortized cost at batch 128: ~10 µs (vs 0.6 ms CPU) → ~60× per-eval.
- Wall-clock is gated by inference-thread duty cycle, not per-eval cost. Realistic iteration: **well under 5 min at 32C / 4060**, likely 1-2 min once Lever 2 saturates the batcher.
- Training cost for an extended run drops from days to hours on the same box.
