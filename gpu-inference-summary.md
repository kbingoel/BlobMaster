# GPU Batched Inference — Summary & Conclusion

**Date**: 2026-04-21
**Hardware**: AMD Ryzen 9 7950X (16C/32T) + NVIDIA RTX 4060 8 GB, 128 GB DDR5
**Model**: BlobNet Transformer — d=128, 8 heads, 8 layers, FFN=512, ~1.63M params, ~50M FLOPs/forward

---

## Verdict

GPU-batched inference for self-play is **not beneficial** for this model size and workload.
The RTX 4060 remains used for **training only** (gradient computation). Self-play inference
continues on CPU via ONNX Runtime with 32 parallel single-threaded sessions.

---

## Profiling Runs (2026-04-15, commit `97ed493`)

Both runs: 64 games, 5-player/7-card, 5 det x 100 sims MCTS, 32 threads.

### End-to-End Iteration Timing

| Config | Self-Play (s) | Training (s) | Total (s) | Total (min) |
|--------|--------------|-------------|-----------|-------------|
| CPU ONNX x32 (batch=1) | ~678 | ~10 | 687.6 | 11.5 |
| GPU CUDA batched (batch<=64) | 715.1 | ~9 | 724.1 | 12.1 |

GPU batched self-play was **5.3% slower** than CPU ONNX despite saturating near max batch.

### GPU Batch Statistics (from `gpu.log`)

| Metric | Value |
|--------|-------|
| Total requests | 5,305,296 |
| Total batches | 87,487 |
| Avg batch size | 60.6 |
| Max batch size | 64 |
| p10 / p50 / p90 / p99 | 62 / 64 / 64 / 64 |
| Evals per decision | ~218 |

Batches were nearly fully packed (p50=64 = concurrent thread count), yet throughput
did not exceed CPU. The batcher was not starved — the GPU forward pass itself is too
cheap to amortize the coordination overhead.

### Throughput Comparison

| Backend | Total Evals | Wall-Clock (s) | Throughput (evals/s) |
|---------|-------------|----------------|---------------------|
| ONNX CPU x32 (batch=1) | ~5.3M | ~678 | ~7,800 |
| CUDA batched x1 stream (avg batch 61) | 5,305,296 | 715.1 | 7,419 |

### Single-Eval Latency (from Section 6.3 benchmarks)

| Backend | Batch | Per-Eval Latency | Notes |
|---------|-------|-----------------|-------|
| ONNX CPU (ort, 1 thread) | 1 | 604 us | Measured in Session 6.3 |
| tch CPU | 1 | ~500 us | Estimated from single-iteration analysis |
| CUDA (amortized) | ~61 | ~135 us | 8.2 ms/batch / 60.6 evals |
| CUDA (kernel only) | 64 | ~10 us | Expected from 50M FLOPs @ 15 TFLOPS |

The gap between CUDA kernel time (~10 us/eval) and measured amortized cost (~135 us/eval)
is entirely coordination overhead: channel send/recv, encoding, H2D/D2H transfer, batch
drain deadline, oneshot scatter.

---

## Why GPU Batching Fails Here

1. **Model is too small.** At ~50M FLOPs, a single forward pass on the RTX 4060 takes
   <0.1 ms of actual compute. CUDA kernel launch overhead (~5-10 us per kernel, ~20+
   kernels per forward) dominates the GPU time.

2. **MCTS is sequential per determinization.** Each tree runs one simulation at a time
   (no virtual loss / leaf queue), so each worker submits one request then blocks.
   Maximum in-flight requests = number of concurrent threads. Batch size is capped by
   thread count, not GPU capacity.

3. **Coordination overhead exceeds compute savings.** The channel round-trip
   (encode -> mpsc send -> batch drain -> pad -> H2D -> forward -> D2H -> oneshot scatter -> recv)
   adds ~125 us per eval that doesn't exist in the CPU ONNX path, where each thread
   runs an independent session with zero synchronization.

4. **CPU ONNX is cache-friendly for this model.** The 1.63M-parameter model fits
   comfortably in L2/L3 cache. At batch=1, ort CPU avoids all GPU transfer costs and
   scales linearly with core count.

---

## Configurations Used

### `bench-baseline.toml` (CPU ONNX)

```toml
[training]
device = "cpu"

[self_play]
num_games = 64
num_threads = 32

[mcts]
num_determinizations = 5
sims_per_determinization = 100

[gpu_eval]
enabled = false
```

### `bench-gpu.toml` (CUDA batched)

```toml
[training]
device = "cuda:0"

[self_play]
num_games = 128
num_threads = 32
concurrent_games = 128

[mcts]
num_determinizations = 5
sims_per_determinization = 100

[gpu_eval]
enabled = true
device = "cuda:0"
max_batch = 128
max_wait_us = 200
pad_to_max_seq = true
```

---

## Historical Context: Python-Era Benchmarks (from `conclusion.md`, 2026-03-14)

For reference, the original Python implementation on the same hardware:

| MCTS Config | Total Sims/Move | Games/Min | Notes |
|-------------|-----------------|-----------|-------|
| Light (2x20) | 40 | 69.1 | 32 MP workers |
| Medium (3x30) | 90 | 36.7 | 32 MP workers |
| Heavy (5x50) | 250 | 16.0 | 32 MP workers |

Full iteration (Python): ~5 min 18s (38s self-play + 4m20s training + 13s checkpoint).
Full iteration (Rust, CPU ONNX): ~11.5 min at 5x100 MCTS — higher sim budget but
still dominated by self-play.

---

## Decision

- **Self-play inference**: CPU ONNX, 32 parallel sessions, batch=1. No change.
- **Training**: GPU (CUDA) via tch-rs. Retained — gradient computation benefits from GPU.
- **GPU batching code** (`gpu_eval.rs`, `BatchedGpuEvaluator`, `batch_sweep` example):
  archived in commit `97ed493`, then removed from working tree.

To revisit GPU inference in the future, the threshold would be a model >10x larger
(~15M+ params, ~500M FLOPs) where GPU compute exceeds coordination overhead.
