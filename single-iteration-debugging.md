Diagnosis — what the single-iteration profile is really telling us
From the 6.3 results and the code paths I read:

ONNX CPU single-eval floor ≈ 0.6 ms/eval (blob-engine/src/onnx.rs:69-111). The model is 1.63M params / ~50M FLOPs; at batch=1 this is launch-overhead bound, not math bound. There is no factor-of-10 left on the CPU path.
MCTS is strictly sequential inside each determinization (blob-engine/src/mcts.rs:296-334): one eval.evaluate() per sim, no virtual loss, no leaf queue. So a single thread running 100 sims = 100 sequential forward passes.
Self-play pool = one OnnxEvaluator per rayon thread (blob-nn/src/self_play.rs:98, blob-engine/src/onnx.rs:45-47). 32 isolated CPU inference streams. No batching anywhere.
Training (tch) already has a device: Device plumbed end-to-end (blob-nn/src/training_loop.rs:70), but self-play inference has zero tch dependency — blob-nn/Cargo.toml:13 only pulls the CPU libtorch.
The evaluator abstraction lives in blob-engine/src/evaluator.rs:30-32 as a sync fn evaluate(&self, &BlobState) -> (Vec<f32>, f32). Nothing about it forces single-sample inference; it just happens that both implementations do that.
Bottom line: the 4060 is idle, CPU is saturated doing 1.28M independent ~0.6 ms inferences per iteration (40M / 32), and no mechanism exists to coalesce them.

Where the "bad proposal" goes wrong
Three real issues with it:

It orders virtual-loss last. Without leaf-parallel MCTS, each worker submits one request and blocks. Steady-state batch = number of workers (32). On a 4060 with a tiny model, launch overhead dominates until batch ≥ ~64; batch=32 is meh, maybe 3× over CPU. The coalescer exists but has nothing interesting to coalesce. Virtual loss + leaf queue is not a "then layer on" — it's the thing that makes the batcher earn its keep.
(B ready) OR (T µs elapsed) is the wrong primary dial for this workload. With sync producers and a fixed worker count, B is never met (producers block as soon as they submit), so T becomes the only real trigger and you're essentially picking a fixed latency/batch trade-off. The right dial is in-flight requests per worker (oversubscription or virtual-loss width) so B can actually fire.
"Avoids a second ONNX→CUDA stack" is a non-reason for the training box. Training already depends on tch; the crate boundary that matters ("blob-engine must never depend on tch", development-plan.md:14) is about the deployment binary blob-bin. A GPU evaluator lives in blob-nn, implements the blob_engine::Evaluator trait, and self-play uses it via a trait object. blob-engine stays clean; blob-bin still links only ort. The real cost of tch CUDA is a bigger libtorch download and LD_LIBRARY_PATH at runtime (you already have the note for this per memory) — not a crate-layering violation.
The correct approach
Think of it as three independent levers, attack them in the order of most-leverage-per-line-of-code:

Lever 1 — GPU-resident batched evaluator (prerequisite)
New crate module: blob_nn::gpu_eval::BatchedGpuEvaluator. Owns one BlobNet on Device::Cuda(0) (shared with training — no export round-trip, no ONNX in the self-play hot loop; the ONNX export stays only for best_model.onnx / deployment).
Gate it behind a cuda cargo feature so Windows/deployment builds stay CPU-only.
Implements blob_engine::Evaluator (sync API, no trait changes needed). Internally:
An MPSC (crossbeam_channel or flume) of (EncodedState, oneshot::Sender<Resp>).
A single dedicated inference thread that drains up to max_batch requests, calls pad_batch ([blob-nn/src/input.rs]) + model.forward_* once on CUDA, scatters results back via the oneshots.
Sync evaluator call: encode → send(req) → recv() — the worker thread blocks, but the GPU gets a real batch.
Flush policy: batch_ready(N) OR deadline(T µs) is fine, but the critical point is keeping the inference thread always running back-to-back forward passes — as soon as forward i finishes, immediately kick off forward i+1 on whatever's queued. The deadline matters only at tail.
Only one inference thread. Two CUDA-stream inference threads is tempting but not worth it on a 4060 for a 50M-FLOP model; you want ALL GPU time spent on forward passes, not context switching.
Lever 2 — Feed it enough work (the actual win)
Without this, Lever 1 gives you ~3× and calls it a day. Pick one of these two patterns; I recommend starting with 2A because it's ~50 lines of code:

2A — Oversubscribe games. Spawn num_games_in_flight = 128 or 192 worker threads (not num_cpus). Each runs one game sequentially. Each blocks on the evaluator most of the time, so CPU oversubscription is cheap. Target steady-state batch = ~128 at the GPU. One-line change in blob-nn/src/self_play.rs:71-74 + a concurrent_games knob distinct from core count.

2B — Virtual loss + leaf queue inside MCTS. More complex; requires touching blob-engine/src/mcts.rs:296-334. Per tree, keep K leaves in flight; add a "virtual loss" increment on visit_count / value_counts on selection so sibling selections diverge; commit the real update on result arrival. This multiplies batch by K per thread. The right answer if 2A's game-level parallelism tops out (e.g. 5-card rounds where 85 decisions × 5 dets dominates and per-tree latency is what matters for wall-clock).

Reality for your hardware: 2A alone probably hits the 5-min iteration gate. 2B is the cleaner long-term answer and also raises search quality (leaf batching is standard AlphaZero). My recommendation: do 2A first, measure, add 2B only if needed — most teams never need both on a single-GPU box.

Lever 3 — Remove friction around it
These are boring but necessary:

Stop re-loading ONNX per iteration in self-play. blob-nn/src/self_play.rs:98 loads from disk per game; switch self-play to take an Arc<dyn Evaluator> and construct the evaluator once per iteration from the in-memory VarStore. ONNX export becomes a promotion-only step, not a hot-loop step.
Pre-pad sequences to a fixed MAX_SEQ inside the batcher so CUDA kernel plans stay cached. A MAX_SEQ = 58 pad is ~15% extra work but eliminates recompile-per-shape cost (called out in the 6.3 "held-in-reserve" list at development-plan.md:593 and it applies even more strongly to GPU).
Reuse EncodedState's hand_card_indices slice in run_encoded-equivalent — the ONNX path re-encodes inside evaluate() (blob-engine/src/onnx.rs:121); the batched GPU path should encode on the submitting worker thread (parallel, off the critical path) and ship only the encoded tensors to the inference thread.
Lever 4 — Instrumentation (ship with Lever 1, not after)
Non-negotiable data to log on day one so you can tell which lever is stuck:

Batch-size histogram (p10/p50/p90/p99 per iteration). If p50 < 32 after Lever 2A, producers aren't actually blocking → something's serializing.
Per-request breakdown: submit → picked_up → gpu_done → recv timestamps. Distinguishes "GPU busy" from "queue starved" from "scatter slow".
Inference-thread duty cycle: fraction of wall-clock doing forward(). Target > 90%. Under 50% means Lever 2 insufficient.
H2D / D2H bytes per iteration: the proposal mentions PCIe time — useful to have, but on this model (seq ≤ 58, feat=48, bs=128 → ~3 MB per forward) PCIe is not plausibly the bottleneck; the metric is there to rule out regressions, not as a primary signal.
Sequencing
Add cuda feature to blob-nn, wire libtorch-CUDA download (or document env-var override, matching your existing LD_LIBRARY_PATH note).
Write BatchedGpuEvaluator + inference-thread harness. Unit-test against DummyEvaluator-equivalent tch model. Parity test vs the existing tch-CPU forward at batch=1 within 1e-5 before touching self-play.
Swap self_play_iteration to accept &(dyn Evaluator + Sync) instead of &Path; caller in TrainingLoop::run_iteration builds one batched evaluator per iteration.
Raise concurrent_games to ~128 (Lever 2A). Measure: batch-size histogram, iteration wall-clock, GPU SM utilization (nvidia-smi dmon).
Decision point. If iteration < 5 min and GPU duty > 70%: done. If not, add virtual loss + leaf queue (Lever 2B).
Only after wall-clock is green: prune the now-dead OnnxEvaluator usage from the self-play path (keep it for deployment).