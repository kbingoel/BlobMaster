potential optimisations (not checked) worth adressing when starting from scratch, in no order:


delta-buffer idea
It's a good idea and it's actually how AlphaZero/MuZero-style trainers typically do it. The math: capacity 500k, ~48,640 examples/iter → window = ~10.3 iters. Currently each iter_N/buffer.bin is a snapshot of that whole window, so adjacent iters share ~90% of their data. Per-iter deltas would cut each file from ~200 MB to ~20 MB — a ~10× reduction, matching your estimate.

At 125 iters with retention disabled (training_loop.rs:768 — pruning was turned off 2026-04-29 for weight-evolution analysis), that's a swing from ~25 GB down to ~2.5 GB. Almost certainly what would have prevented today's crash.

Sketch of the change:

Per iter, write only the new examples generated that iter: iter_N/examples.bin (~20 MB).
Resume: walk iter dirs newest → oldest, load examples.bin from each, push into a fresh buffer until you hit buffer_capacity or run out of iters. Result is identical to today's hot resume — the FIFO's contents are exactly "last N iters of self-play".
The old buffer.bin save path stays alive long enough to read existing runs (run-2026-05-14, the sweep dirs), then deprecate.
Trade-offs worth knowing about:

Mid-window corruption matters now. Today, if iter_124/buffer.bin is fine and iter_125/buffer.bin is corrupt, resume from 125 falls back to cold, but 124 is still a clean hot-resume option. With deltas, a corrupt iter_119/examples.bin punches a hole in the reconstructed window — you'd want logic to either skip that iter and continue further back, or accept a partially-full buffer.
Order matters: the buffer is a circular FIFO. Loading deltas oldest-first preserves chronological order; doing it newest-first works for "fill until full" but the write-index won't match the original. For training (uniform sampling) that's fine; for any analysis that depends on insertion order it's not.
Bookkeeping per delta: should include the iter number and example count, so reconstruction can sanity-check (e.g. warn if a delta's count doesn't match its iter's logged examples= field).
Capacity changes become natural: just read more or fewer prior iters. Today the load path has to honor the saved capacity to avoid losing data.
One-time migration concern: this current run (run-2026-05-14) already has the old-format snapshots. Either keep the old loader for back-compat, or do a one-shot conversion script that splits the iter-N snapshot into iter-N's delta by diffing against iter-(N-1)'s snapshot.





**async training**
Quick read of your stack (key files)
blob-nn/src/training_loop.rs:582-632 — train_one_step: samples 512 from buffer → encodes on the calling thread → forward/backward on GPU. Two steps per call (bid + play sub-batches).
blob-nn/src/training_loop.rs:649-683 — train_phase: steps_per_epoch = ceil(buffer_len / batch_size), repeated up to epochs_per_iteration, with relative early-stop.
blob-engine/src/replay.rs:1-12 — explicit "store raw BlobState, re-encode at sample time" decision. This is the single-CPU-thread cost.
blob-nn/src/train.rs:127-185 — LrSchedule: linear warmup over warmup_steps only inside iteration 0, then iteration-relative cosine to MIN_LR.
[blob-nn/src/engine.rs] — self_play_iteration uses rayon map_init so each worker holds one ONNX session and reuses across games.
[blob-train/config.sample.toml] — batch_size=512, epochs_per_iteration=10, buffer_capacity=500_000, num_games=118, num_threads=32, device="cuda:0".
Epochs and batch size, concretely on your numbers
Batch size 512: at every gradient step you sample 512 examples from the buffer, encode them, and run one forward+backward+optimizer step. Two such steps fire per train_one_step call (one batch sampled per phase: bidding and playing). It's the GPU's "mouthful." Bigger = smoother gradient, more GPU work per host roundtrip; smaller = noisier gradient, more frequent updates per example seen. 512 is a sane default for a 1.63M-param transformer on an 8GB card.
steps_per_epoch is derived: ceil(buffer_len / batch_size). So at full buffer it's ceil(500000 / 512) = 977 steps. One "epoch" is not one pass over distinct examples — it's "977 random samples from the buffer with replacement." Because sampling is uniform with replacement, in expectation each example is touched ~once per epoch.
Epochs per iteration = 10: ten such passes, so each example in the buffer is sampled ~10 times over an iteration before the next 80k self-play examples mix in. Combined with the 9-iter buffer, the average example contributes to ~10×9=90 gradient steps over its lifetime in the buffer. That number is the real "how hard are we training on each datum" knob, and it's what changing epochs moves. Lower epochs = data churn dominates, smoother but slower learning. Higher epochs = more updates per example, faster per iter but higher overfit risk on a stale buffer (exactly the regression cold_buffer_post_resume_epochs was added to mitigate at training_loop.rs:653-662).
Early stop: epoch_early_stop_rel=0.005 cuts the 10-epoch budget short if the loss stops dropping fast enough. So sometimes you only run 3-4 epochs.
On warmup: your schedule does have one (linear over warmup_steps inside iter 0 only), but it's stepwise, not "LR=0 until buffer fills." The reason buffer-fill doesn't need a true zero-LR phase is that early gradients on tiny buffers are still pointed in the right direction — they're just noisy. The Run-3 incident on 2026-04-29 (the cold-buffer-on-resume regression) is the same idea inverted: peak-LR + tiny buffer + many epochs = overfit the small fresh slice. That's the practical edge of "warmup matters."

CPU↔GPU during training, and why training looks single-threaded
The hot loop in train_one_step is:


buffer.sample_batch(512)      ← calling thread, CPU
  for each of 512 BlobStates:
    encode(state)             ← CPU; reb**async training**
Quick read of your stack (key files)
blob-nn/src/training_loop.rs:582-632 — train_one_step: samples 512 from buffer → encodes on the calling thread → forward/backward on GPU. Two steps per call (bid + play sub-batches).
blob-nn/src/training_loop.rs:649-683 — train_phase: steps_per_epoch = ceil(buffer_len / batch_size), repeated up to epochs_per_iteration, with relative early-stop.
blob-engine/src/replay.rs:1-12 — explicit "store raw BlobState, re-encode at sample time" decision. This is the single-CPU-thread cost.
blob-nn/src/train.rs:127-185 — LrSchedule: linear warmup over warmup_steps only inside iteration 0, then iteration-relative cosine to MIN_LR.
[blob-nn/src/engine.rs] — self_play_iteration uses rayon map_init so each worker holds one ONNX session and reuses across games.
[blob-train/config.sample.toml] — batch_size=512, epochs_per_iteration=10, buffer_capacity=500_000, num_games=118, num_threads=32, device="cuda:0".
Epochs and batch size, concretely on your numbers
Batch size 512: at every gradient step you sample 512 examples from the buffer, encode them, and run one forward+backward+optimizer step. Two such steps fire per train_one_step call (one batch sampled per phase: bidding and playing). It's the GPU's "mouthful." Bigger = smoother gradient, more GPU work per host roundtrip; smaller = noisier gradient, more frequent updates per example seen. 512 is a sane default for a 1.63M-param transformer on an 8GB card.
steps_per_epoch is derived: ceil(buffer_len / batch_size). So at full buffer it's ceil(500000 / 512) = 977 steps. One "epoch" is not one pass over distinct examples — it's "977 random samples from the buffer with replacement." Because sampling is uniform with replacement, in expectation each example is touched ~once per epoch.
Epochs per iteration = 10: ten such passes, so each example in the buffer is sampled ~10 times over an iteration before the next 80k self-play examples mix in. Combined with the 9-iter buffer, the average example contributes to ~10×9=90 gradient steps over its lifetime in the buffer. That number is the real "how hard are we training on each datum" knob, and it's what changing epochs moves. Lower epochs = data churn dominates, smoother but slower learning. Higher epochs = more updates per example, faster per iter but higher overfit risk on a stale buffer (exactly the regression cold_buffer_post_resume_epochs was added to mitigate at training_loop.rs:653-662).
Early stop: epoch_early_stop_rel=0.005 cuts the 10-epoch budget short if the loss stops dropping fast enough. So sometimes you only run 3-4 epochs.
On warmup: your schedule does have one (linear over warmup_steps inside iter 0 only), but it's stepwise, not "LR=0 until buffer fills." The reason buffer-fill doesn't need a true zero-LR phase is that early gradients on tiny buffers are still pointed in the right direction — they're just noisy. The Run-3 incident on 2026-04-29 (the cold-buffer-on-resume regression) is the same idea inverted: peak-LR + tiny buffer + many epochs = overfit the small fresh slice. That's the practical edge of "warmup matters."

CPU↔GPU during training, and why training looks single-threaded
The hot loop in train_one_step is:


buffer.sample_batch(512)      ← calling uilds entity-token tensor from a 410B BlobState
  pad_batch(...)              ← CPU; produces a CPU tensor
  to_device(cuda:0)            ← PCIe copy
  forward / backward           ← GPU
  optimizer.step()             ← GPU
That outer loop runs on whichever single thread called train_phase. There's no rayon, no async, no producer pipeline. So while the GPU executes the forward/backward (~maybe 10-30ms for a 1.63M-param batch=512 on a 4060), the CPU is idle. While the CPU re-encodes and copies, the GPU is idle. That's the "1 thread + GPU not pegged" you observed — both halves are running, just never simultaneously.

The reason it's single-threaded is deliberate decoupling, not a bug: replay.rs:8-11 chose to store raw BlobStates and re-encode at sample time so encoder changes don't invalidate stored buffers. The cost is paid every gradient step. The fix would be either (a) parallelize sample_batch re-encoding via rayon (one-line change, gets you ~10×, free), or (b) cache encoded tensors in the buffer (faster still but breaks the encoder-evolution property). On a 100-iter run, that single change is probably worth more than any of the 7.4 levers you ruled out.

(And per your earlier discussion: you don't have validated training profiling, so the actual GPU vs CPU split is inferred from wall_clock_secs - 552. The single-thread re-encoding is the most obvious culprit but you should instrument to be sure.)

Async actor-learner — would it be viable from scratch?
Yes, and on this hardware it's a particularly good fit. Your intuition is right.

Why your hardware especially likes it. Self-play and training use disjoint resources: 32 logical CPU cores running ORT sessions for self-play, 1 GPU running libtorch for training. In sync iter-mode they take turns using the machine even though they don't compete for the same hardware. Async lets both run flat-out:

N rayon workers (say, 28-30) running self-play forever, dropping examples into a shared buffer.
1 trainer thread (or 2 — one for sample/encode, one for GPU step) consuming from the buffer.
A small "publish" thread that writes a fresh model.onnx every K gradient steps; self-play workers hot-swap their OnnxEvaluator on next game start.
1-2 cores left over for OS / export / eval.
Why AlphaZero-style is forgiving of off-policy lag. Quoting back what we settled before: targets are MCTS visit distributions and z-scored final scores, not the model's own outputs. They're effectively supervised labels. A sample that's 9 iters old is fine because some past good model computed it via tree search; one more iter of staleness is noise on noise.

The throughput math actually self-corrects. At your numbers: self-play ~145 decisions/sec across 32T (~9 min for 118×~680 dec), trainer at 512×2 examples per ~0.3-0.4s ≈ ~3000 examples/sec consumed. That's a ~20× consume:produce ratio, which means each example gets sampled ~20 times before being evicted from a 500k buffer. Today you sample each example ~10 times (10 epochs at full buffer); async would land you in the same neighborhood by happenstance — and you'd tune it by either capping trainer rate, widening the model, or making the buffer larger.

What gets harder.

LR schedule: cosine has to key off gradient steps or wall time, not iters. (Your existing LrSchedule is half-prepared for this — train.rs:140-185 reads iteration and step_in_run already.)
Eval anchors: "iter K vs anchor" stops being a thing. You'd eval on step milestones (e.g., every 5,000 steps) or wall-time milestones.
Reproducibility: thread timing affects sampling order, so two runs from the same seed don't bit-match. Usually fine for research, but worth knowing.
Stop/start "cooking" semantics get nicer, not harder: pausing the trainer leaves self-play running and the buffer freshening; restarting picks up with no cold-buffer regime. The Run-3 mitigation you needed in iter-mode (cold_buffer_post_resume_epochs) literally goes away.
ONNX hot-swap: needs file-rename atomicity + workers checking mtime/version on game boundary. Not hard, but not free.
Single-CPU-thread re-encoding bites you in both architectures and should be fixed regardless.
The warmup question, rephrased for async. You don't run training at LR=0 until the buffer fills; you just don't start the trainer until the buffer has, say, 50k examples (or 1 full iter's worth). That's a launch-condition gate, not an LR schedule. Your linear warmup over the first ~1000 grad steps still applies on top of that, for the same reason it applies in sync: the freshly-initialized network is fragile.

Concrete sketch of "if I redid this from scratch":

One persistent ReplayBuffer shared via Arc<Mutex<...>> or a lock-free MPMC ring (preferred — your buffer is already a circular FIFO).
Self-play actors: rayon thread pool, each owns an OnnxEvaluator, polls a "current model version" AtomicU64, hot-swaps when bumped.
Trainer: dedicated thread; loop = (parallel sample+encode → GPU step → maybe publish ONNX → maybe eval).
LR keyed off global_step; warmup gate keyed off buffer.len() >= warmup_threshold.
Replace "iter K complete" with "step S complete" in metrics.jsonl. Eval cadence by step.
STOP file works the same; on receipt, drain self-play in-flight games, save buffer + model, exit.
This is a significant refactor (the train driver, LR schedule, eval orchestrator, and pruning logic all touch iteration as a first-class concept), but every individual change is local and the win is real on your specific hardware mix. For V2 of this project, I'd build it this way from day one.

The harder honest question is whether V1 should be retrofitted now. My take: not while a 100-iter run is in flight and you're trying to validate the original pipeline. The single-thread re-encoding fix is the strict-prerequisite, low-risk speedup; do that first regardless. Async is a V2 redesign, not a 7.x patch.