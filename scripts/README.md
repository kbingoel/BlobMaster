# Scripts

## `blobmaster-train.sh`
Runs the release `blobmaster-train` with the downloaded libtorch on the library path and `libtorch_cuda.so` preloaded (see "Runtime env" below), passing its arguments through:

```bash
cargo build --release -p blob-train
scripts/blobmaster-train.sh pretrain --config blob-train/pretrain.sample.toml --output checkpoints/<run>
scripts/blobmaster-train.sh export --output <dir> [--checkpoint <checkpoint dir>] [--check]
```

`blobmaster-train` links libtorch since Phase 4, so even `export` needs the library path.

`train` (Phase 5) also needs `blobmaster` built beside it (`cargo build --release -p blob-bin -p blob-train`): its actors run as `blobmaster selfplay` and its benches as `blobmaster bench`, in processes without libtorch (AGENTS.md, Training). Launch it detached and read `<run>/status.md`:

```bash
mkdir -p checkpoints/<run>
setsid nohup scripts/blobmaster-train.sh train --config blob-train/train.sample.toml --output checkpoints/<run> \
  > checkpoints/<run>/train.log 2>&1 < /dev/null &
touch checkpoints/<run>/STOP      # save and exit; continue with: train --resume --output checkpoints/<run>
touch checkpoints/<run>/PAUSE     # freeze until removed
touch checkpoints/<run>/FINISH    # final publish and benches now
```

## `plot_rl_run.py`
Draws a `train` run into `<run>/plots/`, reading `metrics.jsonl`, the reports in `bench/` and the checkpoints in `models/`; every x-axis is running hours. Modelled on the gen-1 charts below (CI bands, P against its opponents' bid success, convergence, held out vs training, weight evolution):
- `00_overview.png`: the run on one page;
- `01_strength.png`: network-only benches vs rule bot 2 and the rule bot, search vs rule bot 2, with 95% CI bands, absolute and paired;
- `02_bids.png`: bids made by cards dealt and 0-bids, P (solid) against its opponents (dashed);
- `03_learning.png`: training losses, P's agreement with search, P's and V's change per publish, learner pace;
- `04_generalization.png`: held out, validation vs an equal training sample; drift from rule bot 2 and V's last-trick error on fixed teacher states;
- `05_selfplay.png`: search health (disagreement with P, KL, bid entropy, bids played off the top), bids made, throughput;
- `06_weights.png`: change per publish by layer (P and V heatmaps) and distance from the warm start; needs torch, `--no-weights` skips it;
- `08_value_margin.png` (runs from 2026-10-07): search's margin over P alone on the same deals, V on the fixed rounds of an earlier run, V on its stream of P-alone rounds, and the learner's pace (P + V steps, V-only updates).

Run it with `( unset LD_PRELOAD; .venv/bin/python scripts/plot_rl_run.py checkpoints/<run> )`.

## `export_onnx.py`
Writes a model directory (`policy.onnx`, `value.onnx`, `meta.json`; gen-2.md §5.3) for `OnnxPolicy` / `OnnxValue`; see the file header. Run it through `blobmaster-train export`, which uses the repo's `.venv` and strips `LD_PRELOAD`. `--checkpoint` takes a learner checkpoint directory (`policy.ot`, `value.ot`, `meta.json`) and loads both networks strictly; without it, both are random-init. `pretrain` exports its final checkpoint to `<run>/model` itself.

Both ONNX files carry the encoder's layout id in their metadata, and the Rust evaluators refuse a model with another. The script's token widths and `LAYOUT_ID` mirror `blob-engine/src/encoder.rs`; the Rust test `export_script_mirrors_feature_widths` checks them.

`--check` compares PyTorch and ONNX Runtime on random in-range inputs and fails above `PARITY_GATE` (1e-5, relative above 1). Trained weights pass; a random *tch* init (`save_random_checkpoint`) doesn't (~2e-5 for P), because tch initializes with ~2.5× torch's weight scale. The authoritative check is the Rust parity test on real game states (`blob-nn/tests/onnx_tch_parity.rs`, P and V, 1e-5).

## `visualize_strength.py`, `visualize_weight_evolution.py`
Plot a run's metrics and the evolution of its weights. They still read gen-1 outputs (`strength.csv`, per-iteration `metrics.jsonl`, `iter_*` directories) and get re-pointed at gen-2 outputs with the Phase-5 driver, whose evaluator produces the strength series they plot (gen-2.md §4).

## Runtime env for libtorch (Linux, CUDA)

`tch` with `download-libtorch` drops libtorch into
`target/{debug,release}/build/torch-sys-*/out/libtorch/libtorch/lib`. Binaries
that link it (the `blob-nn` and `blob-train` tests, `blobmaster-train`) need the dynamic loader,
and CUDA's lazy symbol resolution, to find it explicitly; `blobmaster-train.sh`
does this:

```bash
LIBTORCH_DIR="$(find target/release/build -maxdepth 6 -type d -name lib -path '*/libtorch/libtorch/lib' | head -n1)"
LD_LIBRARY_PATH="$LIBTORCH_DIR" cargo test --release -p blob-nn -p blob-train   # CPU
LD_LIBRARY_PATH="$LIBTORCH_DIR" LD_PRELOAD="$LIBTORCH_DIR/libtorch_cuda.so" <binary>   # GPU runs
```

Without `LD_LIBRARY_PATH`: load fails at startup (missing `libtorch_cpu.so`).
Without `LD_PRELOAD=libtorch_cuda.so`: a CPU-only libtorch (`pretrain` refuses
`cuda`; anything else silently runs on the CPU) or cryptic CUDA symbol errors —
the CUDA backend isn't pulled in eagerly otherwise. Never let
`LD_PRELOAD` reach the venv's Python (`import torch` crashes on the ABI
mismatch); `blobmaster-train export` removes it.

Swap `target/release` → `target/debug` for debug builds. Re-run the `find` if
`torch-sys` rebuilds (hash in the path changes).

The pinned venv (Python 3.12.3, `torch==2.5.1+cu124`, `onnxruntime==1.24.4`,
`onnx==1.21.0`, `numpy==2.4.4`), the `tch` pin and the driver notes are in
AGENTS.md, Runtime environment.
