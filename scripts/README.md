# Scripts

## `export_onnx.py`
Exports a `blob-nn` checkpoint (`model.ot`) to ONNX for `OnnxEvaluator`; see the file header. Run it through

```bash
./target/release/blobmaster-train export --checkpoint <dir or model.ot> --output <model.onnx> [--check]
```

which uses the repo's `.venv` and strips `LD_PRELOAD`. The script's token widths mirror `blob-engine/src/encoder.rs`; the Rust test `export_script_mirrors_feature_widths` checks them.

## `visualize_strength.py`, `visualize_weight_evolution.py`
Plot a run's metrics and the evolution of its weights. They still read gen-1 outputs (`strength.csv`, per-iteration `metrics.jsonl`, `iter_*` directories) and get re-pointed at gen-2 outputs together with the learner (gen-2.md §4).

## Runtime env for libtorch (Linux, CUDA)

`tch` with `download-libtorch` drops libtorch into
`target/{debug,release}/build/torch-sys-*/out/libtorch/libtorch/lib`. Binaries
that link it (the `blob-nn` tests now, the learner from Phase 4) need the
dynamic loader, and CUDA's lazy symbol resolution, to find it explicitly:

```bash
LIBTORCH_DIR="$(find target/release/build -maxdepth 6 -type d -name lib -path '*/libtorch/libtorch/lib' | head -n1)"
LD_LIBRARY_PATH="$LIBTORCH_DIR" cargo test --release -p blob-nn      # CPU
LD_LIBRARY_PATH="$LIBTORCH_DIR" LD_PRELOAD="$LIBTORCH_DIR/libtorch_cuda.so" <binary>   # GPU runs
```

Without `LD_LIBRARY_PATH`: load fails at startup (missing `libtorch_cpu.so`).
Without `LD_PRELOAD=libtorch_cuda.so`: CPU fallback or cryptic CUDA symbol
errors — the CUDA backend isn't pulled in eagerly otherwise. Never let
`LD_PRELOAD` reach the venv's Python (`import torch` crashes on the ABI
mismatch); `blobmaster-train export` removes it.

Swap `target/release` → `target/debug` for debug builds. Re-run the `find` if
`torch-sys` rebuilds (hash in the path changes).
