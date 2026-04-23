# Scripts

## `export_onnx.py`
Exports a trained `blob-nn` checkpoint to ONNX for `OnnxEvaluator`. See file header.

## Runtime env for `blob-train` (Linux, CUDA libtorch)

`tch` with `download-libtorch` drops libtorch into
`target/{debug,release}/build/torch-sys-*/out/libtorch/libtorch/lib`. The binary
links against it, but at runtime the dynamic loader and CUDA's lazy symbol
resolution both need to find it explicitly:

```bash
LIBTORCH_DIR="$(find target/release/build -maxdepth 6 -type d -name lib -path '*/libtorch/libtorch/lib' | head -n1)"
export LD_LIBRARY_PATH="$LIBTORCH_DIR:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$LIBTORCH_DIR/libtorch_cuda.so"
./target/release/blob-train <args>
```

Without `LD_LIBRARY_PATH`: load fails at startup (missing `libtorch_cpu.so`).
Without `LD_PRELOAD=libtorch_cuda.so`: CPU fallback or cryptic CUDA symbol
errors — the CUDA backend isn't pulled in eagerly otherwise.

Swap `target/release` → `target/debug` for debug builds. Re-run the `find` if
`torch-sys` rebuilds (hash in the path changes).
