"""Session 7.4b deferred-lever sweep for overnight 2026-04-27.

Reuses `scripts.export_onnx.load_calibration` plus the prepped FP32 graph
from `quantize_int8`, then runs three independent INT8 quantization levers
that were parked at the end of 7.4b:

    * `s8s8`     — `activation_type = QInt8` (symmetric activations)
    * `entropy`  — `CalibrationMethod.Entropy` (KL-min instead of MinMax)
    * `exclude-block N` — exclude all nodes whose name contains
                          `/encoder/<N>/` (in addition to the LN/Softmax/head
                          exclusions already baked into the 7.4b path).

For each lever we (1) write `model.int8.onnx` into `--out-dir/<lever>/`,
(2) run validate_int8 over the same calibration BCAL we built it with,
(3) print a one-line JSON summary to stdout and append a JSONL row to
`--results-jsonl`.

This is a one-off sweep helper; it deliberately does not modify
`scripts/export_onnx.py` so the regular training-time INT8 path is
untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Reuse pieces from export_onnx so the BCAL format and exclusion list stay
# in one place.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_onnx import _BcalDataReader, load_calibration  # noqa: E402


HEAD_PREFIXES = (
    "/value_head/",
    "/fc1/",
    "/fc2/",
    "/fc1_1/",
    "/fc2_1/",
)
EXCLUDED_OPS = {"LayerNormalization", "Softmax"}

POLICY_AGREEMENT_GATE = 0.95
VALUE_SIGN_AGREEMENT_GATE = 1.0


def prep_graph(fp32_path: Path) -> Path:
    """Run quant_pre_process and return the path to the prepped ONNX."""
    from onnxruntime.quantization.shape_inference import (  # type: ignore
        quant_pre_process,
    )

    tf = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tf.close()
    out = Path(tf.name)
    quant_pre_process(
        input_model_path=str(fp32_path),
        output_model_path=str(out),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
    )
    return out


def collect_baseline_exclusions(prepped_path: Path) -> list[str]:
    """Same exclusion list as `quantize_int8` baseline — LN/Softmax + heads."""
    import onnx  # type: ignore

    graph = onnx.load(str(prepped_path)).graph
    excluded: list[str] = []
    for n in graph.node:
        if n.op_type in EXCLUDED_OPS:
            excluded.append(n.name)
            continue
        if n.name and n.name.startswith(HEAD_PREFIXES):
            excluded.append(n.name)
    return excluded


def collect_block_node_names(prepped_path: Path, block_idx: int) -> list[str]:
    """Names of every node belonging to transformer block `block_idx`.

    The transformer encoder export names blocks with prefix
    `/encoder/layers.<i>/...` from the PyTorch nn.Sequential. We match by
    substring so we catch sub-graph nodes (qkv split, scale, softmax, etc.)
    that the head-prefix list above doesn't cover.
    """
    import onnx  # type: ignore

    needles = (
        f"/encoder/layers.{block_idx}/",
        f"/encoder/layers/{block_idx}/",
        f"/layers.{block_idx}/",
    )
    graph = onnx.load(str(prepped_path)).graph
    matched: list[str] = []
    for n in graph.node:
        if not n.name:
            continue
        if any(needle in n.name for needle in needles):
            matched.append(n.name)
    return matched


def quantize_with_lever(
    prepped_path: Path,
    out_path: Path,
    states: list[dict],
    *,
    activation_type_name: str,
    calibrate_method_name: str,
    extra_exclusions: list[str],
) -> int:
    """Run quantize_static with the chosen lever options.

    Returns the total number of excluded node names actually used.
    """
    from onnxruntime.quantization import (  # type: ignore
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    activation_type = {
        "QUInt8": QuantType.QUInt8,
        "QInt8": QuantType.QInt8,
    }[activation_type_name]
    calibrate_method = {
        "MinMax": CalibrationMethod.MinMax,
        "Entropy": CalibrationMethod.Entropy,
        "Percentile": CalibrationMethod.Percentile,
    }[calibrate_method_name]

    baseline_excl = collect_baseline_exclusions(prepped_path)
    nodes_to_exclude = list(dict.fromkeys(baseline_excl + extra_exclusions))

    reader = _BcalDataReader(states)
    quantize_static(
        model_input=str(prepped_path),
        model_output=str(out_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=activation_type,
        calibrate_method=calibrate_method,
        nodes_to_exclude=nodes_to_exclude,
        per_channel=True,
        reduce_range=False,
    )
    return len(nodes_to_exclude)


def run_validation(fp32_path: Path, int8_path: Path, states: list[dict]) -> dict:
    import numpy as np  # type: ignore
    import onnxruntime as ort  # type: ignore

    sess_fp = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_q = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    bid_match = 0
    play_match = 0
    value_sign_match = 0
    for st in states:
        feed = {
            "features": st["features"],
            "token_types": st["token_types"],
            "chrono_indices": st["chrono_indices"],
            "attention_mask": st["attention_mask"],
        }
        bf, pf, vf = sess_fp.run(None, feed)
        bq, pq, vq = sess_q.run(None, feed)
        if int(np.argmax(bf[0])) == int(np.argmax(bq[0])):
            bid_match += 1
        if int(np.argmax(pf[0])) == int(np.argmax(pq[0])):
            play_match += 1
        if np.sign(vf[0]) == np.sign(vq[0]):
            value_sign_match += 1
    n = len(states)
    return {
        "n": n,
        "bid_argmax_agree": bid_match / n if n else 0.0,
        "play_argmax_agree": play_match / n if n else 0.0,
        "value_sign_agree": value_sign_match / n if n else 0.0,
    }


def lever_label(args: argparse.Namespace) -> str:
    if args.lever == "exclude-block":
        return f"exclude-block-{args.block_idx}"
    return args.lever


def configure_lever(args: argparse.Namespace, prepped_path: Path) -> dict:
    if args.lever == "s8s8":
        return {
            "activation_type_name": "QInt8",
            "calibrate_method_name": "MinMax",
            "extra_exclusions": [],
        }
    if args.lever == "entropy":
        return {
            "activation_type_name": "QUInt8",
            "calibrate_method_name": "Entropy",
            "extra_exclusions": [],
        }
    if args.lever == "exclude-block":
        names = collect_block_node_names(prepped_path, args.block_idx)
        if not names:
            sys.stderr.write(
                f"[int8_levers] WARN: no nodes matched encoder block "
                f"{args.block_idx}; lever will degenerate to baseline\n"
            )
        return {
            "activation_type_name": "QUInt8",
            "calibrate_method_name": "MinMax",
            "extra_exclusions": names,
        }
    raise ValueError(f"unknown lever: {args.lever}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fp32", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--lever",
        choices=("s8s8", "entropy", "exclude-block"),
        required=True,
    )
    p.add_argument("--block-idx", type=int, default=-1)
    p.add_argument("--results-jsonl", type=Path, required=True)
    args = p.parse_args()

    if args.lever == "exclude-block" and args.block_idx < 0:
        sys.stderr.write("[int8_levers] --lever exclude-block requires --block-idx\n")
        sys.exit(2)

    label = lever_label(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    int8_path = args.out_dir / "model.int8.onnx"

    states = load_calibration(args.calibration)
    if not states:
        sys.stderr.write(f"[int8_levers] {args.calibration} has no states\n")
        sys.exit(2)

    prepped = prep_graph(args.fp32)
    try:
        cfg = configure_lever(args, prepped)
        n_excl = quantize_with_lever(
            prepped,
            int8_path,
            states,
            activation_type_name=cfg["activation_type_name"],
            calibrate_method_name=cfg["calibrate_method_name"],
            extra_exclusions=cfg["extra_exclusions"],
        )
    finally:
        if prepped.exists():
            prepped.unlink()

    metrics = run_validation(args.fp32, int8_path, states)
    bid_pass = metrics["bid_argmax_agree"] >= POLICY_AGREEMENT_GATE
    val_pass = metrics["value_sign_agree"] >= VALUE_SIGN_AGREEMENT_GATE

    row = {
        "lever": label,
        "n_calibration_states": metrics["n"],
        "bid_argmax_agree": metrics["bid_argmax_agree"],
        "play_argmax_agree": metrics["play_argmax_agree"],
        "value_sign_agree": metrics["value_sign_agree"],
        "bid_gate_pass": bid_pass,
        "value_gate_pass": val_pass,
        "static_gate_pass": bid_pass and val_pass,
        "nodes_excluded_total": n_excl,
        "int8_path": str(int8_path),
    }

    args.results_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.results_jsonl.open("a") as fh:
        fh.write(json.dumps(row) + "\n")

    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
