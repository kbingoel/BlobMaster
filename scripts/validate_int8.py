"""Session 7.4b INT8-vs-FP32 agreement gate.

Runs both ONNX models on every state in a BCAL calibration file and reports:

  * argmax-policy agreement (per-state, masked): fraction of states where
    INT8 and FP32 pick the same `bid_policy` argmax. Gate: ≥ 95%.
  * value-sign agreement: fraction of states where the two value heads
    agree on `sign(value)`. Gate: 100%.
  * argmax `play_scores` agreement over hand-card slots (raw — caller in
    Rust applies the legality mask, but the slot ordering is identical
    between FP32 and INT8 so a raw argmax is a valid invariant). Reported
    for context, not gated by the dev plan.

Exit status: 0 iff both gates pass; 1 otherwise.

Typical invocation (after `export_onnx.py --int8-out ... --calibration ...`):

    python scripts/validate_int8.py \
        --fp32 path/to/model.onnx \
        --int8 path/to/model.int8.onnx \
        --calibration path/to/calibration.bin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Re-use the BCAL reader from export_onnx so the format stays in one place.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_onnx import load_calibration  # noqa: E402


POLICY_AGREEMENT_GATE = 0.95
VALUE_SIGN_AGREEMENT_GATE = 1.0


def run_models(fp32: Path, int8: Path, states: list[dict]):
    import onnxruntime as ort  # type: ignore

    sess_fp = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    sess_q = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])

    bid_match = 0
    play_match = 0
    play_total = 0
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
        # play_scores are per-token; the meaningful comparison is the argmax
        # over the variable-length sequence. Matches mean both nets pick the
        # same token as "best", which downstream legality masking can only
        # *reduce* the top-1 disagreement rate from.
        if int(np.argmax(pf[0])) == int(np.argmax(pq[0])):
            play_match += 1
        play_total += 1
        if np.sign(vf[0]) == np.sign(vq[0]):
            value_sign_match += 1
    n = len(states)
    return {
        "n": n,
        "bid_argmax_agree": bid_match / n if n else 0.0,
        "play_argmax_agree": play_match / play_total if play_total else 0.0,
        "value_sign_agree": value_sign_match / n if n else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fp32", type=Path, required=True)
    p.add_argument("--int8", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    args = p.parse_args()

    states = load_calibration(args.calibration)
    if not states:
        sys.stderr.write(
            f"[validate_int8] {args.calibration} contains no states\n"
        )
        sys.exit(2)
    print(f"[validate_int8] running on {len(states)} states")
    r = run_models(args.fp32, args.int8, states)

    print(f"  bid argmax agreement   : {r['bid_argmax_agree']:.4f}")
    print(f"  play argmax agreement  : {r['play_argmax_agree']:.4f}")
    print(f"  value sign agreement   : {r['value_sign_agree']:.4f}")

    bid_ok = r["bid_argmax_agree"] >= POLICY_AGREEMENT_GATE
    val_ok = r["value_sign_agree"] >= VALUE_SIGN_AGREEMENT_GATE
    if bid_ok and val_ok:
        print("[validate_int8] PASS — INT8 model meets the 7.4b gates")
        sys.exit(0)
    if not bid_ok:
        sys.stderr.write(
            f"[validate_int8] FAIL: bid argmax {r['bid_argmax_agree']:.4f} "
            f"< {POLICY_AGREEMENT_GATE}\n"
        )
    if not val_ok:
        sys.stderr.write(
            f"[validate_int8] FAIL: value sign {r['value_sign_agree']:.4f} "
            f"< {VALUE_SIGN_AGREEMENT_GATE}\n"
        )
    sys.exit(1)


if __name__ == "__main__":
    main()
