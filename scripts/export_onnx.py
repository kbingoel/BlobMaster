"""Export a trained BlobNet to ONNX.

Session 3.5 workflow:
    1. In Rust, call `blob_nn::train::save_checkpoint(&vs, iter, dir)` — this
       writes `{dir}/model.ot`, a libtorch-compatible tensor archive.
    2. Run this script:
           python scripts/export_onnx.py \
               --weights path/to/model.ot \
               --out path/to/model.onnx
    3. Load the resulting `.onnx` from `blob_engine::OnnxEvaluator::from_file`.

The PyTorch model definition below mirrors `blob-nn/src/{input,transformer,
heads,model}.rs` parameter-for-parameter. Any change to the Rust network must
be reflected here, otherwise `VarStore.load_from_*` will fail at weight
renaming or the ONNX outputs will disagree with the tch forward pass.

Parity check (after export, before shipping):
    python scripts/export_onnx.py --weights model.ot --out model.onnx --check
will run 100 random inputs through both the PyTorch model and the exported
ONNX graph and report max absolute difference; target < 1e-5.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Architecture constants (keep in sync with blob-nn/src/*.rs) -----------

D_MODEL = 128
N_HEADS = 8
HEAD_DIM = D_MODEL // N_HEADS
FFN_DIM = 512
N_LAYERS = 8
DROPOUT = 0.1
LN_EPS = 1e-5

HAND_DIM = 30
PLAYED_DIM = 48
PLAYER_DIM = 29
CONTEXT_DIM = 13
FEAT_DIM = PLAYED_DIM  # right-padded feature width
MAX_CHRONO = 52

NUM_BIDS = 14
PLAY_MLP_HIDDEN = 32
HEAD_HIDDEN = 64

TT_CLS, TT_CONTEXT, TT_PLAYER, TT_HAND, TT_PLAYED = 0, 1, 2, 3, 4

# ---- Model -----------------------------------------------------------------


class InputProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hand_proj = nn.Linear(HAND_DIM, D_MODEL)
        self.played_proj = nn.Linear(PLAYED_DIM, D_MODEL)
        self.player_proj = nn.Linear(PLAYER_DIM, D_MODEL)
        self.context_proj = nn.Linear(CONTEXT_DIM, D_MODEL)
        self.cls = nn.Parameter(torch.randn(D_MODEL) * 0.02)
        self.chrono_embed = nn.Embedding(MAX_CHRONO, D_MODEL)

    def forward(self, features, token_types, chrono_indices, attention_mask):
        hand_out = self.hand_proj(features[..., :HAND_DIM])
        played_out = self.played_proj(features[..., :PLAYED_DIM])
        player_out = self.player_proj(features[..., :PLAYER_DIM])
        context_out = self.context_proj(features[..., :CONTEXT_DIM])
        b, s = token_types.shape
        cls_out = self.cls.view(1, 1, D_MODEL).expand(b, s, D_MODEL)

        def m(v: int) -> torch.Tensor:
            return (token_types == v).to(features.dtype).unsqueeze(-1)

        out = (
            cls_out * m(TT_CLS)
            + context_out * m(TT_CONTEXT)
            + player_out * m(TT_PLAYER)
            + hand_out * m(TT_HAND)
            + played_out * m(TT_PLAYED)
        )
        out = out + self.chrono_embed(chrono_indices) * m(TT_PLAYED)
        return out * attention_mask.to(features.dtype).unsqueeze(-1)


class MHSA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x, attention_mask):
        b, s, _ = x.shape
        qkv = self.qkv(x).view(b, s, 3, N_HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = q @ k.transpose(-2, -1) / math.sqrt(HEAD_DIM)
        key_pad = (~attention_mask).view(b, 1, 1, s)
        scores = scores.masked_fill(key_pad, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        ctx = (attn @ v).transpose(1, 2).contiguous().view(b, s, D_MODEL)
        return self.out(ctx)


class FFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, FFN_DIM)
        self.fc2 = nn.Linear(FFN_DIM, D_MODEL)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.attn = MHSA()
        self.ln2 = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.ffn = FFN()

    def forward(self, x, attention_mask):
        h = x + self.attn(self.ln1(x), attention_mask)
        return h + self.ffn(self.ln2(h))


class TransformerEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Block() for _ in range(N_LAYERS)])

    def forward(self, x, attention_mask):
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x * attention_mask.to(x.dtype).unsqueeze(-1)


class PlayingHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, PLAY_MLP_HIDDEN)
        self.fc2 = nn.Linear(PLAY_MLP_HIDDEN, 1)

    def scores(self, h):
        return self.fc2(F.gelu(self.fc1(h))).squeeze(-1)


class BiddingHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, HEAD_HIDDEN)
        self.fc2 = nn.Linear(HEAD_HIDDEN, NUM_BIDS)

    def logits(self, h):
        cls = h[:, 0, :]
        return self.fc2(F.gelu(self.fc1(cls)))


class ValueHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, HEAD_HIDDEN)
        self.fc2 = nn.Linear(HEAD_HIDDEN, 1)

    def forward(self, h):
        cls = h[:, 0, :]
        return torch.tanh(self.fc2(F.gelu(self.fc1(cls)))).squeeze(-1)


class BlobNet(nn.Module):
    """ONNX-exportable wrapper. Returns (bid_policy, play_scores, value).

    - bid_policy: softmax over 14 bids with no legality mask baked in; the
      Rust `OnnxEvaluator` re-applies the per-state legal-bid mask.
    - play_scores: raw per-position scores [B, S]; masked + softmaxed by
      `OnnxEvaluator` using `legal_plays(state)` and the encoder's
      `hand_card_indices`.
    - value: scalar in [-1, 1].
    """

    def __init__(self) -> None:
        super().__init__()
        self.input = InputProjection()
        self.transformer = TransformerEncoder()
        self.play_head = PlayingHead()
        self.bid_head = BiddingHead()
        self.value_head = ValueHead()

    def forward(self, features, token_types, chrono_indices, attention_mask):
        x = self.input(features, token_types, chrono_indices, attention_mask)
        h = self.transformer(x, attention_mask)
        play_scores = self.play_head.scores(h)
        bid_logits = self.bid_head.logits(h)
        bid_policy = torch.softmax(bid_logits, dim=-1)
        value = self.value_head(h)
        return bid_policy, play_scores, value


# ---- Weight loading --------------------------------------------------------

# Mapping from VarStore path → PyTorch parameter path. Both sides follow
# the same sub-module layout because of how BlobNet::new(vs) builds the
# graph, so the mapping is mostly identity. Any divergence goes here.


def _rust_to_torch_key(rust_key: str) -> str:
    # VarStore uses `/` as a path separator; PyTorch uses `.`.
    return rust_key.replace("/", ".")


def load_varstore_into(model: BlobNet, weights_path: Path) -> None:
    """Load a tch VarStore archive into the PyTorch BlobNet.

    `VarStore::save` writes a dict of named tensors in libtorch's zip
    serialization format. `torch.load` with `weights_only=True` handles it
    in recent PyTorch versions; older versions need `weights_only=False`.
    """
    try:
        raw = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(weights_path, map_location="cpu")

    remapped = {}
    for k, v in raw.items():
        remapped[_rust_to_torch_key(k)] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        sys.stderr.write(
            f"[export_onnx] warning: missing={missing} unexpected={unexpected}\n"
        )


# ---- Export ---------------------------------------------------------------


def make_dummy_inputs(batch: int = 1, seq: int = 29) -> tuple[torch.Tensor, ...]:
    features = torch.randn(batch, seq, FEAT_DIM)
    token_types = torch.zeros(batch, seq, dtype=torch.long)
    chrono = torch.zeros(batch, seq, dtype=torch.long)
    mask = torch.ones(batch, seq, dtype=torch.bool)
    return features, token_types, chrono, mask


def export(model: BlobNet, out_path: Path) -> None:
    model.eval()
    features, token_types, chrono, mask = make_dummy_inputs()
    torch.onnx.export(
        model,
        (features, token_types, chrono, mask),
        str(out_path),
        input_names=["features", "token_types", "chrono_indices", "attention_mask"],
        output_names=["bid_policy", "play_scores", "value"],
        dynamic_axes={
            "features": {0: "batch", 1: "seq"},
            "token_types": {0: "batch", 1: "seq"},
            "chrono_indices": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "bid_policy": {0: "batch"},
            "play_scores": {0: "batch", 1: "seq"},
            "value": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )


def parity_check(model: BlobNet, onnx_path: Path, n_trials: int = 100) -> None:
    import onnxruntime as ort  # type: ignore

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model.eval()
    max_diff = 0.0
    for _ in range(n_trials):
        seq = int(torch.randint(5, 50, (1,)).item())
        features, token_types, chrono, mask = make_dummy_inputs(batch=1, seq=seq)
        with torch.no_grad():
            tb, tp, tv = model(features, token_types, chrono, mask)
        ob, op, ov = sess.run(
            None,
            {
                "features": features.numpy(),
                "token_types": token_types.numpy(),
                "chrono_indices": chrono.numpy(),
                "attention_mask": mask.numpy(),
            },
        )
        for a, b in ((tb.numpy(), ob), (tp.numpy(), op), (tv.numpy(), ov)):
            d = float((a - b).__abs__().max())
            max_diff = max(max_diff, d)
    print(f"[parity] max abs diff over {n_trials} trials: {max_diff:.3e}")
    if max_diff > 1e-5:
        sys.stderr.write(
            f"[parity] WARNING: exceeds 1e-5 tolerance ({max_diff:.3e})\n"
        )
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, help="VarStore .ot file")
    p.add_argument("--out", type=Path, required=True, help="output .onnx path")
    p.add_argument("--check", action="store_true", help="run parity check after export")
    args = p.parse_args()

    torch.manual_seed(0)
    model = BlobNet()
    if args.weights is not None:
        load_varstore_into(model, args.weights)
    export(model, args.out)
    print(f"[export_onnx] wrote {args.out}")
    if args.check:
        parity_check(model, args.out)


if __name__ == "__main__":
    main()
