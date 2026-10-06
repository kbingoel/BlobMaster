"""Export a gen-2 model directory: the policy net P and the value net V as ONNX.

Run it through `blobmaster-train export`, which picks the repo's venv and
strips LD_PRELOAD:

    blobmaster-train export --output <dir> [--checkpoint <checkpoint dir>] [--check]

or directly:

    python scripts/export_onnx.py --out-dir <dir> [--checkpoint <checkpoint dir>] [--check]

It writes `<dir>/policy.onnx`, `<dir>/value.onnx` and `<dir>/meta.json`
(gen-2.md §5.3). Both ONNX files carry the encoder layout id and the network
name in their metadata (`blob_layout_id`, `blob_network`); the Rust
`OnnxPolicy` / `OnnxValue` refuse a file with another layout (gen-2.md §5.5
item 10).

- `PolicyNet` and `ValueNet` mirror `blob-nn/src/{input,transformer,heads,
  model}.rs` parameter for parameter, so a learner checkpoint (`policy.ot`,
  `value.ot`: tch VarStores, plus `meta.json` with the learner step) loads
  strictly. Any change to the Rust networks must be made here too, or the
  weights won't load or the outputs will disagree with tch.
- Without `--checkpoint`, both nets are random-init (seeded): a model for
  exercising `bench` and `play`.

`--check` runs random inputs through PyTorch and the exported graphs and
reports the max absolute difference per network against `PARITY_GATE`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Architecture constants (keep in sync with blob-nn/src/*.rs) -----------

D_MODEL = 128
N_HEADS = 8
HEAD_DIM = D_MODEL // N_HEADS
FFN_DIM = 512
P_LAYERS = 8
V_LAYERS = 4
DROPOUT = 0.1
LN_EPS = 1e-5

# The encoder layout (`blob-engine/src/encoder.rs`). The Rust test
# `export_script_mirrors_feature_widths` checks these lines.
LAYOUT_ID = "layout-3"
HAND_DIM = 32
PLAYED_DIM = 49
PLAYER_DIM = 28
CONTEXT_DIM = 16
OPP_HAND_DIM = 41
FEAT_DIM = 49  # right-padded feature width: the widest token type
assert FEAT_DIM == max(HAND_DIM, PLAYED_DIM, PLAYER_DIM, CONTEXT_DIM, OPP_HAND_DIM)
MAX_CHRONO = 52

NUM_BIDS = 14
PLAY_MLP_HIDDEN = 32
HEAD_HIDDEN = 64

TT_CLS, TT_CONTEXT, TT_PLAYER, TT_HAND, TT_PLAYED = 0, 1, 2, 3, 4
TT_OPP_HAND = 5

# `--check` passes when |PyTorch − ONNX Runtime| <= PARITY_GATE · max(1, |PyTorch|)
# on every output: absolute below 1, relative above (P's raw play scores).
PARITY_GATE = 1e-5

INPUT_NAMES = ["features", "token_types", "chrono_indices", "attention_mask"]
SEQ_AXES = {name: {0: "batch", 1: "seq"} for name in INPUT_NAMES}

# ---- Model -----------------------------------------------------------------


class InputProjection(nn.Module):
    """One projection per token type, plus CLS and the chronological embedding
    of played cards. `opponent_cards` adds V's projection for opponents'
    hand cards."""

    def __init__(self, opponent_cards: bool) -> None:
        super().__init__()
        self.hand_proj = nn.Linear(HAND_DIM, D_MODEL)
        self.played_proj = nn.Linear(PLAYED_DIM, D_MODEL)
        self.player_proj = nn.Linear(PLAYER_DIM, D_MODEL)
        self.context_proj = nn.Linear(CONTEXT_DIM, D_MODEL)
        self.opp_hand_proj = nn.Linear(OPP_HAND_DIM, D_MODEL) if opponent_cards else None
        self.cls = nn.Parameter(torch.randn(D_MODEL) * 0.02)
        self.chrono_embed = nn.Embedding(MAX_CHRONO, D_MODEL)

    def forward(self, features, token_types, chrono_indices, attention_mask):
        b, s = token_types.shape

        def m(v: int) -> torch.Tensor:
            return (token_types == v).to(features.dtype).unsqueeze(-1)

        out = (
            self.cls.view(1, 1, D_MODEL).expand(b, s, D_MODEL) * m(TT_CLS)
            + self.context_proj(features[..., :CONTEXT_DIM]) * m(TT_CONTEXT)
            + self.player_proj(features[..., :PLAYER_DIM]) * m(TT_PLAYER)
            + self.hand_proj(features[..., :HAND_DIM]) * m(TT_HAND)
            + self.played_proj(features[..., :PLAYED_DIM]) * m(TT_PLAYED)
        )
        if self.opp_hand_proj is not None:
            out = out + self.opp_hand_proj(features[..., :OPP_HAND_DIM]) * m(TT_OPP_HAND)
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
    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Block() for _ in range(n_layers)])

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


class SeatValueHead(nn.Module):
    """Expected ŝ ∈ [0, 1] at every token; read at the player tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, HEAD_HIDDEN)
        self.fc2 = nn.Linear(HEAD_HIDDEN, 1)

    def forward(self, h):
        return torch.sigmoid(self.fc2(F.gelu(self.fc1(h)))).squeeze(-1)


class PolicyNet(nn.Module):
    """P: the acting seat's own view → (bid_policy, play_scores).

    - bid_policy: softmax over all 14 bids; `OnnxPolicy` re-applies the
      state's legal-bid mask.
    - play_scores: raw per-token scores [B, S]; `OnnxPolicy` masks and
      softmaxes them over the legal hand cards.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input = InputProjection(opponent_cards=False)
        self.transformer = TransformerEncoder(P_LAYERS)
        self.play_head = PlayingHead()
        self.bid_head = BiddingHead()

    def forward(self, features, token_types, chrono_indices, attention_mask):
        x = self.input(features, token_types, chrono_indices, attention_mask)
        h = self.transformer(x, attention_mask)
        bid_policy = torch.softmax(self.bid_head.logits(h), dim=-1)
        return bid_policy, self.play_head.scores(h)


class ValueNet(nn.Module):
    """V: the whole deal → seat_values [B, S], the expected ŝ of each seat at
    its player token (relative-seat order, the seat to move first)."""

    def __init__(self) -> None:
        super().__init__()
        self.input = InputProjection(opponent_cards=True)
        self.transformer = TransformerEncoder(V_LAYERS)
        self.value_head = SeatValueHead()

    def forward(self, features, token_types, chrono_indices, attention_mask):
        x = self.input(features, token_types, chrono_indices, attention_mask)
        return self.value_head(self.transformer(x, attention_mask))


NETWORKS = {
    # name: (file, output names, last token type the network reads)
    "policy": ("policy.onnx", ["bid_policy", "play_scores"], TT_PLAYED),
    "value": ("value.onnx", ["seat_values"], TT_OPP_HAND),
}

# ---- Weight loading --------------------------------------------------------


def _rust_to_torch_key(rust_key: str) -> str:
    # tch's VarStore archive names parameters like
    # `transformer|layer0|attn|qkv|weight`; PyTorch uses `.` separators and
    # nn.ModuleList indices (`transformer.layers.0.attn.qkv.weight`).
    k = rust_key.replace("|", ".").replace("/", ".")
    return re.sub(r"\.layer(\d+)\.", r".layers.\1.", k)


def load_varstore_into(model: nn.Module, weights_path: Path) -> None:
    """Load a tch VarStore archive into `model`. A missing or unexpected
    parameter is an error: the two definitions have drifted apart."""
    # `VarStore::save` writes a TorchScript archive (zip of named tensors).
    try:
        module = torch.jit.load(str(weights_path), map_location="cpu")
        raw = {name: p.detach() for name, p in module.named_parameters(recurse=True)}
        for name, buf in module.named_buffers(recurse=True):
            raw.setdefault(name, buf.detach())
    except Exception:
        raw = torch.load(weights_path, map_location="cpu", weights_only=True)

    remapped = {_rust_to_torch_key(k): v for k, v in raw.items()}
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        sys.exit(f"[export_onnx] {weights_path}: missing={missing} unexpected={unexpected}")


# ---- Export ---------------------------------------------------------------


def random_inputs(batch: int, seq: int, max_token_type: int) -> tuple[torch.Tensor, ...]:
    # Uniform in [0, 1], the encoder's range: every feature is a flag, a
    # one-hot entry or a count scaled to [0, 1], except the over/under-bid
    # ratio, which stays within [-1, 12].
    features = torch.rand(batch, seq, FEAT_DIM)
    token_types = torch.randint(0, max_token_type + 1, (batch, seq))
    token_types[:, 0] = TT_CLS
    chrono = torch.randint(0, MAX_CHRONO, (batch, seq))
    mask = torch.ones(batch, seq, dtype=torch.bool)
    return features, token_types, chrono, mask


def export(model: nn.Module, network: str, out_dir: Path) -> Path:
    file, outputs, max_tt = NETWORKS[network]
    path = out_dir / file
    model.eval()
    axes = dict(SEQ_AXES)
    for name in outputs:
        axes[name] = {0: "batch"} if name == "bid_policy" else {0: "batch", 1: "seq"}
    torch.onnx.export(
        model,
        random_inputs(1, 29, max_tt),
        str(path),
        input_names=INPUT_NAMES,
        output_names=outputs,
        dynamic_axes=axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    graph = onnx.load(str(path))
    onnx.helper.set_model_props(graph, {"blob_layout_id": LAYOUT_ID, "blob_network": network})
    onnx.save(graph, str(path))
    return path


def parity_check(model: nn.Module, network: str, path: Path, n_trials: int = 100) -> float:
    """Worst |PyTorch − ONNX Runtime| / max(1, |PyTorch|) over random inputs."""
    import onnxruntime as ort  # type: ignore

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    _, _, max_tt = NETWORKS[network]
    model.eval()
    max_abs, max_scaled = 0.0, 0.0
    for _ in range(n_trials):
        seq = int(torch.randint(5, 50, (1,)).item())
        inputs = random_inputs(2, seq, max_tt)
        with torch.no_grad():
            want = model(*inputs)
        want = want if isinstance(want, tuple) else (want,)
        got = sess.run(None, {name: t.numpy() for name, t in zip(INPUT_NAMES, inputs)})
        for a, b in zip(want, got):
            diff = abs(a.numpy() - b)
            max_abs = max(max_abs, float(diff.max()))
            max_scaled = max(max_scaled, float((diff / abs(a.numpy()).clip(min=1.0)).max()))
    print(f"[parity] {network}: over {n_trials} trials, max abs diff {max_abs:.3e}, scaled {max_scaled:.3e}")
    return max_scaled


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, required=True, help="model directory to write")
    p.add_argument("--checkpoint", type=Path, help="learner checkpoint directory (default: random init)")
    p.add_argument("--check", action="store_true", help="run a parity check after export")
    args = p.parse_args()

    torch.manual_seed(0)
    policy = PolicyNet()
    torch.manual_seed(1)
    value = ValueNet()
    learner_step = None
    if args.checkpoint is not None:
        load_varstore_into(policy, args.checkpoint / "policy.ot")
        load_varstore_into(value, args.checkpoint / "value.ot")
        learner_step = json.loads((args.checkpoint / "meta.json").read_text())["learner_step"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"policy": export(policy, "policy", args.out_dir), "value": export(value, "value", args.out_dir)}
    meta = {
        "layout_id": LAYOUT_ID,
        "learner_step": learner_step,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "policy": {"layers": P_LAYERS},
        "value": {"layers": V_LAYERS},
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[export_onnx] wrote {args.out_dir}: {', '.join(str(p.name) for p in paths.values())}, meta.json")

    if args.check:
        worst = max(parity_check(policy, "policy", paths["policy"]), parity_check(value, "value", paths["value"]))
        if worst > PARITY_GATE:
            sys.exit(f"[parity] exceeds the {PARITY_GATE:.0e} tolerance ({worst:.3e})")


if __name__ == "__main__":
    main()
