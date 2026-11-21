#!/usr/bin/env python3
"""Convert an InstantHMR checkpoint to a CoreML .mlpackage (mlprogram, iOS 17+).

Verified on Linux (coremltools 9.0 + torch 2.7.1) against pytorch/medium-commercial;
the result runs only on Apple hardware, so speed and numerics must be checked on a Mac.

    python tools/coreml/convert_to_coreml.py --ckpt pytorch/medium-commercial/instanthmr-medium-commercial.pth \
        --out build/InstantHMRMediumCommercial.mlpackage --precision fp16

Why this script is not a one-liner (both blockers are real, measured on 2026-11-21):
  1. coremltools >= 7 has no ONNX frontend any more, so the exported .onnx cannot be
     converted: the route has to go through the PyTorch checkpoint (this script).
  2. torch.jit.trace turns nn.MultiheadAttention into `_native_multi_head_attention`, which
     coremltools does not implement, and after disabling torch's MHA fast path the traced
     graph still contains `aten::Int` nodes from shape arithmetic inside attention, which
     coremltools fails to lower ("only 0-dimensional arrays can be converted to Python
     scalars"). `PlainAttn` below re-implements the same attention math with only
     matmul/softmax/unflatten/flatten, so no shape integers reach the graph.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "training"))

import torch
torch.backends.mha.set_fastpath_enabled(False)          # keep the unfused MHA path when tracing
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import coremltools as ct

sys.path.insert(0, str(REPO / "tools"))
from export_onnx import Network, load_network  # noqa: E402

NETWORK_OUTPUTS = ["mhr_params", "shape_params", "cam_trans", "joints_2d"]
IMAGE_SIZE = 288


class PlainAttn(nn.Module):
    """`nn.MultiheadAttention` math with no shape arithmetic (unflatten/flatten/matmul only)."""

    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        self.num_heads = int(mha.num_heads)
        self.embed_dim = int(mha.embed_dim)
        self.head_dim = self.embed_dim // self.num_heads
        self.batch_first = bool(mha.batch_first)
        self.in_proj_weight = nn.Parameter(mha.in_proj_weight.detach().clone())
        self.in_proj_bias = nn.Parameter(mha.in_proj_bias.detach().clone()) if mha.in_proj_bias is not None else None
        self.out_proj = nn.Linear(int(mha.out_proj.in_features), int(mha.out_proj.out_features))
        with torch.no_grad():
            self.out_proj.weight.copy_(mha.out_proj.weight)
            if mha.out_proj.bias is not None:
                self.out_proj.bias.copy_(mha.out_proj.bias)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False, **kw):
        if not self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        qw, kw, vw = self.in_proj_weight.chunk(3, dim=0)
        qb, kb, vb = (None, None, None) if self.in_proj_bias is None else self.in_proj_bias.chunk(3, dim=0)
        H, hd = self.num_heads, self.head_dim
        q = F.linear(query, qw, qb).unflatten(-1, (H, hd)).permute(0, 2, 1, 3)      # B,H,L,hd
        k = F.linear(key, kw, kb).unflatten(-1, (H, hd)).permute(0, 2, 3, 1)        # B,H,hd,S
        v = F.linear(value, vw, vb).unflatten(-1, (H, hd)).permute(0, 2, 1, 3)       # B,H,S,hd
        s = torch.matmul(q, k) * (1.0 / math.sqrt(hd))
        if key_padding_mask is not None:
            s = s.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(1), -1e9)
        w = torch.softmax(s, dim=-1)
        o = torch.matmul(w, v).permute(0, 2, 1, 3).flatten(2, 3)
        o = self.out_proj(o)
        return (o if self.batch_first else o.transpose(0, 1)), None


def replace_mha(module: nn.Module) -> int:
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, PlainAttn(child))
            n += 1
        else:
            n += replace_mha(child)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    args = ap.parse_args()

    net, cfg = load_network(args.ckpt)
    print(f"replaced {replace_mha(net)} attention blocks")
    module = Network(net).eval()
    ex = (torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE), torch.zeros(1, 3))
    with torch.no_grad():
        eager = module(*ex)
        traced = torch.jit.trace(module, ex)
        traced_out = traced(*ex)
    diff = max(float((a - b).abs().max()) for a, b in zip(eager, traced_out))
    n_int = sum(1 for n in traced.inlined_graph.nodes() if n.kind() == "aten::Int")
    print(f"trace diff {diff:.2e}, aten::Int nodes {n_int} (must be 0)")

    ml = ct.convert(
        traced, source="pytorch",
        inputs=[ct.TensorType(name="image", shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32),
                ct.TensorType(name="cliff_cond", shape=(1, 3), dtype=np.float32)],
        outputs=[ct.TensorType(name=n) for n in NETWORK_OUTPUTS],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ml.save(str(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
