#!/usr/bin/env python3
"""Export InstantHMR to ONNX: one network graph per checkpoint, and one shared MHR body graph.

network  (--ckpt): image (B, 3, S, S) ImageNet-normalised RGB crop, cliff_cond (B, 3)
                   -> mhr_params (B, 204), shape_params (B, 45), cam_trans (B, 3),
                      joints_2d (B, 70, 2) in [-1, 1] over the crop
body     (--body): mhr_params, shape_params -> joints_3d (B, 70, 3), vertices (B, 18439, 3),
                   metres, body-centred (add cam_trans for camera space)

The network graph stores its crop size and conditioning form in the model metadata, the body
graph its mesh faces, so instanthmr.InstantHMR configures itself from the files.

    python tools/export_onnx.py --ckpt runs/medium/best_student_model_v3.pth --output models/instanthmr-medium.onnx
    python tools/export_onnx.py --body --output models/mhr-body.onnx
"""
import argparse
import base64
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))

import train as T  # noqa: E402
from mhr_body import MHRBody  # noqa: E402

NETWORK_OUTPUTS = ["mhr_params", "shape_params", "cam_trans", "joints_2d"]
BODY_OUTPUTS = ["joints_3d", "vertices"]


class Network(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image, cliff_cond):
        out = self.net(image, cliff_cond)
        return tuple(out[k] for k in NETWORK_OUTPUTS)


def load_network(ckpt: Path):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.get("model_state_dict", state).items()}
    if "head_2d_logits.weight" not in state or any(k.startswith("head_3d.") for k in state):
        sys.exit(f"[error] {ckpt} was not written by training/train.py")
    cfg, _ = T.config_from_checkpoint(state, ckpt)
    cfg.kp2d_bins = state["head_2d_logits.bias"].shape[0] // 2
    cfg.kp2d_range = float(state["kp2d_bin_centers"][-1])
    net = T.InstantHMRStudent(cfg, pretrained=False).eval()
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        sys.exit(f"[error] checkpoint and model disagree: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return net, cfg


def export(module, inputs: dict, outputs: list, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(module, tuple(inputs.values()), str(path), input_names=list(inputs), output_names=outputs,
                      dynamic_axes={n: {0: "batch"} for n in [*inputs, *outputs]}, opset_version=17, dynamo=False)
    model = onnx.load(str(path))
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(model, str(path))

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = session.run(None, {k: v.numpy() for k, v in inputs.items()})
    with torch.no_grad():
        want = module(*inputs.values())
    worst = {n: float(np.abs(g - w.numpy()).max()) for n, g, w in zip(outputs, got, want)}
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB); onnxruntime vs PyTorch, max abs diff: "
          + ", ".join(f"{n} {v:.1e}" for n, v in worst.items()))
    if max(worst.values()) > 1e-3:
        sys.exit("[error] the ONNX graph disagrees with the PyTorch model")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--ckpt", type=Path, help="checkpoint written by training/train.py or train_ddp.py")
    what.add_argument("--body", action="store_true", help="export the MHR body graph instead")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mhr-model", type=Path, default=REPO / "training" / "checkpoints" / "mhr_model.pt",
                    help="MHR TorchScript rig the body is read from")
    args = ap.parse_args()

    if args.body:
        body = MHRBody.from_rig(args.mhr_model).eval()
        faces = torch.jit.load(str(args.mhr_model), map_location="cpu").character_torch.mesh.faces.numpy()
        inputs = {"mhr_params": torch.randn(2, 204) * 0.3, "shape_params": torch.randn(2, 45)}
        export(body, inputs, BODY_OUTPUTS, args.output,
               {"mesh_faces": base64.b64encode(faces.astype("<i4").tobytes()).decode()})
        return

    net, cfg = load_network(args.ckpt)
    size = int(cfg.image_size)
    inputs = {"image": torch.randn(2, 3, size, size), "cliff_cond": torch.randn(2, 3) * 0.3}
    export(Network(net).eval(), inputs, NETWORK_OUTPUTS, args.output,
           {"image_size": str(size), "cliff_focal": str(bool(cfg.cliff_focal)).lower()})


if __name__ == "__main__":
    main()
