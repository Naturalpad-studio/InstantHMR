#!/usr/bin/env python3
"""Export InstantHMR to ONNX: one network graph per checkpoint, and one shared MHR body graph.

network  (--ckpt): image (B, 3, S, S) ImageNet-normalised RGB crop, cliff_cond (B, 3)
                   -> mhr_params (B, 204), shape_params (B, 45), cam_trans (B, 3),
                      joints_2d (B, 70, 2) in [-1, 1] over the crop
body     (--body): mhr_params, shape_params -> joints_3d (B, 70, 3), vertices (B, 18439, 3),
                   metres, body-centred (add cam_trans for camera space)
fit      (--ckpt --fit 1|2): one person, the network on 1 crop (0.9 of the box) or 2 crops (0.9 and 0.7, batched),
                   then the MHR kinematic fit (training/kinematic_fit.py), in fp16:
                   image (N, S, S, 3) uint8 RGB crops, cliff_cond (N, 3), crop_box (N, 4) each crop's pixel square
                   (x1, y1, w, h) in the frame, cam (1, 3) = (focal, cx, cy) in pixels
                   -> mhr_params (1, 204), shape_params (1, 45), cam_trans (1, 3), joints_2d (1, 70, 2) in pixels

The network graph stores its crop size and conditioning form in the model metadata, the body
graph its mesh faces, so instanthmr.InstantHMR configures itself from the files.

    python tools/export_onnx.py --ckpt runs/medium/best_student_model_v3.pth --output models/instanthmr-medium.onnx
    python tools/export_onnx.py --body --output models/mhr-body.onnx
    python tools/export_onnx.py --ckpt runs/medium/best_student_model_v3.pth --fit 2 --output models/instanthmr-medium-fit2.onnx
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
from kinematic_fit import FitHead, MHRKinematics  # noqa: E402
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


class FitNetwork(torch.nn.Module):
    """Network + kinematic fit for one person, in ``dtype``; uint8 crops in, float32 out."""

    def __init__(self, net, kin, n_crops, dtype):
        super().__init__()
        self.net, self.dtype = net, dtype
        self.head = FitHead(kin, net.kp2d_bin_centers, n_crops)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("istd", 1.0 / (torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255.0))

    def forward(self, image, cliff_cond, crop_box, cam):
        x = (image.float().permute(0, 3, 1, 2) - self.mean) * self.istd     # cast first: TensorRT has no uint8 tensors
        out = self.net(x.to(self.dtype), cliff_cond.to(self.dtype))
        mp, t, fused, _, _ = self.head(out["mhr_params"], out["cam_trans"], out["joints_2d_logits"],
                                       crop_box.to(self.dtype), cam.to(self.dtype))
        joints_2d = crop_box[:1, None, :2] + 0.5 * crop_box[:1, None, 2:] + fused.float() * crop_box[:1, None, 2:3]
        return mp.float(), out["shape_params"][:1].float(), t.float(), joints_2d


def atan2_poly(y, x):
    """atan2 from basic operations (max error ~1e-5 rad): onnxruntime's CUDA provider has no Atan kernel, so the
    native operator would send the rotation decoding to the CPU and prevent CUDA-graph capture."""
    ax, ay = x.abs(), y.abs()
    a = torch.minimum(ax, ay) / torch.maximum(ax, ay).clamp_min(1e-12)
    s = a * a
    r = a * (0.9998660 + s * (-0.3302995 + s * (0.1801410 + s * (-0.0851330 + s * 0.0208351))))
    r = torch.where(ay > ax, 1.5707963267948966 - r, r)
    r = torch.where(x < 0, 3.141592653589793 - r, r)
    return torch.where(y < 0, -r, r)


def _attention_in_fp32(module: torch.nn.Module) -> None:
    """Run every attention block in float32 inside an fp16 graph. Exported attention computes the query-key scores
    as a plain matrix product, and in fp16 they overflow (65504) on some models and images: medium-commercial
    reaches 81 x 31 per dimension over 64 dimensions, which turns the softmax into NaN on onnxruntime's CUDA
    provider. PyTorch's fused kernel never stores those scores in fp16, so the PyTorch model does not show it."""
    for m in module.modules():
        if isinstance(m, torch.nn.MultiheadAttention):
            m.float()

            def forward(query, key, value, *args, _forward=m.forward, **kwargs):
                kwargs = {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in kwargs.items()}
                out, weights = _forward(query.float(), key.float(), value.float(), *args, **kwargs)
                return out.to(query.dtype), weights
            m.forward = forward


def export_fit(net, cfg, n_crops: int, path: Path, mhr_model: Path, fp32: bool) -> None:
    if not torch.cuda.is_available() and not fp32:
        sys.exit("[error] an fp16 export needs a CUDA GPU to trace the network; pass --fp32 on a CPU-only machine")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if fp32 else torch.float16
    kin = MHRKinematics.from_rig(mhr_model, REPO / "training" / "assets" / "mhr_j127_to_kp70.npy")
    graph = FitNetwork(net, kin, n_crops, dtype).eval().to(dev, dtype)
    graph.mean.data, graph.istd.data = graph.mean.float(), graph.istd.float()
    if not fp32:
        _attention_in_fp32(graph)
    size = int(cfg.image_size)
    scales = (0.9, 0.7)[:n_crops]
    side = torch.tensor([600.0 * 1.2 * s for s in scales])                # a person ~600 px tall in a 1280 x 720 frame
    box = torch.stack([640 - side / 2, 360 - side / 2, side, side], 1)
    inputs = {"image": torch.randint(0, 255, (n_crops, size, size, 3), dtype=torch.uint8),
              "cliff_cond": torch.tensor([[0.0, 0.0, 0.5]]).repeat(n_crops, 1), "crop_box": box,
              "cam": torch.tensor([[1100.0, 640.0, 360.0]])}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.atan2, atan2 = atan2_poly, torch.atan2                           # traced into the graph instead of Atan
    try:                                       # no torch.no_grad(): it selects a fused attention kernel ONNX lacks
        torch.onnx.export(graph, tuple(v.to(dev) for v in inputs.values()), str(path), input_names=list(inputs),
                          output_names=NETWORK_OUTPUTS, opset_version=17, dynamo=False)
    finally:
        torch.atan2 = atan2
    model = onnx.load(str(path))
    try:                                       # optional: folds the constants the fp16 trace leaves behind
        from onnxsim import simplify
        model, ok = simplify(model)
        assert ok, "onnxsim could not validate the simplified graph"
    except ImportError:
        pass
    for key, value in {"image_size": str(size), "cliff_focal": str(bool(cfg.cliff_focal)).lower(),
                       "crops": str(n_crops), "crop_scales": ",".join(map(str, scales)),
                       "precision": "fp32" if fp32 else "fp16",
                       "fit": "1 Levenberg-Marquardt step, 10 Newton-Schulz steps, prior 30 / 0.01"}.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(model, str(path))

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = dict(zip(NETWORK_OUTPUTS, session.run(None, {k: v.numpy() for k, v in inputs.items()})))
    ref = FitNetwork(net, kin, n_crops, torch.float32).float().eval().cpu()
    with torch.no_grad():
        want = dict(zip(NETWORK_OUTPUTS, (o.numpy() for o in ref(*inputs.values()))))
    if not all(np.isfinite(v).all() for v in got.values()):
        sys.exit("[error] the ONNX graph returns non-finite values")
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {'fp32' if fp32 else 'fp16'}); onnxruntime vs PyTorch "
          f"fp32: pose {np.abs(got['mhr_params'][:, 3:68] - want['mhr_params'][:, 3:68]).max():.1e} rad, "
          f"joints_2d {np.abs(got['joints_2d'] - want['joints_2d']).max():.2f} px")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--ckpt", type=Path, help="checkpoint written by training/train.py or train_ddp.py")
    what.add_argument("--body", action="store_true", help="export the MHR body graph instead")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fit", type=int, choices=(1, 2), help="with --ckpt: export the network + kinematic fit graph, "
                    "on 1 crop or 2 crops per person")
    ap.add_argument("--fp32", action="store_true", help="with --fit: float32 instead of float16")
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
    if args.fit:
        export_fit(net, cfg, args.fit, args.output, args.mhr_model, args.fp32)
        return
    size = int(cfg.image_size)
    inputs = {"image": torch.randn(2, 3, size, size), "cliff_cond": torch.randn(2, 3) * 0.3}
    export(Network(net).eval(), inputs, NETWORK_OUTPUTS, args.output,
           {"image_size": str(size), "cliff_focal": str(bool(cfg.cliff_focal)).lower()})


if __name__ == "__main__":
    main()
