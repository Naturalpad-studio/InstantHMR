#!/usr/bin/env python3
"""Score an InstantHMR model on EMDB-1 or 3DPW test under the paper's protocol.

Protocol: every 10th frame; ground-truth person boxes (EMDB's own boxes; for 3DPW, the extent of
the 2D keypoints with confidence > 0.1, padded by 20%); the calibrated focal length; the MHR mesh
(with Meta's pose correctives) converted to SMPL by Meta's official fitting
(facebookresearch/MHR, tools/mhr_smpl_conversion); EMDB-1 scored on the 24 SMPL joints, 3DPW on
the 14 joints regressed from the mesh with the H36M regressor. Both joint sets are printed.

    python benchmark/evaluate.py --dataset emdb --data <EMDB root> --model medium \\
        --smpl-dir <SMPL files> --mhr-repo <MHR clone> --mhr-model training/checkpoints/mhr_model.pt
    python benchmark/evaluate.py --dataset 3dpw --data <3DPW root> ...

See benchmark/README.md for the downloads.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from benchlib import joints as J  # noqa: E402
from benchlib import metrics as M  # noqa: E402
from instanthmr import InstantHMR, resolve_model  # noqa: E402
from make_3dpw_gt import _install_chumpy_stub, _to_numpy, load_smpl, smpl_forward  # noqa: E402

MM = 1000.0


class _FacesOnlyRig:
    """What Meta's MHR -> SMPL conversion reads from an MHR model: its faces."""

    def __init__(self, faces):
        self.character = types.SimpleNamespace(mesh=types.SimpleNamespace(faces=faces))

    def to(self, _device):
        return self


class _AnyName(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})


def emdb_frames(root: Path, stride: int) -> list[dict]:
    """EMDB-1: frames with `good_frames_mask` set and not in `bboxes.invalid_idxs`, EMDB's boxes."""
    out = []
    for pkl in sorted(root.glob("P*/*/*_data.pkl")):
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        if not d["emdb1"]:
            continue
        good, bad = np.asarray(d["good_frames_mask"], bool), set(np.ravel(d["bboxes"]["invalid_idxs"]).tolist())
        s = d["smpl"]
        for f in range(0, int(d["n_frames"]), stride):
            if good[f] and f not in bad:
                out.append(dict(image=pkl.parent / "images" / f"{f:05d}.jpg", bbox=np.float32(d["bboxes"]["bboxes"][f]),
                                K=np.asarray(d["camera"]["intrinsics"]), seq=str(d["name"]),
                                gender=str(d["gender"]), betas=np.ravel(s["betas"])[:10],
                                pose=np.concatenate([s["poses_root"][f], s["poses_body"][f]]), trans=s["trans"][f],
                                extrinsics=np.asarray(d["camera"]["extrinsics"][f])))
    return out


def threedpw_frames(root: Path, stride: int) -> list[dict]:
    """3DPW test: `campose_valid` people with >= 4 confident 2D keypoints, box = their extent + 20%."""
    out = []
    for pkl in sorted((root / "sequenceFiles" / "test").glob("*.pkl")):
        with open(pkl, "rb") as f:
            seq = pickle.load(f, encoding="latin1")
        for f in range(0, len(seq["poses"][0]), stride):
            for p in range(len(seq["poses"])):
                kp = np.asarray(seq["poses2d"][p][f])
                ok = kp[2] > 0.1
                if not seq["campose_valid"][p][f] or ok.sum() < 4:
                    continue
                (x1, y1), (x2, y2) = kp[:2, ok].min(1), kp[:2, ok].max(1)
                w, h = x2 - x1, y2 - y1
                out.append(dict(image=root / "imageFiles" / seq["sequence"] / f"image_{f:05d}.jpg",
                                bbox=np.float32([x1 - 0.2 * w, y1 - 0.2 * h, x2 + 0.2 * w, y2 + 0.2 * h]),
                                K=np.asarray(seq["cam_intrinsics"]), seq=seq["sequence"],
                                gender=str(seq["genders"][p]), betas=np.ravel(seq["betas"][p])[:10],
                                pose=seq["poses"][p][f], trans=seq["trans"][p][f], extrinsics=seq["cam_poses"][f]))
    return out


def ground_truth(samples: list[dict], smpl_dir: Path, device) -> tuple[np.ndarray, np.ndarray]:
    """(N, 24, 3) SMPL joints and (N, 6890, 3) vertices, camera space, metres."""
    models, joints, verts = {}, [], []
    for s in samples:
        g = "MALE" if s["gender"].lower().startswith("m") else "FEMALE"
        if g not in models:
            models[g] = load_smpl(smpl_dir / f"SMPL_{g}.pkl", device)
        m = models[g]
        betas = torch.zeros(m["shapedirs"].shape[-1], device=device)
        betas[:10] = torch.as_tensor(np.float32(s["betas"]), device=device)
        v, j = smpl_forward(m, torch.as_tensor(np.float32(s["pose"])[None], device=device), betas,
                            torch.as_tensor(np.float32(s["trans"])[None], device=device))
        E = torch.as_tensor(np.float32(s["extrinsics"]), device=device)
        verts.append((v[0] @ E[:3, :3].T + E[:3, 3]).cpu().numpy())
        joints.append((j[0] @ E[:3, :3].T + E[:3, 3]).cpu().numpy())
    return np.stack(joints), np.stack(verts)


def predict(samples: list[dict], model: Path, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hmr = InstantHMR(model, device=device, max_batch=1)
    params, shapes, trans = [], [], []
    for s in tqdm(samples, desc="InstantHMR", leave=False):
        rgb = cv2.cvtColor(cv2.imread(str(s["image"])), cv2.COLOR_BGR2RGB)
        p = hmr.predict(rgb, [s["bbox"]], focal=float(s["K"][0, 0]))[0]
        params.append(p.mhr_params), shapes.append(p.shape_params), trans.append(p.cam_trans)
    return np.stack(params), np.stack(shapes), np.stack(trans)


def mhr_vertices(rig, params: np.ndarray, shapes: np.ndarray, device) -> np.ndarray:
    """The full rig's mesh (pose correctives included), body-centred, metres, camera axes."""
    out = []
    with torch.no_grad():
        for i in range(0, len(params), 256):
            p, s = (torch.as_tensor(a[i:i + 256], device=device) for a in (params, shapes))
            v, _ = rig(s, p, torch.zeros(len(p), 72, device=device), True)
            out.append((v * torch.tensor([0.01, -0.01, -0.01], device=device)).cpu().numpy())
    return np.concatenate(out)


def official_smpl_fit(verts: np.ndarray, cam_trans: np.ndarray, faces: np.ndarray, mhr_repo: Path, smpl_dir: Path,
                      device) -> tuple[np.ndarray, np.ndarray]:
    """Meta's MHR -> SMPL conversion, called unchanged: (N, 24, 3) joints and (N, 6890, 3) vertices.

    It only reads the rig's faces from the `mhr` package and imports `pymomentum.solver` without using
    it for this direction, so both are replaced by stand-ins and neither needs installing.
    """
    import smplx
    from smplx.utils import Struct

    _install_chumpy_stub()
    with open(smpl_dir / "SMPL_NEUTRAL.pkl", "rb") as f:
        neutral = pickle.load(f, encoding="latin1")
    neutral = Struct(**{k: _to_numpy(neutral[k]) for k in ("v_template", "shapedirs", "posedirs", "J_regressor",
                                                            "weights", "f", "kintree_table")})
    neutral.f, neutral.kintree_table = neutral.f.astype(np.int64), neutral.kintree_table.astype(np.int64)
    rig = _FacesOnlyRig(faces)
    mhr_module = types.ModuleType("mhr.mhr")
    mhr_module.MHR = _FacesOnlyRig
    sys.modules.update({"mhr": types.ModuleType("mhr"), "mhr.mhr": mhr_module,
                        "pymomentum": types.ModuleType("pymomentum"),
                        "pymomentum.solver": _AnyName("pymomentum.solver")})
    tool = mhr_repo / "tools" / "mhr_smpl_conversion"
    sys.path.insert(0, str(tool))
    cwd = os.getcwd()
    os.chdir(tool)
    try:
        from conversion import Conversion

        smpl = smplx.SMPL(model_path=str(smpl_dir), data_struct=neutral, gender="neutral").to(device)
        conversion = Conversion(mhr_model=rig, smpl_model=smpl, method="pytorch")
        joints, vertices = [], []
        for i in range(0, len(verts), 512):
            items = [dict(mhr_model_params=np.zeros(1, np.float32), shape_params=np.zeros(1, np.float32),
                          expr_params=np.zeros(1, np.float32), pred_vertices=verts[k], pred_cam_t=cam_trans[k])
                     for k in range(i, min(i + 512, len(verts)))]
            res = conversion.convert_sam3d_output_to_smpl(items, return_smpl_parameters=True)
            p = {k: torch.as_tensor(v, device=device).float() for k, v in res.result_parameters.items()}
            with torch.no_grad():
                o = smpl(betas=p["betas"], body_pose=p["body_pose"], global_orient=p["global_orient"],
                         transl=p["transl"])
            joints.append(o.joints[:, :24].cpu().numpy()), vertices.append(o.vertices.cpu().numpy())
    finally:
        os.chdir(cwd)
    return np.concatenate(joints), np.concatenate(vertices)


def score(joint_set: str, jp, vp, jg, vg, h36m_regressor) -> dict:
    """PA-MPJPE, MPJPE and PVE in mm. "24": SMPL joints rooted at the mid-hip (joints 1, 2);
    "14": the H36M-regressed joints rooted at their pelvis. PVE uses the same root as MPJPE."""
    vp, vg = vp.astype(np.float64) * MM, vg.astype(np.float64) * MM
    if joint_set == "24":
        jp, jg = jp.astype(np.float64) * MM, jg.astype(np.float64) * MM
        rp, rg = jp[:, 1:3].mean(1), jg[:, 1:3].mean(1)
    else:
        jp = np.einsum("jv,nvc->njc", h36m_regressor, vp)[:, J.J14_FROM_H36M17]
        jg = np.einsum("jv,nvc->njc", h36m_regressor, vg)[:, J.J14_FROM_H36M17]
        rp, rg = J.pelvis(jp), J.pelvis(jg)
    return dict(PA_MPJPE=float(M.pa_mpjpe(jp, jg).mean()), MPJPE=float(M.mpjpe(jp, jg, root=(rp, rg)).mean()),
                PVE=float(M.mpjpe(vp, vg, root=(rp, rg)).mean()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=("emdb", "3dpw"))
    ap.add_argument("--data", type=Path, required=True, help="EMDB root (P0/, P1/, ...) or 3DPW root "
                                                             "(sequenceFiles/, imageFiles/)")
    ap.add_argument("--model", default="medium", help="model name or .onnx path")
    ap.add_argument("--smpl-dir", type=Path, required=True,
                    help="SMPL_MALE.pkl, SMPL_FEMALE.pkl, SMPL_NEUTRAL.pkl, J_regressor_h36m.npy")
    ap.add_argument("--mhr-repo", type=Path, required=True, help="clone of github.com/facebookresearch/MHR")
    ap.add_argument("--mhr-model", type=Path, default=HERE.parent / "training" / "checkpoints" / "mhr_model.pt")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--limit", type=int, help="score only the first N frames (a quick check)")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--save", type=Path, help="also save the SMPL predictions (.npz)")
    args = ap.parse_args()

    samples = (emdb_frames if args.dataset == "emdb" else threedpw_frames)(args.data, args.stride)[:args.limit]
    print(f"{args.dataset}: {len(samples)} frames")
    params, shapes, cam_trans = predict(samples, resolve_model(args.model), args.device)
    rig = torch.jit.load(str(args.mhr_model), map_location=args.device).eval()
    verts = mhr_vertices(rig, params, shapes, args.device)
    faces = rig.character_torch.mesh.faces.cpu().numpy()
    jp, vp = official_smpl_fit(verts, cam_trans, faces, args.mhr_repo, args.smpl_dir, args.device)
    jg, vg = ground_truth(samples, args.smpl_dir, args.device)
    h36m = np.load(args.smpl_dir / "J_regressor_h36m.npy").astype(np.float64)

    published = "24" if args.dataset == "emdb" else "14"
    results = {js: score(js, jp, vp, jg, vg, h36m) for js in ("24", "14")}
    for js, r in results.items():
        mark = "  <- paper" if js == published else ""
        print(f"J{js}: PA-MPJPE {r['PA_MPJPE']:.1f}  MPJPE {r['MPJPE']:.1f}  PVE {r['PVE']:.1f} mm{mark}")
    if args.save:
        np.savez(args.save, joints=jp, vertices=vp, mhr_params=params, shape_params=shapes, cam_trans=cam_trans,
                 results=json.dumps(results))


if __name__ == "__main__":
    main()
