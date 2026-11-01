#!/usr/bin/env python3
"""Foot lock on the held-out semelle participants.

    python benchmark/eval_footlock.py --ckpt <best_student_model_v3.pth> --out benchmark/results/<name>_footlock.json

Inputs are datasets_pipeline/semelle/label.py `eval` files: per labelled frame
(every 5th video frame, 8 fps) the participant's detector box, insole contact
per foot and the 8 mocap foot keypoints in the camera frame. Frames are
undistorted to the centred pinhole the training split uses, the box with them,
and the model sees the runtime crop (val3dpw._preprocess) at the calibrated focal.

Two modes for a --prev-token checkpoint, one for any other:
  masked    previous pose always withheld (single-frame model)
  feedback  previous pose = the model's own prediction on the previous
            labelled frame; withheld on the first frame and after a gap of
            more than 20 video frames (the longest pair it was trained on)

Metrics, over consecutive labelled frames 5 video frames apart (125 ms):
  slide_mm      mean displacement of a foot (mean of its 4 keypoints, camera
                frame) while the insole says it is in contact in both frames
  mocap_slide_mm  the same for the mocap feet: what a planted foot really moves
  foot_err_mm   foot keypoint error against the mocap after removing each
                trial's median offset (the depth bias label.py also removes)
  contact_f1    per-foot contact head against the insole (prev-token only)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "instanthmr_distill_train"))
sys.path.insert(0, str(ROOT / "datasets_pipeline/semelle"))
import train_distill_mhr_only as T  # noqa: E402
import val3dpw  # noqa: E402

FEET = T.FOOT_KP.tolist()
MAX_GAP = 20


def run_trial(path, model, mhr, cfg, mode, device):
    import label as L
    z = np.load(path)
    cam = L.vc.Camera(**json.loads(str(z["camera"])))
    stem = path.stem                                  # <session>_<trial>_<cam>
    session, rest = stem.split("_", 1)
    trial, cam_name = rest.rsplit("_", 1)
    cap = cv2.VideoCapture(str(L.MANIP / session / trial / f"{cam_name}_manip.mp4"))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K, dist, Kc = L.centred_K(cam, w, h)
    maps = cv2.initUndistortRectifyMap(K, dist, None, Kc, (w, h), cv2.CV_32FC1)
    want = {int(f): i for i, f in enumerate(z["frame"])}
    feet = np.full((len(want), 8, 3), np.nan)
    logits = np.full((len(want), 2), np.nan)
    prev, last = None, -10 ** 9
    f = 0
    while want and f <= max(want):
        if f not in want:
            cap.grab(); f += 1
            continue
        ok, bgr = cap.read()
        if not ok:
            break
        i = want[f]
        rgb = cv2.cvtColor(cv2.remap(bgr, *maps, cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        box = L.undistort_box(z["bbox"][i], K, dist, Kc)
        img, cliff = val3dpw._preprocess(rgb, box, h, w, focal=cam.focal_px if cfg.cliff_focal else None,
                                         size=cfg.image_size)
        img = torch.from_numpy(img)[None].to(device)
        cliff = torch.from_numpy(cliff)[None].to(device)
        args = ()
        if cfg.prev_token:
            use = mode == "feedback" and prev is not None and f - last <= MAX_GAP
            pp = torch.from_numpy(prev if use else np.zeros(T.PREV_DIM, np.float32))[None].to(device)
            args = (pp, torch.tensor([1.0 if use else 0.0], device=device))
        with torch.no_grad():
            o = model(img, cliff, *args)
            kp = mhr.regress_keypoints(mhr.to_vision(
                mhr.get_joints(o["mhr_params"].float(), o["shape_params"].float())[..., :3]))[0].cpu().numpy()
        t = o["cam_trans"][0].float().cpu().numpy()
        feet[i] = (kp + t)[FEET]
        if "contact_logits" in o:
            logits[i] = o["contact_logits"][0].float().cpu().numpy()
        zt = max(float(t[2]), 0.1)
        prev = np.concatenate([kp.reshape(-1), [t[0] / zt, t[1] / zt, np.log(zt)]]).astype(np.float32)
        last = f
        f += 1
    return z, feet, logits


def lock(z, feet, planted):
    """Causal foot lock: while a foot is called planted on consecutive labelled
    frames, shift its 4 keypoints so their mean sits at the running average of
    the predicted foot since contact onset (one frame's error does not hold for
    the whole stance); released, the raw prediction comes back (``jump`` is
    that step). Camera frame, so it assumes the static semelle cameras."""
    fr = z["frame"]
    out = feet.copy()
    jump = []
    for s in range(2):
        sl = slice(4 * s, 4 * s + 4)
        total, n = None, 0
        for k in range(len(fr)):
            ok = np.isfinite(feet[k, sl]).all()
            if n and planted[k, s] and ok and fr[k] - fr[k - 1] == 5:
                total += feet[k, sl].mean(0); n += 1
                out[k, sl] = feet[k, sl] - feet[k, sl].mean(0) + total / n
                continue
            if n and k and fr[k] - fr[k - 1] == 5 and not planted[k, s]:
                jump.append(np.linalg.norm(out[k, sl].mean(0) - out[k - 1, sl].mean(0)))
            total, n = (feet[k, sl].mean(0).copy(), 1) if planted[k, s] and ok else (None, 0)
    return out, jump


def score(z, feet, logits):
    fr = z["frame"]
    mocap = z["feet_cam"].astype(np.float64)
    c, v = z["contact"], z["contact_valid"]
    pair = np.where(np.diff(fr) == 5)[0]
    slide, mslide = [], []
    for k in pair:
        for s in range(2):
            sl = slice(4 * s, 4 * s + 4)
            if v[k, s] and v[k + 1, s] and c[k, s] and c[k + 1, s]:
                d = np.linalg.norm(feet[k + 1, sl].mean(0) - feet[k, sl].mean(0))
                dm = np.linalg.norm(mocap[k + 1, sl].mean(0) - mocap[k, sl].mean(0))
                if np.isfinite(d) and np.isfinite(dm):
                    slide.append(d); mslide.append(dm)
    ok = np.isfinite(feet).all((1, 2)) & np.isfinite(mocap).all((1, 2))
    off = np.median((mocap[ok] - feet[ok]).mean(1), axis=0) if ok.any() else np.zeros(3)
    err = np.linalg.norm(mocap[ok] - off - feet[ok], axis=-1).ravel()
    tp = fp = fn = 0
    if np.isfinite(logits).any():
        m = v > 0
        pr = logits > 0
        tp, fp, fn = int((pr & (c > 0) & m).sum()), int((pr & (c == 0) & m).sum()), int((~pr & (c > 0) & m).sum())
    return dict(slide=slide, mslide=mslide, err=err.tolist(), tp=tp, fp=fp, fn=fn)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--eval-dir", default=str(ROOT / "data/semelle_work/eval"))
    p.add_argument("--out", required=True)
    p.add_argument("--max-trials", type=int, default=None)
    p.add_argument("--lock", action="store_true",
                   help="also score the feet after lock(): '+lock' planted = the "
                        "contact head (prev-token only), '+oracle' = the insole")
    args = p.parse_args()
    device = "cuda"
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
    cfg, prov = T.config_from_checkpoint(state, args.ckpt)
    model = T.InstantHMRStudent(cfg, pretrained=False).to(device).eval()
    model.load_state_dict(state)
    mhr = T.MHRForwardPass(cfg.mhr_model_path, device, kp_regressor=np.load(cfg.kp_regressor_path))
    files = sorted(Path(args.eval_dir).glob("*.npz"))[: args.max_trials]
    modes = ("masked", "feedback") if cfg.prev_token else ("single",)
    res = {}
    variants = [""]
    if args.lock:
        variants += (["+lock"] if cfg.prev_token else []) + ["+oracle"]
    for mode in modes:
        acc = {v: dict(slide=[], mslide=[], err=[], jump=[], tp=0, fp=0, fn=0) for v in variants}
        for fp in files:
            z, feet, logits = run_trial(fp, model, mhr, cfg, mode, device)
            for var in variants:
                jump = []
                if var == "+lock":
                    feet_v, jump = lock(z, feet, logits > 0)
                elif var == "+oracle":
                    feet_v, jump = lock(z, feet, (z["contact"] > 0) & (z["contact_valid"] > 0))
                else:
                    feet_v = feet
                s = score(z, feet_v, logits)
                s["jump"] = jump
                for k in ("slide", "mslide", "err", "jump"):
                    acc[var][k] += s[k]
                for k in ("tp", "fp", "fn"):
                    acc[var][k] += s[k]
        for var in variants:
            a = acc[var]
            f1 = 2 * a["tp"] / max(2 * a["tp"] + a["fp"] + a["fn"], 1) if a["tp"] + a["fn"] else None
            res[mode + var] = dict(slide_mm=1000 * float(np.mean(a["slide"])),
                                   slide_median_mm=1000 * float(np.median(a["slide"])),
                                   mocap_slide_mm=1000 * float(np.mean(a["mslide"])),
                                   foot_err_mm=1000 * float(np.mean(a["err"])),
                                   contact_f1=f1, n_contact_pairs=len(a["slide"]))
            if var:
                res[mode + var]["release_jump_mm"] = 1000 * float(np.mean(a["jump"])) if a["jump"] else None
            print(mode + var, json.dumps(res[mode + var]))
    Path(args.out).write_text(json.dumps(dict(ckpt=args.ckpt, trials=len(files), **res), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
