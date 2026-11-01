"""Previous-pose feedback and jitter for the video benchmarks (3DPW, EMDB).

A ``--prev-token`` model can be fed its own earlier prediction. Here every
person-frame at frame ``f`` gets the prediction made at frame ``f - lag`` of
the same track (sequence, person), or the previous pose withheld when that
frame has none (the first ``lag`` frames, a missed detection, a non-finite
prediction). At 30 fps the default lag of 5 frames is 167 ms, inside the
training pairs (Harmony4D and semelle, 5-20 frames apart).

Frames are processed in rounds of ``lag`` frames per track, all tracks
together: round ``r`` only needs round ``r - 1``, so batches stay large.

``accel`` is the standard video-HMR jitter number: the second difference over
three consecutive frames of pelvis-centred joints, in mm per frame^2, for the
prediction (``accel_mm``) and against the GT's (``accel_err_mm``).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

import train_distill_mhr_only as T


def _track_rounds(samples, lag):
    """Index batches in dependency order, and frame -> index per track."""
    tracks = defaultdict(dict)
    for i, s in enumerate(samples):
        tracks[(s["sequence"], s["person"])][s["frame"]] = i
    rounds = defaultdict(list)
    for frames in tracks.values():
        f0 = min(frames)
        for f, i in frames.items():
            rounds[(f - f0) // lag].append(i)
    return [sorted(rounds[r]) for r in sorted(rounds)], tracks


@torch.no_grad()
def predict_feedback(model, ds, mhr_module, device, lag=5, use_amp=True,
                     exact_landmarks=False, batch_size=64, num_workers=6):
    """Same return as ``eval_3dpw_ckpt.predict``: (N, 70, 3) mm, indices, dropped."""
    model.eval()
    rounds, tracks = _track_rounds(ds.samples, lag)
    batches = [r[k:k + batch_size] for r in rounds for k in range(0, len(r), batch_size)]
    loader = torch.utils.data.DataLoader(ds, batch_sampler=batches,
                                         num_workers=num_workers, pin_memory=True)
    prev = {}                                   # index -> PREV_DIM vector
    kp_all = {}
    n_dropped = 0
    zeros = np.zeros(T.PREV_DIM, np.float32)
    for ids, b in zip(batches, loader):
        pp, hp = [], []
        for i in ids:
            s = ds.samples[i]
            j = tracks[(s["sequence"], s["person"])].get(s["frame"] - lag)
            v = prev.get(j) if j is not None else None
            pp.append(zeros if v is None else v)
            hp.append(0.0 if v is None else 1.0)
        img = b["image"].to(device, non_blocking=True)
        cc = b["cliff_cond"].to(device, non_blocking=True)
        pp = torch.from_numpy(np.stack(pp)).to(device)
        hp = torch.tensor(hp, device=device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(img, cc, pp, hp)
        kp = mhr_module.get_native_keypoints(out["mhr_params"].float(),
                                             out["shape_params"].float(),
                                             exact=exact_landmarks).float().cpu().numpy()
        t = out["cam_trans"].float().cpu().numpy()
        ok = b["ok"].bool().numpy()
        for k, i in enumerate(ids):
            if not ok[k]:
                continue
            if not (np.isfinite(kp[k]).all() and np.isfinite(t[k]).all()):
                n_dropped += 1
                continue
            kp_all[i] = kp[k]
            z = max(float(t[k, 2]), 0.1)
            prev[i] = np.concatenate([kp[k].reshape(-1),
                                      [t[k, 0] / z, t[k, 1] / z, math.log(z)]]).astype(np.float32)
    idx = np.array(sorted(kp_all))
    return np.stack([kp_all[i] for i in idx]) * 1000.0, idx, n_dropped


def accel(pred, gt, root_p, root_g, samples):
    """(pred, gt) are (N, J, 3) mm in the same joint set, ``samples`` the
    matching sample dicts; triplets are three consecutive frames of one track."""
    p = pred - root_p[:, None]
    g = gt - root_g[:, None]
    pos = {(s["sequence"], s["person"], s["frame"]): n for n, s in enumerate(samples)}
    tri = [(pos[(q, r, f - 1)], n, pos[(q, r, f + 1)])
           for (q, r, f), n in pos.items()
           if (q, r, f - 1) in pos and (q, r, f + 1) in pos]
    if not tri:
        return dict(accel_mm=None, accel_err_mm=None, n_triplets=0)
    a, b, c = (np.array(x) for x in zip(*tri))
    ap = p[a] - 2 * p[b] + p[c]
    ag = g[a] - 2 * g[b] + g[c]
    return dict(accel_mm=float(np.linalg.norm(ap, axis=-1).mean()),
                accel_err_mm=float(np.linalg.norm(ap - ag, axis=-1).mean()),
                n_triplets=len(tri))
