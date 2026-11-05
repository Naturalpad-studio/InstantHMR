#!/usr/bin/env python3
"""Floor plane per video camera, for the height-based contact labels of --contact-floor.

    python tools/build_contact_floor.py --split-dir <root>/sam3d_gt_harmony4d --out <file>.npz

The trainer's motion rule (a foot that moved < contact_thr between paired
frames is planted, > noncontact_thr is not) fails on Harmony4D: a foot that
stays within 1 cm of the floor still moves a median 3.1 cm over 5 frames there
(pivots, slides, label jitter), so 42% of its pairs come out "both feet in the
air" against ~4% with both feet more than 5 cm up. Height above the floor is
what separates them, and the floor of a static camera is the plane most foot
keypoints lie on.

For every camera track (the annotation stem without `_<frame>_p<person>`, the
key `train.PAIR_NAME` pairs on), RANSAC fits a plane through
the 8 foot keypoints (joints_3d + cam_trans, camera frame) of all its frames:
score = points within 2 cm minus 5x points more than 5 cm below it, normal
pointing to the side most points lie on. Calibrated against insole-measured
contact (368k foot labels): planted = lowest foot point < 2 cm above the plane
and moved < 5 cm, in the air = > 5 cm above it agrees with the insoles 97.3%,
and no insole-loaded foot is ever > 5 cm up.

Output: keys (K,) str, normal (K, 3), point (K, 3), inlier (K,) = fraction of
foot points within 2 cm, n (K,) = frames. Fits with inlier < --min-inlier are
dropped, so their samples keep the motion rule.
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

import numpy as np

FOOT = [13, 15, 16, 17, 14, 18, 19, 20]
PAIR_NAME = re.compile(r"^(.*)_(\d+)_p(\d+)$")      # = train.PAIR_NAME


def fit_floor(P: np.ndarray, iters: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n /= nn
        d = (P - a) @ n
        if np.median(d) < 0:
            n, d = -n, -d
        score = ((d > -0.02) & (d < 0.02)).sum() - 5 * (d < -0.05).sum()
        if best is None or score > best[0]:
            best = (score, n, a, float(((d > -0.02) & (d < 0.02)).mean()))
    return best[1:]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-dir", required=True, help="a sam3d_gt_<name> folder (has annotations/)")
    p.add_argument("--out", required=True)
    p.add_argument("--min-frames", type=int, default=30)
    p.add_argument("--min-inlier", type=float, default=0.25)
    args = p.parse_args()

    groups = collections.defaultdict(list)
    for f in (Path(args.split_dir) / "annotations").iterdir():
        m = PAIR_NAME.match(f.stem)
        if m:
            groups[m.group(1)].append(f)
    print(f"{len(groups)} camera tracks in {args.split_dir}", flush=True)

    keys, normals, points, inliers, ns = [], [], [], [], []
    dropped = 0
    for i, (k, files) in enumerate(sorted(groups.items())):
        if len(files) < args.min_frames:
            dropped += 1
            continue
        P = []
        for f in files:
            with np.load(f) as z:
                P.append(z["joints_3d"][FOOT] + z["cam_trans"])
        P = np.concatenate(P).astype(np.float64)
        P = P[np.isfinite(P).all(1)]
        n, a, inl = fit_floor(P)
        if inl < args.min_inlier:
            dropped += 1
            continue
        keys.append(k); normals.append(n); points.append(a); inliers.append(inl); ns.append(len(files))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(groups)}", flush=True)
    np.savez(args.out, keys=np.array(keys), normal=np.array(normals, np.float32),
             point=np.array(points, np.float32), inlier=np.array(inliers, np.float32),
             n=np.array(ns))
    inl = np.array(inliers)
    print(f"wrote {args.out}: {len(keys)} floors ({dropped} tracks dropped: < {args.min_frames} frames "
          f"or inlier < {args.min_inlier}); inlier fraction pct 10/50/90 = "
          f"{np.round(np.percentile(inl, [10, 50, 90]), 2) if len(inl) else '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
