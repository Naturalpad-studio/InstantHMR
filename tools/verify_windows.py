#!/usr/bin/env python3
"""Verify the precomputed-window corpus (build_split.py --windows) and the
--preset v4 dataset on real rows.

    python tools/verify_windows.py --split aic_train \
        --annotation-dir ~/sam3d_data/annotations/aic_train \
        --image-dir ~/sam3d_data/images/aic \
        --ref <train_distill_mhr_only.py before the change> --n 64

Checks:

1. Builder: the default and --context 2.0 outputs are byte-identical to HEAD's.
2. Window build: both size folders hold the same people and the same npz, which
   is the standard npz plus win_boxes; row 0 is within 1 px of bbox_square,
   rows 1.. are integer exact squares; each tile is K*S x S.
3. Every window is the RUNTIME crop: instanthmr.inference's own _preprocess,
   given the detector box that window stands for (the tight box itself for
   window 0), returns the same pixels (PNG build: exact) and the same CLIFF
   vector, and for windows 1.. the labels land where the runtime crop puts the
   annotated keypoints (window 0's box can be 1 px off square, and the label
   normalisation, unchanged since g8, uses its width for both axes). Also reports PSNR against the
   runtime crop for a JPEG q95 build (the cluster corpus).
4. Augmented v4 (baseline geometry on top): the loss's reprojection chain lands
   on the joints_2d labels.
5. The window framings follow the v3 range: side log-uniform 0.6-2.0x the
   tight box, centre shift within +/-15% of the side.
6. With crop_windows off, baseline and v2 samples are bit-identical to --ref.
7. A standard split, and tiles of the wrong size, are refused.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
BUILDER = REPO / "datasets_pipeline" / "build_split.py"
sys.path.insert(0, str(REPO / "instanthmr_distill_train"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import train_distill_mhr_only as T  # noqa: E402
import train_distill_jz as JZ  # noqa: E402
from instanthmr.inference import InstantHMR  # noqa: E402
from verify_v3_preset import reproject, same, seeded  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
K = 4
SIZES = (224, 288)
fails: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(f"  {'OK ' if ok else 'BAD'} {msg}")
    if not ok:
        fails.append(msg)


def build(builder: Path, out: Path, args, extra: list[str], fmt: str = "png") -> None:
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'datasets_pipeline'}:{REPO}")
    cmd = [sys.executable, str(builder), "--split", args.split,
           "--annotation-dir", str(args.annotation_dir),
           "--image-dir", str(args.image_dir), "--output-dir", str(out),
           "--max-samples", str(args.n), "--originals", "none", "--crop-format", fmt,
           *extra]
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)


def tree_diff(a: Path, b: Path) -> int:
    fa = {p.relative_to(a) for p in a.rglob("*") if p.is_file()}
    fb = {p.relative_to(b) for p in b.rglob("*") if p.is_file()}
    if fa != fb or not fa:
        return max(1, len(fa ^ fb))
    return sum((a / f).read_bytes() != (b / f).read_bytes() for f in fa)


def load(d: Path, name: str) -> dict:
    with np.load(d / "annotations" / f"{name}.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def psnr(a, b) -> float:
    mse = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def runtime(img_rgb, box, size, focal):
    """instanthmr.inference's crop + CLIFF for detector box `box`."""
    stub = types.SimpleNamespace(cliff_focal=True, input_size=size,
                                 _focal_px=lambda h, w, f: f)
    h, w = img_rgb.shape[:2]
    crop, x1, y1, side, cliff = InstantHMR._preprocess(stub, img_rgb, box, h, w, focal)
    u8 = np.clip((crop.transpose(1, 2, 0) * STD + MEAN) * 255.0, 0, 255)
    return u8, np.array([x1, y1, side]), cliff


def detector_box(win: np.ndarray, size: int, img_rgb, focal):
    """The detector box whose 1.2x square is exactly `win` at the runtime:
    side / 1.2 in float lands a hair off the integer and the runtime's floor /
    ceil then cut one pixel more, so nudge it by ulps until it does not."""
    c, half = (win[:2] + win[2:]) / 2, (win[2] - win[0]) / 2.4
    for t in range(64):
        det = np.concatenate([c - half, c + half])
        out = runtime(img_rgb, det, size, focal)
        x1, y1, side = out[1]          # the runtime cuts floor(x1) .. ceil(x1 + side)
        if [math.floor(x1), math.floor(y1), math.ceil(x1 + side), math.ceil(y1 + side)] \
                == [int(v) for v in win]:
            return det, out
        half = np.nextafter(half, np.inf if t % 2 else -np.inf)
    raise RuntimeError(f"no detector box reproduces {win}")


def to_uint8(img: torch.Tensor) -> np.ndarray:
    return np.clip((img.numpy().transpose(1, 2, 0) * STD + MEAN) * 255.0, 0, 255)


def cfg_for(preset: str, size: int = 224):
    cfg = T.DistillConfig()
    cfg.cliff_focal = True
    cfg.crop_centre_fix = True
    cfg.image_size = size
    if preset == "v2":
        JZ.apply_v2_preset(cfg)
    if preset == "v4":
        JZ.apply_v4_preset(cfg)
    return cfg


def dataset(mod, root, cfg, augment=True, **over):
    kw = dict(geom_p=cfg.geom_p, geom_rot_deg=cfg.geom_rot_deg,
              geom_scale_range=cfg.geom_scale_range, geom_trans=cfg.geom_trans,
              geom_flip_p=cfg.geom_flip_p, geom_scale_max=cfg.geom_scale_max,
              cliff_follows_aug=cfg.cliff_follows_aug, crop_centre_fix=cfg.crop_centre_fix,
              cliff_focal=cfg.cliff_focal, occl_p=cfg.occl_p, occl_scale=cfg.occl_scale,
              jpeg_p=cfg.jpeg_p, jpeg_quality=cfg.jpeg_quality, image_size=cfg.image_size)
    if cfg.crop_windows:
        kw["crop_windows"] = True
    kw.update(over)
    return mod.SAM3DStudentDataset(str(root), augment=augment, **kw)


def main() -> None:  # noqa: C901
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="aic_train")
    p.add_argument("--annotation-dir", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--n", type=int, default=64)
    args = p.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="verify_windows_"))
    head = tmp / "build_split_head.py"
    head.write_text(subprocess.run(
        ["git", "-C", str(REPO), "show", "HEAD:datasets_pipeline/build_split.py"],
        check=True, capture_output=True, text=True).stdout)

    print("[1] builder defaults vs HEAD")
    build(head, tmp / "head" / "sam3d_gt_aic", args, [])
    build(BUILDER, tmp / "std" / "sam3d_gt_aic", args, [])
    build(BUILDER, tmp / "std_nh" / "sam3d_gt_aic", args, ["--no-hands"])
    check(tree_diff(tmp / "head", tmp / "std") == 0, "default build byte-identical to HEAD")
    ctx = ["--context", "2.0", "--body-size", "448", "--no-hands"]
    build(head, tmp / "head_ctx", args, ctx)
    build(BUILDER, tmp / "ctx", args, ctx)
    check(tree_diff(tmp / "head_ctx", tmp / "ctx") == 0, "--context 2.0 build byte-identical to HEAD")

    win = ["--windows", str(K), "--window-sizes", ",".join(map(str, SIZES)), "--no-hands"]
    build(BUILDER, tmp / "win{size}" / "sam3d_gt_aic", args, win)
    build(BUILDER, tmp / "winjpg{size}" / "sam3d_gt_aic", args, win, fmt="jpg")
    std = tmp / "std" / "sam3d_gt_aic"
    wd = {s: tmp / f"win{s}" / "sam3d_gt_aic" for s in SIZES}
    wj = {s: tmp / f"winjpg{s}" / "sam3d_gt_aic" for s in SIZES}

    print("[2] window build layout")
    names = sorted(q.stem for q in (std / "annotations").glob("*.npz"))
    sets = [sorted(q.stem for q in (wd[s] / "annotations").glob("*.npz")) for s in SIZES]
    check(all(x == names for x in sets) and names, f"{len(names)} people in every folder")
    bad_npz = bad_box = bad_tile = 0
    for nm in names:
        a, b, c = (load(tmp / "std_nh" / "sam3d_gt_aic", nm), load(wd[SIZES[0]], nm),
                   load(wd[SIZES[1]], nm))
        wb = b.pop("win_boxes")
        bad_npz += (b.keys() != a.keys() or not all(same(a[k], b[k]) for k in a)
                    or not np.array_equal(wb, c.pop("win_boxes"))
                    or not all(same(b[k], c[k]) for k in b))
        sides = wb[1:, 2:] - wb[1:, :2]
        bad_box += (wb.shape != (K, 4) or np.abs(wb[0] - a["bbox_square"]).max() > 1
                    or not np.array_equal(wb[1:], np.round(wb[1:]))
                    or not np.array_equal(sides[:, 0], sides[:, 1]))
        for s in SIZES:
            tile = cv2.imread(str(wd[s] / "body_crops" / f"{nm}.png"))
            bad_tile += tile.shape != (s, K * s, 3)
    check(bad_npz == 0, f"npz = standard --no-hands npz + win_boxes, same in both folders "
                        f"({bad_npz} bad)")
    check(bad_box == 0, f"win_boxes: row 0 within 1 px of bbox_square, rows 1.. integer "
                        f"squares ({bad_box} bad)")
    check(bad_tile == 0, f"tiles K*S x S ({bad_tile} bad)")

    print("[3] each window is the runtime crop")
    for s in SIZES:
        cfg = cfg_for("v4", s)
        plain = dataset(T, wd[s].parent, cfg, augment=False).transform
        ds = dataset(T, wd[s].parent, cfg, augment=True, geom_p=0.0)
        ds.transform = plain                       # no colour / occlusion noise
        dsj = dataset(T, wj[s].parent, cfg, augment=True, geom_p=0.0)
        dsj.transform = plain
        exact = cl = lab = 0.0
        ps = {k: [] for k in range(K)}
        n_k = np.zeros(K, int)
        for i in range(len(ds)):
            nm = Path(ds.pairs[i][1]).stem
            a = load(wd[s], nm)
            img = cv2.cvtColor(cv2.imread(str(args.image_dir / a["image_relpath"].item())),
                               cv2.COLOR_BGR2RGB)
            f = float(a["cam_focal_length"][1])
            for k in range(K):
                # the draw inside the dataset is random.randrange(K): pick k
                for seed in range(10_000):
                    random.seed(seed)
                    if random.randrange(K) == k:
                        break
                smp = seeded(ds, i, seed)
                smj = seeded(dsj, i, seed)
                box = a["win_boxes"][k].astype(np.float64)
                if k == 0:
                    rt, (x1, y1, side), cliff = runtime(img, a["bbox"].astype(np.float64), s, f)
                else:
                    _, (rt, (x1, y1, side), cliff) = detector_box(box, s, img, f)
                exact = max(exact, float(np.abs(to_uint8(smp["image"]) - rt).max()))
                ps[k].append(psnr(to_uint8(smj["image"]), rt))
                cl = max(cl, float(np.abs(smp["cliff_cond"].numpy() - cliff).max()))
                j = a["joints_2d"].astype(np.float64)
                want = np.stack([(j[:, 0] - x1) / side, (j[:, 1] - y1) / side], 1) * 2 - 1
                vis = smp["joints_2d_vis"].numpy() > 0
                err = np.abs(smp["joints_2d"].numpy() - want)[vis] * s / 2
                if k:
                    lab = max(lab, float(err.max()))
                n_k[k] += 1
        check(exact < 0.01, f"{s} px: every window pixel-equal to the runtime crop "
                            f"(max |diff| {exact:.4f} /255, PNG build)")
        check(cl < 1e-5, f"{s} px: CLIFF = runtime's for that box (max diff {cl:.1e})")
        check(lab < 1e-3, f"{s} px: labels = keypoints in the runtime crop (max {lab:.1e} px)")
        msg = ", ".join(f"w{k} {np.mean(ps[k]):.1f}" for k in range(K))
        check(min(np.mean(ps[k]) for k in range(K)) > 36.0,
              f"{s} px JPEG q95 build vs runtime crop, PSNR dB: {msg}")

    print("[4] augmented v4: reprojection chain")
    for s in SIZES:
        ds = dataset(T, wd[s].parent, cfg_for("v4", s), augment=True, geom_p=1.0)
        rep, shapes, aug = 0.0, set(), 0
        for t in range(200):
            smp = seeded(ds, t % len(ds), 500 + t)
            vis = smp["joints_2d_vis"].numpy() > 0
            err = np.abs(reproject(smp, s) - smp["joints_2d"].numpy())[vis] * s / 2
            rep = max(rep, float(err.max()))
            shapes.add(tuple(smp["image"].shape))
            aug += int(float(smp["aug_active"]) > 0.5)
        check(rep < 1e-3 and aug == 200, f"{s} px: max {rep:.2e} px over 200 augmented samples")
        check(shapes == {(3, s, s)}, f"{s} px: output shapes {shapes}")

    print("[5] framing range")
    ms, sh = [], []
    for nm in names:
        a = load(wd[224], nm)
        t = a["bbox"].astype(np.float64)
        b = max(t[2] - t[0], t[3] - t[1])
        for w in a["win_boxes"][1:].astype(np.float64):
            side = w[2] - w[0]
            ms.append(side / b)
            sh.append(np.abs((w[:2] + w[2:]) / 2 - (t[:2] + t[2:]) / 2).max() / side)
    ms, sh = np.array(ms), np.array(sh)
    expect = math.log(1.1 / 0.6) / math.log(2.0 / 0.6)
    check(ms.min() > 0.59 and ms.max() < 2.01 and abs(np.mean(ms < 1.1) - expect) < 0.08,
          f"side {ms.min():.2f}-{ms.max():.2f}x, P(<1.1x) {np.mean(ms < 1.1):.2f} vs "
          f"log-uniform {expect:.2f} ({len(ms)} windows)")
    check(sh.max() < 0.15 + 0.01, f"centre shift max {sh.max():.3f} of the side")

    print("[6] crop_windows off: bit-identical to --ref")
    spec = importlib.util.spec_from_file_location("ref_trainer", args.ref)
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    for preset in ("baseline", "v2"):
        cfg = cfg_for(preset)
        new, old = dataset(T, std.parent, cfg), dataset(R, std.parent, cfg)
        diff = 0
        for t in range(24):
            for _ in range(2):           # alternating calls
                a, b = seeded(new, t, 7 + t), seeded(old, t, 7 + t)
                diff += not all(same(a[x], b[x]) for x in a)
        check(diff == 0, f"{preset}: {diff} of 48 samples differ")

    print("[7] refusals")
    for root, size, want in ((std.parent, 224, "win_boxes"), (wd[224].parent, 288, "tile")):
        try:
            dataset(T, root, cfg_for("v4", size))
            check(False, f"{root.name} accepted at {size} px")
        except ValueError as e:
            check(want in str(e), f"refused: {str(e)[:90]}")

    print("[8] window_mode: v4 unchanged, 'standard' = the baseline framing")
    cfg = cfg_for("v4")
    new, old = dataset(T, wd[224].parent, cfg), dataset(R, wd[224].parent, cfg)
    diff = sum(not all(same(a[x], b[x]) for x in a)
               for t in range(24) for _ in range(2)
               for a, b in [(seeded(new, t, 7 + t), seeded(old, t, 7 + t))])
    check(diff == 0, f"v4 (window_mode random): {diff} of 48 samples differ from --ref")
    # Same seeds, baseline augmentation: on the standard corpus vs window 0 of
    # the window corpus. The standard draw consumes no randomness, so every
    # augmentation must match; only the crop box differs (int() vs the runtime's
    # floor / ceil rounding, <= 1 px), so the labels may move by about that.
    base = cfg_for("baseline")
    a_ds = dataset(T, std.parent, base)
    b_ds = dataset(T, wd[224].parent, base, crop_windows=True, window_mode="standard")
    names_a = [Path(p[1]).stem for p in a_ds.pairs]
    cl = lab = 0.0
    same_aug = box0 = True
    for t in range(48):
        i = t % len(b_ds)
        j = names_a.index(Path(b_ds.pairs[i][1]).stem)
        a, b = seeded(a_ds, j, 300 + t), seeded(b_ds, i, 300 + t)
        win0 = load(wd[224], Path(b_ds.pairs[i][1]).stem)["win_boxes"][0]
        box0 &= np.array_equal(np.asarray(b["bbox_square"]), win0)
        same_aug &= all(torch.equal(a[k], b[k]) for k in ("aug_active", "aug_flip", "aug_cos", "aug_sin"))
        cl = max(cl, float((a["cliff_cond"] - b["cliff_cond"]).abs().max()))
        lab = max(lab, float((a["joints_2d"] - b["joints_2d"]).abs().max()) * 112)
    check(box0, "standard: every augmented sample uses window 0")
    check(same_aug and cl < 1e-6, f"standard: same augmentation draws and CLIFF as baseline "
                                  f"on the standard corpus (CLIFF max diff {cl:.1e})")
    check(lab < 3.0, f"standard: labels within {lab:.2f} px of baseline's (box rounding only)")
    ds = dataset(T, wd[288].parent, cfg_for("baseline", 288), crop_windows=True,
                 window_mode="standard", geom_p=1.0)
    rep = 0.0
    for t in range(100):
        smp = seeded(ds, t % len(ds), 900 + t)
        vis = smp["joints_2d_vis"].numpy() > 0
        rep = max(rep, float((np.abs(reproject(smp, 288) - smp["joints_2d"].numpy())[vis]
                              * 144).max()))
    check(rep < 1e-3, f"standard at 288: reprojection chain max {rep:.2e} px")

    print("[9] --windows on the trainer's command line")
    def cfg_cli(*flags):
        old_argv = sys.argv
        sys.argv = ["train_distill_jz.py", "--cliff-focal", *flags]
        try:
            return JZ.build_cfg(JZ.parse_args())
        finally:
            sys.argv = old_argv
    c = cfg_cli("--preset", "baseline", "--windows", "standard")
    check(c.crop_windows and c.window_mode == "standard" and not c.cliff_follows_aug
          and not c.mask_oob_2d and c.occl_p == 0.2 and c.jpeg_p == 0.0 and c.w_reproj == 0.01,
          "baseline + --windows standard: window 0, baseline augmentation and losses")
    c = cfg_cli("--preset", "baseline", "--windows", "random")
    check(c.crop_windows and c.window_mode == "random" and c.cliff_follows_aug
          and c.mask_oob_2d and c.occl_p == 0.2 and c.occl_scale == (0.02, 0.15)
          and c.jpeg_p == 0.0 and c.w_reproj == 0.01,
          "baseline + --windows random: v4 framings, baseline occlusion, no JPEG")
    c4 = cfg_cli("--preset", "v4")
    check(c4.window_mode == "random" and c4.occl_p == 0.7 and c4.jpeg_p == 0.3,
          "--preset v4 unchanged")
    try:
        cfg_cli("--preset", "v3", "--windows", "standard")
        check(False, "--preset v3 --windows accepted")
    except SystemExit as e:
        check("v3" in str(e), "--preset v3 --windows refused")

    print("[10] --window0-p: how often the standard crop is drawn")
    rcfg = cfg_for("baseline")
    for p0 in (0.5, 0.75):
        ds = dataset(T, wd[224].parent, rcfg, crop_windows=True, window0_p=p0)
        counts = np.zeros(K, int)
        for t in range(2000):
            random.seed(5000 + t)
            i = t % len(ds)
            box = np.asarray(ds[i]["bbox_square"])
            wb = load(wd[224], Path(ds.pairs[i][1]).stem)["win_boxes"]
            counts[int(np.argmax([np.array_equal(box, w) for w in wb]))] += 1
        f = counts / counts.sum()
        rest = (1 - p0) / (K - 1)
        check(abs(f[0] - p0) < 0.04 and np.all(np.abs(f[1:] - rest) < 0.04),
              f"p0={p0}: window frequencies {np.round(f, 3).tolist()} "
              f"(expect {p0} then {rest:.3f} each)")
    c = cfg_cli("--preset", "baseline", "--windows", "random", "--window0-p", "0.5")
    check(c.window0_p == 0.5 and c.window_mode == "random", "--window0-p 0.5 reaches the config")
    for bad in (("--preset", "baseline", "--windows", "standard", "--window0-p", "0.5"),
                ("--preset", "baseline", "--window0-p", "0.5"),
                ("--preset", "baseline", "--windows", "random", "--window0-p", "1.5")):
        try:
            cfg_cli(*bad)
            check(False, f"accepted: {' '.join(bad)}")
        except SystemExit:
            check(True, f"refused: {' '.join(bad[2:])}")

    if fails:
        print(f"FAIL ({len(fails)})")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
