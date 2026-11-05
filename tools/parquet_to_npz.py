#!/usr/bin/env python3
"""
Convert the official ``facebook/sam-3d-body-dataset`` parquet annotations into the
per-crop ``images/*.png`` + ``annotations/*.npz`` layout consumed by
``training/train.py``.

These are the ground-truth MHR fits SAM 3D Body was built from. Facts about the parquet:
  * The parquet has no ``cam_trans`` column, but the SAM3D translation is fully
    recoverable. The global translation is the first three MHR ``model_params``:

        cam_trans = [ model_params[0], -model_params[1], -model_params[2] ] / 10

    and the body-centred joints (vision frame, Y down) are

        joints_3d = keypoints_3d[:, :3] * [1, -1, -1]   # negate Y and Z

    Together these reproject keypoints_3d onto the stored full-frame
    keypoints_2d through ``cam_int`` at ~0 px (verified on coco_train).
  * The dataset stores its own per-image focal length in ``cam_int`` (NOT the
    sqrt(H^2+W^2) heuristic), with principal point at the image centre. We copy
    the focal straight from ``cam_int`` so reprojection stays consistent.
  * Persons with ``mhr_valid == False`` have unreliable fits and are skipped by
    default (the 2D in-the-wild splits — coco/mpii/aic — rely on this flag).

NPZ payload (per crop):
    orig_shape          (2,)     [H, W] of the full uncropped image
    bbox                (4,)     [x1, y1, x2, y2] in original image coords (xyxy)
    bbox_square         (4,)     [x1, y1, x2, y2] expanded square crop (1.2x)
    cam_focal_length    (2,)     [fx, fy] taken from cam_int diagonal
    cam_trans           (3,)     [tx, ty, tz] SAM3D translation, vision frame, metres
    mhr_model_params    (204,)   MHR pose + scale params (== parquet model_params)
    shape_params        (45,)    MHR identity blendshapes
    joints_3d           (70, 3)  body-centred joints, vision Y-down (metres)
    joints_2d           (70, 2)  full-frame pixel coords

The same converter works for every SAM 3D Body split — the annotation schema and
the MHR math are dataset-independent (verified: cam_trans recovery reprojects to
~0 px and the principal point is image-centred on coco / mpii / aic / harmony4d /
egohumans alike). Only two things vary per dataset and are handled here:
  * image path resolution (``image_relpath`` mirrors Meta's get_img_name), and
  * the output crop/annotation name, which is built from the FULL relative image
    path so nested-path datasets (harmony4d, egohumans) don't collide.

``--image_dir`` must point at the prepared images for that dataset. For
harmony4d and egohumans these are the **undistorted** frames produced by Meta's
``data/scripts/<dataset>/undistort_*.py`` — run that first.

Usage:
    # 1. Download annotations for a split (needs HF access to the gated repo):
    #    python data/scripts/download.py --save_dir $ANN_DIR --splits coco_train
    # 2. Prepare the matching images (download; undistort for harmony4d/egohumans).
    # 3. Convert (one split -> one output dir under data/):
    python tools/parquet_to_npz.py \\
        --annotation_dir $ANN_DIR/coco_train \\
        --image_dir      $COCO_IMG_DIR \\
        --output_dir     data/sam3d_gt_coco \\
        --validate

    # other splits — identical call, swap the split / image dir / output dir:
    #   mpii_train       --image_dir $MPII_IMG_DIR        --output_dir data/sam3d_gt_mpii
    #   aic_train        --image_dir $AIC_IMG_DIR         --output_dir data/sam3d_gt_aic
    #   harmony4d_train  --image_dir $HARMONY4D_IMG_DIR   --output_dir data/sam3d_gt_harmony4d
    #   egohumans_train  --image_dir $EGOHUMANS_IMG_DIR   --output_dir data/sam3d_gt_egohumans

Resume-safe: existing ``annotations/<name>.npz`` files are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

CROP_SIZE = 224
# Negate Y and Z to map MHR-native (Y-up) coordinates into the vision camera
# frame (Y-down, Z-forward) the trainer uses.
NEG_YZ = np.array([1.0, -1.0, -1.0], dtype=np.float64)
# MHR root translation (model_params[:3]) is stored in decimetres.
TRANS_SCALE = 10.0


def get_square_crop_padded(image: np.ndarray, bbox: np.ndarray, expand: float = 1.2):
    """Square crop around *bbox*, black-padded where it leaves the image."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox.astype(float)

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    size = max(x2 - x1, y2 - y1) * expand
    half = size / 2.0

    sq_x1, sq_y1 = int(cx - half), int(cy - half)
    sq_x2, sq_y2 = int(cx + half), int(cy + half)

    pad_top = max(0, -sq_y1)
    pad_bottom = max(0, sq_y2 - h)
    pad_left = max(0, -sq_x1)
    pad_right = max(0, sq_x2 - w)

    valid_x1, valid_y1 = max(0, sq_x1), max(0, sq_y1)
    valid_x2, valid_y2 = min(w, sq_x2), min(h, sq_y2)
    crop_valid = image[valid_y1:valid_y2, valid_x1:valid_x2]

    crop_square = cv2.copyMakeBorder(
        crop_valid, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )
    sq_bbox = np.array([sq_x1, sq_y1, sq_x2, sq_y2], dtype=np.float32)
    return crop_square, sq_bbox


# ---------------------------------------------------------------------------
# Parquet field helpers
# ---------------------------------------------------------------------------
def _arr(x) -> np.ndarray:
    """Parquet cells come back as numpy object/ragged arrays; normalise them."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    return np.array(x)


def image_relpath(dataset: str, image: str) -> str:
    """Mirror data/scripts/create_webdataset.py:get_img_name for image lookup."""
    if dataset == "coco":
        # e.g. COCO_train2014_000000312652.jpg -> train2014/COCO_train2014_...jpg
        _, split, _ = image.split("_")
        return os.path.join(split, image)
    if dataset == "mpii":
        return os.path.join("images", image)
    if dataset == "aic":
        return os.path.join("train", "images", image)
    # 3dpw / harmony4d / egoexo4d / egohumans / sa1b store a usable relative path
    return image


def to_xyxy(bbox: np.ndarray, fmt: str) -> np.ndarray:
    bbox = bbox.astype(np.float32)
    if fmt == "xyxy":
        return bbox
    if fmt == "xywh":
        x, y, w, h = bbox
        return np.array([x, y, x + w, y + h], dtype=np.float32)
    raise ValueError(f"Unsupported bbox_format: {fmt}")


def cam_trans_from_params(model_params: np.ndarray) -> np.ndarray:
    """Recover SAM3D camera translation (vision frame, metres) from MHR params."""
    t = model_params[:3].astype(np.float64) * NEG_YZ / TRANS_SCALE
    return t.astype(np.float32)


def reproj_error(joints_3d: np.ndarray, cam_trans: np.ndarray,
                 joints_2d: np.ndarray, focal: np.ndarray,
                 orig_shape: np.ndarray) -> float:
    """RMS px error of (joints_3d + cam_trans) projected onto joints_2d.

    Camera model of the training losses: focal from cam_int, principal point at
    the image centre.
    """
    h, w = float(orig_shape[0]), float(orig_shape[1])
    cam = joints_3d + cam_trans[None, :]
    z = np.clip(cam[:, 2], 1e-3, None)
    u = focal[0] * cam[:, 0] / z + w / 2.0
    v = focal[1] * cam[:, 1] / z + h / 2.0
    d = np.stack([u - joints_2d[:, 0], v - joints_2d[:, 1]], axis=-1)
    return float(np.sqrt((d ** 2).mean()))


# ---------------------------------------------------------------------------
# Per-person annotation builder
# ---------------------------------------------------------------------------
def build_annotation(row, img_h: int, img_w: int) -> dict:
    model_params = _arr(row["model_params"]).astype(np.float32).ravel()      # (204,)
    shape_params = _arr(row["shape_params"]).astype(np.float32).ravel()      # (45,)
    cam_int = _arr(row["cam_int"]).astype(np.float32).reshape(3, 3)

    kp3 = _arr(row["keypoints_3d"]).astype(np.float32)                       # (70, 4)
    kp2 = _arr(row["keypoints_2d"]).astype(np.float32)                       # (70, 3)

    joints_3d = (kp3[:, :3].astype(np.float64) * NEG_YZ).astype(np.float32)  # vision frame
    joints_2d = kp2[:, :2].astype(np.float32)                               # full-frame px
    cam_trans = cam_trans_from_params(model_params)  # from model_params[:3] BEFORE zeroing
    focal = np.array([cam_int[0, 0], cam_int[1, 1]], dtype=np.float32)

    # The dataset bakes the global translation into model_params[:3]; the trainer
    # expects a body-centred mesh and carries the translation in cam_trans, so it
    # is zeroed here (the mesh then matches `joints_3d` to ~1e-7).
    model_params = model_params.copy()
    model_params[:3] = 0.0

    bbox = to_xyxy(_arr(row["bbox"]), row["bbox_format"])

    return {
        "orig_shape": np.array([img_h, img_w], dtype=np.int32),
        "bbox": bbox,
        "cam_focal_length": focal,
        "cam_trans": cam_trans,
        "mhr_model_params": model_params,
        "shape_params": shape_params,
        "joints_3d": joints_3d,
        "joints_2d": joints_2d,
    }


def save_person(full_img_rgb, ann, name, images_dir, annotations_dir):
    crop_square, sq_bbox = get_square_crop_padded(full_img_rgb, ann["bbox"], expand=1.2)
    crop_224 = cv2.resize(crop_square, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(images_dir / f"{name}.png"), cv2.cvtColor(crop_224, cv2.COLOR_RGB2BGR))
    ann = dict(ann)
    ann["bbox_square"] = sq_bbox
    np.savez(annotations_dir / f"{name}.npz", **ann)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert sam-3d-body-dataset parquet -> per-person .npz crops.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--annotation_dir", required=True,
                   help="Directory of NNNNNN.parquet files for one split (e.g. .../coco_train).")
    p.add_argument("--image_dir", required=True,
                   help="Root directory of the source dataset images.")
    p.add_argument("--output_dir", required=True,
                   help="Output dir; images/ and annotations/ are created inside.")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Stop after producing N crops (counts existing on disk).")
    p.add_argument("--keep_invalid", action="store_true",
                   help="Keep persons with mhr_valid == False (skipped by default).")
    p.add_argument("--validate", action="store_true",
                   help="Assert reprojection error < 1px on each saved person.")
    p.add_argument("--reproj_tol", type=float, default=1.0,
                   help="Max allowed RMS reprojection error in px when --validate (default 1.0).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ann_dir = Path(args.annotation_dir)
    img_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    images_dir = out_dir / "images"
    annotations_dir = out_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    parquets = sorted(ann_dir.glob("*.parquet"))
    if not parquets:
        print(f"No parquet files in {ann_dir}", file=sys.stderr)
        sys.exit(1)

    existing = {p.stem for p in annotations_dir.glob("*.npz")}
    total = len(existing)
    stats = {"crops": 0, "skipped_existing": 0, "skipped_invalid": 0,
             "missing_image": 0, "failed": 0}

    pbar = tqdm(parquets, desc="shards", unit="shard")
    for pq in pbar:
        df = pd.read_parquet(pq)
        img_cache: dict[str, np.ndarray | None] = {}

        for _, row in df.iterrows():
            dataset = row["dataset"]
            image = row["image"]
            person_id = int(row["person_id"])
            # Use the FULL relative image path (minus extension) as the stem, not
            # just Path(image).stem: nested-path datasets (harmony4d, egohumans)
            # name every frame "00051.jpg" inside each cam/seq folder, so the bare
            # stem would collide across thousands of distinct frames. Flat-named
            # datasets (coco/mpii/aic) have no "/" so their names are unchanged.
            stem = os.path.splitext(image)[0].replace("/", "_").replace(os.sep, "_").replace(" ", "_")
            name = f"{dataset}_{stem}_p{person_id}"

            if name in existing:
                stats["skipped_existing"] += 1
                continue
            if not args.keep_invalid and not bool(row["mhr_valid"]):
                stats["skipped_invalid"] += 1
                continue

            if image not in img_cache:
                ipath = img_dir / image_relpath(dataset, image)
                bgr = cv2.imread(str(ipath))
                img_cache[image] = (
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
                )
            img_rgb = img_cache[image]
            if img_rgb is None:
                stats["missing_image"] += 1
                continue

            h, w = img_rgb.shape[:2]
            try:
                ann = build_annotation(row, h, w)
                if args.validate:
                    err = reproj_error(ann["joints_3d"], ann["cam_trans"],
                                       ann["joints_2d"], ann["cam_focal_length"],
                                       ann["orig_shape"])
                    if err > args.reproj_tol:
                        raise ValueError(f"reproj err {err:.2f}px > {args.reproj_tol}px")
                save_person(img_rgb, ann, name, images_dir, annotations_dir)
            except Exception as e:  # noqa: BLE001
                stats["failed"] += 1
                tqdm.write(f"[fail] {name}: {e}")
                continue

            existing.add(name)
            stats["crops"] += 1
            total += 1
            if args.max_samples and total >= args.max_samples:
                pbar.close()
                _report(stats, total, out_dir)
                return

        pbar.set_postfix(crops=stats["crops"], invalid=stats["skipped_invalid"],
                         missing=stats["missing_image"])

    _report(stats, total, out_dir)


def _report(stats, total, out_dir):
    print("=" * 56)
    print("CONVERSION COMPLETE")
    for k, v in stats.items():
        print(f"  {k:18s}: {v}")
    print(f"  total on disk     : {total}")
    print(f"  output            : {out_dir}")
    print("=" * 56)


if __name__ == "__main__":
    main()
