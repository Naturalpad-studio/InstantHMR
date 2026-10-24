"""Replace a benchmark's person boxes with the deployed detector's.

The published protocols crop around a box nobody has at runtime (projected GT
joints on 3DPW, EMDB's annotated boxes). ``instanthmr.pipeline`` feeds the
pose model RF-DETR's raw box, so this scores the same path: run RF-DETR on
each frame, keep the person detection that overlaps the GT person best, and
crop around it exactly as the phone would. A frame where no detection reaches
``MIN_IOU`` with the GT person is a missed detection: it is dropped and
counted, since there is nothing the pose model could be given.

Detections are cached per image (all persons, so the same cache serves every
person in a frame and every later run).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .threedpw import bbox_from_points

MIN_IOU = 0.3
CACHE_DIR = Path.home() / ".cache" / "instanthmr" / "detboxes"


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def detect_all(image_paths, cache_name: str, variant: str = "medium",
               confidence: float = 0.5) -> dict[str, list]:
    """{image path: [[x1, y1, x2, y2, conf], ...]}, from the cache when present."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{cache_name}_rfdetr-{variant}_c{confidence}.json"
    cache = json.loads(path.read_text()) if path.is_file() else {}
    todo = sorted(set(image_paths) - cache.keys())
    if todo:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the repo, for instanthmr
        from instanthmr.detector import RFDETRDetector
        det = RFDETRDetector(variant=variant, confidence=confidence, max_persons=20)
        for i, p in enumerate(tqdm(todo, desc=f"RF-DETR {variant}", unit="img")):
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            dets = det.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) if bgr is not None else []
            cache[p] = [[*map(float, d["bbox"]), d["confidence"]] for d in dets]
            if i % 2000 == 1999:
                path.write_text(json.dumps(cache))
        path.write_text(json.dumps(cache))
    return cache


def apply_detector_boxes(ds, cache_name: str, variant: str = "medium") -> dict:
    """Swap every ``ds.samples[i]["bbox"]`` for the matched RF-DETR box and drop
    missed detections from ``ds.samples`` and ``ds.gt`` alike. Each sample needs
    ``image_path`` and ``uv`` (the projected GT joints)."""
    cache = detect_all([s["image_path"] for s in ds.samples], cache_name, variant)
    keep, ious = [], []
    for i, s in enumerate(ds.samples):
        ref = bbox_from_points(np.asarray(s["uv"], dtype=np.float64), 1.0)
        cands = cache.get(s["image_path"], [])
        best = max(cands, key=lambda d: _iou(np.asarray(d[:4]), ref), default=None)
        iou = _iou(np.asarray(best[:4]), ref) if best is not None else 0.0
        if iou < MIN_IOU:
            continue
        s["bbox"] = np.asarray(best[:4], dtype=np.float32)
        keep.append(i)
        ious.append(iou)
    n = len(ds.samples)
    ds.samples = [ds.samples[i] for i in keep]
    ds.gt = np.asarray(ds.gt)[keep]
    return dict(n_frames=n, n_missed=n - len(keep),
                miss_rate=(n - len(keep)) / max(n, 1),
                mean_iou=float(np.mean(ious)) if ious else 0.0)
