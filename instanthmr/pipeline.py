"""Person detection followed by InstantHMR, with per-stage timings."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .detector import PersonDetector
from .inference import DETECTOR_TO_GT_BOX, HMRPrediction, InstantHMR
from .models import DEFAULT_FIT, DEFAULT_MODEL, resolve_model
from .smoothing import PoseFilter

BODY_KEYPOINTS = list(range(21)) + [41, 62] + list(range(63, 70))


@dataclass
class FrameResult:
    persons: list[HMRPrediction]
    detector_ms: float
    hmr_ms: float


def keypoint_box(joints_2d: np.ndarray, w: int, h: int, pad: float = 0.15) -> np.ndarray:
    """A detector-like box from a person's 2D keypoints: their extent padded by `pad` of its size per side (the
    ground-truth box convention: on 3DPW the box's long side is 1.30x the keypoints' extent), shrunk to a
    detector's tighter framing, clipped to the frame."""
    k = joints_2d[BODY_KEYPOINTS]
    c, half = (k.min(0) + k.max(0)) / 2, (k.max(0) - k.min(0)) * (0.5 + pad) / DETECTOR_TO_GT_BOX
    return np.array([max(0.0, c[0] - half[0]), max(0.0, c[1] - half[1]),
                     min(float(w), c[0] + half[0]), min(float(h), c[1] + half[1])], np.float32)


class PosePipeline:
    """RF-DETR, then InstantHMR on every detected person.

    `model` is a released model name (nano, tiny, small, medium, medium-commercial) or an .onnx path; `fit` picks
    the network + kinematic-fit graph on 1 or 2 crops per person (0: the network alone). `detector_stride` > 1
    runs the detector every N frames; in between, each person's box follows its 2D keypoints of the previous
    frame. `pose_filter` ("one-euro" or "alpha-beta") filters each person's pose across frames.
    """

    def __init__(self, model: str | Path = DEFAULT_MODEL, fit: int = DEFAULT_FIT, device: str = "cuda",
                 detector: str = "medium", det_confidence: float = 0.5, max_persons: int = 4,
                 detector_stride: int = 1, focal: float | None = None, pose_filter: str | None = None):
        self.hmr = InstantHMR(resolve_model(model, fit), device=device, focal=focal, max_batch=max_persons,
                              pose_filter=PoseFilter(pose_filter) if pose_filter else None)
        self.detector = PersonDetector(detector, det_confidence, max_persons)
        self.detector_stride = max(1, detector_stride)
        self._frame = 0
        self._boxes = np.zeros((0, 4), np.float32)
        self._scores = np.zeros(0, np.float32)
        self.detector(np.zeros((288, 288, 3), np.uint8))
        self.hmr.warmup(max_persons)

    def __call__(self, image_rgb: np.ndarray, timestamp: float | None = None) -> FrameResult:
        t0 = time.perf_counter()
        if self._frame % self.detector_stride == 0:
            self._boxes, self._scores = self.detector(image_rgb)
        t1 = time.perf_counter()
        persons = self.hmr.predict(image_rgb, self._boxes, self._scores, timestamp=timestamp)
        t2 = time.perf_counter()
        h, w = image_rgb.shape[:2]
        self._boxes = np.array([keypoint_box(p.joints_2d, w, h) for p in persons], np.float32).reshape(-1, 4)
        self._frame += 1
        return FrameResult(persons, (t1 - t0) * 1000, (t2 - t1) * 1000)
