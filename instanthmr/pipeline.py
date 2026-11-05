"""Person detection followed by InstantHMR, with per-stage timings."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .detector import PersonDetector
from .inference import HMRPrediction, InstantHMR
from .models import DEFAULT_MODEL, resolve_model


@dataclass
class FrameResult:
    persons: list[HMRPrediction]
    detector_ms: float
    hmr_ms: float


class PosePipeline:
    """RF-DETR, then InstantHMR on every detected person in one batched call.

    `model` is a released model name (nano, tiny, small, medium, medium-commercial) or an .onnx
    path. `detector_stride` > 1 runs the detector every N frames and reuses its boxes in between.
    """

    def __init__(self, model: str | Path = DEFAULT_MODEL, device: str = "cuda", detector: str = "medium",
                 det_confidence: float = 0.5, max_persons: int = 4, detector_stride: int = 1,
                 focal: float | None = None):
        self.hmr = InstantHMR(resolve_model(model), device=device, focal=focal, max_batch=max_persons)
        self.detector = PersonDetector(detector, det_confidence, max_persons)
        self.detector_stride = max(1, detector_stride)
        self._frame = 0
        self._boxes = np.zeros((0, 4), np.float32)
        self._scores = np.zeros(0, np.float32)
        self.detector(np.zeros((288, 288, 3), np.uint8))
        self.hmr.warmup(max_persons)

    def __call__(self, image_rgb: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()
        if self._frame % self.detector_stride == 0:
            self._boxes, self._scores = self.detector(image_rgb)
        t1 = time.perf_counter()
        persons = self.hmr.predict(image_rgb, self._boxes, self._scores)
        t2 = time.perf_counter()
        self._frame += 1
        return FrameResult(persons, (t1 - t0) * 1000, (t2 - t1) * 1000)
