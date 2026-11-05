"""InstantHMR inference with onnxruntime: person crops in, MHR pose, 3D keypoints and mesh out."""
from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CROP_EXPAND = 1.2
# Focal used when the camera is not calibrated, as a multiple of the image diagonal
# (about 50 degrees of horizontal field of view on a 16:9 frame).
FOCAL_FALLBACK_DIAG = 1.05
DEVICES = ("cuda", "tensorrt", "cpu", "coreml")


@dataclass
class HMRPrediction:
    """One person. Pixel coordinates are in the full frame; 3D is in metres, camera axes
    (x right, y down, z forward), with `*_local` centred on the body and `*_cam` in camera space."""

    bbox: np.ndarray                    # (4,) x1, y1, x2, y2
    confidence: float
    joints_2d: np.ndarray               # (70, 2) pixels
    joints_3d_local: np.ndarray         # (70, 3)
    cam_trans: np.ndarray               # (3,)
    mhr_params: np.ndarray              # (204,) MHR model parameters
    shape_params: np.ndarray            # (45,) MHR identity coefficients
    focal_length: float                 # pixels, the camera the 3D is expressed in
    image_shape: tuple[int, int]        # (H, W)
    vertices_local: np.ndarray          # (18439, 3)

    @property
    def joints_3d_cam(self) -> np.ndarray:
        return self.joints_3d_local + self.cam_trans

    @property
    def vertices_cam(self) -> np.ndarray:
        return self.vertices_local + self.cam_trans


def providers_for(device: str, max_batch: int = 1, image_size: int = 288,
                  cache_dir: str | Path = "models/trt_cache") -> list:
    """onnxruntime execution providers for a device name, always ending on the CPU."""
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if device == "tensorrt":
        try:
            import tensorrt_libs  # noqa: F401  (loads libnvinfer for onnxruntime)
        except ImportError as e:
            raise RuntimeError("TensorRT is not installed: pip install tensorrt-cu12==10.15.*") from e
        if "TensorrtExecutionProvider" not in available:
            raise RuntimeError("this onnxruntime build has no TensorRT support: install onnxruntime-gpu")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        shape = lambda n: f"image:{n}x3x{image_size}x{image_size},cliff_cond:{n}x3"  # noqa: E731
        trt = {"trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": str(cache_dir),
               "trt_timing_cache_enable": True, "trt_profile_min_shapes": shape(1),
               "trt_profile_opt_shapes": shape(max_batch), "trt_profile_max_shapes": shape(max_batch)}
        return [("TensorrtExecutionProvider", trt), "CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDA is not available in this onnxruntime build: install onnxruntime-gpu")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "coreml" and "CoreMLExecutionProvider" in available:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class InstantHMR:
    """An exported InstantHMR network and the MHR body graph that turns its output into 3D.

    `device` runs the network ("cuda", "tensorrt", "cpu" or "coreml"); the body graph runs on
    CUDA when the network does, else on the CPU. `focal` is the camera focal length in pixels
    when it is known; otherwise 1.05 x the image diagonal is assumed. `max_batch` is the most
    people predicted in one call (TensorRT builds its engine for batches up to it).
    """

    def __init__(self, model_path: str | Path, body_path: str | Path | None = None, device: str = "cuda",
                 focal: float | None = None, max_batch: int = 4):
        import onnxruntime as ort

        from .models import resolve_body

        if device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}")
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        ort.set_default_logger_severity(3)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3

        size = 288
        if device == "tensorrt":   # the engine profile needs the crop size before the session exists
            probe = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            size = int(probe.get_modelmeta().custom_metadata_map.get("image_size", size))
            del probe
        self.network = ort.InferenceSession(str(model_path), sess_options=options,
                                            providers=providers_for(device, max_batch, size))
        self.provider = self.network.get_providers()[0]
        meta = self.network.get_modelmeta().custom_metadata_map
        self.input_size = int(meta.get("image_size", size))
        self.cliff_focal = meta.get("cliff_focal", "true") == "true"
        self.focal = focal
        self.max_batch = max_batch

        body_path = Path(body_path) if body_path else resolve_body(Path(model_path))
        self.body = ort.InferenceSession(str(body_path), sess_options=options,
                                         providers=providers_for("cuda" if device in ("cuda", "tensorrt") else "cpu"))
        self.faces = np.frombuffer(base64.b64decode(self.body.get_modelmeta().custom_metadata_map["mesh_faces"]),
                                   "<i4").reshape(-1, 3)

    def focal_px(self, h: int, w: int, focal: float | None = None) -> float:
        f = focal if focal is not None else self.focal
        if f is not None:
            return float(f)
        return (FOCAL_FALLBACK_DIAG if self.cliff_focal else 1.0) * math.hypot(h, w)

    def predict(self, image_rgb: np.ndarray, bboxes: Sequence, confidences: Sequence[float] | None = None,
                focal: float | None = None) -> list[HMRPrediction]:
        """All people of one frame in a single call. `bboxes` are tight person boxes, x1 y1 x2 y2."""
        if len(bboxes) == 0:
            return []
        h, w = image_rgb.shape[:2]
        f = self.focal_px(h, w, focal)
        crops, cliffs, squares = zip(*(self._preprocess(image_rgb, np.asarray(b, np.float32), f) for b in bboxes))
        crops, cliffs = np.stack(crops), np.stack(cliffs)
        chunks = [self._run(crops[i:i + self.max_batch], cliffs[i:i + self.max_batch])
                  for i in range(0, len(crops), self.max_batch)]
        outs = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        confidences = confidences if confidences is not None else [1.0] * len(bboxes)

        results = []
        for i, (x0, y0, side) in enumerate(squares):
            j2d = (outs["joints_2d"][i] + 1.0) * 0.5 * side + np.array([x0, y0], np.float32)
            results.append(HMRPrediction(
                bbox=np.asarray(bboxes[i], np.float32), confidence=float(confidences[i]),
                joints_2d=j2d.astype(np.float32), joints_3d_local=outs["joints_3d"][i],
                cam_trans=outs["cam_trans"][i], mhr_params=outs["mhr_params"][i],
                shape_params=outs["shape_params"][i], focal_length=f, image_shape=(h, w),
                vertices_local=outs["vertices"][i]))
        return results

    def _run(self, crops: np.ndarray, cliffs: np.ndarray) -> dict[str, np.ndarray]:
        names = ["mhr_params", "shape_params", "cam_trans", "joints_2d"]
        outs = dict(zip(names, self.network.run(names, {"image": crops, "cliff_cond": cliffs})))
        outs.update(zip(["joints_3d", "vertices"], self.body.run(
            ["joints_3d", "vertices"], {"mhr_params": outs["mhr_params"], "shape_params": outs["shape_params"]})))
        return outs

    def warmup(self, max_batch: int = 1) -> None:
        for n in sorted({1, max_batch}):
            self.network.run(None, {"image": np.zeros((n, 3, self.input_size, self.input_size), np.float32),
                                    "cliff_cond": np.zeros((n, 3), np.float32)})
            self.body.run(None, {"mhr_params": np.zeros((n, 204), np.float32),
                                 "shape_params": np.zeros((n, 45), np.float32)})

    def _preprocess(self, image_rgb: np.ndarray, bbox: np.ndarray, focal: float):
        """The square 1.2x crop the model was trained on, and its CLIFF conditioning vector."""
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = bbox.astype(np.float64)
        cx, cy, size = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)
        if self.cliff_focal:
            cliff = [math.atan((cx - w / 2) / focal), math.atan((cy - h / 2) / focal), size / focal]
        else:
            cliff = [2 * cx / w - 1, 2 * cy / h - 1, size / max(w, h)]

        side = size * CROP_EXPAND
        sx, sy = cx - side / 2, cy - side / 2
        ix1, iy1 = math.floor(sx), math.floor(sy)
        ix2, iy2 = math.ceil(sx + side), math.ceil(sy + side)
        patch = image_rgb[max(0, iy1):min(h, iy2), max(0, ix1):min(w, ix2)]
        patch = cv2.copyMakeBorder(patch, max(0, -iy1), max(0, iy2 - h), max(0, -ix1), max(0, ix2 - w),
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        crop = cv2.resize(patch, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        crop = ((crop.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
        return np.ascontiguousarray(crop), np.asarray(cliff, np.float32), (float(sx), float(sy), float(side))
