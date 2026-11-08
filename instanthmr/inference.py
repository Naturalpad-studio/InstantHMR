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
# The kinematic-fit graphs' crop scales (0.9 and 0.7 of the box) were chosen on ground-truth person boxes; RF-DETR's
# boxes are tighter (median long side 0.866 of the ground-truth box), so they are enlarged by this factor first.
DETECTOR_TO_GT_BOX = 1 / 0.866
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
    """An exported InstantHMR graph and the MHR body graph that turns its output into 3D.

    The graph is either a network (`instanthmr-<size>.onnx`) or a network + kinematic fit
    (`instanthmr-<size>-fit<1|2>.onnx`, read from its metadata), which refines the pose on 1 or 2 crops per
    person. `device` runs it ("cuda", "tensorrt", "cpu" or "coreml"); the body graph runs on CUDA when the
    graph does, else on the CPU. `focal` is the camera focal length in pixels when it is known; otherwise
    1.05 x the image diagonal is assumed. `max_batch` is the most people predicted in one call. `pose_filter`
    (smoothing.PoseFilter) filters each person's pose across frames before the mesh is built.
    """

    def __init__(self, model_path: str | Path, body_path: str | Path | None = None, device: str = "cuda",
                 focal: float | None = None, max_batch: int = 4, pose_filter=None):
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

        probe = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        meta = probe.get_modelmeta().custom_metadata_map
        self.input_size = int(meta.get("image_size", 288))
        self.cliff_focal = meta.get("cliff_focal", "true") == "true"
        self.n_crops = int(meta.get("crops", 0))            # 0: a network graph; 1 or 2: network + kinematic fit
        self.scales = tuple(float(v) for v in meta["crop_scales"].split(",")) if self.n_crops else (1.0,)
        del probe
        if self.n_crops:
            providers = _fit_providers(device, Path(model_path).stem)
        else:
            providers = providers_for(device, max_batch, self.input_size)
        self.network = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
        self.provider = self.network.get_providers()[0]
        self.focal = focal
        self.max_batch = max_batch
        self.pose_filter = pose_filter
        self._binding = None
        if self.n_crops and self.provider == "CUDAExecutionProvider":
            self._bind_cuda_graph(ort)

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
                focal: float | None = None, timestamp: float | None = None) -> list[HMRPrediction]:
        """All people of one frame. `bboxes` are tight person boxes (as a detector gives them), x1 y1 x2 y2;
        `timestamp` (seconds) is only read by the pose filter."""
        if len(bboxes) == 0:
            return []
        h, w = image_rgb.shape[:2]
        f = self.focal_px(h, w, focal)
        boxes = [np.asarray(b, np.float32) for b in bboxes]
        outs = self._estimate_fit(image_rgb, boxes, f) if self.n_crops else self._estimate(image_rgb, boxes, f)
        if self.pose_filter is not None:
            outs["mhr_params"] = self.pose_filter(boxes, outs["mhr_params"], timestamp)
        joints_3d, vertices = self.body.run(["joints_3d", "vertices"], {"mhr_params": outs["mhr_params"],
                                                                        "shape_params": outs["shape_params"]})
        confidences = confidences if confidences is not None else [1.0] * len(boxes)
        return [HMRPrediction(bbox=boxes[i], confidence=float(confidences[i]), joints_2d=outs["joints_2d"][i],
                              joints_3d_local=joints_3d[i], cam_trans=outs["cam_trans"][i],
                              mhr_params=outs["mhr_params"][i], shape_params=outs["shape_params"][i],
                              focal_length=f, image_shape=(h, w), vertices_local=vertices[i])
                for i in range(len(boxes))]

    def _estimate(self, image_rgb, boxes, f) -> dict[str, np.ndarray]:
        """The network graph: one crop per person, batches of up to `max_batch`; 2D in pixels."""
        names = ["mhr_params", "shape_params", "cam_trans", "joints_2d"]
        crops, cliffs, squares = zip(*(self._crop(image_rgb, b, 1.0, f) for b in boxes))
        crops = np.stack([_normalise(c) for c in crops])
        cliffs = np.stack(cliffs)
        chunks = [self.network.run(names, {"image": crops[i:i + self.max_batch], "cliff_cond": cliffs[i:i + self.max_batch]})
                  for i in range(0, len(crops), self.max_batch)]
        outs = {k: np.concatenate([c[j] for c in chunks]) for j, k in enumerate(names)}
        sq = np.asarray(squares, np.float32)                                    # (P, 4) x1, y1, w, h
        outs["joints_2d"] = (outs["joints_2d"] + 1.0) * 0.5 * sq[:, None, 2:] + sq[:, None, :2]
        return outs

    def _estimate_fit(self, image_rgb, boxes, f) -> dict[str, np.ndarray]:
        """The network + fit graph: one call per person on its 1 or 2 crops; 2D in pixels."""
        h, w = image_rgb.shape[:2]
        cam = np.array([[f, w / 2.0, h / 2.0]], np.float32)
        per_person = []
        for b in boxes:
            c, half = (b[:2] + b[2:]) / 2, (b[2:] - b[:2]) * DETECTOR_TO_GT_BOX / 2
            gt_box = np.concatenate([c - half, c + half])
            crops, cliffs, squares = zip(*(self._crop(image_rgb, gt_box, s, f) for s in self.scales))
            per_person.append(self._run_fit({"image": np.stack(crops), "cliff_cond": np.stack(cliffs),
                                             "crop_box": np.asarray(squares, np.float32), "cam": cam}))
        return {k: np.concatenate([o[k] for o in per_person]) for k in per_person[0]}

    def _run_fit(self, feeds: dict) -> dict[str, np.ndarray]:
        names = ["mhr_params", "shape_params", "cam_trans", "joints_2d"]
        if self._binding is None:
            return dict(zip(names, self.network.run(names, feeds)))
        for k, v in feeds.items():
            self._inputs[k].update_inplace(np.ascontiguousarray(v))
        self.network.run_with_iobinding(self._binding)
        return dict(zip(names, (o.numpy() for o in self._binding.get_outputs())))

    def _bind_cuda_graph(self, ort) -> None:
        """ORT CUDA: fixed device buffers for every input and output, so the first run is captured as one CUDA
        graph and every later run replays it."""
        n, s = self.n_crops, self.input_size
        shapes = {"image": ((n, s, s, 3), np.uint8), "cliff_cond": ((n, 3), np.float32),
                  "crop_box": ((n, 4), np.float32), "cam": ((1, 3), np.float32)}
        self._inputs = {k: ort.OrtValue.ortvalue_from_numpy(np.zeros(*v), "cuda", 0) for k, v in shapes.items()}
        self._binding = self.network.io_binding()
        for k, v in self._inputs.items():
            self._binding.bind_ortvalue_input(k, v)
        for k in ("mhr_params", "shape_params", "cam_trans", "joints_2d"):
            self._binding.bind_output(k, "cuda", 0)

    def warmup(self, max_batch: int = 1) -> None:
        image, pose_filter, self.pose_filter = np.zeros((480, 640, 3), np.uint8), self.pose_filter, None
        try:
            for n in sorted({1, max_batch}):
                self.predict(image, [[200, 100, 400, 450]] * n)
        finally:
            self.pose_filter = pose_filter

    def _crop(self, image_rgb: np.ndarray, bbox: np.ndarray, scale: float, focal: float):
        """The square 1.2x crop of the box scaled by `scale` about its centre, as uint8, cut and resized by one
        affine warp (the same sampling as cropping then resizing); its CLIFF vector and pixel square."""
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = bbox.astype(np.float64)
        cx, cy, size = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * scale
        if self.cliff_focal:
            cliff = [math.atan((cx - w / 2) / focal), math.atan((cy - h / 2) / focal), size / focal]
        else:
            cliff = [2 * cx / w - 1, 2 * cy / h - 1, size / max(w, h)]
        side = size * CROP_EXPAND
        ix1, iy1 = math.floor(cx - side / 2), math.floor(cy - side / 2)
        ix2, iy2 = math.ceil(cx - side / 2 + side), math.ceil(cy - side / 2 + side)
        s = self.input_size
        sx, sy = (ix2 - ix1) / s, (iy2 - iy1) / s
        m = np.array([[sx, 0, ix1 + 0.5 * sx - 0.5], [0, sy, iy1 + 0.5 * sy - 0.5]], np.float64)
        crop = cv2.warpAffine(image_rgb, m, (s, s), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return crop, np.asarray(cliff, np.float32), (float(ix1), float(iy1), float(ix2 - ix1), float(iy2 - iy1))


def _normalise(crop: np.ndarray) -> np.ndarray:
    """uint8 HWC -> ImageNet-normalised float32 CHW, the network graphs' input."""
    return np.ascontiguousarray(((crop.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))


def _fit_providers(device: str, name: str, cache_dir: str | Path = "models/trt_cache") -> list:
    """Providers for a fit graph (fixed shapes): TensorRT in fp16 with a cached engine, or ORT CUDA with CUDA graphs."""
    if device == "tensorrt":
        providers = providers_for(device)                 # checks the installation
        cache = Path(cache_dir) / name
        cache.mkdir(parents=True, exist_ok=True)
        return [("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                               "trt_engine_cache_path": str(cache), "trt_timing_cache_enable": True,
                                               "trt_timing_cache_path": str(cache)}), *providers[1:]]
    if device == "cuda":
        providers_for(device)
        return [("CUDAExecutionProvider", {"enable_cuda_graph": "1"}), "CPUExecutionProvider"]
    return providers_for(device)
