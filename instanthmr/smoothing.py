"""Temporal filters for video, all causal (they work on a live camera). People are matched across frames by box
overlap.

`PoseFilter` filters the MHR pose (root rotation and body angles) before the mesh is built, so the mesh, the 3D
keypoints and the pose stay consistent. Its two settings were tuned on 3DPW validation and 56 EgoHumans takes and
checked on EMDB-1 and 3DPW: both lower PA-MPJPE by 0.1-0.3 mm and the jitter by about 27 % (`alpha-beta`) or 35 %
(`one-euro`), with under half a frame of lag.

`TemporalSmoother` smooths the outputs themselves (keypoints, mesh, translation) with the 1 Euro filter (Casiez,
Roussel and Vogel, CHI 2012): smoother still, at the cost of some accuracy and visible lag on fast motion.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from .inference import HMRPrediction

# (mincutoff in Hz, beta) per field, in the field's own units: pixels for joints_2d, metres for 3D.
FIELDS = {
    "joints_2d": (1.0, 0.01),
    "joints_3d_local": (1.0, 4.0),
    "vertices_local": (1.0, 4.0),
    "cam_trans": (0.6, 2.0),
}
POSE = slice(3, 68)          # MHR parameters: root rotation (3:6) and the body angles (6:68), in radians


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _alpha(cutoff, dt):
    return 1.0 / (1.0 + 1.0 / (2 * np.pi * np.maximum(cutoff, 1e-6) * dt))


class _OneEuro:
    """1 Euro filter; `angles` wraps each residual to (-pi, pi], so an angle crossing +/-pi is not a 2 pi jump."""

    def __init__(self, mincutoff: float, beta: float, dcutoff: float = 1.0, angles: bool = False):
        self.mincutoff, self.beta, self.dcutoff, self.angles = mincutoff, beta, dcutoff, angles
        self.x = self.dx = self.t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.x is None:
            self.x, self.dx, self.t = x.astype(np.float64), np.zeros(x.shape), t
            return x
        dt = t - self.t if t > self.t else 1 / 30
        self.t = t
        r = x - self.x
        if self.angles:
            r = _wrap(r)
        a_d = _alpha(self.dcutoff, dt)
        self.dx = a_d * r / dt + (1 - a_d) * self.dx
        self.x = self.x + _alpha(self.mincutoff + self.beta * np.abs(self.dx), dt) * r
        return self.x.astype(np.float32)


class _AlphaBeta:
    """Constant-velocity filter on angles, one step per frame: predict x + v, correct by `a` and `b` times the
    wrapped residual."""

    def __init__(self, a: float, b: float):
        self.a, self.b = a, b
        self.x = self.v = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.x is None:
            self.x, self.v = x.astype(np.float64), np.zeros(x.shape)
            return x
        pred = self.x + self.v
        r = _wrap(x - pred)
        self.x, self.v = pred + self.a * r, self.v + self.b * r
        return self.x.astype(np.float32)


POSE_FILTERS = {
    "one-euro": lambda: _OneEuro(mincutoff=12.0, beta=0.5, dcutoff=1.0, angles=True),
    "alpha-beta": lambda: _AlphaBeta(a=0.8, b=0.005),
}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


class _Tracks:
    """Matches boxes to the previous frame's by overlap; a track survives `max_age` frames without a match."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5):
        self.iou_threshold, self.max_age = iou_threshold, max_age
        self.tracks: dict[int, dict] = {}
        self._next_id = 0

    def assign(self, boxes: Sequence[np.ndarray]) -> list[dict]:
        """The state dict of each box's track (new tracks start empty)."""
        pairs = sorted(((_iou(b, tr["box"]), i, t) for i, b in enumerate(boxes)
                        for t, tr in self.tracks.items()), reverse=True)
        ids, used = [-1] * len(boxes), set()
        for score, i, t in pairs:
            if score >= self.iou_threshold and ids[i] == -1 and t not in used:
                ids[i] = t
                used.add(t)
        for i, b in enumerate(boxes):
            if ids[i] == -1:
                ids[i] = self._next_id
                self.tracks[ids[i]] = {"filters": {}}
                self._next_id += 1
            self.tracks[ids[i]].update(box=np.asarray(b), age=0)
        for t in set(self.tracks) - set(ids):
            self.tracks[t]["age"] += 1
            if self.tracks[t]["age"] > self.max_age:
                del self.tracks[t]
        return [self.tracks[t] for t in ids]


class PoseFilter:
    """`pose_filter(boxes, mhr_params, timestamp)` returns the parameters with the pose filtered per person.

    `kind` is "one-euro" or "alpha-beta". `timestamp` is in seconds; without one, frames are taken 1/30 s apart.
    """

    def __init__(self, kind: str = "one-euro"):
        if kind not in POSE_FILTERS:
            raise ValueError(f"unknown pose filter {kind!r}: choose one of {', '.join(POSE_FILTERS)}")
        self.kind = kind
        self._tracks = _Tracks()
        self._clock = 0.0

    def __call__(self, boxes: Sequence[np.ndarray], mhr_params: np.ndarray, timestamp: float | None = None) -> np.ndarray:
        if timestamp is None:
            self._clock += 1 / 30
            timestamp = self._clock
        out = np.array(mhr_params, np.float32, copy=True)
        for i, track in enumerate(self._tracks.assign(boxes)):
            f = track["filters"].setdefault("pose", POSE_FILTERS[self.kind]())
            out[i, POSE] = f(out[i, POSE], timestamp)
        return out


class TemporalSmoother:
    """`smoother(persons, timestamp)` returns the same people with smoothed fields.

    `beta_scale` trades jitter for lag: raise it if fast motion lags, lower it if a still person
    shakes.
    """

    def __init__(self, beta_scale: float = 1.0, iou_threshold: float = 0.3, max_age: int = 5):
        self.beta_scale = beta_scale
        self._tracks = _Tracks(iou_threshold, max_age)

    def __call__(self, persons: Sequence[HMRPrediction], timestamp: float) -> list[HMRPrediction]:
        out = []
        for track, person in zip(self._tracks.assign([p.bbox for p in persons]), persons):
            fields = {}
            for name, (mincutoff, beta) in FIELDS.items():
                value = getattr(person, name)
                if value is None:
                    continue
                f = track["filters"].setdefault(name, _OneEuro(mincutoff, beta * self.beta_scale))
                fields[name] = f(value, timestamp)
            out.append(replace(person, **fields))
        return out
