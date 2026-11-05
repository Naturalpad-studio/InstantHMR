"""Temporal smoothing of per-person predictions with the 1 Euro filter.

InstantHMR estimates every frame on its own, so a video shows per-frame jitter. The 1 Euro filter
(Casiez, Roussel and Vogel, CHI 2012) smooths each coordinate with a cutoff that rises with its
own speed: still joints are smoothed hard and fast ones barely, so it adds no visible lag. It is
causal, so it works on a live camera. People are matched across frames by box overlap.
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


class _OneEuro:
    def __init__(self, mincutoff: float, beta: float, dcutoff: float = 1.0):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2 * np.pi * np.maximum(cutoff, 1e-6) * dt))

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.x is None:
            self.x, self.dx, self.t = x.astype(np.float64), np.zeros(x.shape), t
            return x
        dt = t - self.t if t > self.t else 1 / 30
        self.t = t
        a_d = self._alpha(self.dcutoff, dt)
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        a = self._alpha(self.mincutoff + self.beta * np.abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        return self.x.astype(np.float32)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


class TemporalSmoother:
    """`smoother(persons, timestamp)` returns the same people with smoothed fields.

    `beta_scale` trades jitter for lag: raise it if fast motion lags, lower it if a still person
    shakes. A track survives `max_age` frames without a match before its filter is reset.
    """

    def __init__(self, beta_scale: float = 1.0, iou_threshold: float = 0.3, max_age: int = 5):
        self.beta_scale, self.iou_threshold, self.max_age = beta_scale, iou_threshold, max_age
        self._tracks: dict[int, dict] = {}
        self._next_id = 0

    def __call__(self, persons: Sequence[HMRPrediction], timestamp: float) -> list[HMRPrediction]:
        ids = self._assign(persons)
        out = []
        for track_id, person in zip(ids, persons):
            filters = self._tracks[track_id]["filters"]
            fields = {}
            for name, (mincutoff, beta) in FIELDS.items():
                value = getattr(person, name)
                if value is None:
                    continue
                f = filters.setdefault(name, _OneEuro(mincutoff, beta * self.beta_scale))
                fields[name] = f(value, timestamp)
            out.append(replace(person, **fields))
        return out

    def _assign(self, persons: Sequence[HMRPrediction]) -> list[int]:
        pairs = sorted(((_iou(p.bbox, tr["box"]), i, t) for i, p in enumerate(persons)
                        for t, tr in self._tracks.items()), reverse=True)
        ids, used = [-1] * len(persons), set()
        for score, i, t in pairs:
            if score >= self.iou_threshold and ids[i] == -1 and t not in used:
                ids[i] = t
                used.add(t)
        for i, p in enumerate(persons):
            if ids[i] == -1:
                ids[i] = self._next_id
                self._tracks[ids[i]] = {"filters": {}}
                self._next_id += 1
            self._tracks[ids[i]].update(box=p.bbox, age=0)
        for t in set(self._tracks) - set(ids):
            self._tracks[t]["age"] += 1
            if self._tracks[t]["age"] > self.max_age:
                del self._tracks[t]
        return ids
