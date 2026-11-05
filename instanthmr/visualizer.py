"""Rerun viewer: the frame with 2D skeletons and timings, the 3D scene, and a latency plot."""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from .inference import HMRPrediction
from .skeleton import SKELETON_EDGES

MESH_COLOR = (200, 160, 130)
JOINT_COLOR = (0, 230, 0)
BONE_COLOR = (255, 230, 0)
BOX_COLOR = (0, 200, 255)


class RerunVisualizer:
    def __init__(self, faces: Optional[np.ndarray] = None, mesh_alpha: float = 0.35, spawn: bool = True,
                 save_path: Optional[str] = None):
        import rerun as rr

        self.rr = rr
        self.faces = faces
        self.albedo = (*MESH_COLOR, int(round(255 * min(max(mesh_alpha, 0.0), 1.0))))
        rr.init("instanthmr", spawn=spawn)
        if save_path:
            rr.save(save_path)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        self._n_logged = 0
        self._layout_sent = False

    def log(self, frame: int, image_rgb: np.ndarray, persons: Sequence[HMRPrediction],
            timings: dict[str, float]) -> None:
        rr = self.rr
        rr.set_time("frame", sequence=frame)
        for name, ms in timings.items():
            rr.log(f"timing/{name}", rr.Scalars(float(ms)))
        for stale in range(len(persons), self._n_logged):
            rr.log(f"world/person_{stale}", rr.Clear(recursive=True))
        self._n_logged = len(persons)

        for i, p in enumerate(persons):
            joints = p.joints_3d_cam
            rr.log(f"world/person_{i}/joints", rr.Points3D(joints, radii=0.012, colors=JOINT_COLOR))
            rr.log(f"world/person_{i}/skeleton",
                   rr.LineStrips3D(joints[np.array(SKELETON_EDGES)], colors=BONE_COLOR, radii=0.004))
            if self.faces is not None and p.vertices_local is not None:
                rr.log(f"world/person_{i}/mesh", rr.Mesh3D(vertex_positions=p.vertices_cam,
                                                           triangle_indices=self.faces, albedo_factor=self.albedo))

        h, w = image_rgb.shape[:2]
        rr.log("camera/image", rr.Image(draw_overlay(image_rgb, persons, timings)))
        if persons:
            depth = float(np.median([p.cam_trans[2] for p in persons]))
            rr.log("camera", rr.Pinhole(width=w, height=h, focal_length=persons[0].focal_length,
                                        principal_point=[w / 2, h / 2], image_plane_distance=max(0.25 * depth, 0.2)))
            if not self._layout_sent:
                self._send_layout(persons[0])
                self._layout_sent = True

    def _send_layout(self, first: HMRPrediction) -> None:
        import rerun.blueprint as rrb

        centre = first.joints_3d_cam.mean(0)
        distance = max(4 * float(np.abs(first.joints_3d_cam - centre).max()), 1.5)
        self.rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
            rrb.Vertical(rrb.Spatial2DView(origin="camera/image", name="Camera"),
                         rrb.TimeSeriesView(origin="timing", name="Latency (ms)"), row_shares=[3, 1]),
            rrb.Spatial3DView(origin="/", name="3D", background=rrb.Background(color=[25, 25, 25]),
                              eye_controls=rrb.EyeControls3D(
                                  position=[float(centre[0]), float(centre[1]), float(centre[2] - distance)],
                                  look_target=centre.tolist(), eye_up=[0.0, -1.0, 0.0], kind=rrb.Eye3DKind.Orbital)),
            column_shares=[2, 3])))


def draw_overlay(image_rgb: np.ndarray, persons: Sequence[HMRPrediction], timings: dict[str, float]) -> np.ndarray:
    """The frame with each person's box and 2D skeleton, and the stage timings top left."""
    out = image_rgb.copy()
    h, w = out.shape[:2]
    thick = max(2, round(min(h, w) / 400))
    for p in persons:
        x1, y1, x2, y2 = p.bbox.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, thick)
        pts = p.joints_2d.astype(int)
        for a, b in SKELETON_EDGES:
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), BONE_COLOR, thick, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(out, tuple(pt), thick + 1, JOINT_COLOR, -1, cv2.LINE_AA)

    lines = [f"{name:<9s}{ms:6.1f} ms" for name, ms in timings.items()]
    total = sum(timings.values())
    if total > 0:
        lines.append(f"{'total':<9s}{total:6.1f} ms ({1000 / total:.0f} fps)")
    scale = max(0.5, min(h, w) / 1200)
    line_h = int(28 * scale)
    panel = out.copy()
    cv2.rectangle(panel, (8, 8), (int(300 * scale) + 16, 16 + line_h * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.55, out, 0.45, 0, out)
    for k, line in enumerate(lines):
        cv2.putText(out, line, (16, 8 + line_h * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7,
                    (255, 255, 255), max(1, round(scale * 1.5)), cv2.LINE_AA)
    return out
