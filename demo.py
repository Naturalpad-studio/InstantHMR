#!/usr/bin/env python3
"""InstantHMR demo: people in an image, a video or a live camera, shown in the Rerun viewer.

    python demo.py --image photo.jpg
    python demo.py --video clip.mp4 --pose-filter one-euro
    python demo.py --camera 0 --model small --fit 1 --device tensorrt
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from instanthmr import DEFAULT_FIT, DEFAULT_MODEL, MODELS, PosePipeline
from instanthmr.smoothing import TemporalSmoother
from instanthmr.visualizer import RerunVisualizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="image file")
    src.add_argument("--video", help="video file")
    src.add_argument("--camera", type=int, metavar="INDEX", help="live camera, e.g. 0")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"{', '.join(MODELS)}, or a path to an .onnx file (default: {DEFAULT_MODEL})")
    p.add_argument("--fit", type=int, default=DEFAULT_FIT, choices=(0, 1, 2),
                   help="kinematic fit on 1 or 2 crops per person, 0 for the network alone (default: "
                        f"{DEFAULT_FIT}; ignored for an .onnx path)")
    p.add_argument("--device", default="cuda", choices=("cuda", "tensorrt", "cpu", "coreml"))
    p.add_argument("--max-persons", type=int, default=4)
    p.add_argument("--detector", default="medium", choices=("nano", "small", "medium", "base", "large"),
                   help="RF-DETR size (default: medium)")
    p.add_argument("--det-confidence", type=float, default=0.5)
    p.add_argument("--detector-stride", type=int, default=1, metavar="N",
                   help="run the detector every N frames and reuse its boxes in between")
    p.add_argument("--focal", type=float, help="camera focal length in pixels, if known")
    p.add_argument("--pose-filter", choices=("one-euro", "alpha-beta"),
                   help="video: filter each person's pose across frames (more accurate and steadier)")
    p.add_argument("--smooth", action="store_true",
                   help="video: smooth the keypoints and mesh instead (steadiest, but lags on fast motion)")
    p.add_argument("--no-mesh", action="store_true", help="show the skeleton only")
    p.add_argument("--mesh-alpha", type=float, default=0.35, help="mesh opacity (default: 0.35)")
    p.add_argument("--save-rrd", metavar="PATH", help="also save the Rerun recording")
    p.add_argument("--no-viewer", action="store_true", help="do not open the Rerun viewer")
    return p.parse_args()


def frames(args):
    """(index, timestamp in s, RGB frame) from the chosen source."""
    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            sys.exit(f"cannot read image {args.image}")
        yield 0, 0.0, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video or f'camera {args.camera}'}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start, i = time.perf_counter(), 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                return
            t = i / fps if args.video else time.perf_counter() - start
            yield i, t, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()


def main() -> None:
    args = parse_args()
    fit = f", kinematic fit on {args.fit} crop{'s' if args.fit > 1 else ''}" if args.fit else ""
    print(f"loading InstantHMR ({args.model}{fit}) on {args.device} and RF-DETR {args.detector}...")
    pipeline = PosePipeline(args.model, fit=args.fit, device=args.device, detector=args.detector,
                            det_confidence=args.det_confidence, max_persons=args.max_persons,
                            detector_stride=args.detector_stride, focal=args.focal, pose_filter=args.pose_filter)
    print(f"InstantHMR runs on {pipeline.hmr.provider}")
    viewer = RerunVisualizer(None if args.no_mesh else pipeline.hmr.faces, args.mesh_alpha,
                             spawn=not args.no_viewer, save_path=args.save_rrd)
    smoother = TemporalSmoother() if args.smooth else None

    history = []
    try:
        for i, t, rgb in frames(args):
            result = pipeline(rgb, t)
            persons = smoother(result.persons, t) if smoother else result.persons
            timings = {"detector": result.detector_ms, "InstantHMR": result.hmr_ms}
            viewer.log(i, rgb, persons, timings)
            history.append((result.detector_ms, result.hmr_ms))
            if i % 30 == 0:
                print(f"frame {i:5d}  {len(persons)} person(s)  detector {result.detector_ms:5.1f} ms  "
                      f"InstantHMR {result.hmr_ms:5.1f} ms")
    except KeyboardInterrupt:
        pass
    if len(history) > 1:
        det, hmr = np.median(np.array(history[1:]), axis=0)
        print(f"median over {len(history) - 1} frames: detector {det:.1f} ms, InstantHMR {hmr:.1f} ms")


if __name__ == "__main__":
    main()
