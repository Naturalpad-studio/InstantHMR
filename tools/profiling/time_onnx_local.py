#!/usr/bin/env python3
"""Model-only latency of exported InstantHMR graphs on this machine.

Batch 1, the graph alone (no detector, no MHR forward pass, no crop), ONNX
Runtime with the CPU provider (all cores, then one thread) and the CUDA
provider. Median and p90 over --iters runs after --warmup. The input size is
read from the ONNX metadata, as instanthmr.inference does. Run it on an idle
machine: anything else on the GPU or the cores shows up in the numbers.

    python tools/profiling/time_onnx_local.py models/family/*.onnx --out bench.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch  # noqa: F401  loads the cuDNN onnxruntime's CUDA provider needs; without it
import onnxruntime as ort  # the provider fails to load and ORT silently runs on the CPU


def time_session(path: str, providers, threads: int | None, warmup: int, iters: int):
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(path, so, providers=providers)
    if sess.get_providers()[0] != providers[0]:
        raise SystemExit(f"{providers[0]} did not load (got {sess.get_providers()}); "
                         "refusing to report a CPU fallback as a GPU number")
    size = int(sess.get_modelmeta().custom_metadata_map.get("image_size", 224))
    feeds = {"image": np.random.randn(1, 3, size, size).astype(np.float32),
             "cliff_cond": np.zeros((1, 3), dtype=np.float32)}
    for _ in range(warmup):
        sess.run(None, feeds)
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        t.append((time.perf_counter() - t0) * 1e3)
    return size, float(np.median(t)), float(np.percentile(t, 90))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("onnx", nargs="+")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    modes = [("cpu_all", ["CPUExecutionProvider"], None),
             ("cpu_1t", ["CPUExecutionProvider"], 1),
             ("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"], None)]
    rows = []
    print(f"{'model':<20s} {'in':>4s}" + "".join(f" {m:>16s}" for m, _, _ in modes))
    for path in args.onnx:
        row = {"model": Path(path).stem}
        for m, prov, th in modes:
            size, med, p90 = time_session(path, prov, th, args.warmup, args.iters)
            row["image_size"], row[m] = size, {"median_ms": med, "p90_ms": p90}
        rows.append(row)
        print(f"{row['model']:<20s} {row['image_size']:>4d}"
              + "".join(f" {row[m]['median_ms']:8.2f} ({row[m]['p90_ms']:5.1f})"
                        for m, _, _ in modes), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(dict(
            machine=dict(cpu=platform.processor() or platform.machine(),
                         onnxruntime=ort.__version__),
            rows=rows), indent=2))


if __name__ == "__main__":
    main()
