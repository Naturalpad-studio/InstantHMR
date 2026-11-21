#!/usr/bin/env python3
"""Pre-download the released InstantHMR ONNX graphs into the Hugging Face cache.

`demo.py` downloads what it needs on first use; this script just does it up front so the
first run is not slowed down by a 80-130 MB download, and so `setup_windows.bat` can
report progress. Files land in the usual HF cache (`~/.cache/huggingface/hub` on Linux,
`%USERPROFILE%\\.cache\\huggingface\\hub` on Windows).

    python tools/prefetch_models.py                          # medium-commercial, fit 2, + body
    python tools/prefetch_models.py --model medium --fit 0,1,2
    python tools/prefetch_models.py --model nano --no-body

Set HTTP_PROXY / HTTPS_PROXY (or run `setup_windows.bat --proxy <url>`) if Hugging Face
is not reachable directly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "momolesang/InstantHMR"
MODELS = ("nano", "tiny", "small", "medium", "medium-commercial")
BODY = "mhr-body.onnx"


def file_for(model: str, fit: int) -> str:
    base = f"instanthmr-{model}.onnx"
    return base if fit == 0 else base.replace(".onnx", f"-fit{fit}.onnx")


def fetch(name: str) -> Path:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(REPO_ID, name))
    mb = path.stat().st_size / 2**20
    print(f"[OK]   {name:<42} {mb:7.1f} MB  ->  {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="medium-commercial", choices=MODELS, help="model size (default: medium-commercial)")
    ap.add_argument("--fit", default="2", help="which fit graphs: 0 (network alone), 1, 2, or a comma list (default: 2)")
    ap.add_argument("--no-body", action="store_true", help="skip the MHR body graph (mhr-body.onnx)")
    args = ap.parse_args()

    fits = [int(x) for x in str(args.fit).replace(" ", "").split(",") if x]
    for fit in fits:
        if fit not in (0, 1, 2):
            ap.error(f"--fit must be 0, 1 or 2, got {fit}")

    total = 0
    for fit in fits:
        total += fetch(file_for(args.model, fit)).stat().st_size
    if not args.no_body:
        total += fetch(BODY).stat().st_size
    print(f"[DONE] downloaded / verified {len(fits) + (0 if args.no_body else 1)} file(s), {total / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
