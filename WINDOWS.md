# InstantHMR on Windows

End-to-end for Windows + an NVIDIA GPU: one-click setup → download the models → run images,
videos and the webcam with the three `.bat` launchers. Measured numbers and the tuning story
behind them are in [`RUN.md`](RUN.md); the Chinese version of this page is
[`WINDOWS.zh.md`](WINDOWS.zh.md).

> Three commands are enough:
> ```bat
> setup_windows.bat
> run_image.bat assets\test_zidane.jpg
> run_video.bat D:\videos\clip.mp4
> ```

---

## 1. Requirements

| | |
| --- | --- |
| OS | Windows 10/11 x64 |
| Python | 3.10 – 3.12 (**3.12 recommended**, started through `py -3.12`; do not let a system-wide 3.13/3.14 be the default) |
| GPU | NVIDIA, driver 525+ (tested on an RTX 3090 with CUDA 12.6) |
| Disk | ~5 GB free (torch in the venv is ~3 GB, the models ~0.3 GB) |
| Network | Hugging Face and PyPI on first run; use a proxy or a mirror if they are slow |

---

## 2. One-click setup: `setup_windows.bat`

Double-click it, or from a shell:

```bat
setup_windows.bat                                    :: everything: venv + packages + models
setup_windows.bat --check                            :: report only, install nothing
setup_windows.bat --proxy http://127.0.0.1:21882     :: route pip and HF through a proxy
setup_windows.bat --model medium --fit 0,1,2         :: another model size / several fit graphs
```

It does four things:

1. creates `.venv` with `py -3.12 -m venv .venv` (reuses an existing one);
2. installs the **CUDA** torch/torchvision from
   `https://download.pytorch.org/whl/cu126`, falling back to CPU wheels with a warning — the
   person detector then runs on the CPU and is roughly 4x slower;
3. installs `requirements.txt` (onnxruntime-gpu, rfdetr, rerun-sdk, huggingface_hub, ...);
4. runs `tools\prefetch_models.py` to pull the ONNX graphs into the Hugging Face cache.

It ends with a report, e.g. `torch 2.9.1+cu126 | cuda: True NVIDIA GeForce RTX 3090` and the
onnxruntime providers (they should include `CUDAExecutionProvider`). It is safe to re-run;
`--check` tells you whether the current environment is usable.

---

## 3. Models: what gets downloaded, and where

The demo downloads what it needs from
[Hugging Face `momolesang/InstantHMR`](https://huggingface.co/momolesang/InstantHMR) into the
standard cache (`%USERPROFILE%\.cache\huggingface\hub`), so nothing has to be placed in
`models\` by hand. `setup_windows.bat` / `tools\prefetch_models.py` just do it up front.

| File | Use | Size | Licence |
| --- | --- | --- | --- |
| `instanthmr-medium-commercial.onnx` | default model, network alone | 123.4 MB | commercial use allowed |
| `instanthmr-medium-commercial-fit1.onnx` | `--fit 1`, kinematic fit on one crop | 79.3 MB | commercial use allowed |
| `instanthmr-medium-commercial-fit2.onnx` | `--fit 2`, two crops, steadiest root joint (**recommended**) | 79.4 MB | commercial use allowed |
| `mhr-body.onnx` | MHR body graph, decodes pose to an 18,439-vertex mesh | 12.7 MB (12.1 MiB) | LICENSE-SAM, as the repo |
| `instanthmr-medium.onnx` / `-fit1` / `-fit2` | more accurate, non-commercial | 93.7 / 79.3 / 79.4 MB | **SAM License, non-commercial** |
| `instanthmr-{nano,tiny,small}*.onnx` | smaller and faster, less accurate | 56 – 65 MB | non-commercial |

Notes:

- A `-fitN` graph **must be used with `--fit N`**; mixing them fails or gives wrong meshes.
- For commercial work only `medium-commercial` is available; every other size is SAM License,
  non-commercial.
- PyTorch weights (fine-tuning): `pytorch\medium-commercial\instanthmr-medium-commercial.pth`
  (129.4 MB) plus `run_config.json` beside it, then export with
  `python tools\export_onnx.py --ckpt <pth> --output models\instanthmr-medium-commercial.onnx`
  (`--fit 1|2` also needs `training\checkpoints\mhr_model.pt`).
- Manual download works too: `https://huggingface.co/momolesang/InstantHMR/resolve/main/<file>`;
  drop it into `models\` and pass `--model models\xxx.onnx`.

---

## 4. The three launchers

All three put the venv's `nvidia\{cudnn,cublas,cuda_runtime,cufft,curand,cuda_nvrtc,nvjitlink}\bin`
directories on `PATH` (the CUDA EP loads those sub-libraries dynamically, otherwise you get
"DLL not found"), check that the model exists and then call `demo.py`. Extra arguments are
passed straight through.

### 4.1 `run_image.bat` — one image

```bat
run_image.bat                                   :: no argument = assets\test_zidane.jpg
run_image.bat D:\photos\a.jpg
run_image.bat D:\photos\a.jpg --no-mesh         :: skeleton only
run_image.bat D:\photos\a.jpg --detector medium --max-persons 2
```

Defaults: `--fit 0` (a single frame has no temporal context to refine), `--detector nano`,
`--device cuda`.

### 4.2 `run_video.bat` — a local video

```bat
run_video.bat D:\videos\clip.mp4
run_video.bat clip.mp4 --smooth
run_video.bat clip.mp4 --focal 880                :: phone 26 mm equivalent, more realistic depth
run_video.bat clip.mp4 --pose-filter alpha-beta
run_video.bat clip.mp4 --no-viewer --save-rrd assets\demo.rrd
```

Defaults: `medium-commercial-fit2` when present, `--fit 2`, `--pose-filter one-euro --smooth` —
the steadiest root-joint combination. Measured here on a 720×1280 phone clip: **~70 ms per frame
(about 14 fps)**.

### 4.3 `run_camera.bat` — the webcam

```bat
run_camera.bat            :: camera 0
run_camera.bat 1          :: second camera
run_camera.bat 0 --detector nano --max-persons 1
```

Defaults to `--detector-stride 3` (detect every third frame, follow the 2D keypoints in between);
the capture loop runs at ~24 fps. `Ctrl+C` in the window stops it.

### 4.4 Changing the defaults without editing the scripts

```bat
set HMR_MODEL=E:\Code\python\InstantHMR\models\instanthmr-medium.fit2.onnx
set HMR_DETECTOR=nano
set HMR_DEVICE=cuda
run_video.bat clip.mp4
```

---

## 5. Parameters

| Option | |
| --- | --- |
| `--model` | model name (`nano`/`tiny`/`small`/`medium`/`medium-commercial`) or a path to an `.onnx` |
| `--fit 0\|1\|2` | kinematic-fit level; must match the `-fitN` graph |
| `--device cuda\|cpu\|tensorrt` | backend for the main graph |
| `--detector nano\|small\|medium\|base\|large` | person detector; nano is the fastest |
| `--detector-stride N` | detect every N frames (2–3 for live use) |
| `--pose-filter one-euro\|alpha-beta` | smooths the pose parameters |
| `--smooth` | smooths `cam_trans` (root position); can be combined |
| `--focal <px>` | override the focal length, which sets the world scale |
| `--max-persons N` | people per frame (default 4) |
| `--no-mesh` / `--no-viewer` / `--save-rrd <file>` | skeleton only / head-less / save a recording |

---

## 6. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `[ERROR] virtual env not found` | run `setup_windows.bat` first |
| `DLL load failed ... cudnn64_9.dll` | use the three `.bat` files (they add the nvidia `bin` directories); a bare `python demo.py` elsewhere is missing that PATH |
| Detector takes 90–120 ms, slower than the model | torch is the CPU build: `setup_windows.bat --check` shows `cuda: False`; reinstall the cu126 wheels |
| `Registering cuda ... failed` in the log | onnxruntime-gpu and torch disagree on CUDA; install `onnxruntime-gpu==1.26.*` with `torch 2.9.1+cu126` |
| Root joint jumps around | make sure the `-fit2` graph is used with `--fit 2`, add `--smooth`, and give a realistic `--focal` |
| Downloads stall / SSL errors | retry through a proxy: `setup_windows.bat --proxy http://127.0.0.1:21882` |
| Start from scratch | delete `.venv` and re-run `setup_windows.bat` (the model cache lives elsewhere and is kept) |

---

## 7. Measured performance (RTX 3090)

| Case | Detector | Model | End to end |
| --- | --- | --- | --- |
| Single image (warm) | ~30 ms | ~15 ms (fit2 graph) | — |
| 465-frame 720×1280 video | 34.7 ms | 35.8 ms | ~70 ms/frame (14 fps) |
| Webcam 640×480 (stride 3) | ~36 ms | — | capture ~24 fps |

The detector's CUDA fast path (`_cuda_predictor()` in `instanthmr/detector.py`) is what turns
90–120 ms into 25–36 ms. Pushing further means `compile=True` with fp16 (16 ms per frame, but a
~141 s build on first use, so it is not practical for a launcher).
