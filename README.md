# InstantHMR: Real-Time Human Mesh Recovery on Your Phone

<p align="center">
  <a href="https://huggingface.co/momolesang/InstantHMR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffc107" alt="Hugging Face models"></a>
  <a href="https://hal.science/hal-05854387"><img src="https://img.shields.io/badge/HAL-hal--05854387-b31b1b" alt="Paper on HAL"></a>
</p>

InstantHMR recovers the 3D body of every person in an image: pose, body shape, 70 keypoints in 2D
and 3D, and a full body mesh in the [Momentum Human Rig](https://github.com/facebookresearch/MHR)
(MHR). It runs in about 5 ms per person on a phone NPU and in about 1 ms on a desktop GPU.

**New in v1.1:** a kinematic fit inside the model graph refines each pose against the network's own
2D keypoints (`--fit`, on by default), and two pose filters for video (`--pose-filter`). See
[Kinematic fit](#kinematic-fit).

<p align="center">
  <img src="assets/demo_desktop.gif" height="300" alt="InstantHMR on a desktop GPU, shown in the Rerun viewer">
  <img src="assets/demo_phone.gif" height="300" alt="InstantHMR on a Samsung Galaxy S23">
</p>

## Models

Four sizes, trained on the released annotations of the
[SAM 3D Body dataset](https://huggingface.co/datasets/facebook/sam-3d-body-dataset).
Accuracy is PA-MPJPE (lower is better) under the protocol of the paper; latency is per person,
for the model alone on the GPU as measured in the paper (the desktop column with TensorRT directly,
up to a 6,890-vertex mesh).

| Model | Parameters | EMDB-1 (mm) | 3DPW (mm) | RTX 4070, TensorRT (ms) | Galaxy S23 NPU (ms) | Licence |
|---|---|---|---|---|---|---|
| `nano` | 20.0 M | 61.7 | 46.6 | 0.70 | 4.60 | SAM License, non-commercial |
| `tiny` | 20.3 M | 58.0 | 44.0 | 0.73 | 4.84 | SAM License, non-commercial |
| `small` | 24.4 M | 53.3 | 41.7 | 0.84 | 5.38 | SAM License, non-commercial |
| `medium` (default) | 32.2 M | 48.8 | 40.0 | 0.98 | 5.49 | SAM License, non-commercial |
| `medium-commercial` | 32.2 M | 72.2 | 50.7 | 0.98 | 5.49 | SAM License |

`medium-commercial` is the medium model trained only on images that allow commercial use
(Harmony4D and part of COCO). The weights download automatically from
[Hugging Face](https://huggingface.co/momolesang/InstantHMR) on first use.

### Kinematic fit

Every model also comes with a kinematic fit in the same graph. The network predicts the MHR pose
and, separately, the person's 2D keypoints; the 2D keypoints are about twice as accurate as the
projection of the 3D pose. The fit moves the pose (one Levenberg-Marquardt step through the MHR
skeleton, each keypoint weighted by its confidence) so that the projected body lands on those 2D
keypoints. It has no learned parameters and runs inside the graph in fp16, on the GPU or the phone
NPU. `--fit 1` runs it on one crop of each person (0.9 of the box); `--fit 2` (default) adds a
zoomed-in crop (0.7 of the box) in the same batch and fuses the 2D of both.

| Model | EMDB-1 (mm) | 3DPW (mm) | RTX 4070, TensorRT (ms) | Galaxy S23 NPU (ms) |
|---|---|---|---|---|
| `nano`, `--fit 0 / 1 / 2` | 61.7 / 57.2 / 55.9 | 46.6 / 44.5 / 44.1 | 0.70 / 0.85 / 1.10 | 4.60 / 5.49 / 8.73 |
| `tiny`, `--fit 0 / 1 / 2` | 58.0 / 53.8 / 52.6 | 44.0 / 42.4 / 42.0 | 0.73 / 0.87 / 1.13 | 4.84 / 5.84 / 9.20 |
| `small`, `--fit 0 / 1 / 2` | 53.3 / 49.2 / 48.1 | 41.7 / 40.3 / 40.0 | 0.84 / 1.00 / 1.30 | 5.38 / 6.37 / 10.36 |
| `medium`, `--fit 0 / 1 / 2` | 48.8 / 45.2 / 44.4 | 40.0 / 39.4 / 39.3 | 0.98 / 1.12 / 1.53 | 5.49 / 6.40 / 10.22 |
| `medium-commercial`, `--fit 0 / 1 / 2` | 72.2 / 68.8 / 68.3 | 50.7 / 47.7 / 47.5 | 0.98 / 1.12 / 1.53 | 5.49 / 6.40 / 10.22 |

The fit graphs take one person per call, and their crop scales assume the person box of the
benchmarks; the demo enlarges RF-DETR's tighter boxes to match (by 1/0.866, measured on 3DPW).

## Install

Python 3.10 to 3.12, and on Linux or Windows an NVIDIA driver 525 or newer.

```bash
git clone https://github.com/Naturalpad-studio/InstantHMR.git
cd InstantHMR
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

The first `pip` line installs PyTorch for CUDA 12, which the person detector uses and onnxruntime
shares, so a single set of CUDA libraries is installed. On a Mac, or without an NVIDIA GPU, skip
it.

## Demo

```bash
python demo.py --image photo.jpg
python demo.py --video clip.mp4 --pose-filter one-euro
python demo.py --camera 0 --model small --fit 1
```

The [Rerun](https://rerun.io) viewer opens with the frame and its 2D skeletons, the 3D scene, and
the latency of each stage. People are found by the [RF-DETR](https://github.com/roboflow/rf-detr)
detector, then InstantHMR runs on each of them.

| Option | |
|---|---|
| `--model` | `nano`, `tiny`, `small`, `medium` (default), `medium-commercial`, or a path to an `.onnx` file |
| `--fit` | kinematic fit on `2` crops per person (default), `1` crop, or `0` for the network alone |
| `--device` | `cuda` (default), `tensorrt`, `cpu` or `coreml` |
| `--pose-filter` | videos and cameras: `one-euro` or `alpha-beta` filters each person's pose before the mesh is built; both were tuned on 3DPW and EgoHumans, lower the error slightly and the jitter by a third, and lag less than half a frame |
| `--smooth` | videos and cameras: smooths the keypoints and mesh instead; steadier still, but it lags on fast motion |
| `--focal` | the camera's focal length in pixels, if you know it; it places people at the right depth |
| `--detector` | RF-DETR size: `nano`, `small`, `medium` (default), `base`, `large` |
| `--max-persons` | most people per frame (default 4) |
| `--detector-stride` | run the detector every N frames; in between, each person's box follows their 2D keypoints |
| `--no-mesh` | skeleton only |
| `--save-rrd` | also save the recording, to replay with `rerun file.rrd` |

The latencies in the tables above are the model alone, as measured in the paper. The demo's timings
are what a frame costs end to end in Python. On an RTX 4070 with TensorRT, the InstantHMR stage takes
2.8 ms per person with `--fit 1` (0.25 ms to cut the crop on the CPU, 1.3 ms for the graph, 1.2 ms
for the keypoints and the full 18,439-vertex mesh) and 3.3 ms with `--fit 2`; with CUDA, 4.7 and
5.7 ms. The fit graphs are faster end to end than the network alone (3.3 ms with TensorRT), because
they take the crop as 8-bit pixels and run in fp16. The detector is the slow stage (about 27 ms for
RF-DETR medium); for a faster live demo, use `--detector nano` or `--detector-stride 3`.

## Use it from Python

```python
import cv2
from instanthmr import PosePipeline

pipeline = PosePipeline("medium", fit=2, device="cuda")       # pose_filter="one-euro" for video
image = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)
for person in pipeline(image).persons:
    person.joints_2d        # (70, 2) keypoints, pixels
    person.joints_3d_cam    # (70, 3) keypoints, metres, camera frame (x right, y down, z forward)
    person.vertices_cam     # (18439, 3) mesh, metres, camera frame; faces: pipeline.hmr.faces
    person.mhr_params       # (204,) MHR pose and skeleton parameters
    person.shape_params     # (45,) MHR body shape
```

With your own person boxes, use `InstantHMR` directly:
`InstantHMR(resolve_model("medium", fit=2)).predict(image, boxes)`. The 70 keypoints are listed in
`instanthmr.JOINT_NAMES`.

## TensorRT

TensorRT runs InstantHMR in fp16 on NVIDIA GPUs. Install it next to onnxruntime and pick the
device:

```bash
pip install tensorrt-cu12==10.15.*
python demo.py --camera 0 --device tensorrt
```

The first run builds an engine for your GPU and caches it in `models/trt_cache/`: about two
minutes for a network graph (it covers batches of up to `--max-persons` people), and four to ten
minutes for a fit graph (one person per call, one engine per graph). Later runs start in under a
second. With CUDA, the fit graphs are replayed as a single CUDA graph, so they need no extra
installation to be fast.

## Training and evaluation

- [`training/`](training/README.md): train the released models, or your own, from the SAM 3D Body
  annotations, and export them to ONNX.
- [`datasets_pipeline/`](datasets_pipeline/README.md): download the annotations and images and
  build the training data.
- [`benchmark/`](benchmark/README.md): reproduce the paper's EMDB-1 and 3DPW numbers.

## Licence

The code is under the [Apache-2.0](LICENSE) licence, except
[`training/mhr_cont.py`](training/mhr_cont.py) and [`training/assets/`](training/assets/), which
are derived from SAM 3D Body and are under the [SAM License](LICENSE-SAM).

The weights, all five models (with and without the fit) and the shared `mhr-body.onnx`, are under the
[SAM License](LICENSE-SAM): they are trained on the SAM 3D Body annotations, which Meta releases
under that licence, and they contain tables from the SAM 3D Body checkpoint. The SAM License
allows commercial use, under the conditions it lists (trade controls, no military use, among
others). In addition, the `nano`, `tiny`, `small` and `medium` weights are for non-commercial
use only, because some of their training images (SA-1B, AI Challenger, and COCO photos under
non-commercial Flickr licences) are licensed for research or non-commercial use only.
`medium-commercial` was trained only on images that allow commercial use and has no such
restriction. The body model inside `mhr-body.onnx` is Meta's
[MHR](https://github.com/facebookresearch/MHR) (Apache-2.0).

## Acknowledgements

InstantHMR is trained on the annotations Meta released with
[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) and uses Meta's
[MHR](https://github.com/facebookresearch/MHR) body model. People are detected with
[RF-DETR](https://github.com/roboflow/rf-detr) and shown with [Rerun](https://rerun.io). Training
was granted access to the HPC resources of IDRIS under the allocation AD011017116 made by GENCI.

## Citation

<!-- ticket 20: add the arXiv identifier. -->
```bibtex
@misc{adjel2026instanthmr,
  title        = {InstantHMR: Real-Time Human Mesh Recovery on Your Phone},
  author       = {Adjel, Mohamed and Chalabi, Kahina and Bonnet, Vincent},
  year         = {2026},
  howpublished = {HAL preprint hal-05854387},
  url          = {https://hal.science/hal-05854387},
  note         = {LAAS report no. 26486}
}
```
