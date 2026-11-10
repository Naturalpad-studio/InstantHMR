# Training InstantHMR

This folder trains the InstantHMR models from the released ground-truth
annotations of [SAM 3D Body](https://huggingface.co/datasets/facebook/sam-3d-body-dataset)
(the MHR fits Meta used to build SAM 3D Body). It is not a distillation setup:
the labels are the dataset's, not a teacher model's output.

| file | what it is |
|---|---|
| `train.py` | The trainer: dataset, augmentation, model, MHR forward pass, losses, metrics, and a single-GPU loop. It also holds the self-checks (`--self-test`, `--overfit-test`). |
| `train_ddp.py` | The entry point for real runs. It imports `train.py` and adds multi-GPU training (torchrun or Slurm), per-dataset mixture sampling, resume, and 3DPW validation. Without a launcher it runs as one process on one GPU. |
| `val3dpw.py` | 3DPW validation used by `train_ddp.py --val-3dpw`. |
| `mhr_cont.py` | The continuous MHR head (`--cont-head`), transcribed from SAM 3D Body. |
| `assets/` | Small files the trainer reads: the keypoint regressor, the body-size bounds, the continuous-head and landmark tables, and the MHR-to-J14 adapter for 3DPW validation. |

A change to the model or a loss goes in `train.py`, so both entry points get it.

`mhr_cont.py` and `assets/` are derived from SAM 3D Body and are under the
[SAM License](../LICENSE-SAM), not Apache-2.0. A model trained with this folder
inherits it: it learns from annotations Meta releases under that licence, and
`--cont-head` stores SAM 3D Body tables in its weights.

## Setup

```bash
pip install -r training/requirements.txt
```

The trainer needs Meta's MHR rig, `mhr_model.pt` (Apache-2.0, 700 MB), from the
[MHR release](https://github.com/facebookresearch/MHR/releases):

```bash
curl -L -o mhr_assets.zip https://github.com/facebookresearch/MHR/releases/latest/download/assets.zip
mkdir -p training/checkpoints && unzip -j mhr_assets.zip assets/mhr_model.pt -d training/checkpoints
```

## Data

`datasets_pipeline/` downloads the annotations and images and writes one folder
per dataset, `sam3d_gt_<dataset>/{images,annotations}`; see
[`datasets_pipeline/README.md`](../datasets_pipeline/README.md).

The released models read a **window corpus**: for each person, four square
crops cut from the full photo (the standard 1.2x crop and three other
framings), stored side by side at 288 px. Build it per dataset with:

```bash
python datasets_pipeline/build_split.py \
    --split coco_train --annotation-dir <annotations>/coco_train --image-dir <coco images> \
    --windows 4 --window-sizes 288 --no-hands \
    --output-dir 'data_win{size}/sam3d_gt_coco'
```

The released mix uses SA-1B, AI Challenger, Harmony4D, COCO and MPII.

## Training the released models

The four sizes differ only in `--backbone` (`hgnetv2_b0`, `_b1`, `_b2`, `_b4`
for nano, tiny, small, medium). They were trained on 4 GPUs with 64 crops each:

```bash
torchrun --standalone --nproc_per_node=4 training/train_ddp.py \
    --data_root data_win288 --output_dir runs/medium \
    --backbone hgnetv2_b4 --image-size 288 \
    --preset baseline --losses rebalanced \
    --windows random --window0-p 0.75 \
    --mix sa1b=0.30,aic=0.18,harmony4d=0.10,coco=0.09,mpii=0.03 \
    --epochs 600 --samples-per-epoch 400000 --batch_size 64 \
    --lr 3e-4 --lr-scaling sqrt --seed 0 \
    --w_reproj 0.01 --w_keypoints3d 0.35 --w-verts 0.35 --n-verts 595 \
    --cont-head --exact-landmarks --cliff-focal --crop-centre-fix \
    --bound-scales --anomaly-safe-fallback \
    --val-3dpw <3DPW>/sequenceFiles --val-3dpw-images <3DPW>/imageFiles --val-3dpw-gt h36m
```

- `--batch_size` is per GPU, and `--lr-scaling sqrt` multiplies `--lr` by the
  square root of the number of GPUs (6e-4 on 4 GPUs). On another number of
  GPUs, the effective batch and the learning rate change with it.
- `--windows random --window0-p 0.75` uses the standard crop 75% of the time
  and one of the three other framings otherwise.
- **Medium-commercial** (M-C) is the medium recipe with
  `--windows standard` (no `--window0-p`) and
  `--mix harmony4d=0.75,coco=0.25`, on commercially usable images only:
  Harmony4D (MIT) and the 6,742 COCO people whose Flickr licence is CC BY or
  "no known copyright". Build that COCO split with
  `datasets_pipeline/coco_license_subset.py --licenses 4,7` and put it in the
  `--data_root` beside Harmony4D. The released M-C was stopped at epoch 532 of
  600.

These commands rebuild the exact settings recorded in each released model's
`run_config.json`.

## Fine-tuning a released model

The PyTorch checkpoints of the five released models are on
[Hugging Face](https://huggingface.co/momolesang/InstantHMR/tree/main/pytorch),
one folder per model with its `run_config.json`:

```bash
hf download momolesang/InstantHMR --include "pytorch/medium-commercial/*" --local-dir .
```

Start a new run from them with `--init-from`. It loads the weights (and the
EMA copy starts from them too), and keeps a fresh optimiser and learning-rate
schedule. Use the flags of that model's training command above, so that the
model built matches the checkpoint (a mismatched key stops the run), and
replace `--mix`, `--data_root` and the length of the run with your own:

```bash
torchrun --standalone --nproc_per_node=1 training/train_ddp.py \
    --init-from pytorch/medium-commercial/instanthmr-medium-commercial.pth \
    --data_root <data> --output_dir runs/my-finetune \
    --mix mydata=0.5,harmony4d=0.25,coco=0.25 \
    --epochs <fewer> --lr <lower> ...   # the other flags as in the model's command
```

`--mix` keys select the `sam3d_gt_<name>` folders in `--data_root`, so your
data must be in the same format (see `datasets_pipeline/`). Keeping some of
the original data in the mix limits forgetting. Do not resume a released
checkpoint as if it were your own `last.pth`: it holds the weights only, and
resuming continues the original 600-epoch schedule.

Each model's settings (`backbone`, `mix`, `cliff_focal`, ...) are in its
`run_config.json`. Keep `--cliff-focal`: all five models use it, and since it
changes what the box vector means rather than any weight, leaving it out
raises no error and silently misplaces people in depth. The exporter reads it
from the `run_config.json` that `train_ddp.py` writes in `--output_dir`, next
to your new checkpoints.

A run writes `last.pth` every epoch and resumes from it when relaunched with the
same command. `best_student_model_v3.pth` is the checkpoint with the best 3DPW
validation score.

### 3DPW validation

`--val-3dpw` scores the 3DPW validation split every epoch, on 14 joints from
the H36M regressor, as in the 3DPW literature. It needs that ground truth built
once next to `sequenceFiles`:

```bash
python benchmark/make_3dpw_gt.py --sequence-dir <3DPW>/sequenceFiles \
    --smpl-dir <folder with SMPL_MALE.pkl, SMPL_FEMALE.pkl> \
    --regressor <J_regressor_h36m.npy> --out <3DPW>/gt_h36m
```

Leave `--val-3dpw-split` at `validation`: choosing a checkpoint on the test
split makes its test score no longer held out. Without `--val-3dpw`, the run
selects on a 2% slice of the training corpus set aside (`--val-frac`), which
tells you whether training is healthy, not how well the model generalises.

## Checks before a long run

```bash
# Every parameter must get a gradient under DistributedDataParallel; a missing
# one passes every single-GPU run and only aborts a multi-GPU job.
python tools/ddp_smoke.py --backbone hgnetv2_b4 --image-size 288 \
    --w-verts 0.35 --cont-head --exact-landmarks --data_root <data>

# The model must be able to fit 8 images.
python training/train.py --overfit-test --data_root <data>

python -m pytest -q training
```

## Export to ONNX

```bash
python tools/export_onnx.py --ckpt runs/medium/best_student_model_v3.pth --output models/my-model.onnx
python tools/export_onnx.py --body --output models/mhr-body.onnx      # once, shared by every model
python demo.py --image photo.jpg --model models/my-model.onnx
```

The first graph is the network; the second turns its MHR parameters into the 70 3D keypoints and
the mesh, and the demo looks for it next to the model. The exporter reads the training settings
from the `run_config.json` beside the checkpoint, so keep the two together.

To export the network with the kinematic fit (on 1 or 2 crops per person), add `--fit`:

```bash
python tools/export_onnx.py --ckpt runs/medium/best_student_model_v3.pth --fit 2 --output models/my-model-fit2.onnx
python demo.py --video clip.mp4 --model models/my-model-fit2.onnx
```

This graph is fp16 by default, which needs a CUDA GPU to trace (`--fp32` otherwise). It reads the
MHR rig from `training/checkpoints/mhr_model.pt`, like `--body`.
