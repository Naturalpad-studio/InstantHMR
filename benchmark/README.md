# Evaluation

`evaluate.py` scores an InstantHMR model on EMDB-1 and 3DPW test under the protocol of the paper:
every 10th frame, ground-truth person boxes, the calibrated focal length, and Meta's official MHR
to SMPL fitting. EMDB-1 is scored on the 24 SMPL joints and 3DPW on the 14 joints regressed with
the Human3.6M regressor, as in each benchmark's literature; both joint sets are printed.

## Setup

```bash
pip install -r requirements.txt -r benchmark/requirements.txt
git clone https://github.com/facebookresearch/MHR.git ../MHR
curl -L -o mhr_assets.zip https://github.com/facebookresearch/MHR/releases/latest/download/assets.zip
mkdir -p training/checkpoints && unzip -j mhr_assets.zip assets/mhr_model.pt -d training/checkpoints
```

Then put these files in one folder, the `--smpl-dir`:

| File | Where from |
|---|---|
| `SMPL_MALE.pkl`, `SMPL_FEMALE.pkl` | [SMPL](https://smpl.is.tue.mpg.de), "SMPL for Python", renamed |
| `SMPL_NEUTRAL.pkl` | [SMPLify](https://smplify.is.tue.mpg.de), `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`, renamed |
| `J_regressor_h36m.npy` | the data of [SPIN](https://github.com/nkolot/SPIN) (`fetch_data.sh`) |

And the benchmarks:

- **EMDB**: request access at [emdb.ait.ethz.ch](https://eth-ait.github.io/emdb/); `--data` is the
  folder that holds `P0/`, `P1/`, ...
- **3DPW**: from [virtualhumans.mpi-inf.mpg.de/3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/);
  `--data` is the folder that holds `sequenceFiles/` and `imageFiles/`.

## Run

```bash
python benchmark/evaluate.py --dataset emdb --data <EMDB> --model medium \
    --smpl-dir <SMPL files> --mhr-repo ../MHR
python benchmark/evaluate.py --dataset 3dpw --data <3DPW> --model medium \
    --smpl-dir <SMPL files> --mhr-repo ../MHR
```

`--limit 50` scores the first 50 frames as a quick check. The fitting runs on the GPU; a full
benchmark takes a few minutes.

## Expected results

`medium` reproduces Table 2 of the paper:

| | PA-MPJPE (mm) | MPJPE (mm) | PVE (mm) |
|---|---|---|---|
| EMDB-1, 24 joints | 48.8 | 74.9 | 89.1 |
| 3DPW, 14 joints | 40.0 | 64.0 | 76.8 |

The paper ran the PyTorch checkpoints in fp16 and the released ONNX graphs run in fp32, so single
frames differ slightly; the means above agree to 0.003 mm. Meta's fitting is not bitwise
deterministic on a GPU, so a rerun can move them by a few hundredths of a millimetre.
