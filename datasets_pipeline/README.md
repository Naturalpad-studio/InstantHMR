# datasets_pipeline

Builds the training data from the released annotations of
[SAM 3D Body](https://huggingface.co/datasets/facebook/sam-3d-body-dataset) and the original
images of each dataset. One folder per dataset:

```
<data root>/sam3d_gt_<dataset>/
├── annotations/   <name>.npz    MHR parameters, camera, 2D and 3D keypoints of one person
└── images/        <name>.png    the person crop
```

| script | does |
|---|---|
| `fetch_annotations.py` | downloads the annotation shards (gated: accept the terms on Hugging Face first) |
| `probe_split.py` | tells you what a split contains, from its annotations alone |
| `plan_disk.py` | how much disk a split will take |
| `fetch_images.py` | downloads the source images, or prints how to get them |
| `build_split.py` | annotations + images -> the folder above |
| `verify_split.py` | checks a built split |

`registry.py` holds every URL and per-dataset detail; `stream_archives.py` builds a dataset larger
than your disk; `undistort.md` covers the datasets whose frames must be undistorted first.

## Quick start

The released models were trained on a **window corpus**: for each person, four square crops cut
from the full image (the standard 1.2x box crop and three other framings), stored side by side at
288 px. COCO as an example:

```bash
python datasets_pipeline/fetch_annotations.py --splits coco_train --out annotations
python datasets_pipeline/fetch_images.py --dataset coco --dest images/coco
python datasets_pipeline/build_split.py --split coco_train \
    --annotation-dir annotations/coco_train --image-dir images/coco \
    --windows 4 --window-sizes 288 --no-hands --output-dir 'data_win{size}/sam3d_gt_coco'
```

Then train with `--data_root data_win288` (see [`training/README.md`](../training/README.md)).
Without `--windows`, `build_split.py` writes single crops into `data/sam3d_gt_<dataset>/`, which
the trainer also reads. Every script resumes where it stopped.

## What each dataset costs, and how to get it

| dataset | images | route | undistort |
|---|---|---|---|
| coco | 13.5 GB (often mirrored) | `wget` train2014.zip — **only train2014 is referenced** | no |
| mpii | 12.1 GB | `wget` mpii_human_pose_v1.tar.gz | no |
| aic | 36.4 GB | OpenDataLab CLI (login) — where this corpus came from before | no |
| 3dpw | 4.6 GB | licence form → imageFiles.zip | no |
| harmony4d | 352 GB | HF `Jyun-Ting/Harmony4D`, ungated, per-scene zips | **yes** |
| egohumans | 107 GB referenced (550 GB advertised) | Google Drive, per-take tar.gz → `gdown --folder` | **yes** |
| egoexo4d | ~8 GB (procedure) | `egoexo -o $DEST --parts sam_3d_body` (signed licence) | no |
| sa1b | ~11 TB (often mirrored) | terms → `links.txt` → `--links-file` | no |

`fetch_images.py` handles the first two and harmony4d directly. For the rest it
prints the exact commands and exits 2 rather than pretending.

Measured output sizes, from `plan_disk.py` over the shards already on this
machine (PNG crops, 20 % hand yield):

```
split                         persons     frames     body    hands  originals     TOTAL
3dpw_train                     22,405     17,410     1.7G     0.7G       4.4G      6.7G
aic_train                     287,391    169,923    21.6G     8.6G      11.9G     42.1G
harmony4d_train               496,036    248,018    37.2G    14.9G     347.2G    399.3G
mpii_train                      7,135      5,441     0.5G     0.2G       1.1G      1.9G
```

Two things fall out of that table.

**`original_images` is the whole cost.** Everything except Harmony4D fits
comfortably: coco + mpii + aic + 3dpw is roughly 60 GB of originals and 40 GB of
crops. Harmony4D alone is 347 GB of 4K frames and does not fit in 172 GB at any
setting. Build it with `--originals none` — `body_crops` and `hands_crops` are
self-contained, and the npz records `image_relpath`, so the originals for any
subset can be re-fetched later.

**`--crop-format jpg` cuts the crops 3×** (75 KB → 24 KB per 224 crop) and
decodes ~4× faster, which also speeds up training. It is not the default only
because PNG keeps byte-identity with the existing `data/` tree. If you are
rebuilding from scratch anyway, use jpg.

## Datasets you fetch by hand (AIC, 3DPW)

Both are one archive behind a login, so you download and `rsync` them yourself.
The only thing that matters afterwards is **what `--image-dir` points at**: it
is the root that the parquet `image` strings are relative to, and that root
differs per dataset because `image_relpath()` prepends a different prefix.

**AIC** — `image` is a bare `<hash>.jpg`, and the resolver prepends
`train/images/`. So unpack until this exists:

```
$AIC_IMG_DIR/train/images/0000252aea98840a550dac9a78c476ecb9f47ffa.jpg
```

which is exactly the layout already on this machine at
`~/sam3d_data/images/aic`. Then:

```bash
python datasets_pipeline/build_split.py --split aic_train \
    --annotation-dir $ANN_DIR/aic_train --image-dir $AIC_IMG_DIR \
    --data-root data --validate
```

`imageFiles.zip`-style archives often unpack one level deeper than you expect;
if the build reports `missing_image` for everything, the root is wrong by one
directory. A single `--max-samples 20` run tells you in seconds.

**3DPW** — `image` is `<sequence>/image_00386.jpg` and the resolver prepends
nothing, so sequence folders sit directly under the root:

```
$THREEDPW_IMG_DIR/outdoors_climbing_01/image_00386.jpg
```

You need `imageFiles.zip` only. `sequenceFiles.zip` holds the original SMPL
ground truth, which this pipeline never reads — the MHR fits come from the
parquet. Keep it for the 3DPW ground truth (`benchmark/make_3dpw_gt.py`), not for
training data.

Same command, `--split 3dpw_train`. Both are small enough for
`--originals copy` (4.4 GB and 11.9 GB of referenced frames).

## EgoHumans

Google Drive, one `.tar.gz` per take, and the untar needs
`--strip-components=8` or you get eight nested empty parents:

```bash
pip install gdown
gdown --folder 1JD963urzuzV_R_6FOVOtlx8UupwUuknR -O $EGOHUMANS_DATA_DIR
cd $EGOHUMANS_DATA_DIR/01_tagging && tar xzf 001_tagging.tar.gz --strip-components=8
```

The dataset advertises 550 GB, but `egohumans_train` references only **89 takes
across 7 sequences** — 272k persons over 268k 4K frames, ~107 GB of referenced
frames. Get the exact shopping list before downloading anything:

```bash
python datasets_pipeline/probe_split.py \
    --annotation-dir $ANN_DIR/egohumans_train --list-groups 2
```

Only the `exo/` cameras are referenced; `ego/`, `colmap/` and `processed_data/`
inside each tarball are dead weight you can delete after extraction. Frames
must be undistorted (`undistort.md`), then build per take with
`--image-prefix 01_tagging/001_tagging/`, or let `stream_archives.py` do it.

Worth knowing before you spend the 107 GB: it is good body data (multi-view,
272k persons) but useless for hands — sports at distance, so only 21.5% of
hands clear 64 px and the finger articulation is 6.3 mm, the lowest in the
corpus.

## SA-1B

**Check your cluster's shared dataset mirror first** — SA-1B is commonly
mirrored, and if it is, everything below is moot. A mirror that unpacked the
1000 tars in place stores `sa_000000/`, `sa_000001/`, ... while the parquet asks
for a flat `sa_<id>.jpg`; `index_sa1b.py` walks it once and
`build_split.py --image-index` resolves against that index.

Otherwise, accept the terms at
<https://ai.meta.com/datasets/segment-anything-downloads/> to get `links.txt`
(one `sa_NNNNNN.tar <signed-url>` per line), then:

```bash
python datasets_pipeline/stream_archives.py --dataset sa1b \
    --split sa1b_train --annotation-dir $ANN_DIR/sa1b_train \
    --work-dir /scratch/sa1b --data-root data \
    --links-file ~/links.txt --max-archives 50 --build-every 10 \
    -- --originals none --crop-format jpg
```

Any prefix of the tar list is a valid dataset, so you can stop whenever. If you
are paying for the bandwidth, the economics are poor:

- References are spread **uniformly over all 1000 tars** — 994 of 1000 id
  blocks are hit by just 3% of the annotations — so there is no dense subset to
  cherry-pick.
- Only **11.4%** of SA-1B is referenced. Each ~10 GB tar yields ~1,270 useful
  frames ≈ 3,400 crops, i.e. **~2.3 MB of download per crop** against AIC's
  0.13 MB. The full 3.4M crops cost ~11 TB.

### The SA-1B label trap

SA-1B is the one split whose keypoint arrays are not fully populated, and the
two arrays differ:

| | populated |
|---|---|
| `keypoints_3d` + `model_params` + `shape_params` | **100%** |
| `keypoints_2d`, body (0–14) | 100% |
| `keypoints_2d`, feet (15–20) | 97% |
| `keypoints_2d`, hands (21–62) | **8%** |

So the 3D supervision is complete and the split is fully usable — but the 2D
array stores unobserved keypoints as literal `(0, 0)` with a graded confidence
in the third column. `tools/parquet_to_npz.py` drops that column
(`joints_2d = kp2[:, :2]`), so a naive build writes ~45 of 70 keypoints per
person as observations at the image corner. Left unmasked that would drag every
predicted hand keypoint to the top-left of the frame.

`build_split.py` stores `joints_2d_vis` and `joints_3d_conf`, and refuses to cut
a hand crop unless all 21 of that hand's keypoints are visible (true for 0.3% of
SA-1B hands and 100% of every other split).

`train.py` consumes the mask: `loss_2d_native`,
`loss_2d_simcc` and `loss_reproj` are weighted by `joints_2d_vis`, which
defaults to all-ones when a folder does not carry it — so every existing
`data/sam3d_gt_*` folder trains exactly as before (verified to float32
rounding, 6e-8). On SA-1B, `vis > 0` is an exact gate for "this coordinate is
real": zero exceptions in 183k keypoints measured.

## Datasets that do not fit: `stream_archives.py`

Never holds more than one archive: download → extract → undistort → build →
delete. Any prefix of the archive list is a valid dataset, so you can stop when
the disk or the clock says so and resume later.

```bash
python datasets_pipeline/stream_archives.py --dataset harmony4d \
    --split harmony4d_train --annotation-dir $ANN_DIR/harmony4d_train \
    --work-dir /scratch/h4d --data-root data \
    --undistort-python ~/anaconda3/envs/sam_3d_body_data/bin/python \
    -- --originals none --crop-format jpg

python datasets_pipeline/stream_archives.py --dataset sa1b \
    --split sa1b_train --annotation-dir $ANN_DIR/sa1b_train \
    --work-dir /scratch/sa1b --data-root data \
    --links-file ~/sa1b_links.txt --max-archives 200 --build-every 20 \
    -- --originals none
```

`--min-free-gb` (default 25) aborts before an archive that would fill the disk.
Read `undistort.md` before the first Harmony4D run — Meta's script has three
blockers that a fresh clone hits.

## Hand crops

The hand box is free: `joints_2d` is already in full-frame pixel coordinates, so
the 21 keypoints of each hand give a tight box with no extra annotation.

- `--hand-min-px 64` (default) is SAM3D's own validity gate, applied to the raw
  keypoint span. On AIC this keeps ~17 % of hands, at a median 145 source pixels
  upsampled 1.5× to 224.
- `--hand-expand 2.0` stores margin around the hand so you can jitter the box
  *inside* the stored crop at train time. That is what makes a model tolerant of
  a real detector's boxes, and it is why cropping dynamically from originals
  buys nothing.
- Crops are stored **unflipped**. SAM3D trains one right-hand model and mirrors
  the left; flipping a loaded crop is a free numpy slice, so baking the mirror
  into disk would only remove a choice.
- `hand_valid_l` / `hand_valid_r` in the npz say whether a crop exists;
  `hand_box_*` (tight) and `hand_crop_*` (the square actually written) let you
  re-derive any tighter crop without touching the originals.

## Which split to spend money on

`probe_split.py` measured every split from its annotations. *Articulation* is
how far the labels' finger poses sit from ONE constant hand pose, in mm at the
fingertips — an upper bound on what a hand branch could learn from them.
*≥96 px* is the share of hands whose crop would be at least 96 source pixels.

| split | persons | articulation | hands ≥96 px | download |
|---|---|---|---|---|
| **egoexo4d_procedure** | ~350k | **16.1 mm** | **85.5%** | licence + CLI |
| egoexo4d_physical | ~700k | 14.5 mm | 19.3% | licence + CLI |
| harmony4d | 496k | 12.6 mm | 3.4% | 352 GB |
| sa1b | 3.4M | 8.5 mm | 0.3% visible at all | ~11 TB, or free if mirrored |
| 3dpw | 22k | 7.5 mm | 71.4% | 4.6 GB |
| mpii | 7k | 7.4 mm | 11.5% | 12.1 GB |
| aic | 287k | 6.9 mm | 16.6% | 36.4 GB |
| coco | ~42k | 6.5 mm | 3.0% | 13.5 GB |
| egohumans | 272k | 6.3 mm | 6.4% | 107 GB |

COCO, MPII and AIC have no finger keypoints and 3DPW is SMPL, so the official
MHR fits fall back on the prior — a hand branch trained on them learns a
constant. **EgoExo4D procedure** (cooking, bike repair, piano — hands doing
things close to the camera) is the only split in the corpus where both terms
are good at once: 2.5× the articulation of COCO/AIC and two thirds of hands
above 128 px. It is also cheap, ~8 GB of referenced frames for 350k persons.
That is where the licence you are waiting on pays off.

Re-run the check on your own build at any time:

```bash
python datasets_pipeline/verify_split.py --dir data/sam3d_gt_egoexo4d --hand-signal
```
