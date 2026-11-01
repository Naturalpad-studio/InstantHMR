#!/usr/bin/env python3
"""Build a COCO split restricted to photos whose Flickr licence allows
commercial use, as symlinks into an existing built split (nothing is copied).

COCO's annotations are CC-BY 4.0, but our labels are the SAM 3D Body fits and
the photos keep their own Flickr licences, recorded per image in COCO's JSON
(assets/coco2014_train_licenses.csv, extracted from captions_train2014.json):

    1 BY-NC-SA  2 BY-NC  3 BY-NC-ND  4 BY  5 BY-SA  6 BY-ND  7 no known copyright  8 US Gov

Of the 39,665 people in sam3d_gt_coco: 4 -> 6,396, 5 -> 3,537, 6 -> 1,980,
7 -> 346; the rest (27,406) are NonCommercial. Which licences to keep is a
legal choice; --licenses makes it explicit and it is recorded in the output.

    python datasets_pipeline/coco_license_subset.py \
        --src $SCRATCH/instanthmr/data_win288/sam3d_gt_coco \
        --out $SCRATCH/instanthmr/data_win288_com/sam3d_gt_cococom --licenses 4,7

The output is a normal split folder (annotations/, body_crops/, images ->
body_crops), so it sits in a data root beside other splits and needs its own
pair index. Keep it OUT of a data root other runs train on: a new sub-folder
changes that root's index fingerprint and forces a rescan.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE = HERE / "assets" / "coco2014_train_licenses.csv"
NAME = re.compile(r"COCO_train2014_\d{12}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="a built sam3d_gt_coco split")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--licenses", required=True, help="COCO licence ids to keep, e.g. 4,7")
    args = p.parse_args()

    keep = {int(v) for v in args.licenses.split(",")}
    with open(TABLE) as f:
        lic = {r["file_name"]: int(r["license"])
               for r in csv.DictReader(line for line in f if not line.startswith("#"))}
    if args.out.exists():
        sys.exit(f"{args.out} exists; remove it first")

    ann_src = args.src / "annotations"
    crops = args.src / "images"            # body_crops/, or a real images/ dir on old splits
    (args.out / "annotations").mkdir(parents=True)
    (args.out / "body_crops").mkdir()
    os.symlink("body_crops", args.out / "images")

    # One directory listing, not one glob per person: on Lustre each glob re-reads
    # the whole 40k-entry folder.
    crop_of = {e.name.rsplit(".", 1)[0]: e.path for e in os.scandir(crops)}
    kept = dropped = 0
    by_lic: dict[int, int] = {}
    for npz in sorted(ann_src.glob("*.npz")):
        m = NAME.search(npz.stem)
        if m is None or f"{m.group(0)}.jpg" not in lic:
            sys.exit(f"{npz.name}: no COCO 2014 licence on record")
        code = lic[f"{m.group(0)}.jpg"]
        if code not in keep:
            dropped += 1
            continue
        img = crop_of.get(npz.stem)
        if img is None:
            sys.exit(f"{npz.name}: no crop in {crops}")
        os.symlink(npz.resolve(), args.out / "annotations" / npz.name)
        os.symlink(Path(img).resolve(), args.out / "body_crops" / Path(img).name)
        kept += 1
        by_lic[code] = by_lic.get(code, 0) + 1

    (args.out / "LICENSE_FILTER.json").write_text(json.dumps(dict(
        source=str(args.src.resolve()), licenses_kept=sorted(keep),
        people_kept=kept, people_dropped=dropped,
        kept_per_license={str(k): v for k, v in sorted(by_lic.items())}), indent=2))
    print(f"{kept:,} people kept, {dropped:,} dropped; per licence {dict(sorted(by_lic.items()))}")


if __name__ == "__main__":
    main()
