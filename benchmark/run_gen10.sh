#!/bin/bash
# Generation 10 (all 288 px): the baseline family, the standard-crop-probability ablation and
# the commercial-data run. 3DPW test and EMDB-1, every frame, published / GT-joint / RF-DETR boxes.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
R=instanthmr_distill_train/runs
DPW_SEQ=${DPW_SEQ:-$HOME/Downloads/sequenceFiles/sequenceFiles}
DPW_IMG=${DPW_IMG:-$HOME/Downloads/imageFiles}
EMDB=${EMDB:-$HOME/Downloads/EMDB_root}
OUT=benchmark/results
CK="$(ls $R/g10*_b*_288_s0/best_student_model_v3.pth)"
for B in gt-joints detector; do
  $PY benchmark/eval_3dpw_ckpt.py --ckpt $CK --sequence-dir "$DPW_SEQ" --image-root "$DPW_IMG" \
    --split test --gt h36m --adapter $OUT/adapter_j14_h36m_teacher.npz --bbox $B \
    --out $OUT/gen10_3dpw_test_${B}_288.json
done
for B in annotated gt-joints detector; do
  $PY benchmark/eval_emdb_ckpt.py --ckpt $CK --emdb-root "$EMDB" --bbox $B \
    --out $OUT/gen10_emdb1_${B}_288.json
done
echo ALL DONE
