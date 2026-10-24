#!/bin/bash
# Generation-9 v4 family + g8h reference: 3DPW test and EMDB-1, full protocol (stride 1),
# published boxes, ground-truth-joint boxes and RF-DETR boxes. One invocation per input size.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
R=instanthmr_distill_train/runs
DPW_SEQ=${DPW_SEQ:-$HOME/Downloads/sequenceFiles/sequenceFiles}
DPW_IMG=${DPW_IMG:-$HOME/Downloads/imageFiles}
EMDB=${EMDB:-$HOME/Downloads/EMDB_root}
OUT=benchmark/results
C224="$R/g8h_lr1x_e600/g8h_lr1x_e600/best_student_model_v3.pth $(ls $R/g9w_b*_224_s0/best_student_model_v3.pth)"
C288="$(ls $R/g9w_b*_288_s0/best_student_model_v3.pth)"
for S in 224 288; do
  eval CK=\$C$S
  for B in gt-joints detector; do
    $PY benchmark/eval_3dpw_ckpt.py --ckpt $CK --sequence-dir "$DPW_SEQ" --image-root "$DPW_IMG" \
      --split test --gt h36m --adapter $OUT/adapter_j14_h36m_teacher.npz --bbox $B \
      --out $OUT/gen9w_3dpw_test_${B}_$S.json
  done
  for B in annotated gt-joints detector; do
    $PY benchmark/eval_emdb_ckpt.py --ckpt $CK --emdb-root "$EMDB" --bbox $B \
      --out $OUT/gen9w_emdb1_${B}_$S.json
  done
done
echo ALL DONE
