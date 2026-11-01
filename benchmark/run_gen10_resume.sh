#!/bin/bash
# The two passes of run_gen10.sh still missing after the 2026-10-29 interruption.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
CK="$(ls instanthmr_distill_train/runs/g10*_b*_288_s0/best_student_model_v3.pth)"
for B in gt-joints detector; do
  $PY benchmark/eval_emdb_ckpt.py --ckpt $CK --emdb-root "${EMDB:-$HOME/Downloads/EMDB_root}" --bbox $B \
    --out benchmark/results/gen10_emdb1_${B}_288.json
done
echo ALL DONE
