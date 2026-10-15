#!/bin/bash
# 서버: 5판 원본 대조 특징 계산(CPU) → B2 A/B ± L2 비교
source ~/anaconda3/etc/profile.d/conda.sh; conda activate nadan
cd ~/storage/nadan/farmai_img/data; export PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=""
python - <<'PY'
import sys, csv; sys.path.insert(0, "image")
import im14_lookup_feats as M
db = M.DB(); rows = list(csv.DictReader(open("local/im_set_v5/manifest.csv", encoding="utf-8")))
M.run(db, "im_set_v5", M.LOCAL / "im_set_v5" / "img", rows, [False] * len(rows))
PY
python -W ignore -u image/im05n_eval_B_L2.py > ../logs/B_L2_eval.log 2>&1
echo BL2_DONE
