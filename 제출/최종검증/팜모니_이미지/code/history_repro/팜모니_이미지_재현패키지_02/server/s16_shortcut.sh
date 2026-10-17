#!/bin/bash
# 서버: G3 지름길 반증 세트 특징 → 판정 (IM_G3_지름길반증_v1.md)
source ~/anaconda3/etc/profile.d/conda.sh; conda activate nadan
cd ~/storage/nadan/farmai_img/data; export PYTHONIOENCODING=utf-8
echo "== $(date '+%F %T') nvidia-smi =="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=0
for S in im_set_gpt3r im_set_realn im_set_realg; do
  M=local/$S/manifest.csv; I=local/$S/img
  python image/im06_lowlevel.py $S $I $M || echo FAIL_low_$S
  python image/im06b_region_feats.py $S $I $M || echo FAIL_reg_$S
  python -W ignore image/im04_clip_feats.py $S $I $M || echo FAIL_clip_$S
  python -W ignore image/im16_backbone_feats.py B2 $S $I $M || echo FAIL_B2_$S
  python - $S <<'PY' || echo FAIL_look_$S
import sys, csv; sys.path.insert(0, "image")
import im14_lookup_feats as K
s = sys.argv[1]; rows = list(csv.DictReader(open(f"local/{s}/manifest.csv", encoding="utf-8")))
K.run(K.DB(), s, K.LOCAL / s / "img", rows, [False] * len(rows))
PY
done
python -W ignore -u image/im05z_eval_shortcut.py > ../logs/shortcut_eval.log 2>&1 || echo FAIL_eval
echo SC_DONE
