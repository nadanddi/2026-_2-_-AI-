#!/bin/bash
# 서버: GPT v3 시험대(im_set_gpt3) 특징 → 후보 G3 판정 (IM_GPT학습_G3_v1.md)
source ~/anaconda3/etc/profile.d/conda.sh; conda activate nadan
cd ~/storage/nadan/farmai_img/data; export PYTHONIOENCODING=utf-8
echo "== $(date '+%F %T') nvidia-smi =="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=0
S=im_set_gpt3; M=local/$S/manifest.csv; I=local/$S/img
python image/im06_lowlevel.py $S $I $M || echo FAIL_low
python image/im06b_region_feats.py $S $I $M || echo FAIL_reg
python -W ignore image/im04_clip_feats.py $S $I $M || echo FAIL_clip
python -W ignore image/im16_backbone_feats.py B2 $S $I $M || echo FAIL_B2
python - <<'PY' || echo FAIL_look
import sys, csv; sys.path.insert(0, "image")
import im14_lookup_feats as K
rows = list(csv.DictReader(open("local/im_set_gpt3/manifest.csv", encoding="utf-8")))
K.run(K.DB(), "im_set_gpt3", K.LOCAL / "im_set_gpt3" / "img", rows, [False] * len(rows))
PY
python -W ignore -u image/im05x_eval_G3.py > ../logs/G3_eval.log 2>&1 || echo FAIL_eval
echo G3_DONE
