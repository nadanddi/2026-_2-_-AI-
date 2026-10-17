#!/bin/bash
# 서버: GPT v2 시험대(im_set_gpt2) 특징 → 후보 D1c 판정 (IM_부분생성_D1c_v1.md)
source ~/anaconda3/etc/profile.d/conda.sh; conda activate nadan
cd ~/storage/nadan/farmai_img/data; export PYTHONIOENCODING=utf-8
echo "== $(date '+%F %T') nvidia-smi =="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=0
S=im_set_gpt2; M=local/$S/manifest.csv; I=local/$S/img
python image/im06_lowlevel.py $S $I $M || echo FAIL_low
python image/im06b_region_feats.py $S $I $M || echo FAIL_reg
python -W ignore image/im04_clip_feats.py $S $I $M || echo FAIL_clip
python -W ignore image/im16_backbone_feats.py B2 $S $I $M || echo FAIL_B2
python - <<'PY' || echo FAIL_look
import sys, csv; sys.path.insert(0, "image")
import im14_lookup_feats as K
rows = list(csv.DictReader(open("local/im_set_gpt2/manifest.csv", encoding="utf-8")))
K.run(K.DB(), "im_set_gpt2", K.LOCAL / "im_set_gpt2" / "img", rows, [False] * len(rows))
PY
python -W ignore -u image/im05t_eval_D1c.py > ../logs/D1c_eval.log 2>&1 || echo FAIL_eval
echo D1C_DONE
