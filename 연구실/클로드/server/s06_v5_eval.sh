#!/bin/bash
# 서버: 5판 생성 완료 후 → 목록 합치기 → 특징(저수준·영역·CLIP·B1·B2) → A/B/C 비교
source ~/anaconda3/etc/profile.d/conda.sh; conda activate nadan
cd ~/storage/nadan/farmai_img/data; export PYTHONIOENCODING=utf-8
while ! grep -q GEN_V5_DONE ../logs/gen_v5.log; do sleep 120; done
python - <<'PY'
import csv, glob
rows = []
for k in ["filter_blur", "composite_v5b", "ai_gen_xl", "full_style_xl", "partial_style_xl", "partial_gen_xl"]:
    rows += list(csv.DictReader(open(f"local/im_set_v5/manifest_{k}.csv", encoding="utf-8")))
with open("local/im_set_v5/manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("v5 manifest", len(rows))
PY
echo "== $(date '+%F %T') nvidia-smi =="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=0
M=local/im_set_v5/manifest.csv; I=local/im_set_v5/img
python image/im06_lowlevel.py im_set_v5 $I $M; python image/im06b_region_feats.py im_set_v5 $I $M
python -W ignore image/im04_clip_feats.py im_set_v5 $I $M
for b in B1 B2; do python -W ignore image/im16_backbone_feats.py $b im_set_v5 $I $M; done
for b in B2 B1; do python -W ignore -u image/im05m_eval_v5data.py $b > ../logs/v5data_eval_$b.log 2>&1; done
echo V5EVAL_DONE
