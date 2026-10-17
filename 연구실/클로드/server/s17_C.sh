#!/bin/bash
# 서버: 조합 후보 C 판정 준비(시험 세트 조각 특징) → 판정 (IM_조합후보_C_v1.md)
source ~/anaconda3/etc/profile.d/conda.sh; conda activate nadan
cd ~/storage/nadan/farmai_img/data; export PYTHONIOENCODING=utf-8
echo "== $(date '+%F %T') nvidia-smi =="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=0
for S in im_set_gpt3 im_set_gpt3r im_set_realn im_set_realg; do
  [ -f local/feats/${S}_P16.npz ] || python -W ignore image/im23_patch_feats.py $S local/$S/img local/$S/manifest.csv || echo FAIL_P16_$S
done
python -W ignore -u image/im05ad_eval_C.py > ../logs/C_eval.log 2>&1 || echo FAIL_eval
echo C_DONE
