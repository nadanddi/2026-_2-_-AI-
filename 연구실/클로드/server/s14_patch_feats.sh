#!/bin/bash
# 서버: 후보 L 조각 특징(ViT-B/16, 54조각) 추출 (IM_국소패치_L_v1.md)
source ~/anaconda3/etc/profile.d/conda.sh; conda activate nadan
cd ~/storage/nadan/farmai_img/data; export PYTHONIOENCODING=utf-8
echo "== $(date '+%F %T') nvidia-smi =="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=0
PUB=~/storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플
python -W ignore image/im23_patch_feats.py public21 $PUB/images $PUB/sample_labels.csv || echo FAIL_pub
for s in im_set_gpt1 im_set_gpt2 im_set_v6 im_set_sd2 im_set_v4 im_set_v5; do python -W ignore image/im23_patch_feats.py $s local/$s/img local/$s/manifest.csv || echo FAIL_$s; done
echo P16_DONE
