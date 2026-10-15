#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh; conda activate nadan
cd ~/storage/nadan/farmai_img/data; export PYTHONIOENCODING=utf-8
echo "== $(date '+%F %T') nvidia-smi =="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=0
python -W ignore image/im16_backbone_feats.py B2 im_set_r2 local/im_set_r2/img local/im_set_r2/manifest.csv
python -W ignore -u image/im08b_compare_v7.py > ../logs/cmp_v7.log 2>&1
echo CMP_DONE
