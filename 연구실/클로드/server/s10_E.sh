#!/bin/bash
# 서버: 후보 묶음 E 특징 추출(GPU 0) → 비교
source ~/anaconda3/etc/profile.d/conda.sh; conda activate nadan
cd ~/storage/nadan/farmai_img/data; export PYTHONIOENCODING=utf-8
echo "== $(date '+%F %T') nvidia-smi =="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=0
PUB=~/storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플
for v in B2c20 B2c20f B2f SG384; do
  python -W ignore image/im16b_feats_E.py $v public21 $PUB/images $PUB/sample_labels.csv || { echo "FAIL $v"; continue; }
  for s in im_set_v4 im_set_sd2 im_set_v5; do python -W ignore image/im16b_feats_E.py $v $s local/$s/img local/$s/manifest.csv; done
done
python -W ignore -u image/im05o_eval_E.py B2,B2c20,B2c20f,B2f,SG384 > ../logs/E_eval.log 2>&1
echo E_DONE
