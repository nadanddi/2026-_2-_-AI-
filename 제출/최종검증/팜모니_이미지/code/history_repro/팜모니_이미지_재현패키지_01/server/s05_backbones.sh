#!/bin/bash
# 서버: 백본 7종 특징 추출(GPU 0, 가짜 5판 생성과 같은 GPU 공유) → 비교 평가(CPU)
source ~/anaconda3/etc/profile.d/conda.sh; conda activate nadan
cd ~/storage/nadan/farmai_img/data; export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
GPU=0; echo "== $(date '+%F %T') nvidia-smi =="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=$GPU
PUB=~/storage/nadan/공용/대회자료/이미지데이터/참가자_배포/공개샘플
for b in B1 B2 B3 B4 B5 B6 B7; do
  python -W ignore image/im16_backbone_feats.py $b public21 $PUB/images $PUB/sample_labels.csv || { echo "FAIL $b"; continue; }
  for s in im_set_v4 im_set_sd2; do python -W ignore image/im16_backbone_feats.py $b $s local/$s/img local/$s/manifest.csv; done
done
python -W ignore -u image/im05k_backbones.py B1,B2,B3,B4,B5,B6,B7 > ../logs/backbones_eval.log 2>&1
echo BB_DONE
