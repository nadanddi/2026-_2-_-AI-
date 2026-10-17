#!/bin/bash
# 서버: 밤샘 생성 (GPU 0) — ① Qwen-Image-Edit 지시문 편집 700장(nadan_qwen) → ② 되붙임 국소 편집(6판 방식, 마스크 10~30%) 400장(nadan)
source ~/anaconda3/etc/profile.d/conda.sh
cd ~/storage/nadan/farmai_img/data; export PYTHONIOENCODING=utf-8
echo "== $(date '+%F %T') nvidia-smi =="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=0
conda activate nadan_qwen
python -W ignore -u image/im31_qwen_edit.py 700 8 || echo FAIL_qwen
echo "== $(date '+%F %T') Qwen 끝 =="
conda activate nadan
python -W ignore -u image/im15d_pgen_v7.py 400 || echo FAIL_v7
echo NIGHT_DONE
