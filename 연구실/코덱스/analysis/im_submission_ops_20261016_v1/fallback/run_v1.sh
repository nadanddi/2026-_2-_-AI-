#!/bin/bash
set -euo pipefail
source /home/adilab/anaconda3/etc/profile.d/conda.sh
conda activate nadan
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /home/adilab/storage/nadan/codex/im_submission_ops_20261016_v1/fallback
python -B -u check_v1.py > check_v1.log 2>&1
