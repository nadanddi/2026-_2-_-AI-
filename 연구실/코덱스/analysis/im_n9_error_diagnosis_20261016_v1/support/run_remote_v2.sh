#!/bin/bash
set -euo pipefail
source /home/adilab/anaconda3/etc/profile.d/conda.sh
conda activate nadan
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/adilab/storage/nadan/codex/im_n9_error_diagnosis_20261016_v1
python -B -u scan_v2.py
