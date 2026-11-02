#!/bin/bash
set -euo pipefail
source /home/adilab/anaconda3/etc/profile.d/conda.sh
conda activate nadan
export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="" PYTHONDONTWRITEBYTECODE=1
cd /home/adilab/storage/nadan/codex/im_meta_probe_20261016_v1
python -B -u run_v2.py "$1"
