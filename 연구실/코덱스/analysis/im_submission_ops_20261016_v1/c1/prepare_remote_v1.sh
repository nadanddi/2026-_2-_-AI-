#!/bin/bash
set -euo pipefail
source /home/adilab/anaconda3/etc/profile.d/conda.sh
conda activate nadan
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/adilab/storage/nadan/codex/im_submission_ops_20261016_v1/c1
python -B -u prepare_features_v1.py generate > generate_v1.log 2>&1
nvidia-smi --id=0 > gpu_before_features_v1.log
apps=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ -n "$apps" ]]; then echo GPU0_BUSY_STOP;exit 2;fi
python -B -u prepare_features_v1.py features > features_v1.log 2>&1
echo C1_PREPARE_DONE
