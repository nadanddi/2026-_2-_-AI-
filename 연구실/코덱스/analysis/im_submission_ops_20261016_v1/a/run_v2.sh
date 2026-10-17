#!/bin/bash
set -euo pipefail
source /home/adilab/anaconda3/etc/profile.d/conda.sh
conda activate nadan
export PYTHONPATH="" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/adilab/storage/nadan/codex/im_submission_ops_20261016_v1/a
if [[ "$1" == sample || "$1" == score ]]; then
  export CUDA_VISIBLE_DEVICES=0
  nvidia-smi --id=0 > "gpu_before_${1}_v1.log"
  apps=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader,nounits)
  if [[ -n "$apps" ]]; then echo GPU0_BUSY_STOP;exit 2;fi
else export CUDA_VISIBLE_DEVICES="";fi
case "$1" in
sample) python -B -u sample_v1.py ;;
first|rest) python -B -u train_v1.py "$1" ;;
score) python -B -u score_eval_v2.py ;;
*) exit 2 ;;
esac
