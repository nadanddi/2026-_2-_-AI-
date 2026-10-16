#!/bin/bash
set -euo pipefail
source /home/adilab/anaconda3/etc/profile.d/conda.sh
conda activate nadan
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /home/adilab/storage/nadan/codex/im_n10_n11_prepare_20261016_v1
while [[ ! -f validation_r2r5_v1.json ]];do
  if (( 10#$(date +%H%M) >= 1730 ));then echo DEADLINE_STOP;exit 2;fi
  sleep 15
done
while [[ -n "$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader,nounits)" ]];do sleep 10;done
nvidia-smi --id=0 > gpu_before_upper_v1.log
python -B -u validate_upper_v1.py pub > validate_pub_upper_v1.log 2>&1
python -B -u self_checks_upper_v1.py > self_checks_upper_v1.log 2>&1
python -B -u validate_upper_ff_v1.py > validate_r2r5_upper_v1.log 2>&1
echo UPPER_ALL_CHECKS_DONE
