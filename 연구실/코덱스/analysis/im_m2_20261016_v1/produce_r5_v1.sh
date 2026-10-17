#!/bin/bash
set -euo pipefail
source /home/adilab/anaconda3/etc/profile.d/conda.sh
conda activate nadan
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/adilab/storage/nadan/codex/im_m2_20261016_v1
SRC=/home/adilab/storage/nadan/farmai_img/data/image
check_gpu() {
  nvidia-smi --id=0
  apps=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader,nounits)
  if [[ -n "$apps" ]]; then echo "GPU0_BUSY_STOP"; exit 2; fi
}
echo "R5_EXTRACT_START"
python -B -u r5_mechanical_v1.py extract
echo "R5_CPU_START"
python -B -u "$SRC/im64_make_r5.py" cpu > r5_cpu.log 2>&1
echo "R5_CPU_EXIT_ZERO"
check_gpu > gpu_before_generation.log 2>&1
echo "R5_GPU_START"
python -B -u "$SRC/im64_make_r5.py" gpu > r5_gpu.log 2>&1
echo "R5_GPU_EXIT_ZERO"
python -B -u "$SRC/im64_make_r5.py" merge > r5_merge.log 2>&1
python -B -u r5_mechanical_v1.py images
check_gpu > gpu_before_features.log 2>&1
echo "R5_FEATURES_START"
python -B -u "$SRC/im43_h_feats.py" im_set_r5 > r5_features.log 2>&1
python -B -u r5_mechanical_v1.py features
echo "R5_SEALED_READY"
