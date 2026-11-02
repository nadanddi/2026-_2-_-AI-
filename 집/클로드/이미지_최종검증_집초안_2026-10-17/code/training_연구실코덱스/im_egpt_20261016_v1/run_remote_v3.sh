#!/bin/bash
set -eu
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nadan
export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
cd ~/storage/nadan/codex/im_egpt_20261016_v1
python -u run_v3.py
