#!/bin/bash
set -euo pipefail
source /home/adilab/anaconda3/etc/profile.d/conda.sh
conda activate nadan
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" PYTHONDONTWRITEBYTECODE=1
python -B -u /home/adilab/storage/nadan/codex/im_meta_probe_20261016_v1/provenance_v1.py
