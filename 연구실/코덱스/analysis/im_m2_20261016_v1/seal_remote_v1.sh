#!/bin/bash
set -euo pipefail
source /home/adilab/anaconda3/etc/profile.d/conda.sh
conda activate nadan
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="" PYTHONDONTWRITEBYTECODE=1
cd /home/adilab/storage/nadan/codex/im_m2_20261016_v1
test -f r5_integrity_v1.json
test ! -e R5_OPEN_ONCE.json
python -B -u seal_files_v2.py
tar -cf r5_artifacts_v1.tar -C /home/adilab/storage/nadan/farmai_img/data/local im_set_r5 feats_h/im_set_r5.npz
echo "SEALED_BACKUP_READY"
