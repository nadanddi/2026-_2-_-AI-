#!/bin/bash
set -euo pipefail
cd /home/adilab/storage/nadan/codex/im_n10_n11_prepare_20261016_v1
while [[ -n "$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader,nounits)" ]];do
  if [[ $(date +%H%M) -ge 1730 ]];then echo DEADLINE_GPU_BUSY_STOP;exit 2;fi
  sleep 15
done
bash run_v4.sh pub
bash finish_checks_v1.sh
bash run_v4.sh r2r5
echo ALL_GPU_CHECKS_DONE
