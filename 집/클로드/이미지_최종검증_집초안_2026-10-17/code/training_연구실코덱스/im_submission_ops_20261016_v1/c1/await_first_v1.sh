#!/bin/bash
set -euo pipefail
cd /home/adilab/storage/nadan/codex/im_submission_ops_20261016_v1/c1
while ! grep -q C1_PREPARE_DONE prepare_v1.log; do
  if ! kill -0 2539295 2>/dev/null; then
    grep -q C1_PREPARE_DONE prepare_v1.log || { echo PREPARE_NOT_COMPLETE_STOP; exit 2; }
  fi
  sleep 10
done
bash train_remote_v2.sh first > first_train_v1.log 2>&1
echo C1_FIRST_REVIEW_READY
