#!/bin/bash
set -euo pipefail
BASE=/home/adilab/storage/nadan/codex/im_submission_ops_20261016_v1
cd "$BASE/a"
while ! grep -q C1_PREPARE_DONE "$BASE/c1/prepare_v1.log"; do
  if ! kill -0 2539295 2>/dev/null; then
    grep -q C1_PREPARE_DONE "$BASE/c1/prepare_v1.log" || { echo C1_PREPARE_NOT_COMPLETE_STOP;exit 2; }
  fi
  sleep 10
done
bash run_v2.sh sample > sample_v1.log 2>&1
bash run_v2.sh first > first_train_v1.log 2>&1
echo A2_FIRST_REVIEW_READY
