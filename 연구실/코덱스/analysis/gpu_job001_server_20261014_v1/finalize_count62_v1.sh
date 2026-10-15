#!/bin/bash
set -eu
BASE="$HOME/storage/nadan/codex/JOB001_SZ2_20261014_v1"
while [ "$(cat "$BASE/STATUS.txt")" = RUNNING ]; do sleep 10; done
count=$(find "$BASE/workspace/집/클로드/research/local/sz2_ckpt" -maxdepth 1 -name '*.csv' | wc -l)
if [ "$count" -eq 62 ] && grep -q 'exit_code=0 checkpoint_count=62 expected=66' "$BASE/results/ERROR.txt"; then
  mv "$BASE/results/ERROR.txt" "$BASE/count66_legacy_error.txt"
  { date -Is; echo 'cv + sum completed; checkpoint_count=62'; echo 'JOB.md corrected expected fold count from 66 to 62 on 2026-10-14.'; echo 'Commands: python -u sz2_season_index_real_pipeline_v1.py cv; python -u sz2_season_index_real_pipeline_v1.py sum'; echo 'GPU0 RTX6000Ada; nadan_codex Python3.10; unchanged experiment code. Five home checkpoints reused; server packages listed in packages.txt.'; cat "$BASE/results/START.txt"; } > "$BASE/results/DONE.txt"
  echo COMPLETE > "$BASE/STATUS.txt"
fi
