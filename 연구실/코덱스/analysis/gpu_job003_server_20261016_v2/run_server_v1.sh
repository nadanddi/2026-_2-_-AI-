#!/bin/bash
set -uo pipefail
export PS1=""
BASE="$HOME/storage/nadan/codex/JOB003_FT2_20261016_v2"
export CONDA_ENVS_PATH="$HOME/storage/nadan/codex/conda_envs"
export CONDA_PKGS_DIRS="$HOME/storage/nadan/codex/conda_pkgs"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate nadan_codex || exit 90
python -V
python -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version' || exit 91
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=""
export XDG_CACHE_HOME="$BASE/cache"
export MPLCONFIGDIR="$BASE/cache/matplotlib"
export HF_HOME="$BASE/cache/huggingface"
export TORCH_HOME="$BASE/cache/torch"
export TABPFN_MODEL_CACHE_DIR="$BASE/model_cache"
export TABPFN_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$BASE/results"
cd "$BASE/workspace/집/클로드/research" || exit 1
date -Is > "$BASE/results/START.txt"
nvidia-smi > "$BASE/results/gpu_before.txt"
python -m pip freeze > "$BASE/results/packages.txt"
python -u ft2_past_only_training_temp_v1.py cv > "$BASE/results/ft2_lab_v1.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then
  python -u ft2_past_only_training_temp_v1.py sum > "$BASE/results/ft2_lab_sum_v1.log" 2>&1
  rc=$?
fi
python -m zipfile -c "$BASE/results/ft2_ckpt.zip" local/ft2_ckpt
count=$(find local/ft2_ckpt -maxdepth 1 -name '*.csv' | wc -l)
if [ "$rc" -eq 0 ] && [ "$count" -eq 82 ]; then
  { date -Is; echo 'cv + sum completed'; echo "checkpoint_count=$count"; echo 'CUDA_VISIBLE_DEVICES=0; nadan_codex Python 3.10'; echo 'Original experiment files unchanged. Server package versions in packages.txt; same Python3.10 environment as JOB001 and JOB002.'; } > "$BASE/results/DONE.txt"
  echo COMPLETE > "$BASE/STATUS.txt"
else
  { date -Is; echo "exit_code=$rc checkpoint_count=$count expected=82"; cat "$BASE/results/ft2_lab_v1.log"; test ! -f "$BASE/results/ft2_lab_sum_v1.log" || cat "$BASE/results/ft2_lab_sum_v1.log"; } > "$BASE/results/ERROR.txt"
  echo ERROR > "$BASE/STATUS.txt"
fi
