#!/bin/bash
set -euo pipefail
BASE="$HOME/storage/nadan/codex/JOB003_FT2_20261016_v1"
export CONDA_ENVS_PATH="$HOME/storage/nadan/codex/conda_envs"
export CONDA_PKGS_DIRS="$HOME/storage/nadan/codex/conda_pkgs"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate nadan_codex
cd "$BASE"
test ! -e STATUS.txt
echo 'e27e7410183bd9395f9aeca13a58dbf8e259ee69601df9030cacacec4f98f151  bundle_v1.zip' | sha256sum -c -
python - <<'PY'
from pathlib import Path
import zipfile, hashlib, json
root = Path('workspace').resolve()
root.mkdir(exist_ok=True)
with zipfile.ZipFile('bundle_v1.zip') as z:
    for i in z.infolist():
        dest = (root / i.filename).resolve()
        assert dest.is_relative_to(root), i.filename
    z.extractall(root)
    checks = []
    for i in z.infolist():
        if not i.is_dir():
            a, b = z.read(i), (root / i.filename).read_bytes()
            assert a == b, i.filename
            checks.append({'file': i.filename, 'sha256': hashlib.sha256(b).hexdigest()})
Path('bundle_integrity_v1.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2))
(root / 'CLAUDE.md').touch(exist_ok=True)
print('bundle files verified:', len(checks))
PY
mkdir -p model_cache
cp workspace/_tabpfn_cache/tabpfn-v2-regressor.ckpt model_cache/
python -V
export CUDA_VISIBLE_DEVICES=0
python - <<'PY'
import torch, tabpfn
assert torch.cuda.is_available()
print('GPU:', torch.cuda.get_device_name(0))
PY
test -z "$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader)"
echo RUNNING > STATUS.txt
nohup bash run_server_v1.sh > launcher_v1.log 2>&1 < /dev/null &
echo $! > PID.txt
cat PID.txt
