#!/bin/bash
set -euo pipefail
export PS1=""
BASE="$HOME/storage/nadan/codex/JOB003_FT2_20261016_v2"
export CONDA_ENVS_PATH="$HOME/storage/nadan/codex/conda_envs"
export CONDA_PKGS_DIRS="$HOME/storage/nadan/codex/conda_pkgs"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate nadan_codex
cd "$BASE"
test ! -e STATUS.txt
echo 'a307490f6a3a24f8455bbe3c96be29644b5f726308bd62dac045b00db670db70  bundle_v2.zip' | sha256sum -c -
python - <<'PY'
from pathlib import Path
import zipfile, hashlib, json
root = Path('workspace').resolve()
root.mkdir(exist_ok=True)
with zipfile.ZipFile('bundle_v2.zip') as z:
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
python - <<'PY'
from pathlib import Path
import re, json, hashlib
root = Path('workspace').resolve()
files = [p for p in root.rglob('*.py') if not p.is_symlink()]
by_name = {}
for p in files:
    by_name.setdefault(p.name.casefold(), []).append(p)
aliases = []
for p in files:
    for name in set(re.findall(r'[\w-]+\.py', p.read_text(encoding='utf-8-sig'))):
        local = p.parent / name
        if not local.exists():
            matches = [q for q in by_name.get(name.casefold(), []) if q.parent == p.parent]
            if len(matches) == 1:
                local.symlink_to(matches[0].name)
                aliases.append({'alias': str(local), 'target': matches[0].name})
checks = json.loads(Path('bundle_integrity_v1.json').read_text())
for c in checks:
    assert hashlib.sha256((root / c['file']).read_bytes()).hexdigest() == c['sha256'], c['file']
Path('case_aliases_v2.json').write_text(json.dumps(aliases, ensure_ascii=False, indent=2))
print('Windows filename compatibility aliases:', aliases)
print('All original bundle files unchanged:', len(checks))
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
