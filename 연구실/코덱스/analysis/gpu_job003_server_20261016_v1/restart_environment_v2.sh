#!/bin/bash
set -euo pipefail
export PS1=""
BASE="$HOME/storage/nadan/codex/JOB003_FT2_20261016_v1"
export CONDA_ENVS_PATH="$HOME/storage/nadan/codex/conda_envs"
export CONDA_PKGS_DIRS="$HOME/storage/nadan/codex/conda_pkgs"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate nadan_codex
cd "$BASE"
test "$(cat STATUS.txt)" = ERROR
python - <<'PY'
from pathlib import Path
import re, json, hashlib
root = Path('workspace')
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
test -z "$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader)"
mv results results_attempt1
cp STATUS.txt STATUS_attempt1.txt
echo RUNNING > STATUS.txt
nohup bash run_server_v2.sh > launcher_v2.log 2>&1 < /dev/null &
echo $! > PID_v2.txt
cat PID_v2.txt
