"""Install only fixed missing/conflicting packages in a new own-local overlay."""
from pathlib import Path
import json
import subprocess
import sys
import os

H = Path(__file__).resolve().parent
ROOT = H.parents[3]
OUT = ROOT/'집/코덱스/local/tabdpt130_cpu_v1'
SITE = OUT/'site'
ACCESS = ROOT/'집/코덱스/analysis/ec_followup_progress_20261004_v1/tabdpt_public_access_v2.json'
assert json.loads(ACCESS.read_text(encoding='utf-8'))['status']=='PASS'
assert OUT.resolve().is_relative_to((ROOT/'집/코덱스/local').resolve())
assert not SITE.exists(), 'Preserve any partial runtime; use a new version instead.'
assert not (H/'install_report_v1.json').exists()
H.mkdir(parents=True,exist_ok=True)
OUT.mkdir(parents=True,exist_ok=True)
os.environ['PYTHONPATH']=''
packages = ['faiss-cpu==1.12.0','omegaconf==2.3.0','antlr4-python3-runtime==4.9.3',
            'huggingface-hub==0.36.0',
            'tabdpt @ git+https://github.com/layer6ai-labs/TabDPT-inference.git@97e5494431e9527c7edb31cb4dcfc5f00b232fdf']
cmd = [sys.executable,'-m','pip','install','--disable-pip-version-check','--no-deps',
       '--no-warn-script-location','--target',str(SITE),'--cache-dir',str(OUT/'pip_cache'),
       '--report',str(H/'install_report_v1.json'),*packages]
print('Installing fixed 5-package own-local overlay; no model/data access.',flush=True)
subprocess.run(cmd,check=True)
print('OVERLAY_INSTALL_COMPLETE',flush=True)
