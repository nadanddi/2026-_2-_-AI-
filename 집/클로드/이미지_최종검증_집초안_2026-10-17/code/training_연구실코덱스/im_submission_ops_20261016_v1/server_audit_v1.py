"""Capture executed source hashes and server environment after jobs finish."""
import sys,json,hashlib,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
names=('c1/prepare_features_v1.py','c1/train_eval_v2.py','c1/train_eval_v3.py','a/tokens_v1.py','a/sample_v1.py','a/train_v1.py','a/eval_support_v1.py','a/score_eval_v2.py','fallback/check_v1.py','verify_results_v1.py')
out=dict(python=sys.version,source_sha256={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names},gpu_state=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True))
with (HERE/'server_audit_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print('SERVER_AUDIT_SAVED',len(names),flush=True)
