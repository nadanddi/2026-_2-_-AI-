"""Pin final pre-reviewed code/preparation before any actual residual training."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (H/'fit_audit_v1.json').exists()
OUT=ROOT/'집/코덱스/local'/H.name
assert not list(OUT.glob('*_pred.csv')) and not list(OUT.glob('*_model.npz'))
assert sha(H/'run_v3.py')=='be6bab3303e37decad5bd078e56edf569cda5bf0baaea470df971bc53043b172'
assert sha(H/'preparation_v3.json')=='df4ff2c3b7617676c61e4cee4b8e9690f9312a7cca448c0caec189cb5051b860'
assert sha(H/'verify_full_v3.py')=='6875fc1d4062d52edbbda6733c72977d45fe9f52b3daf3046317042e212979bd'
assert sha(H/'verification_math_v4.py')=='6e16ecf93a3d15cf8fe0c77bda55f647c6a360668c694adbf461b0cc0a8152ba'
prep=json.loads((H/'preparation_v3.json').read_text(encoding='utf-8'))
names=['run_v3.py','preparation_v3.json','preregistration_v1.md','verify_full_v3.py','verification_math_v4.py','synthetic_run_v3.json','verification_math_synthetic_v4.json','independent_preparation_v1.json','adversarial_whole_result_v1.json','adversarial_whole_v1.py','register_verifier_v1.py']
assert json.loads((H/'independent_preparation_v1.json').read_text())['status']=='PASS_INDEPENDENT_PREPARED_IDS_FEATURES_MOMENTS'
crit=json.loads((H/'adversarial_whole_result_v1.json').read_text());assert all(x['rejected'] for x in crit['rejections']) and len(crit['rejections'])==19
for k,h in crit['source_sha256'].items():assert sha(H/k)==h
receipt=dict(status='PINNED_BEFORE_ACTUAL_FIT',runner_file='run_v3.py',preparation_file='preparation_v3.json',files={str((H/n).relative_to(ROOT)):sha(H/n) for n in names},config_sha256=hashlib.sha256(json.dumps(prep['config'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),models_fit=0,score_count=0)
with (H/'verification_registration_v1.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print('PINNED_BEFORE_ACTUAL_FIT')
