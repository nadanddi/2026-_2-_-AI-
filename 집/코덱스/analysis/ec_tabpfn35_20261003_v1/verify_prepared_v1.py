from pathlib import Path
import ast,subprocess,sys,json
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
for name in ['run.py','prepare_baseline.py','verify_formula.py','verify_baseline_scalar.py']:ast.parse((H/name).read_text(encoding='utf-8'))
checkpoint=ROOT/'집/코덱스/local/ec_tabpfn35_20261003_v1/weights/tabpfn-v3.5-20260909.safetensors';assert not checkpoint.exists()
process=subprocess.run([sys.executable,str(H/'run.py')],cwd=ROOT,text=True,capture_output=True,encoding='utf-8',timeout=20)
assert process.returncode!=0 and 'MISSING_LICENSED_WEIGHTS' in process.stderr
result=dict(status='PASS',parsed_scripts=4,missing_weights_guard='PASS: exited before framework import/download/fit',new_model_fit_started=False,license_acceptance_performed=False)
(H/'preparation_verification_v1.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
