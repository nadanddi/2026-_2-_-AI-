from pathlib import Path
import json,hashlib,sys
H=Path(__file__).resolve().parent
prep=H.parent/'temp_round9_reproduction_20261008_v1/preparation_v1.json'
p=json.loads(prep.read_text(encoding='utf-8'))
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
code=Path(p['temp_code'])
for n,v in p['temp_package_files'].items():assert sha(code.parent/n)==v
for n,v in p['data_sha'].items():assert sha(code.parent/'data'/n)==v
assert sha(Path(p['cache'])/'tabpfn-v2-regressor.ckpt')==p['checkpoint_sha']
payload={'preparation_sha':sha(prep),'package_files':p['temp_package_files'],'package_table_sha':hashlib.sha256(json.dumps(p['temp_package_files'],sort_keys=True).encode()).hexdigest(),'runtime_sha':sha(H.parent/'temp_repair_20261009_v1/runtime_v1.py'),'script_sha':sha(H/'n4_run_v2.py'),'data_sha':p['data_sha'],'checkpoint_sha':p['checkpoint_sha'],'python':sys.version,'python_exe':sys.executable,'new_fit':0,'status':'SOURCE_INPUT_CHECKPOINT_CURRENT_MATCH'}
with (H/'n4_source_receipt_v1.json').open('x',encoding='utf-8') as f:json.dump(payload,f,ensure_ascii=False,indent=2)
print(payload['status'],len(payload['package_files']),payload['package_table_sha'])
