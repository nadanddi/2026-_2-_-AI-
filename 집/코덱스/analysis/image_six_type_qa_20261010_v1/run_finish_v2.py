import sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
reg=json.loads((HERE/'resume_registration_v1.json').read_text(encoding='utf-8'))
for name,digest in reg['pinned_files'].items():
    if sha(HERE/name)!=digest:raise ValueError('resume registered file changed: '+name)
training=json.loads((HERE/'training_registration_v1.json').read_text(encoding='utf-8'))
if training['qa_verification_sha256']!=sha(HERE/'qa_verification_v2.json'):raise ValueError('registered QA2 hash')
if not json.loads((HERE/'qa_verification_v2.json').read_text(encoding='utf-8'))['all_checks_pass']:raise ValueError('registered QA2 failed')
first=json.loads((HERE/'train_first_v1.json').read_text(encoding='utf-8'))
if sha(first['checkpoint'])!=first['checkpoint_sha256']:raise ValueError('first checkpoint changed')
receipt={'registered_validation_file':'qa_verification_v2.json','registered_validation_sha_match':True,'validation_PASS':True,'trainer_v1_also_reads':'qa_verification_v1.json','all_pinned_files_verified':list(reg['pinned_files']),'first_checkpoint_sha_match':True,'scope':'explicit enforcement of registered stronger QA2 before resume; original first fit used weaker QA1 gate'}
with (HERE/'resume_preflight_v1.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
import train_contract_v3
train_contract_v3.fit('finish')
