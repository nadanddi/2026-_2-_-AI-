from pathlib import Path
import zipfile,hashlib,json
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OUT=ROOT/'집/코덱스/local'/H.name
names=['run_v1.py','run_v2.py','run_v3.py','verify_v1.py','verify_v2.py','verify_v3.py','preparation_v1.json','preparation_v3.json','prelaunch_v1.py','prelaunch_v2.py','prelaunch_v2.json','prelaunch_v3.py','prelaunch_v4.py','prepare_v1.log','prepare_v3.log','upgrade_v2.py','upgrade_v3.py','upgrade_v4.py','partial_audit_v1.log']
names=[n for n in names if (H/n).is_file()];p=OUT/'draft_history_v1.zip'
with zipfile.ZipFile(p,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for n in names:z.write(H/n,n)
with zipfile.ZipFile(p) as z:
    assert z.testzip() is None
    for n in names:assert hashlib.sha256(z.read(n)).hexdigest()==hashlib.sha256((H/n).read_bytes()).hexdigest()
record=dict(status='PASS_DRAFT_HISTORY_PRESERVED_NO_TRAINING_CHANGE',path=str(p.relative_to(ROOT)),sha=hashlib.sha256(p.read_bytes()).hexdigest(),files=names,bytes=p.stat().st_size)
with (H/'draft_history_v1.json').open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
print(json.dumps(record,ensure_ascii=False))
