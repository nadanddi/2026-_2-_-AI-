"""Atomic prediction receipts, strict reuse and independently recomputed scores."""
from pathlib import Path
import hashlib
import json
import math
import os
import uuid

REQUIRED={'input_sha256','code_sha256','environment_sha256','fold_sha256',
          'train_ids_sha256','anchor_ids_sha256','context_ids_sha256','query_ids_sha256',
          'baseline_sha256','postprocess_sha256','seed','candidate_id','validator','fold'}
def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def validate(contract):
    if set(contract)!=REQUIRED:raise ValueError('checkpoint contract fields mismatch')
    for key in REQUIRED:
        if key.endswith('sha256') and (not isinstance(contract[key],str) or len(contract[key])!=64):
            raise ValueError('invalid hash: '+key)
def atomic(path, text):
    path=Path(path);tmp=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    with tmp.open('x',encoding='utf-8') as f:
        f.write(text);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def complete(folder,contract,ids,y,pred):
    validate(contract);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'receipt.json').exists():raise FileExistsError('completed cell is immutable')
    if len(ids)!=len(set(ids)) or len(ids)!=len(y) or len(ids)!=len(pred):raise ValueError('IDs/shape')
    if digest(ids)!=contract['query_ids_sha256']:raise ValueError('query IDs')
    rows=[{'row_id':rid,'y':float(a),'prediction':float(b)} for rid,a,b in zip(ids,y,pred)]
    if not rows or not all(math.isfinite(r['y']) and math.isfinite(r['prediction']) for r in rows):raise ValueError('nonfinite/empty')
    blob=json.dumps(rows,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    atomic(folder/'predictions.json',blob)
    receipt={'status':'COMPLETE','contract':contract,'contract_sha256':digest(contract),
             'prediction_sha256':hashlib.sha256(blob.encode()).hexdigest(),
             'RMSE':math.sqrt(math.fsum((r['prediction']-r['y'])**2 for r in rows)/len(rows)),'rows':len(rows)}
    atomic(folder/'receipt.json',json.dumps(receipt,sort_keys=True,ensure_ascii=False))
    return reuse(folder,contract)
def reuse(folder,contract):
    validate(contract);folder=Path(folder)
    receipt=json.loads((folder/'receipt.json').read_text(encoding='utf-8'))
    if receipt['status']!='COMPLETE' or receipt['contract']!=contract or receipt['contract_sha256']!=digest(contract):
        raise ValueError('receipt contract mismatch')
    raw=(folder/'predictions.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=receipt['prediction_sha256']:raise ValueError('prediction checksum')
    rows=json.loads(raw)
    if len(rows)!=receipt['rows'] or digest([r['row_id'] for r in rows])!=contract['query_ids_sha256']:raise ValueError('receipt IDs')
    if not rows or not all(math.isfinite(r['y']) and math.isfinite(r['prediction']) for r in rows):raise ValueError('nonfinite')
    score=math.sqrt(math.fsum((r['prediction']-r['y'])**2 for r in rows)/len(rows))
    if score!=receipt['RMSE']:raise ValueError('saved metric mismatch')
    return receipt

if __name__=='__main__':
    here=Path(__file__).resolve().parent
    import tempfile
    with tempfile.TemporaryDirectory(prefix='resume_synthetic_',dir=here) as tmp:
        ids=['synthetic_a','synthetic_b'];contract={k:digest(k) for k in REQUIRED}
        contract.update(seed=47,candidate_id='SYNTHETIC_TEST_ONLY',validator='SYNTHETIC',fold=0,query_ids_sha256=digest(ids))
        r=complete(tmp,contract,ids,[0,1],[.1,.9]);assert reuse(tmp,contract)==r
        rejected=[]
        for key in REQUIRED:
            c=dict(contract);c[key]=digest('changed') if key.endswith('sha256') else 'changed'
            try:reuse(tmp,c)
            except ValueError:rejected.append(key)
            else:raise AssertionError(key)
        p=Path(tmp)/'predictions.json';original=p.read_bytes();p.write_bytes(original+b' ')
        try:reuse(tmp,contract)
        except ValueError:rejected.append('corrupt_predictions')
        else:raise AssertionError('corruption accepted')
        p.write_bytes(original)
        receipt=Path(tmp)/'receipt.json';old=receipt.read_bytes();receipt.unlink()
        try:reuse(tmp,contract)
        except FileNotFoundError:rejected.append('missing_receipt_partial_cell')
        else:raise AssertionError('partial accepted')
        receipt.write_bytes(old)
    out=here/'checkpoint_selftest_v1.json';assert not out.exists()
    out.write_text(json.dumps({'status':'PASS','rejected_cases':rejected,'synthetic_only':True,
        'limitations':'실제 학습 runner·프로세스 생존/재시작 감사는 별도 필요'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','rejected_cases':len(rejected)}))
