"""Fresh row/label-presence/MASK and full scoring-population checks for 66 folds."""
import csv,json,hashlib
from pathlib import Path
from collections import defaultdict,Counter
from blk_context_v1 import key,RAW
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DATA=ROOT/'공용/대회자료/정형데이터/참가자_배포'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
registry_path=HERE/'original_validator_endpoint_registry_v1.json'
reg=json.loads(registry_path.read_text(encoding='utf-8'))
assert all(sha(DATA/n)==s for n,s in reg['source_sha256'].items())
X={}
with (DATA/'train_X.csv').open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        if key(r['row_id'])[0] in ['F13','F47']:
            assert r['row_id'] not in X
            X[r['row_id']]=set(r)
present=set()
with (DATA/'train_y.csv').open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        if key(r['row_id'])[0] in ['F13','F47'] and r['sub_ec'].strip():
            assert r['sub_temp'].strip() and r['row_id'] not in present
            present.add(r['row_id'])
assert set(X)==present and len(X)==9600
with (DATA/'test_X.csv').open(encoding='utf-8-sig',newline='') as f:test=list(csv.DictReader(f))
mask=[c for c in test[0] if c!='row_id' and all(not r[c].strip() for r in test)]
assert set(test[0])-{'row_id'}-set(mask)==set(RAW)
assert len(mask)==5
foldmeta=json.loads((HERE/'fold_registry_v1.json').read_text(encoding='utf-8'))
tm={tuple(v) for v in foldmeta['TM_days']}
query_by_validator=defaultdict(list);TM=[]
for original,registered in zip(foldmeta['folds'],reg['folds']):
    assert original['validator']==registered['validator'] and original['fold']==registered['fold']
    assert set(original['query_ids'])==set(registered['ordered_query_ids'])
    tr=set(registered['ordered_train_ids']);va=set(registered['ordered_query_ids']);forbidden=set(registered['input_forbidden_ids'])
    assert not(tr&va or tr&forbidden or va&forbidden) and tr|va|forbidden==present
    assert tr<=set(X) and va<=set(X)
    for f,d in {key(r)[:2] for r in tr|va}:
        assert {key(r)[2] for r in tr|va if key(r)[:2]==(f,d)}==set(range(24))
    query_by_validator[registered['validator']].extend(registered['ordered_query_ids'])
    if registered['validator']=='DIAG10':TM.extend(r for r in registered['ordered_query_ids'] if key(r)[:2] in tm)
counts={k:len(v) for k,v in query_by_validator.items()}
assert counts=={'DIAG10':8640,'P2LOO':1104,'EL1':1104} and len(TM)==len(set(TM))==2664
assert all(len(v)==len(set(v)) for v in query_by_validator.values())
out=HERE/'original_validator_integrity_audit_v1.json';assert not out.exists()
out.write_text(json.dumps({'status':'PASS','folds':66,'F13_F47_X_equals_EC_label_keys':9600,'both_labels_present':9600,
    'evaluation_all_empty_MASK_columns':mask,'model_original_inputs':RAW,'validator_query_rows':counts,'TM_query_rows':len(TM),
    'duplicate_query_ids_within_each_validator':0,'train_query_gap_disjoint':True,'numeric_heldout_labels_parsed':False,
    'source_sha256':{n:sha(DATA/n) for n in ['train_X.csv','train_y.csv','test_X.csv']},'registry_sha256':sha(registry_path),
    'limits':['structural loader/population check; actual per-fold model/context execution pending','validators are separate populations with repeated IDs across them; never pool as independent rows']},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','validator_query_rows':counts,'TM':len(TM)}),flush=True)
