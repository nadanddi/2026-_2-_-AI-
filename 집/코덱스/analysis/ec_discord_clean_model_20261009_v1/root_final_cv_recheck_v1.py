"""Fresh CSV/math.fsum CV check and official/public-label consistency check."""
from pathlib import Path
import sys,csv,json,math,hashlib
from collections import defaultdict
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3];L=ROOT/'집/코덱스/local'/H.name
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=json.loads((H/'final_score_v1.json').read_text(encoding='utf8'));assert s['status']=='FULL_DIAG10'
reg=json.loads((H/'registration_v1.json').read_text(encoding='utf8'))
truth={};days=defaultdict(list);meta=[]
for k in range(10):
    p=L/'cv_predictions'/f'fold{k}.csv';m=json.loads(p.with_suffix('.json').read_text(encoding='utf8'));assert m['sha']==sha(p) and m['registration']==sha(H/'registration_v1.json');meta.append(m)
    for r in csv.DictReader(p.open(encoding='utf8',newline='')):
        if r['arm']=='BASE' and r['seed']=='ensemble':
            assert r['row_id'] not in truth;truth[r['row_id']]=float(r['sub_ec']);days[(r['farm'],int(r['day']))].append(float(r['sub_ec']))
assert len(truth)==8640 and len(days)==360 and all(len(v)==24 for v in days.values())
hi={key:math.fsum(v)/24>=1 for key,v in days.items()}
loss=defaultdict(list);removed=set();counts=defaultdict(int)
for k in range(10):
    for r in csv.DictReader((L/'cv_predictions'/f'fold{k}.csv').open(encoding='utf8',newline='')):
        key=(r['farm'],int(r['day']));remove=r['query_removed'].lower()=='true';err=(float(r['prediction'])-float(r['sub_ec']))**2
        scopes=['all','removed' if remove else 'filtered','high' if hi[key] else 'normal']
        if key[1]>=179:
            scopes.append('pass2_all')
            if not remove:scopes.append('pass2_filtered')
        if remove and r['arm']=='BASE' and r['seed']=='ensemble':removed.add(key)
        for scope in scopes:idx=(scope,r['arm'],r['seed']);loss[idx].append(err)
G={(g['scope'],g['arm'],g['seed']):g for g in s['groups']};groups=[]
for idx,vals in sorted(loss.items()):
    metric=math.sqrt(math.fsum(vals)/len(vals));assert len(vals)==G[idx]['rows'] and abs(metric-G[idx]['rmse'])<1e-12
    groups.append(dict(scope=idx[0],arm=idx[1],seed=idx[2],rows=len(vals),rmse=metric))
assert sum(len(m['fitinfo']) for m in meta)==90
# This runs only after CV score is fixed; parse public labels only for the consistency join.
source=Path(env.DATA)/'train_y.csv';assert sha(source)==reg['pins'][str(source)]
actual={r['row_id']:float(r['sub_ec']) for r in csv.DictReader(source.open(encoding='utf-8-sig',newline='')) if r['row_id'] in truth}
assert set(actual)==set(truth);gap=max(abs(actual[key]-truth[key]) for key in truth);assert gap<1e-12
out=dict(status='PASS',rows=8640,days=360,removed_query_days=len(removed),filtered_days=360-len(removed),candidate_model_count=90,groups=groups,official_vs_public_label_maxdiff=gap,official_extra40_labels_not_scored=True,method='stdlib streaming CSV/math.fsum, producer not imported',score_sha=sha(H/'final_score_v1.json'),script_sha=sha(Path(__file__)))
with (H/'root_final_cv_recheck_v1.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in out.items() if k!='groups'},ensure_ascii=False))
for g in groups:
    if g['seed']=='ensemble':print(json.dumps(g,ensure_ascii=False))
