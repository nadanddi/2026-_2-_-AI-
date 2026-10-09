"""Fresh streaming/f-sum cross-check, no producer import."""
import csv,json,math,hashlib,sys
from pathlib import Path
from collections import defaultdict
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
ROOT=H.parents[3]
L=ROOT/'집/코덱스/local/ec_high_training_hypothesis_20261009_v1/folds'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
truth=defaultdict(list);meta=[]
for k in range(10):
    p=L/f'fold{k}.csv';m=json.loads(p.with_suffix('.json').read_text(encoding='utf8'))
    assert sha(p)==m['sha'] and m['registration']==sha(H/'registration_v4.json')
    meta.append(m)
    for r in csv.DictReader(p.open(encoding='utf8',newline='')):
        if r['arm']=='BASE' and r['seed']=='ensemble':truth[(r['farm'],int(r['day']))].append(float(r['sub_ec']))
assert len(truth)==360 and all(len(v)==24 for v in truth.values())
high={key:math.fsum(v)/24>=1 for key,v in truth.items()}
loss=defaultdict(list);raw=defaultdict(list)
for k in range(10):
    for r in csv.DictReader((L/f'fold{k}.csv').open(encoding='utf8',newline='')):
        key=(r['farm'],int(r['day']));y=float(r['sub_ec']);err=(float(r['prediction'])-y)**2
        scopes=['all','high' if high[key] else 'normal']
        if key[1]>=179 and not high[key]:scopes.append('pass2_normal')
        for scope in scopes:
            idx=(scope,r['arm'],r['seed']);loss[idx].append(err);raw[idx].append((float(r['raw_et'])-y)**2)
expected=json.loads((H/'final_score_v1.json').read_text(encoding='utf8'))
groups=[]
for key,vals in sorted(loss.items()):
    scope,arm,seed=key;rmse=math.sqrt(math.fsum(vals)/len(vals))
    match=[g for g in expected['groups'] if (g['scope'],g['arm'],g['seed'])==key]
    assert len(match)==1 and abs(match[0]['rmse']-rmse)<1e-12 and len(vals)==match[0]['rows']
    groups.append(dict(scope=scope,arm=arm,seed=seed,rows=len(vals),rmse=rmse,raw_et_rmse=math.sqrt(math.fsum(raw[key])/len(vals))))
nfits=sum(bool(i['new_fit']) for m in meta for i in m['fitinfo'])
assert nfits==90
out=dict(status='PASS',days=360,rows=8640,normal_days=sum(not v for v in high.values()),high_days=sum(high.values()),candidate_fits=nfits,baseline_maxdiff=max(m['baseline_final_maxdiff'] for m in meta),groups=groups,method='streaming CSV/math.fsum; producer not imported',script_sha=sha(Path(__file__)),final_score_sha=sha(H/'final_score_v1.json'))
with (H/'root_final_recheck_v1.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in out.items() if k!='groups'},ensure_ascii=False))
for g in groups:
    if g['seed']=='ensemble':print(json.dumps(g,ensure_ascii=False))
