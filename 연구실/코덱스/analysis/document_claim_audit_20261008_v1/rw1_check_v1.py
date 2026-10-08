"""Read-only arithmetic check of completed official-data RW1; no refit, no EL1."""
from pathlib import Path
import csv,json,math,hashlib
from collections import defaultdict
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prep=json.loads((ROOT/'연구실/코덱스/analysis/ec_current14_influence_20261007_v1/preparation_v4.json').read_text(encoding='utf-8'))
allowed={i for f in prep['records'] for i in f['query_ids']}
y={r['row_id']:float(r['sub_ec']) for r in read(ROOT/'공용/대회자료/정형데이터/참가자_배포/train_y.csv') if r['row_id'] in allowed}
rough={(r['farm'],int(r['day'])) for r in read(H/'first_v4/day_inputs.csv') if r['group']=='train_inputs' and r['R2']=='True'}
unknown={(r['farm'],int(r['day'])) for r in read(H/'first_v4/day_inputs.csv') if r['group']=='train_inputs' and r['R2']==''}
rf=ROOT/'연구실/클로드/results/rw1_rough_days_v1.csv'
assert rough=={(r['farm'],int(r['day'])) for r in read(rf)} and len(rough)==30
p=ROOT/'연구실/클로드/local/ec_rw1_all_v1.csv';kept=[];seen=set()
for r in read(p):
    if r['validator'] not in ['DIAG10','A','B']:continue
    assert r['row_id'] in allowed
    assert float(r['sub_ec'])==y[r['row_id']]
    key=(r['validator'],r['validation_fold'],r['row_id']);assert key not in seen;seen.add(key)
    kept.append(r)
def pred(r,tag,s):
    ps=[float(r[f'{tag}_{c}_{s}']) for c in ['et','lgb','mlp']]
    assert all(map(math.isfinite,ps))
    return min(float(r['hi']),max(float(r['lo']),math.fsum(w*z for w,z in zip([.6,.3,.1],ps))))
stats=[]
for v in ['DIAG10','A','B']:
    rows=[r for r in kept if r['validator']==v]
    for scope in ['all','exclude_rough_original','exclude_rough_and_unknown']:
        z=[r for r in rows if scope=='all' or ((r['farm'],int(r['day'])) not in rough and (scope=='exclude_rough_original' or (r['farm'],int(r['day'])) not in unknown))]
        for s in ['47','1414','6464','ensemble']:
            errs={}
            for tag in ['b','c']:
                values=[pred(r,tag,s) if s!='ensemble' else math.fsum(pred(r,tag,ss) for ss in ['47','1414','6464'])/3 for r in z]
                errs[tag]=math.sqrt(math.fsum((p-y[r['row_id']])**2 for r,p in zip(z,values))/len(z))
            stats.append(dict(validator=v,scope=scope,seed=s,rows=len(z),days=len({r['row_id'][:7] for r in z}),base=errs['b'],candidate=errs['c'],percent=100*(errs['c']/errs['b']-1)))
out=H/'rw1_check_v1.csv'
with out.open('x',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
receipt=dict(status='PASS_PUBLIC_LABELS_ROUGH_SET_AND_WEIGHTED_COMPONENT_RMSE',source_sha=sha(Path(__file__)),input_sha=sha(p),rough_sha=sha(rf),output_sha=sha(out),rows=len(kept),cells=len(stats),all_scored_changes_positive=all(r['percent']>0 for r in stats),new_fit=0,EL1_scored=0,limitations=['No refit/training runtime reproduction','No new independent critic available due usage limit','R3 season+DP1 diagnostic; not full current EC14 with SG2'])
with (H/'rw1_check_v1.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
for r in stats:
    if r['scope']=='exclude_rough_original':print(r)
print('PASS',len(stats),'cells',len(kept),'rows')
