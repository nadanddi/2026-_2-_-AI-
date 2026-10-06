"""Independent saved-row SG2 status and score-case/subset scope; fit zero."""
from pathlib import Path
import csv,json,collections,math
H=Path(__file__).resolve().parent;L=H.parents[3]/'연구실/코덱스/local'/H.name
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
rows=read(L/'baseline_rows.csv');counts=collections.defaultdict(collections.Counter)
for r in rows:
    c=counts[(r['seed'],r['farm'])];present=bool(r['candidate_day']);gate=r['gate']=='True';p2=int(r['day'])>=179
    state='pass1_inactive' if not p2 else ('pass2_no_candidate' if not present else ('pass2_gate_true' if gate else 'pass2_gate_false'))
    c[state]+=1
    if r['delta'] and abs(float(r['delta']))>1e-12:c['nonzero_delta']+=1
    c['rows']+=1
D=H/'baseline_score_v1';days=read(D/'days.csv');cases=read(D/'cases.csv')
keys=lambda r:(r['arm'],r['seed'],r['farm'],int(r['day']))
lookup={keys(r):r for r in days};targets={('F47',160),('F47',161),('F13',98),('F13',112)}
assert len(cases)==32 and len({keys(r) for r in cases})==32
assert {keys(r) for r in cases}=={k for k in lookup if (k[2],k[3]) in targets}
for r in cases:
    assert r.keys()==lookup[keys(r)].keys()
    for k,v in r.items():assert v==lookup[keys(r)][k],(keys(r),k)
groups=read(D/'groups.csv');lookupg={(r['arm'],r['seed'],r['group']):r for r in groups}
for r in groups:
    b=lookupg[('BASE',r['seed'],r['group'])]
    assert float(r['base_sse'])==float(b['sse']) and float(r['base_rmse'])==float(b['rmse'])
scope={g:int(lookupg[('BASE','ensemble',g)]['days']) for g in ['all','ordinary','high','ordinary_closed61','ordinary_other268','ordinary_closed_fanlow50','pass1','pass2_public46','F13','F47']}
assert scope==dict(all=360,ordinary=329,high=31,ordinary_closed61=61,ordinary_other268=268,ordinary_closed_fanlow50=50,pass1=314,pass2_public46=46,F13=180,F47=180)
out=dict(status='PASS_BASELINE_SCOPE_AND_CASE_SUBSET',fit=0,SG2_by_seed_farm=[dict(seed=s,farm=f,**c) for (s,f),c in counts.items()],cases=32,scope=scope)
with (H/'critic_verify_baseline_scope_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False))
