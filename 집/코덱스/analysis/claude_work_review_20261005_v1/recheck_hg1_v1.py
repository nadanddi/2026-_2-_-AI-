"""Public-only HG1 completion recheck. No EL1 numeric use or source execution."""
import sys,json,math,collections
sys.dont_write_bytecode=True
from recheck_public_v2 import H,ROOT,L,PUB,rows,sha,rmse,boot,np
pub={r['row_id']:float(r['sub_ec']) for r in rows(PUB) if r['validator']=='DIAG10' and r['seed']=='7'}
assert len(pub)==8640
p=L/'ec3_HG1_all.csv';selected=[];seeds=[29,909,1111]
for r in rows(p):
    if r['validator']=='EL1':continue
    assert r['validator'] in ['DIAG10','DIAG10z'] and r['row_id'] in pub
    q={k:r[k] for k in ['row_id','farm','day','hour','validator']};q['y']=float(r['sub_ec'])
    assert abs(q['y']-pub[r['row_id']])<=1e-12
    for k in [f'{a}_{s}' for a in ['sgb','sg'] for s in seeds]:q[k]=float(r[k]);assert math.isfinite(q[k])
    selected.append(q)
assert len(selected)==2208 and len({(r['validator'],r['row_id']) for r in selected})==2208
scores=[];segments=[]
for v in ['DIAG10','DIAG10z']:
    rs=[r for r in selected if r['validator']==v];assert len(rs)==1104
    days=collections.defaultdict(list)
    for r in rs:days[(r['farm'],r['day'])].append(r)
    assert len(days)==46
    for dd in days.values():
        dm=math.fsum(r['y'] for r in dd)/len(dd)
        for r in dd:r['dm']=dm
    for s in seeds:
        a=rmse([r[f'sgb_{s}'] for r in rs],[r['y'] for r in rs]);b=rmse([r[f'sg_{s}'] for r in rs],[r['y'] for r in rs])
        scores.append(dict(validator=v,seed=s,baseline=a,candidate=b,change_pct=100*(b/a-1)))
        for tag,use in [('high',[r for r in rs if r['dm']>=1]),('ordinary',[r for r in rs if r['dm']<1])]:
            a=rmse([r[f'sgb_{s}'] for r in use],[r['y'] for r in use]);b=rmse([r[f'sg_{s}'] for r in use],[r['y'] for r in use])
            segments.append(dict(validator=v,seed=s,segment=tag,rows=len(use),baseline=a,candidate=b,change_pct=100*(b/a-1)))
rs=[r for r in selected if r['validator']=='DIAG10z']
bm=[math.fsum(r[f'sgb_{s}'] for s in seeds)/3 for r in rs];cm=[math.fsum(r[f'sg_{s}'] for s in seeds)/3 for r in rs]
bb=boot(rs,bm,cm,np.random.default_rng(20261004));assert abs(bb['p_worse']-.0399)<.000051
result=dict(status='PASS_PUBLIC_HG1_RECHECK',rows=2208,scores=scores,segments=segments,bootstrap=bb,author_public_threshold_failed=bb['p_worse']>=.025,source_sha256=sha(ROOT/'집/클로드/research/ec3_HG1_high_gate_anchor_v1.py'),csv_sha256=sha(p),public_sha256=sha(PUB),verifier_sha256=sha(ROOT/'집/코덱스/analysis/claude_work_review_20261005_v1/recheck_hg1_v1.py'),fit=0,predict=0,raw_EC_reads=0,test_reads=0,EL1_rescore=0,limitations=['Arithmetic only; same SG2 preprocessing/reference caveats','R3S+SG2 baseline, not actual EC season_v2','Same46days and author pooled bootstrap; not Codex adoption'])
with (H/'hg1_public_recheck_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print('PASS_PUBLIC_HG1_RECHECK',bb)
