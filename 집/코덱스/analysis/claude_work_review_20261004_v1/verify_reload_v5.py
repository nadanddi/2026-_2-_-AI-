"""Fresh DP2 public-only recheck while family20 first audit runs; no refit/EL1 scoring."""
from pathlib import Path
import sys, json, math, hashlib, re
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'집/클로드/research')); import env
import numpy as np
import pandas as pd
H = Path(__file__).resolve().parent
SRC = ROOT/'집/클로드/research'
DEST = H/'reload_verification_v5.json'
assert not DEST.exists()
SEEDS = [7,101,2024]
VALIDATORS = ['DIAG10','A','B','EXT10','EXT12']
res = dict(status='PASS',git_checked='283fc25',checks=0,experiments={},scope='stored public OOF only; no EL1 scoring or new fit')
def close(x,y,tol=1e-12):
    assert abs(float(x)-float(y)) < tol,(x,y)
    res['checks'] += 1
for tag,prefix in [('DP2','dp2')]:
    path = SRC/'local'/f'ec3_{tag}_all.csv'
    d = pd.concat([x[x.validator.isin(VALIDATORS)] for x in pd.read_csv(path,chunksize=2048,float_precision='round_trip')],ignore_index=True)
    assert len(d)==27720 and not d.duplicated(['validator','validation_fold','row_id']).any()
    assert set(d.validator)==set(VALIDATORS)
    cells = []
    for v in VALIDATORS:
        g = d[d.validator==v]
        for seed in SEEDS:
            ea = g[f'r3s_{seed}'].to_numpy()-g.sub_ec.to_numpy()
            eb = g[f'{prefix}_{seed}'].to_numpy()-g.sub_ec.to_numpy()
            assert np.isfinite(ea).all() and np.isfinite(eb).all()
            a,b = float(np.sqrt(np.mean(ea**2))),float(np.sqrt(np.mean(eb**2)))
            close(a,math.sqrt(math.fsum(float(x)**2 for x in ea)/len(ea)))
            close(b,math.sqrt(math.fsum(float(x)**2 for x in eb)/len(eb)))
            cells.append(dict(validator=v,seed=seed,n=len(g),baseline=a,candidate=b,change_pct=100*(b/a-1)))
    D = d[d.validator=='DIAG10'].copy()
    assert len(D)==8640 and len(D[['farm','day']].drop_duplicates())==360
    rng=np.random.default_rng(20261004)
    ps=[]
    for seed in SEEDS:
        delta=(D[f'{prefix}_{seed}']-D.sub_ec)**2-(D[f'r3s_{seed}']-D.sub_ec)**2
        clusters=D.farm+'_'+(D.day//5).astype(str)
        stats=delta.groupby(clusters).agg(['sum','count'])
        manual=defaultdict(list)
        for key,val in zip(clusters,delta):manual[key].append(float(val))
        keys=sorted(manual)
        assert keys==list(stats.index)
        sums=np.array([math.fsum(manual[k]) for k in keys])
        counts=np.array([len(manual[k]) for k in keys])
        for k,s,c in zip(keys,sums,counts):
            close(s,stats.loc[k,'sum']);close(c,stats.loc[k,'count'])
        ix=rng.integers(0,len(stats),(20000,len(stats)))
        boot=stats['sum'].to_numpy()[ix].sum(1)/stats['count'].to_numpy()[ix].sum(1)
        independently=sums[ix].sum(1)/counts[ix].sum(1)
        close(float((boot>=0).mean()),float((independently>=0).mean()))
        ps.append(float((boot>=0).mean()))
    logpath=SRC/'ec3_DP2_operation_core3_v1.log'
    text=logpath.read_text(encoding='utf-8')
    found=re.search(r'DIAG10 P\(worse\) by seed:\s*(\[[^\]]+\])',text)
    assert found
    oldps=json.loads(found.group(1))
    for p,old in zip(ps,oldps):close(round(p,4),old)
    high=D.groupby(['farm','day']).sub_ec.transform('mean')>=1
    segments=[]
    for name,mask in [('high',high),('ordinary',~high),('late',D.day>=179)]:
        g=D[mask]
        for seed in SEEDS:
            a=(g[f'r3s_{seed}']-g.sub_ec).to_numpy()
            b=(g[f'{prefix}_{seed}']-g.sub_ec).to_numpy()
            ra=math.sqrt(math.fsum(float(z)**2 for z in a)/len(a))
            rb=math.sqrt(math.fsum(float(z)**2 for z in b)/len(b))
            close(ra,float(np.sqrt(np.mean(a*a))));close(rb,float(np.sqrt(np.mean(b*b))))
            segments.append(dict(segment=name,seed=seed,n=len(g),days=len(g[['farm','day']].drop_duplicates()),
                                 baseline=ra,candidate=rb,change_pct=100*(rb/ra-1),
                                 baseline_bias=math.fsum(float(z) for z in a)/len(a),candidate_bias=math.fsum(float(z) for z in b)/len(b)))
    res['experiments'][tag]=dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(d),cells=cells,segments=segments,
                                p_worse=ps,decision='REJECT',all_primary_improve=all(x['change_pct']<0 for x in cells if x['validator'] in ['DIAG10','A','B']))
res['limitations']=['Baseline is R3S, not actual .8R3+.2PFN season-v2.',
                    'Existing same public data reused; no untouched holdout, independent refit, or causal claims.',
                    'DP2 removes six DP1 features jointly, so no causal attribution to ventilation/curtain features.',
                    'Causal prefix source inspected, missing actions filled with zero; no independent full feature replay here.',
                    'EL1 colleague log read as report only, not independently rescored.']
res['source_hashes']={name:hashlib.sha256((SRC/name).read_bytes()).hexdigest() for name in ['ec3_DP2_operation_core3_v1.py','ec3_DP1_daily_operation_pattern_v1.py','ec3_DB1_dong_split_models_v1.py']}
DEST.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status=res['status'],checks=res['checks'],experiments=res['experiments']),ensure_ascii=False,indent=2))
