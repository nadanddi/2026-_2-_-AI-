from pathlib import Path
import sys,math,json
sys.dont_write_bytecode=True
import run as R
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
H=Path(__file__).resolve().parent
o=pd.read_csv(R.OUT/'oof.csv',float_precision='round_trip')
res=json.loads((H/'result_v1.json').read_text(encoding='utf-8'))
checks=0
for seed in [7,101,2024]:
    d=o[(o.validator=='DIAG10')&(o.seed==seed)]
    byday={}
    for r in d.itertuples():byday.setdefault((r.farm,r.day),[]).append((r.candidate-r.y)**2-(r.baseline-r.y)**2)
    rng=np.random.default_rng(20261003+seed);sample=np.zeros(20000)
    for f in ['F13','F47']:
        day=sorted(k[1] for k in byday if k[0]==f)
        values=[math.fsum(byday[(f,x)]) for x in day]
        block=np.array([math.fsum(values[i:i+5]) for i in range(0,len(values),5)])
        idx=rng.integers(len(block),size=(20000,len(block)))
        sample+=np.sum(block[idx],axis=1)
    assert abs(np.mean(sample>=0)-res['bootstrap'][str(seed)][0])<1e-12
    assert np.max(np.abs(np.quantile(sample,[.025,.975])-res['bootstrap'][str(seed)][1]))<1e-10
    checks+=2
lab,core,_,folds,_=R.S.loadec()
name,k,tm,vm=next(f for f in folds if f[0]=='DIAG10' and f[1]==0)
raw=core.identify(pd.read_csv(Path(R.env.DATA)/'train_X.csv',usecols=['row_id']+R.W+core.RAW))
raw=raw[raw.farm.isin(['F13','F47'])]
tr=lab[tm];va=lab[vm];tk=set(zip(tr.farm,tr.day));vk=set(zip(va.farm,va.day))
role,_=R.roles(raw,tk)
with threadpool_limits(limits=2):p,_=R.probs(raw,core.RAW,tk|vk,role,R.prefix(raw,core.RAW))
z=dict(np.load(R.S.OUT/'E_DIAG10_0_cpu.npz'))
b=lab.set_index('row_id').reindex(z['row_id']).reset_index()
bag=np.mean([np.load(R.S.OUT/f'E_DIAG10_0_pfn_{i}.npz')['prediction'] for i in range(1,5)],axis=0)
maxdiff=0
for seed in [7,101,2024]:
    nested=np.clip(core.shrink(.8*z[f'r3_{seed}']+.2*bag,b),z['lo'],z['hi'])
    daily={}
    for row,pred in zip(b.itertuples(),nested):daily.setdefault((row.farm,row.day),[]).append(row.sub_ec-pred)
    mean={key:math.fsum(val)/len(val) for key,val in daily.items()}
    g=o[(o.validator==name)&(o.fold==k)&(o.seed==seed)]
    for r in g.itertuples():
        history=[(q,e) for (f,q),e in mean.items() if f==r.farm and 0<r.day-q<=12 and (q>=179)==(r.day>=179)]
        if history:
            weights=[math.exp(-(r.day-q)/6)*(r.pB*p[(r.farm,q,23)]+(1-r.pB)*(1-p[(r.farm,q,23)])) for q,e in history]
            weighted=math.fsum(w*e for w,(q,e) in zip(weights,history))/math.fsum(weights)
            corr=.1*math.exp(-(r.day-max(q for q,e in history))/6)*weighted
            expected=max(r.baseline+max(-.5,min(.5,corr)),0)
        else:expected=r.baseline
        maxdiff=max(maxdiff,abs(expected-r.candidate));assert abs(expected-r.candidate)<1e-12;checks+=1
(H/'extra_verification_v1.json').write_text(json.dumps(dict(status='PASS',checks=checks,scalar_correction_maxdiff=maxdiff,bootstrap_seeds=3),indent=2),encoding='utf-8')
print('PASS',checks,maxdiff)
