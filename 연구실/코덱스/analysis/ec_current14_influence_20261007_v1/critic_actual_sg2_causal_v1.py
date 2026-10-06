"""Actual baseline ensemble SG2 future/other-farm invariance; fit zero."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));import runner_v4 as R
import numpy as np,pandas as pd
raw,jobs,_=R.prepare();meta=R.M.identify(raw);times=meta.day*24+meta.hour
rows=pd.read_csv(R.L/'baseline_rows.csv',float_precision='round_trip',dtype={'seed':str});checks=[]
for farm in ['F13','F47']:
    k=next(k for k,(t,q,p) in jobs.items() if ((q.farm==farm)&(q.day>=179)).any())
    t,q,_=jobs[k];ref=set(t[['farm','day']].itertuples(index=False,name=None));ec=t.groupby(['farm','day']).sub_ec.mean()
    S=R.SG.prepare(raw,ref);cal=R.SG.ref_calendar(S,ref)
    r=rows[(rows.k==k)&rows.seed.eq('ensemble')].set_index('row_id').loc[q.row_id]
    p=r.pre_sg2.to_numpy();original=R.SG.correct(q[['row_id']],p,S,ec,ref,cal)
    np.testing.assert_allclose(original,r.sg2_raw.to_numpy(),rtol=0,atol=2e-12)
    qt=q.day*24+q.hour;refids=set(t.row_id)
    for frac in [.2,.5,.8]:
        cut=int(qt[q.farm.eq(farm)&q.day.ge(179)].quantile(frac))
        allowed_q=q.farm.eq(farm)&qt.le(cut);allowed_raw=meta.farm.eq(farm)&times.le(cut)
        changed=raw.copy();mask=~changed.row_id.isin(refids)&~allowed_raw;cols=R.M.RAW+R.SG.W
        changed.loc[mask,cols]=changed.loc[mask,cols]*13+97
        altS=R.SG.prepare(changed,ref);pp=p.copy();pp[~allowed_q]=pp[~allowed_q]+1
        altered=R.SG.correct(q[['row_id']],pp,altS,ec,ref,R.SG.ref_calendar(altS,ref))
        np.testing.assert_array_equal(original[allowed_q],altered[allowed_q])
        active=int(np.sum(abs(original[allowed_q]-p[allowed_q])>1e-12));assert active>0
        checks.append(dict(fold=k,farm=farm,cut=cut,allowed_rows=int(allowed_q.sum()),active_correction_rows=active,maxdiff=float(np.max(abs(original[allowed_q]-altered[allowed_q])))))
        print('PASS_ACTUAL_NONVACUOUS_SG2',checks[-1],flush=True)
out=dict(status='PASS_ACTUAL_ENSEMBLE_SG2_SIX_NONVACUOUS_PREFIX_PERTURBATIONS',checks=checks,fit=0,scope='SG2 only, ensemble, one public-P2-containing fold per farm, three cutoff fractions. Not an exhaustive test of all models/folds/hours.')
with (H/'critic_actual_sg2_causal_v1.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
