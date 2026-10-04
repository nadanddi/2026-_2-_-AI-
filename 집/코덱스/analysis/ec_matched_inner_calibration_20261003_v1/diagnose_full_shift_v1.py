"""Descriptive inner/outer calibration transfer; no new recipe or fitting."""
from pathlib import Path
import sys, importlib.util, json, math
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent; ROOT = H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research')); import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1')); import support as S
import numpy as np, pandas as pd
path = H.parent/'ec_nested_high_specialist_20261004_v1/run_v2.py'
sp = importlib.util.spec_from_file_location('shift_inner_id_guard',path); N=importlib.util.module_from_spec(sp); sp.loader.exec_module(N)
OUT = ROOT/'집/코덱스/local'/H.name
dest=H/'full_shift_diagnostic_v1.json'; assert not dest.exists()
lab,core,wv,folds,outer=S.loadec(); assert N.preflight(lab,folds,outer)['status']=='PASS'
fit=json.loads((H/'fit_audit_v1.json').read_text(encoding='utf-8'))
coef={(x['validator'],x['fold'],x['seed'],x['hour']):x['beta'] for x in fit['coefficients']}
o=pd.read_csv(OUT/'oof.csv',float_precision='round_trip')
idx=lab.set_index('row_id'); rows=[]; checks=0
for v,k,tm,vm in folds:
 a,b,z,bag=N.full_inner(v,k,lab[tm],idx)
 for seed in [7,101,2024]:
  baseline=np.clip(core.shrink(.8*z[f'r3_{seed}']+.2*bag,b),float(z['lo']),float(z['hi']))
  q=o[(o.validator==v)&(o.fold==k)&(o.seed==seed)].copy()
  for domain,frame,ycol,predcol in [('inner',b.assign(p=baseline),'sub_ec','p'),('outer',q,'y','baseline')]:
   daily=frame.groupby(['farm','day'])[ycol].mean()
   for hour in [0,6,12,23]:
    x=frame[frame.hour<=hour].groupby(['farm','day'])[predcol].mean()
    values=daily.reindex(x.index).to_numpy(); p=x.to_numpy()
    bet=coef[v,k,seed,hour]; raw=bet[0]+bet[1]*p+bet[2]*np.maximum(p-.6,0)+bet[3]*np.maximum(p-1,0)
    correction=.2*np.clip(raw,-.3,.3)
    for (farm,day),actual,pp,cc in zip(x.index,values,p,correction):
     g=frame[(frame.farm==farm)&(frame.day==day)&(frame.hour<=hour)]
     assert abs(pp-math.fsum(float(z) for z in g[predcol])/len(g))<1e-12
     checks+=1
     rows.append(dict(validator=v,fold=k,seed=seed,hour=hour,domain=domain,farm=farm,day=int(day),
                      x=float(pp),yday=float(actual),required_shift=float(actual-pp),
                      learned_correction=float(cc),high=bool(actual>=1)))
t=pd.DataFrame(rows)
t['prediction_bin']=pd.cut(t.x,[-np.inf,.6,.9,1.1,1.3,np.inf],right=False).astype(str)
summary=t.groupby(['validator','seed','hour','domain','prediction_bin'],observed=True).agg(
 n=('x','size'),high_n=('high','sum'),prediction_mean=('x','mean'),label_mean=('yday','mean'),
 required_shift=('required_shift','mean'),learned_correction=('learned_correction','mean')).reset_index()
summary.to_csv(H/'full_shift_summary_v1.csv',index=False)
t.to_csv(OUT/'full_shift_records_v1.csv',index=False)
selected=[]
for seed in [7,101,2024]:
 for hour in [0,23]:
  for domain in ['inner','outer']:
   g=t[(t.validator=='DIAG10')&(t.seed==seed)&(t.hour==hour)&(t.domain==domain)&(t.x>=.9)]
   assert len(g)>0
   selected.append(dict(seed=seed,hour=hour,domain=domain,occurrences=len(g),
                        unique_days=len(g[['farm','day']].drop_duplicates()),
                        prediction_mean=float(g.x.mean()),label_mean=float(g.yday.mean()),
                        required_shift=float(g.required_shift.mean()),learned_correction=float(g.learned_correction.mean()),
                        high_fraction=float(g.high.mean())))
res=dict(status='PASS',checks=checks,day_occurrences=len(t),selected_high_prediction=selected,
         scope='same completed candidate, descriptive transfer diagnostic; no new fit or threshold selection',
         limitations=['inner days repeat across outer folds; counts are occurrences, not independent days',
                      'yday uses full-day labels for diagnosis only; never inference features',
                      'association and transport mismatch do not identify a causal mechanism'])
dest.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(res,ensure_ascii=False,indent=2))
