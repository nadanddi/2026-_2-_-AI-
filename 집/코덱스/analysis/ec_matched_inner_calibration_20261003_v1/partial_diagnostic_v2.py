from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/statistical_experiments_20261003_v1'));import support as S
import numpy as np,pandas as pd
OUT=ROOT/'집/코덱스/local/ec_matched_inner_calibration_20261003_v1'
lab,core,wv,folds,outer=S.loadec();idx=lab.set_index('row_id');records=[];checks=0
# Frozen first three finished folds. Diagnostic only, never an interim adoption gate.
for k in [0,1,2]:
 cp=dict(np.load(OUT/f'DIAG10_{k}_cpu.npz'));b=idx.reindex(cp['row_id']).reset_index()
 pp=[dict(np.load(OUT/f'DIAG10_{k}_pfn_{c}.npz')) for c in [1,2,3,4]]
 assert all(np.array_equal(z['row_id'],b.row_id) for z in pp)
 bag=np.mean([z['prediction'] for z in pp],axis=0)
 for seed in [7,101,2024]:
  o=pd.read_csv(OUT/f'DIAG10_{k}_{seed}_pred.csv',float_precision='round_trip')
  pred=np.clip(core.shrink(.8*cp[f'r3_{seed}']+.2*bag,b),cp['lo'],cp['hi'])
  inner=b[['farm','day','hour','sub_ec']].copy();inner['prediction']=pred
  query=o.rename(columns={'y':'sub_ec','baseline':'prediction'})
  for scope,g in [('inner',inner),('outer',query)]:
   d=g[g.hour==0];d=d[d.prediction>=1]
   vals=(d.prediction-d.sub_ec).to_numpy()
   bias=float(np.mean(vals)) if len(vals) else None
   if len(vals):assert abs(bias-math.fsum(float(x) for x in vals)/len(vals))<1e-12;checks+=1
   records.append(dict(fold=k,seed=seed,scope=scope,high_pred_days=len(d),bias=bias,low_actual_days=int((d.sub_ec<1).sum()),inner_validation_days=len(set(zip(b.farm,b.day))) if scope=='inner' else None))
  for c in ['baseline','candidate']:
   err=(o[c]-o.y).to_numpy();a=float(np.sqrt(np.mean(err**2)));z=math.sqrt(math.fsum(float(e)**2 for e in err)/len(err));assert abs(a-z)<1e-12;checks+=1
   records.append(dict(fold=k,seed=seed,scope='outer_'+c,n=len(o),rmse=a))
  expected=np.clip(o.baseline+o.correction,o.clip_lo,o.clip_hi);assert np.max(np.abs(expected-o.candidate))<1e-12;checks+=len(o)
R=pd.DataFrame(records);R.to_csv(H/'partial_diagnostic_v2.csv',index=False)
(H/'partial_diagnostic_verification_v2.json').write_text(json.dumps(dict(status='PASS',checks=checks,folds=[0,1,2],scope='partial diagnostic only; no full performance or adoption claim; no stopping rule; high_pred is current hour0 prediction>=1',warning='high_actual flag uses hour0 observed EC, not whole-day average; no adoption inference'),indent=2),encoding='utf-8')
print(R.to_string(index=False));print('CHECKS',checks)
