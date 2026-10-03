from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
OUT=ROOT/'집/코덱스/local/ec_pca_rotation_20261003_v1';results=[];checks=0
for seed in [7,101,2024]:
 paths=[OUT/f'DIAG10_{k}_{seed}.csv' for k in range(10)]
 d=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in paths],ignore_index=True);assert len(d)==8640
 truth=d.groupby(['farm','day']).y.mean()>=1
 d['high']=pd.MultiIndex.from_frame(d[['farm','day']]).map(truth)
 q=d[d.hour==0].copy();q['truth']=pd.MultiIndex.from_frame(q[['farm','day']]).map(truth);assert len(q)==360 and q.truth.sum()==31
 for c in ['baseline','candidate','old_et_shrunk','new_et_raw']:
  high=q.loc[q.truth,c].to_numpy();low=q.loc[~q.truth,c].to_numpy();manual=float(np.mean((high[:,None]>low[None,:]).astype(float)+.5*(high[:,None]==low[None,:])))
  sklearn=float(roc_auc_score(q.truth,q[c]));assert abs(manual-sklearn)<1e-12;checks+=1
  results.append(dict(seed=seed,scope='hour0_rank',prediction=c,n=360,high_days=31,auc=manual))
 contributions=[]
 for label,mask in [('high',d.high),('ordinary',~d.high)]:
  g=d[mask];delta=((g.candidate-g.y)**2-(g.baseline-g.y)**2).to_numpy();sse=math.fsum(float(x) for x in delta);assert abs(sse-float(delta.sum()))<1e-9;checks+=1
  bias=math.fsum(float(x) for x in g.candidate-g.y)/len(g);assert abs(bias-float((g.candidate-g.y).mean()))<1e-12;checks+=1
  contributions.append(sse);results.append(dict(seed=seed,scope='all_hours_error',segment=label,n=len(g),high_days=31,sse_change=sse,candidate_bias=bias))
 all_delta=math.fsum(float(x) for x in (d.candidate-d.y)**2-(d.baseline-d.y)**2);assert abs(sum(contributions)-all_delta)<1e-9;checks+=1
R=pd.DataFrame(results);R.to_csv(H/'rank_scale_diagnostic_v1.csv',index=False)
(H/'rank_scale_verification_v1.json').write_text(json.dumps(dict(status='PASS',checks=checks,scope='posthoc public DIAG10 diagnostic; no gate/threshold selected; AUC does not measure absolute EC or conditional incremental information'),indent=2),encoding='utf-8');print(R.to_string(index=False));print('CHECKS',checks)
