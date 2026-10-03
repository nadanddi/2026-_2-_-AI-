from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'));import env
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
O=ROOT/'집/코덱스/local/ec_tree_median_20261003_v1';records=[];checks=0
for seed in [7,101,2024]:
 d=pd.concat([pd.read_csv(O/f'DIAG10_{k}_{seed}.csv',float_precision='round_trip') for k in range(10)],ignore_index=True);d['state']=d.groupby(['farm','day']).y.transform('mean')>=1;q=d[d.hour==0];assert len(q)==360;hi=q.state.to_numpy()
 for signal in ['baseline','raw_mean','new_et_raw','high_vote_fraction','tree_std','raw_q75']:
  x=q[signal].to_numpy();manual=float(np.mean((x[hi][:,None]>x[~hi][None,:])+0.5*(x[hi][:,None]==x[~hi][None,:])));auc=roc_auc_score(hi,x);assert abs(manual-auc)<1e-12;checks+=1
  records.append(dict(seed=seed,signal=signal,n=len(q),high_days=int(hi.sum()),auc=auc))
pd.DataFrame(records).to_csv(H/'high_vote_diagnostic_v1.csv',index=False)
(H/'high_vote_verification_v1.json').write_text(json.dumps(dict(status='PASS',checks=checks,warning='post-hoc rank diagnostic only; state target is full day mean; predictors are h0 causal; no cutoff, gate, coefficient selected'),indent=2),encoding='utf-8');print(pd.DataFrame(records).to_string(index=False))
