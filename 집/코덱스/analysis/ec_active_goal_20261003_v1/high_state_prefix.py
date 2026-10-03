"""Diagnostic only: distinguish whole-day high-state ranking from causal prefix."""
from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/클로드/research'))
import env
import pandas as pd,numpy as np
from sklearn.metrics import roc_auc_score
p=ROOT/'집/코덱스/local/ec_dc4_integration_20261002_v1/v2_integration_oof.csv'
o=pd.read_csv(p,float_precision='round_trip');o=o[(o.validator=='DIAG10')&(o.seed==7)]
yd=o.groupby(['farm','day']).sub_ec.mean();rows=[];checks=0
for h in [0,3,6,12,18,23]:
    q=o[o.hour<=h].groupby(['farm','day']).season_v2.mean()
    for f in ['all','F13','F47']:
        use=[k for k in q.index if f=='all' or k[0]==f];pred=q.reindex(use).to_numpy();target=yd.reindex(use).to_numpy();y=target>=1
        auc=roc_auc_score(y,pred);a=pred[y];b=pred[~y]
        manual=math.fsum(float(x>z)+.5*float(x==z) for x in a for z in b)/(len(a)*len(b))
        assert abs(auc-manual)<1e-12;checks+=1
        rows.append(dict(farm=f,hour=h,n=len(y),high_days=int(y.sum()),auc=auc,high_prediction_mean=float(pred[y].mean()),high_actual_mean=float(target[y].mean()),ordinary_prediction_mean=float(pred[~y].mean()),ordinary_actual_mean=float(target[~y].mean())))
pd.DataFrame(rows).to_csv(H/'high_state_prefix_diagnostic_v1.csv',index=False)
(H/'high_state_prefix_verification_v1.json').write_text(json.dumps(dict(status='PASS',pairwise_checks=checks,warning='diagnostic only; no threshold selection; one seed; target is day mean whereas prefix predictions are causal partial mean'),ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.DataFrame(rows).query("farm=='all'").to_string(index=False))
