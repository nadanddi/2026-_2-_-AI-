"""Input-derived role diagnostics on known pairs; no predictive correction."""
from pathlib import Path
import sys,json,math
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
sys.path.insert(0,str(ROOT/'집/코덱스/analysis/source_history_20261003_v1'))
import run as A
import pandas as pd,numpy as np
from sklearn.metrics import roc_auc_score,brier_score_loss
o=pd.read_csv(A.OUT/'oof.csv',float_precision='round_trip')
o=o[(o.validator=='DIAG10')&(o.seed==7)].copy()
raw=pd.read_csv(Path(A.env.DATA)/'train_X.csv',usecols=['row_id']+A.W)
raw['farm']=raw.row_id.str[:3];raw['day']=raw.row_id.str[4:7].astype(int);raw['hour']=raw.row_id.str[8:10].astype(int)
keys=set(zip(o.farm,o.day));role,pair=A.roles(raw,keys)
o['role']=[role.get((f,d),np.nan) for f,d in zip(o.farm,o.day)]
o=o[o.role.notna()]
rows=[];checks=0
for farm in ['F13','F47']:
    for h in [0,6,12,23]:
        for period in ['all','early','late']:
            g=o[(o.farm==farm)&(o.hour==h)]
            if period!='all':g=g[(g.day>=179)==(period=='late')]
            if not len(g):continue
            y=g.role.to_numpy();p=g.pB.to_numpy();auc=roc_auc_score(y,p) if len(set(y))==2 else None
            if auc is not None:
                p1=p[y==1];p0=p[y==0];rank=math.fsum(float(a>b)+.5*float(a==b) for a in p1 for b in p0)/(len(p1)*len(p0));assert abs(rank-auc)<1e-12;checks+=1
            rows.append(dict(farm=farm,hour=h,period=period,n=len(g),auc=auc,accuracy=float(np.mean((p>=.5)==y)),brier=brier_score_loss(y,p)))
pd.DataFrame(rows).to_csv(H/'source_classifier_diagnostic_v1.csv',index=False)
(H/'source_classifier_verification_v1.json').write_text(json.dumps(dict(status='PASS',pairwise_auc_checks=checks,warning='role proxy only; query full-day weather used only for diagnostic ground truth, never model features; late sample is small'),ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.DataFrame(rows).query("period == 'all'").to_string(index=False))
