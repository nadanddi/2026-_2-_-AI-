# -*- coding: utf-8 -*-
"""EV3: can INPUTS alone (causal: hours h-3..h of the same record) detect EC25 events (DROP morning, RISE afternoon) vs
non-event hours of the same hour of day?  ExtraTrees classifier, CV grouped by record (5 folds); AUC overall and per pass.
Also: does adding the substrate-temperature LABEL change (oracle) raise AUC? (diagnostic of where the signal lives)."""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import os, numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
X=pd.read_csv(os.path.join(env.DATA,"train_X.csv")); Y=pd.read_csv(os.path.join(env.DATA,"train_y.csv"))
D=X.merge(Y,on="row_id"); D=D[D.row_id.str[:3].isin(["F13","F47"])].copy()
D["rec"]=D.row_id.str[:7]; D["p2"]=D.rec.str[4:7].astype(int)>=179; D["h"]=D.row_id.str[8:].astype(int); D=D.sort_values(["rec","h"]).reset_index(drop=True)
D["ec25"]=D.sub_ec/(1+.02*(D.sub_temp-25)); D["dEC"]=D.groupby("rec").ec25.diff()
V=["in_temp","in_hum","in_co2","out_temp","out_hum","out_rad","act_vent","act_shade","act_thermal","act_heating","act_circfan","act_co2","act_fog"]
F=[]
for v in V:
    for L in range(0,4):
        c=f"{v}_l{L}"; D[c]=D.groupby("rec")[v].shift(L); F.append(c)
    D[f"{v}_d1"]=D[v]-D.groupby("rec")[v].shift(1); F.append(f"{v}_d1")
D["st_d"]=D.groupby("rec").sub_temp.diff(); D["st_d2"]=D.groupby("rec").sub_temp.diff(2)
recs=D.rec.unique(); rng=np.random.default_rng(1); fold=dict(zip(recs,rng.permutation(len(recs))%5)); D["fold"]=D.rec.map(fold)
for kind,hours,sel in (("DROP morning 8-13h",range(8,14),D.dEC<=-.05),("RISE afternoon 13-18h",range(13,19),D.dEC>=.05)):
    Z=D[D.h.isin(hours)&D.dEC.notna()].copy(); Z["y"]=sel[Z.index].astype(int)
    for nm,feat in (("inputs only",F+["h"]),("+ substrate temp change (label, oracle)",F+["h","st_d","st_d2"])):
        pr=np.zeros(len(Z))
        for k in range(5):
            te=(Z.fold==k).values; m=ExtraTreesClassifier(400,min_samples_leaf=10,max_features=.4,random_state=1,n_jobs=2).fit(Z.loc[~te,feat].fillna(-99),Z.y[~te]); pr[te]=m.predict_proba(Z.loc[te,feat].fillna(-99))[:,1]
        a=roc_auc_score(Z.y,pr); a2=roc_auc_score(Z.y[Z.p2],pr[Z.p2.values]) if Z.y[Z.p2].nunique()>1 else np.nan
        print(f"{kind:22s} {nm:42s}: events {int(Z.y.sum())}/{len(Z)} | AUC all {a:.3f} | pass-2 {a2:.3f}")

# --- within-record check (added after first run): does the detector find the event HOUR or just the event-prone DAY? ---
print("\nwithin-record check (only records that contain >=1 event; AUC of event hour vs other hours of the same record window):")
for kind,hours,sel in (("DROP morning 8-13h",range(8,14),D.dEC<=-.05),("RISE afternoon 13-18h",range(13,19),D.dEC>=.05)):
    Z=D[D.h.isin(hours)&D.dEC.notna()].copy(); Z["y"]=sel[Z.index].astype(int)
    pr=np.zeros(len(Z))
    for k in range(5):
        te=(Z.fold==k).values; m=ExtraTreesClassifier(400,min_samples_leaf=10,max_features=.4,random_state=1,n_jobs=2).fit(Z.loc[~te,F+["h"]].fillna(-99),Z.y[~te]); pr[te]=m.predict_proba(Z.loc[te,F+["h"]].fillna(-99))[:,1]
    Z["pr"]=pr; has=Z.groupby("rec").y.transform("max")==1
    W=Z[has]; print(f"  {kind}: day-level AUC (event-day vs not, max score per record) {roc_auc_score(Z.groupby('rec').y.max(),Z.groupby('rec').pr.max()):.3f} | within event-days hour AUC {roc_auc_score(W.y,W.pr):.3f} (rows {len(W)}, events {int(W.y.sum())})")
    # per-record rank AUC average
    aucs=[roc_auc_score(G.y,G.pr) for _,G in W.groupby("rec") if G.y.nunique()>1]; print(f"     mean per-record hour AUC {np.mean(aucs):.3f} over {len(aucs)} records")
