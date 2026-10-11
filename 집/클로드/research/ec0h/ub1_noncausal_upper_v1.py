# -*- coding: utf-8 -*-
"""UB1 (2026-10-12 집 클로드; user "0시 맞추기 포기하지 말고 도전"): NON-CAUSAL upper bound (diagnostic only, never deployable).
Question: is the EC 0 h residual r0 = y(0h) - q(0h) (q = current model DIAG10 OOF, CUR+SG2, seed mean) predictable from the record's
WHOLE-DAY inputs (all 24 h, incl. future hours)?  If not even this works, no causal 0 h estimator from inputs can.
Features: per variable (14 usable inputs) daily mean/std/min/max, 0-5 h mean, 6-11, 12-17, 18-23 h means, hour-0 value, night
rise (5h - 0h), plus q0 and season.  Model: ExtraTrees(500, min_samples_leaf 5) and Ridge; CV grouped by DIAG10 fold.
Report R^2 of r0 (all, pass-2, sealed days) and the RMSE of q0 + predicted r0 vs q0.  Also the same for the day-mean residual.
"""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import os, numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
O=pd.read_csv(os.path.join(env.ROOT,"연구실","클로드","results","ec_rv1_preds_v1.csv")); O=O[(O.tag=="CUR")&(O.validator=="DIAG10")]; O["rec"]=O.row_id.str[:7]
A=O.groupby(["rec","hour"]).agg(q=("sg","mean"),y=("sub_ec","first"),fold=("fold","first")).reset_index()
R=A[A.hour==0].set_index("rec")[["q","y","fold"]]; R["r0"]=R.y-R.q
R["rd"]=A.assign(e=A.y-A.q).groupby("rec").e.mean()
X=pd.read_csv(os.path.join(env.DATA,"train_X.csv")); X["rec"]=X.row_id.str[:7]; X["h"]=X.row_id.str[8:].astype(int); X=X[X.rec.isin(R.index)]
V=["out_temp","out_hum","out_rad","out_wspd","in_temp","in_hum","in_co2","act_vent","act_shade","act_thermal","act_heating","act_circfan","act_co2","act_fog"]
g=X.groupby("rec"); F=pd.concat([g[V].mean().add_suffix("_m"),g[V].std().add_suffix("_s"),g[V].min().add_suffix("_mn"),g[V].max().add_suffix("_mx")],axis=1)
for a,b in ((0,5),(6,11),(12,17),(18,23)): F=F.join(X[(X.h>=a)&(X.h<=b)].groupby("rec")[V].mean().add_suffix(f"_{a}_{b}"))
h0=X[X.h==0].set_index("rec")[V]; h5=X[X.h==5].set_index("rec")[V]; F=F.join(h0.add_suffix("_h0")).join((h5-h0).add_suffix("_rise"))
S=pd.read_csv(r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\0038d811-b5c9-4a0d-913f-084fb31e14ab\scratchpad\build22\팜모니_정형데이터_재현패키지_22\EC\training_day_to_season.csv")
S["rec"]=S.farm+"_"+S.day.map("{:03d}".format); F=F.join(S.set_index("rec").season); F["q0"]=R.q; F["f47"]=(F.index.str[:3]=="F47").astype(int)
D=R.join(F).dropna(subset=["r0"]); feat=[c for c in F.columns]; Xf=D[feat].fillna(D[feat].median())
sealed=(D.act_vent_m<=5)&(D.act_circfan_m<=5); p2=D.index.str[4:7].astype(int)>=179
rm=lambda e: float(np.sqrt(np.mean(np.square(e))))
for tgt in ("r0","rd"):
    for nm,mk in (("ET",lambda: ExtraTreesRegressor(500,min_samples_leaf=5,max_features=.5,random_state=1,n_jobs=2)),("Ridge",lambda: RidgeCV(alphas=np.logspace(-1,4,20)))):
        pr=np.zeros(len(D))
        for k in D.fold.unique():
            te=(D.fold==k).values; Z=Xf.copy()
            if nm=="Ridge": mu,sd=Z[~te].mean(),Z[~te].std()+1e-9; Z=(Z-mu)/sd
            pr[te]=mk().fit(Z[~te],D[tgt][~te]).predict(Z[te])
        y=D[tgt].values; r2=lambda m: 1-np.mean((y[m]-pr[m])**2)/np.mean((y[m]-y[m].mean())**2)
        allm=np.ones(len(D),bool)
        print(f"{tgt} {nm:5s}: CV R2 all {r2(allm):+.3f} | pass-2 {r2(p2):+.3f} | sealed {r2(sealed.values):+.3f} | residual RMSE {rm(y):.3f} -> {rm(y-pr):.3f} (pass-2 {rm(y[p2]):.3f} -> {rm((y-pr)[p2]):.3f})")
