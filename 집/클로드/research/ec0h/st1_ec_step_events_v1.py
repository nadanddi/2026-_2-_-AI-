# -*- coding: utf-8 -*-
"""ST1 (2026-10-12 집 클로드; user: "그 계단 변화가 다른 날도 비슷한지 찾아봐").  Step event = hourly |dEC| >= .30 at hour h (h-1 -> h)
AND the next 3 hours stay within +-.05 of EC(h) (new level persists).  Training labels F13/F47.  Reports count, hour, direction,
size, level before, pass, and inputs at the step vs hour-matched non-step baseline (z of input change at h and level at h)."""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import os, numpy as np, pandas as pd
X=pd.read_csv(os.path.join(env.DATA,"train_X.csv")); Y=pd.read_csv(os.path.join(env.DATA,"train_y.csv"))
D=X.merge(Y,on="row_id"); D=D[D.row_id.str[:3].isin(["F13","F47"])].copy(); D["rec"]=D.row_id.str[:7]; D["h"]=D.row_id.str[8:].astype(int)
D=D.sort_values(["rec","h"]).reset_index(drop=True); D["p2"]=D.rec.str[4:7].astype(int)>=179
g=D.groupby("rec").sub_ec; D["d"]=g.diff()
fut=pd.concat([g.shift(-k) for k in (1,2,3)],axis=1); D["stay"]=(fut.sub(D.sub_ec,axis=0).abs().max(axis=1)<=.05)&fut.notna().all(axis=1)
D["step"]=(D.d.abs()>=.30)&D.stay
S=D[D.step]
print("step events:",len(S),"in",S.rec.nunique(),"records (of",D.rec.nunique(),") | pass-2:",int(S.p2.sum()))
print("direction: down",int((S.d<0).sum()),"up",int((S.d>0).sum())," | size median %.2f max %.2f"%(S.d.abs().median(),S.d.abs().max()))
print("hour distribution:",S.h.value_counts().sort_index().to_dict())
print("EC before step (median) down %.2f / up %.2f"%(D.sub_ec.shift(1)[S.index][S.d<0].median(), D.sub_ec.shift(1)[S.index][S.d>0].median()))
print("\nsteps:"); print(S[["rec","h","d","sub_ec"]].assign(before=D.sub_ec.shift(1)[S.index]).round(2).to_string(index=False))
V=["in_temp","in_hum","in_co2","out_rad","act_vent","act_shade","act_thermal","act_circfan","act_fog","act_co2","act_heating","sub_temp"]
for v in V: D["dv_"+v]=D.groupby("rec")[v].diff()
print("\ninput change at the step hour vs same-hour non-step baseline (z):")
for kind,sel in (("DOWN",S.d<0),("UP",S.d>0)):
    idx=S.index[sel.values]; out=[]
    for v in V:
        z=[]
        for i in idx:
            base=D[(D.h==D.h[i])&(~D.step)]["dv_"+v].dropna()
            if np.isfinite(D["dv_"+v][i]) and base.std()>0: z.append((D["dv_"+v][i]-base.mean())/base.std())
        out.append((v,np.mean(z) if z else np.nan,len(z)))
    print(f"  {kind} (n {len(idx)}):", ", ".join(f"{v} {m:+.2f}" for v,m,n in out))
