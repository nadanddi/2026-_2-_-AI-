# -*- coding: utf-8 -*-
"""EV2 = EV1 on TEMPERATURE-COMPENSATED EC (EC25 = EC / (1 + .02 (sub_temp - 25)), general sensor convention; removes the sensor temperature effect so remaining jumps reflect solution actions). (2026-10-12 집 클로드; user: "양액 작용이 있을 때 다른 데이터들이 어떻게 움직이는지 연구"): event study on TRAINING labels.
Events: within-record hourly EC change dEC_h = y_h - y_(h-1), hours 1..23.  DROP events: dEC <= -T; RISE: dEC >= +T, T in {.05,.10}.
For each input v and lag L in -3..+3 (input change dv at hour h+L), compare mean dv at events vs non-event hours of the SAME hour of
day (hour-matched baseline), standardised by baseline sd -> effect size.  Consistency: same sign in 4 cells (F13/F47 x pass1/2),
min |effect| reported.  Descriptive (no model).
"""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import os, numpy as np, pandas as pd
X=pd.read_csv(os.path.join(env.DATA,"train_X.csv")); Y=pd.read_csv(os.path.join(env.DATA,"train_y.csv"))
D=X.merge(Y,on="row_id"); D=D[D.row_id.str[:3].isin(["F13","F47"])].copy()
D["rec"]=D.row_id.str[:7]; D["farm"]=D.rec.str[:3]; D["p2"]=D.rec.str[4:7].astype(int)>=179; D["h"]=D.row_id.str[8:].astype(int)
D=D.sort_values(["rec","h"]).reset_index(drop=True)
V=["in_temp","in_hum","in_co2","out_temp","out_hum","out_rad","act_vent","act_shade","act_thermal","act_heating","act_circfan","act_co2","act_fog","sub_temp"]
D["ec25"]=D.sub_ec/(1+.02*(D.sub_temp-25)); D["dEC"]=D.groupby("rec").ec25.diff(); print("corr(dEC raw, d sub_temp) %.3f -> corr(dEC25, d sub_temp) %.3f"%(D.groupby("rec").sub_ec.diff().corr(D.groupby("rec").sub_temp.diff()), D.dEC.corr(D.groupby("rec").sub_temp.diff())))
for v in V: D["d_"+v]=D.groupby("rec")[v].diff()
for T in (.05,.10):
    for kind,sel in (("DROP",D.dEC<=-T),("RISE",D.dEC>=T)):
        ev=sel&D.dEC.notna(); print(f"\n== {kind} |dEC|>={T}: events {int(ev.sum())} in {D.loc[ev,'rec'].nunique()} records; by hour mode {D.loc[ev,'h'].value_counts().head(4).to_dict()} | cells n: "+str(D[ev].groupby(['farm','p2']).size().to_dict()))
        rows=[]
        for v in V:
            for L in range(-3,4):
                s=D.groupby("rec")["d_"+v].shift(-L)   # change at hour h+L
                effs=[]
                for (f,p2),G in D.groupby(["farm","p2"]):
                    e=ev[G.index]; sG=s[G.index]
                    if e.sum()<5: effs.append(np.nan); continue
                    base=pd.DataFrame({"s":sG[~e],"h":G.h[~e]}).dropna().groupby("h").s.agg(["mean","std"])
                    z=[(sv-base.at[hh,"mean"])/(base.at[hh,"std"]+1e-9) for sv,hh in zip(sG[e],G.h[e]) if hh in base.index and np.isfinite(sv)]
                    effs.append(np.mean(z) if z else np.nan)
                effs=np.array(effs); ok=np.all(np.isfinite(effs)) and (np.all(effs>0) or np.all(effs<0))
                rows.append(dict(v=v,L=L,mean=np.nanmean(effs),minabs=np.nanmin(np.abs(effs)) if ok else 0.0,consistent=ok))
        R=pd.DataFrame(rows); R=R[R.consistent].sort_values("minabs",ascending=False)
        print(R.head(10).round(3).to_string(index=False))
