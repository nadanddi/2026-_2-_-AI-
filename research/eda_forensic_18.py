# -*- coding: utf-8 -*-
"""Forensic 18: (a) cold-end OOD of test in_temp vs F13/F47 train support; heating strategy shift.
(b) midnight discontinuity of INPUTS in train vs test."""
import env
import numpy as np, pandas as pd, common
tX, ty, sX = common.load_raw()
tX["set"]="train"; sX["set"]="test"
X=pd.concat([tX[tX.farm.isin(["F13","F47"])],sX]).merge(ty[["row_id","sub_temp"]],on="row_id",how="left").sort_values(["farm","t"])
tr=X[X.set=="train"]; te=X[X.set=="test"]
for q in [0.01,0.02,0.05]:
    lim=tr.in_temp.quantile(q)
    print(f"train in_temp q{q}={lim:.1f}: test share below {np.mean(te.in_temp<lim):.3f}")
# night in_temp vs out_temp: heating setpoint
night=lambda d:d[~d.hour.between(7,18)]
for nm,d in [("train",tr),("test",te)]:
    n=night(d); c=n[n.out_temp<3]
    print(nm,"night rows out<3: n=%d in_temp mean %.2f p10 %.2f; heat>=100 share %.2f; in-out %.2f"%(len(c),c.in_temp.mean(),c.in_temp.quantile(.1),(c.act_heating>=100).mean(),(c.in_temp-c.out_temp).mean()))
# sub_temp in train at low in_temp
lo=tr[tr.in_temp<8]
print("train rows in_temp<8:",len(lo),"days",lo.groupby(["farm","day"]).size().to_dict())
print("sub_temp min train F13/F47:",tr.groupby("farm").sub_temp.min().to_dict(), " 1%:",tr.groupby("farm").sub_temp.quantile(.01).round(2).to_dict())
# midnight jumps in inputs
X["ok"]=X.farm.eq(X.farm.shift())&X.t.eq(X.t.shift()+1)
for c in ["in_temp","in_hum","in_co2"]:
    X["d_"+c]=(X[c]-X[c].shift()).abs().where(X.ok)
for s in ["train","test"]:
    g=X[X.set==s]
    r={c: round(g[g.hour==0]["d_"+c].mean()/g[g.hour.isin([22,23,1,2])]["d_"+c].mean(),2) for c in ["in_temp","in_hum","in_co2"]}
    print(s,"input |d| at 00h / mean |d| at 22,23,01,02h:",r, "n00=",g[g.hour==0].d_in_temp.notna().sum())
