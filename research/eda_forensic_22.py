# -*- coding: utf-8 -*-
"""Forensic 22: scan for piecewise-linearly interpolated OUTSIDE-weather days (F13/F47 train+test)
using share of hours whose 2nd difference is within rounding (out_temp, out_wspd, out_rad)."""
import env
import numpy as np, pandas as pd, common
tX, ty, sX = common.load_raw()
X=pd.concat([tX[tX.farm.isin(["F13","F47"])].assign(set="train"),sX.assign(set="test")]).sort_values(["farm","t"]).reset_index(drop=True)
ok=X.farm.eq(X.farm.shift())&X.t.eq(X.t.shift()+1)&X.farm.eq(X.farm.shift(-1))&X.t.eq(X.t.shift(-1)-1)
for c,tol in [("out_temp",0.11),("out_wspd",0.11),("out_rad",1.1),("in_temp",0.11)]:
    d2=(X[c].shift(-1)-2*X[c]+X[c].shift()).abs()
    X[c+"_s"]=(d2<=tol)&ok
    if c=="out_rad": X[c+"_s"]&=(X.out_rad>6)|(X.out_rad.shift()>6)
D=X.groupby(["farm","set","day"])[["out_temp_s","out_wspd_s","out_rad_s","in_temp_s"]].mean().reset_index()
D["score"]=D[["out_temp_s","out_wspd_s","out_rad_s"]].mean(axis=1)
print(D.groupby("set").score.describe().round(3))
print(D.sort_values("score",ascending=False).head(15).round(2).to_string(index=False))
D.to_csv(env.LOCAL+"/eda_forensic_22_days.csv",index=False)
