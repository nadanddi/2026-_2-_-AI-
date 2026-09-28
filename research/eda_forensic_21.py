# -*- coding: utf-8 -*-
"""Forensic 21: the non-floor night out_rad days (linear ramp?) - print weather + indoor."""
import env
import numpy as np, pandas as pd, common
tX, ty, sX = common.load_raw()
X=pd.concat([tX[tX.farm.isin(["F13","F47"])].assign(set="train"),sX.assign(set="test")])
n=X[X.hour.isin([0,1,2,3,4,21,22,23])&(X.out_rad>10)]
print(n.groupby(["farm","set","day"]).size())
for f,d in n.groupby(["farm","day"]).size().index:
    g=X[(X.farm==f)&(X.day.between(d-0,d+0))]
    print("==",f,d); print(g[["hour","out_temp","out_hum","out_rad","out_wspd","in_temp","in_hum","in_co2"]].T.to_string(header=False))
