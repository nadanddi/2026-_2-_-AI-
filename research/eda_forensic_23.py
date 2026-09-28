# -*- coding: utf-8 -*-
"""Forensic 23: partial interpolated-weather days and model error on interpolated-weather train days."""
import env
import numpy as np, pandas as pd, common
tX, ty, sX = common.load_raw()
X=pd.concat([tX[tX.farm.isin(["F13","F47"])].assign(set="train"),sX.assign(set="test")])
for f,d in [("F13",207),("F13",93)]:
    g=X[(X.farm==f)&(X.day==d)]
    print("==",f,d); print(g[["hour","out_temp","out_rad","out_wspd","in_temp"]].T.to_string(header=False))
# EC oof err and temperature residual on interpolated days
E=pd.read_csv(env.LOCAL+"/eda_forensic_10_err.csv")
E["farm"]=E.row_id.str[:3]; E["day"]=E.row_id.str[4:7].astype(int)
bad={("F13",106),("F13",107),("F13",108),("F47",109),("F47",110),("F47",111),("F47",112)}
m=E.apply(lambda r:(r.farm,r.day) in bad,axis=1)
print("EC oof rmse interp-weather days %.3f (n=%d) vs rest %.3f"%(np.sqrt(np.nanmean(E.err[m]**2)),m.sum(),np.sqrt(np.nanmean(E.err[~m]**2))))
near=E.apply(lambda r:any((r.farm,r.day+k) in bad for k in range(-5,6)) and (r.farm,r.day) not in bad,axis=1)
print("EC oof rmse +-5 days around: %.3f; bias on interp days %.3f"%(np.sqrt(np.nanmean(E.err[near]**2)), np.nanmean(E.err[m])))
G=pd.read_pickle(env.LOCAL+"/eda_forensic_7_G.pkl")
G["bad"]=G.apply(lambda r:(r.farm,r.day) in bad,axis=1)
tr=G[G.set=="train"]
print("sub_temp simple-model |res| interp days %.2f vs rest %.2f"%(tr[tr.bad].res.abs().mean(),tr[~tr.bad].res.abs().mean()))
# daily out_rad sum on those days vs typical
dd=X.groupby(["farm","set","day"]).out_rad.sum()
print("daily out_rad sum: interp days",[int(dd[(f,'train',d)]) for f,d in sorted(bad)], "test F13 208",int(dd[("F13","test",208)]),"median all",int(dd.median()))
