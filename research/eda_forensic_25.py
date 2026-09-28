# -*- coding: utf-8 -*-
"""Forensic 25: linear (extrapolating) night reference vs current submission on cold test rows.
Night rows (19-07h): sub_temp ~ in_temp + in_temp lags 1..3 + out_temp, per farm, least squares."""
import env
import numpy as np, pandas as pd, common
tX, ty, sX = common.load_raw()
S=pd.read_csv("submissions/submission_03.csv").rename(columns={"sub_temp":"pred"})
X=pd.concat([tX[tX.farm.isin(["F13","F47"])].assign(set="train"),sX.assign(set="test")]).sort_values(["farm","t"])
X=X.merge(ty[["row_id","sub_temp"]],on="row_id",how="left").merge(S[["row_id","pred"]],on="row_id",how="left")
full=[]
for f,g in X.groupby("farm"):
    g=g.set_index("t").reindex(range(g.t.min(),g.t.max()+1))
    for k in [1,2,3,6]: g[f"l{k}"]=g.in_temp.shift(k)
    g["farm"]=f; full.append(g.reset_index())
X=pd.concat(full); X=X[X.row_id.notna()]
nt=X[~X.hour.between(8,18)].dropna(subset=["in_temp","l1","l2","l3","l6"])
F=["in_temp","l1","l2","l3","l6","out_temp"]
for f in ["F13","F47"]:
    tr=nt[(nt.farm==f)&(nt.set=="train")]; te=nt[(nt.farm==f)&(nt.set=="test")]
    A=np.c_[tr[F].values,np.ones(len(tr))]; c,*_=np.linalg.lstsq(A,tr.sub_temp.values,rcond=None)
    r=tr.sub_temp-A@c; print(f,"night linear rmse train %.3f"%np.sqrt((r**2).mean()),
        " cold(in<8) train rows resid mean %.2f n=%d"%(r[tr.in_temp<8].mean(),(tr.in_temp<8).sum()))
    lin=np.c_[te[F].values,np.ones(len(te))]@c
    for lo,hi in [(-9,6),(6,8),(8,10),(10,99)]:
        m=(te.in_temp>lo)&(te.in_temp<=hi)
        print(f"  test in_temp ({lo},{hi}] n={m.sum():3d}  submission - linear ref = {np.mean(te.pred[m]-lin[m]):+.2f}")
