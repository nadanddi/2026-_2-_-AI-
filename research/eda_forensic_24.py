# -*- coding: utf-8 -*-
"""Forensic 24: cold-end extrapolation. Train relation night sub_temp - in_temp by in_temp bin;
current submission (submissions/submission_03.csv) on test rows by in_temp bin."""
import env
import numpy as np, pandas as pd, common
tX, ty, sX = common.load_raw()
S=pd.read_csv("submissions/submission_03.csv")
te=sX.merge(S,on="row_id")
tr=tX[tX.farm.isin(["F13","F47"])].merge(ty[["row_id","sub_temp"]],on="row_id")
bins=[-5,6,8,10,12,14,18,40]
tr["b"]=pd.cut(tr.in_temp,bins); te["b"]=pd.cut(te.in_temp,bins)
night=lambda d:d[~d.hour.between(8,17)]
a=night(tr).groupby("b").apply(lambda g:pd.Series(dict(n=len(g),sub=g.sub_temp.mean(),gap=(g.sub_temp-g.in_temp).mean())))
b=night(te).groupby("b").apply(lambda g:pd.Series(dict(n=len(g),pred=g.sub_temp.mean(),gap=(g.sub_temp-g.in_temp).mean())))
print(pd.concat([a.add_prefix("train_"),b.add_prefix("test_")],axis=1).round(2))
print("test pred sub_temp min %.2f, share pred<7.3 (train 1%% label) %.3f; train label share<7.3 %.3f"%(te.sub_temp.min(),(te.sub_temp<7.3).mean(),(tr.sub_temp<7.3).mean()))
