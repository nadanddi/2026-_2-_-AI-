# -*- coding: utf-8 -*-
"""Forensic 20: night out_rad floor value (4 vs 5 vs other) as a day-level marker."""
import env
import numpy as np, pandas as pd, common
tX, ty, sX = common.load_raw()
tX["set"]="train"; sX["set"]="test"
X=pd.concat([tX[tX.farm.isin(["F13","F47"])],sX]).merge(ty[["row_id","sub_ec","sub_temp"]],on="row_id",how="left")
n=X[X.hour.isin([0,1,2,3,4,21,22,23])]
print("night out_rad value counts:",n.out_rad.value_counts().head(8).to_dict())
print("train out_rad<4 rows:",(X[X.set=='train'].out_rad<4).sum(), X[(X.set=='train')&(X.out_rad<4)].out_rad.value_counts().to_dict())
d=n.groupby(["farm","set","day"]).out_rad.agg(lambda s: s.value_counts().index[0]).rename("floor").reset_index()
d["pure"]=n.groupby(["farm","set","day"]).out_rad.agg(lambda s:s.nunique()).values
print("per-day night floor:", d.groupby(["set","floor"]).size().to_dict(), " share of days with single night value:", (d.pure==1).mean().round(2))
# within-day: does night value switch? sequence of floors per farm
for f in ["F13","F47"]:
    s=d[d.farm==f].sort_values("day")
    print(f,"".join({4:"4",5:"5"}.get(int(v),"x") for v in s.floor.values))
# relation to EC zigzag
e=X[X.set=="train"].groupby(["farm","day"]).sub_ec.mean().rename("ec").reset_index()
dd=d.merge(e,on=["farm","day"])
for f in ["F13","F47"]:
    q=dd[dd.farm==f].sort_values("day").copy()
    q["res"]=q.ec-q.ec.rolling(5,center=True,min_periods=3).median()
    print(f, q.groupby("floor").res.agg(["size","mean"]).round(3).to_dict())
