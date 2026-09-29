# -*- coding: utf-8 -*-
"""Forensic 17: follow-ups on test oddities: cold/heating block OOD, F47 d223 CO2, all-zero actuator days."""
import env
import numpy as np, pandas as pd, common
tX, ty, sX = common.load_raw()
tX["set"]="train"; sX["set"]="test"
X=pd.concat([tX[tX.farm.isin(["F13","F47"])],sX]).merge(ty[["row_id","sub_temp","sub_ec"]],on="row_id",how="left")
A=["act_vent","act_shade","act_thermal","act_heating","act_circfan","act_co2","act_fog"]
d=X.groupby(["farm","set","day"]).agg(h100=("act_heating",lambda s:(s>=100).mean()), out=("out_temp","mean"), itemp=("in_temp","mean"),
    allzero=("act_vent",lambda s: np.nan), ec=("sub_ec","mean"), st=("sub_temp","mean")).reset_index()
az=X.groupby(["farm","set","day"])[A].apply(lambda g:(g[["act_vent","act_shade","act_thermal","act_circfan"]].sum().sum()==0)).rename("az").reset_index()
d=d.merge(az,on=["farm","set","day"])
print("days with heat100 share>=0.4:", d[d.h100>=0.4].groupby(["farm","set"]).size().to_dict())
print(" train such days:", d[(d.h100>=0.4)&(d.set=="train")][["farm","day","h100","out","itemp","ec","st"]].round(2).values.tolist())
print("days with out_temp mean<3:", d[d.out<3].groupby(["farm","set"]).size().to_dict())
print("in_temp mean<10 days:", d[d.itemp<10].groupby(["farm","set"]).size().to_dict())
print("rows in_temp<8: train %.3f test %.3f"%((X[X.set=='train'].in_temp<8).mean(),(X[X.set=='test'].in_temp<8).mean()))
print("all-curtain/vent/fan zero days:", d[d.az].groupby(["farm","set"]).size().to_dict(), d[d.az][["farm","set","day"]].values.tolist()[:20])
g=X[(X.farm=="F47")&(X.day.isin([222,223,224]))]
print(g[["day","hour","in_temp","in_co2","act_co2","act_vent","out_rad"]].to_string(index=False))
