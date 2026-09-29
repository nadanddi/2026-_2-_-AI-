# -*- coding: utf-8 -*-
"""Forensic 16: test_X block/day oddities vs season-near train days; F13-F47 daily EC co-movement."""
import env
import numpy as np, pandas as pd, common
tX, ty, sX = common.load_raw()
tX["set"]="train"; sX["set"]="test"
X=pd.concat([tX[tX.farm.isin(["F13","F47"])],sX])
X["heat100"]=X.act_heating>=100; X["co2on"]=X.act_co2>0; X["fogon"]=X.act_fog>0
X["vent_on"]=X.act_vent>0; X["night_dt"]=np.where(~X.hour.between(7,18), X.in_temp-X.out_temp, np.nan)
cols=["in_temp","out_temp","in_hum","in_co2","out_rad","heat100","co2on","fogon","vent_on","act_circfan","act_thermal","act_shade","night_dt"]
D=X.groupby(["farm","set","day"])[cols].mean().reset_index()
def block_id(days):
    b=np.zeros(len(days),int); 
    for i in range(1,len(days)): b[i]=b[i-1]+(days[i]!=days[i-1]+1)
    return b
out=[]
for f in ["F13","F47"]:
    for s in ["train","test"]:
        d=D[(D.farm==f)&(D.set==s)].sort_values("day").copy()
        d["blk"]=block_id(d.day.values); out.append(d)
D=pd.concat(out)
late=D[D.day>=175]
pd.set_option("display.width",250)
print(late.groupby(["farm","set","blk"]).agg(d0=("day","min"),d1=("day","max"),**{c:(c,"mean") for c in cols}).round(2).to_string())
# day-level z vs train days>=150 of same farm
ref=D[(D.set=="train")&(D.day>=150)]
z=[]
for f in ["F13","F47"]:
    r=ref[ref.farm==f]; t=D[(D.farm==f)&(D.set=="test")]
    Z=(t[cols]-r[cols].mean())/r[cols].std()
    t=t.assign(zmax=Z.abs().max(axis=1), zcol=Z.abs().idxmax(axis=1))
    z.append(t)
Z=pd.concat(z).sort_values("zmax",ascending=False)
print("\nmost atypical test days vs train days>=150:"); print(Z[["farm","day","zmax","zcol"]+cols].head(10).round(2).to_string(index=False))
# F13 vs F47 daily EC at shifts
e13=ty[ty.farm=="F13"].groupby("day").sub_ec.mean(); e47=ty[ty.farm=="F47"].groupby("day").sub_ec.mean()
for sh in [-3,-2,-1,0,1,2,3]:
    a=e13; b=e47.copy(); b.index=b.index+sh
    j=pd.concat([a,b],axis=1,join="inner").dropna()
    dj=j.diff().dropna()
    print(f"shift F47+{sh}: n={len(j)} corr level {j.corr().iloc[0,1]:.2f}  corr of daily changes {dj.corr().iloc[0,1]:.2f}")
