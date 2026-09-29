# -*- coding: utf-8 -*-
"""Forensic 12: label-side anomalies. X rows without y; label runs, linear runs, resolution segments."""
import env
import numpy as np, pandas as pd, common
tX, ty, sX = common.load_raw()
miss = tX[~tX.row_id.isin(ty.row_id)]
print("X rows without y:", len(miss)); print(miss.farm.value_counts().head(20).to_dict())
print("hours:", miss.hour.value_counts().sort_index().to_dict())
# full days missing?
fd = miss.groupby(["farm","day"]).size()
print("per farm-day sizes value_counts", fd.value_counts().head(10).to_dict())
print("farms", miss.farm.nunique())
# y rows without X ?
print("y rows not in X:", (~ty.row_id.isin(tX.row_id)).sum())
# y NaN in sub_temp
print("sub_temp NaN in ty:", ty.sub_temp.isna().sum(), ty[ty.sub_temp.isna()].farm.value_counts().head().to_dict())
# F13/F47 missing X->y?
print("F13/F47 X without y:", miss.farm.isin(["F13","F47"]).sum())
# inputs of those rows: are they NaN-heavy?
print("in_temp NaN rate among X-without-y %.3f vs overall %.3f"%(miss.in_temp.isna().mean(), tX.in_temp.isna().mean()))
# label runs F13/F47
Y = ty[ty.farm.isin(["F13","F47"])].sort_values(["farm","t"])
for f in ["F13","F47"]:
    y=Y[Y.farm==f]
    for c,tol in [("sub_temp",0.0051),("sub_ec",0.00051)]:
        v=y[c].values; t=y.t.values
        # constant runs
        runs=[];s=0
        for i in range(1,len(v)+1):
            if i==len(v) or v[i]!=v[s] or t[i]-t[s]!=i-s:
                runs.append((i-s,s)); s=i
        rl=np.array([r for r,_ in runs])
        long=[(int(y.day.iloc[s]),int(y.hour.iloc[s]),r,v[s]) for r,s in runs if r>=4]
        # linear runs: d1 equal (within tol) with nonzero slope over >=4 steps
        d1=np.diff(v); cont=np.diff(t)==1
        lin=(np.abs(np.diff(d1))<tol*2)&(np.abs(d1[1:])>tol)&cont[1:]&cont[:-1]
        lr=[];k=0
        for x in lin:
            if x:k+=1
            else:
                if k: lr.append(k)
                k=0
        lr=np.array(lr) if lr else np.array([0])
        print(f,c,"const runs>=3:",(rl>=3).sum(),">=4:",(rl>=4).sum(),"max",rl.max(), "long:",long[:8], "| lin runs(>=2 consecutive equal-slope):",(lr>=2).sum(),"max",lr.max())
    # resolution of sub_temp over time: frac of 2-decimal values by 20-day period
    y2=y.assign(per=y.day//20*20, d2=~np.isclose(y.sub_temp*10,np.round(y.sub_temp*10)), e3=~np.isclose(y.sub_ec*100,np.round(y.sub_ec*100)))
    print(f,"share needing 2 decimals (sub_temp) / 3 decimals (sub_ec) by period:", y2.groupby("per")[["d2","e3"]].mean().round(2).T.to_dict())
