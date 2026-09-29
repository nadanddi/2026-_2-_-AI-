# -*- coding: utf-8 -*-
"""Forensic 1: per-variable resolution, discrete levels, sentinel values; F13/F47 train vs test."""
import env
import numpy as np, pandas as pd, common
tX, ty, sX = common.load_raw()
COLS = ["out_temp","out_hum","out_rad","out_wspd","in_temp","in_hum","in_co2",
        "act_vent","act_shade","act_thermal","act_heating","act_circfan","act_co2","act_fog"]
def decs(v):
    v = v.dropna().values
    out = {}
    for d in range(0,5):
        m = np.isclose(v*10**d, np.round(v*10**d), atol=1e-6)
        out[d] = m
    # number of decimals needed
    need = np.full(len(v), 5)
    for d in range(4,-1,-1):
        need[out[d]] = d
    return pd.Series(need).value_counts(normalize=True).sort_index().round(4).to_dict()
rows=[]
for f in ["F13","F47"]:
    a = tX[tX.farm==f]; b = sX[sX.farm==f]
    for c in COLS:
        for nm, d in [("train",a),("test",b)]:
            v = d[c]
            vc = v.value_counts(normalize=True)
            rows.append(dict(farm=f,col=c,set=nm,n=len(v),na=round(v.isna().mean(),4),
                min=v.min(),max=v.max(),nuniq=v.nunique(),decs=decs(v),
                top3=";".join(f"{k}:{p:.3f}" for k,p in vc.head(3).items()),
                frac_int=round(np.isclose(v.dropna()%1,0).mean(),4)))
r = pd.DataFrame(rows)
pd.set_option("display.width",250); pd.set_option("display.max_colwidth",60); pd.set_option("display.max_rows",200)
print(r.to_string())
r.to_csv(env.LOCAL+"/eda_forensic_1_resolution.csv",index=False)
# labels
for f in ["F13","F47"]:
    y = ty[ty.farm==f]
    for c in ["sub_temp","sub_ec"]:
        print(f,c,"decs",decs(y[c]),"nuniq",y[c].nunique(),"min",y[c].min(),"max",y[c].max())
