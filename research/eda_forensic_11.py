# -*- coding: utf-8 -*-
"""Forensic 11: row-level physical-violation flags, train vs test rates, and label-residual link.
 V1 night in_temp < out_temp - 3 ; V2 night |sub_temp-in_temp|>6 (train only);
 V3 night co2<300 ; V4 |d1 in_co2|>150 at night with act_co2==0 and vent==0;
 V5 in_temp flat >=6h ; V6 |d1 in_temp|>4 at night (hours 19-6)"""
import env
import numpy as np, pandas as pd
G = pd.read_pickle(env.LOCAL+"/eda_forensic_7_G.pkl")
night = ~G.hour.between(7,18)
G["V1"]= night & (G.in_temp < G.out_temp-3)
G["V3"]= night & (G.in_co2<300)
G["V4"]= night & (G.d1_in_co2.abs()>150) & (G.act_co2==0)
G["V5"]= G.temp_run>=6
G["V6"]= night & (G.d1_in_temp.abs()>4)
G["V7"]= night & (G.d1_in_hum.abs()>15)
G["V2"]= night & ((G.sub_temp-G.in_temp).abs()>6)
V=["V1","V3","V4","V5","V6","V7"]
G["Vany"]=G[V].any(axis=1)
print((G.groupby(["farm","set"])[V+["Vany","V2"]].mean()*100).round(2).to_string())
tr=G[G.set=="train"]
for v in V+["Vany"]:
    m=tr[v]
    print(f"{v}: n={m.sum()} |sub_temp res| {tr.res[m].abs().mean():.2f} vs {tr.res[~m].abs().mean():.2f}; P(V2|v)={tr.V2[m].mean():.2f} vs {tr.V2[~m].mean():.3f}")
print("V1 train days:", tr[tr.V1].groupby(["farm","day"]).size().to_dict())
print("V2 train days:", tr[tr.V2].groupby(["farm","day"]).size().to_dict())
print("V1 test days:", G[(G.set=='test')&G.V1].groupby(["farm","day"]).size().to_dict())
G[["row_id","set","V1","V2","V3","V4","V5","V6","V7","Vany"]].to_csv(env.LOCAL+"/eda_forensic_11_flags.csv",index=False)
