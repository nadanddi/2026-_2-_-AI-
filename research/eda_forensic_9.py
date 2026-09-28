# -*- coding: utf-8 -*-
"""Forensic 9: low-CO2 rows - physical (daytime drawdown) or artefact?"""
import env
import numpy as np, pandas as pd
G = pd.read_pickle(env.LOCAL+"/eda_forensic_7_G.pkl")
tr=G[G.set=="train"]
lo=tr[tr.in_co2<300]
print("low co2: day(7-18h) share %.2f, out_rad>100 share %.2f, vent>0 share %.2f"%(lo.hour.between(7,18).mean(), (lo.out_rad>100).mean(), (lo.act_vent>0).mean()))
night=lo[~lo.hour.between(7,19)]
print("night low co2 rows",len(night), night.groupby(["farm","day"]).size().to_dict())
# per-day 159 F47
g=G[(G.farm=="F47")&(G.day.isin([158,159]))][["day","hour","in_temp","in_hum","in_co2","out_rad","act_vent","act_co2","sub_temp","res"]]
print(g.round(2).to_string(index=False))
# season: share of daytime hours (10-16) with co2<350 by 20-day period, train vs test
G["per"]=(G.day//20)*20
print(G[G.hour.between(10,16)].assign(l=lambda d:d.in_co2<350).groupby(["per","set"]).l.mean().unstack().round(3))
