# -*- coding: utf-8 -*-
"""Forensic 4: print hourly traces of suspicious days."""
import env, sys
import numpy as np, pandas as pd
G = pd.read_pickle(env.LOCAL+"/eda_forensic_7_G.pkl")
pd.set_option("display.width",250)
spec = sys.argv[1:] or ["F13:108-110","F13:114-115","F13:47-48","F47:90-91"]
for s in spec:
    f, r = s.split(":"); a,b = map(int,r.split("-"))
    g = G[(G.farm==f)&(G.day.between(a,b))]
    print("==",s)
    print(g[["day","hour","in_temp","in_hum","in_co2","out_temp","out_rad","act_vent","act_heating","act_thermal","sub_temp","pred_lin","res","sub_ec"]].round(2).to_string(index=False))
