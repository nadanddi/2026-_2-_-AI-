# -*- coding: utf-8 -*-
"""Q1 prep: group days by identical 24h outdoor-weather vector across ALL farms (analysis only)."""
import env  # noqa
import numpy as np, pandas as pd
import common
tX, ty, sX = common.load_raw()
W = ["out_temp", "out_hum", "out_rad", "out_wspd"]
a = pd.concat([tX, sX], ignore_index=True).sort_values(["farm", "t"])
cnt = a.groupby(["farm", "day"]).size()
full = cnt[cnt == 24].index
a = a.set_index(["farm", "day"]).loc[full].reset_index()
def key(g):
    v = g.sort_values("hour")[W].round(2).to_numpy()
    return hash(np.nan_to_num(v, nan=-999).tobytes())
keys = a.groupby(["farm", "day"]).apply(key).rename("wk").reset_index()
keys["is_test"] = keys.set_index(["farm", "day"]).index.isin(sX.groupby(["farm", "day"]).size().index)
keys.to_csv(env.LOCAL + "/deep_cal_1_keys.csv", index=False)
tgt = keys[keys.farm.isin(["F13", "F47"])]
oth = keys[~keys.farm.isin(["F13", "F47"])]
print("F13/F47 days", len(tgt), "unique wk", tgt.wk.nunique())
m = tgt.merge(oth[["farm", "day", "wk"]].rename(columns={"farm": "of", "day": "od"}), on="wk", how="left")
print("F13/F47 days with a weather match in other farms:", m.groupby(["farm", "day"]).of.apply(lambda s: s.notna().any()).mean())
print(m.of.value_counts().head(20))
# within other farms: how many share weather among themselves
g = oth.groupby("wk").farm.agg(["size", "nunique"])
print("other-farm weather groups: n", len(g), "size>1", (g["size"] > 1).mean(), "nfarm dist", g["nunique"].value_counts().head())
