# -*- coding: utf-8 -*-
"""(1) midnight-reset features for the stitched F13/F47 series; (2) cost of 1-degree label rounding, simulated on F13/F47."""
import env  # noqa
import numpy as np, pandas as pd
import lightgbm as lgb
from harness import load, views, score

panel, lab_t, _ = load()
v = views(panel)["temp"]
def add_reset(df):
    df = df.sort_values(["farm", "t"]).copy()
    g = df.groupby(["farm", "day"]).in_temp
    for hl in (1, 2, 4, 8):
        df["in_temp_er%d" % hl] = g.transform(lambda s: s.ewm(halflife=hl, ignore_na=True).mean())
    df["in_temp_h0"] = g.transform("first")
    df["in_temp_since0"] = df.in_temp - df.in_temp_h0
    df["in_temp_prevday_last"] = df.groupby("farm").in_temp.shift(1).where(df.hour == 0)
    df["in_temp_prevday_last"] = df.groupby(["farm", "day"]).in_temp_prevday_last.transform("first")
    df["mid_jump"] = df.in_temp_h0 - df.in_temp_prevday_last
    return df
pa = add_reset(panel)
lab = pa[(~pa.is_test) & pa.sub_temp.notna()].reset_index(drop=True)
R = ["in_temp_er1", "in_temp_er2", "in_temp_er4", "in_temp_er8", "in_temp_h0", "in_temp_since0", "mid_jump"]
P = dict(objective="huber", n_estimators=800, learning_rate=0.04, num_leaves=63, min_child_samples=40,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0, n_jobs=2, verbose=-1)
fac = lambda s: lgb.LGBMRegressor(random_state=s, **P)
out = {}
for kind in ("A", "B"):
    for name, cols in [("base view", v), ("+reset", v + R)]:
        r, sd, per = score(lab, "sub_temp", cols, fac, kind=kind, seeds=(7, 101))
        out[(kind, name)] = r; print(kind, name, round(r, 4), np.round(per, 3), flush=True)
# rounding simulation
lab["sub_temp_rnd"] = np.round(lab.sub_temp)
class RoundFit:
    def __init__(self, s): self.m = fac(s)
    def fit(self, X, y): self.m.fit(X, np.round(y)); return self
    def predict(self, X): return self.m.predict(X)
for kind in ("A", "B"):
    r, sd, per = score(lab, "sub_temp", v, lambda s: RoundFit(s), kind=kind, seeds=(7, 101))
    print(kind, "trained on ROUNDED labels (eval on raw)", round(r, 4), np.round(per, 3), flush=True)
