# -*- coding: utf-8 -*-
"""ST1 (structure audit; 2026-10-03 집 클로드).  Follow-up of ST0 (identical-weather
neighbour pairs switch between even- and odd-start phase every ~10-15 records).
train_X inputs only.  For each farm and each neighbour pair (d, d+1), both in
train_X: z-scored outdoor 24h distance (RMSE over 96 values, scaled by the
farm's per-variable SD).  Print the record sequence F13 day 8..60 and F47 day
6..60 compactly (distance, '=' identical), plus the distance distribution
(quantiles) of identical vs non-identical pairs, and of pairs (d, d+2)."""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd

V = ["out_temp", "out_hum", "out_rad", "out_wspd"]
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id"] + V)
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
for f in ("F13", "F47"):
    G = X[X.farm == f]
    sd = G[V].std()
    W = {v: G.pivot_table(index="day", columns="hour", values=v) / sd[v] for v in V}
    days = sorted(W[V[0]].index)
    def dist(a, b):
        x = np.concatenate([W[v].loc[a].values for v in V]); y = np.concatenate([W[v].loc[b].values for v in V])
        ok = ~np.isnan(x) & ~np.isnan(y)
        return float(np.sqrt(np.mean((x[ok] - y[ok]) ** 2)))
    s = set(days)
    d1 = {d: dist(d, d + 1) for d in days if d + 1 in s}
    d2 = {d: dist(d, d + 2) for d in days if d + 2 in s}
    a = np.array(list(d1.values()))
    print("\n%s neighbour (d,d+1) distance quantiles: %s" % (f, np.round(np.quantile(a, [0, .1, .25, .3, .35, .5, .75, .9]), 3)))
    print("%s (d,d+2) distance quantiles: %s" % (f, np.round(np.quantile(list(d2.values()), [0, .1, .25, .5, .75, .9]), 3)))
    print("%s share (d,d+1) < .05: %.3f, < .2: %.3f; (d,d+2) < .05: %.3f" % (
        f, (a < .05).mean(), (a < .2).mean(), (np.array(list(d2.values())) < .05).mean()))
    line = []
    for d in days:
        if d > 120:
            break
        line.append("%d:%s" % (d, ("=" if d1.get(d, 9) < 1e-9 else ("%.2f" % d1[d] if d in d1 else "-"))))
    print("sequence (day:dist to next):")
    for k in range(0, len(line), 12):
        print("   " + "  ".join(line[k:k + 12]))
