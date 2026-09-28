# -*- coding: utf-8 -*-
"""Q1: fuzzy date grouping (near-identical outdoor weather) across F13/F47 days (analysis)."""
import env  # noqa
import numpy as np, pandas as pd
import common
from scipy.sparse.csgraph import connected_components
tX, ty, sX = common.load_raw()
a = pd.concat([tX, sX], ignore_index=True)
a = a[a.farm.isin(["F13", "F47"])].sort_values(["farm", "day", "hour"])
tests = set(zip(sX.farm, sX.day))
days = a.groupby(["farm", "day"]).size()
days = days[days == 24].index.tolist()
W = ["out_temp", "out_hum", "out_rad", "out_wspd"]
X = np.stack([a[(a.farm == f) & (a.day == d)][W].to_numpy(float) for f, d in days])
n = len(days)
sc = np.array([1.0, 5.0, 50.0, 1.0])
D = np.zeros((n, n)); Dv = np.zeros((n, n, 4))
for v in range(4):
    x = X[:, :, v]
    dv = np.nanmean(np.abs(x[:, None, :] - x[None, :, :]), axis=2)
    Dv[:, :, v] = dv
D = (Dv[:, :, :3] / sc[:3]).mean(axis=2)
np.fill_diagonal(D, np.inf)
nn = D.min(axis=1)
print("nearest-neighbour distance quantiles", np.round(np.percentile(nn, [5, 10, 25, 50, 75, 90, 95]), 3))
print("hist", np.histogram(nn, bins=[0, 0.001, 0.01, 0.03, 0.06, 0.1, 0.2, 0.4, 1, 10])[0])
# per-variable for near pairs
i, j = np.where(D < 0.1)
print("pairs D<0.1: %d ; of those exact-equal per var:" % (len(i) // 2), [(Dv[i, j, v] < 1e-9).mean().round(3) for v in range(4)])
for thr in [0.02, 0.05, 0.1, 0.2]:
    nc, lab = connected_components(D < thr, directed=False)
    print("thr", thr, "n groups", nc)
thr = 0.1
nc, lab = connected_components(D < thr, directed=False)
out = pd.DataFrame(days, columns=["farm", "day"]); out["g"] = lab
out["is_test"] = [(f, d) in tests for f, d in days]
out["tmean"] = X[:, :, 0].mean(axis=1)
# order groups by min F13-first-pass day else overall min day
out.to_csv(env.LOCAL + "/deep_cal_6_days.csv", index=False)
comp = out.groupby("g").apply(lambda x: "".join(sorted(x.farm.str[1:])))
print(comp.value_counts().head(12))
# check within-group internal max distance (to catch chaining)
mx = [np.max(D[np.ix_(idx, idx)][~np.eye(len(idx), dtype=bool)]) if len(idx) > 1 else 0 for idx in out.groupby("g").indices.values()]
print("within-group max distance quantiles", np.round(np.percentile(mx, [50, 90, 99, 100]), 3))
np.save(env.LOCAL + "/deep_cal_6_D.npy", D)
