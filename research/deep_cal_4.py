# -*- coding: utf-8 -*-
"""Q1: how discriminative is midnight continuity? null from within-day boundaries."""
import env  # noqa
import numpy as np, pandas as pd
import common
tX, ty, sX = common.load_raw()
k = pd.read_csv(env.LOCAL + "/deep_cal_2_days.csv")
a = pd.concat([tX, sX], ignore_index=True)
a = a[a.farm.isin(["F13", "F47"])]
rep = k.sort_values("day").groupby("date").first().reset_index()
W = ["out_temp", "out_hum", "out_wspd"]
X = np.stack([a[(a.farm == r.farm) & (a.day == r.day)].sort_values("hour")[W].to_numpy(float) for _, r in rep.iterrows()])
sc = np.nanstd(np.diff(X, axis=1).reshape(-1, 3), axis=0)
def cst(x22, x23, y0, y1, wts=(1, 1, 0.3)):
    c = 0
    for vi, w in enumerate(wts):
        c = c + w * (((y0[..., vi] - x23[..., vi]) - 0.5 * ((x23[..., vi] - x22[..., vi]) + (y1[..., vi] - y0[..., vi]))) / sc[vi]) ** 2
    return c
for h in [2, 5, 11, 17, 21]:
    c = cst(X[:, h - 1], X[:, h], X[:, h + 1], X[:, h + 2])
    print("within-day boundary %d|%d: cost median %.2f p90 %.2f" % (h, h + 1, np.nanmedian(c), np.nanpercentile(c, 90)))
C = np.load(env.LOCAL + "/deep_cal_3_cost.npy")
ci = np.array([C[i, i + 1] for i in range(len(X) - 1)])
print("id i->i+1 cost quantiles", np.round(np.percentile(ci, [10, 25, 50, 75, 90]), 2))
# per-variable gap at id i -> i+1 vs within-day hour 23->0 like jump
for vi, v in enumerate(W):
    g = X[1:, 0, vi] - X[:-1, 23, vi]
    wd = X[:, 12, vi] - X[:, 11, vi]
    wn = X[:, 3, vi] - X[:, 2, vi]
    print(v, "|gap id->id+1| median %.3f ; within-night |d| median %.3f ; midday %.3f" % (np.nanmedian(np.abs(g)), np.nanmedian(np.abs(wn)), np.nanmedian(np.abs(wd))))
# print hour 0 and 23 temp for first 30 ids
for i in range(0, 40):
    print(i, rep.farm[i], rep.day[i], "t23 %.1f -> next t0 %.1f | hum23 %.0f -> %.0f" % (X[i, 23, 0], X[i + 1, 0, 0], X[i, 23, 1], X[i + 1, 0, 1]))
