# -*- coding: utf-8 -*-
"""Q1: continuity on fuzzy dates; validate first-pass order; place second-pass-only dates."""
import env  # noqa
import numpy as np, pandas as pd
import common
tX, ty, sX = common.load_raw()
k = pd.read_csv(env.LOCAL + "/deep_cal_7_days.csv")
a = pd.concat([tX, sX], ignore_index=True)
a = a[a.farm.isin(["F13", "F47"])]
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]
nd = k.date.max() + 1
X = np.zeros((nd, 24, 4))
for dt, g in k.groupby("date"):
    mats = [a[(a.farm == f) & (a.day == d)].sort_values("hour")[W].to_numpy(float) for f, d in zip(g.farm, g.day)]
    X[dt] = np.nanmedian(np.stack(mats), axis=0)
np.save(env.LOCAL + "/deep_cal_8_X.npy", X)
sc = np.nanstd(np.diff(X, axis=1).reshape(-1, 4), axis=0)
def C_of(X, wts=(1, 1, 0.3)):
    n = len(X); C = np.zeros((n, n))
    for vi, w in enumerate(wts):
        a23 = X[:, 23, vi]; a22 = X[:, 22, vi]; b0 = X[:, 0, vi]; b1 = X[:, 1, vi]
        pg = 0.5 * ((a23 - a22)[:, None] + (b1 - b0)[None, :])
        C += w * (((b0[None, :] - a23[:, None]) - pg) / sc[vi]) ** 2
    np.fill_diagonal(C, np.inf)
    return np.nan_to_num(C, nan=50)
C = C_of(X)
np.save(env.LOCAL + "/deep_cal_8_C.npy", C)
first = np.arange(113)
rk = np.array([(C[i] < C[i, i + 1]).sum() for i in first])
rkp = np.array([(C[:, i + 1] < C[i, i + 1]).sum() for i in first])
print("first-pass i->i+1: successor rank top1 %.2f top3 %.2f | predecessor rank top1 %.2f top3 %.2f"
      % ((rk == 0).mean(), (rk < 3).mean(), (rkp == 0).mean(), (rkp < 3).mean()))
both = (rk == 0) & (rkp == 0)
print("mutual best (unique link) share: %.2f" % both.mean())
ci = np.array([C[i, i + 1] for i in first])
print("cost i->i+1 quantiles", np.round(np.percentile(ci, [25, 50, 75, 90, 95]), 2))
print("largest first-pass breaks:", [(i, round(ci[i], 1)) for i in np.argsort(-ci)[:10]])
# the 2nd-pass-only run 114..126: consecutive costs
for i in range(114, 126):
    print("  %d->%d cost %.2f  best succ of %d: %s" % (i, i + 1, C[i, i + 1], i, np.argsort(C[i])[:3]))
# insertion of block [s..e] between p and p+1
for s, e in [(115, 125), (114, 114), (126, 126), (115, 121), (122, 125)]:
    sc_ins = [(C[p, s] + C[e, p + 1] - C[p, p + 1], p) for p in range(112)]
    sc_ins.sort()
    print("insert block %d..%d best after p:" % (s, e), [(p, round(v, 2)) for v, p in sc_ins[:5]],
          " start pred rank", np.argsort(C[:, s])[:5], " end succ", np.argsort(C[e])[:5])
# daily descriptors by date for season placement
rad = X[:, :, 3]; fl = np.nanmin(rad, axis=1, keepdims=True)
tm = X[:, :, 0].mean(1)
print("date tmean (first pass, every 8):", [(i, round(tm[i], 1)) for i in range(0, 114, 8)])
print("date tmean 114..126:", [(i, round(tm[i], 1)) for i in range(114, 127)])
