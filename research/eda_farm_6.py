# -*- coding: utf-8 -*-
"""F13/F47 are day-stitched: continuity tests and whether midnight-reset features explain sub_temp better."""
import env  # noqa
import numpy as np, pandas as pd
P = pd.read_pickle(env.LOCAL + "/eda_farm_panels.pkl")
rows = []
for f, p in P.items():
    a = p.in_temp.values; h = p.hour.values
    i0 = np.where((h == 0))[0]; i0 = i0[i0 > 0]
    i1 = np.where(h == 1)[0]
    c_mid = pd.Series(a[i0]).corr(pd.Series(a[i0 - 1]))
    c_1 = pd.Series(a[i1]).corr(pd.Series(a[i1 - 1]))
    dm = p.groupby("day").in_temp.mean()
    rows.append(dict(farm=f, corr_23_0=c_mid, corr_0_1=c_1, daymean_ac1=dm.autocorr(1), daymean_ac7=dm.autocorr(7)))
C = pd.DataFrame(rows).set_index("farm").round(3)
print(C.loc[["F13", "F47", "F50", "F38", "F30", "F09", "F24", "F31"]].to_string())
print("others median:", C.drop(["F13", "F47"]).median().round(3).to_dict())
# midnight sub_temp jump vs air jump in F13/F47
for f in ["F13", "F47"]:
    p = P[f]
    ds = p.sub_temp.diff(); di = p.in_temp.diff(); h0 = (p.hour == 0) & ds.notna() & di.notna()
    b = np.polyfit(di[h0], ds[h0], 1)
    r = np.corrcoef(di[h0], ds[h0])[0, 1]
    # at hour 0: does sub(0) track in(0) of new day, or prev-day state?
    L = p[p.sub_temp.notna()]
    for hh in [0, 1, 2, 3, 6, 12]:
        q = L[L.hour == hh]
        pred_prev = q.ew4.shift(0)  # ewm across midnight
        # within-day reset ewm (start at hour 0)
    print(f, "h0: dsub = %.2f*din + %.2f, r=%.2f, n=%d; sd dsub h0 %.2f" % (b[0], b[1], r, h0.sum(), ds[h0].std()))
# day-reset ewm features: fit sub ~ a + b*ewm_reset(hl) vs across-midnight ewm, per hour-of-day RMSE
def ewm_reset(p, hl):
    return p.groupby("day").in_temp.transform(lambda s: s.ewm(halflife=hl, ignore_na=True).mean())
for f in ["F13", "F47", "F50", "F38"]:
    p = P[f].copy()
    for hl in (2, 4, 8):
        p["er%d" % hl] = ewm_reset(p, hl)
    L = p[p.sub_temp.notna() & (p.is_test != True)]
    out = {}
    for name, cols in [("cross-mid ew2+ew8+ew24", ["ew2", "ew8", "ew24"]), ("reset er2+er4+er8", ["er2", "er4", "er8"]),
                       ("both", ["ew2", "ew8", "ew24", "er2", "er4", "er8"])]:
        A = np.c_[np.ones(len(L)), L[cols].values]; m = np.isfinite(A).all(1)
        c, *_ = np.linalg.lstsq(A[m], L.sub_temp.values[m], rcond=None)
        r = L.sub_temp.values[m] - A[m] @ c
        hr = L.hour.values[m]
        out[name] = (np.sqrt(np.mean(r ** 2)).round(3), np.sqrt(np.mean(r[hr < 4] ** 2)).round(3))
    print(f, "linear RMSE (all, hours0-3):", out)
