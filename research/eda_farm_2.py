# -*- coding: utf-8 -*-
"""Cold-regime shape: binned sub_temp - ew4(in_temp) by ew4 bins, per farm; cold slope; similarity to F13/F47."""
import env  # noqa
import numpy as np, pandas as pd
P = pd.read_pickle(env.LOCAL + "/eda_farm_panels.pkl")
bins = [-5, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 40]
out = {}; slope = []
for f, p in P.items():
    L = p[p.sub_temp.notna() & (p.is_test != True)].copy()
    L["d"] = L.sub_temp - L.ew4
    L["b"] = pd.cut(L.ew4, bins)
    g = L.groupby("b", observed=False).d
    out[f] = g.mean().round(2).where(g.count() >= 20)
    c = L[L.ew4 < 12]
    w = L[(L.ew4 >= 12) & (L.ew4 < 20)]
    def fit(D):
        if len(D) < 50: return (np.nan, np.nan)
        A = np.c_[np.ones(len(D)), D.ew4, D.ew24]
        m = np.isfinite(A).all(1)
        cc, *_ = np.linalg.lstsq(A[m], D.sub_temp.values[m], rcond=None)
        return cc
    cc = fit(c); cw = fit(w)
    slope.append(dict(farm=f, n_cold=len(c), cold_a=cc[0], cold_b4=cc[1] if len(c)>=50 else np.nan,
                      cold_b24=cc[2] if len(c)>=50 else np.nan,
                      warm_a=cw[0], warm_b4=cw[1] if len(w)>=50 else np.nan, warm_b24=cw[2] if len(w)>=50 else np.nan,
                      night_cold_d=L[(L.ew4 < 10) & L.hour.isin([0,1,2,3,4,5,6])].d.mean(),
                      min_sub=L.sub_temp.min(), q01_sub=L.sub_temp.quantile(.01), q01_in=L.in_temp.quantile(.01)))
B = pd.DataFrame(out).T
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
print("mean(sub_temp - ew4) by ew4 bin"); print(B.to_string())
S = pd.DataFrame(slope).set_index("farm").round(3)
print(S.to_string())
# similarity: distance of binned profile to F13/F47 mean across overlapping bins
ref = B.loc[["F13", "F47"]].mean()
dist = ((B - ref) ** 2).mean(axis=1, skipna=True) ** .5
print("profile RMS distance to mean(F13,F47):"); print(dist.sort_values().round(2).to_string())
B.to_csv(env.LOCAL + "/eda_farm_2_bins.csv"); S.to_csv(env.LOCAL + "/eda_farm_2_slopes.csv")
