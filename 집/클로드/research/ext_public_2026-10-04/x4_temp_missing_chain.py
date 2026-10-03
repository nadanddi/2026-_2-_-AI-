# -*- coding: utf-8 -*-
"""X4 (diagnostic, fixed before running; 2026-10-04 집 클로드).  Temperature version
of X3 with public 이레 strawberry data (substrate temp + irrigation/drain/weight).
Our temperature error is mostly day level ('separation days', 6b.32).
(1) Link 2: within-farm day deviation of gap = substrate - inside temp vs TRUE
    irrigation sum, drain sum, substrate weight (same day); cross-farm R2 of gap
    deviation from climate-only vs climate + irrigation info (does the missing info
    add explanatory power?).
(2) Link 1: missing info from climate (from X3: irrigation/drain R2 < 0, weight .19).
(3) Hour level event study: change of substrate temp over the next hour minus change
    of inside temp, hours with irrigation vs without, by hour of day (cold-water
    cooling?).
Reading: a usable chain needs (1) climate+irrigation cross-farm R2 exceeding
climate-only by >= .05."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import env  # noqa: F401
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
OUT = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\2ea435a4-3dc2-434b-ad01-aff27f988a60\scratchpad"
I = pd.read_csv(os.path.join(OUT, "x0_ire_hourly.csv"), parse_dates=["t"]).rename(columns={
    "(양액)공급량": "irr", "(양액)배액량": "drn", "내부습도": "hi", "내부온도": "ti", "배지무게": "wt", "배지온도": "st"})
I = I.dropna(subset=["ti", "st"]).sort_values(["farm", "t"])
I["gap"] = I.st - I.ti; I["hour"] = I.t.dt.hour; I["d"] = I.t.dt.date
D = I.groupby(["farm", "d"]).agg(gap=("gap", "mean"), irr=("irr", "sum"), drn=("drn", "sum"), wt=("wt", "mean"), ti=("ti", "mean"),
                                 tmax=("ti", "max"), tmin=("ti", "min"), hi=("hi", "mean"), hmin=("hi", "min"), n=("st", "count")).reset_index()
D = D[D.n >= 20].copy()
for c in ("gap", "irr", "drn", "wt", "ti", "tmax", "tmin", "hi", "hmin"):
    D[c + "_dv"] = D[c] - D.groupby("farm")[c].transform("mean")
print("[이레] farm-days %d | day gap mean q05 %.2f med %.2f q95 %.2f | within-farm share %.2f | lag1 %.2f" % (
    len(D), D.gap.quantile(.05), D.gap.median(), D.gap.quantile(.95), D.gap_dv.var() / D.gap.var(),
    D.gap_dv.corr(D.groupby("farm").gap_dv.shift(1))))
for c in ("irr", "drn", "wt", "ti", "tmin", "hi"):
    print("  Spearman(gap dev, %s dev) %+.2f" % (c, spearmanr(D.gap_dv, D[c + "_dv"], nan_policy="omit").correlation))
def cv(Z, feats, y):
    Z = Z.dropna(subset=feats + [y]); pr = np.zeros(len(Z))
    for a, b in GroupKFold(5).split(Z, groups=Z.farm):
        pr[b] = ExtraTreesRegressor(300, min_samples_leaf=10, random_state=0, n_jobs=4).fit(Z.iloc[a][feats], Z.iloc[a][y]).predict(Z.iloc[b][feats])
    return 1 - np.sum((Z[y] - pr) ** 2) / np.sum((Z[y] - Z[y].mean()) ** 2)
C = ["ti_dv", "tmax_dv", "tmin_dv", "hi_dv", "hmin_dv"]
a = cv(D, C, "gap_dv"); b = cv(D, C + ["irr_dv", "drn_dv", "wt_dv"], "gap_dv")
print("  cross-farm R2 of day gap deviation: climate-only %.3f | climate + irrigation/drain/weight %.3f | gain %+.3f" % (a, b, b - a))
I["dst"] = I.groupby("farm").st.shift(-1) - I.st
I["dti"] = I.groupby("farm").ti.shift(-1) - I.ti
I["rel"] = I.dst - I.dti
I["irr_on"] = I.irr > 0
E = I.dropna(subset=["rel"])
print("\n(3) next-hour substrate change minus air change (C), irrigation hour vs not:")
tab = E.groupby([pd.cut(E.hour, [-1, 5, 9, 13, 17, 23]), "irr_on"], observed=True).rel.agg(["mean", "count"]).unstack()
print(tab.round(3).to_string())
fe = E.assign(rel_dm=E.rel - E.groupby(["farm", "hour"]).rel.transform("mean"))
print("  within farm x hour: mean rel on irrigation hours %+.3f vs others %+.3f" % (fe[fe.irr_on].rel_dm.mean(), fe[~fe.irr_on].rel_dm.mean()))
print("\nX4 usable chain:", (b - a) >= .05)
