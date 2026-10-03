# -*- coding: utf-8 -*-
"""X1 (diagnostic; 2026-10-04 집 클로드).  Public strawberry data:
(a) 이레: what moves the day-level DRAIN EC within a farm?  irrigation (공급량),
    drain (배액량), drain ratio, substrate weight, substrate temp, inside temp/hum,
    same day and previous day; lag-1 persistence.
(b) both datasets: can CLIMATE-ONLY day features (the only overlap with the
    competition inputs: inside temp / humidity [+CO2 golden]) predict the within-farm
    day-level EC deviation in UNSEEN farms?  GroupKFold by farm, ExtraTrees, R2.
Reading: transfer is worth a model test only if (b) cross-farm R2 >= .10."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import env  # noqa: F401
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
OUT = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\2ea435a4-3dc2-434b-ad01-aff27f988a60\scratchpad"
I = pd.read_csv(os.path.join(OUT, "x0_ire_hourly.csv"), parse_dates=["t"])
I = I.rename(columns={"(양액)공급량": "irr", "(양액)배액EC": "dec", "(양액)배액PH": "dph", "(양액)배액량": "drn", "내부습도": "hi", "내부온도": "ti", "배지무게": "wt", "배지온도": "st"})
I.loc[I.dec <= 0, "dec"] = np.nan
I["d"] = I.t.dt.date
D = I.groupby(["farm", "d"]).agg(dec=("dec", "mean"), irr=("irr", "sum"), drn=("drn", "sum"), wt=("wt", "mean"), st=("st", "mean"),
                                 ti=("ti", "mean"), tmax=("ti", "max"), tmin=("ti", "min"), hi=("hi", "mean"), n=("dec", "count")).reset_index()
D = D[D.n >= 6]
D["ratio"] = D.drn / D.irr.replace(0, np.nan)
cols = ["irr", "drn", "ratio", "wt", "st", "ti", "tmax", "tmin", "hi"]
for c in ["dec"] + cols:
    D[c + "_dv"] = D[c] - D.groupby("farm")[c].transform("mean")
D = D.sort_values(["farm", "d"])
print("[이레] farm-days %d farms %d | drain EC day mean q05 %.2f med %.2f q95 %.2f | within-farm share of variance %.2f | lag1 %.2f" % (
    len(D), D.farm.nunique(), D.dec.quantile(.05), D.dec.median(), D.dec.quantile(.95), D.dec_dv.var() / D.dec.var(),
    D.dec_dv.corr(D.groupby("farm").dec_dv.shift(1))))
print("  Spearman of within-farm day deviation of drain EC with:")
for c in cols:
    same = spearmanr(D.dec_dv, D[c + "_dv"], nan_policy="omit").correlation
    prev = spearmanr(D.dec_dv, D.groupby("farm")[c + "_dv"].shift(1), nan_policy="omit").correlation
    print("   %-6s same day %+.2f | previous day %+.2f" % (c, same, prev))
def xfarm(Dd, feats, y, name):
    Z = Dd.dropna(subset=feats + [y])
    pr = np.zeros(len(Z))
    for a, b in GroupKFold(5).split(Z, groups=Z.farm):
        m = ExtraTreesRegressor(300, min_samples_leaf=5, random_state=0, n_jobs=4).fit(Z.iloc[a][feats], Z.iloc[a][y])
        pr[b] = m.predict(Z.iloc[b][feats])
    r2 = 1 - np.sum((Z[y] - pr) ** 2) / np.sum((Z[y] - Z[y].mean()) ** 2)
    print("  cross-farm R2 %-34s %.3f (n %d)" % (name, r2, len(Z)))
    return r2
print("\n(b) transfer check (within-farm day deviation target)")
r_a = xfarm(D, ["ti_dv", "tmax_dv", "tmin_dv", "hi_dv"], "dec_dv", "이레 climate-only")
r_b = xfarm(D, ["ti_dv", "tmax_dv", "tmin_dv", "hi_dv", "irr_dv", "drn_dv", "wt_dv"], "dec_dv", "이레 climate+irrigation (not in comp)")
G = pd.read_csv(os.path.join(OUT, "x0_golden_hourly.csv"), parse_dates=["t"])
G = G.rename(columns={"토양EC": "ec", "내부습도": "hi", "내부온도": "ti", "내부CO2": "ci", "지습": "hl"})
G["d"] = G.t.dt.date
GD = G.groupby(["farm", "d"]).agg(ec=("ec", "mean"), ti=("ti", "mean"), tmax=("ti", "max"), tmin=("ti", "min"), hi=("hi", "mean"), ci=("ci", "mean"), hl=("hl", "mean")).reset_index()
for c in ["ec", "ti", "tmax", "tmin", "hi", "ci", "hl"]:
    GD[c + "_dv"] = GD[c] - GD.groupby("farm")[c].transform("mean")
r_c = xfarm(GD, ["ti_dv", "tmax_dv", "tmin_dv", "hi_dv", "ci_dv"], "ec_dv", "골든 climate-only")
r_d = xfarm(GD, ["ti_dv", "tmax_dv", "tmin_dv", "hi_dv", "ci_dv", "hl_dv"], "ec_dv", "골든 climate+moisture (not in comp)")
print("\nX1 transfer worth a model test (climate-only R2 >= .10):", max(r_a, r_c) >= .10)
