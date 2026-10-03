# -*- coding: utf-8 -*-
"""X3 (diagnostic, fixed before running; 2026-10-04 집 클로드).
User question: can information we do NOT have (irrigation, drain, substrate weight,
substrate moisture) be inferred from what we DO have (inside temp/hum/CO2/radiation),
and does that inferred information explain substrate/drain EC?
Link 1 (missing <- available): cross-farm (GroupKFold by farm) ExtraTrees R2 / AUC,
  day level (within-farm deviations) and hour level (irrigation event yes/no).
Link 2 (missing -> EC): within-farm Spearman of the TRUE missing variable with EC
  deviation (same day / next day), and cross-farm R2 of EC deviation from the
  cross-fitted PROXY of the missing variable.
Reading: a usable proxy chain needs link-1 R2 >= .2 (or AUC >= .75) AND link-2
  cross-farm R2 >= .05 through the proxy."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import env  # noqa: F401
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
OUT = r"C:\Users\aozks\AppData\Local\Temp\claude\C--work-farmai\2ea435a4-3dc2-434b-ad01-aff27f988a60\scratchpad"
def cv(Z, feats, y, clf=False):
    pr = np.zeros(len(Z))
    for a, b in GroupKFold(5).split(Z, groups=Z.farm):
        M = (ExtraTreesClassifier if clf else ExtraTreesRegressor)(300, min_samples_leaf=10, random_state=0, n_jobs=4)
        M.fit(Z.iloc[a][feats], Z.iloc[a][y])
        pr[b] = M.predict_proba(Z.iloc[b][feats])[:, 1] if clf else M.predict(Z.iloc[b][feats])
    if clf:
        return roc_auc_score(Z[y], pr), pr
    return 1 - np.sum((Z[y] - pr) ** 2) / np.sum((Z[y] - Z[y].mean()) ** 2), pr
# ---------- 이레: irrigation / drain / weight ----------
I = pd.read_csv(os.path.join(OUT, "x0_ire_hourly.csv"), parse_dates=["t"]).rename(columns={
    "(양액)공급량": "irr", "(양액)배액EC": "dec", "(양액)배액량": "drn", "내부습도": "hi", "내부온도": "ti", "배지무게": "wt"})
I.loc[I.dec <= 0, "dec"] = np.nan
I["hour"] = I.t.dt.hour; I["d"] = I.t.dt.date
I = I.sort_values(["farm", "t"])
g = I.groupby(["farm", "d"])
for c in ("ti", "hi"):
    I[c + "_l1"] = I.groupby("farm")[c].shift(1); I[c + "_d1"] = I[c] - I[c + "_l1"]
    I[c + "_dm"] = I[c] - g[c].transform("mean")
I["irr_on"] = (I.irr > 0).astype(int)
H = I.dropna(subset=["ti", "hi", "ti_l1", "hi_l1"])
auc, _ = cv(H, ["hour", "ti", "hi", "ti_d1", "hi_d1", "ti_dm", "hi_dm"], "irr_on", clf=True)
print("[이레] LINK1 hour level: irrigation event (rate %.2f) from inside temp/hum -> cross-farm AUC %.3f" % (H.irr_on.mean(), auc))
D = I.groupby(["farm", "d"]).agg(irr=("irr", "sum"), drn=("drn", "sum"), wt=("wt", "mean"), dec=("dec", "mean"), ti=("ti", "mean"),
                                 tmax=("ti", "max"), tmin=("ti", "min"), hi=("hi", "mean"), hmin=("hi", "min"), n=("dec", "count")).reset_index()
D = D[D.n >= 6].copy()
for c in ("irr", "drn", "wt", "dec", "ti", "tmax", "tmin", "hi", "hmin"):
    D[c + "_dv"] = D[c] - D.groupby("farm")[c].transform("mean")
F = ["ti_dv", "tmax_dv", "tmin_dv", "hi_dv", "hmin_dv"]
for y in ("irr", "drn", "wt"):
    r2, pr = cv(D, F, y + "_dv"); D[y + "_px"] = pr
    s_true = spearmanr(D[y + "_dv"], D.dec_dv).correlation
    s_next = spearmanr(D[y + "_dv"], D.groupby("farm").dec_dv.shift(-1), nan_policy="omit").correlation
    r2b, _ = cv(D.assign(px=D[y + "_px"]), ["px"], "dec_dv")
    print("[이레] %-3s LINK1 day R2 from climate %.3f | LINK2 true %s vs EC dev: same day %+.2f next day %+.2f | EC dev from proxy cross-farm R2 %.3f" % (
        y, r2, y, s_true, s_next, r2b))
# ---------- 골든: substrate moisture ----------
G = pd.read_csv(os.path.join(OUT, "x0_golden_hourly.csv"), parse_dates=["t"]).rename(columns={
    "토양EC": "ec", "내부습도": "hi", "내부온도": "ti", "내부CO2": "ci", "지습": "hl", "내부일사량": "ir"})
G["d"] = G.t.dt.date
GD = G.groupby(["farm", "d"]).agg(ec=("ec", "mean"), hl=("hl", "mean"), ti=("ti", "mean"), tmax=("ti", "max"), tmin=("ti", "min"),
                                  hi=("hi", "mean"), hmin=("hi", "min"), ci=("ci", "mean"), ir=("ir", "mean")).reset_index()
for c in ("ec", "hl", "ti", "tmax", "tmin", "hi", "hmin", "ci", "ir"):
    GD[c + "_dv"] = GD[c] - GD.groupby("farm")[c].transform("mean")
r2, pr = cv(GD.dropna(), ["ti_dv", "tmax_dv", "tmin_dv", "hi_dv", "hmin_dv", "ci_dv", "ir_dv"], "hl_dv")
GG = GD.dropna().assign(px=pr)
r2b, _ = cv(GG, ["px"], "ec_dv")
print("[골든] hl  LINK1 day R2 from climate %.3f | LINK2 true moisture vs EC dev same day %+.2f | EC dev from proxy cross-farm R2 %.3f" % (
    r2, spearmanr(GG.hl_dv, GG.ec_dv).correlation, r2b))
