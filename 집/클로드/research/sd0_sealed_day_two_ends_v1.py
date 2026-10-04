# -*- coding: utf-8 -*-
"""SD0 (diagnostic, fixed before running; 2026-10-04 집 클로드).
On SEALED days (act_vent_zero >= .8) the model compresses: true 0.5-0.9 days are
over-predicted, true 1.5-2.7 days under-predicted (ND0 / HC1).  Compare the two ends
side by side.  DIAG10 R3S seed-mean day error eday = pred mean - label mean.
Groups: OVER (eday >= +.15), UNDER (eday <= -.15), OK (between).
Inputs (no labels): hc0 day features (daily mean / night / day / zero fraction of
every input; structure flags role, 동, pass, record day) and op0 same-동 previous
3/7-date operation features; plus midnight (hour 0) inputs.
Primary statistic: Spearman(feature, eday) over all sealed days, per farm too.
Clue (fixed): the best feature's family-wise permutation p (eday permuted within
farm, 10000, max |Spearman| over all features) < .05 AND the same sign in both
farms.  Also report the OVER vs UNDER medians of the top features."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); O = O[O.validator == "DIAG10"].copy()
O["p"] = O[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
DD = O.groupby(["farm", "day"]).agg(y=("sub_ec", "mean"), pm=("p", "mean")).reset_index(); DD["eday"] = DD.pm - DD.y
H = pd.read_csv(os.path.join(env.LOCAL, "hc0_day_features.csv"))
P = pd.read_csv(os.path.join(env.LOCAL, "op0_features_v1.csv"))
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy(); X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
M0 = X[X.hour == 0].drop(columns=["row_id", "hour"]).add_prefix("h0_").rename(columns={"h0_farm": "farm", "h0_day": "day"})
D = DD.merge(H, on=["farm", "day"]).merge(P[["farm", "day"] + [c for c in P.columns if c.endswith(("_3", "_7")) or c == "seal_streak"]], on=["farm", "day"]).merge(M0, on=["farm", "day"], how="left")
S = D[D.act_vent_zero >= .8].copy()
S["grp"] = np.where(S.eday >= .15, "OVER", np.where(S.eday <= -.15, "UNDER", "OK"))
drop = {"farm", "day", "y", "pm", "eday", "ec", "hi", "role", "dong", "grp"}
FE = [c for c in S.columns if c not in drop and pd.api.types.is_numeric_dtype(S[c]) and S[c].nunique() > 2 or c in ("is_B", "is_second", "late")]
FE = [c for c in FE if c not in drop and S[c].notna().sum() >= 20 and S[c].nunique() > 1]
print("sealed days %d (F13 %d, F47 %d); groups %s" % (len(S), (S.farm == "F13").sum(), (S.farm == "F47").sum(), S.grp.value_counts().to_dict()))
print("label mean by group:", S.groupby("grp").y.mean().round(3).to_dict(), " pred:", S.groupby("grp").pm.mean().round(3).to_dict())
print("pass-2 share by group:", S.groupby("grp").late.mean().round(2).to_dict(), " 동B share:", S.groupby("grp").is_B.mean().round(2).to_dict(),
      " second share:", S.groupby("grp").is_second.mean().round(2).to_dict())

res = []
for c in FE:
    z = S[[c, "eday", "farm"]].dropna()
    rho = spearmanr(z[c], z.eday).correlation
    f = [spearmanr(z[z.farm == fm][c], z[z.farm == fm].eday).correlation for fm in ("F13", "F47")]
    res.append((c, rho, f[0], f[1], len(z)))
R = pd.DataFrame(res, columns=["feat", "rho", "F13", "F47", "n"]).dropna(subset=["rho"])
R["same"] = np.sign(R.F13) == np.sign(R.F47)
R = R.reindex(R.rho.abs().sort_values(ascending=False).index)
print("\nSpearman(feature, day error) on sealed days, top 20 (positive = more OVER-predicted):")
print(R.head(20).round(2).to_string(index=False))

Z = S[FE].rank(); fv = S.farm.values; ev = S.eday.values
def maxabs(e):
    er = pd.Series(e, index=S.index).rank()
    return np.nanmax(np.abs([Z[c].corr(er) for c in R.feat]))
obs = maxabs(ev); rng = np.random.default_rng(20261004); cnt = 0; NP = 10000
for _ in range(NP):
    e = ev.copy()
    for fm in ("F13", "F47"):
        m = fv == fm; e[m] = rng.permutation(e[m])
    cnt += maxabs(e) >= obs
pfw = (cnt + 1) / (NP + 1)
b = R.iloc[0]
print("\nbest %s rho %+.2f (F13 %+.2f, F47 %+.2f), family-wise p %.4f over %d features" % (b.feat, b.rho, b.F13, b.F47, pfw, len(R)))
print("\nOVER vs UNDER medians of top 10 features:")
print(S.groupby("grp")[list(R.feat.head(10))].median().T.round(3).to_string())
S.to_csv(os.path.join(env.LOCAL, "sd0_sealed_days_v1.csv"), index=False)
print("\nSD0 clue:", bool(pfw < .05 and b.same))
