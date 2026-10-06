# -*- coding: utf-8 -*-
"""AC1 (exploration, 2026-10-07 집 클로드).  Are the actuator signals of high-EC days (label day mean >= 1, 31 of
360 labelled DIAG10 days) all in the same direction, and do normal days show the same actuator pattern?
Day-level actuator summaries (hc0_day_features: mean / night / day / zero-share of 7 actuators).
Per variable: median high vs normal, AUC(high vs normal), share of high days on the high-day side of the normal
median ('consistency'), share of normal days on that same side.
Profile view: 24 h x 7 actuator vector (z per farm); for each day, share of high days among its 10 nearest
labelled days of the same farm; how many NORMAL days look like high days (>= 5 of 10 neighbours high)."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
H = pd.read_csv(os.path.join(env.LOCAL, "hc0_day_features.csv"))
ACT = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
cols = [a + s for a in ACT for s in ("_mean", "_night", "_day", "_zero")]
H["hi"] = (H.ec >= 1).astype(int)
rows = []
for c in cols:
    x = H[c]; auc = roc_auc_score(H.hi, x.fillna(x.median()))
    med_n = x[H.hi == 0].median(); side = 1 if x[H.hi == 1].median() >= med_n else -1
    cons_h = ((x[H.hi == 1] - med_n) * side > 0).mean(); cons_n = ((x[H.hi == 0] - med_n) * side > 0).mean()
    rows.append((c, x[H.hi == 1].median(), med_n, auc, cons_h, cons_n))
T = pd.DataFrame(rows, columns=["var", "median_high", "median_normal", "AUC", "high_on_side", "normal_on_side"])
T["sep"] = (T.AUC - .5).abs()
print("high days %d, normal days %d" % (H.hi.sum(), (H.hi == 0).sum()))
print(T.sort_values("sep", ascending=False).head(14).drop(columns="sep").round(2).to_string(index=False))
# combined rule: how consistent across the strongest variables
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id"] + ACT)
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
out = []
for f in ("F13", "F47"):
    Xf = X[X.farm == f].copy()
    for a in ACT:
        Xf[a] = (Xf[a] - Xf[a].mean()) / (Xf[a].std() or 1)
    P = Xf.pivot_table(index="day", columns="hour", values=ACT)
    Hf = H[H.farm == f].set_index("day"); days = [d for d in Hf.index if d in P.index]
    M = P.loc[days].values
    for i, d in enumerate(days):
        dist = np.sqrt(np.nanmean((M - M[i]) ** 2, axis=1)); dist[i] = np.inf
        nb = np.argsort(dist)[:10]
        out.append(dict(farm=f, day=d, hi=Hf.hi[d], ec=Hf.ec[d], sealed=Hf.act_vent_zero[d], share_hi_nb=Hf.hi.values[nb].mean()))
Q = pd.DataFrame(out)
print("\nactuator-profile neighbours (10 nearest days by 24h x 7 actuators):")
print("  high days: mean share of high neighbours %.2f; high days with >= 5/10 high neighbours: %d of %d" % (
    Q[Q.hi == 1].share_hi_nb.mean(), (Q[Q.hi == 1].share_hi_nb >= .5).sum(), Q.hi.sum()))
print("  normal days: mean share %.2f; normal days with >= 5/10 high neighbours ('look like high'): %d of %d (their EC mean %.2f, sealed %.2f)" % (
    Q[Q.hi == 0].share_hi_nb.mean(), (Q[Q.hi == 0].share_hi_nb >= .5).sum(), (Q.hi == 0).sum(),
    Q[(Q.hi == 0) & (Q.share_hi_nb >= .5)].ec.mean(), Q[(Q.hi == 0) & (Q.share_hi_nb >= .5)].sealed.mean()))
print("  normal days with >= 3/10 high neighbours: %d (EC mean %.2f)" % (((Q.hi == 0) & (Q.share_hi_nb >= .3)).sum(), Q[(Q.hi == 0) & (Q.share_hi_nb >= .3)].ec.mean()))
print("  high days with 0 high neighbours (actuator pattern unlike other high days): %s" % Q[(Q.hi == 1) & (Q.share_hi_nb == 0)][["farm", "day", "ec"]].values.tolist())
Q.to_csv(os.path.join(env.LOCAL, "ac1_days_v1.csv"), index=False)
