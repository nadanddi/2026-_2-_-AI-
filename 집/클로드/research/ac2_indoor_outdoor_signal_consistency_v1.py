# -*- coding: utf-8 -*-
"""AC2 (exploration, 2026-10-07 집 클로드; mirror of AC1 for indoor climate and outdoor weather).
Are the indoor / outdoor signals of NORMAL days (label day mean < 1, 329 days) in one consistent direction, and do
high-EC days (31) share that direction?  Day summaries (hc0_day_features) of in_temp / in_hum / in_co2 / out_temp /
out_hum / out_rad / out_wspd (mean, night, day) + in_temp_range, in_out_diff.  Per variable: medians, AUC(high vs
normal), share of NORMAL days on the normal side of the HIGH-day median ('normal consistency'), share of HIGH days on
that same side.  Profile view: 24 h x (3 indoor + 4 outdoor) vector (z per farm); share of normal days among the 10
nearest labelled days; how many HIGH days look normal (>= 9 of 10 neighbours normal)."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
H = pd.read_csv(os.path.join(env.LOCAL, "hc0_day_features.csv")); H["hi"] = (H.ec >= 1).astype(int)
V = ["in_temp", "in_hum", "in_co2", "out_temp", "out_hum", "out_rad", "out_wspd"]
cols = [v + s for v in V for s in ("_mean", "_night", "_day")] + ["in_temp_range", "in_out_diff"]
rows = []
for c in cols:
    x = H[c]; auc = roc_auc_score(H.hi, x.fillna(x.median()))
    med_h = x[H.hi == 1].median(); side = 1 if x[H.hi == 0].median() >= med_h else -1
    rows.append((c, med_h, x[H.hi == 0].median(), auc, ((x[H.hi == 0] - med_h) * side > 0).mean(), ((x[H.hi == 1] - med_h) * side > 0).mean()))
T = pd.DataFrame(rows, columns=["var", "median_high", "median_normal", "AUC", "normal_on_side", "high_on_side"])
T["sep"] = (T.AUC - .5).abs()
print(T.sort_values("sep", ascending=False).drop(columns="sep").round(2).to_string(index=False))
S = H[H.act_vent_zero >= .8]
print("\nwithin SEALED days only (n %d, high %d): AUC of the strongest indoor/outdoor variables:" % (len(S), S.hi.sum()),
      {c: round(roc_auc_score(S.hi, S[c].fillna(S[c].median())), 2) for c in ["in_temp_mean", "in_hum_mean", "in_co2_mean", "out_temp_mean", "out_rad_mean", "in_out_diff"]})
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id"] + V)
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
out = []
for f in ("F13", "F47"):
    Xf = X[X.farm == f].copy()
    for v in V:
        Xf[v] = (Xf[v] - Xf[v].mean()) / (Xf[v].std() or 1)
    P = Xf.pivot_table(index="day", columns="hour", values=V)
    Hf = H[H.farm == f].set_index("day"); days = [d for d in Hf.index if d in P.index]; M = P.loc[days].values
    for i, d in enumerate(days):
        dist = np.sqrt(np.nanmean((M - M[i]) ** 2, axis=1)); dist[i] = np.inf
        nb = np.argsort(dist)[:10]
        out.append(dict(farm=f, day=d, hi=Hf.hi[d], ec=Hf.ec[d], sealed=Hf.act_vent_zero[d], share_norm=1 - Hf.hi.values[nb].mean()))
Q = pd.DataFrame(out)
print("\nindoor+outdoor profile neighbours (10 nearest):")
print("  normal days: mean share of normal neighbours %.2f; normal days with all 10 normal %d of %d" % (
    Q[Q.hi == 0].share_norm.mean(), (Q[Q.hi == 0].share_norm == 1).sum(), (Q.hi == 0).sum()))
print("  high days: mean share of normal neighbours %.2f; high days with >= 9/10 normal neighbours ('look normal'): %d of %d (EC mean %.2f); all 10 normal: %d" % (
    Q[Q.hi == 1].share_norm.mean(), (Q[Q.hi == 1].share_norm >= .9).sum(), Q.hi.sum(), Q[(Q.hi == 1) & (Q.share_norm >= .9)].ec.mean(), (Q[Q.hi == 1].share_norm == 1).sum()))
Q.to_csv(os.path.join(env.LOCAL, "ac2_days_v1.csv"), index=False)
