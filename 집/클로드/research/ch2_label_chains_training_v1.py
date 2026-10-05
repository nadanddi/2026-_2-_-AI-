# -*- coding: utf-8 -*-
"""CH2 (structure, fixed before running; 2026-10-06 집 클로드).  Build SOURCE CHAINS among TRAINING
(labelled) records using their labels - training-data preprocessing, allowed (6.309) - so that later an
evaluation record only has to be placed into a gap of known chains.
Link a -> b (same farm, both labelled, b follows a on the next calendar date in the same 동):
  cost = (|EC_b(0) - EC_a(23) - trend| / .02)^2           EC continuity (labels; trend = mean of the
                                                          two adjacent hourly EC changes, as deep_cal)
       + date_cost(a, b)                                  outdoor midnight continuity (deep_cal_8 form)
       + indoor_cost(a, b) / 4                            indoor + actuator continuity (deep_cal_10 form)
One-to-one assignment over all labelled records of a farm (Hungarian), with a 'no link' option of cost
C0 = 12 for both the outgoing and the incoming side (fixed; a link must beat it).
Report: links, EC jump distribution of links, share of links whose date cost is in the consecutive-date
range (<= 3), agreement with deep_cal_10 links, chains and their lengths, chain-start / chain-end counts
(gaps where unlabelled records, e.g. evaluation records, can sit), and 동 consistency (st_dong_assign)."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]; WT = {"out_temp": 1, "out_hum": 1, "out_wspd": .3, "out_rad": 1}
IN = {"in_temp": 1.0, "in_hum": 5.0, "in_co2": 40.0}; ACT = ["act_heating", "act_thermal", "act_circfan", "act_vent"]
C0 = 12.0
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])].copy()
Y["farm"], Y["day"], Y["hour"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int), Y.row_id.str[8:10].astype(int)
X = X.merge(Y[["row_id", "sub_ec"]], on="row_id")
P = X.pivot_table(index=["farm", "day"], columns="hour", values=W + list(IN) + ACT + ["sub_ec"])
SC = {(f, v): np.nanstd(X[X.farm == f].sort_values(["day", "hour"]).groupby("day")[v].diff()) for f in ("F13", "F47") for v in W}


def jump(f, a, b, v):
    A23, A22, B0, B1 = (P.loc[(f, a), (v, 23)], P.loc[(f, a), (v, 22)], P.loc[(f, b), (v, 0)], P.loc[(f, b), (v, 1)])
    return (B0 - A23) - .5 * ((A23 - A22) + (B1 - B0))


rows = []; links = []
for f in ("F13", "F47"):
    D = sorted(P.loc[f].index)
    n = len(D)
    M = np.full((n, n), 1e6)
    for i, a in enumerate(D):
        for j, b in enumerate(D):
            if a == b:
                continue
            ce = (jump(f, a, b, "sub_ec") / .02) ** 2
            cd = sum(WT[v] * ((jump(f, a, b, v) / SC[(f, v)]) ** 2) for v in W)
            ci = sum(((jump(f, a, b, v) / s) ** 2) for v, s in IN.items())
            ci += sum(abs(P.loc[(f, b), (v, 0)] - P.loc[(f, a), (v, 23)]) / 50 for v in ACT)
            c = ce + cd + ci / 4
            M[i, j] = c if np.isfinite(c) else 1e6
    big = np.full((2 * n, 2 * n), 1e6)
    big[:n, :n] = M
    big[:n, n:] = np.where(np.eye(n) == 1, C0, 1e6)      # a has no successor
    big[n:, :n] = np.where(np.eye(n) == 1, C0, 1e6)      # b has no predecessor
    big[n:, n:] = 0
    r_, c_ = linear_sum_assignment(big)
    for i, j in zip(r_, c_):
        if i < n and j < n:
            a, b = D[i], D[j]
            links.append(dict(farm=f, a=a, b=b, cost=M[i, j], ec_jump=P.loc[(f, b), ("sub_ec", 0)] - P.loc[(f, a), ("sub_ec", 23)],
                              date_cost=sum(WT[v] * ((jump(f, a, b, v) / SC[(f, v)]) ** 2) for v in W)))
L = pd.DataFrame(links)
L.to_csv(os.path.join(env.LOCAL, "ch2_label_links_v1.csv"), index=False)
print("links %d (F13 %d, F47 %d) among %d labelled records" % (len(L), (L.farm == "F13").sum(), (L.farm == "F47").sum(), len(P)))
print("EC jump |.|: median %.4f, share < .02 %.2f, < .05 %.2f, > .2 %.2f" % (L.ec_jump.abs().median(), (L.ec_jump.abs() < .02).mean(),
      (L.ec_jump.abs() < .05).mean(), (L.ec_jump.abs() > .2).mean()))
print("date cost: median %.2f, share <= 3 %.2f" % (L.date_cost.median(), (L.date_cost <= 3).mean()))
DL = pd.read_csv(os.path.join(env.LOCAL, "deep_cal_10_links.csv"))
dset = set(zip(DL.farm, DL.d_from, DL.d_to))
both = [(f, a, b) in dset for f, a, b in zip(L.farm, L.a, L.b)]
print("agreement with deep_cal_10 links: %.2f of CH2 links are deep_cal links" % np.mean(both))
dg = pd.read_csv(os.path.join(env.LOCAL, "st_dong_assign_v1.csv")).set_index(["farm", "day"]).dong
print("same 동 (st_dong_assign) along links: %.2f" % np.mean([dg.get((f, a)) == dg.get((f, b)) for f, a, b in zip(L.farm, L.a, L.b)]))
print("record-day gap b - a distribution:", L.assign(g=L.b - L.a).g.value_counts().head(10).to_dict())
# chains
for f in ("F13", "F47"):
    Lf = L[L.farm == f]; succ = dict(zip(Lf.a, Lf.b)); pred = set(Lf.b)
    D = sorted(P.loc[f].index); starts = [d for d in D if d not in pred]
    lens = []
    for s in starts:
        k, c = 1, s
        while c in succ:
            c = succ[c]; k += 1
        lens.append(k)
    lens = sorted(lens, reverse=True)
    print("%s: chains %d, longest %s, singletons %d, chain ends (gaps for unlabelled records) %d" % (
        f, len(lens), lens[:8], sum(1 for x in lens if x == 1), len(starts)))
