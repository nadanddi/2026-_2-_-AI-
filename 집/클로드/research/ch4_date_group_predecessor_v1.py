# -*- coding: utf-8 -*-
"""CH4 (diagnostic, fixed before running; 2026-10-06 집 클로드).  CH3 failed because the outdoor
date cost left ~10 predecessor candidates.  CH4 narrows the candidates to ONE date:
  date groups: training (labelled) records grouped by identical 24 h outdoor weather (z-RMSE <= .05
               between consecutive members in record order; two 동 of one date share the weather);
  predecessor date of query q: the date group G minimising the outdoor midnight continuity cost from the
               group's hours 22-23 (group mean) to q's hours 0-1 (deep_cal_8 form); q itself and the
               query's own date group excluded;
  source within G (1-2 records): (A) model rule - the record whose hour-23 EC is nearest the model's
               mean prediction for q (day level, upper-bound flavour: uses the full-day model mean);
               (B) indoor rule - lowest indoor + actuator continuity cost;
  level: EC_p(23) + (m_h - m_0) (label level, model's within-day change).
Held-out = pass-2 labelled days of DIAG10 folds (reference = labelled records outside the fold).
Reported: chosen date group size; share of days where the TRUE predecessor (any record of G whose
EC_23 is within .05 of the true EC_q(0)) is inside G; share correct for rules A / B; row RMSE vs model
(R3S seed mean), normal / high; guarded A (only if the two candidates' EC_23 differ by >= .3 or G has
one record, and |EC_p23 - model mean| <= .3).
Clue (fixed): date group contains the true predecessor on >= 70 % of days AND rule A or guarded A beats
the model on pass-2 rows and normal days."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]; WT = {"out_temp": 1, "out_hum": 1, "out_wspd": .3, "out_rad": 1}
IN = {"in_temp": 1.0, "in_hum": 5.0, "in_co2": 40.0}; ACT = ["act_heating", "act_thermal", "act_circfan", "act_vent"]
X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")])
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
P = X.pivot_table(index=["farm", "day"], columns="hour", values=W + list(IN) + ACT)
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])].copy()
Y["farm"], Y["day"], Y["hour"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int), Y.row_id.str[8:10].astype(int)
E = Y.set_index(["farm", "day", "hour"]).sub_ec; labset = set(zip(Y.farm, Y.day))
SC = {(f, v): np.nanstd(X[(X.farm == f) & X.row_id.isin(set(Y.row_id))].sort_values(["day", "hour"]).groupby("day")[v].diff()) for f in ("F13", "F47") for v in W}
ZW = {}
for f in ("F13", "F47"):
    Z = P.loc[f][W].copy()
    for v in W:
        Z[v] = (Z[v] - np.nanmean(Z[v].values)) / np.nanstd(Z[v].values)
    ZW[f] = Z


def twin(f, a, b):
    return np.sqrt(np.nanmean((ZW[f].loc[a].values - ZW[f].loc[b].values) ** 2)) <= .05


def groups(f, ref):
    days = sorted(d for ff, d in ref if ff == f); G = []
    for d in days:
        for g in G:
            if twin(f, g[0], d):
                g.append(d); break
        else:
            G.append([d])
    return G


def gcost(f, g, q):
    c = 0.0
    for v in W:
        A23 = np.nanmean([P.loc[(f, a), (v, 23)] for a in g]); A22 = np.nanmean([P.loc[(f, a), (v, 22)] for a in g])
        B0, B1 = P.loc[(f, q), (v, 0)], P.loc[(f, q), (v, 1)]
        e = ((((B0 - A23) - .5 * ((A23 - A22) + (B1 - B0))) / SC[(f, v)]) ** 2)
        c += WT[v] * (50 if np.isnan(e) else e)
    return c


def icost(f, a, q):
    c = 0.0
    for v, s in IN.items():
        A23, A22, B0, B1 = P.loc[(f, a), (v, 23)], P.loc[(f, a), (v, 22)], P.loc[(f, q), (v, 0)], P.loc[(f, q), (v, 1)]
        e = (((B0 - A23) - .5 * ((A23 - A22) + (B1 - B0))) / s) ** 2; c += 0 if np.isnan(e) else e
    return c + sum(0 if np.isnan(P.loc[(f, q), (v, 0)] - P.loc[(f, a), (v, 23)]) else abs(P.loc[(f, q), (v, 0)] - P.loc[(f, a), (v, 23)]) / 50 for v in ACT)


O = pd.read_csv(os.path.join(env.LOCAL, "hk0_rows_v1.csv"))
Fd = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"), usecols=["farm", "day", "validator", "validation_fold"])
Fd = Fd[Fd.validator == "DIAG10"].drop_duplicates(["farm", "day"])
O = O[O.day >= 179].sort_values(["farm", "day", "hour"]).copy()
O["m0"] = O.groupby(["farm", "day"]).p.transform("first"); O["mm"] = O.groupby(["farm", "day"]).p.transform("mean")
O["dm"] = O.groupby(["farm", "day"]).sub_ec.transform("mean")
res = []
for k in sorted(O.validation_fold.unique()):
    vd = set(zip(Fd[Fd.validation_fold == k].farm, Fd[Fd.validation_fold == k].day))
    ref = {x for x in labset if x not in vd}
    for f in ("F13", "F47"):
        G = groups(f, ref)
        for d in sorted({dd for ff, dd in vd if ff == f and dd >= 179}):
            own = [g for g in G if twin(f, g[0], d)]
            cand = [g for g in G if g not in own]
            costs = [gcost(f, g, d) for g in cand]
            g = cand[int(np.argmin(costs))]
            y0 = E[(f, d, 0)]; mm = O[(O.farm == f) & (O.day == d)].mm.iloc[0]
            e23 = {a: E[(f, a, 23)] for a in g}
            pA = min(g, key=lambda a: abs(e23[a] - mm)); pB = min(g, key=lambda a: icost(f, a, d))
            spread = max(e23.values()) - min(e23.values())
            guard = (len(g) == 1 or spread >= .3) and abs(e23[pA] - mm) <= .3
            res.append(dict(farm=f, day=d, gsize=len(g), true_in=any(abs(v - y0) <= .05 for v in e23.values()),
                            A=e23[pA], B=e23[pB], gA=e23[pA] if guard else np.nan, y0=y0, cost=min(costs),
                            second=sorted(costs)[1] if len(costs) > 1 else np.nan))
Rz = pd.DataFrame(res)
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
print("pass-2 held-out days %d: chosen date-group size %s; TRUE predecessor inside the chosen group: %.0f%%" % (
    len(Rz), Rz.gsize.value_counts().to_dict(), 100 * Rz.true_in.mean()))
for t in ("A", "B", "gA"):
    m = Rz[t].notna(); print("  rule %-2s used %d days, level within .05 of true 0h EC %.0f%%" % (t, m.sum(), 100 * ((Rz[t] - Rz.y0).abs() <= .05)[m].mean()))
O = O.merge(Rz[["farm", "day", "A", "B", "gA"]], on=["farm", "day"])
for t in ("A", "B", "gA"):
    O["pred_" + t] = np.where(O[t].notna(), O[t] + (O.p - O.m0), O.p)
for nm, m in (("pass-2", O.dm.notna()), ("normal", O.dm < 1), ("high", O.dm >= 1)):
    g = O[m]
    print("  %-7s model %.4f | A %.4f | B %.4f | guarded A %.4f" % (nm, r(g.p - g.sub_ec), r(g.pred_A - g.sub_ec), r(g.pred_B - g.sub_ec), r(g.pred_gA - g.sub_ec)))
Rz.to_csv(os.path.join(env.LOCAL, "ch4_days_v1.csv"), index=False)
clue = Rz.true_in.mean() >= .7 and any(r(O["pred_" + t] - O.sub_ec) < r(O.p - O.sub_ec) and r((O["pred_" + t] - O.sub_ec)[O.dm < 1]) < r((O.p - O.sub_ec)[O.dm < 1]) for t in ("A", "gA"))
print("CH4 clue:", clue)
