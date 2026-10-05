# -*- coding: utf-8 -*-
"""MC2 (diagnostic, fixed before running; 2026-10-06 집 클로드).
MC1 failed because the 'adjacent calendar date' from the training-only calendar often did not contain the
true predecessor.  MC2 finds the predecessor DATE directly by outdoor-weather midnight continuity (outdoor
weather is shared by both 동, so it pins the date, not the source), then the SOURCE among that date's
labelled records by indoor continuity (+ actuator state), as in deep_cal_8 / deep_cal_10 costs.
Query side uses only its hour 0 (and hour 1 for the trend term) and earlier/training records.
  date cost   (deep_cal_8): trend-corrected jump of out_temp, out_hum, out_wspd (.3), out_rad from the
              candidate's hours 22-23 to the query's hours 0-1, scaled by the farm's hourly-change SD;
  date pick   candidates = labelled reference records (outside the fold); keep those whose date cost is
              within 1.0 of the minimum (the 1-2 records of the best-matching date);
  source cost (deep_cal_10): indoor in_temp/in_hum/in_co2 trend-corrected jump (scales 1/5/40) +
              heating/thermal/circfan/vent state change /50;  pick the cheapest;
  level       P23 = chosen record's hour-23 EC; MC2 row prediction = P23 for all 24 hours.
Reported on pass-2 labelled days (DIAG10 folds, R3S seed mean as the model):
  share of days whose chosen P23 is within .05 of the true hour-0 EC; row RMSE model vs MC2 vs
  blend .5/.5; by normal / high.  Clue (fixed): within-.05 share >= .7 AND MC2 or blend beats the model
  on pass-2 rows and normal days."""
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
tr_days = {(f, d) for f, d in labset}
SC = {}
for f in ("F13", "F47"):
    T = X[(X.farm == f) & (X.row_id.isin(set(Y.row_id)))]
    for v in W:
        SC[(f, v)] = np.nanstd(T.sort_values(["day", "hour"]).groupby("day")[v].diff())


def date_cost(f, a, q):
    c = 0.0
    for v in W:
        A23, A22, B0, B1 = P.loc[(f, a), (v, 23)], P.loc[(f, a), (v, 22)], P.loc[(f, q), (v, 0)], P.loc[(f, q), (v, 1)]
        pg = .5 * ((A23 - A22) + (B1 - B0)); e = (((B0 - A23) - pg) / SC[(f, v)]) ** 2
        c += WT[v] * (50 if np.isnan(e) else e)
    return c


def src_cost(f, a, q):
    c = 0.0
    for v, s in IN.items():
        A23, A22, B0, B1 = P.loc[(f, a), (v, 23)], P.loc[(f, a), (v, 22)], P.loc[(f, q), (v, 0)], P.loc[(f, q), (v, 1)]
        pg = .5 * ((A23 - A22) + (B1 - B0)); e = (((B0 - A23) - pg) / s) ** 2
        c += 0 if np.isnan(e) else e
    for v in ACT:
        dd = P.loc[(f, q), (v, 0)] - P.loc[(f, a), (v, 23)]
        c += 0 if np.isnan(dd) else abs(dd) / 50
    return c


O = pd.read_csv(os.path.join(env.LOCAL, "hk0_rows_v1.csv"))
F = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"), usecols=["farm", "day", "validator", "validation_fold"])
F = F[F.validator == "DIAG10"].drop_duplicates(["farm", "day"]).set_index(["farm", "day"]).validation_fold
O = O[O.day >= 179].copy()
O["dm"] = O.groupby(["farm", "day"]).sub_ec.transform("mean")
res = []
for (f, d), idx in O.groupby(["farm", "day"]).groups.items():
    k = F[(f, d)]
    ref = [e for (ff, e) in tr_days if ff == f and F.get((ff, e), -1) != k and e != d]
    dc = np.array([date_cost(f, e, d) for e in ref])
    m = dc <= dc.min() + 1.0
    cands = np.array(ref)[m]
    sc = np.array([src_cost(f, e, d) for e in cands])
    pick = cands[int(np.argmin(sc))]
    p23 = E[(f, pick, 23)]; y0 = E[(f, d, 0)]
    res.append(dict(farm=f, day=d, pick=int(pick), n_cand=len(cands), p23=p23, y0=y0, ok=abs(p23 - y0) <= .05,
                    best_any=min(abs(E[(f, e, 23)] - y0) for e in ref)))
D = pd.DataFrame(res)
O = O.merge(D[["farm", "day", "p23"]], on=["farm", "day"])
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
print("pass-2 labelled days %d: chosen P23 within .05 of true 0h EC: %.0f%% (candidates per day median %d); "
      "some labelled record within .05 exists: %.0f%%" % (len(D), 100 * D.ok.mean(), D.n_cand.median(), 100 * (D.best_any <= .05).mean()))
O["bl"] = .5 * O.p + .5 * O.p23
for nm, m in (("pass-2", O.dm.notna()), ("normal", O.dm < 1), ("high", O.dm >= 1)):
    g = O[m]
    print("  %-7s model %.4f | MC2 %.4f | blend %.4f" % (nm, r(g.p - g.sub_ec), r(g.p23 - g.sub_ec), r(g.bl - g.sub_ec)))
D.to_csv(os.path.join(env.LOCAL, "mc2_days_v1.csv"), index=False)
clue = D.ok.mean() >= .7 and any(r(O[c] - O.sub_ec) < r(O.p - O.sub_ec) and r((O[c] - O.sub_ec)[O.dm < 1]) < r((O.p - O.sub_ec)[O.dm < 1]) for c in ("p23", "bl"))
print("MC2 clue:", clue)
