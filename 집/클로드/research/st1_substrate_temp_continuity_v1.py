# -*- coding: utf-8 -*-
"""ST1 (diagnostic, 2026-10-07 집 클로드; user: "쓸 수 있는 데이터가 더 있을 것 같은데").
Unused data check: F13/F47 SUBSTRATE-TEMPERATURE labels (sub_temp, given for all training rows).  Substrate
has thermal inertia, so like EC it should be continuous across midnight within a source.  Can it pick the
true previous-date record (source/date identification, the bottleneck of 6.404) better than the indoor
input continuity used so far (CH3 icost, 36-39 %)?
Truth: CH2 label links a -> b with |EC jump| < .05 (reliable, 6.346).  For each such b:
  candidates = labelled records p != b with outdoor date_cost(p -> b) <= 3 (b's hours 0-1 only, as CH3)
  pick by  I   indoor+actuator continuity icost (CH3 rule)                         [baseline]
           T   |sub_temp_p(23) - sub_temp_b(0)| with b's TRUE label                  [upper bound, not usable]
           Tp  same with b's 0 h sub_temp PREDICTED from b's hour-0 inputs (LightGBM, grouped 10-fold over
               records, training rows outside the record's fold)                     [usable form]
           I+Tp  icost + (Tp gap / s)^2, s = residual sd of the predictor
Reading rule (fixed before running): sub_temp is a new lead only if Tp or I+Tp beats I by >= 15 points of
correct-pick share on pass-2 b records AND on all b records; else record and stop.
Run:  PYTHONPATH="" py -3.12 -u st1_substrate_temp_continuity_v1.py
"""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
import lightgbm as lgb

W = ["out_temp", "out_hum", "out_wspd", "out_rad"]; WT = {"out_temp": 1, "out_hum": 1, "out_wspd": .3, "out_rad": 1}
IN = {"in_temp": 1.0, "in_hum": 5.0, "in_co2": 40.0}; ACT = ["act_heating", "act_thermal", "act_circfan", "act_vent"]
FEAT = W + list(IN) + ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog", "hour"]


def main():
    X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
    Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
    X = X.merge(Y, on="row_id")
    X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
    # out-of-fold substrate-temperature prediction (grouped by record), per farm model with farm flag
    X["rec"] = X.farm + "_" + X.day.astype(str); recs = X.rec.unique()
    fold = {r: i % 10 for i, r in enumerate(np.random.default_rng(0).permutation(recs))}
    X["fold"] = X.rec.map(fold); X["is47"] = (X.farm == "F47").astype(int); X["tp"] = np.nan
    for k in range(10):
        tr, te = X[X.fold != k], X[X.fold == k]
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=.05, num_leaves=31, min_child_samples=20, verbose=-1, random_state=0)
        m.fit(tr[FEAT + ["is47"]], tr.sub_temp); X.loc[te.index, "tp"] = m.predict(te[FEAT + ["is47"]])
    s_res = float(np.std((X.tp - X.sub_temp)[X.hour == 0]))
    print("sub_temp predictor at hour 0: RMSE %.3f (sd of label %.3f)" % (s_res, float(X[X.hour == 0].sub_temp.std())))
    P = X.pivot_table(index=["farm", "day"], columns="hour", values=W + list(IN) + ACT + ["sub_ec", "sub_temp", "tp"])
    SC = {(f, v): np.nanstd(X[X.farm == f].sort_values(["day", "hour"]).groupby("day")[v].diff()) for f in ("F13", "F47") for v in W}

    def jmp(f, a, b, v):
        A23, A22, B0, B1 = P.loc[(f, a), (v, 23)], P.loc[(f, a), (v, 22)], P.loc[(f, b), (v, 0)], P.loc[(f, b), (v, 1)]
        return (B0 - A23) - .5 * ((A23 - A22) + (B1 - B0))

    dcost = lambda f, a, b: sum(WT[v] * ((jmp(f, a, b, v) / SC[(f, v)]) ** 2) for v in W)

    def icost(f, a, b):
        return sum(((jmp(f, a, b, v) / s) ** 2) for v, s in IN.items()) + sum(abs(P.loc[(f, b), (v, 0)] - P.loc[(f, a), (v, 23)]) / 50 for v in ACT)

    L = pd.read_csv(os.path.join(env.LOCAL, "ch2_label_links_v1.csv")); L = L[L.ec_jump.abs() < .05]
    print("reliable CH2 links %d" % len(L))
    # how continuous is sub_temp along true links vs any candidate?
    res = []
    for r in L.itertuples():
        f, a, b = r.farm, int(r.a), int(r.b)
        days = [d for d in P.loc[f].index if d != b]
        cands = [p for p in days if np.isfinite(dcost(f, p, b)) and dcost(f, p, b) <= 3]
        if a not in cands or len(cands) < 2:
            res.append(dict(farm=f, b=b, n=len(cands), a_in=a in cands)); continue
        T0, Tp0 = P.loc[(f, b), ("sub_temp", 0)], P.loc[(f, b), ("tp", 0)]
        g = {p: abs(P.loc[(f, p), ("sub_temp", 23)] - T0) for p in cands}
        gp = {p: abs(P.loc[(f, p), ("sub_temp", 23)] - Tp0) for p in cands}
        ic = {p: icost(f, p, b) for p in cands}
        pick = {"I": min(cands, key=ic.get), "T": min(cands, key=g.get), "Tp": min(cands, key=gp.get),
                "I+Tp": min(cands, key=lambda p: ic[p] + (gp[p] / s_res) ** 2)}
        res.append(dict(farm=f, b=b, n=len(cands), a_in=True, gap_true=g[a], gap_other=np.median([g[p] for p in cands if p != a]),
                        **{k: v == a for k, v in pick.items()}))
    R = pd.DataFrame(res); E = R[R.a_in & (R.n >= 2)]
    print("links with the true predecessor among >= 2 candidates: %d of %d (median candidates %d)" % (len(E), len(R), int(E.n.median())))
    print("sub_temp midnight gap: true predecessor median %.3f vs other candidates median %.3f" % (E.gap_true.median(), E.gap_other.median()))
    for nm, G in (("all", E), ("pass-2 b", E[E.b >= 179]), ("pass-1 b", E[E.b < 179])):
        print("  %-9s n %3d  correct pick: I %.2f  T(true, bound) %.2f  Tp %.2f  I+Tp %.2f  (random %.2f)" % (
            nm, len(G), G["I"].mean(), G["T"].mean(), G["Tp"].mean(), G["I+Tp"].mean(), float((1 / G.n).mean())))
    R.to_csv(os.path.join(env.LOCAL, "st1_links_v1.csv"), index=False)
    best = max(("Tp", "I+Tp"), key=lambda k: E[k].mean())
    p2 = E[E.b >= 179]
    lead = (E[best].mean() - E.I.mean() >= .15) and (p2[best].mean() - p2.I.mean() >= .15)
    print("ST1 reading:", "new lead (%s)" % best if lead else "no lead -> stop")


if __name__ == "__main__":
    main()
