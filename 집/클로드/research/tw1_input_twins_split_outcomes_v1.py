# -*- coding: utf-8 -*-
"""TW1 (exploration, 2026-10-06 집 클로드).  User question: are there days whose given inputs (indoor
climate, actuators, outdoor weather) are almost identical but whose EC - and the model's verdict (under /
over / correct) - split?
Day vector: 24 hours x 14 inputs (in_temp, in_hum, in_co2, 7 actuators, 4 outdoor), each input
standardized over all labelled hours of its farm; distance = RMS over available cells.
Pairs: every labelled day's nearest OTHER labelled day (same farm), excluding the same-date sibling
(other 동, identical outdoor); also reported with the sibling allowed.
Outcome: DIAG10 R3S seed-mean day residual res = label - prediction; class under (res > .15),
over (res < -.15), ok.
Reported: distance distribution; among the closest 10 % / 25 % of nearest pairs: |EC difference| and the
share whose classes differ, vs random same-farm pairs; the 12 most similar pairs with |dEC| >= .3."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
V = ["in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog",
     "out_temp", "out_hum", "out_rad", "out_wspd"]
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id"] + V)
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); O = O[O.validator == "DIAG10"]
O["p"] = O[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
D = O.groupby(["farm", "day"]).agg(y=("sub_ec", "mean"), p=("p", "mean")).reset_index(); D["res"] = D.y - D.p
D["cls"] = np.where(D.res > .15, "under", np.where(D.res < -.15, "over", "ok"))
role = pd.read_csv(os.path.join(env.LOCAL, "st_record_roles_v1.csv")).set_index(["farm", "day"]).role
rows = []; rng = np.random.default_rng(0); rand = []
for f in ("F13", "F47"):
    Xf = X[X.farm == f].copy()
    for v in V:
        Xf[v] = (Xf[v] - Xf[v].mean()) / (Xf[v].std() or 1)
    P = Xf.pivot_table(index="day", columns="hour", values=V)
    Df = D[D.farm == f].set_index("day"); days = [d for d in Df.index if d in P.index]
    M = P.loc[days].values
    for i, d in enumerate(days):
        sib = {d - 1, d + 1} & {e for e in days if (role.get((f, e)) in ("first", "second")) and abs(e - d) == 1 and
                                 ((role.get((f, d)) == "first" and e == d + 1) or (role.get((f, d)) == "second" and e == d - 1))}
        dist = np.sqrt(np.nanmean((M - M[i]) ** 2, axis=1)); dist[i] = np.inf
        d_all = dist.copy()
        for s in sib:
            dist[days.index(s)] = np.inf
        j = int(np.argmin(dist)); ja = int(np.argmin(d_all))
        e = days[j]
        rows.append(dict(farm=f, day=d, nn=e, dist=dist[j], y=Df.y[d], y_nn=Df.y[e], cls=Df.cls[d], cls_nn=Df.cls[e],
                         res=Df.res[d], res_nn=Df.res[e], nn_is_sibling_allowed=days[ja], dist_sib=d_all[ja]))
        k = int(rng.integers(len(days)))
        if k != i:
            rand.append((abs(Df.y[d] - Df.y[days[k]]), Df.cls[d] != Df.cls[days[k]], np.sqrt(np.nanmean((M[k] - M[i]) ** 2))))
R = pd.DataFrame(rows); RA = pd.DataFrame(rand, columns=["dEC", "cls_diff", "dist"])
R["dEC"] = (R.y - R.y_nn).abs(); R["cls_diff"] = R.cls != R.cls_nn
print("nearest-pair input distance (z-RMS): median %.2f, 10%% %.2f, 25%% %.2f | random pairs median %.2f" % (
    R.dist.median(), R.dist.quantile(.1), R.dist.quantile(.25), RA.dist.median()))
for q in (.10, .25, 1.0):
    S = R[R.dist <= R.dist.quantile(q)]
    print("closest %3d%% of nearest pairs (n %d): |dEC| median %.3f, share |dEC|>=.3 %.2f, classes differ %.2f" % (
        100 * q, len(S), S.dEC.median(), (S.dEC >= .3).mean(), S.cls_diff.mean()))
print("random same-farm pairs: |dEC| median %.3f, share >=.3 %.2f, classes differ %.2f" % (RA.dEC.median(), (RA.dEC >= .3).mean(), RA.cls_diff.mean()))
print("class mix of all days:", D.cls.value_counts().to_dict())
T = R[R.dEC >= .3].sort_values("dist").head(12)
print("\n12 most input-similar pairs with |dEC| >= .3:")
print(T[["farm", "day", "nn", "dist", "y", "y_nn", "cls", "cls_nn", "res", "res_nn"]].round(2).to_string(index=False))
R.to_csv(os.path.join(env.LOCAL, "tw1_pairs_v1.csv"), index=False)
