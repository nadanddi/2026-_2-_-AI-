# -*- coding: utf-8 -*-
"""TW2 (exploration, 2026-10-07 집 클로드).  For the worst days of the submission_14 configuration (ER1):
(a) hourly flows (24 h) of EC truth / prediction and inputs -> figure;
(b) which INPUT SUBSET points to the right answer: for each subset (indoor climate, actuators, outdoor,
    each single column, each pair of columns), the 5 labelled days of the same farm with the most similar
    24 h profile of that subset (z per farm, RMS; itself and its same-date sibling excluded) and their
    actual daily EC.  'Right direction' = neighbours' mean EC is closer to the truth than the model is.
Model = WT1/ER1 submission_14-configuration OOF (P2LOO for pass-2 days, DIAG10 for F47 132)."""
import env  # noqa: F401
import os, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
IN = ["in_temp", "in_hum", "in_co2"]; ACT = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
OUT = ["out_temp", "out_hum", "out_rad", "out_wspd"]; V = IN + ACT + OUT
PROB = [("F47", 231), ("F13", 217), ("F47", 229), ("F13", 243), ("F13", 233), ("F13", 214), ("F47", 241), ("F47", 216), ("F47", 132)]
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id"] + V)
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])]
X = X.merge(Y[["row_id", "sub_ec"]], on="row_id")
ec = X.groupby(["farm", "day"]).sub_ec.mean()
role = pd.read_csv(os.path.join(env.LOCAL, "st_record_roles_v1.csv")).set_index(["farm", "day"]).role
E1 = pd.read_csv(os.path.join(env.LOCAL, "er1_P2LOO_days_v1.csv")).set_index(["farm", "day"]).p
E0 = pd.read_csv(os.path.join(env.LOCAL, "er1_DIAG10_days_v1.csv")).set_index(["farm", "day"]).p
pred = {k: (E1.get(k) if k in E1.index else E0.get(k)) for k in PROB}
Z = {}
for f in ("F13", "F47"):
    Xf = X[X.farm == f].copy()
    for v in V:
        Xf[v] = (Xf[v] - Xf[v].mean()) / (Xf[v].std() or 1)
    Z[f] = Xf.pivot_table(index="day", columns="hour", values=V)


def sibling(f, d):
    ro = role.get((f, d))
    return d + 1 if ro == "first" else (d - 1 if ro == "second" else None)


def nbrs(f, d, cols, k=5):
    P = Z[f][cols]; days = [e for e in P.index if e != d and e != sibling(f, d)]
    A = P.loc[days].values; b = P.loc[d].values
    dist = np.sqrt(np.nanmean((A - b) ** 2, axis=1))
    o = np.argsort(dist)[:k]
    return [days[i] for i in o], dist[o]


subsets = {"INDOOR(3)": IN, "ACTUATORS(7)": ACT, "OUTDOOR(4)": OUT, "IN+ACT": IN + ACT, "ALL(14)": V}
for c in V:
    subsets["1:" + c] = [c]
rows = []
for f, d in PROB:
    y, p = ec[(f, d)], pred[(f, d)]
    for nm, cols in subsets.items():
        nb, dist = nbrs(f, d, cols)
        ne = np.array([ec[(f, e)] for e in nb])
        rows.append(dict(day="%s %d" % (f, d), y=y, pred=p, subset=nm, nb_mean=ne.mean(), nb_min=ne.min(), nb_max=ne.max(),
                         n_high=int((ne >= 1).sum()), right=abs(ne.mean() - y) < abs(p - y), nb=",".join(map(str, nb))))
    # best pairs of columns
    for a, b in itertools.combinations(V, 2):
        nb, _ = nbrs(f, d, [a, b]); ne = np.array([ec[(f, e)] for e in nb])
        rows.append(dict(day="%s %d" % (f, d), y=y, pred=p, subset="2:%s+%s" % (a, b), nb_mean=ne.mean(), nb_min=ne.min(), nb_max=ne.max(),
                         n_high=int((ne >= 1).sum()), right=abs(ne.mean() - y) < abs(p - y), nb=",".join(map(str, nb))))
T = pd.DataFrame(rows); T.to_csv(os.path.join(env.LOCAL, "tw2_subset_neighbours_v1.csv"), index=False)
pd.set_option("display.width", 220)
for dd, G in T.groupby("day", sort=False):
    y, p = G.y.iloc[0], G.pred.iloc[0]
    print("\n=== %s  truth %.2f  model %.2f  (%s)" % (dd, y, p, "UNDER" if y > p else "OVER"))
    main = G[~G.subset.str.startswith(("1:", "2:"))]
    print(main[["subset", "nb_mean", "nb_min", "nb_max", "n_high", "right", "nb"]].round(2).to_string(index=False))
    one = G[G.subset.str.startswith("1:")].copy(); one["err"] = (one.nb_mean - y).abs()
    print("  single columns closest to truth:", "; ".join("%s %.2f" % (s[2:], m) for s, m in one.sort_values("err").head(4)[["subset", "nb_mean"]].values),
          "| farthest:", "; ".join("%s %.2f" % (s[2:], m) for s, m in one.sort_values("err").tail(3)[["subset", "nb_mean"]].values))
    two = G[G.subset.str.startswith("2:")].copy(); two["err"] = (two.nb_mean - y).abs()
    print("  pairs pointing right: %d of %d; best pairs:" % (two.right.sum(), len(two)), "; ".join("%s %.2f" % (s[2:], m) for s, m in two.sort_values("err").head(3)[["subset", "nb_mean"]].values))
S = T[T.subset.str.startswith("1:")].groupby("subset").right.mean().sort_values(ascending=False)
print("\nhow often each single column's neighbours point closer to the truth than the model (9 days):")
print(S.round(2).to_string())
# figure: hourly flows
fig, axes = plt.subplots(len(PROB), 4, figsize=(17, 2.3 * len(PROB)), sharex=True)
O = pd.read_csv(os.path.join(env.LOCAL, "ec3_WT1_all.csv"))
S3 = (47, 1414, 6464)
O["cur"] = np.clip(.8 * np.mean([.6 * O["et_%d" % s] + .3 * O["lgb_%d" % s] + .1 * O["mlp_%d" % s] for s in S3], axis=0) + .2 * O.pfn, O.lo, O.hi)
raw = X.set_index(["farm", "day", "hour"])
for i, (f, d) in enumerate(PROB):
    g = raw.loc[(f, d)]; v = "P2LOO" if d >= 179 else "DIAG10"
    q = O[(O.validator == v) & (O.farm == f) & (O.day == d)].sort_values("hour")
    ax = axes[i, 0]; ax.plot(g.index, g.sub_ec, "k-", lw=2, label="truth"); ax.plot(q.hour, q.cur, "r--", label="model")
    ax.set_ylabel("%s %d" % (f, d), fontsize=9); ax.set_title("EC" if i == 0 else "")
    ax = axes[i, 1]; [ax.plot(g.index, (g[c] - g[c].mean()) / (g[c].std() or 1), label=c) for c in IN]; ax.set_title("indoor (z within day)" if i == 0 else "")
    ax = axes[i, 2]; [ax.plot(g.index, g[c], label=c.replace("act_", "")) for c in ACT]; ax.set_title("actuators" if i == 0 else "")
    ax = axes[i, 3]; ax.plot(g.index, g.out_temp, label="out_temp"); ax.plot(g.index, g.out_rad / 50, label="out_rad/50"); ax.set_title("outdoor" if i == 0 else "")
for j in range(4):
    axes[0, j].legend(fontsize=6, ncol=2)
plt.tight_layout(); fp = os.path.join(env.LOCAL, "tw2_problem_days_flows_v1.png"); plt.savefig(fp, dpi=80); print("\nfigure:", fp)
