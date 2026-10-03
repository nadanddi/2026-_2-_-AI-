# -*- coding: utf-8 -*-
"""HC0 (descriptive; 2026-10-04 집 클로드).  Which inputs separate high-EC days
(label day mean >= 1) from the others?  All labelled F13/F47 days (lock excluded).
Day features from train_X: daily mean, night (0-6h) mean, day (9-16h) mean, share of
hours == 0 for actuators; in/out temp, hum, CO2, rad; plus record role / 동 (C6.205),
season index position (record day pass 1 / DC4 not needed: use pass flag + record day).
Univariate AUC (direction-free: max(AUC, 1-AUC)) per farm-pooled and per farm; and
condition table for the strongest simple rules.  Descriptive only."""
import env  # noqa: F401
import importlib.util, json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("dc5", os.path.join(HERE, "ec2_DC5_r3_season_v1.py"))
dc5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc5)
lock = {(z["farm"], int(z["day"])) for z in json.loads(open(dc5.p3.LOCK, encoding="utf-8").read())["selected"]}
V = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal",
     "act_heating", "act_circfan", "act_co2", "act_fog"]
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id"] + V)
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])]
Y["farm"], Y["day"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int)
EC = Y.groupby(["farm", "day"]).sub_ec.mean().rename("ec")
F = {}
g = X.groupby(["farm", "day"])
for v in V:
    F[v + "_mean"] = g[v].mean()
    F[v + "_night"] = X[X.hour <= 6].groupby(["farm", "day"])[v].mean()
    F[v + "_day"] = X[(X.hour >= 9) & (X.hour <= 16)].groupby(["farm", "day"])[v].mean()
    if v.startswith("act_"):
        F[v + "_zero"] = g[v].apply(lambda s: (s == 0).mean())
F["in_temp_range"] = g.in_temp.max() - g.in_temp.min()
F["in_out_diff"] = F["in_temp_mean"] - F["out_temp_mean"]
D = pd.DataFrame(F).join(EC, how="inner").reset_index()
D = D[[(f, d) not in lock for f, d in zip(D.farm, D.day)]]
R = pd.read_csv(os.path.join(env.LOCAL, "st_dong_assign_v1.csv"))
D = D.merge(R[["farm", "day", "role", "dong"]], on=["farm", "day"], how="left")
D["is_B"] = (D.dong == "B").astype(float); D["is_second"] = (D.role == "second").astype(float)
D["late"] = (D.day >= 179).astype(float); D["recday"] = D.day.astype(float)
D["hi"] = (D.ec >= 1).astype(int)
print("days %d, high-EC %d (%.1f%%) | F13 %d/%d, F47 %d/%d" % (len(D), D.hi.sum(), 100 * D.hi.mean(),
      D[D.farm == "F13"].hi.sum(), (D.farm == "F13").sum(), D[D.farm == "F47"].hi.sum(), (D.farm == "F47").sum()))
cols = [c for c in D.columns if c not in ("farm", "day", "ec", "hi", "role", "dong")]
rows = []
for c in cols:
    z = D[[c, "hi", "farm"]].dropna()
    if z[c].nunique() < 2: continue
    a = roc_auc_score(z.hi, z[c]); r = dict(feature=c, auc=max(a, 1 - a), dir="+" if a >= .5 else "-",
                                            high_mean=z[z.hi == 1][c].mean(), other_mean=z[z.hi == 0][c].mean())
    for f in ("F13", "F47"):
        w = z[z.farm == f]
        if w.hi.nunique() == 2:
            aa = roc_auc_score(w.hi, w[c]); r["auc_" + f] = aa if r["dir"] == "+" else 1 - aa
    rows.append(r)
T = pd.DataFrame(rows).sort_values("auc", ascending=False)
print(T.head(25).round(3).to_string(index=False))
T.to_csv(os.path.join(env.LOCAL, "hc0_auc_table.csv"), index=False)
D.to_csv(os.path.join(env.LOCAL, "hc0_day_features.csv"), index=False)
