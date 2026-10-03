# -*- coding: utf-8 -*-
"""HC6 (descriptive; 2026-10-04 집 클로드).  Are the under-predicted high-EC days the
ones where the model was less sure?  High days (n=31, DIAG10, R3S seed mean):
error vs predicted day level; seed disagreement (sd over seeds of the day mean);
within-day prediction path (mean of hours 0-5 vs 18-23); share of hours predicted
>= 1.0.  Spearman with the error (pred - true)."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); O = O[O.validator == "DIAG10"].copy()
S = ["r3s_7", "r3s_101", "r3s_2024"]; O["p"] = O[S].mean(axis=1)
G = O.groupby(["farm", "day"])
D = G.agg(y=("sub_ec", "mean"), p=("p", "mean")).reset_index()
D["seed_sd"] = G[S].mean().std(axis=1).values
D["p_early"] = O[O.hour <= 5].groupby(["farm", "day"]).p.mean().values
D["p_late"] = O[O.hour >= 18].groupby(["farm", "day"]).p.mean().values
D["share_hours_hi"] = G.p.apply(lambda s: (s >= 1).mean()).values
D["y_early"] = O[O.hour <= 5].groupby(["farm", "day"]).sub_ec.mean().values
D["err"] = D.p - D.y
H = D[D.y >= 1].copy()
print("high days %d" % len(H))
for c in ("p", "seed_sd", "p_early", "p_late", "share_hours_hi"):
    print("  Spearman(err, %-15s) %+.2f" % (c, spearmanr(H.err, H[c]).correlation))
H["grp"] = pd.cut(H.err, [-9, -0.5, -0.2, 0.2, 9], labels=["under >.5", "under .2-.5", "within .2", "over"])
print(H.groupby("grp", observed=True).agg(n=("y", "size"), true=("y", "mean"), pred=("p", "mean"), early_pred=("p_early", "mean"),
      late_pred=("p_late", "mean"), early_true=("y_early", "mean"), hours_hi=("share_hours_hi", "mean"), seed_sd=("seed_sd", "mean")).round(3).to_string())
