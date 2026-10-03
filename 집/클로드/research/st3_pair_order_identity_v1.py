# -*- coding: utf-8 -*-
"""ST3 (structure audit; 2026-10-03 집 클로드).  ST2: identical-weather neighbour
pairs = the two records of ONE calendar date; records advance in calendar order;
dates carry 1 or 2 records.  Is the order inside a pair a fixed source (동)?
For each pair (first, second) in train data (both labelled where needed):
paired differences second - first of day means: sub_temp, sub_temp - in_temp,
sub_ec, in_temp, in_hum, in_co2, act_heating, act_vent, act_thermal, act_shade,
act_circfan, act_fog, act_co2.  Sign consistency (share > 0), mean, Wilcoxon p.
If some quantity is consistently different, order = fixed 동 and that quantity is
a 동 fingerprint.  Also: does the 'second' record correlate with parity?"""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

V = ["out_temp", "out_hum", "out_rad", "out_wspd"]
C = ["in_temp", "in_hum", "in_co2", "act_heating", "act_vent", "act_thermal", "act_shade", "act_circfan", "act_fog", "act_co2"]
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id"] + V + C)
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
X = X.merge(Y, on="row_id", how="left")
X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
X["gap"] = X.sub_temp - X.in_temp
DM = X.groupby(["farm", "day"])[C + ["sub_temp", "gap", "sub_ec"]].mean()
for f in ("F13", "F47"):
    G = X[X.farm == f]
    W = G.pivot_table(index="day", columns="hour", values=V)
    days = sorted(W.index); k = 0; pairs = []
    while k < len(days) - 1:
        d = days[k]
        if days[k + 1] == d + 1:
            a, b = W.loc[d].values, W.loc[d + 1].values; ok = ~np.isnan(a) & ~np.isnan(b)
            if ok.sum() >= 80 and np.max(np.abs(a[ok] - b[ok])) < 1e-9:
                pairs.append(d); k += 2; continue
        k += 1
    print("\n%s pairs %d (first record even %d / odd %d)" % (f, len(pairs), sum(d % 2 == 0 for d in pairs), sum(d % 2 for d in pairs)))
    print("  %-12s %4s %8s %8s %9s" % ("quantity", "n", "mean2-1", "share>0", "wilcox p"))
    for q in ["sub_temp", "gap", "sub_ec"] + C:
        dd = np.array([DM.loc[(f, d + 1), q] - DM.loc[(f, d), q] for d in pairs])
        dd = dd[~np.isnan(dd)]
        if len(dd) < 8:
            continue
        p = wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
        print("  %-12s %4d %8.3f %8.2f %9.4f" % (q, len(dd), dd.mean(), (dd > 0).mean(), p))
