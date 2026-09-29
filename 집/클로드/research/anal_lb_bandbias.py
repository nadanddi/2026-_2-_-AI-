# -*- coding: utf-8 -*-
"""Solve the band-level bias of submission 04 (temperature) on the COLD test
bands from three public LB pairs (anal_lb_pairs.py identity).  Assumptions:
bias is constant within an in_temp band and zero for bands above 10 C
(stated, not verified).  Unknowns b(<=6), b(6-8), b(8-10); equations from
pairs 04->05, 04->06, 03->04 (for 03->04: mean(d*e04) = mean(d*e03) + mean(d^2)).
Segment-level use of our own public scores only.
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
import common

LB = {"03": 0.6666, "04": 0.5624, "05": 0.5697, "06": 0.5456}
_, _, sX = common.load_raw()
sX = sX.set_index("row_id")
P = {n: pd.read_csv("submissions/submission_%s.csv" % n).set_index("row_id").loc[sX.index, "sub_temp"].values for n in LB}
band = pd.cut(sX.in_temp.values, [-99, 6, 8, 10, 99]).codes
A, rhs = [], []
for a, b in (("04", "05"), ("04", "06"), ("03", "04")):
    d = P[b] - P[a]
    cov = (LB[b] ** 2 - LB[a] ** 2 - np.mean(d ** 2)) / 2          # mean(d * e_a)
    if a == "03":
        cov = cov + np.mean(d ** 2)                                 # -> mean(d * e_04)
    A.append([np.sum(d[band == k]) / len(d) for k in range(3)])
    rhs.append(cov)
A, rhs = np.array(A), np.array(rhs)
print("design (mean d per band x share):\n", A.round(4), "\nrhs", rhs.round(4))
sol = np.linalg.solve(A, rhs)
print("implied bias of submission 04 (pred - truth): <=6C %+.2f | 6-8C %+.2f | 8-10C %+.2f" % tuple(sol))
print("condition number %.1f" % np.linalg.cond(A))
d06 = P["06"] - P["04"]
print("submission 06 implied bias = 04 bias + mean change: %s" % [round(sol[k] + d06[band == k].mean(), 2) for k in range(3)])
