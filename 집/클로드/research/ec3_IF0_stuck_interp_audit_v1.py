# -*- coding: utf-8 -*-
"""EC stage-3 IF0 (diagnostic, fixed before running; 2026-10-03 집 클로드).
Survey D: interpolated / stuck inputs can carry hidden information (or mislead).
Per F13/F47 row and continuous input (in_temp, in_hum, in_co2, out_temp, out_hum,
out_rad>0 hours only): stuck = equal to previous hour value (|diff|<1e-9) for >=3
consecutive hours; linear = |second difference| < 1e-6 with nonzero slope over
>=3 points (interpolation signature).  Train_X only.
Reports: share by farm x pass x parity; day share vs |R3S DIAG10 day residual|
(Spearman) and vs label day mean.  Reading (fixed): clue if some flag has
share >= 5% in a group AND |Spearman| >= .25 with |residual| (n >= 30 days).
"""
import env  # noqa: F401
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

V = ["in_temp", "in_hum", "in_co2", "out_temp", "out_hum", "out_rad"]


def runs(flag, g):
    # mark rows belonging to runs of >=3 consecutive True within the day
    out = np.zeros(len(flag), bool)
    for _, idx in g.groups.items():
        f = flag[idx].values
        n = 0
        for j in range(len(f) + 1):
            if j < len(f) and f[j]:
                n += 1
            else:
                if n >= 2:          # 2 equal diffs = 3 equal points
                    out[[idx[k] for k in range(j - n - 1, j)]] = True
                n = 0
    return out


def main():
    X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id"] + V)
    X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
    X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
    X = X.sort_values(["farm", "day", "hour"]).reset_index(drop=True)
    g = X.groupby(["farm", "day"])
    F = []
    for v in V:
        d1 = g[v].diff()
        d2 = d1.groupby([X.farm, X.day]).diff()
        st = (d1.abs() < 1e-9)
        if v == "out_rad":
            st &= X[v] > 0
        li = (d2.abs() < 1e-6) & (d1.abs() > 1e-9)
        X["stuck_" + v] = runs(st, g)
        X["lin_" + v] = runs(li, g)
        F += ["stuck_" + v, "lin_" + v]
    X["pas"], X["par"] = (X.day >= 179).astype(int), X.day % 2
    print("share of rows flagged (farm pass parity)")
    print(X.groupby(["farm", "pas", "par"])[F].mean().round(3).T.to_string())
    D = X.groupby(["farm", "day", "pas", "par"])[F].mean().reset_index()
    O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))
    O = O[O.validator == "DIAG10"].copy()
    O["e"] = O[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1) - O.sub_ec
    E = O.groupby(["farm", "day"]).agg(e=("e", "mean"), y=("sub_ec", "mean")).reset_index()
    D = D.merge(E, on=["farm", "day"])
    clues = []
    print("\nSpearman (day flag share vs |resid|, vs label mean) where share >= 5%")
    for (f, ps, par), G in D.groupby(["farm", "pas", "par"]):
        for c in F:
            if G[c].mean() >= .05 and len(G) >= 30 and G[c].nunique() > 2:
                r1 = spearmanr(G[c], G.e.abs()).correlation
                r2 = spearmanr(G[c], G.y).correlation
                print("  %s pass%d par%d %-16s share %.3f n %3d  rho|e| %+.2f  rho y %+.2f" % (f, ps, par, c, G[c].mean(), len(G), r1, r2))
                if abs(r1) >= .25:
                    clues.append((f, ps, par, c, round(r1, 2)))
    print("\nIF0 clues:", clues if clues else "none")


if __name__ == "__main__":
    main()
