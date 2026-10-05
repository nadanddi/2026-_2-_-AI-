# -*- coding: utf-8 -*-
"""MC0 (oracle / upper bound, 2026-10-05 집 클로드).  How well can a labelled day's EC be predicted from
the SAME SOURCE's adjacent calendar dates' labels (EC is continuous across midnight)?  Source links =
deep_cal_10_links.csv (consecutive calendar dates within a source, built from inputs only but with a
global assignment that also uses later inputs -> UPPER BOUND, not a legal feature yet).
For every labelled record r with a labelled predecessor p (link p -> r) and/or successor n (r -> n):
  P23   constant = p's hour-23 EC
  INT   linear interpolation over the 24 hours between p's hour-23 EC and n's hour-0 EC
Row RMSE vs truth, by pass and normal/high; compare with R3S DIAG10 OOF on the same rows."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
L = pd.read_csv(os.path.join(env.LOCAL, "deep_cal_10_links.csv"))
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])].copy()
Y["farm"], Y["day"], Y["hour"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int), Y.row_id.str[8:10].astype(int)
E = Y.set_index(["farm", "day", "hour"]).sub_ec
lab = set(zip(Y.farm, Y.day))
prev = {(f, b): a for f, a, b in zip(L.farm, L.d_from, L.d_to)}
nxt = {(f, a): b for f, a, b in zip(L.farm, L.d_from, L.d_to)}
O = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); O = O[O.validator == "DIAG10"].copy()
O["p"] = O[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
O["P23"] = np.nan; O["INT"] = np.nan; O["gap_p"] = np.nan
for (f, d), idx in O.groupby(["farm", "day"]).groups.items():
    p = prev.get((f, d)); n = nxt.get((f, d))
    a = E.get((f, p, 23), np.nan) if p is not None and (f, p) in lab else np.nan
    b = E.get((f, n, 0), np.nan) if n is not None and (f, n) in lab else np.nan
    hh = O.loc[idx, "hour"].values
    O.loc[idx, "P23"] = a
    if np.isfinite(a) and np.isfinite(b):
        O.loc[idx, "INT"] = a + (b - a) * (hh + 1) / 25
    elif np.isfinite(a):
        O.loc[idx, "INT"] = a
    elif np.isfinite(b):
        O.loc[idx, "INT"] = b
O["dm"] = O.groupby(["farm", "day"]).sub_ec.transform("mean")
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
for nm, m in (("all", O.dm.notna()), ("pass-2", O.day >= 179), ("pass-2 normal", (O.day >= 179) & (O.dm < 1)), ("pass-2 high", (O.day >= 179) & (O.dm >= 1)), ("pass-1", O.day < 179)):
    for c in ("P23", "INT"):
        g = O[m & O[c].notna()]
        print("%-14s %s: rows %5d (%.0f%% of segment)  oracle %.4f  vs R3S same rows %.4f" % (nm, c, len(g), 100 * len(g) / max(1, m.sum()), r(g[c] - g.sub_ec), r(g.p - g.sub_ec)))
both = O[(O.day >= 179) & O.P23.notna()]
print("\npass-2 days with labelled predecessor via chain: %d of %d" % (both.groupby(["farm", "day"]).ngroups, O[O.day >= 179].groupby(["farm", "day"]).ngroups))
