# -*- coding: utf-8 -*-
"""SE1 (diagnostic, fixed before running; 2026-10-03 집 클로드).
F13 and F47 share outdoor weather (same site; ST2: 270 identical-weather record
pairs, record-day offset -8..+8).  Do the two greenhouses' day-level residuals move
together on the same calendar date?  Same-date matches = identical 24h weather
(max abs diff < 1e-9) between an F13 and an F47 record within +-15 record days.
DIAG10 public OOF day residuals: temperature W40G (rebuilt), EC R3S seed mean.
Also residual vs the label of the other farm on that date.
Reading (fixed): shared date factor if Spearman(resid F13, resid F47) >= .30 with
n >= 30 for a target (then: rules check before any feature use, since evaluation
features may only use the same greenhouse's inputs)."""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

V = ["out_temp", "out_hum", "out_rad", "out_wspd"]
A = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id"] + V)
A = A[A.row_id.str[:3].isin(["F13", "F47"])].copy()
A["farm"], A["day"], A["hour"] = A.row_id.str[:3], A.row_id.str[4:7].astype(int), A.row_id.str[8:10].astype(int)
W13 = A[A.farm == "F13"].pivot_table(index="day", columns="hour", values=V)
W47 = A[A.farm == "F47"].pivot_table(index="day", columns="hour", values=V)
M = []
for d in W13.index:
    a = W13.loc[d].values
    for e in range(d - 15, d + 16):
        if e in W47.index:
            b = W47.loc[e].values; ok = ~np.isnan(a) & ~np.isnan(b)
            if ok.sum() >= 80 and np.max(np.abs(a[ok] - b[ok])) < 1e-9:
                M.append((d, e))
M = pd.DataFrame(M, columns=["d13", "d47"])
T = pd.read_csv(os.path.join(env.LOCAL, "temp_TC2_oof.csv")); T = T[T.validator == "DIAG10"].copy()
g = np.where(T.in_temp.isna(), 1, np.clip((T.in_temp - 8) / 2, 0, 1))
T["e"] = T.sub_temp - (0.4 * (T.mask_base_7 + T.mask_base_101) / 2 + (0.2 + 0.4 * (1 - g)) * T.codex_base + 0.4 * g * T.pfn)
T["y"] = T.sub_temp
E = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv")); E = E[E.validator == "DIAG10"].copy()
E["e"] = E.sub_ec - E[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1); E["y"] = E.sub_ec
for name, F in (("temp W40G", T), ("EC R3S", E)):
    D = F.groupby(["farm", "day"]).agg(e=("e", "mean"), y=("y", "mean"))
    rows = [(D.loc[("F13", a)], D.loc[("F47", b)]) for a, b in zip(M.d13, M.d47) if ("F13", a) in D.index and ("F47", b) in D.index]
    x13 = np.array([r[0].e for r in rows]); x47 = np.array([r[1].e for r in rows])
    y13 = np.array([r[0].y for r in rows]); y47 = np.array([r[1].y for r in rows])
    print("%s: same-date matched day pairs %d | rho resid13-resid47 %.2f | rho label13-label47 %.2f | rho resid13-label47 %.2f | rho resid47-label13 %.2f" % (
        name, len(rows), spearmanr(x13, x47).correlation, spearmanr(y13, y47).correlation,
        spearmanr(x13, y47).correlation, spearmanr(x47, y13).correlation))
    late = [i for i, (a, b) in enumerate(zip(M.d13, M.d47)) if ("F13", a) in D.index and ("F47", b) in D.index]
print("matched pairs total %d, late (>=179 both) %d" % (len(M), ((M.d13 >= 179) & (M.d47 >= 179)).sum()))
