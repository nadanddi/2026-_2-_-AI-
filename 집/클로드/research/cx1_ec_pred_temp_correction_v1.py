# -*- coding: utf-8 -*-
"""CX1 (fixed before running; 2026-10-03 집 클로드).  Follow-up required by CX0's
fixed reading (only clue: F47 early, temperature residual on predicted-high-EC
days +.135 vs -.082 C, 16 days; overall rho .06).
Correction: W30G' = W30G + clip(a_farm + b_farm * ecp, -.5, .5), ecp = R3S EC
day-mean prediction (DIAG10 OOF; input-only model), a/b per farm by ridge
(alpha 1 on standardized ecp) fitted on the OTHER DIAG10 temperature folds' days
(day-mean residuals), applied to the held-out fold.  Days without EC OOF: no change.
Rule (user's, available validator only): both seeds better on DIAG10 and paired
block bootstrap (farm x 5-day, 20000) P(worse) < .025; EXT has no matched EC OOF
-> cannot pass the full rule; a DIAG10 pass only marks it for a full test.
"""
import env  # noqa: F401
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

T = pd.read_csv(os.path.join(env.LOCAL, "temp_TC2_oof.csv"))
T = T[T.validator == "DIAG10"].copy()
E = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))
E = E[E.validator == "DIAG10"].copy()
E["ecp"] = E[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
T = T.merge(E.groupby(["farm", "day"]).ecp.mean().reset_index(), on=["farm", "day"], how="left")
r = lambda e: float(np.sqrt(np.mean(np.square(e))))
rng = np.random.default_rng(20261003)
T["cl"] = T.farm + "_" + (T.day // 5).astype(str)
ok = True
for sd in (7, 101):
    w = "w30_base_%d" % sd
    T["res"] = T.sub_temp - T[w]
    corr = np.zeros(len(T))
    for k in T.fold.unique():
        for f in ("F13", "F47"):
            tr = (T.fold != k) & (T.farm == f) & T.ecp.notna()
            te = ((T.fold == k) & (T.farm == f) & T.ecp.notna()).values
            dd = T[tr].groupby("day").agg(res=("res", "mean"), ecp=("ecp", "first"))
            mu, sdv = dd.ecp.mean(), dd.ecp.std()
            m = Ridge(alpha=1.0).fit(((dd.ecp - mu) / sdv).to_frame(), dd.res)
            corr[te] = np.clip(m.predict(((T.ecp[te] - mu) / sdv).to_frame()), -.5, .5)
    T["cx_%d" % sd] = T[w] + corr
    a, b = r(T[w] - T.sub_temp), r(T["cx_%d" % sd] - T.sub_temp)
    L = T.day >= 179
    dd = (T["cx_%d" % sd] - T.sub_temp) ** 2 - (T[w] - T.sub_temp) ** 2
    cl = dd.groupby(T.cl).agg(["sum", "count"]); sm, n = cl["sum"].values, cl["count"].values
    idx = rng.integers(0, len(sm), (20000, len(sm)))
    p = float(((sm[idx].sum(1) / n[idx].sum(1)) >= 0).mean())
    print("seed %d DIAG10 W30G %.4f -> CX1 %.4f (%+.2f%%) late %.4f -> %.4f  P(worse) %.4f" % (
        sd, a, b, 100 * (b / a - 1), r((T[w] - T.sub_temp)[L]), r((T["cx_%d" % sd] - T.sub_temp)[L]), p))
    ok &= (b < a) and p < .025
print("\nCX1 decision:", "DIAG10 PASS -> needs full test" if ok else "FAIL")
