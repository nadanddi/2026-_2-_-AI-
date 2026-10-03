# -*- coding: utf-8 -*-
"""CX0 (diagnostic, fixed before running; 2026-10-03 집 클로드).
Question: do the large temperature day-level errors (W30G, mostly F47 late,
Codex TK1 6.172 / 6.189) and the high-EC states (6.177/6.200) fall on the SAME
days?  If yes, the EC model's input-only prediction (high-EC ranked with AUC
.989) could flag temperature anomaly days at evaluation time (legal: built from
same-greenhouse current/previous inputs only).
Data: DIAG10 public OOF only.  Temperature: temp_TC2_oof.csv W30G (w30_base_7,
seed 101 also); EC: ec2_DC5_oof.csv R3S seed mean and labels.  Day level means.
Measures per farm x pass (early <179 / late >=179) and overall:
  Spearman(temp day residual, EC label day mean), Spearman(temp residual, EC pred
  day mean), Spearman(substrate-indoor gap, EC label), mean temp residual on
  predicted-high-EC days (EC pred day mean >= .9) vs others.
Reading (fixed): clue if |rho(temp resid, EC pred)| >= .30 overall or in a late
farm group with n >= 15, OR mean residual difference (pred-high minus others)
>= .20 C in absolute value with >= 8 pred-high days.  Clue -> separate
pre-registered model test (EC prediction as a temperature feature).
"""
import env  # noqa: F401
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

T = pd.read_csv(os.path.join(env.LOCAL, "temp_TC2_oof.csv"))
T = T[T.validator == "DIAG10"].copy()
T["res"] = T.sub_temp - (T.w30_base_7 + T.w30_base_101) / 2
T["gap"] = T.sub_temp - T.in_temp
TD = T.groupby(["farm", "day"]).agg(res=("res", "mean"), gap=("gap", "mean"), absres=("res", lambda x: np.sqrt(np.mean(x ** 2))))
E = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))
E = E[E.validator == "DIAG10"].copy()
E["p"] = E[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
ED = E.groupby(["farm", "day"]).agg(ec=("sub_ec", "mean"), ecp=("p", "mean"))
D = TD.join(ED, how="inner").reset_index()
D["late"] = D.day >= 179
print("days with both labels (DIAG10): %d (temp %d, EC %d)" % (len(D), len(TD), len(ED)))
clues = []
rows = [("ALL", D)] + [("%s %s" % (f, "late" if l else "early"), g) for (f, l), g in D.groupby(["farm", "late"])]
print("\n%-10s %4s %9s %9s %9s %9s | %5s %8s %8s" % ("group", "n", "r(res,ec)", "r(res,ecP)", "r(gap,ec)", "r(|e|,ecP)", "nHiP", "resHiP", "resOth"))
for name, g in rows:
    a = spearmanr(g.res, g.ec).correlation
    b = spearmanr(g.res, g.ecp).correlation
    c = spearmanr(g.gap, g.ec).correlation
    e = spearmanr(g.absres, g.ecp).correlation
    hi = g.ecp >= 0.9
    rh, ro = g.res[hi].mean(), g.res[~hi].mean()
    print("%-10s %4d %9.2f %9.2f %9.2f %9.2f | %5d %8.3f %8.3f" % (name, len(g), a, b, c, e, hi.sum(), rh, ro))
    if (name == "ALL" or ("late" in name and len(g) >= 15)) and abs(b) >= .30:
        clues.append((name, "rho", round(b, 2)))
    if hi.sum() >= 8 and abs(rh - ro) >= .20:
        clues.append((name, "hi-diff", round(rh - ro, 3)))
print("\nlargest temperature error days (day RMSE) with EC label / pred:")
print(D.sort_values("absres", ascending=False).head(15)[["farm", "day", "res", "gap", "absres", "ec", "ecp"]].round(3).to_string(index=False))
print("\nCX0 clues:", clues if clues else "none")
