# -*- coding: utf-8 -*-
"""TP2 (diagnostic; 2026-10-03 집 클로드).  Compare leaders' leaderboard RMSE with
our 'perfect day level' oracle (keep our within-day shape, replace each day's mean
by the true day mean).  If leaders beat that oracle, they predict the within-day
SHAPE better too, i.e. their gain is not only day offsets.  DIAG10 public OOF.
Temperature: W40G rebuilt from stored members; EC: R3S seed mean (season R3; the
submitted season v2 also contains TabPFN, so this is an approximation)."""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd

r = lambda e: float(np.sqrt(np.mean(np.square(e))))
T = pd.read_csv(os.path.join(env.LOCAL, "temp_TC2_oof.csv"))
T = T[T.validator == "DIAG10"].copy()
g = np.where(T.in_temp.isna(), 1, np.clip((T.in_temp - 8) / 2, 0, 1))
m = (T.mask_base_7 + T.mask_base_101) / 2
T["p"] = 0.4 * m + (0.2 + 0.4 * (1 - g)) * T.codex_base + 0.4 * g * T.pfn
E = pd.read_csv(os.path.join(env.LOCAL, "ec2_DC5_oof.csv"))
E = E[E.validator == "DIAG10"].copy()
E["p"] = E[["r3s_7", "r3s_101", "r3s_2024"]].mean(axis=1)
for name, F, y, lead in (("temp W40G", T, "sub_temp", (0.3030, 0.3885, 0.4188)), ("EC R3S", E, "sub_ec", (0.0507, 0.0544, 0.0594))):
    e = F.p - F[y]
    lvl = e.groupby([F.farm, F.day]).transform("mean")
    late = F.day >= 179
    print("%-9s all: model %.4f | perfect day level %.4f | late: model %.4f perfect %.4f | leaders LB %s" % (
        name, r(e), r(e - lvl), r(e[late]), r((e - lvl)[late]), lead))
