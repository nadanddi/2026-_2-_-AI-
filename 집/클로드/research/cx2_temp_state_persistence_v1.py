# -*- coding: utf-8 -*-
"""CX2 (diagnostic; 2026-10-03 집 클로드).  Is the temperature day-level error a
persistent per-source (동 = record-day parity) state like high EC (C3: EC day
level correlates .65 with the previous same-parity labelled day in pass 2)?
DIAG10 W30G (seeds 7/101 mean) day residual r(d) vs r of the nearest previous
labelled day of the SAME parity (gap <= 6 record days) and of the OPPOSITE parity
(gap <= 3), Spearman per farm x pass.  Also the same for substrate-indoor gap.
Descriptive only (labels of other days; previous-label residual correction was
already tested and rejected by Codex TBIAS 6.178)."""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

T = pd.read_csv(os.path.join(env.LOCAL, "temp_TC2_oof.csv"))
T = T[T.validator == "DIAG10"].copy()
T["res"] = T.sub_temp - (T.w30_base_7 + T.w30_base_101) / 2
T["gap"] = T.sub_temp - T.in_temp
D = T.groupby(["farm", "day"]).agg(res=("res", "mean"), gap=("gap", "mean")).reset_index()
print("%-10s %-4s %5s %8s %5s %8s" % ("group", "var", "nS", "rhoSame", "nO", "rhoOpp"))
for (f, late), g in D.groupby([D.farm, D.day >= 179]):
    s = g.set_index("day")
    for v in ("res", "gap"):
        xs, ys, xo, yo = [], [], [], []
        for d in s.index:
            prev_same = [p for p in s.index if p < d and (d - p) % 2 == 0 and d - p <= 6]
            prev_opp = [p for p in s.index if p < d and (d - p) % 2 == 1 and d - p <= 3]
            if prev_same:
                xs.append(s.loc[max(prev_same), v]); ys.append(s.loc[d, v])
            if prev_opp:
                xo.append(s.loc[max(prev_opp), v]); yo.append(s.loc[d, v])
        rs = spearmanr(xs, ys).correlation if len(xs) > 5 else np.nan
        ro = spearmanr(xo, yo).correlation if len(xo) > 5 else np.nan
        print("%-10s %-4s %5d %8.2f %5d %8.2f" % ("%s %s" % (f, "late" if late else "early"), v, len(xs), rs, len(xo), ro))
