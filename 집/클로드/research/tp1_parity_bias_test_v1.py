# -*- coding: utf-8 -*-
"""TP1 (diagnostic; 2026-10-03 집 클로드).  TP0: F47 pass 2 W40G day residual
+.265 (odd record day) vs -.270 (even); pass 1 -.037 vs -.075; F13 ~0.
Significance of the parity difference of day-mean residuals per farm x pass:
Welch t and a permutation test (20000 label shuffles of parity within group).
Also the same with residuals relative to the 5-day-chunk mean (removes slow
season drift that could alias with parity in a short window)."""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

T = pd.read_csv(os.path.join(env.LOCAL, "temp_TC2_oof.csv"))
T = T[T.validator == "DIAG10"].copy()
g = np.where(T.in_temp.isna(), 1, np.clip((T.in_temp - 8) / 2, 0, 1))
m = (T.mask_base_7 + T.mask_base_101) / 2
T["res"] = T.sub_temp - (0.4 * m + (0.2 + 0.4 * (1 - g)) * T.codex_base + 0.4 * g * T.pfn)
D = T.groupby(["farm", "day"]).res.mean().reset_index()
D["pas"], D["par"], D["chunk"] = (D.day >= 179).astype(int), D.day % 2, D.day // 6
D["resc"] = D.res - D.groupby(["farm", "chunk"]).res.transform("mean")
rng = np.random.default_rng(20261003)
for v in ("res", "resc"):
    print("\n[%s]" % v)
    for (f, p), G in D.groupby(["farm", "pas"]):
        a, b = G[v][G.par == 1].values, G[v][G.par == 0].values
        diff = a.mean() - b.mean()
        t = ttest_ind(a, b, equal_var=False)
        x, n1 = G[v].values, len(a)
        perm = np.array([(lambda z: z[:n1].mean() - z[n1:].mean())(rng.permutation(x)) for _ in range(20000)])
        print("%s pass%d n_odd %3d n_even %3d  odd-even %+.3f  Welch p %.4f  perm p %.4f" % (
            f, p, n1, len(b), diff, t.pvalue, (np.abs(perm) >= abs(diff)).mean()))
