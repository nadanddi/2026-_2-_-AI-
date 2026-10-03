# -*- coding: utf-8 -*-
"""HC4 (descriptive; 2026-10-04 집 클로드).  Above-mean high-EC days: well predicted
(n=4) vs under-predicted (n=11), HC3 groups.  For every day feature: AUC (good vs bad),
whether the two groups separate perfectly, group medians; also hourly profiles of key
actuators.  Chance of a perfect split for one feature/direction = 1/C(15,4) = 1/1365;
with ~70 features x 2 directions, ~0.1 perfect splits are expected by chance.
Also the same comparison restricted to pass 1 (to remove the pass-2 effect)."""
import env  # noqa: F401
import os
import numpy as np, pandas as pd
from math import comb
from sklearn.metrics import roc_auc_score
H = pd.read_csv(os.path.join(env.LOCAL, "hc3_high_day_groups.csv"))
S = H[H.level == "평균보다 높음"].copy()
S["good"] = (S.fit == "잘 맞춤").astype(int)
skip = {"farm", "day", "y", "p", "err", "level", "fit", "grp", "good", "role", "dong", "ec", "hi", "date"}
S["is_B"] = (S.dong == "B").astype(float); S["is_second"] = (S.role == "second").astype(float); S["is_F47"] = (S.farm == "F47").astype(float)
cols = [c for c in S.columns if c not in skip and pd.api.types.is_numeric_dtype(S[c])]
def table(T, tag):
    rows = []
    for c in cols:
        z = T[[c, "good"]].dropna()
        if z.good.nunique() < 2 or z[c].nunique() < 2: continue
        a = roc_auc_score(z.good, z[c]); g, b = z[z.good == 1][c], z[z.good == 0][c]
        perfect = (g.min() > b.max()) or (g.max() < b.min())
        rows.append(dict(feature=c, auc=max(a, 1 - a), good_higher=a >= .5, perfect=perfect, good_med=g.median(), bad_med=b.median(),
                         good_rng="%.2f~%.2f" % (g.min(), g.max()), bad_rng="%.2f~%.2f" % (b.min(), b.max())))
    R = pd.DataFrame(rows).sort_values("auc", ascending=False)
    ng, nb = int(T.good.sum()), int((1 - T.good).sum())
    print("\n[%s] good %d vs bad %d | chance perfect split per feature-direction 1/%d, features %d" % (tag, ng, nb, comb(ng + nb, ng), len(R)))
    print(R.head(15).round(2).to_string(index=False))
    return R
R = table(S, "all above-mean high days")
R1 = table(S[S.late == 0], "pass 1 only")
