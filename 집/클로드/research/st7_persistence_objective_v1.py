# -*- coding: utf-8 -*-
"""ST7 (check of ST6 circularity; 2026-10-03 집 클로드).  ST6 assigned singletons
to a 동 with an input classifier; inputs may track EC, which would inflate the
'same-dong' persistence.  Here only PAIR records (objective A/B from pair order)
are used: EC label day mean vs the previous labelled pair record of the same 동
(A->A, B->B) and of the other 동, within <= 10 record days, per farm x pass.
Also 'two 동 high on the same date?': within pairs, both records with EC >= 1."""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = pd.read_csv(os.path.join(env.LOCAL, "st_record_roles_v1.csv"))
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])].copy()
Y["farm"], Y["day"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int)
EC = Y.groupby(["farm", "day"]).sub_ec.mean()
for f in ("F13", "F47"):
    G = R[(R.farm == f) & R.role.isin(["first", "second"])].copy()
    G["dong"] = np.where(G.role == "second", "B", "A")
    G = G[[(f, d) in EC.index for d in G.day]]
    G["ec"] = [EC[(f, d)] for d in G.day]
    G = G.sort_values("day").reset_index(drop=True)
    for ps in (0, 1):
        H = G[(G.day >= 179) == ps]
        sx, sy, ox, oy = [], [], [], []
        for i, r in H.iterrows():
            prev = H[(H.day < r.day) & (H.day >= r.day - 10)]
            s, o = prev[prev.dong == r.dong], prev[(prev.dong != r.dong) & (prev.day != r.day - 1)]
            if len(s): sx.append(s.iloc[-1].ec); sy.append(r.ec)
            if len(o): ox.append(o.iloc[-1].ec); oy.append(r.ec)
        rr = lambda x, y: spearmanr(x, y).correlation if len(x) > 5 else np.nan
        print("%s %s pair records %d: same-dong prev rho %.2f (n %d) | other-dong prev (not same date) %.2f (n %d)" % (
            f, "late" if ps else "early", len(H), rr(sx, sy), len(sx), rr(ox, oy), len(ox)))
    P = G.pivot_table(index=np.where(G.dong == "B", G.day - 1, G.day), columns="dong", values="ec").dropna()
    print("  same-date pairs with both labels %d: both >=1: %d, A>=1 only %d, B>=1 only %d, corr A-B %.2f, mean A %.3f B %.3f" % (
        len(P), ((P.A >= 1) & (P.B >= 1)).sum(), ((P.A >= 1) & (P.B < 1)).sum(), ((P.A < 1) & (P.B >= 1)).sum(),
        spearmanr(P.A, P.B).correlation, P.A.mean(), P.B.mean()))
