# -*- coding: utf-8 -*-
"""LB1 (descriptive, segment level; rule-5 reading of 2026-09-26): can the
four scored EC submissions tell anything about the mean true EC on the 20
test sealed days?  Score identity: ||p_i - y||^2 - ||p_3 - y||^2 =
||p_i||^2 - ||p_3||^2 - 2<p_i - p_3, y>, so <v, y> is known for v in
span{p1-p3, p2-p3, p5-p3}.  A segment quantity <s, y> is identifiable only to
the extent s lies in that span.  Reports the share of ||s||^2 (and of the
centred contrast) inside the span.  No row-level label recovery.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u lb1_sealed_span_v1.py"""
import env  # noqa
import os
import numpy as np
import pandas as pd
S = env.SUBMIT
f = {1: "01회차_2026-09-22/submission_01_scored.csv", 2: "02회차_2026-09-25/submission_03.csv",
     3: "03회차_2026-09-26/submission_04.csv", 5: "05회차_2026-09-26/submission_06.csv"}
score = {1: 0.2442, 2: 0.2287, 3: 0.2055, 5: 0.2134}
P = {k: pd.read_csv(os.path.join(S, v)).set_index("row_id").sub_ec for k, v in f.items()}
ids = P[3].index
P = {k: v.reindex(ids).values for k, v in P.items()}
D = pd.read_csv(os.path.join(env.LOCAL, "sf1_daytable.csv"))
D = D[D.set == "test"]
sealed = {(a, b) for a, b, s in zip(D.farm, D.day, D.sealed) if s}
p = pd.Series(ids).str.split("_", expand=True)
s = np.array([(a, int(b)) in sealed for a, b in zip(p[0], p[1])], float)
print("test rows %d, sealed rows %d" % (len(ids), s.sum()))
V = np.c_[[P[k] - P[3] for k in (1, 2, 5)]].T
n = len(ids)
known = np.array([(n * (score[k] ** 2 - score[3] ** 2) - (P[k] ** 2).sum() + (P[3] ** 2).sum()) / -2 for k in (1, 2, 5)])
for nm, t in (("sealed indicator", s), ("sealed - nonsealed contrast", s / s.sum() - (1 - s) / (1 - s).sum())):
    b, *_ = np.linalg.lstsq(V, t, rcond=None)
    r = t - V @ b
    print("%-28s share of ||s||^2 inside span: %.1f%%" % (nm, 100 * (1 - (r @ r) / (t @ t))))
for k in (1, 2, 5):
    d = P[k] - P[3]
    print("round %d - round 3: mean change sealed %+.3f nonsealed %+.3f | share of ||d||^2 on sealed rows %.0f%%"
          % (k, d[s == 1].mean(), d[s == 0].mean(), 100 * (d[s == 1] ** 2).sum() / (d ** 2).sum()))
print("round 3 mean prediction: sealed %.3f nonsealed %.3f" % (P[3][s == 1].mean(), P[3][s == 0].mean()))
