# -*- coding: utf-8 -*-
"""VL3: do HX3 'same-bay' cert chains cross greenhouses (F13<->F47)? HX3 make_links has no fa==fb filter (feat4 has). diagnostic only."""
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
from feat4 import *
import numpy as np, pandas as pd
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"
rm = lambda e: float(np.sqrt(np.mean(np.asarray(e) ** 2)))
TL = pd.read_pickle(os.path.join(S3, "truth.pkl"))
T = TL[TL.cert.astype(bool)]
T = T[[bool(TRAIN[int(a)] and TRAIN[int(b)]) for a, b in zip(T.a1, T.b)]]
xf = np.array([FARM[int(a)] != FARM[int(b)] for a, b in zip(T.a1, T.b)])
print(f"train-train cert links {len(T)} | cross-farm {xf.sum()}")
H = r"C:\work\farmai\집\클로드\research\hx8\harness\out_v1"
C = pd.read_csv(H + r"\hx3_cases_cert_Sq_v1.csv")
C["xa"] = [ (isinstance(a, str) and a[:3] != r[:3]) or (isinstance(b, str) and b[:3] != r[:3]) for r, a, b in zip(C.rec, C.ra, C.rb)]
print("cases with anchor in other greenhouse:", int(C.xa.sum()), "/", len(C))
print(C[C.xa][["rec", "side", "k", "m", "ra", "rb", "y", "q", "anc"]].round(3).to_string(index=False))
for s in ("prev", "next"):
    for d in (1, 2):
        G = C[(C.side == s) & (C.d == d)]; X = G[~G.xa]
        print(f"  {s} d={d}: all n {len(G)} q {rm(G.q-G.y):.3f} E1 {rm(G.E1-G.y):.3f} E4p {rm(G.E4p-G.y):.3f} | same-farm-anchor only n {len(X)} q {rm(X.q-X.y):.3f} E1 {rm(X.E1-X.y):.3f} E4p {rm(X.E4p-X.y):.3f}")
# comp mixing farms
comps = C.groupby("comp").rec.apply(lambda s: sorted(set(x[:3] for x in s)))
print("components with targets from both farms:", comps[comps.apply(len) > 1].to_dict())
