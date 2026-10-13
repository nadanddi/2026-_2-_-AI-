# -*- coding: utf-8 -*-
"""HX9 근권 온도 v5 (2026-10-13 집 클로드, 사후 대조): v4 와 같은 Δ잔차 ~ Δ배지온 관계를 보통(hi10 아닌) 연속일 링크에서 본다.
고EC 특유인지(보통 링크에선 ≈0이어야 '고EC 상태에서만 온도 반응이 큼') 확인용. 서술."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import env  # noqa: F401
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); HX9 = os.path.dirname(HERE)
RNG = np.random.default_rng(5)
V = pd.read_csv(os.path.join(HERE, "vars_v1.csv"), index_col="rec")
R = pd.read_csv(os.path.join(HX9, "data", "records.csv"), index_col="rec")
for lab, cond in (("normal->normal", (R.hi10 == 0) & (R.prev_hi10 == 0)), ("normal, prev EC mean 0.7-1.0", (R.hi10 == 0) & (R.prev_ec_mean >= 0.7))):
    L = R[cond & (R.has_sg == 1)].copy(); L = L[[R.at[p, "has_sg"] == 1 for p in L.prev_rec]]
    L["dlny"] = np.log(L.ec_mean.values) - np.log(R.loc[L.prev_rec, "ec_mean"].values)
    L["dlnsg"] = np.log(L.sg_mean.values) - np.log(R.loc[L.prev_rec, "sg_mean"].values)
    L["dres"] = L.dlny - L.dlnsg
    g = pd.factorize(V.loc[L.index, "linkcomp"])[0]
    for v in ["st_mean", "pst_mean"]:
        x = V.loc[L.index, v].values - V.loc[L.prev_rec, v].values
        out = []
        for yv in ["dlny", "dlnsg", "dres"]:
            s = stats.linregress(x, L[yv].values)
            ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; bs = []
            for _ in range(2000):
                ss = RNG.choice(ug, len(ug), replace=True); ii = np.concatenate([idx[u] for u in ss]); bs.append(stats.linregress(x[ii], L[yv].values[ii]).slope)
            out.append(f"{yv} {s.slope:+.4f} CI[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] r {s.rvalue:+.2f}")
        print(f"{lab:30s} n {len(L):3d} d_{v:9s}: " + " | ".join(out))
