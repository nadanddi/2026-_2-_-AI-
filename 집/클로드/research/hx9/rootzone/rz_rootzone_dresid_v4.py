# -*- coding: utf-8 -*-
"""HX9 근권 온도 가족 — v4 (2026-10-13 집 클로드, 사후 점검).
질문: 고EC 사슬 안 연속일 EC 변화가 Δ배지온과 함께 움직이는 것(v2)을 현재 모델이 이미 따라가는가?
 -> 모델 잔차의 연속일 변화 Δ(ln y − ln sg) 와 Δst / Δpst 의 관계. 모델이 이미 따라가면 r ≈ 0.
 또 모델 예측 자체의 연속일 변화 Δln sg 와 Δst 의 기울기(모델이 온도에 얼마나 반응하나)를 실제(Δln y)와 비교.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import env  # noqa: F401
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); HX9 = os.path.dirname(HERE)
RNG = np.random.default_rng(4)
V = pd.read_csv(os.path.join(HERE, "vars_v1.csv"), index_col="rec")
R = pd.read_csv(os.path.join(HX9, "data", "records.csv"), index_col="rec")
L = R[(R.hi10 == 1) & (R.prev_hi10 == 1) & (R.has_sg == 1)].copy()
L = L[[R.at[p, "has_sg"] == 1 for p in L.prev_rec]]
L["dlny"] = np.log(L.ec_mean.values) - np.log(R.loc[L.prev_rec, "ec_mean"].values)
L["dlnsg"] = np.log(L.sg_mean.values) - np.log(R.loc[L.prev_rec, "sg_mean"].values)
L["dres"] = L.dlny - L.dlnsg
for v in ["st_mean", "pst_mean", "pst_min", "in_min"]:
    L["d_" + v] = V.loc[L.index, v].values - V.loc[L.prev_rec, v].values
g = L.ep_id.values.astype(int)
def boot(x, y, B=4000):
    ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; out = []
    for _ in range(B):
        s = RNG.choice(ug, len(ug), replace=True); ii = np.concatenate([idx[u] for u in s])
        if np.std(x[ii]) > 0: out.append(stats.linregress(x[ii], y[ii]).slope)
    return np.percentile(out, [2.5, 97.5])
print(f"hi10->hi10 links with OOF on both ends: n {len(L)} (chains {len(np.unique(g))}, same-farm {int(L.prev_same_farm.sum())})")
for v in ["d_st_mean", "d_pst_mean", "d_pst_min", "d_in_min"]:
    for yv in ["dlny", "dlnsg", "dres"]:
        x = L[v].values; y = L[yv].values; s = stats.linregress(x, y); ci = boot(x, y)
        sf = L.prev_same_farm.values == 1
        print(f"  {yv:6s} ~ {v:11s}: slope {s.slope:+.4f}/C CI[{ci[0]:+.4f},{ci[1]:+.4f}] r {s.rvalue:+.2f} | same-farm r {np.corrcoef(x[sf], y[sf])[0,1]:+.2f} "
              f"(n {sf.sum()}), cross r {np.corrcoef(x[~sf], y[~sf])[0,1]:+.2f} (n {(~sf).sum()})")
print(f"SD dlny {L.dlny.std():.3f}, SD dlnsg {L.dlnsg.std():.3f}, SD dres {L.dres.std():.3f}, corr(dlny, dlnsg) {np.corrcoef(L.dlny, L.dlnsg)[0,1]:+.2f}")
