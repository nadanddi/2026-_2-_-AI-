# -*- coding: utf-8 -*-
"""DHM4 (2026-10-16 집 클로드; 사용자 '하루 수준을 못 맞췄다는 거지, 맞춘 날들과 조건이 완전히 다른 건 아닌가?').  서술(판정 없음), 실행 전 커밋.  온도.
DHM2 자료(어제 정보 없는 기록, 19회차 구성 최종 예측 하루 평균 오차 m; 기록 번호 제외 특징).  기록 단위로 묶음(배치 평균).
(1) 크게 틀린 기록(|m| >= .8)마다 표준화 특징 공간에서 가장 가까운 '맞힌 기록'(|m| <= .2) 거리 vs 맞힌 기록끼리 최근접 거리 분포(같은 날씨 조건이 흔한지)
(2) 크게 틀린 기록과 그 최근접 맞힌 기록을 특징 나란히 + 두 기록 하루 평균 오차
(3) 조건 분포 겹침: 특징마다 틀린 기록 값이 맞힌 기록 분포의 몇 % 분위에 있나(5~95% 안이면 '흔한 조건')
Run:  cd 집/클로드/research && PYTHONPATH="" python -u dhm4_miss_vs_nearest_hit_v1.py
"""
import os
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "dhm2_derived_features_v1.py"), encoding="utf-8").read(); exec(src[src.index("R = pd.read_csv"):src.index("pd.set_option")])
BASE = ["out_max", "out_min", "night_gap", "heat_on", "heat", "co2", "t_max", "t_min", "t_sd", "vent", "thermal", "hum", "f47"]
G = D.groupby("rec").agg(m=("m", "mean"), **{f: (f, "first") for f in BASE}).reset_index()
Z = (G[BASE] - G[BASE].mean()) / G[BASE].std(); Z = Z.fillna(0).values
hit = (G.m.abs() <= .2).values; miss = (G.m.abs() >= .8).values
dist = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(dist, np.inf)
hh = dist[np.ix_(hit, hit)].min(1); mh = dist[np.ix_(miss, hit)]; mn = mh.min(1)
print(f"records: hit {hit.sum()} | miss {miss.sum()} | mid {(~hit & ~miss).sum()}")
print(f"(1) nearest-hit distance: miss->hit median {np.median(mn):.2f} (q25 {np.quantile(mn, .25):.2f}, q75 {np.quantile(mn, .75):.2f}) | hit->hit median {np.median(hh):.2f} (q75 {np.quantile(hh, .75):.2f}, q90 {np.quantile(hh, .9):.2f})")
print(f"    share of miss records whose nearest hit is closer than the hit->hit median: {(mn <= np.median(hh)).mean():.2f} | closer than hit->hit q90: {(mn <= np.quantile(hh, .9)).mean():.2f}")
pd.set_option("display.width", 230)
H = G[hit].reset_index(drop=True); Mi = G[miss].reset_index(drop=True); order = np.argsort(-Mi.m.abs().values)
rows = []
for i in order:
    j = int(np.argmin(mh[i])); a, b = Mi.iloc[i], H.iloc[j]
    rows.append(dict(miss=a.rec, miss_m=round(a.m, 2), hit=b.rec, hit_m=round(b.m, 2), dist=round(mh[i, j], 2), **{f"{f}": f"{a[f]:.1f}|{b[f]:.1f}" for f in ["out_max", "out_min", "night_gap", "heat_on", "t_max", "t_min", "thermal"]}))
print("(2) miss record vs its nearest hit record (cells: miss|hit)"); print(pd.DataFrame(rows).to_string(index=False))
rows = []
for f in BASE:
    hv = G[hit][f].dropna().values; pct = [100 * (hv < v).mean() for v in G[miss][f].values]
    rows.append(dict(feat=f, miss_within_5_95=np.mean([(5 <= p <= 95) for p in pct]), miss_pct_median=np.median(pct)))
print("(3) where miss records sit inside the hit distribution"); print(pd.DataFrame(rows).round(2).to_string(index=False))
