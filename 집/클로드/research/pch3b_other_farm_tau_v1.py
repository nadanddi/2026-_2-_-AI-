# -*- coding: utf-8 -*-
"""PCH3b (2026-10-16 집 클로드).  서술 보조(판정 없음), 실행 전 커밋.  PCH3 M1 의 '어제 밤 상태가 오늘 몇 시간 이어지나'(tau) 를 다른 온실(날짜 연속, 정답 있음)에서 독립적으로 잼.
온실마다 e = 배지 - 선형(실내 현재·하루 안 지연 1·2·3·6h·시각) 잔차; 연속한 날(d-1 -> d)마다 e(전날 23시) 와 e(오늘 h시) 의 회귀 기울기 b_h, b_h = b0 * exp(-h / tau) 최소제곱 격자.
F13·F47 은 기록이 이어붙임이라(1.3) 날짜 연속 아님 -> 제외.  출력: 온실별 tau, 중앙값·분위, 따라가는 속도(6.515)와의 관계는 서술만.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u pch3b_other_farm_tau_v1.py
"""
import os
import env  # noqa: F401
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "in_temp"]); Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"), usecols=["row_id", "sub_temp"])
D = X.merge(Y, on="row_id"); D["farm"] = D.row_id.str[:3]; D["day"] = D.row_id.str[4:7].astype(int); D["hour"] = D.row_id.str[8:10].astype(int)
D = D[~D.farm.isin(["F13", "F47"]) & D.sub_temp.lt(40) & D.in_temp.between(0, 45)].sort_values(["farm", "day", "hour"])
g = D.groupby(["farm", "day"])
for L in (1, 2, 3, 6): D[f"l{L}"] = g.in_temp.shift(L).fillna(D.in_temp)
D["hs"], D["hc"] = np.sin(2 * np.pi * D.hour / 24), np.cos(2 * np.pi * D.hour / 24); F = ["in_temp", "l1", "l2", "l3", "l6", "hs", "hc"]
TAUS = (.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 48, 1e9); out = []
for f, R in D.groupby("farm"):
    R = R.dropna(subset=F + ["sub_temp"]); R = R.assign(e=R.sub_temp - LinearRegression().fit(R[F], R.sub_temp).predict(R[F]))
    P = R.pivot_table(index="day", columns="hour", values="e"); prev = P.shift(1)[23] if 23 in P else None
    if prev is None: continue
    ok = (P.index.to_series().diff() == 1).values & prev.notna().values; b = []
    for h in range(24):
        if h not in P: b.append(np.nan); continue
        m = ok & P[h].notna().values
        b.append(float(np.polyfit(prev.values[m], P[h].values[m], 1)[0]) if m.sum() >= 30 else np.nan)
    b = np.array(b); hh = np.arange(24); k = np.isfinite(b)
    if k.sum() < 12: continue
    best = min(((b0, t, float(((b[k] - b0 * np.exp(-hh[k] / t)) ** 2).sum())) for b0 in np.round(np.arange(.1, 1.21, .05), 2) for t in TAUS), key=lambda x: x[2])
    out.append(dict(farm=f, pairs=int(ok.sum()), b0=best[0], tau=best[1], b_h0=round(b[0], 2), b_h6=round(b[6], 2), b_h12=round(b[12], 2)))
S = pd.DataFrame(out); pd.set_option("display.width", 200); print(S.to_string(index=False))
t = S.tau.replace(1e9, np.inf); print(f"\nfarms {len(S)} | tau median {t.median()} | quartiles {t.quantile(.25)} / {t.quantile(.75)} | b0 median {S.b0.median()} | PCH3 M1 (F13/F47 validators): s .9 tau 8")
