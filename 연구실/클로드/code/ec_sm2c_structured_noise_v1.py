# -*- coding: utf-8 -*-
"""SM2c — 구조적 교란(같은 센서) 반증 시뮬레이션: 깨끗한 F47 날 in_temp·in_hum 에 AR(1) 오프셋(ρ) + 흰잡음을 더해
거친 날 지표 묶음(K1 .951, K2 .541, K3 1.03, K4 −.725, 차분 자기상관 T .41·H .27)을 '동시에' 재현하는 설정이 있는지 격자 탐색 — 2026-10-08 연구실 클로드
판독(사전): 6개 지표 모두 거친 날 값의 ±25% 안(자기상관은 ±.1)에 드는 설정이 있으면 '같은 센서 + 구조적 교란'으로 충분 → 다른 센서 가설은 필요 없음."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
R = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(10)
exec(open(os.path.join(R, "ec_sm1b_noise_simulation_v1.py"), encoding="utf-8").read().split('print("거친 날 실제')[0].split('"""', 2)[2])
def ar(n, rho, s):
    e = rng.normal(0, s * np.sqrt(1 - rho ** 2), n); x = np.empty(n); x[0] = rng.normal(0, s)
    for i in range(1, n): x[i] = rho * x[i - 1] + e[i]
    return x
TGT = np.array([.951, .541, 1.032, -.725, .41, .27])
days = [g for d, g in TR.groupby("day") if d in clean and not g[["sub_temp", "in_temp", "in_hum"]].isna().any().any() and len(g) == 24]
best = []
for rho in (.0, .7, .9, .97):
    for sA in (.3, .6, 1.0, 1.5):
        for sW in (0, .3, .6):
            res = []
            for g in days:
                for _ in range(3):
                    res.append(metrics(g.sub_temp.values, g.in_temp.values + ar(24, rho, sA) + rng.normal(0, sW, 24), g.in_hum.values + 4 * (ar(24, rho, sA) + rng.normal(0, sW, 24))))
            r = np.median(np.array(res), axis=0); tol = np.array([.25 * abs(t) for t in TGT[:4]] + [.1, .1])
            ok = np.all(np.abs(r - TGT) <= tol); best.append((np.sum(np.abs(r - TGT) / tol), rho, sA, sW, r, ok))
best.sort(key=lambda t: t[0])
for sc, rho, sA, sW, r, ok in best[:6]:
    print(f"ρ {rho:.2f} σAR {sA:.1f} σW {sW:.1f}: K1 {r[0]:.3f} K2 {r[1]:.3f} K3 {r[2]:.2f} K4 {r[3]:+.3f} acT {r[4]:+.2f} acH {r[5]:+.2f} | 전부 허용범위 {ok}")
print("재현 설정 수:", sum(b[5] for b in best), "/", len(best))
