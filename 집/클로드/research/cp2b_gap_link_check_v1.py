# -*- coding: utf-8 -*-
"""CP2B (2026-10-14 집 클로드).  CP2 빈 날 잇기(98개)가 같은 동인지 서술 점검(학습 정답으로 — 분석용): 이은 쌍 vs 같은 구간·같은 간격 무작위 쌍의 하루 평균 EC 차, 고EC 상태 일치율."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
T = pd.read_csv(os.path.join(HERE, "cp2_tracks_v1.csv")); D = dict(zip(T.rec, T.dm)); DAY = dict(zip(T.rec, T.day)); SEG = dict(zip(T.rec, T.seg))
rng = np.random.default_rng(1); out = []
for lt in ("day1", "gap"):
    g = T[(T.link_type == lt) & T.prev.notna()]
    d_link = [abs(D[b] - D[a]) for b, a in zip(g.rec, g.prev) if np.isfinite(D[b]) and np.isfinite(D.get(a, np.nan))]
    s_link = [((D[b] >= 1) == (D[a] >= 1)) for b, a in zip(g.rec, g.prev) if np.isfinite(D[b]) and np.isfinite(D.get(a, np.nan))]
    d_rand, s_rand = [], []
    for b, gap in zip(g.rec, g.gap_days):
        pool = [r for r in T[T.train].rec if SEG[r] == SEG[b] and DAY[b] - DAY[r] == gap and r != b and np.isfinite(D[r])]
        if pool and np.isfinite(D[b]):
            r = pool[rng.integers(len(pool))]; d_rand.append(abs(D[b] - D[r])); s_rand.append((D[b] >= 1) == (D[r] >= 1))
    out.append(dict(link=lt, n=len(d_link), mean_abs_dEC_link=np.mean(d_link), mean_abs_dEC_random_same_gap=np.mean(d_rand), high_state_agree_link=np.mean(s_link), high_state_agree_random=np.mean(s_rand)))
print(pd.DataFrame(out).round(3).to_string(index=False))
