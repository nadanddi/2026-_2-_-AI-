# -*- coding: utf-8 -*-
"""SM2b — SM2 보완: (1) Fisher 표 방향 버그 정정, (2) 바뀜 신호의 기계적 원인 대조: 자기 in_temp 를 흰잡음으로 망가뜨린 깨끗한 날에서도
'다른 기록이 더 잘 맞음'이 같은 비율로 생기나, (3) 거친 날 in_temp 가 같은 날짜 다른 기록 in_temp 의 복사/근사인가(시간별 상관·평균 절대차) — 2026-10-08 연구실 클로드"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
R = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(9)
exec(open(os.path.join(R, "ec_sm2_cross_record_match_v1.py"), encoding="utf-8").read().split("K = pd.read_csv")[0].split('"""', 2)[2])
K = pd.read_csv(os.path.join(R, "..", "results", "ec_sm1_days_v1.csv")).set_index(["farm", "day"])
M = pd.read_csv(os.path.join(R, "..", "results", "ec_sm2_cross_v1.csv")); H = M[M.nnb > 0].copy(); H["swap"] = H.best > H.own + .02
a, b = int((H.rough & H.swap).sum()), int((H.rough & ~H.swap).sum()); c, d = int((~H.rough & H.swap).sum()), int((~H.rough & ~H.swap).sum())
print(f"(1) 바뀜: 거친 {a}/{a+b} vs 나머지 {c}/{c+d}; Fisher 단측 p {fisher_exact([[a, b], [c, d]], alternative='greater')[1]:.2e}")
# (2) 기계적 대조: 깨끗한 날 자기 in_temp 에 σ 잡음 → own K1 이 거친 날 수준(중앙 .95)으로 떨어질 때 바뀜 비율
for sg in (.5, 1.0, 1.5):
    sw, own_l = [], []
    for i, k in enumerate(keys):
        if k not in ST or k not in K.index or K.loc[k, "rough"]: continue
        nb = [keys[j] for j in np.where(Dm[i] <= .05)[0] if keys[j] != k]
        if not nb: continue
        it = IT[k] + rng.normal(0, sg, 24); o = k1(ST[k], it); bst = max(k1(ST[k], IT[r]) for r in nb)
        sw.append(bst > o + .02); own_l.append(o)
    print(f"(2) 깨끗한 날 + 잡음 σ {sg}: own K1 중앙 {np.nanmedian(own_l):.3f}, 바뀜 비율 {np.mean(sw):.2f}")
# (3) 복사 검사: in_temp 끼리 같은 날짜 다른 기록과의 최대 상관·최소 평균 절대차
rows = []
for i, k in enumerate(keys):
    if k not in K.index: continue
    nb = [keys[j] for j in np.where(Dm[i] <= .05)[0] if keys[j] != k]
    if not nb: continue
    x = IT[k]; cs = [(r, np.corrcoef(x, IT[r])[0, 1], np.nanmean(np.abs(x - IT[r]))) for r in nb if not np.isnan(IT[r]).any() and not np.isnan(x).any()]
    if not cs: continue
    bc = max(cs, key=lambda t: t[1]); bm = min(cs, key=lambda t: t[2])
    rows.append((k[0], k[1], bool(K.loc[k, "rough"]), bc[1], bm[2]))
C = pd.DataFrame(rows, columns=["farm", "day", "rough", "maxcorr", "minmad"])
print("(3) in_temp 끼리: 최대 상관 중앙 거친 %.3f / 나머지 %.3f; 최소 평균절대차 중앙 거친 %.2f / 나머지 %.2f℃; 평균절대차<.2℃(복사 수준) 거친 %d / 나머지 %d" % (
    C[C.rough].maxcorr.median(), C[~C.rough].maxcorr.median(), C[C.rough].minmad.median(), C[~C.rough].minmad.median(), int((C.rough & (C.minmad < .2)).sum()), int((~C.rough & (C.minmad < .2)).sum())))
