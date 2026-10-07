# -*- coding: utf-8 -*-
"""ND2 v2 — 비평가 지적 반영 (진단 전용) — 2026-10-06 연구실 클로드
v1 문제: 정답 기준 일반 날 정의(국면 오분류 가림), base RMSE 식 오류, 표본 내 1차식, 하위집단 불확실성 없음, '압축 없음' 오판.
v2:
 P1 2차 46일을 '예측 하루평균(sg) ≥ .9' × '정답 하루평균 ≥ 1' 2×2 국면으로 나눠 SSE 몫 (배치별, 실제 시간 RMSE)
 P2 base/sg 실제 시간 RMSE
 P3 예측 기준 일반 날(sg 하루평균 < .9)에서 보정 기울기(정답 ~ 예측): 배치 간 교차
    (DIAG10 에서 a,b 적합 → DIAG10y·EL1 적용, 그 반대도) — 하루 수준만 바꿈
 P4 예측 기준 일반 날 SSE 몫의 하위집단(동 B, 밀폐) 날 단위 부트스트랩 95% 구간, F13 233 제외 민감도
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd

R = os.path.dirname(os.path.abspath(__file__))
H = os.path.join(env.ROOT, u"집", u"클로드", "research", "local")
S = pd.read_csv(os.path.join(H, "ec3_SG2_all.csv"))
S["sg"] = S[["sg_23", "sg_808", "sg_9090"]].mean(axis=1); S["bs"] = S[["base_23", "base_808", "base_9090"]].mean(axis=1)
S["dsg"] = S.groupby(["validator", "farm", "day"]).sg.transform("mean"); S["dy"] = S.groupby(["validator", "farm", "day"]).sub_ec.transform("mean")
info = pd.read_csv(os.path.join(R, "..", "results", "ec_err03_sources_days_v2.csv"))[["farm", "day", "dong", "sealed"]]
S = S.merge(info, on=["farm", "day"], how="left")
rm = lambda x: np.sqrt(np.mean(np.square(x)))

print("=== P2 실제 시간 RMSE (2차 46일) ===")
for v, g in S.groupby("validator"):
    print("  %-8s base %.4f → SG2 %.4f (%+.1f%%)" % (v, rm(g.bs - g.sub_ec), rm(g.sg - g.sub_ec), 100 * (rm(g.sg - g.sub_ec) / rm(g.bs - g.sub_ec) - 1)))

print("\n=== P1 국면 2×2 (예측 하루평균 ≥ .9 = '높게 예측', 정답 하루평균 ≥ 1 = '실제 고EC') — 2차 SSE 몫 ===")
for v, g in S.groupby("validator"):
    g = g.assign(ph=g.dsg >= .9, th=g.dy >= 1, se=(g.sg - g.sub_ec) ** 2)
    T = g.se.sum(); out = []
    for ph in (False, True):
        for th in (False, True):
            m = (g.ph == ph) & (g.th == th)
            nm = {(False, False): "일반 예측·실제 일반", (False, True): "일반 예측·실제 고EC(놓침)", (True, False): "높게 예측·실제 일반(오탐)", (True, True): "높게 예측·실제 고EC"}[(ph, th)]
            out.append("%s %d일 %.0f%%" % (nm, g[m].groupby(["farm", "day"]).ngroups, 100 * g.se[m].sum() / T))
    print("  %-8s " % v + " | ".join(out))

print("\n=== P3 예측 기준 일반 날(sg 하루평균 < .9) 하루 수준 보정 기울기, 배치 간 교차 ===")
D = S.groupby(["validator", "farm", "day"]).agg(y=("dy", "first"), p=("dsg", "first")).reset_index()
N = D[D.p < .9]
fits = {}
for v, q in N.groupby("validator"):
    b, a = np.polyfit(q.p, q.y, 1); fits[v] = (a, b)
    print("  %-8s 일 %d, 정답~예측 기울기 %.2f 절편 %+.3f (예측 sd %.3f, 정답 sd %.3f)" % (v, len(q), b, a, q.p.std(), q.y.std()))
for src in fits:
    a, b = fits[src]
    res = []
    for v, g in S.groupby("validator"):
        if v == src:
            continue
        g = g[g.dsg < .9]
        newlvl = a + b * g.dsg
        pred = g.sg - g.dsg + newlvl
        res.append("%s %+.1f%%" % (v, 100 * (rm(pred - g.sub_ec) / rm(g.sg - g.sub_ec) - 1)))
    print("  %s 에서 적합 → 적용: %s (예측 기준 일반 날 시간 RMSE 변화)" % (src, ", ".join(res)))

print("\n=== P4 예측 기준 일반 날 SSE 몫 — 하위집단, 날 단위 부트스트랩 95%, F13 233 제외 ===")
rng = np.random.default_rng(0)
for v, g in S[S.dsg < .9].groupby("validator"):
    d = g.assign(se=(g.sg - g.sub_ec) ** 2).groupby(["farm", "day"]).agg(se=("se", "sum"), dong=("dong", "first"), sealed=("sealed", "first")).reset_index()
    line = []
    for nm, m in (("동B", d.dong == "B"), ("밀폐", d.sealed == True)):
        share = d.se[m].sum() / d.se.sum()
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(d), len(d)); dd = d.iloc[i]; mm = m.values[i]
            bs.append(dd.se[mm].sum() / dd.se.sum())
        ex = d[~((d.farm == "F13") & (d.day == 233))]; mex = (ex.dong == "B") if nm == "동B" else (ex.sealed == True)
        line.append("%s %d/%d일 몫 %.0f%% [%.0f, %.0f] (F13 233 제외 %.0f%%)" % (nm, m.sum(), len(d), 100 * share, 100 * np.quantile(bs, .025), 100 * np.quantile(bs, .975), 100 * ex.se[mex].sum() / ex.se.sum()))
    top = d.sort_values("se", ascending=False)
    print("  %-8s %s | 상위 3일: %s" % (v, " | ".join(line), ", ".join("%s_%d %.0f%%" % (f, dd, 100 * s / d.se.sum()) for f, dd, s in zip(top.farm[:3], top.day[:3], top.se[:3]))))
