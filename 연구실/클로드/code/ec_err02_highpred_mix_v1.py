# -*- coding: utf-8 -*-
"""고EC로 예측한 날 안의 섞임 진단 (진단 전용, 채택 판정 아님) — 2026-10-06 연구실 클로드

질문: 고EC 크기를 키우면 '고EC로 예측했지만 실제 일반인 날'도 같이 커져서 손해인가?
입력: ec_err01 의 날 단위 표(DIAG10, 계절 R3 + DP1 시드 평균).
1) 예측 하루 평균 문턱별: 진짜 고EC / 일반 날 구성
2) 문턱 이상 날의 수준을 배율 k로 키울 때 득실 (전부 vs 진짜 고EC만 = 오라클 분리)
3) 문턱 이상 날 안에서 예측이 실제 크기를 구별하는지 (순위 상관)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd

R = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(R, "..", "results", "ec_err01_days_DIAG10_v1.csv"))
T = d.sse.sum(); N = d.n.sum()
d["hi"] = d.y >= 1
print("기준 RMSE %.4f, 고EC %d일" % (np.sqrt(T / N), d.hi.sum()))

print("\n[1] 예측 문턱별 구성")
for t in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
    m = d.p >= t
    print("p>=%.1f: %3d일 = 진짜고EC %2d + 일반 %2d (일반 실제평균 %.2f) | 놓친 고EC(p<t) %d"
          % (t, m.sum(), (m & d.hi).sum(), (m & ~d.hi).sum(), d.y[m & ~d.hi].mean(), (~m & d.hi).sum()))


def sse_after(mask, k):
    # 해당 날의 하루 수준만 바꿈: 모양 오차는 그대로, 수준 편향 = k*p - y
    nb = k * d.p - d.y
    lvl = np.where(mask, d.n * nb ** 2, d.sse_level)
    return (d.sse_shape + lvl).sum()

print("\n[2] 예측≥.9 날의 하루 수준을 k배 (전체 RMSE 변화 %)")
m = d.p >= 0.9
print("   k   | 전부 키움 | 진짜고EC만 | 그중 일반날 손해(SSE몫%)")
for k in (1.1, 1.2, 1.3, 1.4, 1.6):
    a = sse_after(m, k); b = sse_after(m & d.hi, k)
    loss = (sse_after(m & ~d.hi, k) - T) / T * 100
    print("  %.1f  | %+6.1f%%  | %+6.1f%%   | %+.1f%%" % (k, 100 * (np.sqrt(a / N) / np.sqrt(T / N) - 1),
                                                       100 * (np.sqrt(b / N) / np.sqrt(T / N) - 1), loss))

print("\n[3] 예측≥.9 날 안에서 예측이 실제 크기를 구별하나")
s = d[m]
print("  Spearman(p, y) = %.2f, n=%d" % (s[["p", "y"]].corr("spearman").iloc[0, 1], len(s)))
s = s.assign(bin=pd.qcut(s.p, 3, labels=["예측 하", "예측 중", "예측 상"]))
print(s.groupby("bin", observed=True).agg(날=("y", "size"), 예측평균=("p", "mean"), 실제평균=("y", "mean"),
                                           실제최소=("y", "min"), 실제최대=("y", "max"), 진짜고EC=("hi", "sum")).round(2).to_string())
print("\n  진짜 고EC 날만: Spearman(p, y) = %.2f, n=%d, 실제 %.2f~%.2f / 예측 %.2f~%.2f"
      % (d[d.hi][["p", "y"]].corr("spearman").iloc[0, 1], d.hi.sum(), d.y[d.hi].min(), d.y[d.hi].max(), d.p[d.hi].min(), d.p[d.hi].max()))
