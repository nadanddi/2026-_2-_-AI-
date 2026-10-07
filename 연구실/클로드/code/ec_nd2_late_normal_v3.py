# -*- coding: utf-8 -*-
"""ND2 v3 — 2차 비평 반영 (진단 전용) — 2026-10-06 연구실 클로드
 L1 국면별 '레버' = 그 국면 날의 하루 수준만 정답으로 바꿨을 때(오라클) 2차 시간 RMSE 감소량, 문턱 .85/.9/.95/1.0
 L2 고EC 날(정답 ≥1) 뺀 2차 RMSE (리더보드 비교용)
 L3 P3 보정 기울기를 정답 ≥1 날 뺀 예측 일반 날로 다시
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
H = os.path.join(env.ROOT, u"집", u"클로드", "research", "local")
S = pd.read_csv(os.path.join(H, "ec3_SG2_all.csv"))
S["sg"] = S[["sg_23", "sg_808", "sg_9090"]].mean(axis=1)
S["dsg"] = S.groupby(["validator", "farm", "day"]).sg.transform("mean"); S["dy"] = S.groupby(["validator", "farm", "day"]).sub_ec.transform("mean")
rm = lambda x: np.sqrt(np.mean(np.square(x)))
print("L1 국면별 오라클 하루 수준 보정 시 2차 시간 RMSE 감소 (SG2 시드평균 기준)")
for v, g in S.groupby("validator"):
    base = rm(g.sg - g.sub_ec)
    print(" %s 기준 %.4f" % (v, base))
    for t in (.85, .9, .95, 1.0):
        cells = []
        for nm, m in (("일반예측·일반", (g.dsg < t) & (g.dy < 1)), ("일반예측·고EC(놓침)", (g.dsg < t) & (g.dy >= 1)),
                      ("높게예측·일반(오탐)", (g.dsg >= t) & (g.dy < 1)), ("높게예측·고EC", (g.dsg >= t) & (g.dy >= 1))):
            p = np.where(m, g.sg - g.dsg + g.dy, g.sg)
            cells.append("%s %d일 %+.1f%%" % (nm, g[m].groupby(["farm", "day"]).ngroups, 100 * (rm(p - g.sub_ec) / base - 1)))
        print("   문턱 %.2f: %s" % (t, " | ".join(cells)))
print("\nL2 고EC 날(정답≥1) 뺀 2차 RMSE: " + ", ".join("%s %.4f(41일)" % (v, rm(g[g.dy < 1].sg - g[g.dy < 1].sub_ec)) for v, g in S.groupby("validator")))
print("\nL3 예측 일반(<.9)·정답 일반(<1) 날만 정답~예측 기울기: " + ", ".join(
    "%s %.2f(%d일)" % (v, np.polyfit(d.p, d.y, 1)[0], len(d)) for v, d in
    S[(S.dsg < .9) & (S.dy < 1)].groupby(["validator", "farm", "day"]).agg(p=("dsg", "first"), y=("dy", "first")).reset_index().groupby("validator")))
