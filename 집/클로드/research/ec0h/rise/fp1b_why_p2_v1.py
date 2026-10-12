# -*- coding: utf-8 -*-
"""FP1b (FP1 사후 서술): 2차 검증기(P2LOO·EL1)에서 왜 나빠졌나 — 지문 문 G 가 참인 고EC 기록별 하루 평균 정답 vs 예측, 검증기별."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp1_fingerprint_level_fix_v1 as F
import numpy as np, pandas as pd
cells = F.load()
out = []
for key, d in cells.items():
    if key[1] not in (47, 101): continue
    g = d[d.G].groupby("rec").agg(y=("y", "mean"), p=("p", "mean"), n=("G", "size"))
    g["validator"] = key[0]; out.append(g.reset_index())
T = pd.concat(out); T["hi"] = T.rec.map(F.ymean) >= 1
pd.set_option("display.width", 200)
print(T[T.hi].pivot_table(index="rec", columns="validator", values="p").join(T[T.hi].groupby("rec").y.first()).round(2).to_string())
print("\nmean ratio y/p on gated high records by validator:", T[T.hi].groupby("validator").apply(lambda g: round(float((g.y / g.p).mean()), 3)).to_dict())
