# -*- coding: utf-8 -*-
# vl2: (1) 확실 연결(cert) 쌍의 밤 사이 점프 |EC0(b)-EC23(a)| 분포 -> F47_230(23시 .456) 이 F13_128/129(0시 .72) 로 이어질 확률 감 잡기
#      (2) 하루 평균 vs 다음날 0시 차이 분포(특히 다음날 0시 < .9 일 때 당일 평균 >= 1.1 인 비율) -> F13_237 수준 결론의 견고성
#      (3) 순열 기준선: E1a 류 '다음날 묶음 남은 기록 0시' 를 무작위 같은 날짜 기록으로 바꿨을 때의 하루 RMSE
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import numpy as np, pandas as pd
from feat4 import X, EC, FARM, NUM, N, TRAIN, S3
rid = np.array([f"{FARM[i]}_{NUM[i]:03d}" for i in range(N)])
t = pd.read_pickle(S3 + r"\truth.pkl"); C = t[t.cert & t.a1.map(lambda a: TRAIN[a]) & t.b.map(lambda b: TRAIN[b])]
a = C.a1.values.astype(int); b = C.b.values.astype(int)
j = np.abs(EC[b, 0] - EC[a, 23]); print("cert links n", len(C), "| overnight |jump| q50/q90/q99/max", np.round(np.quantile(j, [.5, .9, .99, 1]), 3), "| share >= .25:", (j >= .25).mean())
dm = EC[a].mean(1) - EC[b, 0]
print("day mean(a) - EC0(b): q01/q05/q50/q95/q99", np.round(np.quantile(dm, [.01, .05, .5, .95, .99]), 3))
m = EC[b, 0] < .9; print("given EC0(b)<.9: n", int(m.sum()), "| share mean(a)>=1.1:", round(float((EC[a[m]].mean(1) >= 1.1).mean()), 3), "| max mean(a):", round(float(EC[a[m]].mean(1).max()), 3))
m2 = m & (EC[a, 0] < .9); print("given EC0(b)<.9 & EC0(a)<.9: n", int(m2.sum()), "| share mean(a)>=1.1:", round(float((EC[a[m2]].mean(1) >= 1.1).mean()), 3))
# all cert links whose 'a' dropped by >=.3 between own 0h and next-day 0h (intraday flush)
print("share |EC0(b)-mean(a)| > .3:", round(float((np.abs(dm) > .3).mean()), 3))
