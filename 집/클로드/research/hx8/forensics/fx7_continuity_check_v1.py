# -*- coding: utf-8 -*-
# hx8/forensics fx7 v1: fx6 의 '다음날 묶음 = 외기 연속성 최소 비용' 선택이 학습에서 얼마나 맞나(앞날 쪽도 비교). 서술.
# 기준(진단): 진단 연결 c1 ≤ 1 인 학습 쌍 a→b. 묶음 = 외기 24h 동일 학습 기록. 정답 = b 가 고른 묶음 안에 있음.
# 여유(margin) = 2위 비용 / 1위 비용. fx6 의 F13_237 다음날 선택 여유 = .137/.039 = 3.5, 앞날 선택 여유 = .061/.053 = 1.15.
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import numpy as np, pandas as pd
from feat4 import X, IN_, AC, EC, ST, FARM, NUM, N, TRAIN, S3

rid = np.array([f"{FARM[i]}_{NUM[i]:03d}" for i in range(N)])
tr = np.where(TRAIN)[0]
sdO = np.nanstd(X[tr].reshape(-1, 4), 0)
key = {}
for i in tr: key.setdefault(np.round(X[i], 1).tobytes(), []).append(i)
B = list(key.values()); bof = {i: k for k, v in enumerate(B) for i in v}
def pe(i): return X[i, 23] + .5 * (X[i, 23] - X[i, 22])
def pb(i): return X[i, 0] - .5 * (X[i, 1] - X[i, 0])
PE = np.array([pe(v[0]) for v in B]); PB = np.array([pb(v[0]) for v in B])
C = np.nanmean(np.abs(PE[:, None, :] - PB[None, :, :]) / sdO, 2)   # C[x, y]: bundle x then bundle y
np.fill_diagonal(C, np.inf)
t = pd.read_pickle(S3 + r"\truth.pkl")
L = t[(t.c1 <= 1.0)]
rows = []
for _, r in L.iterrows():
    a, b = int(r.a1), int(r.b)
    if not (TRAIN[a] and TRAIN[b]): continue
    xa, xb = bof[a], bof[b]
    o = np.argsort(C[xa]); rows.append(dict(side="next", ok=o[0] == xb, margin=C[xa, o[1]] / max(C[xa, o[0]], 1e-9), seg2=NUM[b] >= 179 or NUM[a] >= 179))
    o = np.argsort(C[:, xb]); rows.append(dict(side="prev", ok=o[0] == xa, margin=C[o[1], xb] / max(C[o[0], xb], 1e-9), seg2=NUM[b] >= 179 or NUM[a] >= 179))
D = pd.DataFrame(rows)
D["mbin"] = pd.cut(D.margin, [0, 1.2, 1.5, 2, 3, np.inf])
pd.set_option("display.width", 200)
print("bundles", len(B), "pairs", len(D) // 2)
print(D.groupby(["side"]).ok.agg(["mean", "size"]).round(3).to_string())
print(D.groupby(["side", "mbin"], observed=True).ok.agg(["mean", "size"]).round(3).to_string())
print("2nd-segment-number pairs only:"); print(D[D.seg2].groupby(["side", "mbin"], observed=True).ok.agg(["mean", "size"]).round(3).to_string())
