# -*- coding: utf-8 -*-
"""verify_leakage-stats vls5 (서술 대조만, 특징·선택·계수에 쓰지 않음): 평가 사슬 날짜 정렬 확인.
0시 외기 4변수가 같은 기록(학습+평가)을 찾아 F13_231..237 / F47_219..231 / F13_217..225 의 같은 날짜 무리를 나열."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import env
from feat4 import X, FARM, NUM, N, TRAIN
import numpy as np
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
key = {i: tuple(np.round(X[i, 0, :], 4)) for i in range(N)}
groups = {}
for i in range(N): groups.setdefault(key[i], []).append(i)
recs = [f"F13_{n:03d}" for n in range(215, 240)] + [f"F47_{n:03d}" for n in range(213, 234)]
seen = set()
for r in recs:
    if r not in idx: continue
    g = groups[key[idx[r]]]
    if tuple(g) in seen: continue
    seen.add(tuple(g))
    print("  same 0h weather:", ", ".join(f"{rid(j)}{'(tr)' if TRAIN[j] else '(TE)'}" for j in g))
