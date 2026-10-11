# -*- coding: utf-8 -*-
"""RZ4 (RZ3 비평 (d) 후속): 이 센서의 EC 온도계수 γ 추정. 기록 안(동·날 고정)에서 ln EC 를 배지온에 회귀(기록 고정효과),
시간대별(밤 0~6·20~23 / 낮 7~19), 농장별, 1·2차별. γ≈.02 면 EC25 보정이 맞고, ≈0 이면 센서가 이미 보정한 값."""
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
TRi = np.where(TRAIN)[0]
def gam(I, hrs):
    L = np.log(np.clip(EC[np.ix_(I, hrs)], 1e-3, None)); T = ST[np.ix_(I, hrs)]
    L = L - np.nanmean(L, 1, keepdims=True); T = T - np.nanmean(T, 1, keepdims=True)
    m = np.isfinite(L) & np.isfinite(T); return (L[m] * T[m]).sum() / (T[m] ** 2).sum(), m.sum()
night = list(range(0, 7)) + list(range(20, 24)); day = list(range(7, 20)); late = list(range(19, 24)); allh = list(range(24))
for f in ("F13", "F47"):
    for p2 in (False, True):
        I = TRi[(FARM[TRi] == f) & ((NUM[TRi] >= 179) == p2)]
        print(f, "pass-2" if p2 else "pass-1", "n", len(I), " | ".join("%s γ %+.4f" % (nm, gam(I, h)[0]) for nm, h in (("all", allh), ("night", night), ("day", day), ("19-23h", late))))
# per-record gamma on night hours, distribution
g = []
for i in TRi:
    T = ST[i, night]; L = np.log(np.clip(EC[i, night], 1e-3, None))
    if np.nanstd(T) > .5: g.append(np.polyfit(T, L, 1)[0])
g = np.array(g); print("per-record night γ (records with sd(T)>.5): n %d median %+.4f IQR %+.4f..%+.4f" % (len(g), np.median(g), *np.quantile(g, [.25, .75])))
