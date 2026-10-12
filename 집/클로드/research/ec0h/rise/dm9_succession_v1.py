# DM9: 날짜 묶음 사이 '다음날' 관계를 외기 자정 연속으로 전부 계산 — 각 묶음의 최선 다음 묶음(연속 값·2등과 차이), 1차 순서와 비교
import sys; sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
from collections import defaultdict
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}" + ("" if TRAIN[i] else "*")
Xf = X.reshape(N, -1); par = list(range(N))
def fd(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for i in range(N):
    eq = (np.abs(Xf[i + 1:] - Xf[i]) < 1e-6).mean(1)
    for j in np.where(eq >= .9)[0] + i + 1: par[fd(j)] = fd(i)
comp = np.array([fd(i) for i in range(N)]); G = defaultdict(list)
for i in range(N): G[comp[i]].append(i)
cs = list(G); rep = {c: G[c][0] for c in cs}
lab = {c: min(G[c], key=lambda i: (NUM[i] >= 179, FARM[i] != "F13", NUM[i])) for c in cs}
name = {c: rid(lab[c]) for c in cs}
M = np.array([[np.nansum(np.abs(X[rep[a], 23] - X[rep[b], 0]) / sdO) if a != b else np.inf for b in cs] for a in cs])
# also hour 22->23 extrapolation check: use 2-hour slope continuity (23h + slope vs 0h)
pos = {c: (np.median([NUM[i] for i in G[c] if FARM[i] == "F13" and NUM[i] < 179]) if any(FARM[i] == "F13" and NUM[i] < 179 for i in G[c]) else np.nan) for c in cs}
order = sorted(cs, key=lambda c: (np.nan_to_num(pos[c], nan=1e9), name[c]))
print("group (label, pass-1 F13 pos) -> best next by outdoor continuity (value), 2nd (value)")
for a in order:
    k = cs.index(a); o = np.argsort(M[k]); b1, b2 = cs[o[0]], cs[o[1]]
    flag = "" if (np.isfinite(pos[a]) and np.isfinite(pos[b1]) and 0 < pos[b1] - pos[a] <= 4) else "  <-- not the next pass-1 date"
    print(f"  {name[a]:9s} @{pos[a]:6.1f} -> {name[b1]:9s} @{pos[b1]:6.1f} ({M[k,o[0]]:.2f}) | 2nd {name[b2]:9s} ({M[k,o[1]]:.2f}){flag}")
