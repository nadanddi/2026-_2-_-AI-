# -*- coding: utf-8 -*-
"""DM10: 확실 연결(정답 EC·배지온 자정 연속, truth cert)만으로 날짜 묶음 사이 '다음날' 그래프를 만들고,
1차 F13 번호 순서의 이웃 묶음들이 실제로 +1일로 이어지는지, 순수 2차 묶음은 어디로 들어가는지(1차의 빈 날짜를 채우는지) 확인."""
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
lab = {c: min(G[c], key=lambda i: (NUM[i] >= 179, FARM[i] != "F13", NUM[i])) for c in G}
nm = lambda c: rid(lab[c])
pos = {c: (np.median([NUM[i] for i in G[c] if FARM[i] == "F13" and NUM[i] < 179]) if any(FARM[i] == "F13" and NUM[i] < 179 for i in G[c]) else np.nan) for c in G}
TL = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TL[TL.cert]
nxt = defaultdict(set); prv = defaultdict(set)
for a, b in zip(TLc.a1, TLc.b):
    ca, cb = comp[int(a)], comp[int(b)]
    if ca != cb: nxt[ca].add(cb); prv[cb].add(ca)
ranked = sorted([c for c in G if np.isfinite(pos[c])], key=lambda c: pos[c])
print("pass-1 date groups in F13 order; '=>' certain next-day groups (R = rank in pass-1 order, P = pure pass-2 group)")
R = {c: k for k, c in enumerate(ranked)}
for c in ranked:
    tg = [("R%d" % R[d]) if d in R else "P:" + nm(d) for d in nxt[c]]
    exp = R[c] + 1
    mark = "" if not tg or ("R%d" % exp) in tg else "   <== next-day is NOT the next pass-1 group"
    print(f"  R{R[c]:3d} {nm(c):9s} @{pos[c]:6.1f} => {tg}{mark}")
pure = [c for c in G if c not in R]
print("\npure pass-2 groups: certain prev / next")
for c in sorted(pure, key=lambda c: nm(c)):
    print(f"  {nm(c):9s} members {sorted(rid(i) for i in G[c])} | prev {[('R%d' % R[d]) if d in R else 'P:' + nm(d) for d in prv[c]]} | next {[('R%d' % R[d]) if d in R else 'P:' + nm(d) for d in nxt[c]]}")
