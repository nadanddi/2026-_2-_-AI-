# -*- coding: utf-8 -*-
"""DM2 (2026-10-12 집 클로드; 사용자: "상대일차이긴 하지만 날짜가 조금 섞여 있는 것 같은 경우를 더 찾아줘").  서술(구조 지도, 예측 특징 아님).
1) 날짜 묶음: 460 기록(학습+평가) 중 외기 24h×4 가 90% 이상 같은 기록끼리 묶음.
2) 달력 번호: 학습 정답 확실 연결(truth cert, a→b 는 다음날)로 묶음 사이 +1 관계를 세우고 BFS 로 상대 날짜(조각별).
   추가로 F13 1차 번호 척도(묶음 안 F13 1차 번호 중앙값)를 '달력 위치'로 씀(1차는 번호≈날짜).
3) 섞임 찾기: (a) 한 날짜 묶음 안에 번호가 멀리 떨어진 기록(1차·2차가 같은 날짜) — 번호 차 > 10;
   (b) 2차 기록의 달력 위치(같은 묶음의 1차 기록 기준)와 번호 순서가 어긋남; (c) 같은 ID 안에서 번호가 늘 때 날짜가 거꾸로 가는 곳.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
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
comp = np.array([fd(i) for i in range(N)])
G = defaultdict(list)
for i in range(N): G[comp[i]].append(i)
print("date groups:", len(G), "| size distribution:", pd.Series([len(v) for v in G.values()]).value_counts().sort_index().to_dict())
# calendar position: F13 pass-1 median number in group; else F47 pass-1 mapped by calendar() offset map; else NaN
vis = np.ones(N, bool); gpos, mp = calendar(vis, {i: comp[i] for i in range(N)})
pos = {}
for c, v in G.items():
    a = [NUM[i] for i in v if FARM[i] == "F13" and NUM[i] < 179]
    b = [mp(i) for i in v if FARM[i] == "F47" and NUM[i] < 179]
    pos[c] = float(np.median(a)) if a else (float(np.median(b)) if b else np.nan)
# relative day by certain links
TL = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TL[TL.cert]
adj = defaultdict(list)
for a, b in zip(TLc.a1, TLc.b):
    ca, cb = comp[int(a)], comp[int(b)]
    if ca != cb: adj[ca].append((cb, 1)); adj[cb].append((ca, -1))
rel = {}; piece = {}; k = 0; bad = 0
for c0 in G:
    if c0 in rel: continue
    k += 1; rel[c0] = 0; piece[c0] = k; Q = [c0]
    while Q:
        c = Q.pop()
        for d, s in adj[c]:
            if d not in rel: rel[d] = rel[c] + s; piece[d] = k; Q.append(d)
            elif rel[d] != rel[c] + s: bad += 1
print("link pieces:", k, "| inconsistent:", bad)
# (a) groups mixing far numbers
print("\n(a) date groups whose members' numbers differ by > 10 (same date, far apart in numbering)  [* = evaluation record]")
rows = []
for c, v in G.items():
    nums = [NUM[i] for i in v]
    if max(nums) - min(nums) > 10:
        rows.append((pos[c], c, v))
for p, c, v in sorted(rows, key=lambda x: (np.nan_to_num(x[0], nan=999), x[1])):
    print(f"  cal {p:6.1f} | " + ", ".join(sorted(rid(i) for i in v)))
print("  count:", len(rows), "of", len(G), "groups")
# (b) pass-2 records: calendar position via group; order vs number
print("\n(b) pass-2 records (number >= 179) with their date's calendar position (from pass-1 members of the same date)")
P2 = sorted([i for i in range(N) if NUM[i] >= 179], key=lambda i: (FARM[i], NUM[i]))
for f in ("F13", "F47"):
    seq = [(NUM[i], pos[comp[i]], rid(i)) for i in P2 if FARM[i] == f]
    print(f"  {f}: " + " ".join(f"{r}@{('%.0f' % p) if np.isfinite(p) else '-'}" for n, p, r in seq))
    ps = [p for n, p, r in seq if np.isfinite(p)]
    inv = sum(1 for x in range(len(ps) - 1) if ps[x + 1] < ps[x] - 3)
    print(f"     with calendar position {len(ps)}/{len(seq)} | backward jumps (>3 days) along numbering: {inv}")
# (c) same-ID numbering vs calendar, pass 1
print("\n(c) pass-1 F47: number vs calendar position offset by block (F13 scale)")
for lo in range(1, 179, 15):
    v = [(NUM[i], pos[comp[i]]) for i in range(N) if FARM[i] == "F47" and lo <= NUM[i] < lo + 15 and np.isfinite(pos[comp[i]])]
    if v: d = [p - n for n, p in v]; print(f"  F47 {lo:3d}-{lo+14:3d}: n {len(v):2d} | calendar - number median {np.median(d):+5.1f} (min {min(d):+.0f}, max {max(d):+.0f})")
# summary table for pass-2 dates that fall inside pass-1 calendar
inside = [i for i in P2 if np.isfinite(pos[comp[i]])]
print(f"\nsummary: pass-2 records whose date also appears in pass 1: {len(inside)}/{len(P2)} (eval {sum(1 for i in inside if not TRAIN[i])}/{sum(1 for i in P2 if not TRAIN[i])})")
pd.DataFrame([dict(rec=rid(i), farm=FARM[i], num=NUM[i], train=bool(TRAIN[i]), group=comp[i], cal=pos[comp[i]], piece=piece[comp[i]], rel=rel[comp[i]], gsize=len(G[comp[i]])) for i in range(N)]).to_csv("dm2_date_map_v1.csv", index=False)
