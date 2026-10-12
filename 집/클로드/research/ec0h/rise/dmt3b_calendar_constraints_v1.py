# -*- coding: utf-8 -*-
"""DMT3b (2026-10-13 집 클로드; '달력 보강' 2차 — DMT3 v1 은 끼워 넣기로 뒤 날짜가 밀려 확실 연결 4개가 깨지고(+3·+15), 외기 붙이기가 엉뚱한 곳(F47_231 → 73일)에 붙어 실패).
방식: 날짜 차 제약을 가중 union-find 로 모아 풂(학습 입력·학습 확실 연결만).
  노드 = 날짜 묶음(DMT 날짜 묶음 + 날짜 없는 학습 기록의 외기 90% 일치 묶음).
  제약 순서(앞의 것이 우선, 모순되면 버림·보고):
    A. 학습 확실 연결 a→b : day(b) = day(a) + 1
    B. DMT 이웃 날 k→k+1 중 외기 자정 비용 ≤ T(끊김 아님) : +1
    C. 외기 연속: '다음날이 비어 있는' 묶음 g1 과 '전날이 비어 있는' 묶음 g2(서로 다른 성분), 비용(g1→g2) ≤ T 이고 서로 최선인 쌍만, 싼 순서로 : +1
  성분 = seg(날짜 차는 같은 seg 안에서만 정의).  T = 확실 연결 외기 자정 비용 99% 분위.
출력 local/dmt3b_days.csv(rec, day, seg).  검산: 버린 제약, 확실 연결 날짜 차, 날짜 없는 학습 기록, seg 수.
대조(서술만): DM11 과 같은 seg 안 기록 쌍 날짜 차 일치율 — DMT2 vs DMT3b, 그리고 핵심 기록 날짜.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
here = os.path.dirname(os.path.abspath(__file__))
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
D = pd.read_csv(os.path.join(here, "dmt_calendar_v1.csv")); D = D[D.split == "train"]
W = np.array([1, 1, .3, .5]); oj = lambda a, b: float(np.sum(W * np.abs(X[a, 23] + .5 * (X[a, 23] - X[a, 22]) - X[b, 0]) / sdO))
TL = pd.read_pickle(S3 + r"\truth.pkl"); TLc = TL[TL.cert]
CL = [(int(a), int(b)) for a, b in zip(TLc.a1, TLc.b) if TRAIN[int(a)] and TRAIN[int(b)]]
Tq = float(np.quantile([oj(a, b) for a, b in CL], .99))
groups = [list(map(idx.get, g.rec)) for _, g in D.dropna(subset=["day"]).sort_values("day").groupby("day", sort=True)]; ndmt = len(groups)
Xf = X.reshape(N, -1)
for i in [idx[r] for r in D[D.day.isna()].rec]:
    for g in groups[ndmt:]:
        if (np.abs(Xf[g[0]] - Xf[i]) < 1e-6).mean() >= .9: g.append(i); break
    else: groups.append([i])
G = len(groups); gof = {i: k for k, g in enumerate(groups) for i in g}
par = list(range(G)); pot = [0] * G  # pot[g] = day(g) - day(parent)
def find(g):
    if par[g] == g: return g, 0
    r, p = find(par[g]); par[g] = r; pot[g] += p; return r, pot[g]
def add(g1, g2, d=1):
    r1, p1 = find(g1); r2, p2 = find(g2)
    if r1 == r2: return p2 - p1 == d
    par[r2] = r1; pot[r2] = p1 + d - p2; return True
gcost = lambda g1, g2: min(oj(a, b) for a in groups[g1] for b in groups[g2])
dropped = []
for a, b in CL:
    if gof[a] != gof[b] and not add(gof[a], gof[b]): dropped.append(("A", rid(a), rid(b)))
for k in range(ndmt - 1):
    if gcost(k, k + 1) <= Tq and not add(k, k + 1): dropped.append(("B", rid(groups[k][0]), rid(groups[k + 1][0])))
cadd = []
while True:
    day = lambda g: find(g)[1]; root = lambda g: find(g)[0]
    occ = {(root(g), day(g)) for g in range(G)}
    tails = [g for g in range(G) if (root(g), day(g) + 1) not in occ]; heads = [g for g in range(G) if (root(g), day(g) - 1) not in occ]
    C = [(gcost(t, h), t, h) for t in tails for h in heads if root(t) != root(h)]; C = [c for c in C if c[0] <= Tq]
    bt = {}; bh = {}
    for c, t, h in C:
        if t not in bt or c < bt[t][0]: bt[t] = (c, h)
        if h not in bh or c < bh[h][0]: bh[h] = (c, t)
    mutual = sorted((c, t, h) for c, t, h in C if bt[t][1] == h and bh[h][1] == t)
    if not mutual: break
    c, t, h = mutual[0]; add(t, h); cadd.append((round(c, 2), [rid(i) for i in groups[t]][:3], [rid(i) for i in groups[h]][:3]))
print(f"T {Tq:.2f} | groups {G} (DMT {ndmt}) | dropped constraints {dropped}")
print("continuity joins:"); [print("   ", x) for x in cadd]
roots = sorted({find(g)[0] for g in range(G)}, key=lambda r: min(find(g)[1] for g in range(G) if find(g)[0] == r))
segid = {r: k for k, r in enumerate(sorted({find(g)[0] for g in range(G)}))}
rows = []
for g in range(G):
    r, d = find(g); m = min(find(x)[1] for x in range(G) if find(x)[0] == r)
    for i in groups[g]: rows.append(dict(rec=rid(i), day=d - m + 1, seg=segid[r]))
out = pd.DataFrame(rows); out.to_csv(os.path.join(env.LOCAL, "dmt3b_days.csv"), index=False)
dd = out.set_index("rec"); chk = pd.Series([dd.day[rid(b)] - dd.day[rid(a)] for a, b in CL if dd.seg[rid(a)] == dd.seg[rid(b)]]).value_counts().to_dict()
print(f"segments {out.seg.nunique()} | sizes(days) {sorted(out.groupby('seg').day.nunique().tolist(), reverse=True)} | certain-link day diffs (same seg) {chk} | train without day {int(TRAIN.sum()) - len(out)}")
D11 = pd.read_csv(os.path.join(here, "dm11_calendar_v1.csv")).set_index("rec").day
for fn in ("dmt2_days.csv", "dmt3b_days.csv"):
    Cc = pd.read_csv(os.path.join(env.LOCAL, fn)).dropna(); Cc["d11"] = Cc.rec.map(D11); Cc = Cc.dropna(); ok = tot = 0
    for _, g in Cc.groupby("seg"):
        m = np.subtract.outer(g.day.values, g.day.values) == np.subtract.outer(g.d11.values, g.d11.values); ok += m.sum(); tot += m.size
    print(f"{fn}: within-seg pair agreement with DM11 {ok / tot:.4f} over {tot} pairs | dated train {len(Cc)}")
for r in ["F47_132", "F47_231", "F13_217", "F13_127", "F13_128", "F13_218", "F47_229", "F13_233", "F47_146", "F13_143"]:
    print("  ", r, "DMT3b day", dd.day.get(r), "seg", dd.seg.get(r), "| DM11", D11.get(r))
