# -*- coding: utf-8 -*-
"""B1 (2026-10-13 집 클로드, hx8/bridge) — 학습 기록 같은 동 연속 날 사슬(진단용 연결) 만들기 + 고EC 에피소드 목록(서술).
연결 규칙(학습끼리만, 정답 사용 = 진단용 오라클): truth.pkl 의 a1 → b 중  (i) 학습 전용 달력 dmt3e_E0 에서 같은 seg 이고 day(b) = day(a)+1  또는
 (ii) 자정 EC 연속 |e0(b) − e23(a)| ≤ .06 이고 c1 ≤ .5 (2차·조각 seg 에서 달력이 없을 때).  한 a 에 여러 b 가 붙으면 c1 최소 하나만.
출력: local 사슬 표(hx8/bridge/local/chains_v1.csv: rec, chain, pos, y, e0, e23, qD(DIAG10 3시드 평균), qE(EL1), qP(P2LOO)).
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
HERE = os.path.dirname(os.path.abspath(__file__)); LOC = os.path.join(HERE, "local"); os.makedirs(LOC, exist_ok=True)
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
E0 = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); D0 = dict(zip(E0.rec, E0.day)); G0 = dict(zip(E0.rec, E0.seg))
TL = pd.read_pickle(S3 + r"\truth.pkl")
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[O.tag == "CUR"].copy(); O["rec"] = O.row_id.str[:7]
Q = O.groupby(["validator", "rec"]).sg.mean().unstack(0)
links = []
for a, b, c1 in zip(TL.a1, TL.b, TL.c1):
    a, b = int(a), int(b)
    if not (TRAIN[a] and TRAIN[b]): continue
    ra, rb = rid(a), rid(b)
    cal = (ra in G0 and rb in G0 and G0[ra] == G0[rb] and D0[rb] == D0[ra] + 1)
    cont = abs(EC[b, 0] - EC[a, 23]) <= 0.06 and c1 <= 0.5
    if cal or cont:
        links.append((ra, rb, c1, cal, cont))
L = pd.DataFrame(links, columns=["a", "b", "c1", "cal", "cont"]).sort_values("c1").drop_duplicates("a").drop_duplicates("b")
print("links", len(L), "cal", L.cal.sum(), "cont-only", (~L.cal & L.cont).sum())
nxt = dict(zip(L.a, L.b)); prv = dict(zip(L.b, L.a))
starts = [r for r in nxt if r not in prv]
rows = []; cid = 0
for s in sorted(starts):
    x, p = s, 0
    while True:
        i = idx[x]
        rows.append(dict(rec=x, chain=cid, pos=p, y=float(np.nanmean(EC[i])), e0=float(EC[i, 0]), e23=float(EC[i, 23]),
                         qD=Q.loc[x, "DIAG10"] if x in Q.index else np.nan, qE=Q.loc[x, "EL1"] if x in Q.index else np.nan,
                         qP=Q.loc[x, "P2LOO"] if x in Q.index else np.nan))
        if x not in nxt: break
        x = nxt[x]; p += 1
    cid += 1
C = pd.DataFrame(rows); C.to_csv(os.path.join(LOC, "chains_v1.csv"), index=False)
print("chains", C.chain.nunique(), "records in chains", len(C), "length dist", C.groupby("chain").size().value_counts().sort_index().to_dict())
pd.set_option("display.width", 250)
hi_ch = C[C.y >= 1.0].chain.unique()
for c in hi_ch:
    G = C[C.chain == c]
    print(f"\nchain {c} len {len(G)}")
    print("  " + " | ".join(f"{r.rec} y{r.y:.2f} q{r.qD:.2f}" for r in G.itertuples()))
print("\nhigh-EC days total", (C.y >= 1).sum(), "of train high", sum(1 for i in range(N) if TRAIN[i] and np.nanmean(EC[i]) >= 1))
