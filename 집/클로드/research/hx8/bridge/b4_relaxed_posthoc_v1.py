# -*- coding: utf-8 -*-
"""B4 (2026-10-13 집 클로드, hx8/bridge) — 사후(post hoc) 서술 전용: b2 FAIL 이 사슬 47 하나 때문인지, 관측 선택 문턱을 넓히면 어떤지.
판정 아님(b2 사전 규칙 FAIL 확정). b2_cases_v1.csv 재사용(재적합 없음). 관측 선택 = 앞 앵커 23시 ≥ θa, 뒤 앵커 0시 ≥ θa, q ≥ θq (평가에서 볼 수 있는 값만).
또한 b2 사전 규칙의 검정력: 사슬 2개면 부호 뒤집기 한쪽 p 최소 .25 → p < .025 는 원리상 불가였음(기록).
"""
import sys, os, itertools
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); LOC = os.path.join(HERE, "local")
A = pd.read_csv(os.path.join(LOC, "b2_cases_v1.csv"))
rm = lambda S, c: np.sqrt(((S[c] - S.y) ** 2).mean())
def flip(S, col):
    g = S.groupby("chain").apply(lambda T: ((T[col] - T.y) ** 2 - (T.q - T.y) ** 2).mean()).values
    obs = g.sum(); n = len(g)
    cnt = sum(np.dot(sg, np.abs(g)) <= obs + 1e-12 for sg in itertools.product([1, -1], repeat=n))
    return cnt / 2 ** n, n, (g < 0).sum()
print("θa θq | 묶음 | n(시드당) 사슬 목표 | RMSE q INT BI1 KB_rw | KB 개선 비율 | 부호뒤집기 p(KB), 사슬 수, KB 좋아진 사슬")
for th_a in (0.8, 0.9, 1.0):
    for th_q in (0.8, 0.9):
        for excl in (False, True):
            for seed in sorted(A.seed.unique())[:1]:
                S = A[(A.seed == seed) & (A.eA >= th_a) & (A.eB >= th_a) & (A.q >= th_q) & A.G.between(6, 10)]
                if excl: S = S[S.chain != 47]
                if len(S) == 0: print(f"{th_a} {th_q} {'no47' if excl else 'all '} | 없음"); continue
                p, nch, nb = flip(S, "KB_rw")
                print(f"{th_a} {th_q} {'no47' if excl else 'all '} | n {len(S):3d} ch {S.chain.nunique()} tgt {S.rec.nunique():2d} | "
                      f"q {rm(S,'q'):.3f} INT {rm(S,'INT'):.3f} BI1 {rm(S,'BI1'):.3f} KB {rm(S,'KB_rw'):.3f} | {(np.abs(S.KB_rw-S.y) < np.abs(S.q-S.y)).mean():.2f} | p {p:.3f} ch {nch} better {nb}")
# 3시드 방향 일치(θa .8, θq .8, 전체)
print("\n3시드 RMSE (θa .8, θq .8, G 6-10, 전체 사슬):")
for seed, S in A[(A.eA >= .8) & (A.eB >= .8) & (A.q >= .8) & A.G.between(6, 10)].groupby("seed"):
    print(f"  seed {seed}: q {rm(S,'q'):.4f} KB {rm(S,'KB_rw'):.4f} INT {rm(S,'INT'):.4f} BI1 {rm(S,'BI1'):.4f}")
