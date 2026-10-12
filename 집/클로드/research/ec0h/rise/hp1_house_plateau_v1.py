# -*- coding: utf-8 -*-
"""HP1 (2026-10-12 집 클로드; DG1 실패에서 나온 가설: '같은 동은 날짜가 멀어도 EC 수준이 비슷할 수 있다(특히 고EC)').  서술 — 학습 기록끼리.
대상 t(학습 400), 후보 = |달력 날짜 차| >= 3 인 학습 기록(어느 ID든; train 은 방향 제한 없음).  운영 지문 fp_dist(t, c, h) 최소 후보를 고름.
h = 6(대상 0..6시만 — 평가에서 0~6시 행에 쓸 수 있는 정보) 과 h = 23(상한).
보고: 고른 후보의 하루 평균 EC 와 대상 0시 EC 의 상관, 무작위 후보(같은 조건) 기준, 그리고 DIAG10 CUR 0시 잔차 r0 = y0 − q0 를 (A − q0) 로 설명하는 기울기·R²(블록 교차적합).
날짜 차 구간(3~7, 8~20, 21+)·대상 수준(고EC y0 ≥ 1 / 보통) 별로 따로.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v6"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat6 import *
import feat6 as F6
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
CAL = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day))
day_of = np.array([DAY.get(rid(i), np.nan) for i in range(N)])
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
O["rec"] = O.row_id.str[:7]; q0 = O[O.hour == 0].groupby("rec").sg.mean()
TR = [i for i in np.where(TRAIN)[0] if np.isfinite(day_of[i])]
rng = np.random.default_rng(20261245); rows = []
for h in (6, 23):
    for t in TR:
        cand = [c for c in TR if c != t and abs(day_of[c] - day_of[t]) >= 3]
        sc = np.nan_to_num(np.array([F6.fp_dist(t, c, h) for c in cand]), nan=99); o = int(np.argmin(sc)); c = cand[o]; rc = cand[int(rng.integers(len(cand)))]
        rows.append(dict(h=h, t=rid(t), y0=float(EC[t, 0]), q0=q0.get(rid(t), np.nan), A=float(np.nanmean(EC[c])), Arand=float(np.nanmean(EC[rc])),
                         dd=abs(day_of[c] - day_of[t]), fp=float(sc[o]), same_id=FARM[c] == FARM[t]))
D = pd.DataFrame(rows).dropna(subset=["q0"]); D["r0"] = D.y0 - D.q0; D["x"] = D.A - D.q0; D["xr"] = D.Arand - D.q0
D["blk"] = D.t.str[:3] + (D.t.str[4:7].astype(int) // 10).astype(str)
def cvfit(G, col):
    pred = np.zeros(len(G))
    for b in G.blk.unique():
        tr = (G.blk != b).values; x, y = G[col].values[tr], G.r0.values[tr]; s = (x * y).sum() / (x * x).sum()
        pred[~tr] = s * G[col].values[~tr]
    return 1 - ((G.r0 - pred) ** 2).sum() / (G.r0 ** 2).sum(), (G[col] * G.r0).sum() / (G[col] ** 2).sum()
for h in (6, 23):
    for nm, s in (("all", lambda G: G.y0 > -9), ("high y0>=1", lambda G: G.y0 >= 1), ("normal y0<1", lambda G: G.y0 < 1)):
        for dn, dsel in (("all dd", lambda G: G.dd >= 3), ("3-7", lambda G: G.dd <= 7), ("8-20", lambda G: (G.dd > 7) & (G.dd <= 20)), ("21+", lambda G: G.dd > 20)):
            G = D[(D.h == h)]; G = G[s(G) & dsel(G)]
            if len(G) < 15: continue
            r2, b = cvfit(G, "x"); r2r, br = cvfit(G, "xr")
            print(f"h {h:2d} | {nm:12s} | {dn:6s} n {len(G):3d} | corr(A, y0) {np.corrcoef(G.A, G.y0)[0,1]:+.2f} (random {np.corrcoef(G.Arand, G.y0)[0,1]:+.2f}) | r0 on (A-q0): slope {b:+.2f}, CV R2 {r2:+.3f} (random {r2r:+.3f})")
D.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hp1_v1.csv"), index=False)
