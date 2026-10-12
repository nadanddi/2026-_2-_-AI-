# -*- coding: utf-8 -*-
"""HP2 (2026-10-12 집 클로드; HP1 후속 서술): 시각 h 마다 대상 0..h 운영 지문으로 고른 '먼 날짜(≥3일) 학습 기록'의 EC 가
그 시각 잔차 r_h = y_h − q_h 를 얼마나 설명하나(블록 교차적합 R², 무작위 기준).  A_h = 고른 기록의 하루 평균 EC.  q = DIAG10 CUR sg."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v6"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat6 import *
import feat6 as F6
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"
CAL = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day))
day_of = np.array([DAY.get(rid(i), np.nan) for i in range(N)])
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
O["rec"] = O.row_id.str[:7]; Q = O.groupby(["rec", "hour"]).sg.mean().unstack()
TR = [i for i in np.where(TRAIN)[0] if np.isfinite(day_of[i]) and rid(i) in Q.index]
rng = np.random.default_rng(20261246); out = []
for h in (3, 6, 9, 12, 15, 18, 21, 23):
    rows = []
    for t in TR:
        cand = [c for c in TR if c != t and abs(day_of[c] - day_of[t]) >= 3]
        sc = np.nan_to_num(np.array([F6.fp_dist(t, c, h) for c in cand]), nan=99); c = cand[int(np.argmin(sc))]; rc = cand[int(rng.integers(len(cand)))]
        q = Q.at[rid(t), h]; rows.append(dict(t=rid(t), r=float(EC[t, h]) - q, x=float(np.nanmean(EC[c])) - q, xr=float(np.nanmean(EC[rc])) - q))
    D = pd.DataFrame(rows); D["blk"] = D.t.str[:3] + (D.t.str[4:7].astype(int) // 10).astype(str)
    def cv(col):
        pred = np.zeros(len(D))
        for b in D.blk.unique():
            tr = (D.blk != b).values; s = (D[col].values[tr] * D.r.values[tr]).sum() / (D[col].values[tr] ** 2).sum(); pred[~tr] = s * D[col].values[~tr]
        return 1 - ((D.r - pred) ** 2).sum() / (D.r ** 2).sum(), (D[col] * D.r).sum() / (D[col] ** 2).sum()
    r2, b = cv("x"); r2r, _ = cv("xr"); p2 = D.t.str[4:7].astype(int) >= 179
    out.append((h, r2, b, r2r)); print(f"h {h:2d}: slope {b:+.2f} | CV R2 {r2:+.3f} (random {r2r:+.3f}) | n {len(D)}", flush=True)
