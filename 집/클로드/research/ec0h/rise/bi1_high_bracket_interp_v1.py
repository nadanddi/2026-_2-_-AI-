# -*- coding: utf-8 -*-
"""BI1 (2026-10-12 집 클로드; '정보 없는 고EC 4기록 크기 추정' — 양 끝 정답 보간이 고EC 에서는 통하나?).  서술(진단용 연결 = 같은 동 식별 완벽 가정).
학습 기록 r 마다 진단용 사슬(truth a1)로 같은 동을 앞뒤로 따라가, r 자신을 빼고 k 일 앞(k = 2~5)의 정답 23시 값 A 와 m 일 뒤(m = 2~5)의 정답 0시 값 B 가 있는 경우:
  보간 수준 I = A + (B − A)·k/(k+m) (r 의 하루 위치), 대상 정답 하루 평균 y, 모델 하루 평균 q(DIAG10 sg).
보고: 고EC(하루 평균 ≥ 1)·보통 날, 간격(k+m)별: |y − I| vs |y − q|, 보간이 더 가까운 비율, 최적 반영 β(모델 → 보간).
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
from feat4 import *
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"; idx = {rid(i): i for i in range(N)}
CAL = pd.read_csv("dm11_calendar_v1.csv"); DAY = dict(zip(CAL.rec, CAL.day))
TL = pd.read_pickle(S3 + r"\truth.pkl"); pre = {rid(int(b)): rid(int(a)) for a, b in zip(TL.a1, TL.b)}
suc = {}
for b, a in pre.items(): suc.setdefault(a, []).append(b)
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
q = O.groupby("rec").sg.mean()
lab = lambda r: r in idx and TRAIN[idx[r]]
def walk_back(r, k):
    x = r
    for _ in range(k):
        x = pre.get(x)
        if x is None: return None
    return x
def walk_fwd(r, m):
    x = r
    for _ in range(m):
        nx = [y for y in suc.get(x, []) if np.isfinite(DAY.get(y, np.nan)) and DAY.get(y) == DAY.get(x, np.nan) + 1]
        if not nx: return None
        x = nx[0]
    return x
rows = []
for r in q.index:
    d = DAY.get(r)
    if not np.isfinite(d or np.nan): continue
    y = float(np.nanmean(EC[idx[r]]))
    for k in range(2, 6):
        a = walk_back(r, k)
        if not (a and lab(a) and DAY.get(a) == d - k): continue
        for m in range(2, 6):
            b = walk_fwd(r, m)
            if not (b and lab(b) and DAY.get(b) == d + m): continue
            A, B = float(EC[idx[a], 23]), float(EC[idx[b], 0]); I = A + (B - A) * k / (k + m)
            rows.append(dict(rec=r, k=k, m=m, gap=k + m, y=y, q=q[r], I=I, hi=y >= 1))
D = pd.DataFrame(rows).drop_duplicates(["rec", "k", "m"])
D["errQ"] = (D.y - D.q).abs(); D["errI"] = (D.y - D.I).abs(); D["x"] = D.I - D.q; D["r"] = D.y - D.q
pd.set_option("display.width", 200)
for nm, S in (("high EC", D[D.hi]), ("normal", D[~D.hi])):
    for g, G in S.groupby(pd.cut(S.gap, [3, 5, 7, 10])):
        if len(G) < 3: continue
        b = float((G.x * G.r).sum() / (G.x ** 2).sum())
        print(f"{nm:8s} gap {str(g):8s} n {len(G):3d} (records {G.rec.nunique():2d}) | median |err| model {G.errQ.median():.3f} vs interp {G.errI.median():.3f} | interp closer {(G.errI < G.errQ).mean():.2f} | best beta {b:.2f}")
print("\nhigh-EC records used:", sorted(D[D.hi].rec.unique()))
