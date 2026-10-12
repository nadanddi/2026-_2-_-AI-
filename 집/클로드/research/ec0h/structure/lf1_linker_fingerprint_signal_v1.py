# -*- coding: utf-8 -*-
"""LF1 (2026-10-12 집 클로드; 사용자: '짝찾기에 운영 지문 넣는 방식').  1단계 서술(신호 확인).
운영 지문 거리 d(b, a; h) = 오늘 기록 b 의 0..h 시 구동기 7종 + 실내 CO2 값과, 후보 전날 기록 a 의 같은 시각 값의 표준화 거리 평균
  (b 는 0..h 만 — 규정상 h 시 행에서 쓸 수 있는 범위; a 는 학습 기록 또는 같은 ID 앞 번호라 하루 전체 사용 가능하나 같은 시각끼리 비교).
(1) 판별력: 진단용 확실 연결(cert) b ← a_true 마다, a_true 와 같은 날짜(DM11)의 다른 기록(= 같은 날 다른 동, 진짜 경쟁자)들 대비
    a_true 가 가장 가까운 비율, a_true 가 경쟁자 하나보다 가까울 확률(AUC), h = 0, 5, 9, 12, 17, 23.
(2) v7 거울 CV 오선택 구제: cv_all400_choice_v7m 에서 고른 전날 ≠ 진짜 전날인 기록마다 d(b, 진짜) < d(b, 고른 것) 비율.
(3) 같은 동 연속 날 vs 다른 동 같은 날 거리 분포 중앙값.
판단: (1) 에서 h ≥ 9 AUC ≥ .75 이고 (2) 구제 비율 ≥ .6 이면 2단계(짝찾기 특징 추가 + 거울 CV)로 진행.
"""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd
H = r"C:\work\farmai\집\클로드\research"; here = os.path.dirname(os.path.abspath(__file__))
X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")])
X = X[X.row_id.str[:3].isin(["F13", "F47"])]; X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:].astype(int)
V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
recs = sorted(X.rec.unique()); ri = {r: i for i, r in enumerate(recs)}
A = np.stack([X.pivot(index="rec", columns="h", values=v).reindex(index=recs, columns=range(24)).values for v in V], 2)
A = (A - np.nanmean(A, (0, 1))) / (np.nanstd(A, (0, 1)) + 1e-9)
def dist(b, a, h):
    d = np.abs(A[ri[b], :h + 1] - A[ri[a], :h + 1]); return float(np.nanmean(np.minimum(d, 4)))
CAL = pd.read_csv(os.path.join(here, "..", "rise", "dm11_calendar_v1.csv")); DAY = dict(zip(CAL.rec, CAL.day))
byday = CAL.dropna(subset=["day"]).groupby("day").rec.apply(list).to_dict()
TL = pd.read_pickle(os.path.join(H, "local", "linker_v3", "truth.pkl")); TL = TL[TL.cert]
sys.path.insert(0, os.path.join(H, "linker_v4"))
from feat4 import FARM, NUM  # noqa: E402
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"
links = [(rid(int(b)), rid(int(a))) for a, b in zip(TL.a1, TL.b)]
HS = (0, 5, 9, 12, 17, 23)
print("(1) true previous day vs other houses on that same date")
res = []
for h in HS:
    top = []; auc = []
    for b, a in links:
        d0 = DAY.get(a)
        if not np.isfinite(d0 if d0 is not None else np.nan): continue
        comp = [x for x in byday.get(d0, []) if x != a and x in ri]
        if not comp: continue
        dt = dist(b, a, h); dc = np.array([dist(b, c, h) for c in comp])
        top.append(float(dt < dc.min())); auc.append(float(np.mean(dt < dc) + .5 * np.mean(dt == dc)))
    res.append(dict(h=h, n=len(top), true_is_closest=np.mean(top), AUC=np.mean(auc)))
print(pd.DataFrame(res).round(3).to_string(index=False))
CH = pd.read_csv(os.path.join(H, "local", "linker_v7", "cv_all400_choice_v7m.csv"), encoding="utf-8-sig")
TP = dict(links)
W = CH[CH.rec.map(TP).notna() & (CH.prev != CH.rec.map(TP))]
print(f"\n(2) v7 mirror-CV wrong choices with a known true previous day: {len(W)} (of which p >= .8: {int((W.p >= .8).sum())})")
for h in HS:
    ok = [dist(r, TP[r], h) < dist(r, c, h) for r, c in zip(W.rec, W.prev) if TP[r] in ri and c in ri]
    okh = [dist(r, TP[r], h) < dist(r, c, h) for r, c, p in zip(W.rec, W.prev, W.p) if p >= .8 and TP[r] in ri and c in ri]
    print(f"   h {h:2d}: true closer than chosen {np.mean(ok):.2f} (n {len(ok)}) | among p>=.8 wrong {np.mean(okh) if okh else float('nan'):.2f} (n {len(okh)})")
print("\n(3) median distance at h 23: same house consecutive days", round(float(np.median([dist(b, a, 23) for b, a in links if a in ri])), 3),
      "| other houses same date as true prev", round(float(np.median([dist(b, c, 23) for b, a in links if np.isfinite(DAY.get(a, np.nan)) for c in byday.get(DAY.get(a), []) if c != a and c in ri])), 3))
