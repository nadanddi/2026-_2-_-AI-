# -*- coding: utf-8 -*-
"""NA4 (2026-10-12 집 클로드; 사용자: '앵커 없는 기록 36개 분류').  서술.
평가 60 중 짝찾기 v7(eval60_predecessor_v7, 0시 판단)에서 '정답 있는 전날·확신 p ≥ .8' 앵커를 못 받은 기록을 원인별로 나눔.
  날짜: DM11 달력(설명용; 평가 날짜는 외기 일치로 붙은 것).  d−1 날짜의 학습(정답) 기록 수(동 무관) = NF1 lab_prev.
분류(위에서부터 먼저 걸리는 것):
  N0 날짜 미확정(DM11 에 날짜 없음)
  N1 정보 구멍: d−1 날짜에 학습 기록 0 (어떤 동이든 전날 정답 없음) — 앵커 원리상 불가
  N2 전날로 평가 기록을 고름(같은 ID 앞 번호) — 진짜 전날이 평가 기록일 수 있음 (사슬 필요)
  N3 정답 있는 기록을 골랐지만 날짜가 d−1 이 아님(날짜 오선택)
  N4 정답 있는 d−1 기록을 골랐지만 확신 p < .8
  N5 그 밖(차단 등)
각 기록: 고른 전날·p·날짜 차, d−1 학습 기록 수, d−1 학습 기록 중 운영 지문(같은 시각 0시·23시 값 거리, LF1 방식)이 가장 가까운 기록과 2등과의 거리 차(서술용, 풀 수 있는지 감 잡기).
12회차/13회차 온도 예측의 0시 값도 붙임(정답 없음).
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
E = pd.read_csv(os.path.join(H, "local", "linker_v7", "eval60_predecessor_v7.csv"), encoding="utf-8-sig").set_index("row")
C = pd.read_csv(os.path.join(H, "ec0h", "rise", "dm11_calendar_v1.csv")); DAY = dict(zip(C.rec, C.day)); SPL = dict(zip(C.rec, C.split))
byday = C.dropna(subset=["day"]).groupby("day").rec.apply(list).to_dict()
X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")])
X = X[X.row_id.str[:3].isin(["F13", "F47"])]; X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:].astype(int)
V = ["act_heating", "act_thermal", "act_vent", "act_shade", "act_circfan", "act_co2", "act_fog", "in_co2"]
recs = sorted(X.rec.unique()); ri = {r: i for i, r in enumerate(recs)}
A = np.stack([X.pivot(index="rec", columns="h", values=v).reindex(index=recs, columns=range(24)).values for v in V], 2)
A = (A - np.nanmean(A, (0, 1))) / (np.nanstd(A, (0, 1)) + 1e-9)
fp = lambda b, a, h: float(np.nanmean(np.minimum(np.abs(A[ri[b], :h + 1] - A[ri[a], :h + 1]), 4)))
anc = E[(E.p_h0 >= .8) & E.prev_is_labelled.astype(bool) & (E.dT_air.abs() <= 1.5)].index
rows = []
for r, c in E.iterrows():
    if r in anc: continue
    d = DAY.get(r); p = c.prev_h0; dp = DAY.get(p)
    labprev = [x for x in byday.get(d - 1, []) if SPL.get(x) == "train"] if d == d else []
    if not np.isfinite(d if d is not None else np.nan): cat = "N0 date unknown"
    elif not labprev: cat = "N1 no labelled record on d-1"
    elif SPL.get(p) == "eval": cat = "N2 chose eval record"
    elif dp != d - 1: cat = "N3 chose labelled, wrong date"
    elif c.p_h0 < .8: cat = "N4 right date, p < .8"
    else: cat = "N5 other (gate)"
    best = second = gap0 = np.nan; bname = ""
    if labprev:
        dd = sorted((fp(r, x, 0), x) for x in labprev)
        best, bname = dd[0]; second = dd[1][0] if len(dd) > 1 else np.nan
    rows.append(dict(rec=r, cat=cat, day=d, prev=p, p=c.p_h0, prev_day=dp, n_lab_prev=len(labprev), fp_best=bname, fp_d0=best, fp_gap=second - best if np.isfinite(second) else np.nan, chose_is_fpbest=(p == bname)))
T = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(f"unanchored eval records: {len(T)} (anchored {len(anc)})"); print(T.cat.value_counts().to_string())
print(); print(T.sort_values(["cat", "day"]).round(3).to_string(index=False))
T.to_csv(os.path.join(H, "na4_v1.csv"), index=False)
