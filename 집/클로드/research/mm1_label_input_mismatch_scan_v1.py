# -*- coding: utf-8 -*-
"""MM1 (2026-10-12 집 클로드; 병목 재조사 — 온도 큰 오차 기록 F47_191·F47_178·F47_189 에서 본 '정답은 다른 기록과 거의 같고 입력은 다름').  서술(포렌식).
가설 H-MIX: 일부 기록은 입력과 정답이 서로 다른 동/날에서 왔다(기록 조립 시 어긋남).
  흔적(맞다면): 그 기록의 정답 곡선(배지온도 24시)이 주변 날짜(DM11 d−1..d+1) 다른 기록 하나와 거의 같고(L 작음), 그 기록과 입력(실내온도)은 크게 다름(I 큼);
               모델 하루 오차가 큼(입력으로 정답을 설명 못 함); 정답 쌍 기록의 입력과 이 기록 정답의 관계는 정상.
  안 보여야 할 것: 무작위 기록에서도 같은 비율로 '정답 거의 같음 + 입력 다름' 쌍이 흔함.
측정: 학습 기록 r 마다 후보 c(주변 날짜 학습 기록, r 제외)에 대해 L = mean_h|ST_r − ST_c|, I = mean_h|IN_r − IN_c|.
  r 의 정답 짝 c* = argmin L.  지표: L* = L(c*), I* = I(c*), 그리고 '같은 동 다음날/전날(진단용 연결)과의 L·I'.
  플래그 F = L* ≤ .6 그리고 I* ≥ 2.5.   EV 13회차 구성(v22) 하루 오차와 비교.  기준선: 같은 동 이웃날 쌍의 L·I 분포.
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")).merge(pd.read_csv(os.path.join(env.DATA, "train_y.csv")), on="row_id")
X = X[X.row_id.str[:3].isin(["F13", "F47"])]; X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:].astype(int)
ST = X.pivot(index="rec", columns="h", values="sub_temp"); IN = X.pivot(index="rec", columns="h", values="in_temp")
C = pd.read_csv(os.path.join(H, "ec0h", "rise", "dm11_calendar_v1.csv")); DAY = dict(zip(C.rec, C.day))
D = pd.read_pickle(os.path.join(H, "ev_validator", "temp", "local_ckpt", "ev_all.pkl")); D["rec"] = D.row_id.str[:7]
D["e"] = D[[f"v22_a_{c}" for c in ["7A", "7B", "101A", "101B"]]].mean(1) - D.sub_temp
DE = D.groupby("rec").e.mean()
recs = [r for r in ST.index if np.isfinite(DAY.get(r, np.nan))]
byday = {}
for r in recs: byday.setdefault(DAY[r], []).append(r)
rows = []
for r in recs:
    d = DAY[r]; cand = [c for dd in (d - 1, d, d + 1) for c in byday.get(dd, []) if c != r]
    if not cand: continue
    L = np.array([np.nanmean(np.abs(ST.loc[r].values - ST.loc[c].values)) for c in cand]); I = np.array([np.nanmean(np.abs(IN.loc[r].values - IN.loc[c].values)) for c in cand])
    k = int(np.argmin(L)); ki = int(np.argmin(I))
    rows.append(dict(rec=r, day=d, p2=int(r[4:7]) >= 179, Lstar=L[k], Istar=I[k], pair=cand[k], pair_dd=DAY[cand[k]] - d, inpair=cand[ki], Imin=I[ki], Lin=L[ki], de=DE.get(r, np.nan)))
T = pd.DataFrame(rows); T["F"] = (T.Lstar <= .6) & (T.Istar >= 2.5)
pd.set_option("display.width", 220)
print(f"records {len(T)} | flagged {int(T.F.sum())} (pass-2 {int((T.F & T.p2).sum())} of {int(T.p2.sum())})")
print("baseline: all records L* median %.2f, I* median %.2f | label-pair = input-pair %.2f" % (T.Lstar.median(), T.Istar.median(), (T.pair == T.inpair).mean()))
print("EV day error |de|: flagged median %.2f mean %.2f vs others median %.2f mean %.2f" % (T[T.F].de.abs().median(), T[T.F].de.abs().mean(), T[~T.F].de.abs().median(), T[~T.F].de.abs().mean()))
print("SSE share of flagged records (EV rows): %.2f" % (D[D.rec.isin(T[T.F].rec)].e.pow(2).sum() / D.e.pow(2).sum()))
print("\ntop |de| records with their label-pair / input-pair:")
print(T.reindex(T.de.abs().sort_values(ascending=False).index).head(20)[["rec", "day", "p2", "de", "Lstar", "Istar", "pair", "pair_dd", "inpair", "Imin", "Lin", "F"]].round(2).to_string(index=False))
print("\nflagged records:"); print(T[T.F].sort_values("de")[["rec", "day", "p2", "de", "Lstar", "Istar", "pair", "pair_dd", "inpair", "Imin", "Lin"]].round(2).to_string(index=False))
T.to_csv(os.path.join(H, "mm1_v1.csv"), index=False)
