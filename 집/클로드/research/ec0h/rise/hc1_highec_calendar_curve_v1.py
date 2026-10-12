# -*- coding: utf-8 -*-
"""HC1 (2026-10-13 집 클로드; 사용자: '전날 없는 고EC 기록 수준 맞추기 새 각도로') 서술(신호 확인).
가설: 고EC 수준은 동과 무관하게 '시즌 날짜'에 따라 움직임(6.690: 74~80일 1.10→1.69, 89~98일 1.79→2.71). 모델은 고EC 를 알아보지만 크기를 누름(기울기 .57~.70).
→ 고EC 로 알아본 기록(모델 하루 평균 예측 ≥ .9)의 하루 수준을, 같은 구간·±k 일 안의 '다른' 학습 고EC 기록(정답 하루 평균 ≥ 1) 수준의 중앙값(날짜 곡선)으로 바꾸거나 섞음.
날짜 = 학습 전용 달력 DMT3e(E0).  검증: DIAG10(rv1 CUR, 시드 평균); 곡선에는 같은 DIAG10 폴드 기록·자기 자신 제외.
보고: 알아본 기록 수·곡선 있는 수, 기록 하루 수준 RMSE(모델 / 곡선 / 반반), 실제 고EC·보통 나눠서, 전체·2차; k = 2, 3, 5. 평가 기록 중 알아본 기록의 날짜 확보 수(0시 외기 쌍둥이)."""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import env
import numpy as np, pandas as pd
from feat4 import X, FARM, NUM, N, TRAIN
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"
C = pd.read_csv(os.path.join(env.LOCAL, "dmt3e_E0_days.csv")); DAY = dict(zip(C.rec, C.day)); SEG = dict(zip(C.rec, C.seg))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
R = O.groupby(["rec", "fold"], as_index=False).agg(y=("sub_ec", "mean"), sg=("sg", "mean"))
R["day"] = R.rec.map(DAY); R["seg"] = R.rec.map(SEG); R["p2"] = R.rec.str[4:7].astype(int) >= 179
fold = dict(zip(R.rec, R.fold)); hi_recs = R[R.y >= 1.0]
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
pd.set_option("display.width", 220)
for k in (2, 3, 5):
    rows = []
    for _, r in R[R.sg >= .9].iterrows():
        nb = hi_recs[(hi_recs.seg == r.seg) & ((hi_recs.day - r.day).abs() <= k) & (hi_recs.fold != r.fold) & (hi_recs.rec != r.rec)]
        rows.append(dict(rec=r.rec, y=r.y, sg=r.sg, p2=r.p2, n_nb=len(nb), curve=nb.y.median() if len(nb) else np.nan, day=r.day))
    T = pd.DataFrame(rows); A = T.dropna(subset=["curve"])
    print(f"\n== k={k}: flagged (sg day-mean >= .9) {len(T)} (actual >=1: {(T.y >= 1).sum()}) | with curve {len(A)} (actual >=1: {(A.y >= 1).sum()}, pass-2 {A.p2.sum()})")
    for nm, S in (("all with curve", A), ("actual high", A[A.y >= 1]), ("actual normal", A[A.y < 1]), ("pass-2", A[A.p2])):
        if len(S): print(f"   {nm:15s} n {len(S):2d} | day-level RMSE model {rm(S.sg - S.y):.3f} | curve {rm(S.curve - S.y):.3f} | half {rm(.5 * (S.sg + S.curve) - S.y):.3f} | curve closer {np.mean(abs(S.curve - S.y) < abs(S.sg - S.y)):.2f}")
    if k == 3: print(A.sort_values("y", ascending=False)[["rec", "day", "y", "sg", "curve", "n_nb", "p2"]].round(3).to_string(index=False))
# eval: flagged records and date availability via exact 0h outdoor twin with a dated training record
P = pd.read_csv(os.path.join(env.ROOT, u"제출", u"13회차_2026-10-12(나)", [f for f in os.listdir(os.path.join(env.ROOT, u"제출", u"13회차_2026-10-12(나)")) if f.endswith(".csv")][0]))
print("\n13th submission columns:", P.columns.tolist()[:5])
