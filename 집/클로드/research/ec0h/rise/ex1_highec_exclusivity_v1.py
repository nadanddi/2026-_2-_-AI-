# -*- coding: utf-8 -*-
"""EX1 (2026-10-13 집 클로드; 사용자: '뭐라도 찾아봐') 서술(신호 확인): '하루에 고EC 동은 보통 하나'(6.690) — 배타성.
같은 날짜 묶음 = 0시 외기 4칸이 정확히 같은 기록들(학습·평가 모두 입력으로 묶음, 정답은 학습만). 날짜별 학습 고EC(하루 평균 ≥ 1) 개수 분포.
대상 r(학습, DIAG10)마다: 같은 날짜의 '다른' 학습 기록 중 고EC 가 있나(slot_taken), 그 수, 대상 모델 예측(sg 하루 평균)·정답.
질문: 모델이 높게(≥ .9) 예측한 기록 중 slot_taken 이면 실제로 보통일 확률이 높은가? (가짜 고EC 거르기)  slot 비었으면 실제 고EC 확률은?"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); sys.path.insert(0, r"C:\work\farmai\집\클로드\research\linker_v4")
import env
import numpy as np, pandas as pd
from feat4 import X, FARM, NUM, N, TRAIN, EC
rid = lambda i: f"{FARM[i]}_{NUM[i]:03d}"
key = {rid(i): tuple(np.round(X[i, 0], 4)) for i in range(N) if FARM[i] in ("F13", "F47")}
grp = {}
for r, k in key.items(): grp.setdefault(k, []).append(r)
YM = {rid(i): float(np.nanmean(EC[i])) for i in range(N) if TRAIN[i]}
sizes = pd.Series([len(v) for v in grp.values()]).value_counts().sort_index().to_dict()
nh = [sum(YM.get(r, 0) >= 1 for r in v if r in YM) for v in grp.values() if all(r in YM for r in v)]
print("date groups", len(grp), "| size distribution", sizes)
print("fully-labelled date groups:", len(nh), "| # high-EC per date:", pd.Series(nh).value_counts().sort_index().to_dict())
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")].copy(); O["rec"] = O.row_id.str[:7]
R = O.groupby("rec").agg(y=("sub_ec", "mean"), sg=("sg", "mean"))
rows = []
for r, row in R.iterrows():
    mates = [m for m in grp[key[r]] if m != r]; lab = [m for m in mates if m in YM]
    rows.append(dict(rec=r, y=row.y, sg=row.sg, p2=int(r[4:7]) >= 179, n_mates=len(mates), n_lab=len(lab), n_high=sum(YM[m] >= 1 for m in lab), n_unlab=len(mates) - len(lab)))
T = pd.DataFrame(rows); T["taken"] = T.n_high > 0; T["hi"] = T.y >= 1
pd.set_option("display.width", 200)
for thr in (.9, .8):
    F = T[T.sg >= thr]
    print(f"\nflagged sg>={thr}: {len(F)} | actual high {F.hi.sum()}")
    print(F.groupby("taken").agg(n=("rec", "size"), actual_high=("hi", "sum"), mean_y=("y", "mean"), mean_sg=("sg", "mean")).round(3).to_string())
print("\nall records: P(actual high | taken) vs not:", T.groupby("taken").hi.mean().round(3).to_dict(), "| n", T.groupby("taken").size().to_dict())
print("\nflagged & taken records:"); print(T[(T.sg >= .9) & T.taken].sort_values("y")[["rec", "y", "sg", "n_lab", "n_high", "n_unlab", "p2"]].round(3).to_string(index=False))
print("\nbig over-predictions (sg - y > .3):"); print(T[T.sg - T.y > .3].sort_values("y")[["rec", "y", "sg", "n_lab", "n_high", "n_unlab", "p2"]].round(3).to_string(index=False))
