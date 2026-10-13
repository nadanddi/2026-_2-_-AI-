# -*- coding: utf-8 -*-
"""DG2 (2026-10-14 집 클로드; 사용자: '1일이지만 그 안에 24개 데이터가 있지 않나').  서술(판정 아님).  EC.
앵커는 전날 23시 EC 하나만 씀.  전날 24시간 곡선(추세·변동)이 오늘 수준에 추가 정보를 주나?
쌍: DG1 과 같은 확실한 1일 쌍(CP2 1일 연결, 잠금 제외), 기본 = DIAG10 CUR OOF.  목표: 오늘 기본 예측 잔차의 하루 평균 r_b = mean_h(y_b − base_b).
설명 변수(전날 a 의 정답 곡선에서만 — 평가에서도 전날이 학습 기록이면 쓸 수 있음):
  F0 = A23 − mean_h base_b(오늘 기본의 하루 평균)  (앵커 정보 그 자체)
  F1 = 전날 추세: A23 − A12 (오후 기울기),  F2 = A23 − A18,  F3 = 전날 하루 평균 − A23,  F4 = 전날 하루 안 표준편차,  F5 = 전날 0시 − 23시
교차 적합(쌍 단위 10겹) 선형 회귀 R2: [F0] vs [F0 + F1..F5];  고EC/보통·1차/2차 분해;  급락(오늘 하루 평균 − 전날 하루 평균 <= −.3) 예측: F1 의 AUC.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
T = pd.read_csv(os.path.join(HERE, "cp2_tracks_v1.csv")); A = pd.read_csv(os.path.join(HERE, "cp2_rows_v1.csv.gz"))
A = A[A.is_train & ~A.lock].sort_values(["rec", "hour"]); Y = {r: g.y.values for r, g in A.groupby("rec")}; B = {r: g.base_oof.values for r, g in A.groupby("rec")}
T = T[T.train & ~T.lock]
P = T[(T.link_type == "day1") & T.prev.isin(Y.keys()) & T.rec.isin(Y.keys())][["rec", "prev"]].values
rows = []
for b, a in P:
    ya, yb, bb = Y[a], Y[b], B[b]
    if np.isnan(ya).any() or np.isnan(yb).any() or np.isnan(bb).any(): continue
    rows.append(dict(b=b, a=a, r=float(np.mean(yb - bb)), F0=ya[23] - bb.mean(), F1=ya[23] - ya[12], F2=ya[23] - ya[18], F3=ya.mean() - ya[23], F4=ya.std(), F5=ya[0] - ya[23],
                     hi=yb.mean() >= 1.0, p2=int(b[4:]) >= 179, fall=(yb.mean() - ya.mean()) <= -.3, ya_hi=ya.mean() >= 1.0))
D = pd.DataFrame(rows); print("pairs", len(D), "| high today", int(D.hi.sum()), "| drops (today - yesterday <= -.3)", int(D.fall.sum()), "| yesterday high", int(D.ya_hi.sum()))
def cv_r2(cols, m=None):
    G = D if m is None else D[m]
    if len(G) < 15: return np.nan, len(G)
    pred = np.zeros(len(G)); X = G[cols].values; y = G.r.values
    for tr, te in KFold(10, shuffle=True, random_state=0).split(X):
        pred[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2), len(G)
for nm, m in (("all", None), ("today normal", ~D.hi), ("today high", D.hi), ("yesterday high", D.ya_hi), ("pass-1", ~D.p2), ("pass-2", D.p2)):
    r0, n = cv_r2(["F0"], m); r1, _ = cv_r2(["F0", "F1", "F2", "F3", "F4", "F5"], m)
    print(f"  {nm:15s} n {n:3d} | CV R2 anchor-only {r0:.3f} -> + yesterday curve {r1:.3f}")
print("single-feature corr with residual (all):", {c: round(float(np.corrcoef(D[c], D.r)[0, 1]), 3) for c in ["F0", "F1", "F2", "F3", "F4", "F5"]})
G = D[D.ya_hi]
if G.fall.nunique() == 2: print(f"yesterday-high pairs {len(G)}: drops {int(G.fall.sum())} | AUC of afternoon trend F1 (lower = falling) for drop: {roc_auc_score(G.fall, -G.F1):.3f} | F3 {roc_auc_score(G.fall, G.F3):.3f}")
print(D[D.fall][["b", "a", "F0", "F1", "F2", "F3", "r"]].round(3).to_string(index=False))
D.to_csv(os.path.join(HERE, "dg2_pairs_v1.csv"), index=False)
