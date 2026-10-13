# -*- coding: utf-8 -*-
"""P2T2B (2026-10-14 집 클로드; 사용자: '보통 날 층과 고EC 날 층을 구분할 수 있겠어?').  서술(사후 — P2T2 결과를 본 뒤, 판정 아님).
P2T2 저장 예측(같은 폴드·시드): ALL(전체 학습) 과 P2C2(2차 + 같은 시기 1차). 두 층: 관문이 '보통' 이면 P2C2, '고EC' 면 ALL.
관문: (오라클) 실제 하루 평균 >= 1.0;  (인과) 그 시각까지 ALL 예측 누적평균 >= 1.0 (지금까지 써온 기준);  참고로 문턱 .8 / 1.2.
보고: 관문 정확도(날 단위: 23시 기준), 두 층 RMSE vs ALL, 잘못 보낸 날의 손해."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import env  # noqa: F401
import numpy as np, pandas as pd
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
D = pd.read_csv(os.path.join(HERE, "p2t2_preds_v1.csv"))
A = D[D.arm == "ALL"].reset_index(drop=True); C = D[D.arm == "P2C2"].reset_index(drop=True)
assert (A[["fold", "seed", "farm", "day", "y"]].values == C[["fold", "seed", "farm", "day", "y"]].values).all()
A["h"] = A.groupby(["fold", "seed", "farm", "day"]).cumcount(); A["cm"] = A.groupby(["fold", "seed", "farm", "day"]).p.transform(lambda s: s.expanding().mean())
rows = []
for nm, g in [("oracle", A.hi.values)] + [(f"causal {t}", (A.cm >= t).values) for t in (.8, 1.0, 1.2)]:
    for sd in (47, 1414, 6464):
        m = (A.seed == sd).values; p = np.where(g, A.p, C.p)[m]; y = A.y.values[m]
        rows.append(dict(gate=nm, seed=sd, all=rm(A.p.values[m] - y), two_layer=rm(p - y), normal=rm((p - y)[~A.hi.values[m]]), high=rm((p - y)[A.hi.values[m]])))
S = pd.DataFrame(rows).groupby("gate")[["all", "two_layer", "normal", "high"]].mean(); S["pct_vs_all"] = 100 * (S.two_layer / S["all"] - 1)
pd.set_option("display.width", 200); print("2-layer (gate normal -> P2C2, high -> ALL), pass-2 validation, seed mean"); print(S.round(4).to_string())
day = A[A.h == 23].groupby(["farm", "day"]).agg(hi=("hi", "first"), cm=("cm", "mean")); day["pred_hi"] = day.cm >= 1.0
print("\nday-level gate at 23h (threshold 1.0, seed-mean cm): true high", int(day.hi.sum()), "| predicted high", int(day.pred_hi.sum()), "| caught", int((day.hi & day.pred_hi).sum()), "| false high", int((~day.hi & day.pred_hi).sum()), "| missed", int((day.hi & ~day.pred_hi).sum()))
print("hour-by-hour share of true-high rows routed high:", {h: round(float(A[(A.h == h) & A.hi].cm.ge(1.0).mean()), 2) for h in (0, 3, 6, 9, 12, 18, 23)})
print("hour-by-hour share of true-normal rows routed high (false):", {h: round(float(A[(A.h == h) & ~A.hi].cm.ge(1.0).mean()), 3) for h in (0, 3, 6, 9, 12, 18, 23)})
