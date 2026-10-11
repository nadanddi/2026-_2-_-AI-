# -*- coding: utf-8 -*-
"""PS0 (2026-10-12 집 클로드, 새 방향 상한 점검): EC 모델의 하루 평균(수준)은 그대로 두고, 하루 안 모양만 물리식으로 교체.
pred_h = m * f_h / mean(f),  f_h = 1 + .02 (T_h − 25),  m = 모델 하루 평균.  T = 정답 배지온(오라클 상한) — 상한이 의미 없으면 중단.
비교: 모델 원 예측 / 물리 모양(오라클 T) / 반반 섞기 / 완전 평평(m).  DIAG10 CUR sg 시드 평균, 1·2차, 고EC(0시 정답 >=1)·보통 따로."""
import sys, os; sys.path.insert(0, r"C:\work\farmai\집\클로드\research"); import env
import numpy as np, pandas as pd
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]
O["rec"] = O.row_id.str[:7]; Q = O.groupby(["rec", "hour"]).sg.mean().unstack(); Y = O.groupby(["rec", "hour"]).sub_ec.first().unstack()
Ty = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Ty["rec"] = Ty.row_id.str[:7]; Ty["hour"] = Ty.row_id.str[8:].astype(int)
T = Ty.pivot(index="rec", columns="hour", values="sub_temp").reindex(Q.index)
f = 1 + .02 * (T - 25); m = Q.mean(axis=1)
P_phys = f.div(f.mean(axis=1), axis=0).mul(m, axis=0); P_flat = pd.DataFrame(np.repeat(m.values[:, None], 24, 1), index=Q.index, columns=Q.columns)
rm = lambda P, idx: float(np.sqrt(((P.loc[idx] - Y.loc[idx]) ** 2).values.mean()))
p2 = Q.index.str[4:7].astype(int) >= 179; hi = Y[0] >= 1
for nm, s in (("pass-1", ~p2), ("pass-2", p2), ("pass-1 high", ~p2 & hi), ("pass-1 normal", ~p2 & ~hi), ("pass-2 high", p2 & hi)):
    idx = Q.index[s]
    if len(idx) == 0: continue
    print(f"{nm:14s} n {len(idx):3d} | model {rm(Q, idx):.4f} | physics-shape(oracle T) {rm(P_phys, idx):.4f} | half {rm((Q + P_phys) / 2, idx):.4f} | flat {rm(P_flat, idx):.4f} | perfect shape (true day mean offset kept) {rm(Y.sub(Y.mean(axis=1), axis=0).add(m, axis=0), idx):.4f}")
