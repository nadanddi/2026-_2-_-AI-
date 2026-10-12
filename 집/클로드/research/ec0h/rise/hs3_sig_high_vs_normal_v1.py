# -*- coding: utf-8 -*-
"""HS3 (2026-10-12 집 클로드; HS2 후속).  서술.  지문(SIG) 기록 중 고EC(학습 17) vs 보통(학습 16)을 무엇이 가르나.
(1) 변수별 하루 평균·0~5시 평균: 두 집단 중앙값과 분리도(AUC).  (2) 모델(DIAG10 sg) 예측: SIG 보통 기록을 높게 냈나.
(3) 평가 SIG 기록 7개(82~88일)의 같은 변수 값이 어느 집단 쪽인가."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))
D = pd.read_csv("hs2_v1.csv", index_col=0)
X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")])
X = X[X.row_id.str[:3].isin(["F13", "F47"])]; X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:].astype(int)
V = ["out_temp", "out_hum", "out_rad", "in_temp", "in_hum", "in_co2", "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]
F = X.groupby("rec")[V].mean().join(X[X.h <= 5].groupby("rec")[V].mean().add_suffix("@0-5")).join(X[(X.h >= 10) & (X.h <= 15)].groupby("rec")[V].mean().add_suffix("@10-15"))
O = pd.read_csv(os.path.join(env.ROOT, u"연구실", u"클로드", "results", "ec_rv1_preds_v1.csv")); O = O[(O.tag == "CUR") & (O.validator == "DIAG10")]; q = O.assign(rec=O.row_id.str[:7]).groupby("rec").sg.mean()
S = D[D.SIG].join(F); S["q"] = q
H = S[(S.split == "train") & (S.y >= 1)]; L = S[(S.split == "train") & (S.y < 1)]
def auc(a, b):
    a, b = a.dropna().values, b.dropna().values
    return np.nan if len(a) == 0 or len(b) == 0 else float(((a[:, None] > b[None, :]).mean() + .5 * (a[:, None] == b[None, :]).mean()))
R = pd.DataFrame([dict(v=c, high_med=H[c].median(), normal_med=L[c].median(), auc=auc(H[c], L[c])) for c in F.columns]).set_index("v")
R["sep"] = (R.auc - .5).abs()
pd.set_option("display.width", 220)
print(f"SIG train records: high {len(H)}, normal {len(L)}")
print(R.sort_values("sep", ascending=False).head(15).round(3).to_string())
print("\nmodel DIAG10 day-mean prediction on SIG normal records:"); print(L[["day", "y", "q"]].round(2).sort_values("day").to_string())
print("\nmodel on SIG high records:"); print(H[["day", "y", "q"]].round(2).sort_values("day").to_string())
top = list(R.sort_values("sep", ascending=False).head(6).index)
print("\neval SIG records vs group medians:"); E = S[S.split == "eval"]
print(pd.concat([E[["day", "pred"] + top], pd.DataFrame([H[top].median().rename("TRAIN high median"), L[top].median().rename("TRAIN normal median")])]).round(2).to_string())
