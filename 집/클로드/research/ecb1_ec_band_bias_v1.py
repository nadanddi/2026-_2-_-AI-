# -*- coding: utf-8 -*-
"""ECB1 (2026-10-15 집 클로드; 사용자 목표 EC .108~.112 — 평가가 추움(DV1) → EC 도 추운 영역 치우침이 있나).  서술 분해(판정 없음), 실행 전 커밋.
nh1 저장 행(DIAG10·EL1·P2LOO, 3시드 NH 평균 = 27번 EC 기본) 의 오차를 실내온도 구간(≤ 6, 6~8, 8~10, 10~12, ≥ 12)·하루 최저 구간·시각대·고EC 여부로 분해:
행 수, RMSE, 평균 치우침(예측 − 정답).  평가 입력의 같은 구간 비율도 함께(정답 없음).
Run:  cd 집/클로드/research && PYTHONPATH="" python -u ecb1_ec_band_bias_v1.py
"""
import os, sys, pickle
import env  # noqa: F401
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
R = pd.read_csv(os.path.join(HERE, "nh1_rows_v1.csv.gz")); MT = pickle.load(open(os.path.join(env.LOCAL, "nh1_mt1_v1.pkl"), "rb"))
mt = np.array([MT[(v, k)][r][h] for v, k, r, h in zip(R.validator, R.fold, R.rec, R.hour)])
R["NH"] = np.mean([(R[f"A1_{s}"].values - .1 * mt) / .9 for s in (1357, 2468, 3579)], axis=0)
X = pd.concat([pd.read_csv(os.path.join(env.DATA, f)) for f in ("train_X.csv", "test_X.csv")]); X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
X["rec"] = X.row_id.str[:7]; X["hour"] = X.row_id.str[8:10].astype(int); X["dmin"] = X.groupby("rec").in_temp.transform("min")
R = R.merge(X[["rec", "hour", "in_temp", "dmin"]], on=["rec", "hour"], how="left")
bands = lambda t: pd.cut(t, [-99, 6, 8, 10, 12, 99], labels=["<=6", "6-8", "8-10", "10-12", ">=12"])
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
pd.set_option("display.width", 200)
for v in ("DIAG10", "EL1", "P2LOO"):
    g = R[R.validator == v].copy(); g["e"] = g.NH - g.y
    for col, nm in (("in_temp", "row in_temp"), ("dmin", "record day-min")):
        t = g.groupby(bands(g[col]), observed=False).agg(rows=("e", "size"), rmse=("e", rm), bias=("e", "mean"), y=("y", "mean"))
        print(f"{v} by {nm}:", t.round(3).to_dict("index"))
    print(f"{v} high-EC days (y day mean >= 1) bias by day-min:", g[g.hi].groupby(bands(g.dmin[g.hi]), observed=False).e.mean().round(3).to_dict(), "| normal:", g[~g.hi].groupby(bands(g.dmin[~g.hi]), observed=False).e.mean().round(3).to_dict())
E = X[X.row_id.isin(pd.read_csv(os.path.join(env.DATA, "test_X.csv")).row_id)]
print("EVAL rows by in_temp band:", E.groupby(bands(E.in_temp), observed=False).size().to_dict(), "| by record day-min:", E.groupby(bands(E.dmin), observed=False).size().to_dict())
