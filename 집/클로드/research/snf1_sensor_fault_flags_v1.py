# -*- coding: utf-8 -*-
"""SNF1 (2026-10-16 집 클로드; 사용자 '센서 쪽 이상 현상이 있으면 걸러내게 하는 게 필요한 것 같다').  서술(판정 없음), 실행 전 커밋(감지 규칙 고정).  온도.
센서 이상 자동 감지기(입력만, 같은 기록 안 — 평가에도 같은 규칙; 평가 규칙상 현재·이전 시각만 쓰도록 '그 시각까지' 판정):
  F1 직선 보간: 연속 센서(외기 온도·습도·일사·풍속, 실내 온도·습도·CO2)에서 그 시각까지 4시간 이상 |Δ²x| <= tol 이고 |Δx| > tol (tol = 열 일차차분 중앙 2%)
     (외기 일사는 밤 0 구간 제외: 값 > 0 일 때만)
  F2 값 멈춤: 실내 온도·CO2, 외기 온도에서 그 시각까지 6시간 이상 같은 값
  F3 실내 온도 불일치: 실내 온도를 같은 시각 외기 온도·일사·난방·환기·보온커튼·시각(sin, cos)으로 학습 자료에서 선형 적합(행 OOF 아님 — 학습 전체 1회 적합, 평가엔 고정식)
     → |잔차| > 4·(잔차 표준편차) 이면 표시
출력: 표시 종류별 학습/평가 행·기록 수; dgx5_rows(CV8·N·CK2·EL1, 19회차 구성 최종 예측)에서 표시 행 vs 나머지 RMSE·편향, 표시 기록의 하루 평균 오차; 평가 표시 기록 목록.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u snf1_sensor_fault_flags_v1.py
"""
import os
import env  # noqa: F401
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
HERE = os.path.dirname(os.path.abspath(__file__))
TR = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); TE = pd.read_csv(os.path.join(env.DATA, "test_X.csv"))
def prep(X, tag):
    X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy(); X["rec"] = X.row_id.str[:7]; X["h"] = X.row_id.str[8:10].astype(int); X["set"] = tag; return X.sort_values(["rec", "h"])
A = pd.concat([prep(TR, "train"), prep(TE, "eval")], ignore_index=True)
CONT = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2"]
TOL = {c: max(np.nanmedian(np.abs(np.diff(A[c].values))[np.abs(np.diff(A[c].values)) > 0]) * .02, 1e-6) for c in CONT}
def causal_run(flag_pt, L):
    out = np.zeros(len(flag_pt), bool); cnt = 0
    for i, f in enumerate(flag_pt):
        cnt = cnt + 1 if f else 0; out[i] = cnt >= L
    return out
F1 = np.zeros(len(A), bool); F2 = np.zeros(len(A), bool); F1c = {}
for r, g in A.groupby("rec"):
    ix = g.index.values
    for c in CONT:
        v = g[c].values.astype(float); d = np.r_[np.nan, np.diff(v)]; dd = np.r_[np.nan, np.diff(d)]
        pt = np.isfinite(dd) & (np.abs(dd) <= TOL[c]) & (np.abs(d) > TOL[c]) & ((v > 0) if c == "out_rad" else True)
        f = causal_run(pt, 3)                                     # 3 consecutive straight steps = >= 4 points on one line, judged at the current hour
        F1[ix] |= f
        if f.any(): F1c.setdefault(c, set()).add(r)
    for c in ("in_temp", "in_co2", "out_temp"):
        v = g[c].values.astype(float); same = np.r_[False, np.isfinite(v[1:]) & (np.diff(v) == 0)]; F2[ix] |= causal_run(same, 5)
A["hs"], A["hc"] = np.sin(2 * np.pi * A.h / 24), np.cos(2 * np.pi * A.h / 24)
PRED = ["out_temp", "out_rad", "act_heating", "act_vent", "act_thermal", "hs", "hc"]
trm = (A.set == "train") & A[PRED + ["in_temp"]].notna().all(1); lr = LinearRegression().fit(A.loc[trm, PRED], A.loc[trm, "in_temp"])
ok = A[PRED + ["in_temp"]].notna().all(1); res = np.full(len(A), np.nan); res[ok] = A.loc[ok, "in_temp"] - lr.predict(A.loc[ok, PRED]); s = np.nanstd(res[trm.values])
F3 = np.abs(np.nan_to_num(res)) > 4 * s
A["F1"], A["F2"], A["F3"] = F1, F2, F3; A["ANY"] = F1 | F2 | F3
pd.set_option("display.width", 200)
print("straight-line (F1) records by column:", {c: len(v) for c, v in F1c.items()})
print(A.groupby("set")[["F1", "F2", "F3", "ANY"]].agg(["sum"]).to_string())
print("records flagged (ANY):", A[A.ANY].groupby("set").rec.nunique().to_dict())
print("eval flagged records:", A[A.ANY & (A.set == "eval")].groupby("rec")[["F1", "F2", "F3"]].sum().to_dict("index"))
R = pd.read_csv(os.path.join(HERE, "dgx5_rows_v1.csv.gz")); R["row_id"] = R.rec + "_" + R.h.map("{:02d}".format)
R = R.merge(A[["row_id", "F1", "F2", "F3", "ANY"]], on="row_id", how="left"); R["e"] = R.fin - R.y
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
for f in ("F1", "F2", "F3", "ANY"):
    m = R[f].fillna(False).astype(bool); recs = set(R.rec[m])
    rin = R.rec.isin(recs)
    print(f"[{f}] flagged rows {int(m.sum())} ({R.rec[m].nunique()} recs) | RMSE flagged rows {rm(R.e[m]) if m.any() else float('nan'):.3f} vs others {rm(R.e[~m]):.3f} | "
          f"flagged-record whole-day RMSE {rm(R.e[rin]):.3f} vs non-flagged records {rm(R.e[~rin]):.3f} | |day mean err| median flagged recs {R[rin].groupby(['layout', 'rec']).e.mean().abs().median():.3f} vs {R[~rin].groupby(['layout', 'rec']).e.mean().abs().median():.3f}")
big = ["F13_107", "F13_109", "F13_139", "F13_157", "F13_175", "F13_217", "F13_244", "F47_018", "F47_095", "F47_096", "F47_105", "F47_123", "F47_132", "F47_145", "F47_178", "F47_191", "F47_214", "F47_239", "F47_240", "F47_241"]
fl = A[A.ANY & (A.set == "train")].groupby("rec")[["F1", "F2", "F3"]].sum()
print("large-error records caught:", {r: fl.loc[r].to_dict() for r in big if r in fl.index}, f"| {sum(r in fl.index for r in big)}/{len(big)} vs base rate {fl.shape[0] / 400:.2f}")
A[["row_id", "set", "F1", "F2", "F3", "ANY"]].to_csv(os.path.join(HERE, "snf1_flags_v1.csv"), index=False)
