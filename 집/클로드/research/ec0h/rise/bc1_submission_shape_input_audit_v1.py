# -*- coding: utf-8 -*-
"""BC1 (2026-10-12 집 클로드, 기초 점검, 서술): (1) 12회차 제출본(submission_19, 읽기만) 예측 모양 — 시간당 변화가 학습 정답의 같은 시각 분포에서
얼마나 벗어나는지(EC·온도), 지그재그(부호 바뀌는 연속 큰 변화) 기록; (2) 평가 입력 점검 — 결측, 멈춘 값(6시간 이상 같은 값), 학습 범위 밖 값,
기록 안 실내 급변(같은 시각 학습 분포 대비 z)."""
import sys, os
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env
import numpy as np, pandas as pd
S = pd.read_csv(r"C:\work\farmai\제출\12회차_2026-10-11(나)\submission_19.csv"); S["rec"] = S.row_id.str[:7]; S["h"] = S.row_id.str[8:].astype(int)
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y = Y[Y.row_id.str[:3].isin(["F13", "F47"])].copy(); Y["rec"] = Y.row_id.str[:7]; Y["h"] = Y.row_id.str[8:].astype(int)
print("(1) submission shape vs training labels: hourly change |d| by hour")
for col in ("sub_ec", "sub_temp"):
    P = S.pivot(index="rec", columns="h", values=col); L = Y.pivot(index="rec", columns="h", values=col)
    dP, dL = P.diff(axis=1).iloc[:, 1:], L.diff(axis=1).iloc[:, 1:]
    q99 = dL.abs().quantile(.99); q50 = dL.abs().median()
    print(f"  {col}: median |d| train {q50.median():.3f} vs submission {dP.abs().median().median():.3f} | cells above train 99th pct of same hour: {int((dP.abs() > q99).sum().sum())} / {dP.size} (expected ~{dP.size//100})")
    ex = (dP.abs() / q99).stack().sort_values(ascending=False).head(12)
    print("   largest (rec, hour, |d| / train p99):", ", ".join(f"{r}@{h} {v:.1f}" for (r, h), v in ex.items()))
    zz = []
    for r in P.index:
        d = dP.loc[r].values; big = np.abs(d) > q99.values
        flips = sum(1 for t in range(len(d) - 1) if big[t] and big[t + 1] and np.sign(d[t]) != np.sign(d[t + 1]))
        if flips: zz.append((r, flips))
    print(f"   zigzag records (consecutive opposite big changes): {zz}")
print("\n(2) evaluation input audit (F13/F47 test_X)")
TX = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); TX = TX[TX.row_id.str[:3].isin(["F13", "F47"])].copy(); TX["rec"] = TX.row_id.str[:7]; TX["h"] = TX.row_id.str[8:].astype(int)
RX = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); RX = RX[RX.row_id.str[:3].isin(["F13", "F47"])].copy(); RX["rec"] = RX.row_id.str[:7]; RX["h"] = RX.row_id.str[8:].astype(int)
cols = [c for c in TX.columns if c not in ("row_id", "rec", "h") and TX[c].notna().any()]
print("  missing cells by column:", {c: int(TX[c].isna().sum()) for c in cols if TX[c].isna().any()})
lo, hi = RX[cols].quantile(.001), RX[cols].quantile(.999)
out = {c: int(((TX[c] < lo[c]) | (TX[c] > hi[c])).sum()) for c in cols}
print("  cells outside training 0.1-99.9% range:", {c: v for c, v in out.items() if v})
for c in ("in_temp", "out_temp"):
    print(f"   {c}: eval min {TX[c].min():.1f} (train min {RX[c].min():.1f}), eval rows < train 1% ({RX[c].quantile(.01):.1f}): {int((TX[c] < RX[c].quantile(.01)).sum())}")
stuck = []
for r, G in TX.groupby("rec"):
    for c in ("in_temp", "in_hum", "in_co2"):
        v = G.sort_values("h")[c].values; run = best = 1
        for t in range(1, 24):
            run = run + 1 if v[t] == v[t - 1] else 1; best = max(best, run)
        if best >= 6: stuck.append((r, c, best))
print("  stuck runs >= 6h in indoor vars:", stuck)
dz = []
sdh = RX.sort_values(["rec", "h"]).groupby("rec")[["in_temp", "in_hum", "in_co2"]].diff().groupby(RX.sort_values(["rec", "h"]).h).std()
for r, G in TX.sort_values(["rec", "h"]).groupby("rec"):
    d = G[["in_temp", "in_hum", "in_co2"]].diff().values
    z = np.nanmax(np.abs(d[1:]) / sdh.values[1:], 0); hh = np.nanargmax(np.abs(d[1:]) / sdh.values[1:], 0) + 1
    for k, c in enumerate(("in_temp", "in_hum", "in_co2")):
        if z[k] > 6: dz.append((r, c, int(hh[k]), round(float(z[k]), 1)))
print("  within-record indoor jumps > 6 sd (same-hour training distribution):", dz)
