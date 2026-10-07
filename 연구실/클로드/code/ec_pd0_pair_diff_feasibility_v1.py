# -*- coding: utf-8 -*-
"""PD0 — 같은 날짜 짝 차이 모델 타당성 검산 (계획 전 전제 검산, 진단 전용) — 2026-10-07 연구실 클로드
질문: 같은 날짜(외기 24h 완전 쌍둥이, z-RMSE ≤ .05) 정답 기록 쌍 (a, b)에서
  b 의 하루 평균 EC 를  y_a + g(입력 b − 입력 a)  로 추정하면, 모델(DIAG10 OOF, 계절 R3+DP1 시드평균)의 하루 평균보다 정확한가?
쌍 범위: 같은 온실 쌍(within) / 다른 온실 쌍(cross, F13↔F47).
g: 릿지(표준화, alpha 1) 와 상수(=Δ 평균, 즉 y_a + 평균차) 와 0(=y_a 그대로). 날짜 묶음 단위 LOO(같은 쌍둥이 묶음 전체를 뺌).
입력 요약(하루 전체 — 타당성 검산이라 시각 인과 아님, 통과 시 계획에서 인과판으로): 실내 온습도·CO2·일사, 구동기 하루 평균 9, 밤 실내온도, 낮 CO2.
비교: 모델 하루 오차(같은 b, DIAG10 OOF). 일반 날(y_b<1)/고EC 따로.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

R = os.path.dirname(os.path.abspath(__file__))
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
for T in (X, Y):
    T["farm"], T["day"], T["hour"] = T.row_id.str[:3], T.row_id.str[4:7].astype(int), T.row_id.str[8:10].astype(int)
X = X[X.farm.isin(["F13", "F47"])]
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]
WV = X.pivot_table(index=["farm", "day"], columns="hour", values=W)
WZ = (WV - WV.mean()) / WV.std()          # 검산용 전역 표준화(학습 자료만)
ym = Y.groupby(["farm", "day"]).sub_ec.mean()
g = X.groupby(["farm", "day"]); night = X[X.hour <= 5].groupby(["farm", "day"]); noon = X[(X.hour >= 10) & (X.hour <= 15)].groupby(["farm", "day"])
ACT = ["act_vent", "act_side", "act_shade", "act_thermal", "act_valve", "act_heating", "act_circfan", "act_co2", "act_fog"]
F = pd.DataFrame({"in_temp": g.in_temp.mean(), "in_hum": g.in_hum.mean(), "in_co2": g.in_co2.mean(), "in_rad": g.in_rad.mean(),
                  "nT": night.in_temp.mean(), "nH": night.in_hum.mean(), "dC": noon.in_co2.mean(), "Trng": g.in_temp.max() - g.in_temp.min(),
                  **{a: g[a].mean() for a in ACT}}).fillna(0)
days = pd.read_csv(os.path.join(R, "..", "results", "ec_err01_days_DIAG10_v1.csv"))[["farm", "day", "p"]].set_index(["farm", "day"]).p
keys = [k for k in WZ.index if k in ym.index]
A = WZ.loc[keys].values
D = np.sqrt(np.nanmean((A[:, None, :] - A[None, :, :]) ** 2, axis=2))
pairs = []
for i in range(len(keys)):
    for j in range(len(keys)):
        if i != j and D[i, j] <= .05:
            pairs.append((keys[i], keys[j]))
# 쌍둥이 묶음(연결 성분) = 날짜
par = list(range(len(keys)))
def fd(x):
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x
idx = {k: i for i, k in enumerate(keys)}
for a, b in pairs:
    par[fd(idx[a])] = fd(idx[b])
P = pd.DataFrame([dict(fa=a[0], da=a[1], fb=b[0], db=b[1], grp=fd(idx[a])) for a, b in pairs])
P["kind"] = np.where(P.fa == P.fb, "within", "cross")
P["ya"] = [ym[(f, d)] for f, d in zip(P.fa, P.da)]; P["yb"] = [ym[(f, d)] for f, d in zip(P.fb, P.db)]
P["pb"] = [days.get((f, d), np.nan) for f, d in zip(P.fb, P.db)]
dX = np.array([F.loc[(fb, db)].values - F.loc[(fa, da)].values for fa, da, fb, db in zip(P.fa, P.da, P.fb, P.db)])
print("쌍둥이 쌍 %d (같은 온실 %d, 다른 온실 %d), 날짜 묶음 %d, 묶음 크기 분포 %s" % (
    len(P), (P.kind == "within").sum(), (P.kind == "cross").sum(), P.grp.nunique(), P.groupby("grp").size().value_counts().sort_index().to_dict()))
rm = lambda e: np.sqrt(np.nanmean(np.square(e)))
for kind in ("within", "cross", "all"):
    m = (P.kind == kind) if kind != "all" else np.ones(len(P), bool)
    Q = P[m].reset_index(drop=True); Z = dX[m]; dy = (Q.yb - Q.ya).values
    est_r = np.full(len(Q), np.nan); est_c = np.full(len(Q), np.nan)
    for gname in Q.grp.unique():
        te = (Q.grp == gname).values; tr = ~te
        sc = StandardScaler().fit(Z[tr]); r = Ridge(alpha=1.0).fit(sc.transform(Z[tr]), dy[tr])
        est_r[te] = Q.ya[te] + r.predict(sc.transform(Z[te])); est_c[te] = Q.ya[te] + dy[tr].mean()
    ok = Q.pb.notna().values
    print("\n[%s] 쌍 %d (모델 OOF 있는 쌍 %d) | |Δy| 중앙 %.3f, Δy 표준편차 %.3f" % (kind, len(Q), ok.sum(), np.median(np.abs(dy)), dy.std()))
    for nm, mm in (("전체", ok), ("일반(y_b<1)", ok & (Q.yb < 1).values), ("고EC(y_b≥1)", ok & (Q.yb >= 1).values)):
        print("  %-11s n=%3d | 모델 %.3f | 짝 정답 그대로 %.3f | +평균차 %.3f | +릿지 Δ %.3f | 모델·릿지 평균 %.3f" % (
            nm, mm.sum(), rm(Q.pb[mm] - Q.yb[mm]), rm(Q.ya[mm] - Q.yb[mm]), rm(est_c[mm] - Q.yb[mm]), rm(est_r[mm] - Q.yb[mm]),
            rm((est_r[mm] + Q.pb[mm]) / 2 - Q.yb[mm])))
    print("  릿지 Δ 예측의 설명력(LOO R²) %.2f" % (1 - np.mean((est_r - Q.ya - dy) ** 2) / dy.var()))
