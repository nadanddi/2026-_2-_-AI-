# -*- coding: utf-8 -*-
"""재분석 22 — 물리 열수지 채널 가설 P1~P4 (2026-10-01, 집 클로드). 사전 고정, 진단(모델 아님).

배지 열수지: dT_sub = 공기 교환 + 바닥·땅 교환 + 관수 열 + 증발 냉각 + 근권 가온 + 일사.
G_C2 하루 오프셋 e_day(DIAG10, 예측−실제; +면 배지가 예측보다 차가움)가 '공기 외 열 출입'의 대리값.
각 채널 지표와 e_day의 부분 순위상관(온실별), 통제: 그날 실내온도 평균·외기 평균.

P1 땅 열 상태: 이전 기록 N개(현재 행 이전, 같은 온실 기록 순서)의 외기 평균. N=6(약 3일), 14(약 1주) 기록. 예측: 낮을수록 e_day ↑ (부호 −)
P2 증발 냉각: 낮(8~17시) 평균 VPD_in × 일사합. 예측: 클수록 e_day ↑ (부호 +)
P3 찬 관수(일사비례): 일사합 × max(0, 15 − 외기 평균). 예측: 클수록 e_day ↑ (부호 +)
P4 근권 가온: 밤(0~6시) max(0, 10 − 외기) × (100 − 난방)/100. 예측: 클수록 e_day ↓ (부호 −)
판정 (실행 전 고정): 두 온실 모두 예측 부호, |부분 ρ| ≥ 0.20, p < 0.05/8 (가설 4 × 온실 2).
합격 채널은 이후 인과 특징 가설로 넘김(별도 사전 고정). 결과 local/re22_physics_channels.txt
"""
import env  # noqa
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

out = open(env.LOCAL + "/re22_physics_channels.txt", "w", encoding="utf-8")
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")

G = pd.read_csv("re17_day_offsets.csv")
X = pd.read_csv(env.DATA + "/train_X.csv")
X = X[X.row_id.str[:3].isin(["F13", "F47"])].copy()
k_ = X.row_id.str.extract(r"^(F\d+)_(\d+)_(\d+)$")
X["farm"], X["day"], X["hour"] = k_[0], k_[1].astype(int), k_[2].astype(int)
X = X.sort_values(["farm", "day", "hour"])
es = 0.6108 * np.exp(17.27 * X.in_temp / (X.in_temp + 237.3))
X["vpd"] = es * (1 - X.in_hum / 100)
rows = []
for f, q in X.groupby("farm"):
    dm = q.groupby("day").agg(out=("out_temp", "mean"), tin=("in_temp", "mean"), rad=("out_rad", "sum"))
    day_v = q[q.hour.between(8, 17)].groupby("day").vpd.mean()
    nq = q[q.hour.between(0, 6)]
    night = nq.groupby("day").apply(lambda z: np.nanmean(np.maximum(0, 10 - z.out_temp) * (100 - z.act_heating) / 100))
    dm["P2"] = day_v.reindex(dm.index) * dm.rad
    dm["P3"] = dm.rad * np.maximum(0, 15 - dm.out)
    dm["P4"] = night.reindex(dm.index)
    outs = dm.out.values
    for N in (6, 14):
        dm[f"P1_{N}"] = [np.nan if i < N else outs[i - N:i].mean() for i in range(len(outs))]  # 이전 기록만
    dm["farm"] = f
    rows.append(dm.reset_index())
D = pd.concat(rows).merge(G[["farm", "day", "temp_offset"]], on=["farm", "day"])

def partial_rho(x, y, Z):
    m = ~(np.isnan(x) | np.isnan(y) | np.isnan(Z).any(1))
    rx, ry = rankdata(x[m]), rankdata(y[m]); RZ = np.c_[np.ones(m.sum()), np.apply_along_axis(rankdata, 0, Z[m])]
    ex = rx - RZ @ np.linalg.lstsq(RZ, rx, rcond=None)[0]; ey = ry - RZ @ np.linalg.lstsq(RZ, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]; n = m.sum()
    from scipy.stats import t as T
    tt = r * np.sqrt((n - 2 - Z.shape[1]) / (1 - r ** 2))
    return r, 2 * T.sf(abs(tt), n - 2 - Z.shape[1]), n

hyp = [("P1_6", -1, "땅 열 상태: 이전 6기록 외기"), ("P1_14", -1, "땅 열 상태: 이전 14기록 외기"),
       ("P2", +1, "증발 냉각: 낮 VPD×일사"), ("P3", +1, "찬 관수: 일사×추위"), ("P4", -1, "근권 가온: 추운 밤×낮은 난방")]
thr = 0.05 / 8
p(f"날 {len(D)}. 판정 기준: 두 온실 예측 부호, |부분ρ|≥0.20, p<{thr:.4f}")
for col, sign, name in hyp:
    res = []
    ok = True
    for f in ("F13", "F47"):
        d = D[D.farm == f]
        r, pv, n = partial_rho(d[col].values, d.temp_offset.values, d[["tin", "out"]].values)
        r0 = spearmanr(d[col], d.temp_offset, nan_policy="omit")[0]
        res.append(f"{f} 부분ρ {r:+.3f} (p {pv:.4f}, n {n}; 단순ρ {r0:+.3f})")
        ok &= (np.sign(r) == sign) and abs(r) >= 0.20 and pv < thr
    p(f"{name:28s} 예측부호 {'+' if sign>0 else '−'} | " + " | ".join(res) + f" → {'합격' if ok else '불합격'}")
