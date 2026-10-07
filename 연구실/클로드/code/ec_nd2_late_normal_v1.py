# -*- coding: utf-8 -*-
"""ND2 — 2차 구간 일반 날(하루 평균 정답 < 1) 하루 수준 오차 해부 (진단 전용, 채택 판정 아님) — 2026-10-06 연구실 클로드

왜: 평가 60일은 모두 2차이고, 9회차 EC .1384 산술상 평가 오차 대부분이 일반 날 하루 수준(ec_err 대화 기록).
기준 예측: SG2 저장 OOF(ec3_SG2_all.csv)의 sg 시드평균(R3S + SG2; 제출 구성에서 DP1·TabPFN 0.2 빠짐), 비교로 base(R3S).
검증 배치 3개(DIAG10, DIAG10y, EL1) — 같은 2차 46일(일반 41일)을 서로 다른 학습 조건에서 예측.
질문:
 Q1 크기: 일반 날 오차 중 하루 수준 몫, 날 집중도, 세 배치 공통으로 크게 틀린 날
 Q2 방향: 과대/과소, 정답 수준에 따른 압축(정답 낮으면 과대·높으면 과소인지), 온실·기록 역할·밀폐·추위별
 Q3 정보: 하루 수준 잔차(정답−예측)와 (a) SG2 이웃 1순위 정답 a1 (b) 같은 날짜 짝 기록 정답
    (c) 같은 온실 달력 ±3일 정답 평균 (d) 정답 사슬 앞 기록 23시(상한, 불법) (e) 하루 입력 요약 의 관계
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
from scipy.stats import spearmanr

R = os.path.dirname(os.path.abspath(__file__))
H = os.path.join(env.ROOT, u"집", u"클로드", "research", "local")
S = pd.read_csv(os.path.join(H, "ec3_SG2_all.csv"))
S["sg"] = S[["sg_23", "sg_808", "sg_9090"]].mean(axis=1); S["bs"] = S[["base_23", "base_808", "base_9090"]].mean(axis=1)
day = S.groupby(["validator", "farm", "day"]).agg(y=("sub_ec", "mean"), sg=("sg", "mean"), bs=("bs", "mean"),
                                                  sse=("sg", lambda x: 0)).reset_index()
# 시간별 SSE 와 하루 수준 몫
S["e"] = S.sg - S.sub_ec
g = S.groupby(["validator", "farm", "day"]).e
day = day.drop(columns="sse").merge(g.apply(lambda x: (x ** 2).sum()).rename("sse").reset_index(), on=["validator", "farm", "day"])
day["n"] = 24; day["bias"] = day.sg - day.y; day["lvl"] = 24 * day.bias ** 2
info = pd.read_csv(os.path.join(R, "..", "results", "ec_err03_sources_days_v2.csv"))
keep = ["farm", "day", "role", "dong", "sealed", "cold", "A1", "CHo", "in_temp", "out_temp", "dT", "in_hum", "in_co2", "vent", "vent0",
        "heat", "thermal", "shade", "circfan", "co2sup", "fog", "out_rad", "wspd"]
day = day.merge(info[keep], on=["farm", "day"], how="left")
# (b) 같은 날짜 짝 기록 정답 / (c) 달력 ±3일 같은 온실 정답 평균
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y["farm"], Y["day"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int)
ym = Y.groupby(["farm", "day"]).sub_ec.mean()
roles = pd.read_csv(os.path.join(H, "st_record_roles_v1.csv")).set_index(["farm", "day"]).role
def sib(f, d):
    r = roles.get((f, d))
    o = d - 1 if r == "second" else d + 1 if r == "first" else None
    return ym.get((f, o), np.nan) if o is not None else np.nan
day["SIB"] = [sib(f, d) for f, d in zip(day.farm, day.day)]
cal = pd.read_csv(os.path.join(H, "cal_own_farm_v1.csv")).set_index(["farm", "day"]).cal
def calnb(f, d):
    c = cal.get((f, d))
    if c is None:
        return np.nan
    v = [ym[(ff, e)] for (ff, e), ce in cal.items() if ff == f and e != d and abs(ce - c) <= 3 and (ff, e) in ym.index]
    return np.mean(v) if v else np.nan
day["CAL3"] = [calnb(f, d) for f, d in zip(day.farm, day.day)]

NV = day[day.y < 1].copy()
NV["r"] = NV.y - NV.sg
print("=== Q1 크기 (2차 일반 41일) ===")
for v, q in NV.groupby("validator"):
    full = day[day.validator == v]
    s = q.sse.sort_values(ascending=False).values
    print("%-8s RMSE 일반 %.4f (base %.4f) | 하루 수준 몫 %.0f%% | 전체 2차 SSE 중 일반 날 %.0f%% | 상위 5날 %.0f%%, 10날 %.0f%% | 과대 %d일·과소 %d일, 평균 편향 %+.3f" % (
        v, np.sqrt(q.sse.sum() / (24 * len(q))), np.sqrt(((q.bs - q.y) ** 2).mean() + (q.sse.sum() / (24 * len(q)) - (q.bias ** 2).mean())),
        100 * q.lvl.sum() / q.sse.sum(), 100 * q.sse.sum() / full.sse.sum(), 100 * s[:5].sum() / s.sum(), 100 * s[:10].sum() / s.sum(),
        (q.bias > 0).sum(), (q.bias < 0).sum(), q.bias.mean()))
W = NV.pivot_table(index=["farm", "day"], columns="validator", values="bias")
W["mean"] = W.mean(axis=1); W["same_sign"] = (np.sign(W[["DIAG10", "DIAG10y", "EL1"]]).nunique(axis=1) == 1)
print("세 배치 하루 편향 상관: DIAG10–DIAG10y %.2f, DIAG10–EL1 %.2f, DIAG10y–EL1 %.2f | 세 배치 같은 부호 %d/41일" % (
    W.DIAG10.corr(W.DIAG10y), W.DIAG10.corr(W.EL1), W.DIAG10y.corr(W.EL1), W.same_sign.sum()))
A = NV.groupby(["farm", "day"]).agg(y=("y", "first"), sg=("sg", "mean"), bias=("bias", "mean"), sse=("sse", "mean"), role=("role", "first"),
                                     dong=("dong", "first"), sealed=("sealed", "first"), cold=("cold", "first"), A1=("A1", "first"),
                                     SIB=("SIB", "first"), CAL3=("CAL3", "first"), CHo=("CHo", "first"), in_temp=("in_temp", "first")).reset_index()
A["r"] = A.y - A.sg
T = A.sse.sum()
print("\n세 배치 평균 기준 최악 12일 (몫 = 일반 41일 SSE 중)")
w = A.sort_values("sse", ascending=False).head(12).assign(몫=lambda x: 100 * x.sse / T)
print(w[["farm", "day", "role", "dong", "sealed", "y", "sg", "bias", "A1", "SIB", "CAL3", "CHo", "in_temp", "몫"]].round(2).to_string(index=False))

print("\n=== Q2 방향 (세 배치 평균 하루 편향) ===")
print("정답 수준과 편향: Spearman(y, 예측−정답) %.2f | 정답 3분위별 평균 편향: %s" % (
    spearmanr(A.y, A.bias)[0], ", ".join("%s %+.3f" % (k, v) for k, v in A.groupby(pd.qcut(A.y, 3, labels=["낮음", "중간", "높음"]), observed=True).bias.mean().items())))
print("예측 폭: 정답 표준편차 %.3f vs 예측 %.3f, 정답→예측 기울기 %.2f" % (A.y.std(), A.sg.std(), np.polyfit(A.y, A.sg, 1)[0]))
for col in ("farm", "role", "dong", "sealed", "cold"):
    t = A.groupby(col).agg(일=("bias", "size"), 평균편향=("bias", "mean"), RMSE=("sse", lambda x: np.sqrt(x.sum() / (24 * len(x)))), 몫=("sse", lambda x: 100 * x.sum() / T))
    print("[%s]\n%s" % (col, t.round(3).to_string()))

print("\n=== Q3 정보: 하루 잔차 r = 정답−예측 과의 관계 (세 배치 평균 예측 기준) ===")
for c, nm in (("A1", "SG2 이웃 1순위 정답"), ("SIB", "같은 날짜 짝 기록 정답"), ("CAL3", "달력 ±3일 같은 온실 정답 평균"), ("CHo", "정답 사슬 앞 기록 23시(상한)")):
    m = A[c].notna()
    z = A[c][m] - A.sg[m]
    rho = spearmanr(z, A.r[m])[0]
    fit = np.polyfit(z, A.r[m], 1)
    res = A.r[m] - np.polyval(fit, z)
    print("  %-26s n=%2d  ρ(x−예측, r) %+.2f | 그 정보로 직접 대체 RMSE %.3f vs 예측 %.3f | 1차식 보정 후 하루 RMSE %.3f" % (
        nm, m.sum(), rho, np.sqrt(np.mean((A[c][m] - A.y[m]) ** 2)), np.sqrt(np.mean(A.r[m] ** 2)), np.sqrt(np.mean(res ** 2))))
IN = ["in_temp", "out_temp", "dT", "in_hum", "in_co2", "vent", "vent0", "heat", "thermal", "shade", "circfan", "co2sup", "fog", "out_rad", "wspd"]
X2 = A.merge(info[["farm", "day"] + [c for c in IN if c != "in_temp"]], on=["farm", "day"])
rr = {c: spearmanr(X2[c], X2.r, nan_policy="omit")[0] for c in IN}
print("  하루 입력 15개와 r 순위상관 상위: " + ", ".join("%s %+.2f" % (k, rr[k]) for k in sorted(rr, key=lambda k: -abs(rr[k]))[:6]))
nul = [max(abs(spearmanr(X2[c], np.random.default_rng(i).permutation(X2.r.values), nan_policy="omit")[0]) for c in IN) for i in range(500)]
print("  (순열 기준: 15개 중 최대 |ρ| 의 95%% 분위 %.2f)" % np.quantile(nul, .95))
A.to_csv(os.path.join(R, "..", "results", "ec_nd2_late_normal_days_v1.csv"), index=False, encoding="utf-8-sig")
