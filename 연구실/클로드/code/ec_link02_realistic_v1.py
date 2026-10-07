# -*- coding: utf-8 -*-
"""LINK2 (LINK1 쌍 특징 재사용, 평가 조건 재현은 파일 아래쪽 블록에 고정) / 원 설명: LINK1 — '같은 출처 앞날' 찾기를 학습된 연결 판별기로 (진단, 실행 전 기준 고정) — 2026-10-06 연구실 클로드

배경: 같은 출처면 0시 EC ≈ 앞날 23시 EC(6.345). 정답으로 고른 앞날을 쓰면 고EC 크기 오차 −51%(ec_err03),
2차 일반 날 .145→.083(6.346). 그러나 입력만 쓴 손 비용식(외기·실내 자정 연속)으로는 앞날 적중 36~41%.
이번 차이: 비용식 대신, 정답 있는 기록 쌍에서 '진짜 연결'을 정답으로 삼아 연결 판별기를 학습한다.

정답 연결(학습·채점용): CH2 방식(EC 자정 끊김 + 외기 + 실내 비용, 1:1 헝가리안)으로 만든 연결 중
  |EC(b,0시) − EC(a,23시)| < .05 인 것. (정답을 쓰는 것은 '정답 만들기'에만; 판별기 입력에는 EC 없음)
판별기 입력(쌍 a→b, 같은 온실, 둘 다 입력만):
  외기 4채널 자정 점프(추세 보정, 표준화 절댓값) + 합, 실내 3채널 점프, 구동기 9개 23시→0시 차,
  하루 운영 서명 거리(구동기 하루 평균 차 9 + 밤 실내온도 차), 기록 번호 차(b−a), 두 기록 구간(1차/2차).
모델: LightGBM 이진 분류, 대상 기록 b 를 달력 묶음(cal_own_farm cal // 8) 단위 5분할 교차적합.
후보: 같은 온실의 정답 있는 다른 모든 기록. 순위 = 판별기 확률 / 비교 = 손 비용식(CH2에서 EC 항 뺀 것).

[고정 기준 — 실행 전]
 S1 (주) 2차 구간 대상 기록의 top-1 적중률: 학습 판별기 ≥ .60 이고 손 비용식보다 +.15 이상 → '정확도 개선 확인'
 S2 (부) 같은 대상에서 '앞날 23시 EC + 모델 하루 모양'의 2차 일반 날 하루 수준 RMSE가 모델(.146 근방)보다 낮음
    확신 문턱 판(판별기 확률 ≥ .5 일 때만 사용, 아니면 모델) 함께 보고. 문턱 .5 고정, 고르지 않음.
 둘 다 진단 기준이며 제출 채택 판정 아님. 실패도 그대로 기록.
"""
import os, sys, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.optimize import linear_sum_assignment

R = os.path.dirname(os.path.abspath(__file__))
H = os.path.join(env.ROOT, u"집", u"클로드", "research", "local")
print("source sha256", hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16])

X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
for T in (X, Y):
    T["farm"], T["day"], T["hour"] = T.row_id.str[:3], T.row_id.str[4:7].astype(int), T.row_id.str[8:10].astype(int)
X = X[X.farm.isin(["F13", "F47"])].merge(Y[["row_id", "sub_ec"]], on="row_id", how="left")
X = X[X.sub_ec.notna()]  # 정답 있는 기록만 (후보·대상)
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]; IN = ["in_temp", "in_hum", "in_co2"]
ACT = ["act_vent", "act_side", "act_shade", "act_thermal", "act_valve", "act_heating", "act_circfan", "act_co2", "act_fog"]
cal = pd.read_csv(os.path.join(H, "cal_own_farm_v1.csv")).set_index(["farm", "day"]).cal

pairs = []; truth = {}
for f in ("F13", "F47"):
    Z = X[X.farm == f]
    D = np.array(sorted(Z.day.unique())); n = len(D)
    piv = {v: Z.pivot(index="day", columns="hour", values=v).reindex(D).values for v in W + IN + ACT + ["sub_ec"]}
    sc = {v: np.nanstd(np.diff(piv[v], axis=1)) for v in W + IN}
    def J(v):  # 자정 점프 행렬 [a, b]: (B0 − A23) − 추세
        A23, A22, B0, B1 = piv[v][:, 23], piv[v][:, 22], piv[v][:, 0], piv[v][:, 1]
        return (B0[None, :] - A23[:, None]) - .5 * ((A23 - A22)[:, None] + (B1 - B0)[None, :])
    jw = {v: J(v) / sc[v] for v in W}; ji = {v: J(v) / sc[v] for v in IN}
    jec = J("sub_ec")
    datec = sum(wt * jw[v] ** 2 for v, wt in zip(W, (1, 1, .3, 1)))
    inc = sum(ji[v] ** 2 for v in IN) / 4 + sum(np.abs(piv[v][:, 0][None, :] - piv[v][:, 23][:, None]) / 50 for v in ACT[:0])
    act_jump = {v: np.abs(piv[v][:, 0][None, :] - piv[v][:, 23][:, None]) for v in ACT}
    inc = inc + sum(act_jump[v] for v in ("act_heating", "act_thermal", "act_circfan", "act_vent")) / 50 / 4
    hand = datec + inc  # CH2 비용에서 EC 항 뺀 손 비용식
    # 정답 연결 (CH2: EC 항 포함, 1:1)
    M = hand + (jec / .02) ** 2
    M = np.where(np.isfinite(M), M, 1e6); np.fill_diagonal(M, 1e6)
    big = np.full((2 * n, 2 * n), 1e6); big[:n, :n] = M
    big[:n, n:] = np.where(np.eye(n) == 1, 12.0, 1e6); big[n:, :n] = np.where(np.eye(n) == 1, 12.0, 1e6); big[n:, n:] = 0
    r_, c_ = linear_sum_assignment(big)
    for i, j in zip(r_, c_):
        if i < n and j < n and abs(jec[i, j]) < .05:
            truth[(f, D[j])] = D[i]
    # 하루 운영 서명
    dmean = {v: np.nanmean(piv[v], axis=1) for v in ACT}
    nightT = np.nanmean(piv["in_temp"][:, :6], axis=1)
    for j, b in enumerate(D):
        for i, a in enumerate(D):
            if i == j:
                continue
            row = dict(farm=f, a=a, b=b, hand=hand[i, j], datec=datec[i, j], gap=b - a,
                       late_a=a >= 179, late_b=b >= 179, cal_gap=cal.get((f, b), np.nan) - cal.get((f, a), np.nan),
                       ec23=piv["sub_ec"][i, 23], y=int(truth.get((f, b)) == a))
            for v in W:
                row["jw_" + v] = abs(jw[v][i, j])
            for v in IN:
                row["ji_" + v] = abs(ji[v][i, j])
            for v in ACT:
                row["ja_" + v] = act_jump[v][i, j]
                row["sd_" + v] = abs(dmean[v][i] - dmean[v][j])
            row["sd_nightT"] = abs(nightT[i] - nightT[j])
            pairs.append(row)
P = pd.DataFrame(pairs)
P["y"] = [int(truth.get((f, b)) == a) for f, a, b in zip(P.farm, P.a, P.b)]
FEAT = [c for c in P.columns if c.startswith(("jw_", "ji_", "ja_", "sd_"))] + ["datec", "hand", "gap", "late_a", "late_b"]
for c in ("late_a", "late_b"):
    P[c] = P[c].astype(int)
print("pairs %d, 정답 연결 %d (F13 %d, F47 %d), 대상 기록 수 %d" % (
    len(P), P.y.sum(), sum(1 for k in truth if k[0] == "F13"), sum(1 for k in truth if k[0] == "F47"), P.groupby(["farm", "b"]).ngroups))

# ======================================================================================
# LINK2 — 평가와 같은 조건 (v1 의 쌍 특징·정답 연결을 그대로 쓰고, 아래만 새로 고정)
#  DIAG10 각 폴드 k: 폴드 k 기록은 '정답 없음'으로 간주 → 후보 a 에서 제외, 판별기 학습 쌍도
#  대상·후보 모두 폴드 k 밖. 폴드 k 의 2차 정답 기록(전부, 정답 앞날 유무와 무관) 에 대해
#  top-1 후보의 23시 EC 를 하루 수준으로, 모델(dp 시드평균) 시간 모양은 유지:
#    pred_h = dp_h − dp_day + level ;  level = ec23(top-1) if p_top ≥ .5 else dp_day   (문턱 .5 고정)
# [고정 기준] 2차 46일 시간 행 RMSE: 확신판 < 모델, 두 온실 모두 < 모델,
#   온실×5기록 묶음 부트스트랩(2000) P(worse) < .025. 진단 기준(제출 채택 판정 아님).
# 함께 보고: 적중률(정답 앞날이 후보에 남아 있는 대상 기준), 일반/고EC 나눔.
# ======================================================================================
DP = pd.read_csv(os.path.join(H, "ec3_DP1_all.csv"))
DP = DP[DP.validator == "DIAG10"].copy()
DP["dp"] = DP[["dp_7", "dp_101", "dp_2024"]].mean(axis=1)
fold = DP.groupby(["farm", "day"]).validation_fold.first()
P["fa"] = [fold.get((f, a), -1) for f, a in zip(P.farm, P.a)]
P["fb"] = [fold.get((f, b), -1) for f, b in zip(P.farm, P.b)]
out = []
for k in sorted(DP.validation_fold.unique()):
    trm = (P.fa != k) & (P.fb != k)
    if not ((P.fb == k) & (P.b >= 179)).any():
        continue  # 이 폴드엔 2차 대상 없음
    mdl = lgb.LGBMClassifier(n_estimators=400, learning_rate=.03, num_leaves=15, min_child_samples=20,
                             subsample=.8, subsample_freq=1, colsample_bytree=.8, scale_pos_weight=20, verbose=-1, random_state=7)
    mdl.fit(P.loc[trm, FEAT], P.loc[trm, "y"])
    tem = (P.fb == k) & (P.fa != k) & (P.b >= 179)
    T = P[tem].copy(); T["prob"] = mdl.predict_proba(T[FEAT])[:, 1]
    for (f, b), g in T.groupby(["farm", "b"]):
        g = g.sort_values("prob", ascending=False)
        t = truth.get((f, b))
        out.append(dict(farm=f, day=b, fold=k, a=g.a.iloc[0], p_top=g.prob.iloc[0], ec_l=g.ec23.iloc[0],
                        a_h=g.sort_values("hand").a.iloc[0], ec_h=g.sort_values("hand").ec23.iloc[0],
                        t_avail=(t is not None) and (t in set(g.a)), hit=(t is not None) and g.a.iloc[0] == t))
O = pd.DataFrame(out)
print("\n[LINK2] 2차 대상 %d일 (DIAG10 폴드별, 폴드 안 기록은 후보 제외)" % len(O))
print("  정답 앞날이 후보에 남아 있음 %d일 (%.0f%%); 그중 판별기 top-1 적중 %.2f, 손 비용식 %.2f" % (
    O.t_avail.sum(), 100 * O.t_avail.mean(), O[O.t_avail].hit.mean(),
    np.mean([truth.get((f, b)) == ah for f, b, ah in zip(O.farm[O.t_avail], O.day[O.t_avail], O.a_h[O.t_avail])])))
print("  확신(p≥.5) %d일: 적중 %.2f / 확신인데 정답 앞날이 후보에 없는 날 %d" % (
    (O.p_top >= .5).sum(), O[O.p_top >= .5].hit.mean(), ((O.p_top >= .5) & ~O.t_avail).sum()))

D2 = DP[DP.day >= 179].merge(O, on=["farm", "day"])
D2["dp_day"] = D2.groupby(["farm", "day"]).dp.transform("mean")
D2["ydm"] = D2.groupby(["farm", "day"]).sub_ec.transform("mean")
D2["lvl"] = np.where(D2.p_top >= .5, D2.ec_l, D2.dp_day)
D2["gate"] = D2.dp - D2.dp_day + D2.lvl
D2["all_l"] = D2.dp - D2.dp_day + D2.ec_l
D2["hand_l"] = D2.dp - D2.dp_day + D2.ec_h
def rm(x):
    return np.sqrt(np.mean(np.square(x)))
for nm, m in (("2차 전체", D2.day > 0), ("2차 일반", D2.ydm < 1), ("2차 고EC", D2.ydm >= 1), ("F13", D2.farm == "F13"), ("F47", D2.farm == "F47")):
    q = D2[m]
    print("  %-7s 일%2d | 모델 %.4f | 확신판 %.4f (%+.1f%%) | 판별기 무조건 %.4f | 손 비용식 무조건 %.4f" % (
        nm, q.groupby(["farm", "day"]).ngroups, rm(q.dp - q.sub_ec), rm(q.gate - q.sub_ec),
        100 * (rm(q.gate - q.sub_ec) / rm(q.dp - q.sub_ec) - 1), rm(q.all_l - q.sub_ec), rm(q.hand_l - q.sub_ec)))
# 부트스트랩: 온실 × 5기록 묶음
D2["blk"] = D2.farm + "_" + (D2.day // 5).astype(str)
B = D2.groupby("blk").apply(lambda g: pd.Series({"a": ((g.dp - g.sub_ec) ** 2).sum(), "b": ((g.gate - g.sub_ec) ** 2).sum(), "n": len(g)}))
rng = np.random.default_rng(1)
idx = rng.integers(0, len(B), (2000, len(B)))
pw = np.mean(B.b.values[idx].sum(1) > B.a.values[idx].sum(1))
g_ok = rm(D2.gate - D2.sub_ec) < rm(D2.dp - D2.sub_ec)
f_ok = all(rm(D2[D2.farm == f].gate - D2[D2.farm == f].sub_ec) < rm(D2[D2.farm == f].dp - D2[D2.farm == f].sub_ec) for f in ("F13", "F47"))
print("  P(worse) %.4f (묶음 %d) → 고정 기준 %s" % (pw, len(B), "통과" if (g_ok and f_ok and pw < .025) else "불합격"))
O.to_csv(os.path.join(R, "..", "results", "ec_link02_days_v1.csv"), index=False, encoding="utf-8-sig")
