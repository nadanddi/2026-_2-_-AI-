# -*- coding: utf-8 -*-
"""LINK1 — '같은 출처 앞날' 찾기를 학습된 연결 판별기로 (진단, 실행 전 기준 고정) — 2026-10-06 연구실 클로드

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

# 교차적합: 대상 b 의 달력 묶음
P["grp"] = [(int(cal.get((f, b), b) // 8) + (0 if f == "F13" else 2)) % 5 for f, b in zip(P.farm, P.b)]
P["prob"] = np.nan
for k in range(5):
    tr, te = P.grp != k, P.grp == k
    mdl = lgb.LGBMClassifier(n_estimators=400, learning_rate=.03, num_leaves=15, min_child_samples=20,
                           subsample=.8, subsample_freq=1, colsample_bytree=.8, scale_pos_weight=20, verbose=-1, random_state=7)
    mdl.fit(P.loc[tr, FEAT], P.loc[tr, "y"])
    P.loc[te, "prob"] = mdl.predict_proba(P.loc[te, FEAT])[:, 1]

# 대상별 순위
res = []
for (f, b), g in P.groupby(["farm", "b"]):
    if (f, b) not in truth:
        continue
    t = truth[(f, b)]
    gl = g.sort_values("prob", ascending=False); gh = g.sort_values("hand")
    res.append(dict(farm=f, b=b, late=b >= 179, rank_l=int(np.where(gl.a.values == t)[0][0]) + 1,
                    rank_h=int(np.where(gh.a.values == t)[0][0]) + 1, p_top=gl.prob.iloc[0], a_l=gl.a.iloc[0], a_h=gh.a.iloc[0],
                    ec_l=gl.ec23.iloc[0], ec_h=gh.ec23.iloc[0], ec_t=g[g.a == t].ec23.iloc[0]))
Rr = pd.DataFrame(res)
print("\n[S1] 정답 연결이 있는 대상의 앞날 적중률 (top-1 / top-3)")
for nm, m in (("전체", Rr.b > 0), ("1차", ~Rr.late), ("2차", Rr.late)):
    s = Rr[m]
    print("  %s n=%3d | 학습 판별기 %.2f / %.2f | 손 비용식 %.2f / %.2f" % (
        nm, len(s), (s.rank_l == 1).mean(), (s.rank_l <= 3).mean(), (s.rank_h == 1).mean(), (s.rank_h <= 3).mean()))
s = Rr[Rr.late]
l1, h1 = (s.rank_l == 1).mean(), (s.rank_h == 1).mean()
print("  S1 판정(2차 top-1 ≥ .60 and 손 비용식 +.15): %s (%.2f vs %.2f)" % ("통과" if (l1 >= .6 and l1 - h1 >= .15) else "불합격", l1, h1))
print("  확신(p_top≥.5) 비율·적중: 2차 %d/%d, 적중 %.2f" % ((s.p_top >= .5).sum(), len(s), (s[s.p_top >= .5].rank_l == 1).mean() if (s.p_top >= .5).any() else np.nan))

# 2차 정답 기록 전체 중 '정답 있는 앞날'이 있는 비율
lab2 = sorted({(f, b) for f, b in zip(P.farm, P.b) if b >= 179})
print("  2차 정답 기록 %d개 중 정답 있는 앞날 연결 %d개 (%.0f%%) — 나머지는 앞날이 정답 없는 기록(평가 등)이거나 없음" % (
    len(lab2), sum(k in truth for k in lab2), 100 * np.mean([k in truth for k in lab2])))

# [S2] 하루 수준: 앞날 23시 EC vs 모델 (DIAG10 dp 시드평균)
day = pd.read_csv(os.path.join(R, "..", "results", "ec_err01_days_DIAG10_v1.csv"))[["farm", "day", "y", "p"]]
Q = s.merge(day, left_on=["farm", "b"], right_on=["farm", "day"])
def rm(x):
    return np.sqrt(np.mean(np.square(x)))
for nm, m in (("2차 일반", Q.y < 1), ("2차 고EC", Q.y >= 1), ("2차 전체", Q.y > -1)):
    q = Q[m]
    gate = np.where(q.p_top >= .5, q.ec_l, q.p)
    print("[S2] %s n=%2d | 모델 %.3f | 판별기 앞날 %.3f | 손 비용식 앞날 %.3f | 확신판 %.3f | 정답 앞날(상한) %.3f" % (
        nm, len(q), rm(q.p - q.y), rm(q.ec_l - q.y), rm(q.ec_h - q.y), rm(gate - q.y), rm(q.ec_t - q.y)))
print("  (하루 수준 RMSE: 앞날 23시 EC를 그날 하루 평균 추정으로 씀. 2차 대상 중 정답 있는 앞날이 있는 날만)")

imp = pd.Series(mdl.feature_importances_, index=FEAT).sort_values(ascending=False)
print("\n판별기 중요도 상위(마지막 분할):", ", ".join("%s %d" % (k, v) for k, v in imp.head(10).items()))
Rr.to_csv(os.path.join(R, "..", "results", "ec_link01_ranks_v1.csv"), index=False, encoding="utf-8-sig")
