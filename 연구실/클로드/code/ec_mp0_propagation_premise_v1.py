# -*- coding: utf-8 -*-
"""MP0 — 다단계 전파 전제 검산: 출처를 '완벽히' 안다고 할 때, 앞날 정답 23시 하나에서 시작해 모델의 하루 안 변화만으로
k 단계 뒤 날의 하루 수준을 얼마나 맞히나 (진단·상한, 출처 식별 오류 없음) — 2026-10-07 연구실 클로드
진짜 사슬: 정답 기록 400 의 CH2 1:1 배정(비용 = (EC 자정 점프/.02)² + 외기 날짜 + 실내/4, 연결 없음 12), EC 점프 < .05 인 연결만.
전파: 출발 노드 s 의 실제 EC23 → 다음 노드 n: EC0_est(n) = 앞 노드 EC23_est, 하루수준_est(n) = EC0_est + (m_mean − m_0)(n),
      EC23_est(n) = EC0_est + (m_23 − m_0)(n). m = 모델 예측(DIAG10 OOF, 계절 R3+DP1 시드 평균; 다른 폴드 학습).
      변형 B: m 대신 '변화 0'(하루 수준 = EC0_est, EC23 = EC0_est).
비교: 같은 노드의 모델 하루 평균 오차. 단계 k = 1..8 별, 그리고 2차 노드만.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
R = os.path.dirname(os.path.abspath(__file__)); H = os.path.join(env.ROOT, u"집", u"클로드", "research", "local")
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]; WT = dict(zip(W, (1, 1, .3, 1))); IN = ["in_temp", "in_hum", "in_co2"]
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
for T in (X, Y):
    T["farm"], T["day"], T["hour"] = T.row_id.str[:3], T.row_id.str[4:7].astype(int), T.row_id.str[8:10].astype(int)
X = X[X.farm.isin(["F13", "F47"])]; Y = Y[Y.farm.isin(["F13", "F47"])]
EC = Y.pivot_table(index=["farm", "day"], columns="hour", values="sub_ec"); labs = list(EC.index)
PV = {v: X.pivot_table(index=["farm", "day"], columns="hour", values=v).reindex(labs) for v in W + IN}
sc = {v: np.nanstd(np.diff(PV[v].values, axis=1)) for v in W + IN}
def J(df):
    M = df.values; A23, A22, B0, B1 = M[:, 23], M[:, 22], M[:, 0], M[:, 1]
    return (B0[None, :] - A23[:, None]) - .5 * ((A23 - A22)[:, None] + (B1 - B0)[None, :])
EJ = J(EC)
C = sum(WT[v] * (J(PV[v]) / sc[v]) ** 2 for v in W) + sum((J(PV[v]) / sc[v]) ** 2 for v in IN) / 4 + (EJ / .02) ** 2
C = np.where(np.isfinite(C), C, 1e6); np.fill_diagonal(C, 1e6); n = len(labs)
big = np.full((2 * n, 2 * n), 1e6); big[:n, :n] = C
big[:n, n:] = np.where(np.eye(n) == 1, 12.0, 1e6); big[n:, :n] = np.where(np.eye(n) == 1, 12.0, 1e6); big[n:, n:] = 0
r_, c_ = linear_sum_assignment(big)
succ = {}
for i, j in zip(r_, c_):
    if i < n and j < n and abs(EJ[i, j]) < .05:
        succ[labs[i]] = labs[j]
pred = {v: k for k, v in succ.items()}
starts = [k for k in labs if k not in pred]
chains = []
for s in starts:
    ch = [s]
    while ch[-1] in succ:
        ch.append(succ[ch[-1]])
    chains.append(ch)
print("사슬 %d, 길이 분포 %s, 다른 ID 연결 %d" % (len(chains), sorted([len(c) for c in chains], reverse=True)[:12], sum(1 for a, b in succ.items() if a[0] != b[0])))
DP = pd.read_csv(os.path.join(H, "ec3_DP1_all.csv")); DP = DP[DP.validator == "DIAG10"]
DP["m"] = DP[["dp_7", "dp_101", "dp_2024"]].mean(axis=1)
M = DP.pivot_table(index=["farm", "day"], columns="hour", values="m")
ym = EC.mean(axis=1)
rows = []
for ch in chains:
    for si in range(len(ch) - 1):
        e23 = EC.loc[ch[si], 23]; e23b = e23
        for k in range(1, 9):
            if si + k >= len(ch):
                break
            nd = ch[si + k]
            if nd not in M.index:
                break
            m = M.loc[nd].values
            lvl = e23 + (np.nanmean(m) - m[0]); nxt = e23 + (m[23] - m[0])
            rows.append(dict(node=nd, k=k, late=nd[1] >= 179, err=lvl - ym[nd], errB=e23b - ym[nd], merr=np.nanmean(m) - ym[nd], y=ym[nd]))
            e23 = nxt
O = pd.DataFrame(rows)
rm = lambda x: np.sqrt(np.mean(np.square(x)))
print("\n단계 k | 쌍 수 | 전파(모델 하루 안 변화) | 전파(변화 0) | 모델 하루 평균   (전체 / 2차 노드)")
for k in range(1, 9):
    q = O[O.k == k]; q2 = q[q.late]
    print("  k=%d | %4d | %.3f | %.3f | %.3f   ||  2차 %3d | %.3f | %.3f | %.3f" % (k, len(q), rm(q.err), rm(q.errB), rm(q.merr), len(q2),
          rm(q2.err) if len(q2) else np.nan, rm(q2.errB) if len(q2) else np.nan, rm(q2.merr) if len(q2) else np.nan))
for nm, m in (("일반 날(y<1)", O.y < 1), ("고EC 날", O.y >= 1)):
    q = O[m]
    print("  %s: k=1~3 전파 %.3f vs 모델 %.3f | k=4~8 전파 %.3f vs 모델 %.3f" % (nm, rm(q[q.k <= 3].err), rm(q[q.k <= 3].merr), rm(q[q.k >= 4].err), rm(q[q.k >= 4].merr)))
O.to_csv(os.path.join(R, "..", "results", "ec_mp0_propagation_v1.csv"), index=False, encoding="utf-8-sig")
