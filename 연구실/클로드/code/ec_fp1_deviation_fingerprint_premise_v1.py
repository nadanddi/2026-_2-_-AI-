# -*- coding: utf-8 -*-
"""FP1 — 날씨·계절을 지운 '같은 날짜 묶음 안 편차' 운영 지문이 출처를 가르나 (전제 검산, 진단; FP0 비평 제안) — 2026-10-07 연구실 클로드
편차 지문 = (기록의 하루 구동기 평균·0 비율·실내 하루 평균) − (같은 날짜 묶음 평균). 묶음 크기 ≥ 3 인 날짜만.
 Q1 같은 날짜 묶음 안 쌍별 '다른 군집' 비율 vs 무작위(묶음 안 배정 순열) — k = 2, 3, 4 (학습 기록만 적합)
 Q2 지속성: 정답 사슬 연결(같은 출처 연속일) 편차 지문 거리 vs 기준선(같은 ID, 기록 번호 차 ≤ 2, 사슬 연결 아님, 같은 날짜 아님)
 Q3 편차 지문으로 같은 날짜 묶음 안 '어느 기록이 사슬 앞날의 다음인가' 맞히기: 앞날 기록 a 의 편차 지문과 가장 가까운 기록 = 예측 b, 적중률 vs 우연(1/묶음 크기)
     (a 의 편차는 a 의 날짜 묶음 기준, 후보의 편차는 그 날짜 묶음 기준 — 입력만 사용)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
R = os.path.dirname(os.path.abspath(__file__))
ACTS = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]; INDOOR = ["in_temp", "in_hum", "in_co2"]
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]
TR = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); TE = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); TR["t"] = 0; TE["t"] = 1
X = pd.concat([TR, TE]); X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
X = X[X.farm.isin(["F13", "F47"])]
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y["farm"], Y["day"], Y["hour"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int), Y.row_id.str[8:10].astype(int)
EC = Y[Y.farm.isin(["F13", "F47"])].pivot_table(index=["farm", "day"], columns="hour", values="sub_ec")
g = X.groupby(["farm", "day"]); isT = g.t.first()
F = pd.concat([g[ACTS].mean().add_suffix("_m"), g[ACTS].agg(lambda s: (s == 0).mean()).add_suffix("_z"), g[INDOOR].mean().add_suffix("_d")], axis=1)
WV = X.pivot_table(index=["farm", "day"], columns="hour", values=W); WZ = (WV - WV[isT == 0].mean()) / WV[isT == 0].std()
keys = list(WZ.index); A = WZ.values; Dm = np.sqrt(np.nanmean((A[:, None, :] - A[None, :, :]) ** 2, axis=2))
par = list(range(len(keys)))
def fd(x):
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x
for i in range(len(keys)):
    for j in np.where(Dm[i] <= .05)[0]:
        par[fd(i)] = fd(j)
grp = pd.Series([fd(i) for i in range(len(keys))], index=pd.MultiIndex.from_tuples(keys))
size = grp.map(grp.value_counts())
Fs = (F - F[isT == 0].mean()) / F[isT == 0].std().replace(0, 1)
Dv = (Fs - Fs.groupby(grp.reindex(Fs.index).values).transform("mean")).fillna(0)
ok = size.reindex(Dv.index) >= 3
rng = np.random.default_rng(0)
print("Q1 편차 지문 군집이 같은 날짜 묶음 안에서 갈리는가 (쌍별 '다른 군집' 비율, 묶음 크기 ≥ 3)")
for k in (2, 3, 4):
    km = KMeans(k, n_init=20, random_state=0).fit(Dv[ok & (isT.reindex(Dv.index) == 0)].values)
    cl = pd.Series(km.predict(Dv.values), index=Dv.index)
    def pdiff(c):
        r = []
        for _, m in grp[ok.values].groupby(grp[ok.values]).groups.items():
            v = c[list(m)].values; r += [v[i] != v[j] for i in range(len(v)) for j in range(i + 1, len(v))]
        return np.mean(r)
    obs = pdiff(cl)
    nul = []
    for _ in range(200):
        c2 = cl.copy()
        for _, m in grp[ok.values].groupby(grp[ok.values]).groups.items():
            m = list(m)
        c2[:] = rng.permutation(cl.values)
        nul.append(pdiff(c2))
    print("  k=%d: %.3f (전체 무작위 %.3f ± %.3f)" % (k, obs, np.mean(nul), np.std(nul)))
# 정답 사슬
labs = list(EC.index)
PV = {v: X.pivot_table(index=["farm", "day"], columns="hour", values=v).reindex(labs) for v in W + INDOOR}
sc = {v: np.nanstd(np.diff(PV[v].values, axis=1)) for v in W + INDOOR}
def J(df):
    M = df.values; A23, A22, B0, B1 = M[:, 23], M[:, 22], M[:, 0], M[:, 1]
    return (B0[None, :] - A23[:, None]) - .5 * ((A23 - A22)[:, None] + (B1 - B0)[None, :])
EJ = J(EC.reindex(labs)); WT = dict(zip(W, (1, 1, .3, 1)))
C = sum(WT[v] * (J(PV[v]) / sc[v]) ** 2 for v in W) + sum((J(PV[v]) / sc[v]) ** 2 for v in INDOOR) / 4 + (EJ / .02) ** 2
C = np.where(np.isfinite(C), C, 1e6); np.fill_diagonal(C, 1e6); n = len(labs)
big = np.full((2 * n, 2 * n), 1e6); big[:n, :n] = C; big[:n, n:] = np.where(np.eye(n) == 1, 12., 1e6); big[n:, :n] = np.where(np.eye(n) == 1, 12., 1e6); big[n:, n:] = 0
r_, c_ = linear_sum_assignment(big)
links = [(labs[i], labs[j]) for i, j in zip(r_, c_) if i < n and j < n and abs(EJ[i, j]) < .05]
lset = set(links) | {(b, a) for a, b in links}
d = lambda a, b: float(np.sqrt(np.mean((Dv.loc[a].values - Dv.loc[b].values) ** 2)))
dl = [d(a, b) for a, b in links if ok[a] and ok[b]]
base = [d((f, x), (f, y)) for f in ("F13", "F47") for x in range(1, 260) for y in (x + 1, x + 2)
        if (f, x) in Dv.index and (f, y) in Dv.index and ((f, x), (f, y)) not in lset and grp[(f, x)] != grp[(f, y)] and ok[(f, x)] and ok[(f, y)] and (f, x) in EC.index and (f, y) in EC.index]
print("Q2 편차 지문 거리: 같은 출처 연속일 중앙 %.3f (n=%d) vs 같은 ID 번호 차 ≤2 비사슬 쌍 중앙 %.3f (n=%d)" % (np.median(dl), len(dl), np.median(base), len(base)))
hits, ch = [], []
for a, b in links:
    if not (ok[a] and ok[b]):
        continue
    cand = [k for k in grp.index[grp == grp[b]] if k != a]
    best = min(cand, key=lambda k: d(a, k))
    hits.append(best == b); ch.append(1 / len(cand))
print("Q3 앞날 편차 지문으로 같은 날짜 묶음에서 다음날 기록 고르기: 적중 %.2f (n=%d) vs 우연 %.2f" % (np.mean(hits), len(hits), np.mean(ch)))
