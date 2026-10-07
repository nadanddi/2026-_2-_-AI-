# -*- coding: utf-8 -*-
"""FP0 — '운영 지문 4군집 = 출처' 인가, 그리고 같은 군집 달력 이웃 정답이 일반 날 하루 수준을 SG2 보다 잘 아는가 (전제 검산, 진단) — 2026-10-07 연구실 클로드
근거: 카탈로그 1.5(일별 구동기 지문 4군집, 같은 군집 연속 자정 EC 점프 .041 vs 바뀐 날 .347), 6.350(정답 사슬 묶음은 출처가 아니라 계절).
지문(하루 전체, 구조 검산용): 구동기 7개 하루 평균·0 비율, 0시 값 7 + 실내 3 (fp_features 와 같은 변수).
 (a) KMeans k=4 (학습 기록만 적합, 표준화, n_init 20, seed 0) → 모든 기록 배정
 (b) 같은 날짜 묶음(외기 24h 완전 쌍둥이) 안 기록들이 서로 다른 군집인 비율 vs 무작위(군집 빈도 유지 순열 200회)
 (c) 정답 사슬(CH2 1:1, EC 점프 < .05) 연결 중 같은 군집 비율 vs 무작위
 (d) 2차 정답 기록 b: SG2 달력(정답 전체, b 와 b±1 기록 제외 참조)에서 |cal − cal_b| ≤ 5 이고 같은 군집인 정답 기록의 하루평균 중앙값 E_fp,
     대조 E_all(군집 무시 같은 창 중앙값). SG2 저장 OOF(DIAG10 시드평균) 하루평균과 하루 RMSE 비교: 전체/예측 일반(<.9) 날.
"""
import os, sys, importlib.util
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research")
sys.path.insert(0, RES)
import env  # noqa
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("sg2", os.path.join(RES, "ec3_SG2_reference_knn_level_v1.py")); sg2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg2)
R = os.path.dirname(os.path.abspath(__file__))
ACTS = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]; INDOOR = ["in_temp", "in_hum", "in_co2"]
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]
TR = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); TE = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); TR["t"] = 0; TE["t"] = 1
X = pd.concat([TR, TE]); X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
X = X[X.farm.isin(["F13", "F47"])]
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y["farm"], Y["day"], Y["hour"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int), Y.row_id.str[8:10].astype(int)
EC = Y[Y.farm.isin(["F13", "F47"])].pivot_table(index=["farm", "day"], columns="hour", values="sub_ec"); ym = EC.mean(axis=1)
g = X.groupby(["farm", "day"])
F = pd.concat([g[ACTS].mean().add_suffix("_m"), g[ACTS].agg(lambda s: (s == 0).mean()).add_suffix("_z"),
               X[X.hour == 0].set_index(["farm", "day"])[ACTS + INDOOR].add_suffix("_h0")], axis=1)
isT = g.t.first()
tr = F[isT == 0]; mu, sd = tr.mean(), tr.std().replace(0, 1)
Zs = ((F - mu) / sd).fillna(0)
km = KMeans(4, n_init=20, random_state=0).fit(Zs[isT == 0].values)
cl = pd.Series(km.predict(Zs.values), index=F.index)
print("(a) 군집 크기(학습/평가):", cl[isT == 0].value_counts().sort_index().to_dict(), cl[isT == 1].value_counts().sort_index().to_dict())
# 날짜 묶음
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
def distinct_rate(c):
    r = [len(set(c[list(m)])) == len(m) for _, m in grp.groupby(grp).groups.items() if len(m) >= 2]
    return np.mean(r), len(r)
obs, ng = distinct_rate(cl)
rng = np.random.default_rng(0)
null = [distinct_rate(pd.Series(rng.permutation(cl.values), index=cl.index))[0] for _ in range(200)]
print("(b) 같은 날짜 묶음(%d개) 모든 기록이 서로 다른 군집: %.2f (무작위 %.2f ± %.2f)" % (ng, obs, np.mean(null), np.std(null)))
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
same = np.mean([cl[a] == cl[b] for a, b in links])
nulls = [np.mean([pa == pb for pa, pb in zip(rng.permutation(cl[[a for a, _ in links]].values), cl[[b for _, b in links]].values)]) for _ in range(200)]
print("(c) 정답 사슬 연결 %d 중 같은 군집: %.2f (무작위 %.2f)" % (len(links), same, np.mean(nulls)))
# (d)
Rr, WVs, hrs, SIG = sg2.prepare_structure()
labset = set(labs)
S = pd.read_csv(os.path.join(RES, "local", "ec3_SG2_all.csv")); S = S[S.validator == "DIAG10"]
S["sg"] = S[["sg_23", "sg_808", "sg_9090"]].mean(axis=1); sgd = S.groupby(["farm", "day"]).sg.mean()
rows = []
for b in sgd.index:
    ref = {x for x in labset if not (x[0] == b[0] and abs(x[1] - b[1]) <= 1)}
    cal = sg2.ref_calendar(Rr, WVs, ref | {b}) if False else sg2.ref_calendar(Rr, WVs, ref)
    E = [e for e in Rr[Rr.farm == b[0]].day if (b[0], e) in ref]
    A1 = WVs.loc[[(b[0], e) for e in E]].values; ok = np.sqrt(np.nanmean((A1 - WVs.loc[b].values) ** 2, axis=1)) <= .05
    if ok.any():
        cb = float(np.mean([cal[(b[0], e)] for e, o in zip(E, ok) if o]))
    else:
        ds = sorted(Rr[Rr.farm == b[0]].day); i = ds.index(b[1]); p = [d for d in ds[:i] if (b[0], d) in cal]
        cb = cal[(b[0], p[-1])] + .1 if p else np.nan
    win = [(b[0], e) for e in E if (b[0], e) in cal and abs(cal[(b[0], e)] - cb) <= 5]
    wf = [k for k in win if cl[k] == cl[b]]
    rows.append(dict(farm=b[0], day=b[1], y=ym[b], sg=sgd[b], efp=np.median([ym[k] for k in wf]) if wf else np.nan,
                     eall=np.median([ym[k] for k in win]) if win else np.nan, nfp=len(wf), nall=len(win)))
D = pd.DataFrame(rows)
rm = lambda x: np.sqrt(np.nanmean(np.square(x)))
for nm, m in (("2차 전체", D.y > -1), ("예측 일반(sg<.9)", D.sg < .9), ("정답 일반(y<1)", D.y < 1)):
    q = D[m & D.efp.notna()]
    print("(d) %-14s n=%2d | SG2 %.3f | 같은 군집 창 중앙값 %.3f | 군집 무시 창 중앙값 %.3f | 반반(SG2·군집) %.3f" % (
        nm, len(q), rm(q.sg - q.y), rm(q.efp - q.y), rm(q.eall - q.y), rm((q.sg + q.efp) / 2 - q.y)))
D.to_csv(os.path.join(R, "..", "results", "ec_fp0_days_v1.csv"), index=False, encoding="utf-8-sig")
