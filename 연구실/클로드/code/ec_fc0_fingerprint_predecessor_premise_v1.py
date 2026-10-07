# -*- coding: utf-8 -*-
"""FC0 — FP1 편차 지문으로 '전날 날짜 묶음 안 같은 출처 기록'을 골라 그 23시 EC 를 하루 수준으로 (전제 검산) — 2026-10-07 연구실 클로드
대상: 2차 정답 기록 b 중 날짜 확정(C1·C2: 1차 정답 기록과 외기 쌍둥이). b 와 실제 평가 60 은 정답 없음.
전날 묶음 P: b 의 날짜 묶음(1차 기록 순서상)의 바로 앞 날짜 묶음(두 ID 포함, 외기 쌍둥이 성분). (PD1 과 같은 구성)
선택: P 의 기록(정답·평가 모두) 중 b 의 편차 지문과 가장 가까운 것. 편차 = 지문 − 그 날짜 묶음 '정답 기록(자기 제외)' 평균.
      고른 기록에 정답이 있으면 v = EC23, 없으면(평가 기록) 결정 안 함.
비교: SG2 저장 OOF(DIAG10 시드평균) 하루평균, 같은 날만. 대조: P 정답 기록 중 무작위, 오라클(P 정답 기록 중 b 의 EC0 에 가장 가까운 EC23 — 상한).
비순환 적중: |EC0(b) − v| < .05.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
R = os.path.dirname(os.path.abspath(__file__)); H = os.path.join(env.ROOT, u"집", u"클로드", "research", "local")
ACTS = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]; INDOOR = ["in_temp", "in_hum", "in_co2"]
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]
TR = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); TE = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); TR["t"] = 0; TE["t"] = 1
X = pd.concat([TR, TE]); X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
X = X[X.farm.isin(["F13", "F47"])]
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv")); Y["farm"], Y["day"], Y["hour"] = Y.row_id.str[:3], Y.row_id.str[4:7].astype(int), Y.row_id.str[8:10].astype(int)
EC = Y[Y.farm.isin(["F13", "F47"])].pivot_table(index=["farm", "day"], columns="hour", values="sub_ec"); ym = EC.mean(axis=1); labset = set(EC.index)
g = X.groupby(["farm", "day"]); isT = g.t.first()
F = pd.concat([g[ACTS].mean().add_suffix("_m"), g[ACTS].agg(lambda s: (s == 0).mean()).add_suffix("_z"), g[INDOOR].mean().add_suffix("_d")], axis=1)
Fs = ((F - F[isT == 0].mean()) / F[isT == 0].std().replace(0, 1)).fillna(0)
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
cls = pd.read_csv(os.path.join(R, "..", "results", "ec_dt0_date_class_v1.csv")).set_index(["farm", "day"]).cls
def dev(k, hidden):
    mem = [m for m in grp.index[grp == grp[k]] if m in labset and m != k and m not in hidden]
    return (Fs.loc[k].values - Fs.loc[mem].values.mean(axis=0)) if mem else None
S = pd.read_csv(os.path.join(H, "ec3_SG2_all.csv")); S = S[S.validator == "DIAG10"]
S["sg"] = S[["sg_23", "sg_808", "sg_9090"]].mean(axis=1); sgd = S.groupby(["farm", "day"]).sg.mean()
rng = np.random.default_rng(0); rows = []
for b in sgd.index:
    if cls.get(b) not in ("C1", "C2"):
        continue
    hidden = {b} | {k for k in keys if isT[k] == 1}
    # b 의 날짜 묶음과 1차 순서상 앞 날짜 묶음 (b 의 묶음에 속한 1차 기록 기준)
    p1 = sorted([k for k in grp.index[grp == grp[b]] if k[1] < 179 and k in labset], key=lambda k: k[1])
    if not p1:
        continue
    f0, d0 = p1[0]
    ds = sorted(d for f, d in keys if f == f0 and d < 179 and d < d0)
    prevg = None
    for d in reversed(ds):
        if grp[(f0, d)] != grp[b]:
            prevg = grp[(f0, d)]; break
    if prevg is None:
        continue
    P = list(grp.index[grp == prevg])
    db = dev(b, hidden)
    if db is None:
        continue
    cand = [(k, dev(k, hidden)) for k in P]
    cand = [(k, v) for k, v in cand if v is not None]
    if not cand:
        continue
    best = min(cand, key=lambda kv: np.sqrt(np.mean((kv[1] - db) ** 2)))[0]
    labP = [k for k in P if k in labset and k not in hidden]
    v = EC.loc[best, 23] if (best in labset and best not in hidden) else np.nan
    rnd = EC.loc[labP[int(rng.integers(len(labP)))], 23] if labP else np.nan
    orc = min((EC.loc[k, 23] for k in labP), key=lambda e: abs(e - EC.loc[b, 0])) if labP else np.nan
    rows.append(dict(farm=b[0], day=b[1], y=ym[b], sg=sgd[b], v=v, rnd=rnd, orc=orc, best_test=best not in labset,
                     hit=abs(EC.loc[b, 0] - v) < .05 if np.isfinite(v) else np.nan, nP=len(P), nPlab=len(labP)))
D = pd.DataFrame(rows)
rm = lambda x: np.sqrt(np.nanmean(np.square(x)))
print("대상(C1·C2, SG2 OOF 있음) %d | 고른 기록이 평가 기록(결정 안 함) %d | 결정 %d, 비순환 적중 %.2f" % (len(D), D.best_test.sum(), D.v.notna().sum(), D.hit.mean()))
for nm, m in (("결정된 날 전체", D.v.notna()), ("결정·정답 일반", D.v.notna() & (D.y < 1)), ("결정·예측 일반", D.v.notna() & (D.sg < .9))):
    q = D[m]
    print("  %-12s n=%2d | SG2 %.3f | 지문 선택 %.3f | 무작위 %.3f | 오라클(상한) %.3f | 반반(SG2·지문) %.3f" % (
        nm, len(q), rm(q.sg - q.y), rm(q.v - q.y), rm(q.rnd - q.y), rm(q.orc - q.y), rm((q.sg + q.v) / 2 - q.y)))
D.to_csv(os.path.join(R, "..", "results", "ec_fc0_days_v1.csv"), index=False, encoding="utf-8-sig")
