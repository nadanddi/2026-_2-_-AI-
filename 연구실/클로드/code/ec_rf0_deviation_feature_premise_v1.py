# -*- coding: utf-8 -*-
"""RF0 — 편차 지문(같은 날짜 묶음 학습 기록 평균 제거)이 현재 모델의 하루 잔차를 설명하나 (전제 검산, 실행 전 고정) — 2026-10-07 연구실 클로드
근거: FP1(6.425) 편차 지문 = 출처 신호. 선택 방식(FC0~3)은 실패 → 연속 특징으로 쓸 가치가 있는지 먼저 잔차로 확인.
대상: 정답 기록 400일 중 DIAG10 OOF(DP1 파일, R3S+DP1 시드평균 = 현 제출 R3 부분) 하루 잔차 r = y − p.
특징: 편차 지문 17개(구동기 7 하루평균·0비율, 실내 3 하루평균 − 같은 날짜 묶음 '다른 정답 기록' 평균) — 하루 전체(상한).
      비교: 날씨 제거 안 한 원래 지문 17개(기존 지문 특징과 같은 계열).
평가: 날짜 묶음 단위 그룹 5분할 교차적합 릿지(alpha 10, 표준화)로 r 예측 → 잔차 설명 R²(교차적합), 하루 RMSE 개선.
      순열(그룹 단위로 특징 섞기 200회) 기준선. 2차 46일과 전체 400일 따로.
[관문(고정)] 편차 지문의 교차적합 R² 가 순열 95% 분위보다 크고, 원래 지문보다 클 것 → 모델 특징 실험(계획·비평)으로. 아니면 종료.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", u"집", u"클로드", "research"))
import env  # noqa
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
R = os.path.dirname(os.path.abspath(__file__)); H = os.path.join(env.ROOT, u"집", u"클로드", "research", "local")
ACTS = ["act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]; INDOOR = ["in_temp", "in_hum", "in_co2"]
W = ["out_temp", "out_hum", "out_wspd", "out_rad"]
TR = pd.read_csv(os.path.join(env.DATA, "train_X.csv")); TE = pd.read_csv(os.path.join(env.DATA, "test_X.csv")); TR["t"] = 0; TE["t"] = 1
X = pd.concat([TR, TE]); X["farm"], X["day"], X["hour"] = X.row_id.str[:3], X.row_id.str[4:7].astype(int), X.row_id.str[8:10].astype(int)
X = X[X.farm.isin(["F13", "F47"])]
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
DP = pd.read_csv(os.path.join(H, "ec3_DP1_all.csv")); DP = DP[DP.validator == "DIAG10"]
DP["p"] = DP[["dp_7", "dp_101", "dp_2024"]].mean(axis=1)
D = DP.groupby(["farm", "day"]).agg(y=("sub_ec", "mean"), p=("p", "mean")); D["r"] = D.y - D.p
labs = set(D.index)
dev = {}
for k in D.index:
    mem = [m for m in grp.index[grp == grp[k]] if m in labs and m != k]
    if mem:
        dev[k] = Fs.loc[k].values - Fs.loc[mem].values.mean(axis=0)
K = [k for k in D.index if k in dev]
Xd = np.array([dev[k] for k in K]); Xr = Fs.loc[K].values; r = D.loc[K, "r"].values; G = grp.loc[K].values
ug = np.unique(G); rng = np.random.default_rng(0); fold_of = {gg: i % 5 for i, gg in enumerate(rng.permutation(ug))}
fo = np.array([fold_of[gg] for gg in G])
def cvpred(Xm, y):
    out = np.zeros(len(y))
    for k in range(5):
        tr, te = fo != k, fo == k
        m = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(Xm[tr], y[tr]); out[te] = m.predict(Xm[te])
    return out
def r2(pred, y):
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
late = np.array([k[1] >= 179 for k in K])
res = {}
for nm, Xm in (("편차 지문", Xd), ("원래 지문", Xr)):
    pr = cvpred(Xm, r)
    null = []
    for _ in range(200):
        perm = {gg: pg for gg, pg in zip(ug, rng.permutation(ug))}
        idx = np.array([np.where(G == perm[gg])[0][0] if (G == perm[gg]).any() else i for i, gg in enumerate(G)])
        Xp = Xm[rng.permutation(len(Xm))]
        null.append(r2(cvpred(Xp, r), r))
    res[nm] = (r2(pr, r), np.quantile(null, .95))
    rm = lambda e: np.sqrt(np.mean(e ** 2))
    print("%s: 교차적합 R² 전체 %.3f (순열 95%% %.3f) | 하루 RMSE 전체 %.4f → %.4f | 2차 %.4f → %.4f | 2차 일반 %.4f → %.4f" % (
        nm, res[nm][0], res[nm][1], rm(r), rm(r - pr), rm(r[late]), rm((r - pr)[late]),
        rm(r[late & (D.loc[K, 'y'].values < 1)]), rm((r - pr)[late & (D.loc[K, 'y'].values < 1)])))
ok = res["편차 지문"][0] > res["편차 지문"][1] and res["편차 지문"][0] > res["원래 지문"][0]
print("[RF0 관문] %s" % ("통과 → 모델 특징 실험 계획으로" if ok else "종료"))
