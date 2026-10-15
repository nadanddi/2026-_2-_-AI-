# -*- coding: utf-8 -*-
"""ANL2-자료 (2026-10-16 집 클로드; ANL1 PASS(6.995) -> 19회차 구성에 붙이기 위한 F13·F47 유사 날 '모양' 값 생성).  자료 생성(판정 없음), 실행 전 커밋.
ANL1 과 같은 검색(0..h 공통 입력 궤적, 다른 온실 날 30개, F13·F47 은 서로 기증에서 뺌 = 다른 49곳만), 기증 온실 수준 제거한 (배지−실내) 모양 평균 gadj.
F13·F47 학습 기록 전부(24시간 입력 있는 날) -> anl2_gadj_v1.csv (rec, hour, gadj).  수준(대상 학습 폴드 시각별 중앙 차)은 판정 스크립트에서 폴드마다 더함.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u anl2_gadj_v1.py
"""
import os, sys
import env  # noqa: F401
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import wilcoxon
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"), usecols=["row_id", "in_temp", "in_hum", "in_co2"]); Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"), usecols=["row_id", "sub_temp"])
D = X.merge(Y, on="row_id"); D["farm"] = D.row_id.str[:3]; D["day"] = D.row_id.str[4:7].astype(int); D["hour"] = D.row_id.str[8:10].astype(int)
D.loc[(D.in_temp < 0) | (D.in_temp > 45), "in_temp"] = np.nan; D.loc[D.in_hum <= 0, "in_hum"] = np.nan; D.loc[D.in_co2 <= 0, "in_co2"] = np.nan
D = D[~(D.sub_temp >= 40) & D.sub_temp.notna()].sort_values(["farm", "day", "hour"]).reset_index(drop=True)
g = D.groupby(["farm", "day"])
for c in ("in_temp", "in_hum", "in_co2"):
    for L in (1, 2, 3, 6): D[f"{c}_l{L}"] = g[c].shift(L).where(D.hour >= L)   # within-day only (rows are contiguous hours)
D["cm_t"] = g.in_temp.transform(lambda s: s.expanding().mean()); D["cmin_t"] = g.in_temp.cummin(); D["cmax_t"] = g.in_temp.cummax(); D["d1_t"] = D.in_temp - D.in_temp_l1
D["dmin"] = g.in_temp.transform("min"); D["fcat"] = D.farm.astype("category")
FEAT = ["hour", "in_temp", "in_hum", "in_co2"] + [f"{c}_l{L}" for c in ("in_temp", "in_hum", "in_co2") for L in (1, 2, 3, 6)] + ["cm_t", "cmin_t", "cmax_t", "d1_t"]
rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
BANDS = [8, 10, 12, 15, 20, 25, 99]
def profile(R):
    gap = R.sub_temp - R.in_temp; b = pd.cut(R.in_temp, BANDS, right=False); pr = gap.groupby(b, observed=False).agg(["mean", "size"])
    ds = R.groupby("day").agg(ss=("sub_temp", "std"), si=("in_temp", "std")); amp = float((ds.ss / ds.si).replace([np.inf, -np.inf], np.nan).median()); sig = float(ds.si.median())
    return pr["mean"].values, pr["size"].values, amp, sig
FARMS = sorted(D.farm.unique()); rng = np.random.default_rng(7)
DON = {}
for f in FARMS:
    R = D[D.farm == f]; keep = (R.dmin <= 12).values | (rng.random(len(R)) < .25); DON[f] = R[keep]
PROF_ALL = {f: profile(D[D.farm == f]) for f in FARMS}
PRM = dict(n_estimators=400, learning_rate=.05, num_leaves=31, min_child_samples=50, subsample=.8, subsample_freq=1, colsample_bytree=.8, random_state=7, verbose=-1, n_jobs=4)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
LAGC = {f"{c}_l{L}": c for c in ("in_temp", "in_hum", "in_co2") for L in (1, 2, 3, 6)}
def linX(A, med, farms=None):
    Z = A[FEAT].copy()
    for k, c in LAGC.items(): Z[k] = Z[k].fillna(A[c])
    Z["d1_t"] = Z["d1_t"].fillna(0.0); Z = Z.fillna(med)
    if farms is not None:
        for f in farms: Z["oh_" + f] = (A.farm.values == f).astype(float)
    return Z.to_numpy(float)
def fit_lr(A, w, te, farms=None):
    med = A[FEAT].median(); Xa = linX(A, med, farms); sc = StandardScaler().fit(Xa, sample_weight=w); lin = Ridge(alpha=1.0).fit(sc.transform(Xa), A.sub_temp.values, sample_weight=w)
    feats = FEAT + (["fcat"] if farms is not None else []); res = A.sub_temp.values - lin.predict(sc.transform(Xa))
    g_ = lgb.LGBMRegressor(**PRM).fit(A[feats], res, sample_weight=w); Xt = linX(te, med, farms); pl = lin.predict(sc.transform(Xt))
    return pl, pl + g_.predict(te[feats])
KNN = 30
V = ["in_temp", "in_hum", "in_co2"]
DD = D.copy(); mu = DD[V].mean(); sd = DD[V].std()
for c in V: DD["z_" + c] = ((DD[c] - mu[c]) / sd[c])
DD["z_in_temp"] = DD.z_in_temp.fillna(0); DD["z_in_hum"] = DD.z_in_hum.fillna(0); DD["z_in_co2"] = DD.z_in_co2.fillna(0)
PZ = DD.pivot_table(index=["farm", "day"], columns="hour", values=["z_in_temp", "z_in_hum", "z_in_co2", "in_temp", "sub_temp"])
okd = PZ["in_temp"].notna().sum(1).eq(24) & PZ["sub_temp"].notna().sum(1).ge(20); PZ = PZ[okd]
ZA = np.stack([PZ["z_in_temp"].values, PZ["z_in_hum"].values, PZ["z_in_co2"].values], 2).astype(np.float32)   # days x 24 x 3
GAPA = (PZ["sub_temp"] - PZ["in_temp"]).values.astype(np.float32); FARMA = PZ.index.get_level_values(0).values; DAYA = PZ.index.get_level_values(1).values
FMG = pd.DataFrame(GAPA).groupby(FARMA).transform(lambda s: s.fillna(s.median()).median() if s.notna().any() else 0).values   # farm-hour median gap (level of each donor farm)
GADJ = np.nan_to_num(GAPA - FMG, nan=0.0)                                                                                    # donor gap with its own farm level removed
rows = []
for T in ("F13", "F47"):
    tmask = FARMA == T; donor = ~np.isin(FARMA, ["F13", "F47"]); Zt, Zd, Gd = ZA[tmask], ZA[donor], GADJ[donor]; tdays = DAYA[tmask]
    cum = np.zeros((len(Zt), len(Zd)), np.float32)
    for h in range(24):
        diff = Zt[:, None, h, :] - Zd[None, :, h, :]; cum += (diff ** 2).sum(-1); nn = np.argpartition(cum, KNN, axis=1)[:, :KNN]
        for d, g in zip(tdays, Gd[nn, h].mean(1)): rows.append(("%s_%03d" % (T, d), h, float(g)))
pd.DataFrame(rows, columns=["rec", "hour", "gadj"]).to_csv(os.path.join(HERE, "anl2_gadj_v1.csv"), index=False); print("rows", len(rows))
