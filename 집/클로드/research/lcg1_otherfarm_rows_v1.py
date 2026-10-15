# -*- coding: utf-8 -*-
"""LCG1-B 자료 (2026-10-16 집 클로드; 사용자 '매개변수도 모델이 데이터를 보고 판단하게' -> 학습형 사슬 게이트, 방법 B 의 학습 자료).  자료 생성(판정 없음), 실행 전 커밋.
PCH4 와 같은 틀(다른 49개 온실, 온실별 SELF_LR 5개 연속 덩어리 OOF, 진짜 전날).  사슬 후보 날마다 시각별 행:
  r = y - 기준 예측,  u = BETA[h] * clip(A - bias_f - x0, +-2) (A = 전날 기준 23시 예측, x0 = 오늘 0시 기준 예측, bias_f = 그 온실 학습 덩어리 23->0시 배지 차),
  특징: h, absu = |A - bias_f - x0|, it0 = 오늘 0시 실내온도.  -> lcg1_otherfarm_rows_v1.csv.gz
Run:  cd 집/클로드/research && PYTHONPATH="" python -u lcg1_otherfarm_rows_v1.py
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
BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy")); BIAS27 = .36; S1, TAU = .9, 8.0
W1 = S1 * np.exp(-np.arange(24) / TAU)
print("BETA:", " ".join(f"{x:.2f}" for x in BETA), flush=True)
BETA = np.load(os.path.join(env.LOCAL, "linker_v7", "ta8_beta.npy"))
def rows_farm(T):
    R = D[D.farm == T].copy(); days = np.array(sorted(R.day.unique()))
    if len(days) < 60 or T in ("F13", "F47"): return None
    blk = {d: min(i * 5 // len(days), 4) for i, d in enumerate(days)}; R["fold"] = R.day.map(blk); R["pred"] = np.nan
    for k in range(5):
        tr = R[(R.fold != k) & R.sub_temp.notna()]; te = R[R.fold == k]; _, p = fit_lr(tr, np.ones(len(tr)), te); R.loc[te.index, "pred"] = p
    R = R[R.sub_temp.notna() & R.pred.notna()]
    P = R.pivot_table(index="day", columns="hour", values="pred"); Yp = R.pivot_table(index="day", columns="hour", values="sub_temp"); It = R.pivot_table(index="day", columns="hour", values="in_temp"); FO = R.groupby("day").fold.first()
    if 0 not in P or 23 not in P: return None
    BF = {}
    for k in range(5):
        dd = [d for d in FO.index[FO != k] if (d - 1) in Yp.index]; BF[k] = float(np.nanmean([Yp.at[d - 1, 23] - Yp.at[d, 0] for d in dd])) if dd else .36
    out = []
    for d in P.index:
        k = FO[d]
        if (d - 1) not in P.index or FO.get(d - 1) != k or pd.isna(P.at[d, 0]) or pd.isna(P.at[d - 1, 23]): continue
        raw = P.at[d - 1, 23] - BF[k] - P.at[d, 0]; it0 = It.at[d, 0] if 0 in It.columns else np.nan
        for h in range(24):
            if h in P.columns and not pd.isna(P.at[d, h]) and not pd.isna(Yp.at[d, h]):
                out.append((T, d, h, Yp.at[d, h] - P.at[d, h], BETA[h] * np.clip(raw, -2, 2), abs(raw), it0))
    print(T, len(out), flush=True); return pd.DataFrame(out, columns=["farm", "day", "h", "r", "u", "absu", "it0"])
with ThreadPoolExecutor(4) as ex: Z = pd.concat([r for r in ex.map(rows_farm, FARMS) if r is not None], ignore_index=True)
Z.to_csv(os.path.join(HERE, "lcg1_otherfarm_rows_v1.csv.gz"), index=False, compression="gzip"); print("rows", len(Z), "farms", Z.farm.nunique())
