# -*- coding: utf-8 -*-
"""DLV1 (2026-10-16 집 클로드; 사용자 '그날 수준이 통째로 어긋나는 상황이 남은 49개 동에도 없었어?').  서술(판정 없음), 실행 전 커밋.  온도.
온실마다 같은 기준 모델(SELF_LR, 날짜 연속 5덩어리 OOF; PCH4 와 같은 틀)의 오차 e 로:
  하루 수준 몫 = sum_d n_d * mean_d(e)^2 / sum e^2,  상위 10% 날 몫,  하루 평균 오차 |m_d| 중앙·90% 분위,
  어제와 이어짐 = 연속한 날(d−1, d) 의 m 상관 (F13·F47 은 기록 이어붙임이라 번호 연속 ≠ 진짜 연속 — 참고만),
  입력으로 설명되나 = m_d 를 그날 입력 요약(실내 평균·최저·최고·밤 평균, 습도·CO2 평균)으로 5덩어리 교차 선형회귀한 R2.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u dlv1_day_level_offsets_all_farms_v1.py
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
from sklearn.linear_model import LinearRegression
def stats(T):
    R = D[D.farm == T].copy(); days = np.array(sorted(R.day.unique()))
    if len(days) < 60: return None
    blk = {d: min(i * 5 // len(days), 4) for i, d in enumerate(days)}; R["fold"] = R.day.map(blk); R["pred"] = np.nan
    for k in range(5):
        tr = R[(R.fold != k) & R.sub_temp.notna()]; te = R[R.fold == k]; _, p = fit_lr(tr, np.ones(len(tr)), te); R.loc[te.index, "pred"] = p
    R = R[R.sub_temp.notna() & R.pred.notna()]; R["e"] = R.pred - R.sub_temp
    g = R.groupby("day"); m = g.e.mean(); n = g.size(); sse = (R.e ** 2).sum(); lvl = float((n * m ** 2).sum() / sse)
    dse = g.e.apply(lambda s: (s ** 2).sum()).sort_values(ascending=False); top = float(dse.head(max(1, len(dse) // 10)).sum() / sse)
    mm = m.sort_index(); cons = mm.index.to_series().diff() == 1; ac = float(np.corrcoef(mm.values[1:][cons.values[1:]], mm.values[:-1][cons.values[1:]])[0, 1]) if cons.sum() > 10 else np.nan
    F = g.agg(t_mean=("in_temp", "mean"), t_min=("in_temp", "min"), t_max=("in_temp", "max"), h_mean=("in_hum", "mean"), c_mean=("in_co2", "mean"))
    F["t_night"] = R[R.hour <= 5].groupby("day").in_temp.mean(); F = F.fillna(F.median()); fo = pd.Series(blk).reindex(F.index).values; pr = np.zeros(len(F))
    for k in range(5):
        tr, te = fo != k, fo == k
        if te.any() and tr.sum() > 10: pr[te] = LinearRegression().fit(F[tr], m[F.index][tr]).predict(F[te])
    r2 = float(1 - ((m[F.index] - pr) ** 2).sum() / ((m[F.index] - m.mean()) ** 2).sum())
    out = dict(farm=T, days=len(days), rmse=float(np.sqrt((R.e ** 2).mean())), level_share=lvl, top10_share=top, absm_med=float(m.abs().median()), absm_q90=float(m.abs().quantile(.9)), lag1_corr=ac, input_R2=r2)
    print(T, {k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()}, flush=True); return out
with ThreadPoolExecutor(4) as ex: S = pd.DataFrame([r for r in ex.map(stats, FARMS) if r is not None])
pd.set_option("display.width", 200); print(S.round(3).to_string(index=False))
O = S[~S.farm.isin(["F13", "F47"])]
print("\nother farms median:", O[["rmse", "level_share", "top10_share", "absm_med", "absm_q90", "lag1_corr", "input_R2"]].median().round(3).to_dict())
print("F13/F47:", S[S.farm.isin(["F13", "F47"])][["farm", "rmse", "level_share", "top10_share", "absm_med", "absm_q90", "lag1_corr", "input_R2"]].round(3).to_dict("records"))
S.to_csv(os.path.join(HERE, "dlv1_v1_table.csv"), index=False)
