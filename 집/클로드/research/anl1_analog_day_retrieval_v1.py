# -*- coding: utf-8 -*-
"""ANL1 (2026-10-16 집 클로드; 사용자 '닮은 온실은 우리가 정하지 말고 모델이 자동으로 빌려와야 5점').  사전 고정(커밋 후 실행).  온도.
유사 날 검색형 보조 예측(자동, 사람 선택 없음) — 51개 온실 시험대(대상마다 날짜 연속 5덩어리 OOF, 기준 = 그 온실 SELF_LR):
  시각 h 에 대상 날의 0..h 시 공통 입력(실내 온도·습도·CO2, 전체 표준화) 궤적과 가장 가까운 다른 온실 날 30개(F13·F47 대상이면 서로는 기증에서 뺌)를 찾고,
  그 날들의 h 시 (배지 − 실내) 에서 '그 기증 온실의 시각별 중앙 차'를 뺀 값(모양만)의 평균 + 대상 온실 학습 덩어리의 시각별 중앙 차(수준) -> 유사 날 예측 = 실내_h + 그 값.
  섞기 beta: 대상의 학습 덩어리 행에서 (y − 기준) ~ beta (유사 − 기준) 최소제곱, [0, 1] 자르기 — 자동.  인과: 0..h 입력만.
판정(사전 고정, alpha .025): 섞은 예측 vs 기준 — 온실 2/3 이상 개선 & Wilcoxon(한쪽, 로그 비) p < .025.  서술: 유사 단독, beta, 낮(8~20시)/밤 비, F13·F47 줄.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u anl1_analog_day_retrieval_v1.py
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
def run_farm(T):
    R = D[D.farm == T].copy(); days = np.array(sorted(R.day.unique()))
    if len(days) < 60: return None
    blk = {d: min(i * 5 // len(days), 4) for i, d in enumerate(days)}; R["fold"] = R.day.map(blk); R["pred"] = np.nan
    for k in range(5):
        tr = R[(R.fold != k) & R.sub_temp.notna()]; te = R[R.fold == k]; _, p = fit_lr(tr, np.ones(len(tr)), te); R.loc[te.index, "pred"] = p
    R = R[R.sub_temp.notna() & R.pred.notna() & R.in_temp.notna()]
    tmask = FARMA == T; donor = ~tmask & ~(np.isin(FARMA, ["F13", "F47"]) & np.isin(T, ["F13", "F47"]))
    if tmask.sum() < 30: return None
    Zt, Zd, Gd = ZA[tmask], ZA[donor], GADJ[donor]; tdays = DAYA[tmask]
    cum = np.zeros((len(Zt), len(Zd)), np.float32); ANA = np.full((len(Zt), 24), np.nan, np.float32)
    for h in range(24):
        diff = Zt[:, None, h, :] - Zd[None, :, h, :]; cum += (diff ** 2).sum(-1)
        nn = np.argpartition(cum, KNN, axis=1)[:, :KNN]; ANA[:, h] = Gd[nn, h].mean(1)
    A = pd.DataFrame(ANA, index=tdays).stack().rename("gadj").reset_index(); A.columns = ["day", "hour", "gadj"]
    R = R.merge(A, on=["day", "hour"], how="inner"); out = dict(farm=T)
    for k in range(5):
        tr = R.fold != k; te = R.fold == k
        lvl = (R[tr].sub_temp - R[tr].in_temp).groupby(R[tr].hour).median()                     # target's own level from training folds only
        R.loc[te, "ana"] = R[te].in_temp + R[te].gadj + R[te].hour.map(lvl).fillna(0)
        R.loc[tr, "ana_tr"] = R[tr].in_temp + R[tr].gadj + R[tr].hour.map(lvl).fillna(0)
        x = (R[tr].ana_tr - R[tr].pred).values; yv = (R[tr].sub_temp - R[tr].pred).values
        beta = float(np.clip((x * yv).sum() / max((x * x).sum(), 1e-9), 0, 1)); R.loc[te, "beta"] = beta
    R["blend"] = R.pred + R.beta * (R.ana - R.pred)
    e0, e1 = R.pred - R.sub_temp, R.blend - R.sub_temp
    out.update(base=float(np.sqrt((e0 ** 2).mean())), blend=float(np.sqrt((e1 ** 2).mean())), ana_alone=float(np.sqrt(((R.ana - R.sub_temp) ** 2).mean())), beta=float(R.beta.mean()))
    for nm, m in (("day", R.hour.between(8, 20)), ("night", ~R.hour.between(8, 20))):
        out[nm + "_ratio"] = float(np.sqrt((e1[m] ** 2).mean()) / np.sqrt((e0[m] ** 2).mean()))
    print(T, {k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()}, flush=True); return out
with ThreadPoolExecutor(4) as ex: S = pd.DataFrame([r for r in ex.map(run_farm, FARMS) if r is not None])
pd.set_option("display.width", 200); print(S.round(3).to_string(index=False))
lr_ = np.log(S.blend / S.base); win = int((lr_ < 0).sum()); n = len(S); p = float(wilcoxon(lr_, alternative="less").pvalue)
print(f"  [desc] analog alone vs base better {int((S.ana_alone < S.base).sum())}/{n} | median beta {S.beta.median():.2f} | day ratio median {S.day_ratio.median():.4f} | night ratio median {S.night_ratio.median():.4f}")
print(S[S.farm.isin(["F13", "F47"])].round(4).to_string(index=False))
print(f"ANL1 blend vs base: farms better {win}/{n} (need >= {int(np.ceil(2 * n / 3))}) | median ratio {np.exp(lr_.median()):.4f} | Wilcoxon p {p:.4g} -> {'PASS' if win >= 2 * n / 3 and p < .025 else 'FAIL'}")
S.to_csv(os.path.join(HERE, "anl1_v1_table.csv"), index=False)
