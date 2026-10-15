# -*- coding: utf-8 -*-
"""PCH4 (2026-10-16 집 클로드; 사용자 '검증 보완' 2번 — 예측 사슬 규칙의 외부·독립 검증).  사전 고정(커밋 후 실행).  온도.
F13·F47 에서 정한 규칙을 그대로 고정해 다른 온실(날짜가 원래 이어진 학습 정답 온실)에 적용: 기록·온실이 완전히 다른 독립 표본, 현장 일반성 근거.
온실마다 모델 = LOGO2 의 SELF_LR(선형 + LGB 잔차, 하루 안 입력만), 날짜 순 5개 연속 덩어리 교차적합(OOF) 예측.
사슬: 같은 덩어리 안에서 전날 d-1 이 있으면 오늘 h시 += w(h) * BETA[h] * clip(A - bias - x0, ±2),  A = 전날 사슬 반영 최종 23시 예측, x0 = 오늘 0시 예측,
  w(h) = .9 * exp(-h/8) (PCH3 M1 그대로), BETA = ta8 (그대로).  bias: chainF = 그 온실 학습 덩어리의 (전날 23시 배지 - 오늘 0시 배지) 평균(현장에서 자기 이력으로 잴 수 있는 값),
  chain27 = F13·F47 값 .36 고정(서술).  링커 없음(진짜 전날) — 현장 상황.
판정(사전 고정, chainF vs base, alpha .025): 온실 2/3 이상에서 전체 RMSE 개선 & Wilcoxon(한쪽, 로그 비) p < .025.  서술: 0~5시 변화, chain27.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u pch4_chain_rule_other_farms_v1.py
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
def run_farm(T):
    R = D[D.farm == T].copy(); days = np.array(sorted(R.day.unique()))
    if len(days) < 60 or T in ("F13", "F47"): return None
    blk = {d: min(i * 5 // len(days), 4) for i, d in enumerate(days)}; R["fold"] = R.day.map(blk)
    R["pred"] = np.nan
    for k in range(5):
        tr = R[(R.fold != k) & R.sub_temp.notna()]; te = R[R.fold == k]
        _, p = fit_lr(tr, np.ones(len(tr)), te); R.loc[te.index, "pred"] = p
    R = R[R.sub_temp.notna() & R.pred.notna()]
    P = R.pivot_table(index="day", columns="hour", values="pred"); Yp = R.pivot_table(index="day", columns="hour", values="sub_temp"); FO = R.groupby("day").fold.first()
    if 0 not in P or 23 not in P: return None
    rows = []
    for k in range(5):
        tr_days = FO.index[FO != k]; dd = [d for d in tr_days if (d - 1) in Yp.index]
        bias_f = float(np.nanmean([Yp.at[d - 1, 23] - Yp.at[d, 0] for d in dd])) if dd else BIAS27   # farm's own 23h -> 0h substrate drop, training folds only
        rows.append((k, bias_f))
    BF = dict(rows); out = {"base": P.copy(), "chainF": P.copy(), "chain27": P.copy()}; nchain = 0
    for d in sorted(P.index):
        k = FO[d]
        if (d - 1) not in P.index or FO.get(d - 1) != k or pd.isna(P.at[d, 0]): continue
        nchain += 1
        for nm, bias in (("chainF", BF[k]), ("chain27", BIAS27)):
            A = out[nm].at[d - 1, 23]
            if pd.isna(A): continue
            u = BETA * np.clip(A - bias - P.at[d, 0], -2, 2); hs = [h for h in range(24) if h in P.columns]
            out[nm].loc[d, hs] = P.loc[d, hs].values + W1[hs] * u[hs]
    res = dict(farm=T, days=len(days), chained_days=nchain)
    for nm, Q in out.items():
        e = (Q - Yp).values.ravel(); e = e[np.isfinite(e)]; res[nm] = float(np.sqrt(np.mean(e ** 2)))
        if nm != "base":
            d0 = (Q - Yp).loc[:, [h for h in range(6) if h in Q.columns]].values.ravel(); b0 = (P - Yp).loc[:, [h for h in range(6) if h in P.columns]].values.ravel()
            m = np.isfinite(d0) & np.isfinite(b0); res[nm + "_0to5"] = float(np.sqrt(np.mean(d0[m] ** 2)) / np.sqrt(np.mean(b0[m] ** 2)) - 1)
    res["biasF_mean"] = float(np.mean(list(BF.values()))); print(T, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, flush=True); return res
with ThreadPoolExecutor(4) as ex: S = pd.DataFrame([r for r in ex.map(run_farm, FARMS) if r is not None])
pd.set_option("display.width", 220); print(S.round(3).to_string(index=False))
n = len(S)
for nm in ("chainF", "chain27"):
    lr_ = np.log(S[nm] / S.base); p = float(wilcoxon(lr_, alternative="less").pvalue); win = int((lr_ < 0).sum())
    tag = (" -> " + ("PASS" if win >= 2 * n / 3 and p < .025 else "FAIL")) if nm == "chainF" else " (desc)"
    print(f"PCH4 {nm}: farms better {win}/{n} (need >= {int(np.ceil(2 * n / 3))}) | median ratio {np.exp(lr_.median()):.4f} | Wilcoxon p {p:.4g} | 0-5h median change {100 * S[nm + '_0to5'].median():+.2f}%{tag}")
S.to_csv(os.path.join(HERE, "pch4_v1_table.csv"), index=False)
