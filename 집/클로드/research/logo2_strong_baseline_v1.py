# -*- coding: utf-8 -*-
"""LOGO2 (2026-10-16 집 클로드; LOGO1 6.973 PASS 이나 기준 SELF 가 외삽 못 하는 트리라 약함 -> 사용자 '진행해').  사전 고정(커밋 후 실행).  온도.
LOGO1 과 같은 시험대(대상 온실의 하루 최저 <= 8도 날 가림, 같은 입력·정제·기증 표본·닮음 거리·SIM 가중) 에 '직선 외삽 되는' 강한 기준을 넣음.
  LR 계열 = 선형(Ridge alpha 1, 표준화; 하루 안 지연 결측은 그 행 현재값으로, 나머지 결측은 학습 중앙값) + LightGBM 잔차(LOGO1 과 같은 설정) - 우리 BASE(물리 선형 + LGB 잔차) 꼴.
  풀링 LR 은 선형에 온실 원핫(온실별 절편) 추가, 잔차 LGB 에 온실 범주.
  SELF_LR : 대상 학습 날만.   SIM_LR : 대상 + SIM 가중 기증(LOGO1 과 같은 가중).   (서술) SELF_LIN 선형만, POOL_LR, BLEND = .5 SELF_LR + .5 SIM_LR, LOGO1 의 SIM(LGB).
판정(단일 주 비교 SIM_LR vs SELF_LR, alpha .025): 대상 2/3 이상에서 추운 행(<= 8도) RMSE 개선 & Wilcoxon(한쪽, 로그 비) p < .025 -> PASS.
Run:  cd 집/클로드/research && PYTHONPATH="" python -u logo2_strong_baseline_v1.py
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
def run_target(T):
    R = D[D.farm == T]; cold_days = set(R.day[R.dmin <= 8]); te = R[R.day.isin(cold_days)]; tr = R[~R.day.isin(cold_days)]
    if len(cold_days) < 3 or (te.in_temp <= 8).sum() < 20 or tr.day.nunique() < 30: return None
    pm, pn, pa, ps = profile(tr); donors = [f for f in FARMS if f != T and not ({T, f} == {"F13", "F47"})]; dist = {}
    for f in donors:
        qm, qn, qa, _ = PROF_ALL[f]; ok = (pn >= 30) & (qn >= 30) & np.isfinite(pm) & np.isfinite(qm)
        dg = float(np.sqrt(np.mean((pm[ok] - qm[ok]) ** 2))) if ok.sum() >= 2 else np.nan; da = (pa - qa) * ps if np.isfinite(pa) and np.isfinite(qa) else np.nan
        dist[f] = float(np.sqrt(dg ** 2 + da ** 2)) if np.isfinite(dg) and np.isfinite(da) else np.nan
    dv = pd.Series(dist); dv = dv.fillna(dv.max() * 2 if dv.notna().any() else 1.0); tau = float(dv.median()); ws = np.exp(-(dv / tau) ** 2)
    out = dict(farm=T, cold_rows=int((te.in_temp <= 8).sum()), top3=",".join(ws.sort_values(ascending=False).index[:3]))
    m8 = (te.in_temp <= 8).values; m6 = (te.in_temp <= 6).values; P = {}; y = te.sub_temp.values
    P["SELF_LIN"], P["SELF_LR"] = fit_lr(tr, np.ones(len(tr)), te)
    for nm, wv in (("POOL", pd.Series(1.0, index=dv.index)), ("SIM", ws)):
        parts, wts = [tr], [np.ones(len(tr))]; tot = float(sum(wv[f] for f in donors))
        for f in donors:
            Rd = DON[f]; parts.append(Rd); wts.append(np.full(len(Rd), len(tr) * wv[f] / tot / len(Rd)))
        A = pd.concat(parts); Aw = np.concatenate(wts)
        _, P[nm + "_LR"] = fit_lr(A, Aw, te, [T] + donors)
        if nm == "SIM": P["SIM"] = lgb.LGBMRegressor(**PRM).fit(A[FEAT + ["fcat"]], A.sub_temp, sample_weight=Aw).predict(te[FEAT + ["fcat"]])
    P["BLEND"] = .5 * P["SELF_LR"] + .5 * P["SIM_LR"]
    for nm, p in P.items():
        out[f"{nm}_le8"] = rm((p - y)[m8]); out[f"{nm}_le6"] = rm((p - y)[m6]) if m6.sum() >= 10 else np.nan; out[f"{nm}_b8"] = float((p - y)[m8].mean())
    print(T, "done", {k: round(v, 3) for k, v in out.items() if k.endswith("_le8")}, flush=True); return out
with ThreadPoolExecutor(4) as ex: res = [r for r in ex.map(run_target, FARMS) if r is not None]
S = pd.DataFrame(res); pd.set_option("display.width", 250)
print(S[["farm", "cold_rows", "top3"] + [c for c in S.columns if c.endswith("_le8")] + ["SELF_LR_b8", "SIM_LR_b8"]].round(3).to_string(index=False))
n = len(S); lr_ = np.log(S.SIM_LR_le8 / S.SELF_LR_le8); win = int((lr_ < 0).sum()); p = float(wilcoxon(lr_, alternative="less").pvalue)
for a_, b_ in (("SELF_LR", "SELF_LIN"), ("SIM", "SELF_LR"), ("POOL_LR", "SELF_LR"), ("SIM_LR", "POOL_LR"), ("BLEND", "SELF_LR"), ("BLEND", "SIM_LR")):
    l2 = np.log(S[f"{a_}_le8"] / S[f"{b_}_le8"]); print(f"  [desc] {a_} vs {b_}: better {int((l2 < 0).sum())}/{n}, median ratio {np.exp(l2.median()):.3f}, Wilcoxon p {wilcoxon(l2, alternative='less').pvalue:.4f}")
k6 = S.SELF_LR_le6.notna(); print(f"  [desc] <=6 rows SIM_LR vs SELF_LR: better {int((S.SIM_LR_le6 < S.SELF_LR_le6)[k6].sum())}/{int(k6.sum())}")
print(S[S.farm.isin(["F13", "F47"])].round(3).T.to_string())
print(f"LOGO2 SIM_LR vs SELF_LR (<=8 rows): better {win}/{n} (need >= {int(np.ceil(2 * n / 3))}) | median ratio {np.exp(lr_.median()):.3f} | Wilcoxon p {p:.4f} -> {'PASS' if win >= 2 * n / 3 and p < .025 else 'FAIL'}")
S.to_csv(os.path.join(HERE, "logo2_v1_table.csv"), index=False)
