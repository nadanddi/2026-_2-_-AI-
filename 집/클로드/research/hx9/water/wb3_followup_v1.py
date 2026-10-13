# -*- coding: utf-8 -*-
"""HX9 / water  WB3 — WB2 후속 점검 (2026-10-13 집 클로드).  WB2 결과를 본 뒤 정한 점검이므로 '사후' 표시.
 (a) WB2 전체 기록 다변량에서 GBM all47 이 +.087 이득 → ep_mean_et_sofar 는 hi10 기록에만 값이 있어(결측 = 정답 표지),
     중앙값 대치가 고EC 여부를 새게 함.  정답 유래 변수를 빼고 다시 (all46 / D 만 / usable 만) + 순열 귀무.
 (b) Y3(링크 위 하루 변화) 에서 전날 증산 쪽(prev_rad_in_sum +.27, lag2id_et_total +.37, d_et_total -.22) 이 CI 0 제외.
     평균 회귀 착시 검사: EC(b) ~ EC(a) + 변수 편상관(사슬 CI), 그리고 모든 clean 링크(319쌍)로 넓혀 같은 방향인지.
 (c) Y3 47개 가족 순열 최대 |r| p.
"""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
rng = np.random.default_rng(1013)

R = pd.read_csv(os.path.join(DATA, "records.csv")).set_index("rec")
F = pd.read_csv(os.path.join(HERE, "vars_v1.csv")).set_index("rec").loc[R.index]
CD = pd.read_csv(os.path.join(HERE, "vars_v1_columns.csv")); VARS = list(CD.variable); USAB = dict(zip(CD.variable, CD.usability))
R = R.join(F)
par = {r: r for r in R.index}
def fd(x):
    while par[x] != x:
        par[x] = par[par[x]]; x = par[x]
    return x
for b, a in R.prev_rec.items():
    if isinstance(a, str): par[fd(b)] = fd(a)
R["chain_all"] = [fd(r) for r in R.index]
AL2 = R[R.has_sg == 1].copy(); HI2 = R[(R.hi10 == 1) & (R.has_sg == 1)].copy()


def mv_logo(df, ycol, feats, base_col, groups, model, yv=None):
    y = df[ycol].values if yv is None else yv; g = df[groups].values
    X = df[feats].values.astype(float); b = df[base_col].values.astype(float)
    pb, pm = np.zeros(len(y)), np.zeros(len(y))
    for u in np.unique(g):
        te = g == u; tr = ~te
        med = np.nanmedian(X[tr], 0); med = np.where(np.isfinite(med), med, 0)
        Xtr = np.where(np.isfinite(X[tr]), X[tr], med); Xte = np.where(np.isfinite(X[te]), X[te], med)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Xtr = np.c_[(Xtr - mu) / sd, b[tr]]; Xte = np.c_[(Xte - mu) / sd, b[te]]
        pb[te] = LinearRegression().fit(b[tr, None], y[tr]).predict(b[te, None])
        m = RidgeCV(alphas=np.logspace(-1, 4, 26)) if model == "ridge" else GradientBoostingRegressor(
            n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, min_samples_leaf=3, random_state=7)
        pm[te] = m.fit(Xtr, y[tr]).predict(Xte)
    sst = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - pb) ** 2) / sst, 1 - np.sum((y - pm) ** 2) / sst, int(np.sum(np.abs(y - pm) < np.abs(y - pb))), len(y)

print("=== (a) all-record Y2 multivariate without label-derived variable ===")
NOLAB = [v for v in VARS if v != "ep_mean_et_sofar"]
DONLY = [v for v in NOLAB if USAB[v] == "D"]
USE = [v for v in NOLAB if USAB[v] in ("U", "U5", "Uid")]
for nm, feats in (("all46", NOLAB), ("D_only", DONLY), ("usable", USE)):
    for model in ("ridge", "gbm"):
        r2b, r2m, imp, n = mv_logo(AL2, "resid_mean", feats, "sg_mean", "chain_all", model)
        print(f"Y2 all | {nm:7s} {model:5s}: R2cv base {r2b:+.3f} -> {r2m:+.3f} (gain {r2m - r2b:+.3f}); improved {imp}/{n}; "
              f"hi10 among improved {int(0)}")
nul = []
for k in range(20):
    r2b, r2m, _, _ = mv_logo(AL2, "resid_mean", NOLAB, "sg_mean", "chain_all", "gbm", yv=rng.permutation(AL2.resid_mean.values))
    nul.append(r2m - r2b)
print(f"null (permuted, gbm all46, 20 reps): gain mean {np.mean(nul):+.3f}, max {np.max(nul):+.3f}")
for model in ("ridge", "gbm"):
    r2b, r2m, imp, n = mv_logo(HI2, "resid_mean", NOLAB, "sg_mean", "ep_id", model)
    print(f"Y2 hi10 | all46 {model}: R2cv base {r2b:+.3f} -> {r2m:+.3f}; improved {imp}/{n}")


def clus_ci_fn(fn, df, gcol, nb=2000):
    ug = df[gcol].unique(); idx = {u: np.where(df[gcol].values == u)[0] for u in ug}; out = []
    for _ in range(nb):
        s = np.concatenate([idx[u] for u in rng.choice(ug, len(ug), replace=True)])
        v = fn(df.iloc[s])
        if np.isfinite(v): out.append(v)
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def pcor(df, x, y, z):
    d = df[[x, y] + z].dropna()
    if len(d) < 6 or d[x].std() < 1e-12: return np.nan
    Z = np.c_[np.ones(len(d)), d[z].values]
    ex = d[x].values - Z @ np.linalg.lstsq(Z, d[x].values, rcond=None)[0]
    ey = d[y].values - Z @ np.linalg.lstsq(Z, d[y].values, rcond=None)[0]
    return np.corrcoef(ex, ey)[0, 1] if ex.std() > 1e-12 else np.nan

print("\n=== (b) Y3 regression-to-mean check: partial r(var, EC_b | EC_a) ===")
PR = R[(R.hi10 == 1) & R.prev_rec.notna()].copy()
PR = PR[[R.at[p, "hi10"] == 1 and R.at[p, "ep_id"] == R.at[b, "ep_id"] for b, p in PR.prev_rec.items()]]
PR["Y3"] = PR.ec_mean - PR.prev_ec_mean
AP = R[R.prev_rec.notna()].copy(); AP["Y3"] = AP.ec_mean - AP.prev_ec_mean
CHK = ["prev_rad_in_sum", "lag2id_et_total", "d_et_total", "prev_et_total", "et_total", "rad_in_sum", "vpd_day", "prev_sealed_hours",
       "tdiff_day", "et_3link", "rad_out_3d_dmt"]
rows = []
for v in CHK:
    d = dict(variable=v, usability=USAB[v])
    d["r_prevEC_hi"] = PR[[v, "prev_ec_mean"]].dropna().corr().iloc[0, 1]
    d["r_Y3_hi"] = PR[[v, "Y3"]].dropna().corr().iloc[0, 1]
    d["pr_ECb|ECa_hi"] = pcor(PR, v, "ec_mean", ["prev_ec_mean"])
    lo, hi = clus_ci_fn(lambda q: pcor(q, v, "ec_mean", ["prev_ec_mean"]), PR, "ep_id"); d["ci_hi"] = f"[{lo:+.2f},{hi:+.2f}]"
    d["n_hi"] = int(PR[[v, "Y3"]].dropna().shape[0])
    d["r_Y3_all"] = AP[[v, "Y3"]].dropna().corr().iloc[0, 1]
    d["pr_ECb|ECa_all"] = pcor(AP, v, "ec_mean", ["prev_ec_mean"])
    lo, hi = clus_ci_fn(lambda q: pcor(q, v, "ec_mean", ["prev_ec_mean"]), AP, "chain_all", nb=1000); d["ci_all"] = f"[{lo:+.2f},{hi:+.2f}]"
    d["pr_all_resid"] = pcor(AP[AP.has_sg == 1], v, "resid_mean", ["prev_ec_mean", "sg_mean"])
    d["n_all"] = int(AP[[v, "Y3"]].dropna().shape[0])
    rows.append(d)
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== (c) Y3 family-wise record permutation (47 vars, n=27) ===")
Xm = PR[VARS].values.astype(float); y = PR.Y3.values
def allr(yv):
    out = np.full(len(VARS), np.nan)
    for j in range(len(VARS)):
        m = np.isfinite(Xm[:, j])
        if m.sum() >= 5 and np.std(Xm[m, j]) > 0: out[j] = np.corrcoef(Xm[m, j], yv[m])[0, 1]
    return out
obs = allr(y); mx = np.array([np.nanmax(np.abs(allr(rng.permutation(y)))) for _ in range(2000)])
o = pd.Series(obs, index=VARS).dropna().sort_values(key=lambda s: -s.abs()).head(6)
print(f"95% of max|r| under null = {np.percentile(mx, 95):.3f}")
for v, r in o.items(): print(f"  {v:20s} r_Y3 {r:+.3f}  p_fw {np.mean(mx >= abs(r)):.3f}")
print("\nhi10 sealed_0_5 values:", R[R.hi10 == 1].sealed_0_5.value_counts().to_dict(), "| normal:", R[R.hi10 == 0].sealed_0_5.value_counts().sort_index().to_dict())
