# -*- coding: utf-8 -*-
"""VC1 (2026-10-13 집 클로드, adversarial verification of hx9/crosstarget CT1). Read-only on CT1 outputs.

Targets of refutation (CT1 reported 44/44 verdict 'none'; there is no signal/weak finding, so the job is to check
(a) the near-miss / CI-excludes-0 sub-findings quoted in the report, (b) the multivariate upper bound, (c) alignment &
leakage, (d) obvious variables CT1 did not build).

Checks fixed before running:
 C0 alignment: recompute 4 CT1 variables from ct_common arrays and compare with vars_v1.csv (must match to 1e-9).
 C1 near-miss list (any |r|>=.30 or a CI excluding 0 in results_v1.csv): drop-one-chain range + sign flips,
    chain-mean correlation (n chains), within-chain r, record-permutation p, Bonferroni over 44 vars x 4 targets = 176,
    partial r controlling farm + season (out_temp_d, in_temp_d), and for Y3/prev_* also prev_ec_mean (regression to mean:
    Y3 = EC(b)-EC(a) is mechanically anti-correlated with anything correlated with EC(a)).
 C2 multivariate bound: nested forward selection (1..3 vars chosen inside each LOCO training fold) and LOO-record ridge.
 C3 missed variables (raw sub_temp level, input-predicted sub_temp r3, predecessor sub_temp level / last hour, EC25-space
    residual) with the same statistics.
Run:  cd 집\\클로드\\research && PYTHONPATH="" python -u hx9/verify_crosstarget/vc1_record_level_v1.py
"""
import os
import sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
CT = os.path.join(os.path.dirname(HERE), "crosstarget")
sys.path.insert(0, CT)
import ct_common_v1 as C  # noqa: E402  (verbatim CT1 construction, writes nothing)

rng = np.random.default_rng(1013)
V = pd.read_csv(os.path.join(CT, "vars_v1.csv"))
assert (V.rec.values == C.recs).all(), "row order mismatch"
R = C.R.copy()
assert (R.rec.values == C.recs).all()

# ---------------- C0 alignment ----------------
chk = {
    "D_sumneg": np.nansum(np.where(C.Dz < -2, -C.DRES, 0), 1),
    "T_res_mean": C.TRES.mean(1),
    "L_res_mean": C.LRES.mean(1),
    "D3_sumneg": np.nansum(np.where(C.D3z < -2, -C.D3RES, 0), 1),
}
for k, v in chk.items():
    print("C0 %s max abs diff vs vars_v1: %.2e" % (k, np.nanmax(np.abs(v - V[k].values))))
# EC never enters predictors: correlation of each variable with EC hourly shape is not a leakage test; instead confirm
# that the expectation models' inputs contain no EC (by construction in ct_common: IN, ST, inputs only) -> code-read.

# ---------------- targets (same as CT1) ----------------
R["Y1"] = R.ec_mean
R["Y2"] = R.resid_mean
ecm = R.set_index("rec").ec_mean
epi = R.set_index("rec").ep_id
R["Y3"] = [R.ec_mean.iloc[i] - ecm[p] if isinstance(p, str) and p in ecm.index and np.isfinite(R.ep_id.iloc[i])
           and epi.get(p, np.nan) == R.ep_id.iloc[i] else np.nan for i, p in enumerate(R.prev_rec)]
D = pd.concat([R.reset_index(drop=True).drop(columns=[c for c in V.columns if c != "rec" and c in R.columns]), V.drop(columns="rec")], axis=1)
assert not D.columns.duplicated().any()
D["comp"] = C.comp_codes

# extra (missed) variables
ST, IN, T3, EC = C.ST, C.IN, C.T3, C.EC
D["x_st_mean"] = ST.mean(1)
D["x_st_night_min"] = ST[:, list(range(0, 6))].min(1)
D["x_T3_mean"] = T3.mean(1)                     # input-only model prediction of sub_temp (usable at h=23 as day mean)
D["x_T3_h0_5"] = T3[:, :6].mean(1)              # usable from hour 5
Vi = D.set_index("rec")
D["x_prev_st_mean"] = [Vi.loc[p, "x_st_mean"] if isinstance(p, str) and p in Vi.index else np.nan for p in D.prev_rec]
st23 = pd.Series(ST[:, 23], index=C.recs)
D["x_prev_st_h23"] = [st23[p] if isinstance(p, str) and p in st23.index else np.nan for p in D.prev_rec]
D["x_st_h0_minus_prev_h23"] = ST[:, 0] - D.x_prev_st_h23.values
EC25 = EC / (1 + 0.02 * (ST - 25))
SG25 = C.SG / (1 + 0.02 * (ST - 25))
D["x_resid25"] = np.nanmean(EC25 - SG25, 1)     # model residual in 25C-compensated space (target-side check)
D["x_T3res_minus"] = D.x_st_mean - D.x_T3_mean  # == T_res_mean (sanity)
EXTRA = ["x_st_mean", "x_st_night_min", "x_T3_mean", "x_T3_h0_5", "x_prev_st_mean", "x_prev_st_h23",
         "x_st_h0_minus_prev_h23"]
print("sanity T_res_mean vs st-T3: %.2e" % np.nanmax(np.abs(D.x_T3res_minus - D.T_res_mean)))

HI = D[D.hi10 == 1].copy()
HI2 = HI[HI.has_sg == 1].copy()
ALL = D[D.has_sg == 1].copy()
print("n HI %d (chains %d) | HI2 %d (chains %d) | Y3 pairs %d (chains %d)" % (
    len(HI), HI.ep_id.nunique(), len(HI2), HI2.ep_id.nunique(), HI.Y3.notna().sum(), HI.loc[HI.Y3.notna(), "ep_id"].nunique()))
print("chain sizes HI2:", HI2.ep_id.value_counts().sort_index().to_dict())
NTEST = 44 * 4


def pr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5 or np.std(x[m]) < 1e-12:
        return np.nan
    return np.corrcoef(x[m], y[m])[0, 1]


def perm_p(x, y, nb=20000):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    r0 = abs(np.corrcoef(x, y)[0, 1])
    xs = (x - x.mean()) / x.std()
    ys = (y - y.mean()) / y.std()
    cnt = 0
    for _ in range(nb // 1000):
        P = np.array([rng.permutation(ys) for _ in range(1000)])
        cnt += np.sum(np.abs(P @ xs / len(xs)) >= r0 - 1e-12)
    return (cnt + 1) / (nb + 1)


def partial(x, y, Z):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(1)
    if m.sum() < 8:
        return np.nan, int(m.sum())
    Zm = np.c_[np.ones(m.sum()), Z[m]]
    rx = x[m] - Zm @ np.linalg.lstsq(Zm, x[m], rcond=None)[0]
    ry = y[m] - Zm @ np.linalg.lstsq(Zm, y[m], rcond=None)[0]
    return np.corrcoef(rx, ry)[0, 1], int(m.sum())


def drop_one(x, y, g):
    out = {}
    for u in np.unique(g[np.isfinite(g)]):
        k = g != u
        out[int(u)] = pr(x[k], y[k])
    return out


def chain_mean_r(x, y, g):
    df = pd.DataFrame({"x": x, "y": y, "g": g}).dropna()
    cm = df.groupby("g")[["x", "y"]].mean()
    return (np.corrcoef(cm.x, cm.y)[0, 1] if len(cm) >= 4 else np.nan), len(cm)


def within_r(x, y, g):
    df = pd.DataFrame({"x": x, "y": y, "g": g}).dropna()
    wx = df.x - df.groupby("g").x.transform("mean")
    wy = df.y - df.groupby("g").y.transform("mean")
    return np.corrcoef(wx, wy)[0, 1] if wx.std() > 1e-12 else np.nan


NEAR = [("D3_sumneg", "Y2"), ("D3_nneg2", "Y2"), ("D3_res_sd", "Y2"), ("prev_T_res_mean", "Y2"), ("dT_res_mean", "Y2"),
        ("D_sumneg", "Y3"), ("D_nneg2", "Y3"), ("D_res_sd", "Y3"), ("T_res_sd", "Y3"), ("prev_amp_ratio", "Y3"),
        ("prev_T_res_mean", "Y3"), ("T_res_mean", "Y1"), ("D_npos3", "Y1"), ("prev_T_res_mean", "Y1"),
        ("prev_D_sumneg", "Y2all"), ("prev_D_nneg2", "Y2all"), ("D3_res_sd", "Y2all")]
NEAR += [(v, t) for v in EXTRA for t in ["Y1", "Y2", "Y3"]]
rows = []
for v, t in NEAR:
    if t == "Y2all":
        S, y, g = ALL, ALL.Y2.values, ALL.comp.values.astype(float)
    elif t == "Y2":
        S, y, g = HI2, HI2.Y2.values, HI2.ep_id.values
    else:
        S, y, g = HI, HI[t].values, HI.ep_id.values
    x = S[v].values.astype(float)
    r = pr(x, y)
    p = perm_p(x, y)
    do = drop_one(x, y, g) if t != "Y2all" else {}
    dov = np.array(list(do.values())) if do else np.array([np.nan])
    cmr, nch = chain_mean_r(x, y, g) if t != "Y2all" else (np.nan, 0)
    wr = within_r(x, y, g)
    season = S[["out_temp_d", "in_temp_d"]].values.astype(float)
    farm = (S.farm.values == "F13").astype(float)[:, None]
    p_s, _ = partial(x, y, np.c_[farm, season])
    p_e = partial(x, y, S[["prev_ec_mean"]].values.astype(float))[0] if t in ("Y3", "Y2", "Y1") else np.nan
    p_es = partial(x, y, np.c_[farm, season, S[["prev_ec_mean"]].values.astype(float)])[0] if t in ("Y3", "Y2", "Y1") else np.nan
    rx_prevec = pr(x, S.prev_ec_mean.values.astype(float))
    # pass split (only meaningful for Y2all)
    rp1 = pr(x[S.pass2.values == 0], y[S.pass2.values == 0])
    rp2 = pr(x[S.pass2.values == 1], y[S.pass2.values == 1])
    n = int((np.isfinite(x) & np.isfinite(y)).sum())
    rows.append(dict(var=v, target=t, n=n, r=r, perm_p=p, p_bonf176=min(1, p * NTEST),
                     drop1_min=np.nanmin(dov), drop1_max=np.nanmax(dov), drop1_signflip=int(np.nansum(np.sign(dov) != np.sign(r))),
                     worst_drop=(min(do, key=lambda k: abs(do[k])) if do else -1),
                     r_chain_means=cmr, n_chains=nch, r_within_chain=wr, r_partial_farm_season=p_s,
                     r_partial_prevEC=p_e, r_partial_all=p_es, r_x_prevEC=rx_prevec, r_pass1=rp1, r_pass2=rp2,
                     n_pass2=int(((S.pass2.values == 1) & np.isfinite(x) & np.isfinite(y)).sum())))
T = pd.DataFrame(rows)
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)
print("\n== C1/C3 near-miss and missed variables ==")
print(T.round(3).to_string(index=False))
T.to_csv(os.path.join(HERE, "vc1_nearmiss_v1.csv"), index=False)

# regression-to-mean mechanics for prev_T_res_mean
print("\n-- regression-to-mean check (HI, Y3 pairs) --")
m = HI.Y3.notna()
print("r(prev_T_res_mean, prev_ec_mean) %.3f | r(prev_ec_mean, Y3) %.3f | r(T_res_mean, ec_mean) all HI %.3f" % (
    pr(HI.loc[m, "prev_T_res_mean"].values, HI.loc[m, "prev_ec_mean"].values),
    pr(HI.loc[m, "prev_ec_mean"].values, HI.loc[m, "Y3"].values), pr(HI.T_res_mean.values, HI.ec_mean.values)))
print("r(prev_T_res_mean, Y1 of b) %.3f | r(prev_T_res_mean, prev resid) %.3f" % (
    pr(HI.loc[m, "prev_T_res_mean"].values, HI.loc[m, "Y1"].values),
    pr(HI.loc[m, "prev_T_res_mean"].values, HI.loc[m, "prev_rec"].map(D.set_index("rec").resid_mean).values)))

# ---------------- C2 multivariate bound: nested selection ----------------
SAMEDAY = [c for c in V.columns if c != "rec" and not c.startswith("prev_") and not c.startswith("d")] + \
          ["D_res_sd", "D_nneg2", "D_npos2", "D_nneg3", "D_npos3", "D_sumneg", "D_sumpos", "D_minres", "D_maxres",
           "D_firstneg_h", "D_neg_morn", "D_neg_aft", "D_neg_night", "D_netanom", "D3_res_sd", "D3_nneg2", "D3_sumneg"]
SAMEDAY = list(dict.fromkeys(SAMEDAY))
print("\n== C2 nested forward-selection LOCO (vars %d) ==" % len(SAMEDAY))


def ols_fit_pred(Xtr, ytr, Xte):
    A = np.c_[np.ones(len(Xtr)), Xtr]
    c = np.linalg.lstsq(A, ytr, rcond=None)[0]
    return np.c_[np.ones(len(Xte)), Xte] @ c


def nested(S, ycol, gcol, k):
    y = S[ycol].values
    g = S[gcol].values
    base = S.sg_mean.fillna(S.sg_mean.mean()).values
    X = S[SAMEDAY].values.astype(float)
    med = np.nanmedian(X, 0)
    X = np.where(np.isfinite(X), X, med)
    pb, pa = np.empty(len(y)), np.empty(len(y))
    chosen = []
    for u in np.unique(g):
        te, tr = g == u, g != u
        pb[te] = ols_fit_pred(base[tr, None], y[tr], base[te, None])
        sel = []
        for _ in range(k):  # greedy inner LOCO selection
            best, bj = np.inf, None
            for j in range(X.shape[1]):
                if j in sel:
                    continue
                cols = sel + [j]
                sse = 0
                for u2 in np.unique(g[tr]):
                    te2, tr2 = tr & (g == u2), tr & (g != u2)
                    pp = ols_fit_pred(np.c_[base[tr2], X[tr2][:, cols]], y[tr2], np.c_[base[te2], X[te2][:, cols]])
                    sse += np.sum((y[te2] - pp) ** 2)
                if sse < best:
                    best, bj = sse, j
            sel.append(bj)
        chosen.append([SAMEDAY[j] for j in sel])
        pa[te] = ols_fit_pred(np.c_[base[tr], X[tr][:, sel]], y[tr], np.c_[base[te], X[te][:, sel]])
    sst = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - pb) ** 2) / sst, 1 - np.sum((y - pa) ** 2) / sst, int(np.sum(np.abs(y - pa) < np.abs(y - pb))), chosen


for k in (1, 2, 3):
    for name, S, yc in [("Y2hi", HI2, "Y2"), ("Y1hi", HI, "Y1")]:
        rb, ra, imp, ch = nested(S, yc, "ep_id", k)
        print("k=%d %s: R2cv base %.3f -> aug %.3f (gain %+.3f), improved %d/%d; chosen per fold: %s" % (
            k, name, rb, ra, ra - rb, imp, len(S), [c for c in ch][:8]))

# LOO-record ridge (optimistic: chain neighbours stay in training) as an extra upper bound
print("\n-- LOO-record ridge (optimistic, chain-mates in training) --")
for name, S, yc in [("Y2hi", HI2, "Y2"), ("Y1hi", HI, "Y1")]:
    y = S[yc].values
    base = S.sg_mean.fillna(S.sg_mean.mean()).values
    X = S[SAMEDAY].values.astype(float)
    X = np.where(np.isfinite(X), X, np.nanmedian(X, 0))
    Xb = np.c_[base, X]
    for a in (10, 100, 1000):
        p = np.empty(len(y))
        pb = np.empty(len(y))
        for i in range(len(y)):
            tr = np.arange(len(y)) != i
            mu, sd = Xb[tr].mean(0), Xb[tr].std(0) + 1e-9
            p[i] = Ridge(alpha=a).fit((Xb[tr] - mu) / sd, y[tr]).predict(((Xb[i] - mu) / sd)[None])[0]
            pb[i] = ols_fit_pred(base[tr, None], y[tr], base[[i], None])[0]
        sst = np.sum((y - y.mean()) ** 2)
        print("  %s alpha %d: R2 base %.3f | ridge %.3f" % (name, a, 1 - np.sum((y - pb) ** 2) / sst, 1 - np.sum((y - p) ** 2) / sst))
print("done")
