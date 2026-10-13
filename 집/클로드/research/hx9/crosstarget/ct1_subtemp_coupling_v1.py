# -*- coding: utf-8 -*-
"""CT1 (hx9 family: cross-target coupling with substrate-temperature dynamics), 2026-10-13 집 클로드.

Question (user): 고EC 의 크기 예측에 인과관계가 조금이라도 있는 데이터 전부 확인. This family: irrigation pulses
should change sub_temp (water temperature != substrate) and EC at the same time; a flush cools/warms the substrate and
drops EC.  sub_temp is the OTHER target's label -> every variable built from it is DIAGNOSTIC ONLY for an evaluation
row (the eval record's own sub_temp is unknown).  Exception noted: variables of a same-ID PREDECESSOR that is a
training record would be computable if the predecessor can be found causally (linker) -- here truth links are used,
so they are still diagnostic.

What is different from earlier tests (catalog 6.544(5), 6.545, 6.634, 6b.36, 6b.39, 6.457):
 - earlier: raw sub_temp 'sudden drops' (count/day, EC change at that hour) vs EC over ALL days, season-confounded.
 - here: anomalies relative to an in_temp-driven thermal expectation (level and first-order derivative model,
   cross-fitted), plus substrate thermal-dynamics proxies of water content (amplitude ratio, relaxation rate,
   peak lag), plus the full sub_temp model residual (r3 OOF) -- tested against high-EC MAGNITUDE (level among hi10,
   residual y-sg among hi10, within-episode day-to-day change), with chain-level CIs and leave-one-chain-out CV gain.

================================================================================================================
PRE-REGISTERED FINGERPRINTS (written before any number below was computed)
If irrigation pulses / flushes drive high-EC magnitude, I expect:
 P1  Among hi10 records, FEWER / SMALLER anomalous negative sub_temp steps (less irrigation, less dilution) go with
     HIGHER EC level and a more positive model residual: r(D_sumneg, Y1) < 0 and r(D_sumneg, Y2) < 0 (|r| >= .35).
 P2  Within high episodes, a day with a big anomalous step (flush) has a LOWER day mean than the previous day:
     r(D_sumneg(b), dEC) < 0, and r(prev_D_sumneg, dEC) < 0 (flush yesterday -> lower today).
 P3  Hourly: at anomalous sub_temp step hours the temperature-compensated EC (EC25) steps down in the same or next
     1-2 hours (dilution), clearly more than at matched control hours (same farm x hour), strongest in hi10 records.
 P4  Water-content proxies: a wetter substrate has larger heat capacity -> lower amplitude ratio / slower relaxation
     (lower k_relax) / longer peak lag; wetter = more diluted -> r(amp_ratio, Y1) > 0, r(k_relax, Y1) > 0.
 P5  Day-level sub_temp model residual and EC model residual co-move (shared unobserved irrigation factor):
     corr(T_res_mean, Y2) > 0 (cool irrigation water -> colder substrate AND diluted EC) among hi10 and all records.
Must NOT be seen (otherwise the mechanism reading is wrong even if some r is large):
 N1  the association disappears within episodes (only between-chain) -> house fingerprint, not irrigation;
 N2  the anomaly is fully reproduced by actuators/outdoor inputs (then it is air/operation, not water);
 N3  opposite sign in pass 1 vs pass 2;
 N4  the effect hangs on one chain (leave-one-chain-out CV gain <= 0).
Verdict rule (fixed before running): signal = |r_Y2hi| >= .35 AND chain-bootstrap 95% CI excludes 0 AND LOCO CV R2
gain > 0; weak = two of those three, or |r| >= .35 with CI excluding 0 for Y1 or Y3; none otherwise.
================================================================================================================

Run:  cd 집\\클로드\\research && PYTHONPATH="" python -u hx9/crosstarget/ct1_subtemp_coupling_v1.py
"""
import os
import sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
OUT = HERE
RNG = np.random.default_rng(20261013)
NB = 2000

R = pd.read_csv(os.path.join(DATA, "records.csv"))
H = pd.read_csv(os.path.join(DATA, "hourly.csv"))
H = H.sort_values(["rec", "hour"]).reset_index(drop=True)
assert (H.groupby("rec").size() == 24).all()
recs = H.rec.unique()
R = R.set_index("rec").loc[recs].reset_index()
NR = len(recs)
INP = ["out_temp", "out_hum", "out_rad", "out_wspd", "in_temp", "in_hum", "in_co2",
       "act_vent", "act_shade", "act_thermal", "act_heating", "act_circfan", "act_co2", "act_fog"]


def arr(col):
    return H[col].values.reshape(NR, 24).astype(float)


def interp_rows(A):
    B = A.copy()
    for i in range(B.shape[0]):
        m = np.isfinite(B[i])
        if m.all():
            continue
        if m.sum() == 0:
            B[i] = np.nan
            continue
        B[i] = np.interp(np.arange(24), np.where(m)[0], B[i, m])
    return B


ST = arr("sub_temp")
EC = arr("sub_ec")
SG = arr("sg")
IN = interp_rows(arr("in_temp"))
FARM = R.farm.values
HOUR = np.tile(np.arange(24), (NR, 1))

# ---------------- chain / component groups -----------------
par = {r: r for r in recs}


def fd(x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


for r_, p_ in zip(R.rec, R.prev_rec):
    if isinstance(p_, str) and p_ in par:
        par[fd(r_)] = fd(p_)
COMP = np.array([fd(r) for r in recs])
comp_codes = pd.factorize(COMP)[0]
print("link components:", len(set(COMP)), "largest", pd.Series(COMP).value_counts().max())

# ---------------- r3 OOF sub_temp model -----------------
z = np.load(os.path.join(env.LOCAL, "oof_temp_diag.npz"), allow_pickle=True)
r3 = pd.Series(z["r3"], index=z["row_id"])
rid = (H.rec + "_" + H.hour.map(lambda h: "%02d" % h)).values
assert set(rid) <= set(r3.index)
T3 = r3.loc[rid].values.reshape(NR, 24)
TRES = ST - T3

# ---------------- expectation models (cross-fitted, per farm) -----------------
gkf = GroupKFold(n_splits=10)


def harm(h):
    return np.c_[np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(4 * np.pi * h / 24), np.cos(4 * np.pi * h / 24)]


def crossfit(Xh, yh, rec_idx, alpha=1.0):
    """Xh, yh rows; rec_idx = record index per row. per-farm ridge, GroupKFold by link component."""
    pred = np.full(len(yh), np.nan)
    for f in ["F13", "F47"]:
        m = (FARM[rec_idx] == f) & np.isfinite(yh) & np.isfinite(Xh).all(1)
        idx = np.where(m)[0]
        g = comp_codes[rec_idx[idx]]
        for tr, te in gkf.split(idx, groups=g):
            mu = Xh[idx[tr]].mean(0)
            sd = Xh[idx[tr]].std(0) + 1e-9
            mdl = Ridge(alpha=alpha).fit((Xh[idx[tr]] - mu) / sd, yh[idx[tr]])
            pred[idx[te]] = mdl.predict((Xh[idx[te]] - mu) / sd)
    return pred


# level model L: ST_h ~ in_temp lags 0..4 (lag filled with In_0 before hour 0) + EWMA(hl 3h) + hour harmonics
def lagged(A, k):
    B = np.empty_like(A)
    B[:, k:] = A[:, :24 - k]
    B[:, :k] = A[:, [0]]
    return B


def ewma(A, hl):
    a = 1 - 0.5 ** (1 / hl)
    B = np.empty_like(A)
    B[:, 0] = A[:, 0]
    for h in range(1, 24):
        B[:, h] = a * A[:, h] + (1 - a) * B[:, h - 1]
    return B


rec_rows = np.repeat(np.arange(NR), 24)
hrs = np.tile(np.arange(24), NR)
XL = np.c_[np.stack([lagged(IN, k).ravel() for k in range(5)], 1), ewma(IN, 3).ravel(), ewma(IN, 8).ravel(),
           harm(hrs), (hrs < 4).astype(float)]
LPRED = crossfit(XL, ST.ravel(), rec_rows).reshape(NR, 24)
LRES = ST - LPRED

# derivative model D2: dST_h ~ (In_h - ST_{h-1}), (In_{h-1}-ST_{h-1}), dIn_h, dIn_{h-1}, dIn_{h-2}, harmonics ; h = 1..23
dST = np.full((NR, 24), np.nan)
dST[:, 1:] = np.diff(ST, axis=1)
dIN = np.full((NR, 24), np.nan)
dIN[:, 1:] = np.diff(IN, axis=1)
STm1 = np.full((NR, 24), np.nan)
STm1[:, 1:] = ST[:, :-1]
INm1 = np.full((NR, 24), np.nan)
INm1[:, 1:] = IN[:, :-1]
dINm1 = np.nan_to_num(lagged(np.nan_to_num(dIN), 1))
dINm2 = np.nan_to_num(lagged(np.nan_to_num(dIN), 2))
dINm1[:, :2] = 0
dINm2[:, :3] = 0
XD = np.c_[(IN - STm1).ravel(), (INm1 - STm1).ravel(), np.nan_to_num(dIN).ravel(), dINm1.ravel(), dINm2.ravel(), harm(hrs)]
yD = dST.ravel()
DPRED = crossfit(XD, yD, rec_rows).reshape(NR, 24)
DRES = dST - DPRED

# D3: derivative model with ALL inputs (levels + 1h changes + lag-1/2 changes) -> unexplained-by-any-input step
FULL = {v: interp_rows(arr(v)) for v in INP}
cols3 = [XD]
for v in INP:
    A = FULL[v]
    d = np.zeros_like(A)
    d[:, 1:] = np.diff(A, axis=1)
    cols3 += [A.ravel()[:, None], d.ravel()[:, None], lagged(d, 1).ravel()[:, None], lagged(d, 2).ravel()[:, None]]
XD3 = np.concatenate(cols3, 1)
XD3 = np.where(np.isfinite(XD3), XD3, 0.0)
D3PRED = crossfit(XD3, yD, rec_rows).reshape(NR, 24)
D3RES = dST - D3PRED


def r2(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    return 1 - np.sum((y[m] - p[m]) ** 2) / np.sum((y[m] - y[m].mean()) ** 2)


print("expectation models (cross-fitted R2): L level %.3f | D2 derivative %.3f | D3 derivative(all inputs) %.3f | r3 level %.3f"
      % (r2(ST.ravel(), LPRED.ravel()), r2(yD, DPRED.ravel()), r2(yD, D3PRED.ravel()), r2(ST.ravel(), T3.ravel())))

# robust sigma per farm
SIG = {}
for f in ["F13", "F47"]:
    v = DRES[FARM == f][:, 1:].ravel()
    v = v[np.isfinite(v)]
    SIG[f] = 1.4826 * np.median(np.abs(v - np.median(v)))
    v3 = D3RES[FARM == f][:, 1:].ravel()
    v3 = v3[np.isfinite(v3)]
    SIG[f + "_3"] = 1.4826 * np.median(np.abs(v3 - np.median(v3)))
print("robust sigma of D2 residual (deg C/h):", {k: round(v, 3) for k, v in SIG.items()})
sig = np.array([SIG[f] for f in FARM])[:, None]
sig3 = np.array([SIG[f + "_3"] for f in FARM])[:, None]

# ---------------- per-record variables -----------------
V = pd.DataFrame({"rec": recs})
NIGHT = list(range(0, 6))
DAY = list(range(9, 17))
EVE = list(range(17, 24))
V["L_res_mean"] = LRES.mean(1)
V["L_res_night"] = LRES[:, NIGHT].mean(1)
V["L_res_day"] = LRES[:, DAY].mean(1)
V["L_res_eve"] = LRES[:, EVE].mean(1)
V["L_res_h0"] = LRES[:, 0]
V["L_res_sd"] = LRES.std(1)
Dz = DRES / sig
V["D_res_sd"] = np.nanstd(DRES[:, 1:], 1)
neg2 = Dz < -2
pos2 = Dz > 2
V["D_nneg2"] = np.nansum(neg2, 1)
V["D_npos2"] = np.nansum(pos2, 1)
V["D_nneg3"] = np.nansum(Dz < -3, 1)
V["D_npos3"] = np.nansum(Dz > 3, 1)
V["D_sumneg"] = np.nansum(np.where(neg2, -DRES, 0), 1)       # magnitude (deg C) of anomalous cooling steps
V["D_sumpos"] = np.nansum(np.where(pos2, DRES, 0), 1)
V["D_minres"] = np.nanmin(DRES[:, 1:], 1)
V["D_maxres"] = np.nanmax(DRES[:, 1:], 1)
fn = np.where(neg2.any(1), np.argmax(np.nan_to_num(neg2), 1), 24)
V["D_firstneg_h"] = fn
V["D_neg_morn"] = np.nansum(neg2[:, 6:12], 1)
V["D_neg_aft"] = np.nansum(neg2[:, 12:18], 1)
V["D_neg_night"] = np.nansum(neg2[:, 18:24], 1) + np.nansum(neg2[:, 1:6], 1)
V["D_netanom"] = np.nansum(DRES[:, 1:], 1)
D3z = D3RES / sig3
V["D3_res_sd"] = np.nanstd(D3RES[:, 1:], 1)
V["D3_nneg2"] = np.nansum(D3z < -2, 1)
V["D3_sumneg"] = np.nansum(np.where(D3z < -2, -D3RES, 0), 1)
ampST = ST.max(1) - ST.min(1)
ampIN = IN.max(1) - IN.min(1)
V["amp_ratio"] = ampST / np.maximum(ampIN, 0.5)
V["peak_lag"] = ((np.argmax(ST, 1) - np.argmax(IN, 1) + 12) % 24) - 12
kr, cr = [], []
for i in range(NR):
    x = (IN[i, :-1] - ST[i, :-1])
    y = np.diff(ST[i])
    kr.append(np.polyfit(x, y, 1)[0] if np.std(x) > 1e-6 else np.nan)
    a, b = np.diff(ST[i])[2:], np.diff(IN[i])[:-2]
    cr.append(np.corrcoef(a, b)[0, 1] if np.std(a) > 1e-6 and np.std(b) > 1e-6 else np.nan)
V["k_relax"] = kr
V["coup_r_lag2"] = cr
V["st_amp"] = ampST
V["st_in_gap_mean"] = (ST - IN).mean(1)
V["st_in_gap_night"] = (ST - IN)[:, NIGHT].mean(1)
V["st_in_gap_day"] = (ST - IN)[:, DAY].mean(1)
V["T_res_mean"] = TRES.mean(1)
V["T_res_night"] = TRES[:, NIGHT].mean(1)
V["T_res_day"] = TRES[:, DAY].mean(1)
V["T_res_h0"] = TRES[:, 0]
V["T_res_sd"] = TRES.std(1)
SAMEDAY = [c for c in V.columns if c != "rec"]
# predecessor (clean link, truth -> diagnostic) versions
Vi = V.set_index("rec")
for c in ["D_sumneg", "D_nneg2", "L_res_mean", "amp_ratio", "T_res_mean", "k_relax"]:
    V["prev_" + c] = [Vi.loc[p, c] if isinstance(p, str) and p in Vi.index else np.nan for p in R.prev_rec]
V["dL_res_mean"] = V.L_res_mean - V.prev_L_res_mean
V["dT_res_mean"] = V.T_res_mean - V.prev_T_res_mean
VARS = [c for c in V.columns if c != "rec"]
print("variables built:", len(VARS))
V.to_csv(os.path.join(OUT, "vars_v1.csv"), index=False)

# ---------------- targets -----------------
R["Y1"] = R.ec_mean
R["Y2"] = R.resid_mean
ecm = R.set_index("rec").ec_mean
epi = R.set_index("rec").ep_id
R["Y3"] = [R.ec_mean.iloc[i] - ecm[p] if isinstance(p, str) and p in ecm.index and np.isfinite(R.ep_id.iloc[i])
           and epi.get(p, np.nan) == R.ep_id.iloc[i] else np.nan for i, p in enumerate(R.prev_rec)]
R["Y3all"] = [R.ec_mean.iloc[i] - ecm[p] if isinstance(p, str) and p in ecm.index else np.nan for i, p in enumerate(R.prev_rec)]
EC25 = EC / (1 + 0.02 * (ST - 25))
R["Y5"] = np.log(EC25[:, 23] / EC25[:, 0])      # within-day temperature-compensated change
D = pd.concat([R[["rec", "farm", "pass2", "hi10", "ep_id", "has_sg", "sg_mean", "ec_mean", "Y1", "Y2", "Y3", "Y3all", "Y5"]],
               V[VARS]], axis=1)
D["comp"] = comp_codes
HI = D[D.hi10 == 1].copy()
HI2 = HI[HI.has_sg == 1].copy()
print("n hi10 %d | hi10 with OOF %d | within-episode pairs %d | all with OOF %d | all clean links %d"
      % (len(HI), len(HI2), HI.Y3.notna().sum(), (D.has_sg == 1).sum(), D.Y3all.notna().sum()))


def corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5 or np.std(x[m]) < 1e-12:
        return np.nan, np.nan, int(m.sum())
    return stats.pearsonr(x[m], y[m])[0], stats.spearmanr(x[m], y[m])[0], int(m.sum())


def boot_ci(x, y, g, nb=NB):
    m = np.isfinite(x) & np.isfinite(y)
    x, y, g = x[m], y[m], g[m]
    ug = np.unique(g)
    idx_by = {u: np.where(g == u)[0] for u in ug}
    out = []
    for _ in range(nb):
        s = RNG.choice(ug, len(ug), replace=True)
        ii = np.concatenate([idx_by[u] for u in s])
        if np.std(x[ii]) < 1e-12 or np.std(y[ii]) < 1e-12:
            continue
        out.append(np.corrcoef(x[ii], y[ii])[0, 1])
    return np.nanpercentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


def loco_gain(x, y, base, g):
    """OLS y~1+base vs y~1+base+x, leave-one-group-out; returns (R2cv_base, R2cv_aug, gain, n_improved, n)."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(base)
    x, y, base, g = x[m], y[m], base[m], g[m]
    pb, pa = np.empty(len(y)), np.empty(len(y))
    for u in np.unique(g):
        te = g == u
        tr = ~te
        Xb = np.c_[np.ones(tr.sum()), base[tr]]
        Xa = np.c_[Xb, x[tr]]
        cb = np.linalg.lstsq(Xb, y[tr], rcond=None)[0]
        ca = np.linalg.lstsq(Xa, y[tr], rcond=None)[0]
        pb[te] = np.c_[np.ones(te.sum()), base[te]] @ cb
        pa[te] = np.c_[np.ones(te.sum()), base[te], x[te]] @ ca
    sst = np.sum((y - y.mean()) ** 2)
    rb = 1 - np.sum((y - pb) ** 2) / sst
    ra = 1 - np.sum((y - pa) ** 2) / sst
    return rb, ra, ra - rb, int(np.sum(np.abs(y - pa) < np.abs(y - pb))), int(len(y))


rows = []
for v in VARS:
    x1 = HI[v].values.astype(float)
    x2 = HI2[v].values.astype(float)
    xa = D[v].values.astype(float)
    r1p, r1s, n1 = corr(x1, HI.Y1.values)
    r2p, r2s, n2 = corr(x2, HI2.Y2.values)
    ci_lo, ci_hi = boot_ci(x2, HI2.Y2.values, HI2.ep_id.values)
    ci1 = boot_ci(x1, HI.Y1.values, HI.ep_id.values)
    m_all = D.has_sg.values == 1
    rap, ras, na = corr(xa[m_all], D.Y2.values[m_all])
    ca_lo, ca_hi = boot_ci(xa[m_all], D.Y2.values[m_all], D.comp.values[m_all], nb=1000)
    r3p, r3s, n3 = corr(x1, HI.Y3.values)
    ci3 = boot_ci(x1, HI.Y3.values, HI.ep_id.values)
    r3ap, _, n3a = corr(xa, D.Y3all.values)
    r5p, _, n5 = corr(x1, HI.Y5.values)
    r5ap, _, n5a = corr(xa, D.Y5.values)
    # pass-1 / pass-2 on all-record residual
    p1 = m_all & (D.pass2.values == 0)
    p2 = m_all & (D.pass2.values == 1)
    rp1 = corr(xa[p1], D.Y2.values[p1])[0]
    rp2 = corr(xa[p2], D.Y2.values[p2])[0]
    rh1 = corr(x2[HI2.pass2.values == 0], HI2.Y2.values[HI2.pass2.values == 0])[0]
    g = loco_gain(x2, HI2.Y2.values, HI2.sg_mean.values, HI2.ep_id.values)
    g1 = loco_gain(x1, HI.Y1.values, HI.sg_mean.fillna(HI.sg_mean.mean()).values, HI.ep_id.values)
    ga = loco_gain(xa[m_all], D.Y2.values[m_all], D.sg_mean.values[m_all], D.comp.values[m_all])
    # within-episode (demeaned) correlation for Y1 -> house-fingerprint check (N1)
    hx = HI[[v, "Y1", "ep_id"]].dropna()
    if len(hx) > 5:
        wx = hx[v] - hx.groupby("ep_id")[v].transform("mean")
        wy = hx.Y1 - hx.groupby("ep_id").Y1.transform("mean")
        rw = np.corrcoef(wx, wy)[0, 1] if wx.std() > 1e-12 else np.nan
    else:
        rw = np.nan
    # delta-variable vs delta EC within episodes
    rows.append(dict(variable=v, n_hi=n1, r_Y1=r1p, rho_Y1=r1s, ci_Y1_lo=ci1[0], ci_Y1_hi=ci1[1], r_Y1_within_ep=rw,
                     n_hi_sg=n2, r_Y2hi=r2p, rho_Y2hi=r2s, ci_Y2hi_lo=ci_lo, ci_Y2hi_hi=ci_hi,
                     cv_base_Y2hi=g[0], cv_aug_Y2hi=g[1], cv_gain_Y2hi=g[2], n_improved_Y2hi=g[3],
                     cv_gain_Y1=g1[2], n_all=na, r_Y2all=rap, rho_Y2all=ras, ci_Y2all_lo=ca_lo, ci_Y2all_hi=ca_hi,
                     cv_gain_Y2all=ga[2], r_Y2all_pass1=rp1, r_Y2all_pass2=rp2, r_Y2hi_pass1=rh1,
                     n_Y3=n3, r_Y3=r3p, rho_Y3=r3s, ci_Y3_lo=ci3[0], ci_Y3_hi=ci3[1], n_Y3all=n3a, r_Y3all=r3ap,
                     r_Y5hi=r5p, n_Y5hi=n5, r_Y5all=r5ap))
res = pd.DataFrame(rows)

# family-wise null for max |r| (record-level permutation; anti-conservative because records in chains are dependent)
def maxnull(Xmat, y, nb=2000):
    m = np.isfinite(y)
    Xm = Xmat[m]
    y = y[m]
    mx = []
    Z = (Xm - np.nanmean(Xm, 0)) / (np.nanstd(Xm, 0) + 1e-12)
    Z = np.nan_to_num(Z)
    for _ in range(nb):
        yp = RNG.permutation(y)
        yz = (yp - yp.mean()) / yp.std()
        mx.append(np.max(np.abs(Z.T @ yz / len(yz))))
    return np.percentile(mx, 95)


thr_Y2hi = maxnull(HI2[SAMEDAY].values.astype(float), HI2.Y2.values)
thr_Y1 = maxnull(HI[SAMEDAY].values.astype(float), HI.Y1.values)
thr_Y2all = maxnull(D.loc[D.has_sg == 1, SAMEDAY].values.astype(float), D.loc[D.has_sg == 1, "Y2"].values, nb=500)
print("family-wise 95%% max|r| null (same-day vars, record permutation): Y1 %.3f | Y2hi %.3f | Y2all %.3f"
      % (thr_Y1, thr_Y2hi, thr_Y2all))


def verdict(r):
    a = abs(r.r_Y2hi) >= .35 if np.isfinite(r.r_Y2hi) else False
    b = (r.ci_Y2hi_lo > 0) or (r.ci_Y2hi_hi < 0)
    c = r.cv_gain_Y2hi > 0
    if a and b and c:
        return "signal"
    y1 = np.isfinite(r.r_Y1) and abs(r.r_Y1) >= .35 and ((r.ci_Y1_lo > 0) or (r.ci_Y1_hi < 0))
    y3 = np.isfinite(r.r_Y3) and abs(r.r_Y3) >= .35 and ((r.ci_Y3_lo > 0) or (r.ci_Y3_hi < 0))
    if (a + b + c) >= 2 or y1 or y3:
        return "weak"
    return "none"


res["verdict"] = res.apply(verdict, axis=1)
res.to_csv(os.path.join(OUT, "results_v1.csv"), index=False)
pd.set_option("display.width", 250)
show = ["variable", "n_hi", "r_Y1", "r_Y1_within_ep", "n_hi_sg", "r_Y2hi", "rho_Y2hi", "ci_Y2hi_lo", "ci_Y2hi_hi",
        "cv_gain_Y2hi", "n_improved_Y2hi", "r_Y2all", "r_Y2all_pass1", "r_Y2all_pass2", "cv_gain_Y2all", "n_Y3", "r_Y3", "r_Y3all",
        "r_Y5hi", "verdict"]
print(res[show].round(3).to_string(index=False))

# ---------------- multivariate upper bound -----------------
def loco_ridge(X, y, g, alphas=(1, 10, 100, 1000)):
    p = np.empty(len(y))
    for u in np.unique(g):
        te = g == u
        tr = ~te
        best, ba = np.inf, alphas[0]
        for a in alphas:
            sse = 0
            for u2 in np.unique(g[tr]):
                te2 = tr & (g == u2)
                tr2 = tr & (g != u2)
                mu, sd = X[tr2].mean(0), X[tr2].std(0) + 1e-9
                mdl = Ridge(alpha=a).fit((X[tr2] - mu) / sd, y[tr2])
                sse += np.sum((y[te2] - mdl.predict((X[te2] - mu) / sd)) ** 2)
            if sse < best:
                best, ba = sse, a
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        p[te] = Ridge(alpha=ba).fit((X[tr] - mu) / sd, y[tr]).predict((X[te] - mu) / sd)
    return p


def loco_gbm(X, y, g, seeds=(0, 1, 2)):
    p = np.zeros(len(y))
    for s in seeds:
        for u in np.unique(g):
            te = g == u
            mdl = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
                                            min_samples_leaf=3, random_state=s).fit(X[~te], y[~te])
            p[te] += mdl.predict(X[te]) / len(seeds)
    return p


def loco_base(y, base, g):
    p = np.empty(len(y))
    for u in np.unique(g):
        te = g == u
        c = np.linalg.lstsq(np.c_[np.ones((~te).sum()), base[~te]], y[~te], rcond=None)[0]
        p[te] = np.c_[np.ones(te.sum()), base[te]] @ c
    return p


def impute(X):
    X = X.copy()
    med = np.nanmedian(X, 0)
    i = np.where(~np.isfinite(X))
    X[i] = np.take(med, i[1])
    return X


print("\n== multivariate non-causal upper bound (leave-one-chain-out) ==")
MV = {}
for name, dset, ycol, gcol, vlist in [("Y2hi same-day", HI2, "Y2", "ep_id", SAMEDAY),
                                       ("Y2hi +prev", HI2, "Y2", "ep_id", VARS),
                                       ("Y1hi same-day", HI, "Y1", "ep_id", SAMEDAY),
                                       ("Y2all same-day", D[D.has_sg == 1], "Y2", "comp", SAMEDAY)]:
    y = dset[ycol].values
    g = dset[gcol].values
    if name.startswith("Y2all"):
        # group k-fold 10 by component for speed
        gg = np.zeros(len(y), int)
        for k, (_, te) in enumerate(GroupKFold(10).split(y, groups=g)):
            gg[te] = k
        g = gg
    base = dset.sg_mean.fillna(dset.sg_mean.mean()).values
    X = impute(dset[vlist].values.astype(float))
    pb = loco_base(y, base, g)
    Xb = np.c_[base, X]
    pr = loco_ridge(Xb, y, g, alphas=(1, 10, 100, 1000) if len(y) < 100 else (10, 100, 1000))
    pg = loco_gbm(Xb, y, g) if len(y) < 100 else loco_gbm(Xb, y, g, seeds=(0,))
    sst = np.sum((y - y.mean()) ** 2)
    out = dict(n=len(y), r2_base=1 - np.sum((y - pb) ** 2) / sst, r2_ridge=1 - np.sum((y - pr) ** 2) / sst,
               r2_gbm=1 - np.sum((y - pg) ** 2) / sst, imp_ridge=int(np.sum(np.abs(y - pr) < np.abs(y - pb))),
               imp_gbm=int(np.sum(np.abs(y - pg) < np.abs(y - pb))))
    # chance reference: same pipeline with y permuted within... (shuffle variables across records, 20 reps, ridge only)
    nulls = []
    for k in range(20 if len(y) < 100 else 3):
        Xp = X[RNG.permutation(len(y))]
        prp = loco_ridge(np.c_[base, Xp], y, g, alphas=(1, 10, 100, 1000) if len(y) < 100 else (10, 100, 1000))
        nulls.append(1 - np.sum((y - prp) ** 2) / sst)
    out["r2_ridge_null_med"] = float(np.median(nulls))
    out["r2_ridge_null_p95"] = float(np.percentile(nulls, 95))
    MV[name] = out
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()})
pd.DataFrame(MV).T.to_csv(os.path.join(OUT, "multivariate_v1.csv"))

# ---------------- hourly coupling: ST anomaly steps vs EC25 steps -----------------
print("\n== hourly: D2 anomaly vs EC25 steps ==")
dlE = np.full((NR, 24), np.nan)
dlE[:, 1:] = np.diff(np.log(EC25), axis=1)
# control: subtract farm x hour mean of dlE
ctrl = np.zeros_like(dlE)
for f in ["F13", "F47"]:
    m = FARM == f
    ctrl[m] = np.nanmean(dlE[m], 0)
dlEc = dlE - ctrl
hi_mask = (R.hi10.values == 1)
hrows = []
for lab, msk in [("all", np.ones(NR, bool)), ("hi10", hi_mask), ("normal", ~hi_mask)]:
    for j in [-1, 0, 1, 2]:
        a = DRES[msk][:, 1:]
        b = np.full_like(a, np.nan)
        src = dlEc[msk]
        for h in range(1, 24):
            if 1 <= h + j <= 23:
                b[:, h - 1] = src[:, h + j]
        r_ = corr(a.ravel(), b.ravel())
        hrows.append(dict(set=lab, lag=j, r_Dres_dlogEC25=r_[0], rho=r_[1], n=r_[2]))
    # event study: neg anomaly hours
    for kind, ev in [("neg2", Dz < -2), ("pos2", Dz > 2)]:
        evm = ev & msk[:, None]
        for j in [-1, 0, 1, 2]:
            vals, gid = [], []
            for i, h in zip(*np.where(evm)):
                if 1 <= h + j <= 23 and np.isfinite(dlEc[i, h + j]):
                    vals.append(dlEc[i, h + j])
                    gid.append(i)
            vals, gid = np.array(vals), np.array(gid)
            if len(vals) < 5:
                continue
            ug = np.unique(gid)
            bs = []
            for _ in range(1000):
                s = RNG.choice(ug, len(ug))
                bs.append(np.mean(np.concatenate([vals[gid == u] for u in s])))
            hrows.append(dict(set=lab, lag=j, event=kind, n_events=len(vals), mean_dlogEC25_vs_ctrl=vals.mean(),
                              ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
HR = pd.DataFrame(hrows)
HR.to_csv(os.path.join(OUT, "hourly_coupling_v1.csv"), index=False)
print(HR.round(4).to_string(index=False))

# converse: big EC25 drops in hi10 records -> D2 anomaly around them
print("\n-- hi10 big EC25 drops (dlogEC25 < -0.05 after hour control): mean D2 residual z at h-1, h, h+1 --")
for lab, msk in [("hi10", hi_mask), ("normal", ~hi_mask)]:
    evs = [(i, h) for i, h in zip(*np.where((dlEc < -0.05) & msk[:, None]))]
    for j in [-1, 0, 1]:
        v = [Dz[i, h + j] for i, h in evs if 1 <= h + j <= 23 and np.isfinite(Dz[i, h + j])]
        print("  %s lag %+d: n %d mean z %.3f (share z<-2 %.3f; base share %.3f)" % (
            lab, j, len(v), np.mean(v) if v else np.nan, np.mean(np.array(v) < -2) if v else np.nan,
            np.nanmean(Dz[msk][:, 1:] < -2)))

# sub_temp model residual vs EC residual at hourly level (within-day and between-day)
print("\n-- r3 sub_temp residual vs EC residual (sub_ec - sg) --")
ERES = EC - SG
for lab, msk in [("all", np.ones(NR, bool)), ("hi10", hi_mask)]:
    m = msk & np.isfinite(ERES).all(1)
    a, b = TRES[m], ERES[m]
    rb = corr(a.mean(1), b.mean(1))
    aw = (a - a.mean(1, keepdims=True)).ravel()
    bw = (b - b.mean(1, keepdims=True)).ravel()
    rw_ = corr(aw, bw)
    print("  %s: between-day r %.3f (rho %.3f, n %d) | within-day pooled r %.3f (n %d)" % (lab, rb[0], rb[1], rb[2], rw_[0], rw_[2]))

# ---------------- can inputs reproduce the irrigation-step signal? -----------------
print("\n== input reproducibility ==")
# hourly: predict D2 neg anomaly from inputs (levels, changes, lags) + hour, GroupKFold by component
Xh = [hrs[:, None]]
for v in INP:
    A = FULL[v]
    d = np.zeros_like(A)
    d[:, 1:] = np.diff(A, axis=1)
    Xh += [A.ravel()[:, None], d.ravel()[:, None], lagged(d, 1).ravel()[:, None], lagged(d, 2).ravel()[:, None]]
Xh = np.concatenate(Xh, 1)
yev = (Dz < -2).ravel().astype(float)
yres = DRES.ravel()
valid = np.isfinite(DRES.ravel()) & (hrs >= 1)
gi = comp_codes[rec_rows]
p_ev = np.full(len(yev), np.nan)
p_hr = np.full(len(yev), np.nan)
p_res = np.full(len(yev), np.nan)
vi = np.where(valid)[0]
for tr, te in GroupKFold(5).split(vi, groups=gi[vi]):
    a, b = vi[tr], vi[te]
    c = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(Xh[a], yev[a])
    p_ev[b] = c.predict_proba(Xh[b])[:, 1]
    c0 = HistGradientBoostingClassifier(max_iter=100, random_state=0).fit(np.c_[hrs[a], FARM[rec_rows[a]] == "F13"], yev[a])
    p_hr[b] = c0.predict_proba(np.c_[hrs[b], FARM[rec_rows[b]] == "F13"])[:, 1]
    rg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(Xh[a], yres[a])
    p_res[b] = rg.predict(Xh[b])
print("hourly anomalous cooling step (D2 z<-2, base rate %.3f): AUC inputs %.3f | AUC hour+farm only %.3f | R2 of D2 residual from inputs %.3f"
      % (yev[valid].mean(), roc_auc_score(yev[valid], p_ev[valid]), roc_auc_score(yev[valid], p_hr[valid]), r2(yres[valid], p_res[valid])))
# record level: predict key per-record ST variables from input day summaries (24h -> usable only at h=23)
summ = [c for c in R.columns if c.endswith("_d") or c.endswith("_n")]
Xr = R[summ].copy()
for v in INP:
    Xr[v + "_mean"] = np.nanmean(FULL[v], 1)
Xr = impute(Xr.values.astype(float))
rrows = []
for v in ["D_sumneg", "D_nneg2", "D_res_sd", "L_res_mean", "L_res_night", "amp_ratio", "k_relax", "peak_lag", "T_res_mean", "D3_sumneg"]:
    y = V[v].values.astype(float)
    m = np.isfinite(y)
    p = np.full(len(y), np.nan)
    for tr, te in GroupKFold(10).split(np.where(m)[0], groups=comp_codes[m]):
        a, b = np.where(m)[0][tr], np.where(m)[0][te]
        mu, sd = Xr[a].mean(0), Xr[a].std(0) + 1e-9
        p[b] = Ridge(alpha=30).fit((Xr[a] - mu) / sd, y[a]).predict((Xr[b] - mu) / sd)
    rrows.append(dict(variable=v, r2cv_from_inputs=r2(y, p)))
RR = pd.DataFrame(rrows)
RR.to_csv(os.path.join(OUT, "input_reproducibility_v1.csv"), index=False)
print(RR.round(3).to_string(index=False))

print("\nverdict counts:", res.verdict.value_counts().to_dict())
print("done")
