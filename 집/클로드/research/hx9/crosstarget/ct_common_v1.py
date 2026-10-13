# -*- coding: utf-8 -*-
"""CT common (2026-10-13 집 클로드): verbatim copy of ct1_subtemp_coupling_v1.py lines 42-220 (data, groups, r3 OOF, L/D2/D3 expectation models, robust sigma) so follow-ups do not re-run or overwrite CT1 outputs."""
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

Dz = DRES / sig
D3z = D3RES / sig3
