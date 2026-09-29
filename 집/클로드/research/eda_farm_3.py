# -*- coding: utf-8 -*-
"""Targeted pooling test: add cold-regime rows of greenhouses whose sub_temp~in_temp response resembles F13/F47.
Held-out rows are always F13/F47; donors only ever train.  Indoor-only features (all farms have them)."""
import env  # noqa
import sys, time
import numpy as np, pandas as pd
import lightgbm as lgb
import common
import features_v2 as F2
from common import split_mask, rmse, TARGET_FARMS
from harness import folds

SIM = ["F50", "F38", "F30", "F42"]
DIS = ["F09", "F12", "F31", "F24"]
tX, ty, sX = common.load_raw()
big = F2.build(tX, sX, farms=TARGET_FARMS + SIM + DIS)
big = big.merge(ty[["row_id", "sub_temp"]], on="row_id", how="left")
big["is_test"] = big.row_id.isin(set(sX.row_id))
lab = big[(~big.is_test) & big.sub_temp.notna()].reset_index(drop=True)
FEAT = ["in_temp", "in_hum", "in_co2", "vpd_in", "root_dh", "in_temp_ewm2", "in_temp_ewm6", "in_temp_ewm24",
        "in_temp_dev6", "in_temp_dev24", "in_temp_lag1", "in_temp_lag2", "in_temp_lag3", "in_temp_d1", "in_temp_d3",
        "in_temp_r24m", "in_temp_r24s", "in_temp_r72m", "in_temp_r168m", "vpd_in_r24m", "in_temp_tdmean",
        "in_temp_dm_pd1", "in_hum_dm_pd1", "in_co2_dm_pd1", "vpd_in_dm_pd1", "in_temp_dmax_pd1", "in_temp_dmin_pd1",
        "in_temp_dm_pd7m", "hr_sin", "hr_cos", "farm_id"]
BASE = ["in_temp_ewm2", "in_temp_ewm6", "in_temp_ewm24"]
lab.loc[~lab.farm.isin(TARGET_FARMS), "farm_id"] = np.nan
is_t = lab.farm.isin(TARGET_FARMS).values

def linfit(D):
    A = np.c_[np.ones(len(D)), D[BASE].values]
    m = np.isfinite(A).all(1)
    c, *_ = np.linalg.lstsq(A[m], D.sub_temp.values[m], rcond=None)
    return c
def linpred(D, c):
    return np.c_[np.ones(len(D)), D[BASE].values] @ c

P = dict(objective="l2", n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=40,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.7, n_jobs=2, verbose=-1)

def cold_fold():
    fd = {}
    for f in TARGET_FARMS:
        d = lab[lab.farm == f].groupby("day").in_temp.mean().sort_values()
        fd[f] = set(d.index[:25])
    return fd

def run(fd, donors, cold_thr, resid, align=True):
    trm, vam = split_mask(lab, fd)
    trm = trm & is_t  # split_mask only buffers target farms; donors all pass, restricted below
    tr = lab[trm].copy(); va = lab[vam & is_t].copy()
    if donors:
        D = lab[lab.farm.isin(donors)].copy()
        if cold_thr is not None:
            D = D[D.in_temp_ewm6 < cold_thr]
        D = D.copy()
    else:
        D = lab.iloc[:0].copy()
    ytr = tr.sub_temp.values.astype(float); yD = D.sub_temp.values.astype(float)
    if resid:
        # per-farm linear baseline, each fitted on that farm's own training labels
        btr = np.zeros(len(tr)); bva = np.zeros(len(va))
        for f in TARGET_FARMS:
            c = linfit(tr[tr.farm == f]); btr[(tr.farm == f).values] = linpred(tr[tr.farm == f], c)
            bva[(va.farm == f).values] = linpred(va[va.farm == f], c)
        bD = np.zeros(len(D))
        for f in D.farm.unique():
            full = lab[lab.farm == f]; c = linfit(full); m = (D.farm == f).values; bD[m] = linpred(D[m], c)
        ytr = ytr - btr; yD = yD - bD
    else:
        bva = 0.0
        if align and len(D):
            off_t = np.nanmean(tr.sub_temp - tr.in_temp_ewm6)
            for f in D.farm.unique():
                full = lab[lab.farm == f]; m = (D.farm == f).values
                yD[m] += off_t - np.nanmean(full.sub_temp - full.in_temp_ewm6)
    X = pd.concat([tr[FEAT], D[FEAT]]); y = np.r_[ytr, yD]
    ps = []
    for s in (1, 2):
        m = lgb.LGBMRegressor(random_state=s, **P).fit(X, y)
        ps.append(m.predict(va[FEAT]))
    p = np.mean(ps, 0) + bva
    return va, p

def lin_only(fd):
    trm, vam = split_mask(lab, fd); tr = lab[trm & is_t]; va = lab[vam & is_t]
    p = np.zeros(len(va))
    for f in TARGET_FARMS:
        c = linfit(tr[tr.farm == f]); m = (va.farm == f).values; p[m] = linpred(va[m], c)
    return va, p

VAR = [("T only", None, None, False), ("T+SIM cold<12", SIM, 12, False), ("T+SIM all", SIM, None, False),
       ("T+DIS cold<12", DIS, 12, False),
       ("R: T only", None, None, True), ("R: T+SIM cold<12", SIM, 12, True), ("R: T+SIM all", SIM, None, True),
       ("R: T+DIS cold<12", DIS, 12, True)]
res = []
for kind in ("A", "COLD"):
    fds = folds("A") if kind == "A" else [cold_fold()]
    allv = {}
    for name, dn, thr, rs in VAR + [("LIN only", None, None, None)]:
        vs, pp = [], []
        for fd in fds:
            va, p = lin_only(fd) if name == "LIN only" else run(fd, dn, thr, rs)
            vs.append(va); pp.append(p)
        va = pd.concat(vs); p = np.concatenate(pp); y = va.sub_temp.values
        c1 = (va.in_temp_ewm6 < 9.72).values; c2 = (va.in_temp_ewm6 < 12).values
        r = dict(cv=kind, var=name, n=len(y), all=rmse(p, y), n_c12=int(c2.sum()),
                 cold12=rmse(p[c2], y[c2]) if c2.any() else np.nan,
                 n_c972=int(c1.sum()), cold972=rmse(p[c1], y[c1]) if c1.any() else np.nan,
                 bias_c12=float(np.mean(p[c2] - y[c2])) if c2.any() else np.nan)
        print(r, flush=True); res.append(r)
R = pd.DataFrame(res).round(4); print(R.to_string())
R.to_csv(env.LOCAL + "/eda_farm_3.csv", index=False)
