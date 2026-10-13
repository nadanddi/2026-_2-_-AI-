# -*- coding: utf-8 -*-
"""CT2 follow-up of CT1 (2026-10-13 집 클로드). Post-hoc (decided after reading CT1 output) -> descriptive only.

CT1 saw: in hi10 records, hours with a big temperature-compensated EC drop (dlogEC25 < -0.05 after farm x hour
control) had a negative mean D2 sub_temp step residual (z about -0.7..-0.8), while the forward event study (sub_temp
anomaly hours -> EC25 change) was null.  Here:
 (1) cluster-bootstrap CI (records) for mean D2 z around hi10 EC drops vs the same records' hours at the same
     hour-of-day (placebo: 1000 draws of random hours matched on record set and hour-of-day);
 (2) same with raw EC drops, EC25 rises, D3 (all-input) residual, and normal records;
 (3) per-episode means (is it one chain?);
 (4) can inputs alone flag those hi10 EC-drop hours? (input classifier for D2 cooling steps, AUC on the EC-drop hours)
Prediction (written before running, from the irrigation mechanism): if irrigation with cooler water causes the drops,
mean D2 z at h..h+1 < 0 with CI excluding 0 and below the placebo band, also for raw EC, in most episodes; D3 z also <0
(not explained by inputs).  Must NOT see: effect only in one episode, or effect vanishing for raw EC.

Run:  cd 집\\클로드\\research && PYTHONPATH="" python -u hx9/crosstarget/ct2_drop_cooccurrence_v1.py
"""
import os
import sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ct_common_v1 as ct1  # same construction as CT1 (verbatim copy), does not write files
NR, FARM, EC, ST, Dz, D3z, R = ct1.NR, ct1.FARM, ct1.EC, ct1.ST, ct1.Dz, ct1.D3z, ct1.R
rng = np.random.default_rng(7)
EC25 = EC / (1 + 0.02 * (ST - 25))


def dctrl(A):
    d = np.full((NR, 24), np.nan)
    d[:, 1:] = np.diff(np.log(A), axis=1)
    c = np.zeros_like(d)
    for f in ["F13", "F47"]:
        m = FARM == f
        c[m] = np.r_[np.nan, np.nanmean(d[m][:, 1:], 0)]
    return d - c


dE25 = dctrl(EC25)
dE = dctrl(EC)
hi = R.hi10.values == 1
ep = R.ep_id.values


def study(ev_mask, Z, rec_mask, label):
    out = []
    evs = [(i, h) for i, h in zip(*np.where(ev_mask & rec_mask[:, None]))]
    for j in [-1, 0, 1]:
        pts = [(i, Z[i, h + j], h + j) for i, h in evs if 1 <= h + j <= 23 and np.isfinite(Z[i, h + j])]
        if len(pts) < 5:
            continue
        ii = np.array([p[0] for p in pts])
        zz = np.array([p[1] for p in pts])
        hh = np.array([p[2] for p in pts])
        ug = np.unique(ii)
        bs = [np.mean(np.concatenate([zz[ii == u] for u in rng.choice(ug, len(ug))])) for _ in range(1000)]
        # placebo: same records, same hour-of-day, random other day among the same record set
        pool = np.where(rec_mask)[0]
        pl = []
        for _ in range(1000):
            rr = rng.choice(pool, len(pts))
            pl.append(np.nanmean(Z[rr, hh]))
        out.append(dict(set=label, lag=j, n=len(zz), n_rec=len(ug), mean_z=zz.mean(), ci_lo=np.percentile(bs, 2.5),
                        ci_hi=np.percentile(bs, 97.5), placebo_lo=np.percentile(pl, 2.5), placebo_hi=np.percentile(pl, 97.5),
                        share_lt2=np.mean(zz < -2), base_share_lt2=np.nanmean(Z[rec_mask][:, 1:] < -2)))
    return out


rows = []
for ev_name, ev in [("EC25 drop<-.05", dE25 < -0.05), ("rawEC drop<-.05", dE < -0.05), ("EC25 rise>+.05", dE25 > 0.05)]:
    for zname, Z in [("D2", Dz), ("D3", D3z)]:
        for lab, msk in [("hi10", hi), ("normal", ~hi)]:
            for r_ in study(ev, Z, msk, lab):
                r_.update(event=ev_name, resid=zname)
                rows.append(r_)
T = pd.DataFrame(rows)[["event", "resid", "set", "lag", "n", "n_rec", "mean_z", "ci_lo", "ci_hi", "placebo_lo", "placebo_hi",
                        "share_lt2", "base_share_lt2"]]
T.to_csv(os.path.join(HERE, "ct2_drop_cooccurrence_v1.csv"), index=False)
pd.set_option("display.width", 250)
print(T.round(3).to_string(index=False))

print("\n-- per episode (hi10, EC25 drop, D2 z mean over lags 0..1) --")
for e in np.unique(ep[hi]):
    m = hi & (ep == e)
    evs = [(i, h) for i, h in zip(*np.where((dE25 < -0.05) & m[:, None]))]
    z = [Dz[i, h + j] for i, h in evs for j in (0, 1) if 1 <= h + j <= 23 and np.isfinite(Dz[i, h + j])]
    print("  ep %d: records %d, drop events %d, mean z %.2f" % (e, m.sum(), len(evs), np.mean(z) if z else np.nan))

# (4) inputs only: hourly classifier for D2 cooling steps (z<-2) -> AUC on hi10 EC25-drop hours
INP = ct1.INP
hrs = np.tile(np.arange(24), NR)
rec_rows = np.repeat(np.arange(NR), 24)
Xh = [hrs[:, None]]
for v in INP:
    A = ct1.FULL[v]
    d = np.zeros_like(A)
    d[:, 1:] = np.diff(A, axis=1)
    Xh += [A.ravel()[:, None], d.ravel()[:, None], ct1.lagged(d, 1).ravel()[:, None], ct1.lagged(d, 2).ravel()[:, None]]
Xh = np.concatenate(Xh, 1)
yev = (Dz < -2).ravel().astype(float)
valid = np.isfinite(Dz.ravel()) & (hrs >= 1)
gi = ct1.comp_codes[rec_rows]
p = np.full(len(yev), np.nan)
vi = np.where(valid)[0]
for tr, te in GroupKFold(5).split(vi, groups=gi[vi]):
    c = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(Xh[vi[tr]], yev[vi[tr]])
    p[vi[te]] = c.predict_proba(Xh[vi[te]])[:, 1]
P = p.reshape(NR, 24)
m = hi[:, None] & np.isfinite(dE25) & np.isfinite(P)
lab = (dE25 < -0.05)[m]
print("\ninput-predicted sub_temp cooling-step probability as detector of hi10 EC25-drop hours: AUC %.3f (n events %d of %d hours)"
      % (roc_auc_score(lab, P[m]), lab.sum(), m.sum()))
# same within-record (rank of hour inside its record) to remove day-level effects
wr = []
for i in np.where(hi)[0]:
    mm = np.isfinite(dE25[i]) & np.isfinite(P[i])
    l = (dE25[i] < -0.05)[mm]
    if 0 < l.sum() < mm.sum():
        wr.append(roc_auc_score(l, P[i][mm]))
print("within-record AUC (mean over %d hi10 records with >=1 drop): %.3f" % (len(wr), np.mean(wr)))
print("done")
