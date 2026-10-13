# -*- coding: utf-8 -*-
"""VC2 (2026-10-13 verifier): try to refute CT2 hi10 'EC25 drop <-> sub_temp cooling step' co-occurrence.
Checks (fixed before running): (a) record x hour-of-day demeaned D2 z (removes house/day offsets and hour pattern);
(b) same-record control = mean of that record's non-event hours; (c) leave-one-episode-out; (d) ep1 removed;
(e) EC25 coefficient sensitivity (0, 0.02, 0.03). Refuted if demeaned effect CI includes 0 or flips without one episode."""
import os, sys
sys.path.insert(0, r"C:\work\farmai\집\클로드\research")
import env  # noqa
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "crosstarget"))
import ct_common_v1 as ct
NR, FARM, EC, ST, Dz, R = ct.NR, ct.FARM, ct.EC, ct.ST, ct.Dz, ct.R
hi = R.hi10.values == 1; ep = R.ep_id.values
rng = np.random.default_rng(11)

def dctrl(A):
    d = np.full((NR, 24), np.nan); d[:, 1:] = np.diff(np.log(A), axis=1)
    c = np.zeros_like(d)
    for f in ["F13", "F47"]:
        m = FARM == f; c[m] = np.r_[np.nan, np.nanmean(d[m][:, 1:], 0)]
    return d - c

# hour-of-day mean within hi set, then record mean removed
def demean(Z, msk):
    Zd = Z.copy()
    hm = np.nanmean(Z[msk], 0); Zd = Zd - hm
    Zd = Zd - np.nanmean(Zd, 1, keepdims=True)
    return Zd

def eff(ev, Z, msk, lag, excl=None):
    m = msk.copy()
    if excl is not None: m &= ~excl
    Zd = demean(Z, m)
    ii, vv, ctl = [], [], []
    for i in np.where(m)[0]:
        hs = [h for h in np.where(ev[i])[0] if 1 <= h + lag <= 23 and np.isfinite(Zd[i, h + lag])]
        if not hs: continue
        evh = set(h + lag for h in hs)
        non = [k for k in range(1, 24) if k not in evh and np.isfinite(Zd[i, k])]
        for h in hs:
            ii.append(i); vv.append(Zd[i, h + lag] - np.mean(Zd[i, non]))
    ii = np.array(ii); vv = np.array(vv)
    if len(vv) < 5: return np.nan, np.nan, np.nan, 0
    ug = np.unique(ii)
    bs = [np.mean(np.concatenate([vv[ii == u] for u in rng.choice(ug, len(ug))])) for _ in range(1000)]
    return vv.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(vv)

for coef in [0.0, 0.02, 0.03]:
    E25 = EC / (1 + coef * (ST - 25)); ev = dctrl(E25) < -0.05
    for lab, msk in [("hi10", hi), ("normal", ~hi)]:
        for lag in (-1, 0, 1):
            m_, lo, hi_, n = eff(ev, Dz, msk, lag)
            print("coef %.2f %-6s lag %+d: demeaned within-record diff %.3f CI [%.3f, %.3f] n %d" % (coef, lab, lag, m_, lo, hi_, n))
ev = dctrl(EC / (1 + 0.02 * (ST - 25))) < -0.05
print("-- leave-one-episode-out (hi10, coef .02) --")
for e in np.unique(ep[hi]):
    out = [eff(ev, Dz, hi, lag, excl=(ep == e))[0] for lag in (-1, 0, 1)]
    print("  drop ep %d (%d rec): lag-1 %.3f lag0 %.3f lag+1 %.3f" % (e, (hi & (ep == e)).sum(), *out))
# hi10 vs normal contrast in raw D2 residual level (house offset)
print("mean Dz hi10 %.3f normal %.3f ; hours 9-18 hi10 %.3f normal %.3f" % (np.nanmean(Dz[hi][:, 1:]), np.nanmean(Dz[~hi][:, 1:]),
      np.nanmean(Dz[hi][:, 9:19]), np.nanmean(Dz[~hi][:, 9:19])))
print("hour profile of hi10 EC25 drops:", np.bincount(np.where(ev[hi])[1], minlength=24).tolist())
print("done")
