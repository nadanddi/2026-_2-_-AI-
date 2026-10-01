# -*- coding: utf-8 -*-
"""RD1: what method produced the fractional (restored-looking) in_temp runs in
the 48 integer farms?  The signature found here is the candidate detector for
restored stretches in F13/F47 train (where every value has a decimal).
2026-10-02 집 클로드.

Candidate generators, each scored on long runs (>= 6 h) by max |error|:
  LIN    straight line between the adjacent integer rows
  PCHIP  shape-preserving cubic through 3 integer rows each side
  CSPL   natural cubic spline through 3 integer rows each side
  HUM    per-farm linear regression in_temp ~ in_hum + in_co2 (+hour sin/cos)
         fitted on that farm's integer rows; applied to the run rows
  HUMR   HUM residual of the run rows vs residual of integer rows (a model-made
         value would sit unusually close to the regression line)
  SEAS   same hour mean of the +-3 neighbouring days (integer rows)
Also: lag-1 autocorrelation of the 1st difference inside runs vs integer rows.

Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u rd1_restore_signature_v1.py
"""
import env  # noqa: F401
import os

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator

X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
p = X.row_id.str.split("_", expand=True)
X["farm"], X["day"], X["hour"] = p[0], p[1].astype(int), p[2].astype(int)
X = X[~X.farm.isin(["F13", "F47", "F32"])].sort_values(["farm", "day", "hour"]).reset_index(drop=True)
X["t"] = X.day * 24 + X.hour
X["fr"] = X.in_temp.notna() & (np.abs(X.in_temp - np.round(X.in_temp)) > 1e-9)
X["hs"], X["hc"] = np.sin(2 * np.pi * X.hour / 24), np.cos(2 * np.pi * X.hour / 24)

rows = []
hum_int_res = []
for f, g in X.groupby("farm"):
    g = g.reset_index(drop=True)
    tt, v, fr = g.t.values, g.in_temp.values, g.fr.values
    # HUM regression on integer rows
    m = (~g.fr) & g[["in_temp", "in_hum", "in_co2"]].notna().all(axis=1)
    A = np.c_[np.ones(m.sum()), g.loc[m, ["in_hum", "in_co2", "hs", "hc"]].values]
    beta, *_ = np.linalg.lstsq(A, g.loc[m, "in_temp"].values, rcond=None)
    full = np.c_[np.ones(len(g)), g[["in_hum", "in_co2", "hs", "hc"]].values] @ beta
    hum_int_res.append(np.abs(full[m.values] - v[m.values]))
    i = 0
    while i < len(g):
        if not fr[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(g) and fr[j + 1] and tt[j + 1] == tt[j] + 1:
            j += 1
        L = j - i + 1
        if L >= 6 and i >= 3 and j + 3 < len(g):
            li, ri = np.arange(i - 3, i), np.arange(j + 1, j + 4)
            anc = np.r_[li, ri]
            if not np.isnan(v[anc]).any() and not fr[anc].any() and tt[ri[-1]] - tt[li[0]] == ri[-1] - li[0]:
                seg = v[i:j + 1]
                x = tt[i:j + 1]
                lin = v[i - 1] + (v[j + 1] - v[i - 1]) * (x - tt[i - 1]) / (tt[j + 1] - tt[i - 1])
                pch = PchipInterpolator(tt[anc], v[anc])(x)
                csp = CubicSpline(tt[anc], v[anc], bc_type="natural")(x)
                hum = full[i:j + 1]
                seas = []
                for k in range(i, j + 1):
                    nb = g[(g.hour == g.hour[k]) & (abs(g.day - g.day[k]) <= 3) & (g.day != g.day[k]) & ~g.fr]
                    seas.append(nb.in_temp.mean())
                seas = np.array(seas)
                d1 = np.diff(seg)
                rows.append(dict(farm=f, L=L, LIN=np.max(np.abs(lin - seg)), PCHIP=np.max(np.abs(pch - seg)),
                                 CSPL=np.max(np.abs(csp - seg)), HUM=np.max(np.abs(hum - seg)),
                                 HUMR=np.mean(np.abs(hum - seg)), SEAS=np.nanmax(np.abs(seas - seg)),
                                 ac1=np.corrcoef(d1[:-1], d1[1:])[0, 1] if L >= 5 else np.nan))
        i = j + 1
R = pd.DataFrame(rows)
print("long runs with 3 clean anchors each side:", len(R))
for c in ("LIN", "PCHIP", "CSPL", "HUM", "SEAS"):
    print("  %-6s median max|err| %.2f   share <= 0.15: %.1f%%   <= 0.5: %.1f%%"
          % (c, R[c].median(), 100 * (R[c] <= .15).mean(), 100 * (R[c] <= .5).mean()))
hi = np.concatenate(hum_int_res)
print("  HUM mean |resid| on run rows %.2f vs integer rows %.2f" % (R.HUMR.mean(), hi.mean()))

# 1st-difference autocorrelation: runs vs integer 24-h windows
ac_int = []
for (f, d), g in X[~X.fr].groupby(["farm", "day"]):
    if len(g) == 24 and g.in_temp.notna().all():
        d1 = np.diff(g.in_temp.values)
        if d1.std() > 0:
            ac_int.append(np.corrcoef(d1[:-1], d1[1:])[0, 1])
print("  lag-1 autocorr of 1st diff: runs median %.2f vs integer days median %.2f"
      % (R.ac1.median(), np.nanmedian(ac_int)))

# split: straight-line runs vs the rest
R["lin"] = R.LIN <= 0.15
for nm, h in (("straight-line runs", R[R.lin]), ("other runs", R[~R.lin])):
    print("  %-19s n=%d  median len %.0f  ac1 median %.2f  PCHIP<=0.5 %.0f%%  SEAS median %.2f"
          % (nm, len(h), h.L.median(), h.ac1.median(), 100 * (h.PCHIP <= .5).mean(), h.SEAS.median()))
