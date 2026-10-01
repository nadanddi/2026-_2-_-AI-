# -*- coding: utf-8 -*-
"""IF2: follow-up to int_frac_audit_v1 (fractional in_temp rows in the 48
integer farms are mostly NOT straight-line fills).  Where do they come from?

  A. split runs: short (<=4 h) vs long (>=6 h); straight-line share in each
  B. copy test: does a long fractional run (>=6 h) match another farm's
     in_temp series (any time shift) after rounding - i.e. was the gap filled
     with another greenhouse's record?  Match = every hour |a - b| <= 0.55
     (b integer-rounded) and in_hum within 1.5 as well.  Also against F13/F47/F32.
  C. does the run look like real sensor data: roughness (mean |2nd diff|) of
     fractional runs vs F13/F47 in_temp (0.1 resolution, real) vs integer rows
  D. labels: share of the 5,025 unlabelled rows that are fractional; sub_temp
     on fractional labelled rows - integer or not

Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u int_frac_audit_v2.py
"""
import env  # noqa: F401
import os

import numpy as np
import pandas as pd

X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
Y = pd.read_csv(os.path.join(env.DATA, "train_y.csv"))
X = X.merge(Y, on="row_id", how="left")
p = X.row_id.str.split("_", expand=True)
X["farm"], X["day"], X["hour"] = p[0], p[1].astype(int), p[2].astype(int)
X = X.sort_values(["farm", "day", "hour"]).reset_index(drop=True)
X["t"] = X.day * 24 + X.hour
X["fr"] = X.in_temp.notna() & (np.abs(X.in_temp - np.round(X.in_temp)) > 1e-9)
INT = ~X.farm.isin(["F13", "F47", "F32"])
X.loc[~INT, "fr"] = False

# runs
runs = []
for f, g in X[INT].groupby("farm"):
    t, v, fr, idx = g.t.values, g.in_temp.values, g.fr.values, g.index.values
    i, n = 0, len(g)
    while i < n:
        if not fr[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and fr[j + 1] and t[j + 1] == t[j] + 1:
            j += 1
        dev = np.nan
        if i > 0 and j + 1 < n and t[i - 1] == t[i] - 1 and t[j + 1] == t[j] + 1 \
                and not np.isnan(v[i - 1]) and not np.isnan(v[j + 1]):
            line = v[i - 1] + (v[j + 1] - v[i - 1]) * (t[i:j + 1] - t[i - 1]) / (t[j + 1] - t[i - 1])
            dev = float(np.max(np.abs(line - v[i:j + 1])))
        runs.append(dict(farm=f, i0=idx[i], i1=idx[j], len=j - i + 1, dev=dev))
        i = j + 1
R = pd.DataFrame(runs)
print("A. runs", len(R))
for nm, s in (("short<=4", R.len <= 4), ("mid5", R.len == 5), ("long>=6", R.len >= 6)):
    h = R[s]
    print("   %-9s runs %4d rows %5d  straight-line(<=0.051) %.0f%%"
          % (nm, len(h), h.len.sum(), 100 * (h.dev[h.dev.notna()] <= .051).mean()))

# B. copy test against every farm's series
series = {f: (g.t.values, g.in_temp.values, g.in_hum.values) for f, g in X.groupby("farm")}
long = R[R.len >= 6]
hits = []
for _, r in long.iterrows():
    a = X.loc[r.i0:r.i1, "in_temp"].values
    ah = X.loc[r.i0:r.i1, "in_hum"].values
    L = len(a)
    found = []
    for f, (t, v, h) in series.items():
        if len(v) < L:
            continue
        W = np.lib.stride_tricks.sliding_window_view(v, L)
        ok = np.all(np.abs(W - a) <= 0.55, axis=1)
        if f == r.farm:   # exclude itself
            ok[X.index.get_loc(r.i0) - X.index.get_loc(X[X.farm == f].index[0])] = False
        for k in np.where(ok)[0]:
            Wh = h[k:k + L]
            if np.all(np.abs(Wh - ah) <= 1.5) and np.all(np.diff(t[k:k + L]) == 1):
                found.append((f, int(t[k] // 24), int(t[k] % 24)))
    hits.append(dict(farm=r.farm, day=int(X.loc[r.i0, "day"]), hour=int(X.loc[r.i0, "hour"]), len=L, n=len(found), where=found[:3], cross=sum(1 for x in found if x[0] != r.farm)))
H = pd.DataFrame(hits)
print("\nB. long runs (>=6 h) with a copy elsewhere (temp & hum match every hour): %d of %d"
      % ((H.n > 0).sum(), len(H)))
print(H[H.n > 0].head(15).to_string())
print("   runs with a match in a DIFFERENT farm: %d" % (H.cross > 0).sum())
print(H[H.cross > 0].to_string())


# C. roughness
def rough(vals_by_group):
    out = []
    for v in vals_by_group:
        if len(v) >= 3:
            out.append(np.abs(np.diff(v, 2)))
    return float(np.mean(np.concatenate(out))) if out else np.nan


fr_runs = [X.loc[r.i0:r.i1, "in_temp"].values for _, r in long.iterrows()]
tgt = [g.in_temp.dropna().values for _, g in X[X.farm.isin(["F13", "F47"])].groupby(["farm", "day"])]
intd = [g.in_temp.values for _, g in X[INT & ~X.fr].groupby(["farm", "day"]) if g.in_temp.notna().all() and len(g) == 24]
print("\nC. mean |2nd diff| in_temp: fractional long runs %.3f | F13/F47 days (real 0.1 res) %.3f | integer days %.3f"
      % (rough(fr_runs), rough(tgt), rough(intd[:5000])))

# D. labels
lab_missing = INT & X.sub_temp.isna() & X.in_temp.notna()
print("\nD. unlabelled rows in integer farms: %d, of which fractional in_temp %.1f%%"
      % (lab_missing.sum(), 100 * X.fr[lab_missing].mean()))
fl = X[X.fr & X.sub_temp.notna()]
print("   labelled fractional rows %d: sub_temp fractional %.2f%%" %
      (len(fl), 100 * (np.abs(fl.sub_temp - np.round(fl.sub_temp)) > 1e-9).mean()))
by = X[INT].groupby("farm").agg(fr=("fr", "mean"), unl=("sub_temp", lambda s: s.isna().mean()))
print("   per-farm corr(frac share, unlabelled share) %.2f" % by.corr().iloc[0, 1])
print(by.sort_values("fr", ascending=False).head(8).round(3).to_string())
