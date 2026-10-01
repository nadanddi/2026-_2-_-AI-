# -*- coding: utf-8 -*-
"""IF3: fair smoothness test + cross-farm copies.  Fractional runs are rounded
to integers first so rounding noise is equal; compared with integer rows of the
same farm over windows of the same length and hours.  Also counts IF2 copy
hits that point to a DIFFERENT farm.
Run:  cd 집/클로드/research && PYTHONPATH="" py -3.12 -u int_frac_audit_v3.py"""
import env  # noqa: F401
import os
import numpy as np
import pandas as pd
X = pd.read_csv(os.path.join(env.DATA, "train_X.csv"))
p = X.row_id.str.split("_", expand=True)
X["farm"], X["day"], X["hour"] = p[0], p[1].astype(int), p[2].astype(int)
X = X[~X.farm.isin(["F13", "F47", "F32"])].sort_values(["farm", "day", "hour"]).reset_index(drop=True)
X["t"] = X.day * 24 + X.hour
X["fr"] = X.in_temp.notna() & (np.abs(X.in_temp - np.round(X.in_temp)) > 1e-9)
rng = np.random.default_rng(1)
fr_r, int_r, fr_raw = [], [], []
for f, g in X.groupby("farm"):
    v, fr, t = g.in_temp.values, g.fr.values, g.t.values
    i = 0
    while i < len(g):
        if not fr[i]:
            i += 1; continue
        j = i
        while j + 1 < len(g) and fr[j + 1] and t[j + 1] == t[j] + 1:
            j += 1
        L = j - i + 1
        if L >= 6:
            fr_r.append(np.abs(np.diff(np.round(v[i:j + 1]), 2)).mean())
            fr_raw.append(np.abs(np.diff(v[i:j + 1], 2)).mean())
            # matched integer windows: same farm, same start hour, no fractional/NaN rows
            cand = [k for k in range(len(g) - L) if t[k] % 24 == t[i] % 24 and not fr[k:k + L].any()
                    and not np.isnan(v[k:k + L]).any() and t[k + L - 1] - t[k] == L - 1]
            if cand:
                ks = rng.choice(cand, size=min(20, len(cand)), replace=False)
                int_r.append(np.mean([np.abs(np.diff(v[k:k + L], 2)).mean() for k in ks]))
            else:
                int_r.append(np.nan)
        i = j + 1
a, b = np.array(fr_r), np.array(int_r)
ok = ~np.isnan(b)
print("long runs %d (matched %d)" % (len(a), ok.sum()))
print("mean |2nd diff| rounded fractional run %.3f vs same-farm same-hour integer windows %.3f (raw fractional %.3f)"
      % (a[ok].mean(), b[ok].mean(), np.mean(fr_raw)))
print("share of runs smoother than their matched integer windows: %.1f%%" % (100 * (a[ok] < b[ok]).mean()))
