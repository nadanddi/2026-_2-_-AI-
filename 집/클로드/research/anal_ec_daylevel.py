# -*- coding: utf-8 -*-
"""Where could day-level EC information come from?  (anal_ec_noncausal.py:
even full-day inputs leave a day-level error of 0.17; a 0.128 total needs
~0.09.)  Here only LABELS of other days are used as predictors of a day's
mean EC, evaluated like DIAG10: each 5-day block is held out and may use
labels outside the block only.

Predictors of day (farm f, day d) mean EC:
  GLOB   mean of all training day means (baseline)
  FARM   mean of farm f's training days
  PAR2   mean of farm f days d-2, d+2 (same-source chain, catalog 1.14)
  NEAR   mean of farm f days within +-7 outside the block
  SIB    mean of same-calendar-date siblings (other records whose 24-h
         out_temp vector nearly equals day d's), any farm, outside the block
  XF     the other farm (F47 for F13) same day index
Each alone, then a ridge stack fitted across folds (fold-out).
Also reported: how many TEST days have each predictor available.

Analysis only.  Run:  cd research && PYTHONPATH="" <python> -u anal_ec_daylevel.py
"""
import env  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

import common
from common import TARGET_FARMS
from harness import load
from anal_q1_errors import diag_folds

TOL = 0.35   # out_temp 24-h RMSE below which two days count as the same date


def main():
    tX, ty, sX = common.load_raw()
    _, _, lab = load()
    D = lab.groupby(["farm", "day"]).sub_ec.mean().rename("ec").reset_index()
    allx = pd.concat([tX, sX], ignore_index=True)
    allx = allx[allx.farm.isin(TARGET_FARMS)]
    W = allx.pivot_table(index=["farm", "day"], columns="hour", values="out_temp").reindex(columns=range(24))
    W = W[W.notna().sum(1) >= 20]
    wk = list(W.index)
    Wv = W.values
    test_days = set(map(tuple, sX[["farm", "day"]].drop_duplicates().values))

    # same-date siblings by weather vector
    sib = {}
    for i, k in enumerate(wk):
        d = np.sqrt(np.nanmean((Wv - Wv[i]) ** 2, axis=1))
        sib[k] = [wk[j] for j in np.where(d < TOL)[0] if wk[j] != k]
    ec = {(f, d): v for f, d, v in D[["farm", "day", "ec"]].itertuples(index=False)}
    print("days with >=1 labelled sibling: training %d/%d, test %d/%d"
          % (sum(any(s in ec for s in sib.get(k, [])) for k in ec), len(ec),
             sum(any(s in ec for s in sib.get(k, [])) for k in test_days), len(test_days)), flush=True)

    rows = []
    for fi, fd in enumerate(diag_folds(lab)):
        held = {(f, d) for f in fd for d in fd[f]}
        avail = {k: v for k, v in ec.items() if k not in held}
        glob = np.mean(list(avail.values()))
        for k in held:
            if k not in ec:
                continue
            f, d = k
            fm = np.mean([v for (ff, _), v in avail.items() if ff == f])

            def m(keys):
                v = [avail[x] for x in keys if x in avail]
                return np.mean(v) if v else np.nan
            rows.append(dict(fold=fi, farm=f, day=d, y=ec[k], GLOB=glob, FARM=fm,
                             PAR2=m([(f, d - 2), (f, d + 2)]),
                             NEAR=m([(f, x) for x in range(d - 7, d + 8) if x != d]),
                             SIB=m(sib.get(k, [])),
                             XF=m([("F47" if f == "F13" else "F13", d)])))
    R = pd.DataFrame(rows)
    print("\nday-level RMSE of each predictor (days where available; n)")
    for c in ("GLOB", "FARM", "PAR2", "NEAR", "SIB", "XF"):
        g = R[c].notna()
        print("  %-5s %.4f  (GLOB on same days %.4f, n=%d)"
              % (c, np.sqrt(((R[c][g] - R.y[g]) ** 2).mean()), np.sqrt(((R.GLOB[g] - R.y[g]) ** 2).mean()), g.sum()))
    # stack (fill missing with FARM), fold-out
    Z = R[["FARM", "PAR2", "NEAR", "SIB", "XF"]].copy()
    for c in Z:
        Z[c] = Z[c].fillna(R.FARM)
    pred = np.full(len(R), np.nan)
    for fi in R.fold.unique():
        tr, te = R.fold != fi, R.fold == fi
        pred[te] = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Z[tr], R.y[tr]).predict(Z[te])
    print("\nstack (label predictors only) day-level RMSE %.4f | our causal model day-level err ~0.201"
          % np.sqrt(((pred - R.y) ** 2).mean()))
    print("corr of day means: PAR2 %.2f  NEAR %.2f  SIB %.2f  XF %.2f"
          % tuple(R[["y", c]].dropna().corr().iloc[0, 1] for c in ("PAR2", "NEAR", "SIB", "XF")))


if __name__ == "__main__":
    main()
